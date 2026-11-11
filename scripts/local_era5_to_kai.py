"""Step 13 data from a local ERA5 0.25 deg hourly archive instead of the WeatherBench2 zarr.

Same outputs and layout as scripts/wb2_to_kai.py --all-hours (``<out>/kai6h/<year>.nc`` and the
WeatherBench1-style ``<out>/era5_data``), built from one ``YYYYMMDD.nc`` per UTC day (0.25 deg,
721x1440, lat N -> S, 13 levels, vars z q t u v u10 v10 mslp tcwv sp skt t2m tisr, ...):

    * 00/06/12/18 UTC snapshots, first-order conservative regridding (exact cell-overlap area
      weights, separable in lat/lon) onto ClimODE's 64x32 grid without poles
    * tisr and skt are the archive's own fields (wb2_to_kai.py had to compute tisr and copy t2m)
    * orography / lsm from the 0.25 deg constant files, regridded the same way

``--daily-mean`` (the tendency core +, docs/kai_tendency_plus.md): only the daily means of the four snapshots,
``<out>/kaiday/<year>.npy`` (D, 67, H, W) float32, the cache scripts/run_kai_tendency.py reads; no 6-hourly file
(12.8 GB a year at 1.40625 deg). ``--grid 1.5`` = KAI's own grid, 121 x 240 with the two pole rows (their cells
are the half cells poleward of 89.25 deg, as in WeatherBench2's 240x121 conservative files).

``--src-daily`` (with ``--daily-mean``): the archive already holds DAILY MEANS (ERA5_0p25_DAILY). The files of a
day are found by themselves (``--pattern auto``: the date, else the month, else the year in the file name, up to
two folders deep; one file with everything or one file per variable; a monthly / yearly file is read at the day's
position on its time axis). Missing skt := t2m, missing tisr is computed (mean of the 1 h accumulations ending at
00/06/12/18 UTC, as scripts/wb2_to_kai.py). ``--check --src-daily`` prints what was found.

The archive is only read. One process per year:

    python scripts/local_era5_to_kai.py --src /lustre/home/mahmed/ERA5/hourly --check --years 2016 --out data/era5_5p625
    python scripts/local_era5_to_kai.py --src /lustre/home/mahmed/ERA5/hourly --out data/era5_5p625 \
        --years 1979 1980 ... 2020 --workers 20
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import netCDF4 as nc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract  # noqa: E402

H, W, DEG, LAT, LON = None, None, None, None, None
GRIDS = {"5.625": (32, 64), "2.8125": (64, 128), "1.40625": (128, 256), "0.703125": (256, 512), "1.5": (121, 240)}


def set_grid(deg):
    """Equiangular grid without poles: ClimODE's 64x32, or a 2, 4, 8 times finer grid whose cells split the 64x32
    cells exactly in both directions (longitudes shifted accordingly), so that the area-weighted average of the
    fine cells IS the 5.625 deg analysis (scripts/coarsen_ens.py) and forecasts score on the same truth."""
    global H, W, DEG, LAT, LON
    H, W = GRIDS[deg]
    DEG = deg
    if deg == "1.5":  # KAI's grid: equiangular WITH poles (pgcfm.contract), not nested in the 64x32 grid
        LAT, LON = np.linspace(-90.0, 90.0, H), 1.5 * np.arange(W)
        return
    LAT = -90 + float(deg) * (np.arange(H) + 0.5)  # S -> N
    LON = float(deg) * np.arange(W) if deg == "5.625" else -2.8125 + float(deg) * (np.arange(W) + 0.5)


set_grid("5.625")
SURF = {"t2m": "t2m", "sp": "sp", "tisr": "tisr", "tcwv": "tcwv", "msl": "mslp", "skt": "skt"}
CLIMODE = {"z": ("z", 500, "geopotential_500"), "t": ("t", 850, "temperature_850"),
           "t2m": ("t2m", None, "2m_temperature"), "u10": ("u10", None, "10m_u_component_of_wind"),
           "v10": ("v10", None, "10m_v_component_of_wind")}
HOURS = (0, 6, 12, 18)
ALIASES = {"mslp": ("mslp", "msl", "mean_sea_level_pressure"), "time": ("time", "valid_time", "date"),
           "level": ("level", "pressure_level", "isobaricInhPa", "plev", "lev"),
           "latitude": ("latitude", "lat"), "longitude": ("longitude", "lon"),
           "z": ("z", "geopotential"), "q": ("q", "specific_humidity"), "t": ("t", "temperature"),
           "u": ("u", "u_component_of_wind"), "v": ("v", "v_component_of_wind"),
           "t2m": ("t2m", "2t", "2m_temperature"), "sp": ("sp", "surface_pressure"),
           "tisr": ("tisr", "toa_incident_solar_radiation"), "tcwv": ("tcwv", "total_column_water_vapour"),
           "skt": ("skt", "skin_temperature")}
# grid means of a day that tell the units apart (z: geopotential, not height; pressure in Pa, not hPa; q in kg/kg)
UNITS = {"z500": (4.5e4, 6.0e4), "t850": (255.0, 295.0), "q700": (5e-4, 8e-3), "t2m": (255.0, 300.0),
         "sp": (8.5e4, 1.02e5), "msl": (9.8e4, 1.03e5), "tcwv": (5.0, 45.0), "skt": (255.0, 300.0)}
PATTERN = "{ymd}.nc"  # file of one UTC day under --src; --pattern "{year}/{ymd}.nc" for per-year folders


def var(d, name):
    """d[name] under the archive's own spelling (ERA5 CDS: msl, valid_time, pressure_level)."""
    for n in ALIASES.get(name, (name,)):
        if n in d.variables:
            return d[n]
    raise KeyError(f"{name} (tried {ALIASES.get(name, (name,))}) not in {sorted(d.variables)}")


def levels_of(d):
    """Pressure levels of a file in hPa (a Pa axis is converted)."""
    lev = np.asarray(var(d, "level")[:], np.float64)
    return [int(round(x)) for x in (lev / 100.0 if lev.max() > 2000 else lev)]


def day_path(src, day):
    ymd = str(day).replace("-", "")
    return os.path.join(src, PATTERN.format(ymd=ymd, year=ymd[:4], ymd_dash=str(day)))


def check(src, year):
    """Print what the converter needs from one day file and stop with a clear message if it is not there."""
    day = np.datetime64(f"{year}-01-01")
    f = day_path(src, day)
    print("file:", f, "exists:", os.path.exists(f))
    with nc.Dataset(f) as d:
        print("variables:", sorted(d.variables))
        t = var(d, "time")
        hrs = [x.hour for x in nc.num2date(t[:], t.units)]
        lev = levels_of(d)
        print("hours:", hrs, "levels:", lev, "lat:", var(d, "latitude").shape, "lon:", var(d, "longitude").shape)
        need_v = list(contract.UPPER_AIR_VARS) + [n for n in SURF.values() if n != "mslp"] + ["u10", "v10"]
        miss = [v for v in need_v if v not in d.variables] + ([] if any(a in d.variables for a in ALIASES["mslp"]) else ["msl"])
        miss_h = [h for h in HOURS if h not in hrs]
        miss_l = [p for p in contract.LEVELS_HPA if p not in lev]
    ok = not (miss or miss_h or miss_l)
    print("missing variables:", miss, "missing hours:", miss_h, "missing levels:", miss_l)
    print("OK: the converter can read this archive" if ok else
          "NOT OK: send this output (daily means without 00/06/12/18 UTC cannot feed the 6-hourly model)")
    return ok


def overlap(src_lo, src_hi, dst_lo, dst_hi):
    """(n_dst, n_src) overlap lengths of intervals."""
    return np.clip(np.minimum(src_hi[None], dst_hi[:, None]) - np.maximum(src_lo[None], dst_lo[:, None]), 0, None)


def weights(src_lat, src_lon):
    """Row-normalized area weights: out = Wy @ x @ Wx.T for x (n_lat_src, n_lon_src)."""
    d = abs(src_lat[1] - src_lat[0])
    lo, hi = np.clip(src_lat - d / 2, -90, 90), np.clip(src_lat + d / 2, -90, 90)
    s = lambda q: np.sin(np.deg2rad(q))  # noqa: E731
    wy = overlap(s(lo), s(hi), s(np.clip(LAT - float(DEG) / 2, -90, 90)), s(np.clip(LAT + float(DEG) / 2, -90, 90)))
    dl = src_lon[1] - src_lon[0]
    wx = np.zeros((W, len(src_lon)))
    for shift in (-360.0, 0.0, 360.0):  # periodic in longitude
        wx += overlap(src_lon - dl / 2 + shift, src_lon + dl / 2 + shift, LON - float(DEG) / 2, LON + float(DEG) / 2)
    return wy / wy.sum(1, keepdims=True), wx / wx.sum(1, keepdims=True)


def regrid(x, wy, wx):
    """(..., 721, 1440) -> (..., H, W) float32."""
    return (wy @ np.asarray(x, np.float64) @ wx.T).astype(np.float32)


# ---------------------------------------------------------------------------- a daily-mean archive (--src-daily)
def _keys(day):
    """Spellings of the date, the month and the year that a file name can carry: {placeholder: text}, three sets."""
    d = str(day)
    return ({"ymd": d.replace("-", ""), "ymd_dash": d, "ymd_us": d.replace("-", "_")},
            {"ym": d[:7].replace("-", ""), "ym_dash": d[:7], "ym_us": d[:7].replace("-", "_")}, {"year": d[:4]})


def _fmt(pattern, day):
    k = {}
    for part in _keys(day):
        k.update(part)
    return pattern.format(var="*", **k)


def daily_patterns(src, day, pattern="auto"):
    """Patterns (relative to ``src``) of the files that hold one day of a daily-mean archive. ``auto``: the .nc
    files up to two folders deep whose path carries the date, else the month, else the year; their names with the
    date replaced by placeholders are the patterns of every other day (no directory scan per day)."""
    if pattern != "auto":
        return [pattern]
    full, month, year = _keys(day)
    for keys in (full, month, year):
        hits = sorted({f for depth in ("", "*", os.path.join("*", "*")) for k in keys.values() for ext in ("nc", "nc4")
                       for f in glob.glob(os.path.join(src, depth, f"*{k}*.{ext}"))})
        if hits:
            out = []
            for f in hits:
                rel = os.path.relpath(f, src)
                for part in (full, month, year):  # the longest spelling first; the year also in a folder name
                    for name, text in sorted(part.items(), key=lambda kv: -len(kv[1])):
                        rel = rel.replace(text, "{" + name + "}")
                out.append(rel)
            return sorted(set(out))
    return []


def daily_files(src, day, patterns):
    return sorted({f for q in patterns for f in glob.glob(os.path.join(src, _fmt(q, day)))})


def _time_index(d, day, path=""):
    """Position of ``day`` on the file's time axis: None = no time axis, -1 = the day is not in this file.
    A file with one time (or none) belongs to the day when its time says so or when its name carries the date
    (a daily mean may be stamped at the end of its day)."""
    named = any(k in os.path.basename(path) for k in _keys(day)[0].values())
    try:
        tv = var(d, "time")
        tt = nc.num2date(np.atleast_1d(tv[:]), tv.units, getattr(tv, "calendar", "standard"))
        dates = [np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}") for t in tt]
    except (KeyError, AttributeError, ValueError):  # no time axis, or one without units
        return None if named else -1
    if len(dates) == 1:
        return (None if tv.ndim == 0 else 0) if (named or dates[0] == day) else -1
    return dates.index(day) if day in dates else -1


def _take(v, ti):
    """One day of a variable as (levels or nothing, lat, lon); a leading time axis is indexed, others of size 1 dropped."""
    x = v[ti] if (ti is not None and v.dimensions[0] in ALIASES["time"]) else v[:]
    x = np.asarray(x)
    if x.ndim > 2 and x.shape[0] == 1 and (v.dimensions[0] in ALIASES["time"] or x.ndim > 3):
        x = x[0]
    return x


def daily_fields(files, day, wy_wx=None):
    """(67, H, W) KAI channels of one day from a daily-mean archive, the orography slot left 0; ``notes``: what was
    filled in. The variables are looked up in every file of the day."""
    f = np.zeros((contract.N_CHANNELS, H, W), dtype=np.float32)
    need = {v: None for v in contract.UPPER_AIR_VARS}
    need.update({c: None for c in SURF})
    for path in files:
        with nc.Dataset(path) as d:
            d.set_auto_mask(False)
            ti = _time_index(d, day, path)
            if ti == -1:
                continue
            if wy_wx is None:
                wy_wx = weights(np.asarray(var(d, "latitude")[:], np.float64), np.asarray(var(d, "longitude")[:], np.float64))
            for name in [n for n, got in need.items() if got is None]:
                try:
                    v = var(d, SURF.get(name, name))
                except KeyError:
                    continue
                x = _take(v, ti)
                if name in contract.UPPER_AIR_VARS:
                    lev = levels_of(d)
                    x = x[[lev.index(p) for p in contract.LEVELS_HPA]]
                    f[contract.level_slice(name)] = regrid(x, *wy_wx)
                else:
                    f[contract.CHANNEL_INDEX[name]] = regrid(x.reshape(x.shape[-2:]), *wy_wx)
                need[name] = path
    notes = []
    if need["skt"] is None and need["t2m"] is not None:
        f[contract.CHANNEL_INDEX["skt"]] = f[contract.CHANNEL_INDEX["t2m"]]
        need["skt"] = "t2m"
        notes.append("skt := t2m")
    if need["tisr"] is None:
        from scripts.wb2_to_kai import tisr_1h
        t = [np.datetime64(f"{day}T{h:02d}") for h in HOURS]
        f[contract.CHANNEL_INDEX["tisr"]] = tisr_1h(t, LAT, LON).mean(0)
        need["tisr"] = "computed"
        notes.append("tisr computed")
    miss = [n for n, got in need.items() if got is None]
    if miss:
        raise KeyError(f"{day}: {miss} not found in {files}")
    return f, notes, wy_wx


def check_daily(src, year, pattern):
    """Print what the daily-mean reader finds for the first day of the year; True when it can build that day."""
    day = np.datetime64(f"{year}-01-01")
    pats = daily_patterns(src, day, pattern)
    files = daily_files(src, day, pats)
    print("source:", src, "| patterns:", pats, "| files of", day, ":", len(files))
    if not files:
        for root in [src] + sorted(q for q in glob.glob(os.path.join(src, "*")) if os.path.isdir(q))[:2]:
            print("listing of", root, ":", sorted(os.listdir(root))[:30] if os.path.isdir(root) else "NOT A DIRECTORY")
        print('NOT OK: no file of that day found. Send this output, or give the layout with --pattern, e.g. '
              '"{year}/{ymd}.nc", "{var}_{year}.nc", "{year}/*.nc" (placeholders ymd, ymd_dash, ymd_us, ym, ym_dash, year, var)')
        return False
    for path in files[:20]:
        with nc.Dataset(path) as d:
            print("file:", path)
            print("  variables:", {k: (v.dimensions, v.shape) for k, v in d.variables.items()})
            try:
                tv = var(d, "time")
                tt = nc.num2date(np.atleast_1d(tv[:]), tv.units)
                print("  times:", len(tt), "first", tt[0], "last", tt[-1], "| position of the day:", _time_index(d, day, path))
            except KeyError:
                print("  no time axis")
            try:
                print("  levels (hPa):", levels_of(d))
            except KeyError:
                print("  no level axis")
    try:
        f, notes, _ = daily_fields(files, day)
        ok = bool(np.isfinite(f).all())
        print("filled in:", notes or "nothing", "| all values finite:", ok)
        warn = 0
        for c in ("z500", "t850", "q700", "u250", "t2m", "sp", "msl", "tcwv", "tisr", "skt"):
            x = f[contract.CHANNEL_INDEX[c]]
            lo, hi = UNITS.get(c, (-np.inf, np.inf))
            bad = not lo <= float(x.mean()) <= hi
            warn += bad
            print(f"  {c}: min {x.min():.6g} mean {x.mean():.6g} max {x.max():.6g}"
                  + (f"   WARNING: a mean of {lo:g} to {hi:g} is expected (other units?)" if bad else ""))
        if warn:
            print("WARNING: the physical terms of the core assume ERA5's units (z = geopotential m2 s-2, sp / msl Pa, "
                  "q kg/kg, tcwv kg m-2). Send this output before training on these files.")
    except Exception as e:  # noqa: BLE001  (the message is the result of the check)
        print("NOT OK:", repr(e))
        print("Send this output.")
        return False
    print("OK: the converter can read this archive as daily means" if ok else "NOT OK: non-finite values, send this output")
    return ok


def day_fields(path, wy, wx):
    """(4, 66 + 1 unused orog slot, H, W) KAI channels and {climode var: (4, H, W)} of one day."""
    f = np.zeros((len(HOURS), contract.N_CHANNELS, H, W), dtype=np.float32)
    cm = {}
    with nc.Dataset(path) as d:
        d.set_auto_mask(False)
        tv = var(d, "time")
        hrs = [t.hour for t in nc.num2date(tv[:], tv.units)]
        ti = [hrs.index(h) for h in HOURS]
        lev = levels_of(d)
        for v in contract.UPPER_AIR_VARS:
            sl = contract.level_slice(v)
            for k, ti_ in enumerate(ti):
                x = var(d, v)[ti_]
                f[k, sl] = regrid(x[[lev.index(p) for p in contract.LEVELS_HPA]], wy, wx)
        for c, name in SURF.items():
            f[:, contract.CHANNEL_INDEX[c]] = regrid(var(d, name)[ti], wy, wx)
        for short, (name, level, _) in CLIMODE.items():
            if level is None:
                cm[short] = f[:, contract.CHANNEL_INDEX[name]] if name in SURF else regrid(d[name][ti], wy, wx)
            else:
                cm[short] = f[:, contract.CHANNEL_INDEX[f"{name}{level}"]]
    return f, cm


def times_of(year):
    t = np.arange(np.datetime64(f"{year}-01-01T00"), np.datetime64(f"{year + 1}-01-01T00"), np.timedelta64(6, "h"))
    return t


def write_kai(path, f, year):
    T = len(f)
    with nc.Dataset(path + ".tmp", "w", format="NETCDF4") as out:
        out.createDimension("time", T)
        out.createDimension("channel", contract.N_CHANNELS)
        out.createDimension("latitude", H)
        out.createDimension("longitude", W)
        out.createVariable("fields", "f4", ("time", "channel", "latitude", "longitude"),
                           zlib=True, complevel=1, shuffle=True)[:] = f
        t = out.createVariable("time", "i8", ("time",))
        t.units, t.calendar = f"hours since {year}-01-01", "proleptic_gregorian"
        t[:] = 6 * np.arange(T, dtype=np.int64)
        ch = out.createVariable("channel", str, ("channel",))
        for i, n in enumerate(contract.CHANNEL_NAMES):
            ch[i] = n
        out.createVariable("latitude", "f8", ("latitude",))[:] = LAT
        out.createVariable("longitude", "f8", ("longitude",))[:] = LON
        out.title = f"ERA5 (local 0.25 deg hourly, conservative to {W}x{H}) 6-hourly snapshots, KAI channel layout"
    os.replace(path + ".tmp", path)


def write_climode(root, year, cm):
    import xarray as xr
    t = times_of(year)
    for short, (_, _, folder) in CLIMODE.items():
        os.makedirs(os.path.join(root, folder), exist_ok=True)
        p = os.path.join(root, folder, f"{folder}_{year}_{DEG}deg.nc")
        if os.path.exists(p):
            continue
        xr.Dataset({short: (("time", "lat", "lon"), cm[short])}, coords={"time": t, "lat": LAT, "lon": LON}) \
            .to_netcdf(p + ".tmp", format="NETCDF4")
        os.replace(p + ".tmp", p)


def constants(root, orog_file, lsm_file):
    import xarray as xr
    p = os.path.join(root, "constants", f"constants_{DEG}deg.nc")
    if os.path.exists(p):
        return
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with xr.open_dataset(orog_file) as o, xr.open_dataset(lsm_file) as m:
        wy, wx = weights(o.latitude.values.astype(np.float64), o.longitude.values.astype(np.float64))
        z = regrid(o["z"].values[0], wy, wx)
        lsm = regrid(m["lsm"].values[0], wy, wx)
    lon2d, lat2d = np.meshgrid(LON, LAT)
    xr.Dataset({"orography": (("lat", "lon"), z / contract.CONST.gravity), "lsm": (("lat", "lon"), lsm),
                "geopotential_at_surface": (("lat", "lon"), z),
                "lat2d": (("lat", "lon"), lat2d), "lon2d": (("lat", "lon"), lon2d)},
               coords={"lat": LAT, "lon": LON}).to_netcdf(p + ".tmp")
    os.replace(p + ".tmp", p)


def year_job(src, out, year, orog_z, pattern=PATTERN, deg="5.625", daily=False, n_days=None, src_daily=False):
    global PATTERN
    PATTERN = pattern
    set_grid(deg)
    if src_daily:  # the archive holds daily means already: regrid them, day by day
        day_file = os.path.join(out, "kaiday", f"{year}.npy")
        if os.path.exists(day_file):
            return f"{year} exists"
        days = np.arange(np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01"))[:n_days]
        pats = daily_patterns(src, days[0], pattern)
        if not daily_files(src, days[0], pats):
            raise FileNotFoundError(f"{year}: no file of {days[0]} under {src} (patterns {pats}); run --check --src-daily")
        os.makedirs(os.path.dirname(day_file), exist_ok=True)
        f = np.lib.format.open_memmap(day_file + ".tmp.npy", mode="w+", dtype=np.float32,
                                      shape=(len(days), contract.N_CHANNELS, H, W))
        wy_wx, seen = None, set()
        for i, day in enumerate(days):
            files = daily_files(src, day, pats)
            if not files and pattern == "auto":  # e.g. month folders: look again, keep what is found
                pats = sorted(set(pats) | set(daily_patterns(src, day, pattern)))
                files = daily_files(src, day, pats)
            fd, notes, wy_wx = daily_fields(files, day, wy_wx)
            fd[contract.CHANNEL_INDEX["orog"]] = orog_z
            if not np.isfinite(fd).all():
                raise RuntimeError(f"{day}: non-finite values")
            f[i] = fd
            seen.update(notes)
            if (i + 1) % 60 == 0:
                print(f"  {year}: {i + 1}/{len(days)} days", flush=True)
        f.flush()
        del f
        os.replace(day_file + ".tmp.npy", day_file)
        return f"{year} done (daily means of the archive{', ' + ', '.join(sorted(seen)) if seen else ''})"
    kai = os.path.join(out, "kai6h", f"{year}.nc")
    root = os.path.join(out, "era5_data")
    day_file = os.path.join(out, "kaiday", f"{year}.npy")
    if daily and os.path.exists(day_file):
        return f"{year} exists"
    if not daily and os.path.exists(kai) and os.path.exists(
            os.path.join(root, "geopotential_500", f"geopotential_500_{year}_{DEG}deg.nc")):
        return f"{year} exists"
    days = np.arange(np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01"))[:n_days]
    with nc.Dataset(day_path(src, days[0])) as d:
        wy, wx = weights(np.asarray(var(d, "latitude")[:], np.float64), np.asarray(var(d, "longitude")[:], np.float64))
    if daily:  # the tendency core +: the mean of the four snapshots of every day, written day by day
        os.makedirs(os.path.dirname(day_file), exist_ok=True)
        f = np.lib.format.open_memmap(day_file + ".tmp.npy", mode="w+", dtype=np.float32,
                                      shape=(len(days), contract.N_CHANNELS, H, W))
        for i, day in enumerate(days):
            fd, _ = day_fields(day_path(src, day), wy, wx)
            fd[:, contract.CHANNEL_INDEX["orog"]] = orog_z
            if not np.isfinite(fd).all():
                raise RuntimeError(f"{day}: non-finite values")
            f[i] = fd.mean(0)
            if (i + 1) % 60 == 0:
                print(f"  {year}: {i + 1}/{len(days)} days", flush=True)
        f.flush()
        del f
        os.replace(day_file + ".tmp.npy", day_file)
        return f"{year} done (daily means)"
    f = np.empty((4 * len(days), contract.N_CHANNELS, H, W), dtype=np.float32)
    cm = {k: np.empty((4 * len(days), H, W), dtype=np.float32) for k in CLIMODE}
    for i, day in enumerate(days):
        fd, cd = day_fields(day_path(src, day), wy, wx)
        f[4 * i:4 * i + 4] = fd
        for k in cm:
            cm[k][4 * i:4 * i + 4] = cd[k]
        if (i + 1) % 60 == 0:
            print(f"  {year}: {i + 1}/{len(days)} days", flush=True)
    f[:, contract.CHANNEL_INDEX["orog"]] = orog_z
    if not np.isfinite(f).all():
        raise RuntimeError(f"{year}: non-finite values")
    os.makedirs(os.path.dirname(kai), exist_ok=True)
    write_kai(kai, f, year)
    write_climode(root, year, cm)
    return f"{year} done"


def main(argv=None):
    global PATTERN
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/lustre/home/mahmed/ERA5/hourly",
                   help="one file per UTC day with hourly fields (ERA5_0p25_DAILY holds daily means: --check says NOT OK)")
    p.add_argument("--grid", default="5.625", choices=sorted(GRIDS), help="output grid in degrees")
    p.add_argument("--orog", default="/lustre/home/mahmed/ERA5_0p25_DAILY/era5_orography_0p25.nc")
    p.add_argument("--lsm", default="/lustre/home/mahmed/ERA5_0p25_DAILY/era5_land_sea_mask_0p25.nc")
    p.add_argument("--out", required=True)
    p.add_argument("--years", type=int, nargs="+", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--pattern", default=None,
                   help='day file under --src: "{ymd}.nc" (default), "{year}/{ymd}.nc", ...; with --src-daily the '
                        'default is "auto" (found from the first day), wildcards and {var} are allowed')
    p.add_argument("--src-daily", action="store_true",
                   help="--src holds daily means (ERA5_0p25_DAILY): they are regridded as they are; needs --daily-mean")
    p.add_argument("--check", action="store_true", help="only check the first year's first day file and stop")
    p.add_argument("--daily-mean", action="store_true",
                   help="write only <out>/kaiday/<year>.npy, the daily means of 00/06/12/18 UTC (tendency core +)")
    p.add_argument("--days", type=int, default=None, help="only the first n days of every year (smoke test)")
    a = p.parse_args(argv)
    if a.src_daily and not (a.daily_mean or a.check):
        p.error("--src-daily gives daily means only: add --daily-mean (the 6-hourly files need the hourly archive)")
    a.pattern = a.pattern or ("auto" if a.src_daily else PATTERN)
    PATTERN = a.pattern
    set_grid(a.grid)
    if a.check:
        sys.exit(0 if (check_daily(a.src, a.years[0], a.pattern) if a.src_daily else check(a.src, a.years[0])) else 1)
    constants(os.path.join(a.out, "era5_data"), a.orog, a.lsm)
    import xarray as xr
    with xr.open_dataset(os.path.join(a.out, "era5_data", "constants", f"constants_{DEG}deg.nc")) as c:
        orog_z = c["geopotential_at_surface"].values.astype(np.float32)
    with ProcessPoolExecutor(a.workers) as ex:
        n = len(a.years)
        for r in ex.map(year_job, [a.src] * n, [a.out] * n, a.years, [orog_z] * n, [a.pattern] * n, [a.grid] * n,
                        [a.daily_mean] * n, [a.days] * n, [a.src_daily] * n):
            print(r, flush=True)


if __name__ == "__main__":
    main()
