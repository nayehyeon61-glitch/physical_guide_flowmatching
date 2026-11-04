"""Step 13 data from a local ERA5 0.25 deg hourly archive instead of the WeatherBench2 zarr.

Same outputs and layout as scripts/wb2_to_kai.py --all-hours (``<out>/kai6h/<year>.nc`` and the
WeatherBench1-style ``<out>/era5_data``), built from one ``YYYYMMDD.nc`` per UTC day (0.25 deg,
721x1440, lat N -> S, 13 levels, vars z q t u v u10 v10 mslp tcwv sp skt t2m tisr, ...):

    * 00/06/12/18 UTC snapshots, first-order conservative regridding (exact cell-overlap area
      weights, separable in lat/lon) onto ClimODE's 64x32 grid without poles
    * tisr and skt are the archive's own fields (wb2_to_kai.py had to compute tisr and copy t2m)
    * orography / lsm from the 0.25 deg constant files, regridded the same way

The archive is only read. One process per year:

    python scripts/local_era5_to_kai.py --src /lustre/home/mahmed/ERA5/hourly --out data/era5_5p625 \
        --years 1979 1980 ... 2018 --workers 20
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import netCDF4 as nc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract  # noqa: E402

H, W = 32, 64
LAT = -90 + 5.625 * (np.arange(H) + 0.5)  # -87.1875 .. 87.1875, S -> N
LON = 5.625 * np.arange(W)
SURF = {"t2m": "t2m", "sp": "sp", "tisr": "tisr", "tcwv": "tcwv", "msl": "mslp", "skt": "skt"}
CLIMODE = {"z": ("z", 500, "geopotential_500"), "t": ("t", 850, "temperature_850"),
           "t2m": ("t2m", None, "2m_temperature"), "u10": ("u10", None, "10m_u_component_of_wind"),
           "v10": ("v10", None, "10m_v_component_of_wind")}
HOURS = (0, 6, 12, 18)


def overlap(src_lo, src_hi, dst_lo, dst_hi):
    """(n_dst, n_src) overlap lengths of intervals."""
    return np.clip(np.minimum(src_hi[None], dst_hi[:, None]) - np.maximum(src_lo[None], dst_lo[:, None]), 0, None)


def weights(src_lat, src_lon):
    """Row-normalized area weights: out = Wy @ x @ Wx.T for x (n_lat_src, n_lon_src)."""
    d = abs(src_lat[1] - src_lat[0])
    lo, hi = np.clip(src_lat - d / 2, -90, 90), np.clip(src_lat + d / 2, -90, 90)
    s = lambda q: np.sin(np.deg2rad(q))  # noqa: E731
    wy = overlap(s(lo), s(hi), s(LAT - 2.8125), s(LAT + 2.8125))
    dl = src_lon[1] - src_lon[0]
    wx = np.zeros((W, len(src_lon)))
    for shift in (-360.0, 0.0, 360.0):  # periodic in longitude
        wx += overlap(src_lon - dl / 2 + shift, src_lon + dl / 2 + shift, LON - 2.8125, LON + 2.8125)
    return wy / wy.sum(1, keepdims=True), wx / wx.sum(1, keepdims=True)


def regrid(x, wy, wx):
    """(..., 721, 1440) -> (..., 32, 64) float32."""
    return (wy @ np.asarray(x, np.float64) @ wx.T).astype(np.float32)


def day_fields(path, wy, wx):
    """(4, 66 + 1 unused orog slot, 32, 64) KAI channels and {climode var: (4, 32, 64)} of one day."""
    f = np.zeros((len(HOURS), contract.N_CHANNELS, H, W), dtype=np.float32)
    cm = {}
    with nc.Dataset(path) as d:
        d.set_auto_mask(False)
        hrs = [t.hour for t in nc.num2date(d["time"][:], d["time"].units)]
        ti = [hrs.index(h) for h in HOURS]
        lev = list(np.asarray(d["level"][:]).astype(int))
        for v in contract.UPPER_AIR_VARS:
            sl = contract.level_slice(v)
            for k, ti_ in enumerate(ti):
                x = d[v][ti_]
                f[k, sl] = regrid(x[[lev.index(p) for p in contract.LEVELS_HPA]], wy, wx)
        for c, name in SURF.items():
            f[:, contract.CHANNEL_INDEX[c]] = regrid(d[name][ti], wy, wx)
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
        out.title = "ERA5 (local 0.25 deg hourly, conservative to 64x32) 6-hourly snapshots, KAI channel layout"
    os.replace(path + ".tmp", path)


def write_climode(root, year, cm):
    import xarray as xr
    t = times_of(year)
    for short, (_, _, folder) in CLIMODE.items():
        os.makedirs(os.path.join(root, folder), exist_ok=True)
        p = os.path.join(root, folder, f"{folder}_{year}_5.625deg.nc")
        if os.path.exists(p):
            continue
        xr.Dataset({short: (("time", "lat", "lon"), cm[short])}, coords={"time": t, "lat": LAT, "lon": LON}) \
            .to_netcdf(p + ".tmp", format="NETCDF4")
        os.replace(p + ".tmp", p)


def constants(root, orog_file, lsm_file):
    import xarray as xr
    p = os.path.join(root, "constants", "constants_5.625deg.nc")
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


def year_job(src, out, year, orog_z):
    kai = os.path.join(out, "kai6h", f"{year}.nc")
    root = os.path.join(out, "era5_data")
    if os.path.exists(kai) and os.path.exists(os.path.join(root, "geopotential_500", f"geopotential_500_{year}_5.625deg.nc")):
        return f"{year} exists"
    days = np.arange(np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01"))
    with nc.Dataset(os.path.join(src, str(days[0]).replace("-", "") + ".nc")) as d:
        wy, wx = weights(np.asarray(d["latitude"][:], np.float64), np.asarray(d["longitude"][:], np.float64))
    f = np.empty((4 * len(days), contract.N_CHANNELS, H, W), dtype=np.float32)
    cm = {k: np.empty((4 * len(days), H, W), dtype=np.float32) for k in CLIMODE}
    for i, day in enumerate(days):
        fd, cd = day_fields(os.path.join(src, str(day).replace("-", "") + ".nc"), wy, wx)
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
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/lustre/home/mahmed/ERA5/hourly")
    p.add_argument("--orog", default="/lustre/home/mahmed/ERA5_0p25_DAILY/era5_orography_0p25.nc")
    p.add_argument("--lsm", default="/lustre/home/mahmed/ERA5_0p25_DAILY/era5_land_sea_mask_0p25.nc")
    p.add_argument("--out", required=True)
    p.add_argument("--years", type=int, nargs="+", required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)
    constants(os.path.join(a.out, "era5_data"), a.orog, a.lsm)
    import xarray as xr
    with xr.open_dataset(os.path.join(a.out, "era5_data", "constants", "constants_5.625deg.nc")) as c:
        orog_z = c["geopotential_at_surface"].values.astype(np.float32)
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(year_job, [a.src] * len(a.years), [a.out] * len(a.years), a.years,
                        [orog_z] * len(a.years)):
            print(r, flush=True)


if __name__ == "__main__":
    main()
