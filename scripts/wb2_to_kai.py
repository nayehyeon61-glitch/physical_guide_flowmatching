"""Real ERA5 at ClimODE's resolution (5.625 deg, 32x64) from the public WeatherBench2 zarr.

Writes two things for the ClimODE comparison (docs/step11_climode.md):

1. KAI-format yearly files (``<out>/kai/<year>.nc``, var ``fields`` (T, 67, 32, 64)) with one
   snapshot per day at ``--hour`` UTC, so a 24 h PG-CFM step lands on ClimODE's 6-hourly
   truth at leads 24, 48, ... h. ``--all-hours`` writes every 6 h time instead, to
   ``<out>/kai6h/<year>.nc`` (step 13 sequences; zlib-compressed, about 0.8 GB a year raw).
   Channel order and units follow pgcfm.contract.
   Departures from the KAI contract (all forced by what WB2 offers at this grid):
     * instantaneous snapshots, not daily means (ClimODE scores instantaneous fields)
     * grid 64x32 equiangular *without* poles (lat -87.1875..87.1875), same as ClimODE
     * tisr is computed (ERA5 convention: 1 h accumulation ending at the valid time),
       the zarr's radiation flux is all NaN
     * skt is not in the zarr: skt := t2m (copy)
2. WeatherBench1-style files for ClimODE's own loader (``<out>/era5_data/<var>/<year>.nc`` with
   vars z, t, t2m, u10, v10 at 6 h, plus ``constants/constants_5.625deg.nc``).

    python scripts/wb2_to_kai.py --out data/era5_5p625 --years 2006 2007 ... 2018
"""

import argparse
import os
import sys

import netCDF4 as nc
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract  # noqa: E402

URLS = {"64x32": "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
        "128x64": "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_conservative.zarr"}
URL = URLS["64x32"]
UPPER = {"z": "geopotential", "q": "specific_humidity", "t": "temperature",
         "u": "u_component_of_wind", "v": "v_component_of_wind"}
SURF = {"t2m": "2m_temperature", "sp": "surface_pressure", "tcwv": "total_column_water_vapour",
        "msl": "mean_sea_level_pressure"}
CLIMODE = {"z": ("geopotential", 500, "geopotential_500"), "t": ("temperature", 850, "temperature_850"),
           "t2m": ("2m_temperature", None, "2m_temperature"),
           "u10": ("10m_u_component_of_wind", None, "10m_u_component_of_wind"),
           "v10": ("10m_v_component_of_wind", None, "10m_v_component_of_wind")}
SOLAR_CONSTANT = 1361.0


def open_wb2(grid: str = "64x32"):
    return xr.open_zarr(URLS[grid], storage_options={"client_kwargs": {"trust_env": True}})


def tisr_1h(times, lat_deg, lon_deg, n_sub=12):
    """ERA5-style TOA incident solar radiation [J m-2]: integral over the hour ending at each time."""
    lat = np.deg2rad(lat_deg)[:, None]
    out = np.zeros((len(times), len(lat_deg), len(lon_deg)), dtype=np.float32)
    for i, t in enumerate(times):
        t = np.datetime64(t, "s")
        doy = (t - t.astype("datetime64[Y]")).astype(float) / 86400.0
        acc = 0.0
        for s in range(n_sub):
            tt = doy - (s + 0.5) / n_sub / 24.0  # fractional day of year, sub-step mid-point
            decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (tt + 284 - 1) / 365.0)
            utc = (tt % 1.0) * 24.0
            ha = np.deg2rad(15.0 * (utc + lon_deg[None, :] / 15.0 - 12.0))
            cz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
            acc = acc + SOLAR_CONSTANT * np.clip(cz, 0.0, None) * 3600.0 / n_sub
        out[i] = acc
    return out


def kai_year(ds, year, hour, path, days_step=None, grid="64x32"):
    """``hour=None``: every 6 h time of the year; ``days_step`` keeps only every n-th day (all its times:
    the step 18 resolution test needs 6 h triples, not whole years)."""
    sel = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    if hour is not None:
        sel = sel.sel(time=sel.time.dt.hour == hour)
    if days_step:
        sel = sel.sel(time=((sel.time.dt.dayofyear - 1) % days_step) == 0)
    times = sel.time.values
    lat, lon = sel.latitude.values, sel.longitude.values
    T, H, W = len(times), len(lat), len(lon)
    f = np.empty((T, contract.N_CHANNELS, H, W), dtype=np.float32)
    for v, name in UPPER.items():
        x = sel[name].sel(level=contract.LEVELS_HPA).transpose("time", "level", "latitude", "longitude").values
        f[:, contract.level_slice(v)] = x
    for v, name in SURF.items():
        f[:, contract.CHANNEL_INDEX[v]] = sel[name].transpose("time", "latitude", "longitude").values
    f[:, contract.CHANNEL_INDEX["skt"]] = f[:, contract.CHANNEL_INDEX["t2m"]]
    f[:, contract.CHANNEL_INDEX["tisr"]] = tisr_1h(times, lat, lon)
    f[:, contract.CHANNEL_INDEX["orog"]] = ds.geopotential_at_surface.transpose("latitude", "longitude").values
    if not np.isfinite(f).all():
        raise RuntimeError(f"{year}: non-finite values")
    with nc.Dataset(path + ".tmp", "w", format="NETCDF4") as out:
        out.createDimension("time", T)
        out.createDimension("channel", contract.N_CHANNELS)
        out.createDimension("latitude", H)
        out.createDimension("longitude", W)
        out.createVariable("fields", "f4", ("time", "channel", "latitude", "longitude"),
                           zlib=hour is None, complevel=1, shuffle=True)[:] = f
        t = out.createVariable("time", "i8", ("time",))
        t.units, t.calendar = f"hours since {year}-01-01", "proleptic_gregorian"
        t[:] = ((times - np.datetime64(f"{year}-01-01")) / np.timedelta64(1, "h")).astype(np.int64)
        ch = out.createVariable("channel", str, ("channel",))
        for i, n in enumerate(contract.CHANNEL_NAMES):
            ch[i] = n
        out.createVariable("latitude", "f8", ("latitude",))[:] = lat
        out.createVariable("longitude", "f8", ("longitude",))[:] = lon
        when = "6-hourly" if hour is None else f"{hour:02d} UTC"
        out.title = f"ERA5 (WeatherBench2 {grid} conservative) {when} snapshots, KAI channel layout"
    os.replace(path + ".tmp", path)
    return f


def climode_year(ds, year, root):
    sel = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    for short, (name, level, folder) in CLIMODE.items():
        os.makedirs(os.path.join(root, folder), exist_ok=True)
        x = sel[name] if level is None else sel[name].sel(level=level, drop=True)
        x = x.transpose("time", "latitude", "longitude").rename(latitude="lat", longitude="lon").rename(short)
        p = os.path.join(root, folder, f"{folder}_{year}_5.625deg.nc")
        x.load().to_dataset().to_netcdf(p + ".tmp", format="NETCDF4")
        os.replace(p + ".tmp", p)


def constants(ds, root):
    os.makedirs(os.path.join(root, "constants"), exist_ok=True)
    lat, lon = ds.latitude.values, ds.longitude.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    c = xr.Dataset({"orography": (("lat", "lon"), ds.geopotential_at_surface.transpose("latitude", "longitude").values
                                  / contract.CONST.gravity),
                    "lsm": (("lat", "lon"), ds.land_sea_mask.transpose("latitude", "longitude").values),
                    "lat2d": (("lat", "lon"), lat2d), "lon2d": (("lat", "lon"), lon2d)},
                   coords={"lat": lat, "lon": lon})
    c.to_netcdf(os.path.join(root, "constants", "constants_5.625deg.nc"))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--years", type=int, nargs="+", required=True)
    p.add_argument("--hour", type=int, default=12, help="UTC hour of the daily KAI snapshot")
    p.add_argument("--all-hours", action="store_true", help="every 6 h time, to <out>/kai6h")
    p.add_argument("--no-climode", action="store_true")
    p.add_argument("--grid", default="64x32", choices=list(URLS), help="128x64 = 2.8125 deg (step 18)")
    p.add_argument("--days-step", type=int, default=None, help="keep every n-th day only (all its times)")
    a = p.parse_args(argv)
    ds = open_wb2(a.grid)
    kai = os.path.join(a.out, "kai6h" if a.all_hours else "kai")
    os.makedirs(kai, exist_ok=True)
    if not a.no_climode and not os.path.exists(os.path.join(a.out, "era5_data", "constants", "constants_5.625deg.nc")):
        constants(ds, os.path.join(a.out, "era5_data"))
    for y in a.years:
        path = os.path.join(kai, f"{y}.nc")
        if not os.path.exists(path):
            kai_year(ds, y, None if a.all_hours else a.hour, path, a.days_step, a.grid)
        if not a.no_climode:
            climode_year(ds, y, os.path.join(a.out, "era5_data"))
        print(f"{y} done", flush=True)


if __name__ == "__main__":
    main()
