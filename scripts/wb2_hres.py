"""Step 14 (H-res): ClimODE's five variables at 2.8125 deg (64 x 128) from the public WeatherBench2 zarr, for the
zero-shot resolution test of the step 12 / 14 cores (scripts/run_step14_hres.py).

    source   1959-2022-6h-128x64_equiangular_conservative.zarr (the same conservative regridding as the
             64 x 32 store of scripts/wb2_to_kai.py, latitudes without the poles, -88.59 .. 88.59)
    output   <out>/era5_hres_<year>.npz: x (T, 5, 64, 128) float32 every 6 h in physical units (z500, t850,
             t2m, u10, v10, the order of scripts/run_step11_climode.py VARS), lat, lon;
             <out>/era5_hres_constants.npz: orography (surface geopotential) and the land-sea mask

    python scripts/wb2_hres.py --out /home/claude/era5_hres --years 2017 2018
"""

import argparse
import os

import numpy as np
import xarray as xr

URL = "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_conservative.zarr"
FIELDS = [("geopotential", 500), ("temperature", 850), ("2m_temperature", None), ("10m_u_component_of_wind", None),
          ("10m_v_component_of_wind", None)]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--years", type=int, nargs="+", required=True)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    ds = xr.open_zarr(URL, storage_options={"client_kwargs": {"trust_env": True}})
    lat, lon = ds.latitude.values, ds.longitude.values
    order = np.argsort(lat)  # ascending, as the 5.625 deg files
    const = os.path.join(a.out, "era5_hres_constants.npz")
    if not os.path.exists(const):
        np.savez(const, orography=ds.geopotential_at_surface.transpose("latitude", "longitude").values[order],
                 lsm=ds.land_sea_mask.transpose("latitude", "longitude").values[order], lat=lat[order], lon=lon)
    for y in a.years:
        path = os.path.join(a.out, f"era5_hres_{y}.npz")
        if os.path.exists(path):
            continue
        sel = ds.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))
        xs = []
        for name, level in FIELDS:
            v = sel[name] if level is None else sel[name].sel(level=level, drop=True)
            xs.append(v.transpose("time", "latitude", "longitude").values[:, order].astype(np.float32))
            print(f"[hres] {y} {name} {xs[-1].shape}", flush=True)
        x = np.stack(xs, 1)
        if not np.isfinite(x).all():
            raise RuntimeError(f"{y}: non-finite values")
        np.savez_compressed(path, x=x, lat=lat[order], lon=lon)
        print(f"[hres] {path} {x.shape}", flush=True)


if __name__ == "__main__":
    main()
