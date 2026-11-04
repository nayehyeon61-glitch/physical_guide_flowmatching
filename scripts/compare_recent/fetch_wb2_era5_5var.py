"""ERA5 z500, t850, t2m, u10, v10 at 5.625 deg (WB2 64x32 conservative, the same zarr scripts/wb2_to_kai.py
uses for our truth) as one float32 array per year: <out>/<year>.npy (T, 5, 32, 64), 6-hourly, lat ascending.

    python scripts/compare_recent/fetch_wb2_era5_5var.py --out data/wb2_5var --years 2019 2020
"""
import argparse, os, time
import numpy as np, xarray as xr

URL = "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
VARS = [("geopotential", 500), ("temperature", 850), ("2m_temperature", None),
        ("10m_u_component_of_wind", None), ("10m_v_component_of_wind", None)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--years", type=int, nargs="+", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ds = xr.open_zarr(URL, storage_options={"client_kwargs": {"trust_env": True}})
    np.savez(os.path.join(a.out, "static.npz"), lat=ds.latitude.values, lon=ds.longitude.values,
             orography=ds.geopotential_at_surface.transpose("latitude", "longitude").values / 9.80665,
             lsm=ds.land_sea_mask.transpose("latitude", "longitude").values)
    for y in a.years:
        f = os.path.join(a.out, f"{y}.npy")
        if os.path.exists(f):
            continue
        t0 = time.time()
        sel = ds.sel(time=str(y))
        arr = []
        for name, lev in VARS:
            x = sel[name] if lev is None else sel[name].sel(level=lev)
            arr.append(x.transpose("time", "latitude", "longitude").values.astype(np.float32))
        np.save(f + ".tmp.npy", np.stack(arr, 1))
        os.replace(f + ".tmp.npy", f)
        print(y, np.stack(arr, 1).shape, f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
