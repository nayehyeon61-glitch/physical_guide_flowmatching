"""Score WeatherBench2's published 2020 forecasts (64x32 conservative) on our 179 ICs, 24..144 h.

Nothing is run: the forecasts are read from gs://weatherbench2 (public). Models without 12 UTC ICs
(SphericalCNN: 00 UTC) are scored at the 00 UTC IC 12 h before each of our ICs and are flagged.
Ensembles are scored with all members and with the first 8 (our member count; the non-fair CRPS
estimator favours larger ensembles).

    python scripts/compare_recent/wb2_forecasts.py --models keisler pangu ... --out runs/step27/wb2
"""
import argparse, os, sys, time
import numpy as np, xarray as xr
sys.path.insert(0, os.path.dirname(__file__))
import common as C

B = "https://storage.googleapis.com/weatherbench2/datasets/"
MODELS = {  # name: (path, member dim or None)
    "keisler": ("keisler/2020-64x32_equiangular_conservative.zarr", None),
    "sphericalcnn": ("sphericalcnn/2020-64x32_equiangular_conservative.zarr", None),
    "pangu": ("pangu/2018-2022_0012_64x32_equiangular_conservative.zarr", None),
    "graphcast": ("graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-64x32_equiangular_conservative.zarr", None),
    "fuxi": ("fuxi/2020-64x32_equiangular_conservative.zarr", None),
    "hres": ("hres/2016-2022-0012-64x32_equiangular_conservative.zarr", None),
    "neuralgcm_det": ("neuralgcm_deterministic/2020-64x32_equiangular_conservative.zarr", None),
    "neuralgcm_ens": ("neuralgcm_ens/2020-64x32_equiangular_conservative.zarr", "realization"),
    "ifs_ens": ("ifs_ens/2018-2022-64x32_equiangular_conservative.zarr", "number"),
    "gencast": ("gencast/2020-64x32_equiangular_conservative.zarr", "sample"),
}
VARS = [("geopotential", 500), ("temperature", 850), ("2m_temperature", None),
        ("10m_u_component_of_wind", None), ("10m_v_component_of_wind", None)]


def fetch(name):
    path, mdim = MODELS[name]
    ds = xr.open_zarr(B + path, storage_options={"client_kwargs": {"trust_env": True}})
    pt = ds.prediction_timedelta
    if not np.issubdtype(pt.dtype, np.timedelta64):  # some archives store plain integers with units attr
        u = {"hours": "h", "days": "D", "minutes": "m"}[pt.attrs.get("units", "hours")]
        ds = ds.assign_coords(prediction_timedelta=pt.values.astype(f"timedelta64[{u}]").astype("timedelta64[ns]"))
    times = C.ic_times()
    shift = np.timedelta64(0, "h")
    if np.isin(times, ds.time.values).mean() < 0.9:
        shift = np.timedelta64(12, "h")  # 00 UTC-only archives
        assert np.isin(times - shift, ds.time.values).sum() > 0.9 * len(times), name
    ok = np.isin(times - shift, ds.time.values)
    leads = np.array([np.timedelta64(h, "h") + shift for h in C.LEADS]).astype("timedelta64[ns]")
    assert np.isin(leads, ds.prediction_timedelta.values).all(), (name, ds.prediction_timedelta.values[:8])
    sel = ds.sel(time=((times - shift)[ok]).astype("datetime64[ns]"), prediction_timedelta=leads)
    M = sel.sizes[mdim] if mdim else 1
    arr = np.full((ok.sum(), M, len(leads), 5, 32, 64), np.nan, np.float32)
    for k, (v, lev) in enumerate(VARS):
        if v not in sel:
            continue
        x = sel[v] if lev is None else sel[v].sel(level=lev)
        if mdim is None:
            x = x.expand_dims("m")
            mdim_ = "m"
        else:
            mdim_ = mdim
        x = x.transpose("time", mdim_, "prediction_timedelta", "latitude", "longitude")
        assert np.allclose(x.latitude.values, C.static()["lat"]), name
        arr[:, :, :, k] = x.values
    return arr, ok, bool(shift)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--out", default=os.path.join(C.R, "runs", "step27", "wb2"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ics = np.array(C.ics_2020())
    for name in a.models:
        f = os.path.join(a.out, f"{name}.npz")
        if os.path.exists(f):
            continue
        t0 = time.time()
        arr, ok, shifted = fetch(name)
        print(name, arr.shape, f"fetched {time.time() - t0:.0f} s, ICs {ok.sum()}/{len(ok)}, 00UTC shift={shifted}", flush=True)
        meta = {"source": B + MODELS[name][0], "members": arr.shape[1], "ic_shift_h": 12 if shifted else 0,
                "ics_used": ics[ok].tolist()}
        sc = C.score(arr, ics=ics[ok].tolist())
        sc = {k: _pad(v, ok) for k, v in sc.items()}
        C.save(name, sc, f, meta)
        if arr.shape[1] > 8:
            sc8 = {k: _pad(v, ok) for k, v in C.score(arr[:, :8], ics=ics[ok].tolist()).items()}
            C.save(name + "_m8", sc8, os.path.join(a.out, f"{name}_m8.npz"), dict(meta, members=8))
        del arr


def _pad(v, ok):
    out = np.full((len(ok),) + v.shape[1:], np.nan)
    out[ok] = v
    return out


if __name__ == "__main__":
    main()
