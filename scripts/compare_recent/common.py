"""Shared pieces for the recent-model comparison (docs/step27_recent_models.md).

Same ICs, truth and scores as docs/results_2026_11 (scripts/compare_step13_climode.py):
  ICs   = the 179 ClimODE-convention 2020 ICs (12 UTC every 2 days, 2020-08-30 dropped), 6 h index into 2020
  truth = WB2 ERA5 64x32 conservative (data/wb2_5var/<year>.npy, the zarr our era5_5p625 files come from)
  rmse  = per-IC sqrt(lat-weighted spatial mean), then mean over ICs (ensemble mean for ensembles)
  acc   = ClimODE's ACC with the test-year time mean as climatology
  spread= sqrt(lat-weighted mean of the member variance, ddof=1)
  crps_ens = unweighted spatial mean of the ensemble CRPS in ClimODE min-max units (2006-2018 min / max)
  crps_w   = lat-weighted spatial mean of the ensemble CRPS in physical units (WB2 / Andrae convention)
"""
import json, os
import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VARS = ["z", "t", "t2m", "u10", "v10"]
LEADS = [24, 48, 72, 96, 120, 144]
DATA = os.path.join(R, "data", "wb2_5var")
PAIRED = os.path.join(R, "docs", "results_2026_11", "scores", "raw", "blend_ngcm_2020_climode_paired.json")


def ics_2020():
    return [int(i) for i in json.load(open(PAIRED))["ics"]]


def ic_times(ics=None):
    ics = ics_2020() if ics is None else ics
    return np.datetime64("2020-01-01T00") + np.array(ics) * np.timedelta64(6, "h")


def truth_year(y):
    return np.load(os.path.join(DATA, f"{y}.npy"), mmap_mode="r")


def static():
    return dict(np.load(os.path.join(DATA, "static.npz")))


def minmax():
    f = os.path.join(DATA, "minmax_2006_2018.json")
    if not os.path.exists(f):
        lo, hi = np.full(5, np.inf), np.full(5, -np.inf)
        for y in range(2006, 2019):
            t = np.asarray(truth_year(y))
            lo, hi = np.minimum(lo, t.min((0, 2, 3))), np.maximum(hi, t.max((0, 2, 3)))
        json.dump({"max": hi.tolist(), "min": lo.tolist()}, open(f, "w"))
    d = json.load(open(f))
    return np.array(d["max"]), np.array(d["min"])


def crps_ens(obs, ens):
    """Fair-free (properscoring) ensemble CRPS, ens (M, ...) obs (...): E|X-y| - 0.5 E|X-X'|."""
    M = ens.shape[0]
    t1 = np.abs(ens - obs[None]).mean(0)
    if M == 1:
        return t1
    s = np.sort(ens, 0)  # E|X-X'| via sorted members: 2/M^2 sum_i (2i-M-1) x_(i)
    k = (2 * np.arange(1, M + 1) - M - 1).reshape((M,) + (1,) * (ens.ndim - 1))
    return t1 - (k * s).sum(0) / M ** 2


def score(ens, ics=None, leads=LEADS, name=""):
    """ens (n_ic, M, L, 5, 32, 64) physical, NaN for a variable the model lacks -> dict metric -> (n_ic, L, 5)."""
    ics = ics_2020() if ics is None else ics
    tr = truth_year(2020)
    clim = np.asarray(tr).mean(0)
    lat = static()["lat"]
    w = np.cos(np.deg2rad(lat)); w = (w / w.mean())[:, None]
    mx, mn = minmax()
    sc = (mx - mn)[:, None, None]
    n, M, L = ens.shape[:3]
    out = {k: np.full((n, L, 5), np.nan) for k in ("rmse", "acc", "spread", "crps_ens", "crps_w")}
    for e, i0 in enumerate(ics):
        for j, h in enumerate(leads):
            obs = np.asarray(tr[i0 + h // 6], np.float64)
            mem = ens[e, :, j].astype(np.float64)
            P = mem.mean(0)
            out["rmse"][e, j] = np.sqrt(((P - obs) ** 2 * w).mean((-1, -2)))
            pa, oa = P - clim, obs - clim
            pa = pa - pa.mean((-1, -2), keepdims=True); oa = oa - oa.mean((-1, -2), keepdims=True)
            out["acc"][e, j] = (w * pa * oa).sum((-1, -2)) / np.sqrt((w * pa ** 2).sum((-1, -2)) * (w * oa ** 2).sum((-1, -2)))
            if M > 1:
                out["spread"][e, j] = np.sqrt((mem.var(0, ddof=1) * w).mean((-1, -2)))
            c = crps_ens(obs, mem)
            out["crps_w"][e, j] = (c * w).mean((-1, -2))
            out["crps_ens"][e, j] = (c / sc).mean((-1, -2))
    return out


def save(name, sc, path, meta=None):
    np.savez_compressed(path, **sc, leads=np.array(LEADS), vars=np.array(VARS), ics=np.array(ics_2020()),
                        meta=json.dumps(meta or {}))
    print(f"[{name}] saved {path}")
    for met in ("rmse", "crps_w", "spread"):
        m = np.nanmean(sc[met], 0)
        print(f"  {met} z500 24/72/144: {m[0,0]:.1f} {m[2,0]:.1f} {m[5,0]:.1f}  t850: {m[0,1]:.2f} {m[2,1]:.2f} {m[5,1]:.2f}")
