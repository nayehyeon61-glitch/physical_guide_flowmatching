"""Diagnostic of the integrated physical terms (docs/kai_dycore_integration.md): on ERA5 daily means, how well
does (a) the instantaneous operator tendency G(x) x 1 day and (b) dinosaur's 24 h integration D(x) - x predict the
observed daily increment x(t+1) - x(t)? Correlation and amplitude ratio per variable (area-weighted, anomalies
over the sample), plus timing and a free dycore run of N days (stability)."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract
from pgcfm.dycore_day import DinosaurDay

LV = {v: contract.level_slice(v) for v in ("z", "q", "t", "u", "v")}


def fields(x):
    f = {v: x[:, LV[v]] for v in ("u", "v", "t", "q", "z")}
    f["sp"] = x[:, contract.CHANNEL_INDEX["sp"]]
    return f


def stat(pred, obs, w):
    pa = pred - pred.mean(0, keepdims=True); oa = obs - obs.mean(0, keepdims=True)
    num = (pa * oa * w).sum(); den = np.sqrt((pa ** 2 * w).sum() * (oa ** 2 * w).sum())
    return float(num / den), float(np.sqrt((pa ** 2 * w).sum() / (oa ** 2 * w).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2016)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--dt", type=float, default=30.0)
    p.add_argument("--dfi", type=float, default=0.0)
    p.add_argument("--free-days", type=int, default=10)
    p.add_argument("--gfeat", action="store_true", help="also the instantaneous operator G (torch)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    x = np.load(f"data/era5_5p625/kaiday/{a.year}.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    days = np.sort(rng.choice(len(x) - 1 - a.free_days, a.n, replace=False))
    x0 = np.asarray(x[days], np.float64); x1 = np.asarray(x[days + 1], np.float64)
    orog = x0[0, contract.CHANNEL_INDEX["orog"]]
    lat = contract.grid_latitudes(32) if 32 not in getattr(contract, "_LATITUDE_OVERRIDES", {}) else None
    lat = np.linspace(-87.1875, 87.1875, 32)
    w = np.cos(np.radians(lat))[None, None, :, None]
    t0 = time.time()
    dy = DinosaurDay(orog, layers=a.layers, dt_minutes=a.dt, dfi_hours=a.dfi)
    inc = {k: [] for k in ("u", "v", "t", "q", "z", "sp")}
    for i in range(0, a.n, a.batch):
        t1 = time.time()
        r = dy(fields(x0[i:i + a.batch]))
        for k in inc:
            inc[k].append(r[k])
        print(f"batch {i // a.batch}: {time.time() - t1:.1f} s", flush=True)
    inc = {k: np.concatenate(v, 0) for k, v in inc.items()}
    obs = fields(x1 - x0)
    res = {"setup": vars(a), "dycore": {}}
    for k in ("u", "v", "t", "q", "z"):
        c, r = stat(inc[k], obs[k], w)
        res["dycore"][k] = {"corr": c, "amp": r,
                            "by_level": [stat(inc[k][:, j:j + 1], obs[k][:, j:j + 1], w)[0] for j in range(12)]}
        print(f"dycore {k}: corr {c:.3f} amp {r:.2f}  levels " + " ".join(f"{v:.2f}" for v in res['dycore'][k]['by_level']))
    c, r = stat(inc["sp"][:, None], obs["sp"][:, None], w)
    res["dycore"]["sp"] = {"corr": c, "amp": r}
    print(f"dycore sp: corr {c:.3f} amp {r:.2f}")
    if a.gfeat:
        import torch
        sys.path.insert(0, "scripts")
        from pgcfm.kai_tend import operator_features
        mean = x0.mean((0, 2, 3)); std = x0.std((0, 2, 3)) + 1e-6
        z = torch.from_numpy(((x0 - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float32))
        feats = operator_features(mean, std, z[:1])
        g, m = feats.raw(z[:, contract.DYNAMIC_CHANNELS])
        g = g.numpy() * 86400.0; m = m.numpy()
        res["G"] = {}
        for j, k in enumerate(("u", "v", "t", "q")):
            gg = g[:, 12 * j:12 * j + 12] * m[:, 12 * j:12 * j + 12]
            c, r = stat(gg, obs[k] * m[:, 12 * j:12 * j + 12], w)
            cd, rd = stat(inc[k] * m[:, 12 * j:12 * j + 12], obs[k] * m[:, 12 * j:12 * j + 12], w)
            res["G"][k] = {"corr": c, "amp": r, "dycore_same_mask": [cd, rd]}
            print(f"G x 1 day {k}: corr {c:.3f} amp {r:.2f}   (dycore on the same points: {cd:.3f} {rd:.2f})")
    # free run of the dycore alone from ERA5 (stability over days), skill of persistence vs dycore
    if a.free_days:
        s = x0[:a.batch].copy()
        fr = []
        for d in range(a.free_days):
            r = dy(fields(s))
            for k in ("u", "v", "t", "q", "z"):
                s[:, LV[k]] += r[k]
            s[:, contract.CHANNEL_INDEX["sp"]] += r["sp"]
            truth = np.asarray(x[days[:a.batch] + d + 1], np.float64)
            row = {}
            for k in ("u", "t", "z"):
                e = np.sqrt(((s[:, LV[k]] - truth[:, LV[k]]) ** 2 * w).mean() / w.mean())
                ep = np.sqrt(((x0[:a.batch, LV[k]] - truth[:, LV[k]]) ** 2 * w).mean() / w.mean())
                row[k] = [float(e), float(ep)]
            row["finite"] = bool(np.isfinite(s).all()); row["umax"] = float(np.abs(s[:, LV["u"]]).max())
            fr.append(row)
            print(f"free day {d + 1}: " + " ".join(f"{k} rmse {row[k][0]:.2f} (persist {row[k][1]:.2f})" for k in ("u", "t", "z")) + f" |u|max {row['umax']:.0f}", flush=True)
        res["free"] = fr
    res["seconds"] = time.time() - t0
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
