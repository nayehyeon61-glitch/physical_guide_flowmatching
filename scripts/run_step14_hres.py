"""Step 14, the owner's H-res: is a core trained on 32 x 64 usable zero-shot on 64 x 128 (discretization
invariance), and is that a property of the FNO and not of the ResNet?

The FM generator only makes 32 x 64 members, so H-res is tested on the deterministic cores of H-dyn
(scripts/run_step12_core.py, trained on 5.625 deg 2006-2015), with the 2.8125 deg ERA5 of scripts/wb2_hres.py:

    native   the core run directly on 64 x 128: the IC frames, the static fields (position, orography,
             land-sea mask) and the time features at 64 x 128, the same normalization (5.625 deg training
             statistics), the same weights
    interp   the core run on 32 x 64 (its training grid) and its output interpolated bilinearly to 64 x 128
             (longitude periodic): what a resolution-bound model can do
    persist  the 64 x 128 IC
Scores against the 64 x 128 truth on ClimODE's ICs (12 UTC every second day, 360 for 2017-2018): RMSE and
ACC per lead (cos-latitude weights, ACC about the test year's mean as in step 11), and the skill
1 - RMSE / RMSE_persistence. The owner's criterion: the native 64 x 128 skill keeps >= 70 % of the 32 x 64 skill.

    python scripts/run_step14_hres.py --data /home/claude/era5 --hres /home/claude/era5_hres --out runs/step14 \
        --cores runs/step12/core_jump6.pt runs/step14/core_jump6_fno.pt runs/step14/core_jump6_fno_b1.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm.core import static_features  # noqa: E402
from scripts.run_step11_climode import VARS, climode_truth  # noqa: E402
from scripts.run_step12_core import LEADS_H, forecast, load_static  # noqa: E402
from scripts.run_step14_hdyn import load_core  # noqa: E402


def scores(P, O, clim, lat):
    """(n, L, 5, H, W) -> rmse, acc (n, L, 5)."""
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None]
    rmse = np.sqrt(((P - O) ** 2 * w).mean((-1, -2)))
    pa, oa = P - clim, O - clim
    pa, oa = pa - pa.mean((-1, -2), keepdims=True), oa - oa.mean((-1, -2), keepdims=True)
    acc = (w * pa * oa).sum((-1, -2)) / np.sqrt((w * pa ** 2).sum((-1, -2)) * (w * oa ** 2).sum((-1, -2)))
    return rmse, acc


def upsample(P: np.ndarray, lat_c, lon_c, lat_f, lon_f) -> np.ndarray:
    """(..., Hc, Wc) -> (..., Hf, Wf): linear in latitude (clamped at the outermost rows), periodic in longitude."""
    sh = P.shape
    x = torch.from_numpy(P.reshape(-1, 1, *sh[-2:]))
    Wc = sh[-1]
    x = torch.cat([x[..., -1:], x, x[..., :1]], -1)  # one periodic column each side
    gy = np.interp(lat_f, lat_c, np.arange(len(lat_c))) / (len(lat_c) - 1) * 2 - 1
    ix = (lon_f - lon_c[0]) / (lon_c[1] - lon_c[0])  # fine longitudes in coarse index units
    gx = (ix + 1) / (Wc + 1) * 2 - 1
    grid = torch.from_numpy(np.stack(np.meshgrid(gx, gy), -1).astype(np.float32))[None]
    y = F.grid_sample(x.float(), grid.expand(x.shape[0], -1, -1, -1), mode="bilinear", align_corners=True)
    return y.numpy().reshape(*sh[:-2], len(lat_f), len(lon_f))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder with era5_data (5.625 deg)")
    p.add_argument("--hres", required=True, help="folder of scripts/wb2_hres.py")
    p.add_argument("--out", default="runs/step14")
    p.add_argument("--cores", nargs="+", required=True)
    p.add_argument("--train-years", type=int, nargs="+", default=list(range(2006, 2016)))
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--max-ics", type=int, default=None)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    era5 = os.path.join(a.data, "era5_data")
    tr = np.concatenate([climode_truth(era5, y) for y in a.train_years])
    mean, std = tr.mean((0, 2, 3)), tr.std((0, 2, 3))
    del tr
    lat_c, lon_c, oro_c, lsm_c = load_static(era5)
    cst = np.load(os.path.join(a.hres, "era5_hres_constants.npz"))
    lat_f, lon_f = cst["lat"], cst["lon"]
    st_c = static_features(lat_c, lon_c, oro_c, lsm_c)
    st_f = static_features(lat_f, lon_f, cst["orography"].astype(np.float32), cst["lsm"].astype(np.float32))
    nz = lambda t: torch.from_numpy(((t - mean[:, None, None]) / std[:, None, None]).astype(np.float32))  # noqa: E731
    den = lambda t: t.numpy() * std[:, None, None] + mean[:, None, None]  # noqa: E731
    data = {}
    for y in a.test_years:
        xf = np.load(os.path.join(a.hres, f"era5_hres_{y}.npz"))["x"]
        n_ic = (len(xf) - 2 - 24) // 8 + 1
        n_ic = min(n_ic, a.max_ics or n_ic)
        ics = [2 + 8 * e for e in range(n_ic)]
        data[y] = (xf, climode_truth(era5, y), ics)
    res = {"args": vars(a)}
    keys = ("native", "interp", "coarse", "persist_f", "persist_c")
    for pt in a.cores:
        core, _ = load_core(pt, st_c)
        tag = os.path.basename(pt)[:-3]
        acc_ = {k: {"rmse": [], "acc": []} for k in keys}
        for y, (xf, xc, ics) in data.items():
            Of = np.stack([xf[[i + h // 6 for h in LEADS_H]] for i in ics])
            Oc = np.stack([xc[[i + h // 6 for h in LEADS_H]] for i in ics])
            core.static = st_c
            Pc = den(forecast(core, nz(xc), ics, 24))
            core.static = st_f
            Pf = den(forecast(core, nz(xf), ics, 24))
            preds = {"native": (Pf, Of, xf, lat_f), "interp": (upsample(Pc, lat_c, lon_c, lat_f, lon_f), Of, xf, lat_f),
                     "coarse": (Pc, Oc, xc, lat_c),
                     "persist_f": (np.repeat(xf[ics][:, None], len(LEADS_H), 1), Of, xf, lat_f),
                     "persist_c": (np.repeat(xc[ics][:, None], len(LEADS_H), 1), Oc, xc, lat_c)}
            for k, (P, O, x, lat) in preds.items():
                r, c = scores(P, O, x.mean(0), lat)
                acc_[k]["rmse"].append(r), acc_[k]["acc"].append(c)
        out = {k: {q: {str(h): {v: float(np.concatenate(d[q])[:, j, i].mean()) for i, v in enumerate(VARS)}
                       for j, h in enumerate(LEADS_H)} for q in d} for k, d in acc_.items()}
        for k, pk in (("native", "persist_f"), ("interp", "persist_f"), ("coarse", "persist_c")):
            out[k]["skill"] = {h: {v: 1 - out[k]["rmse"][h][v] / out[pk]["rmse"][h][v] for v in VARS}
                               for h in out[k]["rmse"]}
        res[tag] = out
        for k in ("coarse", "native", "interp"):
            print(f"[{tag}] {k:6s} z500 RMSE " + " ".join(f"{out[k]['rmse'][str(h)]['z']:.0f}" for h in LEADS_H)
                  + " | skill vs persistence " + " ".join(f"{out[k]['skill'][str(h)]['z']:.2f}" for h in LEADS_H)
                  + " | t2m 24 h " + f"{out[k]['rmse']['24']['t2m']:.2f}", flush=True)
    with open(os.path.join(a.out, "hres.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
