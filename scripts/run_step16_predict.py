"""Step 16 core on ClimODE's ICs: deterministic 144 h rollouts of a trained DynCore
(scripts/run_step16_dynpinn.py) at the ICs 2 + 8 e of every year, written in the step 12 core format so
``scripts/run_step13_guide.py --core <prefix>`` puts them in the mean slot (core_band: the core's mean
with the FM members' deviations, closed-form band Gamma on the validation year).

    <run>/<tag>_pred.npz        keys "2017", "2018": (n_ic, 6, 5, H, W) physical ClimODE variables at 24..144 h
    <run>/<tag>_pred_2016.npz   pred, ics: the validation year (Gamma and the MBM are fitted there)

u10 / v10 come from u1000 / v1000 by the step 11 per-cell regression fitted on the validation year.

    python scripts/run_step16_predict.py --kai /home/claude/era5/kai6h --era5 /home/claude/era5/era5_data \
        --run runs/step16 --tag dyn_lp0
"""

import argparse
import json
import os
import sys

import netCDF4 as nc
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm import contract  # noqa: E402
from pgcfm.core import static_features  # noqa: E402
from pgcfm.dyncore import DynCore, DynCoreConfig  # noqa: E402
from pgcfm.runs import load_years  # noqa: E402
from scripts.run_step11_climode import VARS, climode_truth  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402
from scripts.run_step13_pinn import gather  # noqa: E402
from scripts.run_step13_seqfm import DYN_IDX  # noqa: E402

DYN = contract.DYNAMIC_CHANNELS
K = 24
LEAD_STEPS = [3, 7, 11, 15, 19, 23]  # 24 .. 144 h


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--kai", required=True)
    p.add_argument("--era5", required=True)
    p.add_argument("--run", default="runs/step16")
    p.add_argument("--tag", required=True)
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--chunk", type=int, default=12)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-ics", type=int, default=0)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    info = json.load(open(os.path.join(a.run, f"{a.tag}.json")))
    stats = os.path.join(a.run, "stats")
    mean, std = np.load(os.path.join(stats, "global_mean.npy")), np.load(os.path.join(stats, "global_std.npy"))
    with nc.Dataset(os.path.join(a.kai, f"{a.val_year}.nc")) as d:
        contract.register_latitudes(np.asarray(d["latitude"][:]))
    lat, lon, oro, lsm = load_static(a.era5)
    c = info["cfg"]
    cfg = DynCoreConfig(**{**c, "blocks": tuple(c["blocks"])})
    model = DynCore(cfg, static_features(lat, lon, oro, lsm), torch.ones(contract.N_DYNAMIC), mean[DYN], std[DYN], lat)
    model.load_state_dict(torch.load(os.path.join(a.run, f"{a.tag}.pt"), map_location="cpu"))
    model.eval()
    unscale = lambda q: q * std[DYN][:, None, None] + mean[DYN][:, None, None]  # noqa: E731
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None]
    sfc, out = {}, {}
    for y in [a.val_year] + a.test_years:
        data = load_years(a.kai, [y], mean, std)
        truth = climode_truth(a.era5, y)
        if y == a.val_year:  # u10 / v10 from u1000 / v1000 (step 11), fitted on the validation year
            xv = unscale(data[y][:, DYN].numpy())
            for v in ("u10", "v10"):
                x_, t_ = xv[:, DYN_IDX[v]], truth[:, VARS.index(v)]
                b = ((x_ - x_.mean(0)) * (t_ - t_.mean(0))).sum(0) / ((x_ - x_.mean(0)) ** 2).sum(0).clip(1e-6)
                sfc[v] = (t_.mean(0) - b * x_.mean(0), b)
            del xv
        n_ic = (len(truth) - 2 - K - 1) // 8 + 1
        n_ic = min(n_ic, a.max_ics or n_ic)
        ics = [2 + 8 * e for e in range(n_ic)]
        pred = []
        with torch.no_grad():
            for i in range(0, n_ic, a.chunk):
                part = ics[i:i + a.chunk]
                x0, d, _, hour0 = gather(data, [y] * len(part), part, K)
                tisr = (x0[:, None] + d)[:, :, contract.DYNAMIC_CHANNELS.index(contract.CHANNEL_INDEX["tisr"])]
                r = model.rollout(x0, hour0, K, tisr)
                r = r[:, LEAD_STEPS]
                e = unscale(r.numpy())
                v5 = np.stack([e[..., DYN_IDX[v], :, :] for v in VARS], -3)
                for v in ("u10", "v10"):
                    j = VARS.index(v)
                    v5[..., j, :, :] = sfc[v][0] + sfc[v][1] * v5[..., j, :, :]
                pred.append(v5.astype(np.float32))
        pred = np.concatenate(pred)
        tgt = np.stack([truth[np.asarray(ics) + 4 * (k + 1)] for k in range(6)], 1)
        rmse = np.sqrt(((pred - tgt) ** 2 * w).mean((0, -1, -2)))
        print(f"[{a.tag} {y}] {n_ic} ICs, RMSE 24..144 h: z500 " + " ".join(f"{q:.0f}" for q in rmse[:, 0])
              + " | t850 " + " ".join(f"{q:.2f}" for q in rmse[:, 1]) + " | t2m " + " ".join(f"{q:.2f}" for q in rmse[:, 2]),
              flush=True)
        out[y] = (pred, np.asarray(ics), rmse)
        del data
    prefix = os.path.join(a.run, a.tag)
    np.savez(prefix + f"_pred_{a.val_year}.npz", pred=out[a.val_year][0], ics=out[a.val_year][1])
    np.savez(prefix + "_pred.npz", **{str(y): out[y][0] for y in a.test_years})
    with open(prefix + "_pred.json", "w") as f:
        json.dump({str(y): {"rmse": out[y][2].tolist(), "n_ic": len(out[y][1])} for y in out}, f, indent=1)


if __name__ == "__main__":
    main()
