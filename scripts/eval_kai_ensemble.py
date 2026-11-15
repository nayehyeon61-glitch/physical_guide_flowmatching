"""Ensemble evaluation of a noise core (docs/kai_dycore_integration.md): M members per IC of the test year, the core
alone, L days. Per lead: fair CRPS, RMSE of the ensemble mean, spread / error ratio, ACC of the mean, activity of
one member and of the mean (65 scored channels, normalized by the channel std), and physical-unit RMSE of the mean
+ spread for t2m, t850, z500, u850, msl. ``--det`` tags are run once without noise (their own rows).

    python scripts/eval_kai_ensemble.py --out runs/kaitend --tags core_sb --det core_gnD core_gng --members 8
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_kai_tendency as R  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.kai_tend import DYN, TISR, core_rollout  # noqa: E402

VARS = ["t2m", "t850", "z500", "u850", "msl"]


def main():
    p = R.parser()
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--det", nargs="*", default=[])
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--chunk", type=int, default=4, help="ICs per batch (x members)")
    p.add_argument("--json", default=None)
    p.add_argument("--hint-noise", type=float, default=0.0,
                   help="--tags of dycore-hint cores: noise of this many daily-increment std added to the state the dycore "
                        "integrates (the core itself sees the clean state), new every day and member")
    a = p.parse_args()
    torch.set_num_threads(a.threads) if getattr(a, "threads", 0) else None
    torch.manual_seed(a.seed)
    tr = R.load_years(a, a.train_years)
    mean, std, inc, clim = R.stats(a, tr)
    del tr
    te = R.load_years(a, [a.test_year])[a.test_year]
    lat = R.lat_of(os.path.join(a.data, f"{a.test_year}.nc"), te.shape[-2], a.era5)
    tez = torch.from_numpy(R.zscore(te, mean, std))
    cz = torch.from_numpy(R.zscore(clim, mean, std))[:, DYN]
    L, M = a.leads, a.members
    ics = np.arange(0, len(tez) - L, a.ic_stride)
    w = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)
    w = (w / w.mean()).view(1, 1, -1, 1)
    sc = [i for i, c in enumerate(DYN) if c != TISR]
    vi = [DYN.index(contract.CHANNEL_INDEX[v]) for v in VARS]
    vstd = torch.as_tensor(std[[contract.CHANNEL_INDEX[v] for v in VARS]], dtype=torch.float32)
    res = {"ics": int(len(ics)), "members": M, "leads": L, "arms": {}}

    def wm(x):  # area mean over the grid -> (..., C)
        return (x * w).mean((-2, -1))

    for tag, m_ in [(t, M) for t in a.tags] + [(t, 1) for t in a.det]:
        core = R.build_core(a, mean, std, inc, lat, load=True, tag=tag)
        core.noise_on = m_ > 1
        if getattr(core, "nz", 0):
            core.features.noise_on = m_ > 1
        if a.hint_noise > 0 and m_ > 1:
            keep = torch.ones(len(DYN)); keep[DYN.index(TISR)] = 0
            core.features.noise_scale = (a.hint_noise * core.t6.cpu() * keep.view(1, -1, 1, 1)).float()
            core.features.noise_gen = torch.Generator().manual_seed(a.seed)
        acc = {k: 0.0 for k in ("crps", "mse", "var", "acc_n", "acc_p", "acc_t", "act_m", "act_e", "act_t", "vmse", "vvar")}
        for i in range(0, len(ics), a.chunk):
            j = ics[i:i + a.chunk]
            B = len(j)
            x0 = tez[j][:, DYN].repeat_interleave(m_, 0)
            doy = torch.tensor(j).repeat_interleave(m_)
            tgt = torch.stack([tez[j + k + 1] for k in range(L)], 1)  # (B, L, 67, H, W)
            tis = tgt[:, :, TISR].repeat_interleave(m_, 0)
            with torch.no_grad():
                y = core_rollout(core, x0, doy, L, tis).view(B, m_, L, len(DYN), *x0.shape[-2:])
            t = tgt[:, :, DYN]
            c = torch.stack([cz[(torch.tensor(j) + k + 1) % len(cz)] for k in range(L)], 1)
            mu = y.mean(1)
            e1 = (y - t[:, None]).abs().mean(1)
            e2 = (y[:, :, None] - y[:, None]).abs().sum((1, 2)) / (m_ * (m_ - 1)) if m_ > 1 else torch.zeros_like(e1)
            acc["crps"] += wm(e1 - 0.5 * e2).sum(0)                      # (L, C)
            acc["mse"] += wm((mu - t) ** 2).sum(0)
            acc["var"] += (wm(y.var(1, unbiased=True)).sum(0) if m_ > 1 else 0.0)
            fa, ta = mu - c, t - c
            acc["acc_n"] += wm(fa * ta).sum(0)
            acc["acc_p"] += wm(fa ** 2).sum(0)
            acc["acc_t"] += wm(ta ** 2).sum(0)
            acc["act_m"] += wm((y[:, 0] - c) ** 2).sum(0)
            acc["act_e"] += wm(fa ** 2).sum(0)
            acc["act_t"] += wm(ta ** 2).sum(0)
            print(tag, i + B, "/", len(ics), "nan states", getattr(core.features, "n_nan", 0), flush=True)
        n = len(ics)
        mse, var = acc["mse"] / n, (acc["var"] / n if m_ > 1 else torch.zeros_like(acc["mse"]))
        out = {
            "crps": (acc["crps"] / n)[:, sc].mean(1).tolist(),
            "nrmse_mean": torch.sqrt(mse[:, sc]).mean(1).tolist(),
            "ssr": torch.sqrt((1 + 1 / m_) * var[:, sc].mean(1) / mse[:, sc].mean(1)).tolist(),
            "acc_mean": (acc["acc_n"] / torch.sqrt(acc["acc_p"] * acc["acc_t"]))[:, sc].mean(1).tolist(),
            "activity_member": torch.sqrt(acc["act_m"] / acc["act_t"])[:, sc].mean(1).tolist(),
            "activity_mean": torch.sqrt(acc["act_e"] / acc["act_t"])[:, sc].mean(1).tolist(),
            "vars": {v: {"rmse_mean": (torch.sqrt(mse[:, vi[q]]) * vstd[q]).tolist(),
                         "spread": (torch.sqrt((1 + 1 / m_) * var[:, vi[q]]) * vstd[q]).tolist(),
                         "crps": ((acc["crps"] / n)[:, vi[q]] * vstd[q]).tolist()} for q, v in enumerate(VARS)},
            "nan_states": int(getattr(core.features, "n_nan", 0)), "members": m_,
        }
        res["arms"][tag if m_ > 1 else tag + "~det"] = out  # ~det: one run without noise
        print(tag, "crps", [round(v, 4) for v in out["crps"]], "\n  nrmse", [round(v, 3) for v in out["nrmse_mean"]],
              "\n  ssr", [round(v, 2) for v in out["ssr"]], "\n  acc", [round(v, 3) for v in out["acc_mean"]],
              "\n  act member", [round(v, 2) for v in out["activity_member"]], "mean", [round(v, 2) for v in out["activity_mean"]], flush=True)
        json.dump(res, open(a.json or os.path.join(a.out, "ens_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
