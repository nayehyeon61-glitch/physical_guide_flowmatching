"""Step 14, the owner's H-dyn (2026-10-26): does a learned vector field make flow-dependent spread on its own?

Members start from the same analysis plus isotropic perturbations (a stationary random field with the same
statistics for every case), so any case-to-case difference of the spread is made by the rollout's tangent
linear Phi_t, E ||dx(t)||^2 = tr(Phi_t Sigma_0 Phi_t^T). No calibration (no Gamma, no sigma field).

    cores     trained step 12 / 14 cores (scripts/run_step12_core.py; ``--cores`` = their .pt files, the
              matching .json gives the config): jump6 with the ResNet (step 12), the FNO, the norm-bounded FNO
    members   M per IC: the three input frames (IC - 12 h, - 6 h, IC) all shifted by the same perturbation
              eps_v * g, g a smooth Gaussian random field of unit RMS per variable (Gaussian filter of
              ``--corr-cells`` grid cells, longitude periodic), eps_v = ``--eps`` x the 6 h tendency std of
              variable v (normalized units); 24 steps of 6 h, the 24 h multiples saved
    output    <out>/<core tag>_ens_<year>.npz in the step 13 format (E_minus_x0, O, X0, ics, leads), so
              scripts/score_step13.py (CRPS, SSR, rank) and scripts/diag_step13_flowdep.py (case-to-case spread,
              spread-fifth error ratio) score them unchanged; <out>/hdyn_<suffix>.json: spread growth
              (domain spread / initial perturbation size per lead, mean and CV over cases), and the a priori
              Lipschitz bound of D where the core is an FNO

    python scripts/run_step14_hdyn.py --data /home/claude/era5 --out runs/step14 \
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
from pgcfm.core import Core, CoreConfig, static_features  # noqa: E402
from scripts.run_step11_climode import climode_truth  # noqa: E402
from scripts.run_step12_core import LEADS_H, ic_frames, load_static  # noqa: E402


def load_core(pt, static):
    cfg_json = json.load(open(pt[:-3] + ".json"))
    cfg = dict(cfg_json["cfg"])
    cfg["blocks"] = tuple(cfg["blocks"])
    tend = {int(h): torch.tensor(v) for h, v in cfg_json["tend_std"].items()}
    core = Core(CoreConfig(**cfg), static, tend)
    core.load_state_dict(torch.load(pt))
    return core.eval(), cfg_json


def smooth_field(shape, cells: float, gen) -> torch.Tensor:
    """(n, C, H, W) Gaussian random field: white noise, Gaussian filter (longitude periodic, latitude
    replicate), unit RMS per sample and variable."""
    g = torch.randn(shape, generator=gen)
    r = int(np.ceil(3 * cells))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / cells) ** 2)
    k = k / k.sum()
    n, C, H, W = shape
    g = g.reshape(n * C, 1, H, W)
    g = F.conv2d(F.pad(g, (r, r, 0, 0), mode="circular"), k.view(1, 1, 1, -1))
    g = F.conv2d(F.pad(g, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    g = g.reshape(n, C, H, W)
    return g / g.pow(2).mean((-2, -1), keepdim=True).sqrt()


@torch.no_grad()
def ensemble(core, x, ics, M, eps, cells, seed, batch=16):
    """(n_ic, M, 6, 5, H, W) normalized states at 24 h multiples; (n_ic,) initial domain perturbation size."""
    gen = torch.Generator().manual_seed(seed)
    out, d0 = [], []
    for b in range(0, len(ics), batch):
        idx = ics[b:b + batch]
        fr = ic_frames(x, idx)  # (B, 3, 5, H, W)
        B = len(idx)
        pert = smooth_field((B * M,) + fr.shape[2:], cells, gen) * eps.view(1, -1, 1, 1)
        frm = fr[:, None].expand(-1, M, -1, -1, -1, -1).reshape(B * M, *fr.shape[1:]) + pert[:, None]
        hours = torch.tensor([6.0 * i for i in idx]).repeat_interleave(M)
        p = core.rollout(frm, hours, 24)[:, 3::4]
        out.append(p.reshape(B, M, *p.shape[1:]))
        d0.append(pert.reshape(B, M, -1).pow(2).mean(-1).mean(1).sqrt())
    return torch.cat(out), torch.cat(d0)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder with era5_data (scripts/wb2_to_kai.py)")
    p.add_argument("--out", default="runs/step14")
    p.add_argument("--cores", nargs="+", required=True)
    p.add_argument("--train-years", type=int, nargs="+", default=list(range(2006, 2016)))
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--eps", type=float, default=0.5, help="perturbation size in units of the 6 h tendency std")
    p.add_argument("--corr-cells", type=float, default=1.5, help="Gaussian filter width of the perturbation")
    p.add_argument("--suffix", default="", help="appended to the output names (e.g. a second --eps)")
    p.add_argument("--max-ics", type=int, default=None)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    era5 = os.path.join(a.data, "era5_data")
    tr = np.concatenate([climode_truth(era5, y) for y in a.train_years])
    mean, std = tr.mean((0, 2, 3)), tr.std((0, 2, 3))
    del tr
    lat, lon, oro, lsm = load_static(era5)
    static = static_features(lat, lon, oro, lsm)
    res = {"args": vars(a)}
    for pt in a.cores:
        core, info = load_core(pt, static)
        tag = os.path.basename(pt)[:-3] + a.suffix
        eps = a.eps * core.s6.view(-1)
        r = {"lipschitz_bound_D": info.get("lipschitz_bound_D"), "clip_frac": info.get("clip_frac"),
             "det_test_rmse_z500": {h: info["test"]["rmse"][h]["z"] for h in info["test"]["rmse"]}}
        g_all = []
        for y in a.test_years:
            truth = climode_truth(era5, y)
            x = torch.from_numpy((truth - mean[:, None, None]) / std[:, None, None])
            n_ic = (len(x) - 2 - 24) // 8 + 1
            n_ic = min(n_ic, a.max_ics or n_ic)
            ics = [2 + 8 * e for e in range(n_ic)]
            P, d0 = ensemble(core, x, ics, a.members, eps, a.corr_cells, seed=y)
            # spread growth: domain spread (area-weighted, normalized units, all variables) / initial size
            w = torch.from_numpy(np.cos(np.deg2rad(lat)).astype(np.float32)).view(1, 1, -1, 1)
            sp = (P.var(1, unbiased=True) * w / w.mean()).mean((-2, -1)).mean(-1).sqrt()  # (n, 6)
            g_all.append((sp / d0[:, None]).numpy())
            Pd = P.numpy() * std[:, None, None] + mean[:, None, None]
            O = np.stack([truth[[i + h // 6 for h in LEADS_H]] for i in ics])
            X0 = truth[ics]
            np.savez(os.path.join(a.out, f"{tag}_ens_{y}.npz"),
                     E_minus_x0=(Pd - X0[:, None, None]).astype(np.float16), O=O.astype(np.float32),
                     X0=X0.astype(np.float32), ics=np.array(ics), leads=np.array(LEADS_H))
        g = np.concatenate(g_all)
        r["spread_growth_mean"] = g.mean(0).tolist()
        r["spread_growth_cv"] = (g.std(0) / g.mean(0)).tolist()
        res[tag] = r
        print(f"[{tag}] spread growth (domain spread / initial) by lead " + " ".join(f"{v:.2f}" for v in g.mean(0))
              + " | case CV " + " ".join(f"{v:.2f}" for v in g.std(0) / g.mean(0))
              + (f" | a priori Lip(D) <= {r['lipschitz_bound_D']:.3g}" if r["lipschitz_bound_D"] else ""),
              flush=True)
    with open(os.path.join(a.out, f"hdyn{a.suffix}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
