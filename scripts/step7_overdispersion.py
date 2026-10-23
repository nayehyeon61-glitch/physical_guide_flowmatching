"""Step 7 side test: two hypotheses for the CFM over-dispersion (owner, 2026-10-23).

H1 "the data are too small": the spread is estimation noise of a model fit on
   few pairs, so it should shrink as the number of training pairs grows, and the
   ensemble should be better calibrated on the TRAINING pairs than on held-out ones.
   Test: the same grid CFM (step 6 recipe, same iterations) on 1 / 4 / 12 years of
   90 days, scored on the same 2020 starts, 1-day SSR / spread / lon lag-1 on
   train and test pairs.
H2 "the PINN's physical effect is too small": a stronger physics pull should then
   remove the excess spread. Test: the untrained gradient flow of the PINN residual
   (step 6 ``GradientFlowProjection``) with step sizes eta 0.5 / 2 / 8 on the 4-year
   CFM, next to the same flow on the Laplacian energy (non-physical control).

    python scripts/step7_overdispersion.py --out runs/step7_od
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm.coupling import PGCFM, estimate_scales  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset, kai_data_config  # noqa: E402
from pgcfm.latent import member_structure  # noqa: E402
from pgcfm.projection import GradientFlowProjection, freeze  # noqa: E402
from pgcfm.train import run_phase, take, validate_one_step  # noqa: E402
from scripts.make_balanced_era5 import main as make_balanced  # noqa: E402
from scripts.train_pgcfm import DEFAULTS, build  # noqa: E402

YEARS = {1: [2001], 4: [2001, 2002, 2003, 2004], 12: list(range(2001, 2013))}


def one_step(transition, guide, batches, members, steps):
    r = validate_one_step(PGCFM(transition, guide), batches, n_members=members, steps=steps)
    r.update(member_structure(transition, batches, members, steps))
    return {k: v for k, v in r.items() if not k.startswith("res_obs")}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/step7_od")
    p.add_argument("--n-years", type=int, nargs="+", default=[1, 4, 12])
    p.add_argument("--iters", type=int, default=1440)
    p.add_argument("--etas", type=float, nargs="+", default=[0.5, 2.0, 8.0])
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--steps", type=int, default=8)
    a = p.parse_args(argv)
    torch.set_num_threads(os.cpu_count())
    te = os.path.join(a.out, "data", "test")
    if not os.path.exists(os.path.join(te, "2020.nc")):
        make_balanced(["--out", te, "--years", "2020", "--days", "32", "--n-lat", "25", "--n-lon", "48",
                       "--seed", "1000"])
    res, path = {"args": vars(a)}, os.path.join(a.out, "results.json")
    for n in a.n_years:
        tr = os.path.join(a.out, "data", f"train{n}")
        if not os.path.exists(os.path.join(tr, "stats", "global_std.npy")):
            make_balanced(["--out", tr, "--years", *map(str, YEARS[n]), "--days", "90", "--n-lat", "25",
                           "--n-lon", "48", "--seed", "0"])
        mean_p, std_p = os.path.join(tr, "stats", "global_mean.npy"), os.path.join(tr, "stats", "global_std.npy")
        dcfg = kai_data_config(mean_p, std_p)
        pair_ds = PGCFMSequenceDataset(dcfg, tr, horizon=1)
        loader = torch.utils.data.DataLoader(pair_ds, batch_size=8, shuffle=True, drop_last=True)
        test_b = take(torch.utils.data.DataLoader(PGCFMSequenceDataset(dcfg, te, horizon=1, train=False),
                                                  batch_size=8), 2)
        train_b = take(torch.utils.data.DataLoader(pair_ds, batch_size=8, shuffle=False), 2)
        torch.manual_seed(0)
        m = build(dict(DEFAULTS, model_channels=32, channel_mult=[1, 2, 2], num_res_blocks=1),
                  25, 48, np.load(mean_p), np.load(std_p))
        m.guide.scales = estimate_scales(m.guide, take(loader, 10))
        run_phase(m, "cfm", loader, a.iters, 2e-4, log_every=a.iters // 4)
        freeze(m.cfm), freeze(m.guide)
        r = {"n_pairs": len(pair_ds), "test": one_step(m.cfm, m.guide, test_b, a.members, a.steps),
             "train": one_step(m.cfm, m.guide, train_b, a.members, a.steps)}
        if n == 4:
            for eta in a.etas:
                for ctl in ("physics", "smooth"):
                    gf = GradientFlowProjection(m.cfm, m.guide, eta=eta, control=ctl)
                    r[f"gradflow_{ctl}_{eta:g}"] = one_step(gf, m.guide, test_b, a.members, a.steps)
        res[f"years{n}"] = r
        print(f"[{n} years, {len(pair_ds)} pairs]", json.dumps(r, indent=None, default=float), flush=True)
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
