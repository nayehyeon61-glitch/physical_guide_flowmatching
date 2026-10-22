"""Step 4 follow-up: hydrostatic balance as a hard constraint instead of a penalty.

Trains the same CFM as ``run_step4_pgcfm.py`` (same data, seed, iterations)
and scores it with and without ``HydrostaticProjection`` (T diagnosed from
the generated z). No retraining: this isolates what the constraint costs or
buys in skill when it is satisfied exactly.

    python scripts/run_step4_projection.py --out runs/step4 --n-lat 25 --n-lon 48 --iters-cfm 1200
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm.coupling import PGCFM, HydrostaticProjection, estimate_scales  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset, kai_data_config  # noqa: E402
from pgcfm.train import run_phase, take  # noqa: E402
from scripts.run_step4_pgcfm import evaluate, print_table  # noqa: E402
from scripts.train_pgcfm import DEFAULTS, build  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/step4")
    p.add_argument("--n-lat", type=int, default=25)
    p.add_argument("--n-lon", type=int, default=48)
    p.add_argument("--iters-cfm", type=int, default=1200)
    p.add_argument("--extra-cfm", type=int, default=300, help="the cfm_more iterations of the cfm arm")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--n-starts", type=int, default=4)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--steps", type=int, default=8)
    a = p.parse_args(argv)
    torch.set_num_threads(os.cpu_count())
    tr, te = os.path.join(a.out, "data", "train"), os.path.join(a.out, "data", "test")
    mean_p, std_p = os.path.join(tr, "stats", "global_mean.npy"), os.path.join(tr, "stats", "global_std.npy")
    dcfg = kai_data_config(mean_p, std_p)
    train_ds = PGCFMSequenceDataset(dcfg, tr, horizon=1)
    test_ds = PGCFMSequenceDataset(dcfg, te, horizon=a.horizon, train=False)
    pair_ds = PGCFMSequenceDataset(dcfg, te, horizon=1, train=False)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=a.batch, shuffle=True, drop_last=True)
    test_batch = torch.utils.data.default_collate([test_ds[5 * i] for i in range(a.n_starts)])
    val_batches = take(torch.utils.data.DataLoader(pair_ds, batch_size=a.batch), 2)

    cfg = dict(DEFAULTS, model_channels=32, channel_mult=[1, 2, 2], num_res_blocks=1, sample_steps=2)
    torch.manual_seed(0)
    m = build(cfg, train_ds.img_shape_x, train_ds.img_shape_y, np.load(mean_p), np.load(std_p))
    m.guide.scales = estimate_scales(m.guide, take(loader, 10))
    run_phase(m, "cfm", loader, a.iters_cfm, 2e-4, log_every=max(1, a.iters_cfm // 4))
    torch.manual_seed(1)
    run_phase(m, "cfm", loader, a.extra_cfm, 5e-5, log_every=max(1, a.extra_cfm // 2))

    path = os.path.join(a.out, "results.json")
    results = json.load(open(path))
    results["cfm_rerun"] = evaluate(m, test_batch, val_batches, a.members, a.steps)
    proj = PGCFM(HydrostaticProjection(m.cfm, m.guide), m.guide)
    results["cfm_hydproj"] = evaluate(proj, test_batch, val_batches, a.members, a.steps)
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    for n in ("cfm_rerun", "cfm_hydproj"):
        print(n, json.dumps({k: round(v, 4) for k, v in results[n]["one_step"].items()}))
    print_table(results, ["cfm", "cfm_rerun", "cfm_hydproj", "pgcfm_0.1"], a.horizon)


if __name__ == "__main__":
    main()
