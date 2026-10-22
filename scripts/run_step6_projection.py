"""Step 6 experiment: frozen PINN -> ODE projection of the CFM cloud -> sequence loss.

Protocol on the balanced synthetic data of step 4 (same grid, same CFM recipe):
    shared  : CFM alone for --iters-cfm (phase "cfm"), then frozen
    pinn    : bridge + closure on observed pairs until the held-out physics loss
              stops improving (``projection.train_pinn``), then frozen
    arms (matched updates, --iters-proj each):
      cfm          CFM alone, + --iters-proj more CFM iterations (step 4 baseline)
      proj_crps    projection ODE, rollout CRPS only (lambda_P = 0)
      proj_phys    projection ODE, rollout CRPS + lambda_P x frozen-PINN residual
      proj_phys_gr same, physics gradient capped at --grad-ratio x CRPS gradient
      proj_smooth  projection ODE, rollout CRPS + lambda x Laplacian control (no physics)
      proj_k1      proj_crps with horizon 1 only (what the sequence loss adds)
      gradflow     no training: normalized gradient flow of the frozen PINN loss
Scores (``run_step4_pgcfm.evaluate``) on held-out 2020 starts, normalized units.

    python scripts/run_step6_projection.py --out runs/step6 [--quick]
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import metrics  # noqa: E402
from pgcfm.coupling import PGCFM, estimate_scales  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset, kai_data_config  # noqa: E402
from pgcfm.physics.primitive import ResidualScales  # noqa: E402
from pgcfm.projection import (GradientFlowProjection, ODEProjection, ProjectionConfig,  # noqa: E402
                              freeze, train_pinn, train_projection)
from pgcfm.train import run_phase, take  # noqa: E402
from scripts.make_balanced_era5 import main as make_balanced  # noqa: E402
from scripts.run_step4_pgcfm import KEEP, evaluate, member_residuals  # noqa: E402
from scripts.train_pgcfm import DEFAULTS, build  # noqa: E402


@torch.no_grad()
def compression(proj, batches, members, steps):
    """How the projection moves a member: spread before/after, RMS displacement,
    and the RMS of the ensemble-mean shift, for 1-day proposals."""
    acc = {}
    for b in batches:
        x, y, st, d = b["x0"], b["targets"][:, 0], b["static"], b["doy"][:, 1]
        rep = lambda t: t.repeat(members, *([1] * (t.dim() - 1)))  # noqa: E731
        g = torch.Generator().manual_seed(3)
        raw = proj.cfm.sample(rep(x), rep(st), rep(d), steps=steps, generator=g)
        out = proj.project(rep(x), raw, rep(st), rep(d))
        r, o = raw.view(members, *x.shape)[..., KEEP, :, :], out.view(members, *x.shape)[..., KEEP, :, :]
        for k, v in {"spread_raw": metrics.ens_spread(r).pow(2).mean().sqrt(),
                     "spread_proj": metrics.ens_spread(o).pow(2).mean().sqrt(),
                     "rmse_raw": metrics.ens_mean_rmse(r, y[:, KEEP]).pow(2).mean().sqrt(),
                     "rmse_proj": metrics.ens_mean_rmse(o, y[:, KEEP]).pow(2).mean().sqrt(),
                     "displacement": metrics._wmean((o - r) ** 2).mean().sqrt(),
                     "mean_shift": metrics._wmean((o.mean(0) - r.mean(0)) ** 2).mean().sqrt()}.items():
            acc.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


@torch.no_grad()
def pinn_report(model, batches):
    """Observed-pair residual per equation: linear bridge without closure vs the frozen PINN."""
    g, lin, pinn = model.guide, {}, {}
    for b in batches:
        x, y, st, d = b["x0"], b["targets"][:, 0], b["static"], b["doy"][:, 0]
        m = g.mask(x, y, st)
        for k, v in g.per_equation(g.residuals(x, y, st, d, use_bridge=False, use_closure=False)[0], m).items():
            lin.setdefault(k, []).append(v)
        for k, v in g.per_equation(g.residuals(x, y, st, d)[0], m).items():
            pinn.setdefault(k, []).append(v)
    return {"linear": {k: float(np.mean(v)) for k, v in lin.items()},
            "frozen_pinn": {k: float(np.mean(v)) for k, v in pinn.items()}}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/step6")
    p.add_argument("--n-lat", type=int, default=25)
    p.add_argument("--n-lon", type=int, default=48)
    p.add_argument("--train-years", type=int, nargs="+", default=[2001, 2002, 2003, 2004])
    p.add_argument("--train-days", type=int, default=90)
    p.add_argument("--iters-cfm", type=int, default=1200)
    p.add_argument("--pinn-max-iters", type=int, default=2000)
    p.add_argument("--iters-proj", type=int, default=240)
    p.add_argument("--lambda-p", type=float, default=0.1)
    p.add_argument("--grad-ratio", type=float, default=0.5)
    p.add_argument("--curriculum", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--arms", nargs="+",
                   default=["cfm", "proj_crps", "proj_phys", "proj_phys_gr", "proj_smooth", "proj_k1", "gradflow"])
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--n-starts", type=int, default=4)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.train_years, a.train_days, a.iters_cfm, a.pinn_max_iters, a.iters_proj = [2001], 12, 4, 4, 3
        a.horizon, a.n_starts, a.members, a.steps, a.n_lat, a.n_lon = 2, 1, 2, 2, 17, 32
        a.curriculum = [1, 2]
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    root = os.path.join(a.out, "data")
    tr, te = os.path.join(root, "train"), os.path.join(root, "test")
    if not os.path.exists(os.path.join(tr, "stats", "global_std.npy")):
        make_balanced(["--out", tr, "--years", *map(str, a.train_years), "--days", str(a.train_days),
                       "--n-lat", str(a.n_lat), "--n-lon", str(a.n_lon), "--seed", "0"])
        test_days = a.horizon + 2 + 5 * a.n_starts
        make_balanced(["--out", te, "--years", "2020", "--days", str(test_days),
                       "--n-lat", str(a.n_lat), "--n-lon", str(a.n_lon), "--seed", "1000"])
    mean_p, std_p = os.path.join(tr, "stats", "global_mean.npy"), os.path.join(tr, "stats", "global_std.npy")
    dcfg = kai_data_config(mean_p, std_p)
    K = max(a.curriculum)
    pair_ds = PGCFMSequenceDataset(dcfg, tr, horizon=1)
    seq_ds = PGCFMSequenceDataset(dcfg, tr, horizon=K)
    test_ds = PGCFMSequenceDataset(dcfg, te, horizon=a.horizon, train=False)
    tpair_ds = PGCFMSequenceDataset(dcfg, te, horizon=1, train=False)
    loader = torch.utils.data.DataLoader(pair_ds, batch_size=8, shuffle=True, drop_last=True)
    seq_loader = torch.utils.data.DataLoader(seq_ds, batch_size=a.batch, shuffle=True, drop_last=True)
    test_batch = torch.utils.data.default_collate([test_ds[5 * i] for i in range(a.n_starts)])
    val_batches = take(torch.utils.data.DataLoader(tpair_ds, batch_size=8), 2)

    cfg = dict(DEFAULTS, model_channels=32, channel_mult=[1, 2, 2], num_res_blocks=1)
    torch.manual_seed(0)
    base = build(cfg, pair_ds.img_shape_x, pair_ds.img_shape_y, np.load(mean_p), np.load(std_p))
    res_path, base_path = os.path.join(a.out, "results.json"), os.path.join(a.out, "base.pt")
    if os.path.exists(base_path):  # shared CFM + frozen PINN from an earlier invocation (--arms split)
        ck = torch.load(base_path)
        base.load_state_dict(ck["model"])
        base.guide.scales = ResidualScales(**ck["scales"])
        with open(res_path) as f:
            res = json.load(f)
        res["args_" + "_".join(a.arms)] = vars(a)
        print("loaded shared CFM + frozen PINN from", base_path, flush=True)
    else:
        base.guide.scales = estimate_scales(base.guide, take(loader, 10))
        res = {"args": vars(a), "scales": vars(base.guide.scales), "n_train_pairs": len(pair_ds),
               "n_train_sequences": len(seq_ds)}
        print("b_j:", res["scales"], flush=True)

        t = time.time()
        run_phase(base, "cfm", loader, a.iters_cfm, 2e-4, log_every=max(1, a.iters_cfm // 10))
        res["shared_cfm_seconds"] = time.time() - t

        # stage 1: PINN to convergence on observed pairs, then frozen
        t = time.time()
        res["pinn_history"] = train_pinn(base, loader, val_batches, max_iters=a.pinn_max_iters,
                                         window=max(1, min(100, a.pinn_max_iters // 4)))
        res["pinn_seconds"] = time.time() - t
        res["pinn_report"] = pinn_report(base, val_batches)
        print("observed-pair residual, linear vs frozen PINN:", json.dumps(res["pinn_report"]), flush=True)
        torch.save({"model": base.state_dict(), "scales": vars(base.guide.scales)}, base_path)
        with open(res_path, "w") as f:
            json.dump(res, f, indent=1)
    freeze(base.guide)

    x_k = test_batch["x0"][:, None][..., KEEP, :, :]
    o = test_batch["targets"][..., KEEP, :, :]
    res["persistence_lead_rmse"] = metrics._wmean((x_k - o) ** 2).mean(dim=(0, 2)).sqrt().tolist()
    res["truth_lead_amp"] = metrics._wmean(o ** 2).mean(dim=(0, 2)).sqrt().tolist()
    res["truth_lead_residual"] = member_residuals(
        base, test_batch["x0"], test_batch["targets"][:, None], test_batch["static"], test_batch["doy"])

    def save():
        with open(res_path, "w") as f:
            json.dump(res, f, indent=1)

    for name in a.arms:
        torch.manual_seed(1)
        m = copy.deepcopy(base)
        t, logs = time.time(), []
        if name == "cfm":
            for q in m.cfm.parameters():
                q.requires_grad_(True)
            run_phase(m, "cfm", loader, a.iters_proj, 5e-5, log_every=max(1, a.iters_proj // 5))
            model = m
        elif name == "gradflow":
            model = PGCFM(GradientFlowProjection(m.cfm, m.guide), m.guide)
        else:
            pc = ProjectionConfig(curriculum=(1,) if name == "proj_k1" else tuple(a.curriculum),
                                  cfm_steps=a.steps,
                                  lambda_p=a.lambda_p if name in ("proj_phys", "proj_phys_gr", "proj_smooth") else 0.0,
                                  control="smooth" if name == "proj_smooth" else "physics",
                                  grad_ratio=a.grad_ratio if name == "proj_phys_gr" else 0.0)
            proj = ODEProjection(m.cfm, pc, pair_ds.img_shape_x, pair_ds.img_shape_y)
            logs = train_projection(proj, m.guide, seq_loader, a.iters_proj, lr=2e-4,
                                    log_every=max(1, a.iters_proj // 6))
            model = PGCFM(proj, m.guide)
        secs = time.time() - t
        r = evaluate(model, test_batch, val_batches, a.members, a.steps)
        r["train_seconds"] = secs
        if name != "cfm":
            r["compression"] = compression(model.cfm, val_batches, a.members, a.steps)
        if logs:
            last = logs[-max(1, len(logs) // 6):]
            r["train_log_last"] = {k: float(np.mean([h[k] for h in last if k in h])) for k in logs[-1]}
        res[name] = r
        print(f"[{name}] {secs:.0f}s 1-day", json.dumps({k: round(v, 4) for k, v in r["one_step"].items()
                                                          if not k.startswith("res_obs")}), flush=True)
        if "compression" in r:
            print(f"[{name}] compression", json.dumps({k: round(v, 4) for k, v in r["compression"].items()}),
                  flush=True)
        save()
    print_table(res, [n for n, v in res.items() if isinstance(v, dict) and "lead_rmse" in v], a.horizon)
    return res


def print_table(r, names, K):
    leads = sorted({0, min(2, K - 1), min(4, K - 1), K - 1})
    print("\nlead: RMSE / spread / CRPS / SSR / det(eps=0) RMSE / member amp   (truth amp, persistence)")
    for l in leads:
        print(f" d{l + 1}: truth amp {r['truth_lead_amp'][l]:.3f}  persistence {r['persistence_lead_rmse'][l]:.3f}")
        for n in names:
            x = r[n]
            print(f"   {n:13s} {x['lead_rmse'][l]:.3f} / {x['lead_spread'][l]:.3f} / {x['lead_crps'][l]:.3f} / "
                  f"{x['lead_ssr'][l]:.2f} / {x['lead_det_rmse'][l]:.3f} / {x['lead_member_amp'][l]:.3f}")
    print("\nindependent residual of member transitions (normalized by b_j), lead 1 / last")
    for n in ["truth"] + list(names):
        x = r["truth_lead_residual"] if n == "truth" else r[n]["lead_member_residual"]
        print(f"   {n:13s} " + "  ".join(f"{eq} {v[0]:.2f}/{v[-1]:.2f}" for eq, v in x.items()))


if __name__ == "__main__":
    main()
