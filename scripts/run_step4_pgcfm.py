"""Step 4 experiment: CFM alone vs PG-CFM on balanced synthetic KAI-format data.

Protocol (appendix C, matched theta updates):
    shared   : phase "cfm" for --iters-cfm
    cfm      : + "cfm_more" for --iters-joint            (CFM-only baseline)
    pgcfm_L  : + "guide" for --iters-guide (theta frozen) + "joint" for
               --iters-joint with lambda_P = L          (one arm per --lambdas)

Scores on held-out 2020 starts, all in normalized units:
    1-day     CRPS / ens-mean RMSE / spread / SSR / tendency ratio and the
              physics residual of members, both along the trained bridge with
              closure (what L_phys sees) and along the linear bridge without
              closure (independent: h_psi and C_chi cannot make it small)
    rollout   per lead: RMSE, spread, CRPS, independent residual of each
              member transition (x_k^m, x_{k+1}^m), member amplitude,
              persistence and zero-noise (eps = 0) RMSE

    python scripts/run_step4_pgcfm.py --out runs/step4 [--quick]
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
from pgcfm.cfm import TISR_DYN, rollout  # noqa: E402
from pgcfm.coupling import deterministic_rollout, estimate_scales  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset, kai_data_config  # noqa: E402
from pgcfm.train import run_phase, take, validate_one_step  # noqa: E402
from scripts.make_balanced_era5 import main as make_balanced  # noqa: E402
from scripts.train_pgcfm import DEFAULTS, build  # noqa: E402

KEEP = [i for i in range(66) if i != TISR_DYN]


@torch.no_grad()
def member_residuals(model, x0, ens, static, doy):
    """Independent residual of every member transition. ens (B,M,K,66,H,W) -> {eq: (K,)}."""
    B, M, K = ens.shape[:3]
    g = model.guide
    out = {}
    prev = x0[:, None].expand(B, M, *x0.shape[1:])
    st = static[:, None].expand(B, M, *static.shape[1:]).reshape(B * M, *static.shape[1:])
    for k in range(K):
        x, y = prev.reshape(B * M, *x0.shape[1:]), ens[:, :, k].reshape(B * M, *x0.shape[1:])
        R, _ = g.residuals(x, y, st, doy[:, k].repeat_interleave(M), use_bridge=False, use_closure=False)
        m = g.mask(x, y, st)
        for eq, v in g.per_equation(R, m).items():
            out.setdefault(eq, []).append(v)
        prev = ens[:, :, k]
    return out


@torch.no_grad()
def bridge_residual(model, batches):
    """Residual of generated members along the TRAINED bridge with closure (what L_phys sees)."""
    g, acc = model.guide, {}
    for b in batches:
        x, y, st = b["x0"], b["targets"][:, 0], b["static"]
        y_hat = model.cfm.sample(x, st, b["doy"][:, 1], steps=8)
        R, _ = g.residuals(x, y_hat, st, b["doy"][:, 0])
        for eq, v in g.per_equation(R, g.mask(x, y, st)).items():
            acc.setdefault(eq, []).append(v)
        R, _ = g.residuals(x, y, st, b["doy"][:, 0])
        for eq, v in g.per_equation(R, g.mask(x, y, st)).items():
            acc.setdefault("obs_" + eq, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def evaluate(model, test_batch, val_batches, members, steps):
    t0 = time.time()
    x0, st, doy, obs = test_batch["x0"], test_batch["static"], test_batch["doy"], test_batch["targets"]
    ens = rollout(model.cfm, x0, st, doy, members, steps=steps, seed=1)  # (B,M,K,66,H,W)
    det = deterministic_rollout(model.cfm, x0, st, doy, steps=steps)  # (B,K,66,H,W)
    e = ens.transpose(0, 1)[..., KEEP, :, :]  # (M,B,K,C,H,W)
    o = obs[..., KEEP, :, :]
    res = {
        "one_step": validate_one_step(model, val_batches, n_members=members, steps=steps),
        "bridge_residual": bridge_residual(model, val_batches),
        "lead_rmse": metrics.ens_mean_rmse(e, o).pow(2).mean(dim=(0, 2)).sqrt().tolist(),
        "lead_spread": metrics.ens_spread(e).pow(2).mean(dim=(0, 2)).sqrt().tolist(),
        "lead_crps": [float(metrics.crps(e[:, :, k], o[:, k]).mean()) for k in range(o.shape[1])],
        "lead_det_rmse": metrics._wmean((det[..., KEEP, :, :] - o) ** 2).mean(dim=(0, 2)).sqrt().tolist(),
        "lead_member_amp": metrics._wmean(e ** 2).mean(dim=(0, 1, 3)).sqrt().tolist(),
        "lead_member_residual": member_residuals(model, x0, ens, st, doy),
        "eval_seconds": time.time() - t0,
    }
    M = members
    res["lead_ssr"] = [((M + 1) / M) ** 0.5 * s / r for s, r in zip(res["lead_spread"], res["lead_rmse"])]
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/step4")
    p.add_argument("--n-lat", type=int, default=33)
    p.add_argument("--n-lon", type=int, default=64)
    p.add_argument("--train-years", type=int, nargs="+", default=[2001, 2002, 2003, 2004])
    p.add_argument("--train-days", type=int, default=90)
    p.add_argument("--iters-cfm", type=int, default=1500)
    p.add_argument("--iters-guide", type=int, default=300)
    p.add_argument("--iters-joint", type=int, default=500)
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 1.0])
    p.add_argument("--smooth-control", action="store_true",
                   help="extra arm: same schedule with a non-physical Laplacian penalty instead of L_phys")
    p.add_argument("--grad-ratios", type=float, nargs="*", default=[],
                   help="extra arms lambda_P = max(lambdas) with the physics gradient capped at r x CFM's")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--n-starts", type=int, default=4)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.train_years, a.train_days, a.iters_cfm, a.iters_guide, a.iters_joint = [2001], 10, 4, 2, 2
        a.horizon, a.n_starts, a.members, a.steps, a.n_lat, a.n_lon = 2, 1, 2, 2, 17, 32
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
    train_ds = PGCFMSequenceDataset(dcfg, tr, horizon=1)
    test_ds = PGCFMSequenceDataset(dcfg, te, horizon=a.horizon, train=False)
    pair_ds = PGCFMSequenceDataset(dcfg, te, horizon=1, train=False)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=a.batch, shuffle=True, drop_last=True)
    starts = [5 * i for i in range(a.n_starts)]
    test_batch = torch.utils.data.default_collate([test_ds[i] for i in starts])
    val_batches = take(torch.utils.data.DataLoader(pair_ds, batch_size=a.batch), 2)

    cfg = dict(DEFAULTS, model_channels=32, channel_mult=[1, 2, 2], num_res_blocks=1,
               ramp_iters=max(1, a.iters_joint // 3), sample_steps=2)
    torch.manual_seed(0)
    base = build(cfg, train_ds.img_shape_x, train_ds.img_shape_y, np.load(mean_p), np.load(std_p))
    base.guide.scales = estimate_scales(base.guide, take(loader, 10))
    results = {"args": vars(a), "scales": vars(base.guide.scales), "n_train_pairs": len(train_ds)}
    print("b_j:", results["scales"], flush=True)

    t = time.time()
    run_phase(base, "cfm", loader, a.iters_cfm, 2e-4, log_every=max(1, a.iters_cfm // 10))
    results["shared_cfm_seconds"] = time.time() - t

    x_k = test_batch["x0"][:, None][..., KEEP, :, :]
    o = test_batch["targets"][..., KEEP, :, :]
    results["persistence_lead_rmse"] = metrics._wmean((x_k - o) ** 2).mean(dim=(0, 2)).sqrt().tolist()
    results["truth_lead_amp"] = metrics._wmean(o ** 2).mean(dim=(0, 2)).sqrt().tolist()
    results["truth_lead_residual"] = member_residuals(
        base, test_batch["x0"], test_batch["targets"][:, None], test_batch["static"], test_batch["doy"])

    arms = [("cfm", None, 0.0, "physics")] + [(f"pgcfm_{lam:g}", lam, 0.0, "physics") for lam in a.lambdas] + \
        [(f"pgcfm_{max(a.lambdas):g}_gr{r:g}", max(a.lambdas), r, "physics") for r in a.grad_ratios] + \
        ([(f"smooth_{max(a.lambdas):g}", max(a.lambdas), 0.0, "smooth")] if a.smooth_control else [])
    for name, lam, gr, ctl in arms:
        torch.manual_seed(1)
        m = copy.deepcopy(base)
        t = time.time()
        if lam is None:
            run_phase(m, "cfm", loader, a.iters_joint, 5e-5, log_every=max(1, a.iters_joint // 5))
            logs = []
        else:
            m.guide.cfg.lambda_p, m.guide.cfg.grad_ratio, m.guide.cfg.control = lam, gr, ctl
            run_phase(m, "guide", loader, a.iters_guide, 1e-3, log_every=max(1, a.iters_guide // 3))
            logs = run_phase(m, "joint", loader, a.iters_joint, 5e-5, log_every=max(1, a.iters_joint // 5))
        secs = time.time() - t
        r = evaluate(m, test_batch, val_batches, a.members, a.steps)
        r["train_seconds"] = secs
        if logs:
            r["joint_log_last50"] = {k: float(np.mean([h[k] for h in logs[-50:] if k in h])) for k in logs[-1]}
        results[name] = r
        print(f"[{name}] 1-day", json.dumps({k: round(v, 4) for k, v in r["one_step"].items()}), flush=True)
        with open(os.path.join(a.out, "results.json"), "w") as f:
            json.dump(results, f, indent=1)
    print_table(results, [n for n, *_ in arms], a.horizon)
    return results


def print_table(r, names, K):
    leads = sorted({0, min(2, K - 1), min(4, K - 1), K - 1})
    print("\nlead: RMSE / spread / CRPS / SSR / det(eps=0) RMSE / member amp   (truth amp, persistence)")
    for l in leads:
        print(f" d{l + 1}: truth amp {r['truth_lead_amp'][l]:.3f}  persistence {r['persistence_lead_rmse'][l]:.3f}")
        for n in names:
            x = r[n]
            print(f"   {n:12s} {x['lead_rmse'][l]:.3f} / {x['lead_spread'][l]:.3f} / {x['lead_crps'][l]:.3f} / "
                  f"{x['lead_ssr'][l]:.2f} / {x['lead_det_rmse'][l]:.3f} / {x['lead_member_amp'][l]:.3f}")
    print("\nindependent residual of member transitions (normalized by b_j), lead 1 / last")
    for n in ["truth"] + names:
        x = r["truth_lead_residual"] if n == "truth" else r[n]["lead_member_residual"]
        print(f"   {n:12s} " + "  ".join(f"{eq} {v[0]:.2f}/{v[-1]:.2f}" for eq, v in x.items()))


if __name__ == "__main__":
    main()
