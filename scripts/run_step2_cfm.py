"""Step 2 experiment: CFM alone on synthetic KAI-format ERA5.

Trains the one-step CFM twice (tisr generated vs tisr given as forcing),
then scores 1-step and K-step member-wise rollouts against the truth and
against an oracle ensemble drawn from the synthetic generator itself.

The synthetic generator is an AR(1) in z-score space (make_synthetic_era5):
    a_{k+1} = 0.8 a_k + 0.6 eta,   x = mean + 0.5 std a
so the true conditional p(x_{k+1} | x_k) is known and can be sampled. That
oracle gives the spread a perfect model should have, which separates
"under/over-dispersion of the CFM" from "the task is hard".

    python scripts/run_step2_cfm.py --out runs/step2 [--quick]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract, metrics  # noqa: E402
from pgcfm.cfm import TISR_DYN, CFMConfig, CFMTransition, rollout  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset, kai_data_config  # noqa: E402
from scripts.make_synthetic_era5 import main as make_synthetic, smooth_noise  # noqa: E402

DYN = np.array(contract.DYNAMIC_CHANNELS)
# Channels whose synthetic transition is the plain AR(1) (q has abs(), sp/tisr are special).
Q_DYN = [i for i, c in enumerate(DYN) if contract.CHANNEL_NAMES[c].startswith("q")]
SP_DYN = list(DYN).index(contract.CHANNEL_INDEX["sp"])
AR_DYN = [i for i in range(len(DYN)) if i not in Q_DYN + [SP_DYN, TISR_DYN]]


def make_data(root, n_lat, n_lon, train_years, train_days, test_days):
    tr, te = os.path.join(root, "train"), os.path.join(root, "test")
    make_synthetic(["--out", tr, "--years", *map(str, train_years), "--days", str(train_days),
                    "--n-lat", str(n_lat), "--n-lon", str(n_lon), "--seed", "0"])
    make_synthetic(["--out", te, "--years", "2020", "--days", str(test_days),
                    "--n-lat", str(n_lat), "--n-lon", str(n_lon), "--seed", "1000"])
    stats = os.path.join(tr, "stats")
    cfg = kai_data_config(os.path.join(stats, "global_mean.npy"), os.path.join(stats, "global_std.npy"))
    return tr, te, cfg


def train(model, ds, iters, batch, lr, seed, log_every=100):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    loader = torch.utils.data.DataLoader(ds, batch_size=min(batch, len(ds)), shuffle=True, drop_last=True)
    it, hist, t0 = 0, [], time.time()
    model.train()
    while it < iters:
        for b in loader:
            loss = model.loss(b["x0"], b["targets"][:, 0], b["static"], b["doy"][:, 1])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            hist.append(loss.item())
            it += 1
            if it % log_every == 0:
                print(f"  it {it:5d}  loss {np.mean(hist[-log_every:]):.4f}  ({time.time() - t0:.0f}s)", flush=True)
            if it >= iters:
                break
    model.eval()
    return hist, time.time() - t0


def sp_baseline(static):
    """Normalized orography-only surface pressure (the generator's sp baseline). (B,H,W)"""
    o, i = contract.CHANNEL_INDEX["orog"], contract.CHANNEL_INDEX["sp"]
    orog = static[:, 0].double() * (contract.REFERENCE_STD[o] + contract.KAI_STD_EPS) + contract.REFERENCE_MEAN[o]
    sp = 101325.0 * torch.exp(-orog / (contract.CONST.r_dry * 255.0))
    return ((sp - contract.REFERENCE_MEAN[i]) / (contract.REFERENCE_STD[i] + contract.KAI_STD_EPS)).float()


def oracle_rollout(x0, static, targets, n_members, seed):
    """Sample the synthetic generator's own transition in z-score space.

    x0 (B,66,H,W), targets (B,K,66,H,W) (only tisr is read: it is deterministic).
    Returns (B,M,K,66,H,W). q channels are approximate (the generator's abs() is ignored).
    """
    rng = np.random.default_rng(seed)
    B, K = targets.shape[:2]
    H, W = x0.shape[-2:]
    std = contract.REFERENCE_STD[DYN]
    scale = torch.tensor(std / (std + contract.KAI_STD_EPS), dtype=torch.float32)[:, None, None]
    sp0 = sp_baseline(static)
    out = torch.empty((B, n_members, K) + x0.shape[1:])
    for b in range(B):
        for m in range(n_members):
            x = x0[b].clone()
            for k in range(K):
                eta = torch.tensor(smooth_noise(rng, (contract.N_CHANNELS,), H, W)[DYN], dtype=torch.float32)
                nxt = 0.8 * x + 0.3 * scale * eta
                # sp = baseline(orog) + 0.1 std a, i.e. 0.2x the amplitude of the other channels
                nxt[SP_DYN] = sp0[b] + 0.8 * (x[SP_DYN] - sp0[b]) + 0.06 * scale[SP_DYN, 0] * eta[SP_DYN]
                nxt[TISR_DYN] = targets[b, k, TISR_DYN]
                out[b, m, k] = x = nxt
    return out


def score(ens, obs):
    """ens (M,K,C,H,W), obs (K,C,H,W) -> per lead/channel tensors (K,C)."""
    return {
        "rmse": metrics.ens_mean_rmse(ens, obs),
        "spread": metrics.ens_spread(ens),
        "crps": metrics.crps(ens, obs),
    }


def summarize(scores_list):
    """Average (K,C) scores over start dates, then SSR from averaged MSE/var."""
    out = {}
    for key in ["rmse", "spread", "crps"]:
        if key == "crps":
            out[key] = torch.stack([s[key] for s in scores_list]).mean(0)
        else:
            out[key] = torch.stack([s[key] ** 2 for s in scores_list]).mean(0).sqrt()
    return out


def group(x, idx):
    return x[..., idx].mean(-1)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/step2")
    p.add_argument("--n-lat", type=int, default=25)
    p.add_argument("--n-lon", type=int, default=48)
    p.add_argument("--train-years", type=int, nargs="+", default=list(range(2001, 2009)))
    p.add_argument("--train-days", type=int, default=120)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--n-starts", type=int, default=4)
    p.add_argument("--members", type=int, default=16)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--modes", nargs="+", default=["target", "forcing"])
    p.add_argument("--quick", action="store_true", help="tiny run for CI / smoke tests")
    a = p.parse_args(argv)
    if a.quick:
        a.train_years, a.train_days, a.iters, a.horizon, a.n_starts, a.members, a.steps = \
            [2001], 8, 4, 2, 1, 3, 2
    os.makedirs(a.out, exist_ok=True)

    test_days = a.horizon + 1 + 3 * a.n_starts
    tr, te, dcfg = make_data(os.path.join(a.out, "data"), a.n_lat, a.n_lon, a.train_years, a.train_days, test_days)
    train_ds = PGCFMSequenceDataset(dcfg, tr, horizon=1)
    test_ds = PGCFMSequenceDataset(dcfg, te, horizon=a.horizon)
    starts = [3 * i for i in range(a.n_starts)]
    batch = torch.utils.data.default_collate([test_ds[i] for i in starts])
    obs = batch["targets"]  # (B,K,66,H,W)
    print(f"train samples {len(train_ds)}, test starts {starts}, grid {a.n_lat}x{a.n_lon}", flush=True)

    results = {"args": vars(a), "channels": [contract.CHANNEL_NAMES[c] for c in DYN]}
    oracle = oracle_rollout(batch["x0"], batch["static"], obs, a.members, seed=1)
    ens_by_name = {"oracle": oracle}

    for mode in a.modes:
        print(f"[{mode}] training", flush=True)
        torch.manual_seed(0)
        model = CFMTransition(CFMConfig(tisr_mode=mode), a.n_lat, a.n_lon)
        ckpt = os.path.join(a.out, f"cfm_{mode}.pt")
        setup = {k: getattr(a, k) for k in ["n_lat", "n_lon", "train_years", "train_days", "iters", "batch", "lr"]}
        if os.path.exists(ckpt):  # resume an interrupted run without retraining
            saved = torch.load(ckpt)
            if saved.get("setup") != setup:
                raise RuntimeError(f"{ckpt} was trained with {saved.get('setup')}, not {setup}; "
                                   "use a new --out or delete the checkpoint")
            model.load_state_dict(saved["model"])
            model.eval()
            hist, secs = saved["hist"], saved["secs"]
        else:
            hist, secs = train(model, train_ds, a.iters, a.batch, a.lr, seed=0,
                               log_every=max(1, a.iters // 10))
            torch.save({"model": model.state_dict(), "hist": hist, "secs": secs, "setup": setup}, ckpt)
        print(f"[{mode}] rollout", flush=True)
        t0 = time.time()
        tisr = obs[:, :, TISR_DYN:TISR_DYN + 1] if mode == "forcing" else None
        ens = rollout(model, batch["x0"], batch["static"], batch["doy"], a.members, tisr=tisr, steps=a.steps)
        results[mode] = {
            "n_params": sum(p.numel() for p in model.parameters()),
            "train_seconds": secs,
            "rollout_seconds": time.time() - t0,
            "loss_first100": float(np.mean(hist[:100])),
            "loss_last100": float(np.mean(hist[-100:])),
        }
        ens_by_name[mode] = ens

    # Persistence and the (deterministic) oracle mean as reference points.
    for name, ens in ens_by_name.items():
        s = summarize([score(ens[b], obs[b]) for b in range(obs.shape[0])])
        amp = metrics._wmean(ens.pow(2)).mean(dim=(0, 1)).sqrt()  # (K,C): field amplitude
        r = results.setdefault(name, {})
        r["lead_rmse_nontisr"] = group(s["rmse"], [i for i in range(66) if i != TISR_DYN]).tolist()
        r["lead_spread_nontisr"] = group(s["spread"], [i for i in range(66) if i != TISR_DYN]).tolist()
        r["lead_crps_nontisr"] = group(s["crps"], [i for i in range(66) if i != TISR_DYN]).tolist()
        r["lead_rmse_ar"] = group(s["rmse"], AR_DYN).tolist()
        r["lead_spread_ar"] = group(s["spread"], AR_DYN).tolist()
        r["lead_crps_ar"] = group(s["crps"], AR_DYN).tolist()
        r["lead_amp_ar"] = group(amp, AR_DYN).tolist()
        r["lead_tisr_rmse"] = s["rmse"][:, TISR_DYN].tolist()
        r["lead_tisr_spread"] = s["spread"][:, TISR_DYN].tolist()
        r["lead_tisr_crps"] = s["crps"][:, TISR_DYN].tolist()
        r["lead1_spread_per_channel"] = s["spread"][0].tolist()
        r["lead1_rmse_per_channel"] = s["rmse"][0].tolist()
    obs_amp = metrics._wmean(obs.pow(2)).mean(0).sqrt()
    results["truth_lead_amp_ar"] = group(obs_amp, AR_DYN).tolist()
    pers = batch["x0"][:, None].expand_as(obs)
    results["persistence_lead_rmse_ar"] = group(
        metrics._wmean((pers - obs) ** 2).mean(0).sqrt(), AR_DYN).tolist()

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print_table(results, a)
    return results


def print_table(r, a):
    names = ["oracle"] + list(a.modes)
    K = a.horizon
    leads = sorted({0, min(2, K - 1), min(4, K - 1), min(9, K - 1), K - 1})
    print("\nAR channels (z,t,u,v,t2m,tcwv,msl,skt): RMSE / spread / CRPS / amplitude")
    for name in names:
        x = r[name]
        row = "  ".join(f"d{l + 1}: {x['lead_rmse_ar'][l]:.3f}/{x['lead_spread_ar'][l]:.3f}/"
                        f"{x['lead_crps_ar'][l]:.3f}/{x['lead_amp_ar'][l]:.3f}" for l in leads)
        print(f"  {name:8s} {row}")
    print("  truth amp " + "  ".join(f"d{l + 1}: {r['truth_lead_amp_ar'][l]:.3f}" for l in leads))
    print("tisr: RMSE / spread / CRPS")
    for name in names:
        x = r[name]
        print(f"  {name:8s} " + "  ".join(f"d{l + 1}: {x['lead_tisr_rmse'][l]:.4f}/{x['lead_tisr_spread'][l]:.4f}/"
                                          f"{x['lead_tisr_crps'][l]:.4f}" for l in leads))


if __name__ == "__main__":
    main()
