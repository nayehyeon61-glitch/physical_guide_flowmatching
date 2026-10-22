"""Step 2 follow-up diagnostics on the trained forcing-mode checkpoint.

1. Heun step count S vs lead-1 spread/RMSE (is over-dispersion a sampler artifact?)
2. Spatial structure of member deviations vs the oracle (white leftover noise?)
3. Continue training 3000 more iterations (is it under-training?)

    python scripts/step2_diagnostics.py --run runs/step2
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import metrics  # noqa: E402
from pgcfm.cfm import TISR_DYN, CFMConfig, CFMTransition, rollout  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset  # noqa: E402
from scripts.run_step2_cfm import AR_DYN, oracle_rollout, train  # noqa: E402
from pgcfm.data import kai_data_config  # noqa: E402


def lead1(model, batch, obs, steps, members=16):
    tisr = obs[:, :1, TISR_DYN:TISR_DYN + 1]
    ens = rollout(model, batch["x0"], batch["static"], batch["doy"][:, :2], members, tisr=tisr, steps=steps)
    ens = ens[:, :, 0].transpose(0, 1)  # (M,B,C,H,W)
    o = obs[:, 0]
    sp = metrics.ens_spread(ens).pow(2).mean(0).sqrt()[AR_DYN].mean().item()
    rm = metrics.ens_mean_rmse(ens, o).pow(2).mean(0).sqrt()[AR_DYN].mean().item()
    cr = metrics.crps(ens, o).mean(0)[AR_DYN].mean().item()
    return {"spread": sp, "rmse": rm, "crps": cr}


def deviation_structure(ens):
    """ens (B,M,C,H,W): spectral share of member deviations and lon lag-1 correlation (AR channels)."""
    dev = (ens - ens.mean(1, keepdim=True))[:, :, AR_DYN]
    H = dev.shape[-2]
    pz = torch.fft.rfft(dev[..., H // 6:H - H // 6, :], dim=-1).abs().pow(2).mean(dim=(0, 1, 2, 3))
    return {
        "zonal_k>4_share": (pz[5:].sum() / pz.sum()).item(),
        "zonal_k>12_share": (pz[13:].sum() / pz.sum()).item(),
        "lon_lag1_corr": ((dev * dev.roll(1, -1)).mean() / dev.pow(2).mean()).item(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/step2")
    p.add_argument("--extra-iters", type=int, default=3000)
    a = p.parse_args()
    data = os.path.join(a.run, "data")
    stats = os.path.join(data, "train", "stats")
    dcfg = kai_data_config(os.path.join(stats, "global_mean.npy"), os.path.join(stats, "global_std.npy"))
    train_ds = PGCFMSequenceDataset(dcfg, os.path.join(data, "train"), horizon=1)
    test_ds = PGCFMSequenceDataset(dcfg, os.path.join(data, "test"), horizon=1)
    batch = torch.utils.data.default_collate([test_ds[i] for i in range(0, 24, 2)])
    obs = batch["targets"]
    H, W = obs.shape[-2:]

    model = CFMTransition(CFMConfig(tisr_mode="forcing"), H, W)
    model.load_state_dict(torch.load(os.path.join(a.run, "cfm_forcing.pt"))["model"])
    model.eval()
    out = {"n_starts": obs.shape[0]}
    orc = oracle_rollout(batch["x0"], batch["static"], obs, 16, seed=1)[:, :, 0].transpose(0, 1)
    out["oracle"] = {
        "spread": metrics.ens_spread(orc).pow(2).mean(0).sqrt()[AR_DYN].mean().item(),
        "rmse": metrics.ens_mean_rmse(orc, obs[:, 0]).pow(2).mean(0).sqrt()[AR_DYN].mean().item(),
        "crps": metrics.crps(orc, obs[:, 0]).mean(0)[AR_DYN].mean().item(),
    }
    out["steps"] = {S: lead1(model, batch, obs, S) for S in [1, 2, 4, 8, 16, 32]}
    tisr = obs[:, :1, TISR_DYN:TISR_DYN + 1]
    ens = rollout(model, batch["x0"], batch["static"], batch["doy"][:, :2], 16, tisr=tisr, steps=8)[:, :, 0]
    out["deviation_structure"] = {"model": deviation_structure(ens),
                                  "oracle": deviation_structure(orc.transpose(0, 1))}
    print(json.dumps(out, indent=1), flush=True)

    hist, secs = train(model, train_ds, a.extra_iters, 16, 1e-4, seed=1, log_every=500)
    out["after_extra_training"] = {"iters_total": 3000 + a.extra_iters, "loss_last100": float(np.mean(hist[-100:])),
                                   "S8": lead1(model, batch, obs, 8)}
    print(json.dumps(out["after_extra_training"], indent=1), flush=True)
    with open(os.path.join(a.run, "diagnostics.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
