"""Train PG-CFM on KAI-format ERA5 (appendix C), CFM-only or coupled.

Data: the KAI yearly NetCDF files (``fields[T,67,121,240]``), one directory
per split, and KAI's ``global_mean.npy`` / ``global_std.npy``. On the owner's
cluster that is train 1980-2011, valid 2012 (configs/pgcfm_era5.yaml).

    # full PG-CFM: CFM alone -> bridge/closure on observed pairs -> joint
    python scripts/train_pgcfm.py --config configs/pgcfm_era5.yaml --out runs/pgcfm

    # CFM-only baseline with the same backbone and the same theta updates
    python scripts/train_pgcfm.py --config configs/pgcfm_era5.yaml --out runs/cfm --phases cfm cfm_more

Each phase writes ``<out>/<phase>.pt``; ``--resume`` restarts after the
last finished phase. ``cfm_more`` is the CFM-only continuation that matches
the joint phase in iterations.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm.cfm import CFMConfig, CFMTransition  # noqa: E402
from pgcfm.coupling import PGCFM, PhysicsConfig, PhysicsGuide, estimate_scales  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset, kai_data_config  # noqa: E402
from pgcfm.physics.primitive import ResidualScales  # noqa: E402
from pgcfm.train import run_phase, take, validate_one_step  # noqa: E402

DEFAULTS = dict(
    train_dir=None, valid_dir=None, stats_dir=None,
    phases=["cfm", "guide", "joint"],
    iters={"cfm": 100000, "guide": 5000, "joint": 20000, "cfm_more": 20000},
    lr={"cfm": 2e-4, "guide": 1e-3, "joint": 5e-5, "cfm_more": 5e-5},
    batch=8, workers=8, device="cuda" if torch.cuda.is_available() else "cpu",
    tisr_mode="target", model_channels=64, channel_mult=[1, 2, 2, 4], num_res_blocks=2,
    lambda_p=0.1, ramp_iters=5000, lambda_h=1e-2, lambda_s=0.0, sample_steps=2, n_sub=4,
    gamma={"u": 1.0, "v": 1.0, "T": 1.0, "mass": 1.0, "hyd": 1.0, "q": 0.0},
    pole_cap_deg=85.0, scale_batches=50, val_batches=10, val_members=4, val_every=2000,
    seed=0,
)


def load_config(path):
    cfg = dict(DEFAULTS)
    if path:
        from ruamel.yaml import YAML
        with open(path) as f:
            user = YAML(typ="safe").load(f) or {}
        for k, v in user.items():
            if isinstance(cfg.get(k), dict) and isinstance(v, dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    return cfg


def build(cfg, n_lat, n_lon, mean, std, scales=None):
    cfm = CFMTransition(CFMConfig(tisr_mode=cfg["tisr_mode"], model_channels=cfg["model_channels"],
                                  channel_mult=tuple(cfg["channel_mult"]),
                                  num_res_blocks=cfg["num_res_blocks"]), n_lat, n_lon)
    pcfg = PhysicsConfig(lambda_p=cfg["lambda_p"], ramp_iters=cfg["ramp_iters"], lambda_h=cfg["lambda_h"],
                         lambda_s=cfg["lambda_s"], gamma=dict(cfg["gamma"]), n_sub=cfg["n_sub"],
                         sample_steps=cfg["sample_steps"], pole_cap_deg=cfg["pole_cap_deg"])
    guide = PhysicsGuide(n_lat, n_lon, mean, std, pcfg, scales or ResidualScales())
    return PGCFM(cfm, guide)


def save(path, model, cfg, done):
    torch.save({"model": model.state_dict(), "config": cfg, "phases_done": done,
                "scales": vars(model.guide.scales)}, path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--out", required=True)
    p.add_argument("--phases", nargs="+")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--set", nargs="*", default=[], help="override, e.g. --set lambda_p=0.3 batch=4")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v) if v[:1] in "[{0123456789-." or v in ("true", "false") else v
    if a.phases:
        cfg["phases"] = a.phases
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    dev = torch.device(cfg["device"])

    stats = cfg["stats_dir"]
    mean_p, std_p = os.path.join(stats, "global_mean.npy"), os.path.join(stats, "global_std.npy")
    dcfg = kai_data_config(mean_p, std_p)
    train_ds = PGCFMSequenceDataset(dcfg, cfg["train_dir"], horizon=1)
    valid_ds = PGCFMSequenceDataset(dcfg, cfg["valid_dir"], horizon=1, train=False)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True, drop_last=True,
                                         num_workers=cfg["workers"], pin_memory=dev.type == "cuda",
                                         persistent_workers=cfg["workers"] > 0)
    vloader = torch.utils.data.DataLoader(valid_ds, batch_size=cfg["batch"], shuffle=False,
                                          num_workers=min(2, cfg["workers"]))
    n_lat, n_lon = train_ds.img_shape_x, train_ds.img_shape_y
    model = build(cfg, n_lat, n_lon, np.load(mean_p), np.load(std_p)).to(dev)
    print(f"train {len(train_ds)} pairs, valid {len(valid_ds)}, grid {n_lat}x{n_lon}, "
          f"theta {sum(q.numel() for q in model.cfm.parameters()):,}, "
          f"psi {sum(q.numel() for q in model.guide.parameters()):,}, device {dev}", flush=True)

    done = []
    if a.resume:
        for ph in reversed(cfg["phases"]):
            ck = os.path.join(a.out, f"{ph}.pt")
            if os.path.exists(ck):
                state = torch.load(ck, map_location=dev)
                model.load_state_dict(state["model"])
                model.guide.scales = ResidualScales(**state["scales"])
                done = state["phases_done"]
                print(f"resumed after {done}", flush=True)
                break

    vb = take(vloader, cfg["val_batches"])
    log_path = os.path.join(a.out, "log.jsonl")

    for ph in cfg["phases"]:
        if ph in done:
            continue
        if ph in ("guide", "joint") and "scales" not in done:
            # b_j from observed pairs (sec. 5.1), fixed for the rest of training
            model.guide.scales = estimate_scales(model.guide, [
                {k: v.to(dev) for k, v in b.items()} for b in take(loader, cfg["scale_batches"])])
            done.append("scales")
            print("residual scales b_j:", vars(model.guide.scales), flush=True)

        def on_log(it, avg, ph=ph):
            rec = {"phase": ph, "it": it, **avg}
            if it % cfg["val_every"] == 0 or it == cfg["iters"][ph]:
                rec["val"] = validate_one_step(model, vb, cfg["val_members"], device=dev)
                model.train()
                print("    val " + " ".join(f"{k} {v:.4g}" for k, v in rec["val"].items()), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        base = "joint" if ph == "joint" else ("cfm" if ph == "cfm_more" else ph)
        run_phase(model, base, loader, cfg["iters"][ph], cfg["lr"][ph], device=dev,
                  log_every=min(100, cfg["iters"][ph]), on_log=on_log)
        done.append(ph)
        save(os.path.join(a.out, f"{ph}.pt"), model, cfg, done)
    print("done:", done)


if __name__ == "__main__":
    main()
