"""Step 6 on KAI-format ERA5: frozen PINN -> ODE projection of the CFM cloud -> sequence loss.

Starts from a CFM checkpoint of ``train_pgcfm.py`` (phase ``cfm``), which stays frozen.

    stage "pinn": bridge h_psi + closure C_chi on observed pairs until the
                  held-out physics loss stops improving, then frozen  -> <out>/pinn.pt
    stage "proj": projection ODE g_phi trained with the unrolled rollout CRPS
                  (2 members, horizon curriculum) [+ lambda_P x frozen PINN]
                                                                    -> <out>/proj.pt

    python scripts/train_projection.py --config configs/projection_era5.yaml \
        --cfm-ckpt runs/cfm/cfm.pt --out runs/proj
    torchrun --standalone --nproc_per_node 4 scripts/train_projection.py ...   # batch is per GPU

``--resume`` skips a stage whose checkpoint exists. ``--set lambda_p=0.1 grad_ratio=0.5``
overrides yaml values. ``infer_pgcfm.py --ckpt runs/proj/proj.pt`` writes forecast.nc.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm.coupling import estimate_scales  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset, kai_data_config  # noqa: E402
from pgcfm.physics.primitive import ResidualScales  # noqa: E402
from pgcfm.projection import ODEProjection, ProjectionConfig, freeze, train_pinn, train_projection  # noqa: E402
from pgcfm.train import broadcast_module, is_main, take, validate_one_step  # noqa: E402
from scripts.train_pgcfm import build  # noqa: E402

DEFAULTS = dict(
    train_dir=None, valid_dir=None, stats_dir=None,
    stages=["pinn", "proj"],
    batch=2, workers=8, device="cuda" if torch.cuda.is_available() else "cpu", seed=0,
    # stage 1
    pinn_max_iters=20000, pinn_lr=1e-3, pinn_window=500, pinn_tol=0.01, pinn_patience=3, pinn_batch=8,
    scale_batches=50, scale_floor_frac=0.1, scales=None,
    # stage 3
    proj_iters=20000, proj_lr=2e-4, model_channels=64, channel_mult=[1, 2, 2, 4], num_res_blocks=1,
    ode_method="rk4", ode_steps=2, curriculum=[1, 2, 3, 5], members=2, cfm_steps=8, bptt=True,
    lambda_p=0.0, control="physics", grad_ratio=0.5,
    val_batches=10, val_members=4, val_every=2000,
)


def load_config(path):
    cfg = dict(DEFAULTS)
    if path:
        from ruamel.yaml import YAML
        with open(path) as f:
            cfg.update(YAML(typ="safe").load(f) or {})
    return cfg


def projection_config(cfg) -> ProjectionConfig:
    return ProjectionConfig(model_channels=cfg["model_channels"], channel_mult=tuple(cfg["channel_mult"]),
                            num_res_blocks=cfg["num_res_blocks"], ode_method=cfg["ode_method"],
                            ode_steps=cfg["ode_steps"], curriculum=tuple(cfg["curriculum"]),
                            members=cfg["members"], lambda_p=cfg["lambda_p"], control=cfg["control"],
                            grad_ratio=cfg["grad_ratio"], cfm_steps=cfg["cfm_steps"], bptt=cfg["bptt"])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--cfm-ckpt", required=True, help="train_pgcfm.py checkpoint with a trained CFM (cfm.pt)")
    p.add_argument("--out", required=True)
    p.add_argument("--stages", nargs="+")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v) if v[:1] in "[{0123456789-." or v in ("true", "false") else v
    if a.stages:
        cfg["stages"] = a.stages
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        local = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            torch.distributed.init_process_group("nccl")
            dev = torch.device("cuda", local)
        else:
            torch.distributed.init_process_group("gloo")
            dev = torch.device("cpu")
    else:
        dev = torch.device(cfg["device"])
    log = print if is_main() else (lambda *x, **k: None)

    state = torch.load(a.cfm_ckpt, map_location=dev)
    base_cfg = state["config"]
    stats = cfg["stats_dir"]
    mean_p, std_p = os.path.join(stats, "global_mean.npy"), os.path.join(stats, "global_std.npy")
    dcfg = kai_data_config(mean_p, std_p)
    K = max(cfg["curriculum"])
    pair_ds = PGCFMSequenceDataset(dcfg, cfg["train_dir"], horizon=1)
    seq_ds = PGCFMSequenceDataset(dcfg, cfg["train_dir"], horizon=K)
    valid_ds = PGCFMSequenceDataset(dcfg, cfg["valid_dir"], horizon=1, train=False)

    def loader(ds, bs):
        sampler = torch.utils.data.DistributedSampler(ds, seed=cfg["seed"]) if world > 1 else None
        return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=sampler is None, sampler=sampler,
                                           drop_last=True, num_workers=cfg["workers"],
                                           pin_memory=dev.type == "cuda", persistent_workers=cfg["workers"] > 0)

    n_lat, n_lon = pair_ds.img_shape_x, pair_ds.img_shape_y
    model = build(base_cfg, n_lat, n_lon, np.load(mean_p), np.load(std_p)).to(dev)
    model.load_state_dict(state["model"])
    vb = [{k: v.to(dev) for k, v in b.items()} for b in take(torch.utils.data.DataLoader(
        valid_ds, batch_size=cfg["pinn_batch"], num_workers=2), cfg["val_batches"])]
    log_path = os.path.join(a.out, "log.jsonl")

    def write(rec):
        if is_main():
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    # -- stage 1: PINN to convergence, then frozen ---------------------------
    pinn_ck = os.path.join(a.out, "pinn.pt")
    if a.resume and os.path.exists(pinn_ck):
        ck = torch.load(pinn_ck, map_location=dev)
        model.load_state_dict(ck["model"])
        model.guide.scales = ResidualScales(**ck["scales"])
        log("resumed frozen PINN", flush=True)
    elif "pinn" in cfg["stages"]:
        pl = loader(pair_ds, cfg["pinn_batch"])
        if cfg.get("scales"):
            sc = ResidualScales(**cfg["scales"])
        elif state.get("scales") and "guide" in state.get("phases_done", []):
            sc = ResidualScales(**state["scales"])
        else:
            sc = estimate_scales(model.guide, [{k: v.to(dev) for k, v in b.items()}
                                               for b in take(pl, cfg["scale_batches"])],
                                 floor_frac=cfg["scale_floor_frac"])
        if world > 1:
            obj = [vars(sc)]
            torch.distributed.broadcast_object_list(obj, 0)
            sc = ResidualScales(**obj[0])
        model.guide.scales = sc
        broadcast_module(model)
        log("residual scales b_j:", vars(sc), flush=True)
        hist = train_pinn(model, pl, vb, max_iters=cfg["pinn_max_iters"], lr=cfg["pinn_lr"],
                          window=cfg["pinn_window"], tol=cfg["pinn_tol"], patience=cfg["pinn_patience"],
                          device=dev, log=lambda *x: log(*x, flush=True))
        for h in hist:
            write({"stage": "pinn", **h})
        if is_main():
            torch.save({"model": model.state_dict(), "config": base_cfg, "scales": vars(model.guide.scales),
                        "pinn_history": hist}, pinn_ck)
    freeze(model.guide)

    # -- stage 3: projection ODE with the sequence loss ----------------------
    pcfg = projection_config(cfg)
    proj = ODEProjection(model.cfm, pcfg, n_lat, n_lon).to(dev)
    model.cfm = proj  # validate_one_step / evaluation see the projected sampler
    broadcast_module(proj.field)
    log(f"projection g_phi {sum(q.numel() for q in proj.field.parameters()):,} params, "
        f"curriculum {pcfg.curriculum}, lambda_p {pcfg.lambda_p} ({pcfg.control})", flush=True)

    def on_log(it, avg):
        rec = {"stage": "proj", "it": it, **avg}
        if it % cfg["val_every"] == 0 or it == cfg["proj_iters"]:
            rec["val"] = validate_one_step(model, vb, cfg["val_members"], steps=pcfg.cfm_steps, device=dev)
            proj.train()
            log("    val " + " ".join(f"{k} {v:.4g}" for k, v in rec["val"].items()), flush=True)
        write(rec)

    if "proj" in cfg["stages"]:
        train_projection(proj, model.guide, loader(seq_ds, cfg["batch"]), cfg["proj_iters"], lr=cfg["proj_lr"],
                         device=dev, log_every=min(100, cfg["proj_iters"]), log=lambda *x: log(*x, flush=True),
                         on_log=on_log)
        if is_main():
            torch.save({"model": model.state_dict(), "config": base_cfg, "projection": cfg,
                        "scales": vars(model.guide.scales)}, os.path.join(a.out, "proj.pt"))
    log("done")
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
