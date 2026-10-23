"""Step 7 on KAI-format ERA5: DCT latent PG-CFM (``pgcfm/latent.py``), single GPU.

    stage "stage1": DCT statistics, then encoder -> latent PINN -> decoder jointly
                    (reconstruction + prediction + lambda_P x primitive-equation residual
                    of the decoded pair); the PINN is frozen               -> <out>/stage1.pt
    stage "stage2": ``coupling: anchor`` (parallel): latent CFM around the frozen PINN step,
                    encoder/decoder frozen (``stage2_mode: frozen``, recommended) or retrained;
                    ``coupling: serial``: CFM members -> trainable PINN Q per member -> decoder,
                    CFM and Q trained together (``train_serial``)         -> <out>/latent.pt
    stage "decft" : optional low-LR decoder fine-tune on decoded CRPS       -> <out>/latent.pt

    python scripts/train_latent.py --config configs/latent_era5.yaml --out runs/latent
    python scripts/infer_pgcfm.py --ckpt runs/latent/latent.pt ...   # forecast.nc as for the other steps

``--resume`` skips stages whose checkpoint exists; ``--set k=v`` overrides yaml values.
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm.coupling import PGCFM, estimate_scales  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset, kai_data_config  # noqa: E402
from pgcfm.latent import (LatentConfig, LatentPGCFM, finetune_decoder, floors, train_serial,  # noqa: E402
                          train_stage1, train_stage2)
from pgcfm.physics.primitive import ResidualScales  # noqa: E402
from pgcfm.projection import freeze  # noqa: E402
from pgcfm.train import take, validate_one_step  # noqa: E402
from scripts.train_pgcfm import DEFAULTS as BASE_DEFAULTS, build  # noqa: E402

DEFAULTS = dict(
    train_dir=None, valid_dir=None, stats_dir=None,
    stages=["stage1", "stage2"],
    batch=8, workers=8, device="cuda" if torch.cuda.is_available() else "cpu", seed=0,
    k_lat=24, k_lon=48, latent=512, enc_hidden=[1024, 512], pinn_hidden=[1024, 1024], act="tanh", cfm_width=1024,
    carry_highpass=True, coupling="serial", serial_iters=20000, serial_lr=1e-3, serial_steps=4, serial_members=2,
    dct_fit_batches=200, scale_batches=50, scale_floor_frac=0.1, scales=None,
    stage1_iters=50000, stage1_lr=1e-3, lambda_p=0.1, grad_ratio=0.5,
    stage2_iters=50000, stage2_lr=1e-3, stage2_mode="frozen", ae_lr=1e-4,
    decft_iters=2000, decft_lr=1e-5,
    val_batches=10, val_members=4,
)


def load_config(path):
    cfg = dict(DEFAULTS)
    if path:
        from ruamel.yaml import YAML
        with open(path) as f:
            cfg.update(YAML(typ="safe").load(f) or {})
    return cfg


def latent_config(cfg) -> LatentConfig:
    return LatentConfig(k_lat=cfg["k_lat"], k_lon=cfg["k_lon"], latent=cfg["latent"],
                        enc_hidden=tuple(cfg["enc_hidden"]), dec_hidden=tuple(reversed(cfg["enc_hidden"])),
                        pinn_hidden=tuple(cfg["pinn_hidden"]), act=cfg["act"], cfm_width=cfg["cfm_width"],
                        carry_highpass=cfg["carry_highpass"], coupling=cfg["coupling"])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config")
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
    dev = torch.device(cfg["device"])
    log = lambda *x: print(*x, flush=True)  # noqa: E731

    stats = cfg["stats_dir"]
    mean_p, std_p = os.path.join(stats, "global_mean.npy"), os.path.join(stats, "global_std.npy")
    dcfg = kai_data_config(mean_p, std_p)
    pair_ds = PGCFMSequenceDataset(dcfg, cfg["train_dir"], horizon=1)
    valid_ds = PGCFMSequenceDataset(dcfg, cfg["valid_dir"], horizon=1, train=False)
    loader = torch.utils.data.DataLoader(pair_ds, batch_size=cfg["batch"], shuffle=True, drop_last=True,
                                         num_workers=cfg["workers"], pin_memory=dev.type == "cuda",
                                         persistent_workers=cfg["workers"] > 0)
    n_lat, n_lon = pair_ds.img_shape_x, pair_ds.img_shape_y
    base_cfg = dict(BASE_DEFAULTS)  # only the physics guide (b_j, residual settings) is used
    base = build(base_cfg, n_lat, n_lon, np.load(mean_p), np.load(std_p)).to(dev)
    vb = [{k: v.to(dev) for k, v in b.items()} for b in take(torch.utils.data.DataLoader(
        valid_ds, batch_size=cfg["batch"], num_workers=2), cfg["val_batches"])]
    model = LatentPGCFM(latent_config(cfg), n_lat, n_lon).to(dev)
    log_path = os.path.join(a.out, "log.jsonl")

    def write(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def save(path, done):
        torch.save({"latent": model.state_dict(), "latent_config": dataclasses.asdict(model.cfg),
                    "config": base_cfg, "model": base.state_dict(), "scales": vars(base.guide.scales),
                    "train_cfg": cfg, "stages_done": done}, path)

    def report(tag):
        f = floors(model, vb)
        log(f"[{tag}] floors " + " ".join(f"{k} {v:.4g}" for k, v in f.items()))
        rec = {"stage": tag, "floors": f}
        if tag != "stage1":
            v = validate_one_step(PGCFM(model, base.guide), vb, cfg["val_members"], device=dev)
            log(f"[{tag}] val " + " ".join(f"{k} {x:.4g}" for k, x in v.items()))
            rec["val"] = v
        write(rec)

    s1_ck, lat_ck = os.path.join(a.out, "stage1.pt"), os.path.join(a.out, "latent.pt")
    done = []
    if a.resume and os.path.exists(lat_ck):  # latest checkpoint: skip every stage it already holds
        ck = torch.load(lat_ck, map_location=dev)
        model.load_state_dict(ck["latent"])
        base.guide.scales = ResidualScales(**ck["scales"])
        freeze(model.pinn)
        done = ck.get("stages_done", ["stage1", "stage2"])
        log("resumed", ", ".join(done))
    elif a.resume and os.path.exists(s1_ck):
        ck = torch.load(s1_ck, map_location=dev)
        model.load_state_dict(ck["latent"], strict=False)  # stage 1 may come from the other coupling
        base.guide.scales = ResidualScales(**ck["scales"])
        freeze(model.pinn)
        log("resumed stage 1")
    elif "stage1" in cfg["stages"]:
        base.guide.scales = ResidualScales(**cfg["scales"]) if cfg.get("scales") else estimate_scales(
            base.guide, [{k: v.to(dev) for k, v in b.items()} for b in take(loader, cfg["scale_batches"])],
            floor_frac=cfg["scale_floor_frac"])
        log("residual scales b_j:", vars(base.guide.scales))
        fit = take(loader, cfg["dct_fit_batches"])
        model.fit_dct((b["x0"] for b in fit), (b["static"] for b in fit))
        freeze(base.guide)
        hist = train_stage1(model, base.guide, loader, cfg["stage1_iters"], lr=cfg["stage1_lr"],
                            lambda_p=cfg["lambda_p"], grad_ratio=cfg["grad_ratio"],
                            log_every=min(500, cfg["stage1_iters"]), log=log, device=dev)
        write({"stage": "stage1", "last": hist[-1]})
        model.fit_residual_scale(take(loader, cfg["scale_batches"]))
        report("stage1")
        save(s1_ck, ["stage1"])
    freeze(base.guide)

    run2 = "stage2" in cfg["stages"] and "stage2" not in done
    if run2 and cfg["coupling"] == "serial":
        train_serial(model, base.guide, loader, cfg["serial_iters"], lr=cfg["serial_lr"], lambda_p=cfg["lambda_p"],
                     grad_ratio=cfg["grad_ratio"], members=cfg["serial_members"], steps=cfg["serial_steps"],
                     log_every=min(500, cfg["serial_iters"]), log=log, device=dev)
        report("stage2")
        save(os.path.join(a.out, "latent.pt"), ["stage1", "stage2"])
    elif run2:
        train_stage2(model, loader, cfg["stage2_iters"], mode=cfg["stage2_mode"], lr=cfg["stage2_lr"],
                     ae_lr=cfg["ae_lr"], log_every=min(500, cfg["stage2_iters"]), log=log, device=dev)
        report("stage2")
        save(os.path.join(a.out, "latent.pt"), ["stage1", "stage2"])
    if "decft" in cfg["stages"] and "decft" not in done:
        finetune_decoder(model, loader, cfg["decft_iters"], lr=cfg["decft_lr"], log=log, device=dev)
        report("decft")
        save(os.path.join(a.out, "latent.pt"), ["stage1", "stage2", "decft"])
    log("done")


if __name__ == "__main__":
    main()
