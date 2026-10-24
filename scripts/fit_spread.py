"""Step 8 on real data: fit the ensemble calibration of a step 7 serial checkpoint on ERA5
validation years and write ``calib.json`` for ``infer_pgcfm.py --spread``.

The validation initial states are split in two: the first half fits every map (scalar inflation,
DCT band inflation, per-lead post-hoc scale; tangent-space inflation reuses the scalar), the second
half scores them all against the uncalibrated serial chain (CRPS, SSR, range coverage, hydrostatic
residual per lead). The best map on the second half is written unless ``--mode`` picks one.
It also prints the H2 check (variance before/after Q, hydrostatic tangent vs normal part).

    python scripts/fit_spread.py --ckpt runs/latent/latent.pt --valid-dir data/valid \
        --stats-dir data/train/stats --out runs/latent/calib.json [--n-ics 32 --members 8]
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm.cfm import rollout  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset, kai_data_config  # noqa: E402
from pgcfm.latent import LatentConfig, LatentPGCFM  # noqa: E402
from pgcfm.physics.primitive import ResidualScales  # noqa: E402
from pgcfm.projection import freeze  # noqa: E402
from pgcfm.spread import (KEEP, DCTBands, EnsembleCalibrated, contraction_report, ensemble_scores,  # noqa: E402
                          fit_bands, fit_posthoc, fit_scalar, posthoc_scale)
from scripts.run_step4_pgcfm import member_residuals  # noqa: E402
from scripts.train_pgcfm import build  # noqa: E402

ALPHAS = (1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0)


def load_serial(ckpt, H, W, mean, std, dev):
    state = torch.load(ckpt, map_location=dev)
    if "latent_config" not in state:
        raise ValueError(f"{ckpt} is not a train_latent.py checkpoint")
    lc = {k: tuple(v) if isinstance(v, list) else v for k, v in state["latent_config"].items()}
    if lc.get("coupling") != "serial":
        raise ValueError("spread calibration expects the serial coupling (CFM -> Q -> decoder)")
    base = build(state["config"], H, W, mean, std).to(dev)
    base.load_state_dict(state["model"])
    base.guide.scales = ResidualScales(**state["scales"])
    m = LatentPGCFM(LatentConfig(**lc), H, W).to(dev)
    m.load_state_dict(state["latent"])
    freeze(m)
    freeze(base.guide)
    return m.eval(), base.guide


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--valid-dir", required=True)
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-ics", type=int, default=32, help="validation initial states (half fit, half score)")
    p.add_argument("--ic-stride", type=int, default=7)
    p.add_argument("--leads", type=int, default=10)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--chunk", type=int, default=4, help="initial states per rollout batch (GPU memory)")
    p.add_argument("--n-bands", type=int, default=3)
    p.add_argument("--mode", choices=["inflate", "band", "posthoc", "tangent_inflate"],
                   help="write this map instead of the best one on the scoring half")
    p.add_argument("--no-tangent", action="store_true", help="skip tangent_inflate (12 vjps per step)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    dev = torch.device(a.device)

    mean_p, std_p = os.path.join(a.stats_dir, "global_mean.npy"), os.path.join(a.stats_dir, "global_std.npy")
    ds = PGCFMSequenceDataset(kai_data_config(mean_p, std_p), a.valid_dir, horizon=a.leads, train=False)
    H, W = ds.img_shape_x, ds.img_shape_y
    m, guide = load_serial(a.ckpt, H, W, np.load(mean_p), np.load(std_p), dev)
    bands = DCTBands(H, W, m.cfg.k_lat, m.cfg.k_lon, a.n_bands)
    idx = [i * a.ic_stride for i in range(a.n_ics) if i * a.ic_stride < len(ds)]
    half = len(idx) // 2
    if half < 1:
        raise ValueError(f"{len(ds)} validation sequences: too few for --n-ics {a.n_ics} / --ic-stride {a.ic_stride}")

    def batches(ids):
        for i in range(0, len(ids), a.chunk):
            b = torch.utils.data.default_collate([ds[j] for j in ids[i:i + a.chunk]])
            yield {k: v.to(dev) for k, v in b.items()}

    fit_ids, score_ids = idx[:half], idx[half:]

    def roll(tr, ids, seed=1):
        tr.layout(a.members, "interleave")
        out = []
        with torch.no_grad():
            for b in batches(ids):
                out.append((rollout(tr, b["x0"], b["static"], b["doy"], a.members, steps=a.steps, seed=seed), b))
        return out

    def mean_crps(tr):
        e = [ensemble_scores(ens.transpose(0, 1)[..., KEEP, :, :], b["targets"][..., KEEP, :, :])["lead_crps"]
             for ens, b in roll(tr, fit_ids)]
        return float(np.mean(e))

    def wrap(mode, **kw):
        return EnsembleCalibrated(m, mode, guide=guide, bands=bands, **kw)

    fits = {}
    alpha, grid = fit_scalar(lambda al: mean_crps(wrap("inflate", alpha=al)), ALPHAS)
    fits["inflate"] = {"mode": "inflate", "alpha": alpha, "grid": grid}
    print("[fit] inflate", json.dumps({f"{k:g}": round(v, 4) for k, v in grid.items()}), flush=True)
    ba, trace = fit_bands(lambda al: mean_crps(wrap("band", band_alpha=al)), alpha, bands.n)
    fits["band"] = {"mode": "band", "band_alpha": ba, "n_bands": bands.n}
    print("[fit] band", [round(v, 3) for v in ba], flush=True)
    fit_ens = roll(wrap("serial"), fit_ids)
    ens_all = torch.cat([e for e, _ in fit_ens])[..., KEEP, :, :]
    obs_all = torch.cat([b["targets"] for _, b in fit_ens])[..., KEEP, :, :]
    fits["posthoc"] = {"mode": "posthoc", "alpha_per_lead": fit_posthoc(ens_all, obs_all)}
    print("[fit] posthoc", [round(v, 2) for v in fits["posthoc"]["alpha_per_lead"]], flush=True)
    if not a.no_tangent:
        fits["tangent_inflate"] = {"mode": "tangent_inflate", "alpha": alpha}

    gm = SimpleNamespace(guide=guide)

    def evaluate(tr, ens_map=None):
        parts = []
        for ens, b in roll(tr, score_ids):
            ens = ens_map(ens) if ens_map is not None else ens
            r = ensemble_scores(ens.transpose(0, 1)[..., KEEP, :, :], b["targets"][..., KEEP, :, :])
            with torch.no_grad():
                r["hyd"] = member_residuals(gm, b["x0"], ens, b["static"], b["doy"])["hyd"]
            parts.append(r)
        keys = ("lead_crps", "lead_ssr", "lead_coverage", "hyd")
        return {k: np.mean([q[k] for q in parts], axis=0).tolist() for k in keys}

    table = {"serial": evaluate(wrap("serial"))}
    for name, f in fits.items():
        if name == "posthoc":
            table[name] = evaluate(wrap("serial"), lambda e: posthoc_scale(e, f["alpha_per_lead"]))
        else:
            table[name] = evaluate(wrap(f["mode"], alpha=f.get("alpha", 1.0), band_alpha=f.get("band_alpha")))
    leads = sorted({0, min(4, a.leads - 1), a.leads - 1})  # day 1 / 5 / last
    for name, r in table.items():
        print(f"[score] {name:16s} CRPS mean {np.mean(r['lead_crps']):.4f} | days {[i + 1 for i in leads]} CRPS "
              + " / ".join(f"{r['lead_crps'][i]:.3f}" for i in leads) + " SSR "
              + " / ".join(f"{r['lead_ssr'][i]:.2f}" for i in leads) + " hyd "
              + " / ".join(f"{r['hyd'][i]:.2f}" for i in leads), flush=True)

    one = take_one_step(ds, a, dev)
    h2 = contraction_report(m, guide, one, a.members, a.steps)
    print("[H2]", json.dumps({k: round(v, 5) for k, v in h2.items()}), flush=True)

    choice = a.mode or min(fits, key=lambda n: np.mean(table[n]["lead_crps"]))
    calib = dict(fits[choice], fitted_on=a.valid_dir, n_fit_ics=len(fit_ids), members=a.members,
                 scores=table, contraction=h2, candidates=fits)
    with open(a.out, "w") as f:
        json.dump(calib, f, indent=1)
    print(f"wrote {a.out}: mode {choice}", flush=True)
    return calib


def take_one_step(ds, a, dev, n=2):
    """A few one-step validation pairs for the H2 check (targets[:, :1])."""
    ids = [i * a.ic_stride + 3 for i in range(n * a.chunk) if i * a.ic_stride + 3 < len(ds)]
    out = []
    for i in range(0, len(ids), a.chunk):
        b = torch.utils.data.default_collate([ds[j] for j in ids[i:i + a.chunk]])
        b = {k: v.to(dev) for k, v in b.items()}
        b["targets"] = b["targets"][:, :1]
        out.append(b)
    return out


if __name__ == "__main__":
    main()
