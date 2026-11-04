"""Step 13, stage 1: pretrain the PINN of the sequence PG-CFM (pgcfm/seqdct.py) on 6 h KAI sequences.

    data    ERA5 5.625 deg 32x64, the 66 KAI channels every 6 h (scripts/wb2_to_kai.py --all-hours)
    sample  IC x(t) at any 6 h time, increments d_k = x(t + 6k h) - x(t), k = 1..K
    input   d~ = d + alpha * eps, eps ~ Gaussian with the per-lead, per-coefficient std of d (spatial
            DCT), alpha ~ U(0, alpha_max) per sample: stand-ins for the stage-2 FM members
    loss    project: |P(d~) - d~|^2 + lambda_P x primitive-equation loss of P(d~)
            denoise: |P(d~) - d|^2  + lambda_P x the same
            (|.|^2 latitude weighted, in units of the 6 h tendency variance per channel; the physics
            loss is guide.loss with b_j = the RMS of ERA5's own 6 h residual, on ``--pairs`` random
            consecutive pairs per sample)

Gates (validation year, docs/step13_sequence.md):
    identity   P(d) stays d on the truth, and the truth residual is unchanged
    physics    the residual of P(d~) falls from that of d~ toward ERA5's level
    spread     spread of P over M perturbations of one truth / spread of the perturbations,
               in the low (corrected) and the other DCT modes: about 1 = no contraction

    python scripts/run_step13_pinn.py --kai /home/claude/era5/kai6h --era5 /home/claude/era5/era5_data \
        --out runs/step13 --objective project [--quick]

Long runs (docs/step15_training_system.md): ``--device cuda [--amp]``, ``--cache DIR`` (memory-mapped
years instead of RAM), ``--val-every N`` (loss on fixed validation windows, the best state is kept and
selected, ``--patience`` stops early), ``--max-hours`` (saves and exits 75; the same command resumes),
``<tag>.pt_metrics.csv`` with the curves. The random streams do not depend on these options, so a run
with the defaults reproduces the earlier scripts step for step.
"""

import argparse
import dataclasses
import json
import math
import os
import sys

import netCDF4 as nc
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm import contract  # noqa: E402
from pgcfm.balance import BalanceProjector, PositiveMass, ageostrophic  # noqa: E402
from pgcfm.core import static_features  # noqa: E402
from pgcfm.coupling import PhysicsConfig, PhysicsGuide, estimate_scales  # noqa: E402
from pgcfm.runs import (EXIT_BUDGET, FitConfig, Prefetch, autocast, device_from, fit, load_years,  # noqa: E402,F401
                        run_record, tendency_std)
from pgcfm.seqdct import (PINNConfig, PINNProjector, SeqDCT, band_spread, residual_rms,  # noqa: E402
                          sequence_residual_loss)
from scripts.run_step11_climode import channel_stats  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402

DYN, OROG = contract.DYNAMIC_CHANNELS, contract.STATIC_CHANNELS


def pick(data, years, K, rng, n):
    """n random window starts: (years, indices)."""
    ys = rng.choice(years, n)
    return ys, [int(rng.integers(0, len(data[y]) - K)) for y in ys]


def gather(data, ys, idx, K):
    """Windows -> x0 (n, 66, H, W), d (n, K, 66, H, W), orog (n, 1, H, W), hour0 (n,)."""
    seq = torch.stack([data[y][i:i + K + 1] for y, i in zip(ys, idx)])
    x0 = seq[:, 0, DYN]
    return x0, seq[:, 1:, DYN] - x0[:, None], seq[:, 0, OROG], torch.tensor([6.0 * i for i in idx])


def sample(data, years, K, rng, n):
    """n random sequences (pick + gather)."""
    return gather(data, *pick(data, years, K, rng, n), K)


def wmse(a, b, w, t6):
    """Latitude-weighted MSE in units of the 6 h tendency variance per channel."""
    return (((a - b) / t6) ** 2 * w).mean()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--kai", required=True, help="folder of 6-hourly KAI files <year>.nc")
    p.add_argument("--era5", required=True, help="era5_data folder (constants for the static features)")
    p.add_argument("--out", default="runs/step13")
    p.add_argument("--objective", default="project", choices=["project", "denoise"])
    p.add_argument("--train-years", type=int, nargs="+", default=[2012, 2013, 2014, 2015])
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--K", type=int, default=24)
    p.add_argument("--k-lat", type=int, default=8)
    p.add_argument("--k-lon", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--pairs", type=int, default=4, help="consecutive pairs per sample in the physics loss")
    p.add_argument("--lambda-p", type=float, default=0.05)
    p.add_argument("--alpha-max", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--n-val", type=int, default=32)
    p.add_argument("--val-chunk", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--projector", default="pinn", choices=["pinn", "balance"],
                   help="balance: step 14 H-bal, the full-mode norm-projected FNO Pi_theta of pgcfm/balance.py")
    p.add_argument("--bound", type=float, default=0.8, help="balance: per-block operator norm bound B (0 = none)")
    p.add_argument("--fno-width", type=int, default=32)
    p.add_argument("--fno-modes", type=int, nargs=2, default=[32, 33], help="all modes of the doubled 64 x 64 grid")
    p.add_argument("--fno-layers", type=int, default=4)
    p.add_argument("--no-positive", action="store_true", help="balance: without the mass-conserving q >= 0 layer")
    p.add_argument("--lambda-geo", type=float, default=0.0,
                   help="weight of the ageostrophic wind (in units of ERA5's own) inside the physics loss")
    p.add_argument("--json-suffix", default="", help="appended to the json name (gates rerun on a checkpoint)")
    p.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N or auto")
    p.add_argument("--amp", action="store_true", help="bf16 autocast on cuda (not validated against fp32 here)")
    p.add_argument("--cache", default=None, help="folder of memory-mapped normalized years (built once)")
    p.add_argument("--prefetch", type=int, default=2, help="batches gathered ahead in a thread (0 = none)")
    p.add_argument("--val-every", type=int, default=0, help="training loss on fixed validation windows every N "
                   "steps; the best state is kept (<tag>.pt.best) and selected at the end (0 = off)")
    p.add_argument("--n-val-train", type=int, default=8, help="validation windows of --val-every")
    p.add_argument("--patience", type=int, default=0, help="validations without improvement before stopping")
    p.add_argument("--select", default="best", choices=["best", "last"])
    p.add_argument("--max-hours", type=float, default=0.0, help="wall-clock budget; saves and exits 75")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.train_years, a.iters, a.K, a.hidden, a.n_val = a.train_years[:1], 3, 4, 8, 4
        a.val_chunk, a.fno_width, a.fno_layers, a.n_val_train = 2, 8, 2, 2
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = device_from(a.device)
    os.makedirs(a.out, exist_ok=True)
    tag = f"pinn_{a.objective}" if a.projector == "pinn" else \
        f"pibal_{a.objective}_b{a.bound:g}_g{a.lambda_geo:g}" + ("_nopos" if a.no_positive else "")
    tag += "_quick" if a.quick else ""
    log = lambda s: print(s, flush=True)  # noqa: E731

    # -- data, normalization, physics guide at a 6 h step ------------------------------------
    stats = os.path.join(a.out, "stats")
    channel_stats(a.kai, a.train_years, stats)
    mean, std = np.load(os.path.join(stats, "global_mean.npy")), np.load(os.path.join(stats, "global_std.npy"))
    data = load_years(a.kai, a.train_years + [a.val_year], mean, std, cache=a.cache)
    with nc.Dataset(os.path.join(a.kai, f"{a.val_year}.nc")) as d:
        lat_kai = np.asarray(d["latitude"][:])
    contract.register_latitudes(lat_kai)  # 32 rows without poles
    lat, lon, oro, lsm = load_static(a.era5)
    w = torch.from_numpy(np.cos(np.deg2rad(lat)).astype(np.float32))
    w = (w / w.mean()).view(1, 1, 1, -1, 1).to(dev)
    t6 = tendency_std(data, a.train_years, DYN).clamp_min(1e-6).view(1, 1, -1, 1, 1).to(dev)
    guide = PhysicsGuide(32, 64, mean, std, PhysicsConfig(dt_seconds=6 * 3600.0))
    est = []
    for _ in range(10):
        x0, d, orog, _ = sample(data, a.train_years, 1, rng, 16)
        est.append({"x0": x0, "targets": (x0 + d[:, 0])[:, None], "static": orog, "doy": torch.zeros(16, 1)})
    guide.scales = estimate_scales(guide, est)
    for q in guide.parameters():
        q.requires_grad_(False)
    guide.to(dev)
    pert = SeqDCT(a.K, contract.N_DYNAMIC, 32, 64, spatial=True)
    pert.fit(sample(data, a.train_years, a.K, rng, 16)[1] for _ in range(16))
    pert.to(dev)
    cfg = PINNConfig(K=a.K, k_lat=a.k_lat, k_lon=a.k_lon, hidden=a.hidden)
    if a.projector == "pinn":
        P = PINNProjector(cfg, static_features(lat, lon, oro, lsm))
    else:
        P = BalanceProjector(static_features(lat, lon, oro, lsm), mean[DYN], std[DYN], lat, width=a.fno_width,
                             modes=a.fno_modes, n_layers=a.fno_layers, bound=a.bound or None,
                             positive=not a.no_positive)
    P.to(dev)
    pm = PositiveMass(mean[DYN], std[DYN], lat).to(dev)
    with torch.no_grad():  # ERA5's own ageostrophic wind, the unit of the R_geo loss
        x0, d, orog, _ = (q.to(dev) for q in sample(data, a.train_years, 1, rng, 16))
        b_geo = float(ageostrophic(guide, x0 + d[:, 0], orog)[0].pow(2).mean().sqrt())
    res = {"args": vars(a), "cfg": dataclasses.asdict(cfg), "scales": vars(guide.scales), "b_geo": b_geo,
           "n_params": sum(q.numel() for q in P.parameters()), "run": run_record(dev)}
    log(f"[stage 1 {a.objective} {a.projector}] {res['n_params']} parameters, b_j {res['scales']}, "
        f"ERA5 ageostrophic wind {b_geo:.2f} m/s, {res['run']['device_name']} commit {res['run']['commit']}")

    # -- training ---------------------------------------------------------------------------------
    ck = os.path.join(a.out, f"{tag}.pt")
    project = a.objective == "project"
    if os.path.exists(ck):
        P.load_state_dict(torch.load(ck, map_location=dev))
        res["resumed"] = True
    else:
        opt = torch.optim.Adam(P.parameters(), lr=a.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
        g = torch.Generator().manual_seed(a.seed + 1)  # alpha and the perturbations (CPU: device-independent)

        def picks(r):  # every numpy draw of a step, in the order of the original loop
            ys, idx = pick(data, a.train_years, a.K, r, a.batch)
            return ys, idx, r.choice(a.K, a.pairs, replace=False)

        def gather_(p):
            ys, idx, pairs = p
            return (*gather(data, ys, idx, a.K), torch.from_numpy(pairs))

        def loss_of(x0, d, orog, hour0, pairs, gen):
            x0, d, orog, hour0 = (q.to(dev) for q in (x0, d, orog, hour0))
            alpha = (torch.rand(len(x0), generator=gen).view(-1, 1, 1, 1, 1) * a.alpha_max).to(dev)
            dt_ = d + alpha * pert.noise(d.shape, gen) * P.keep
            out = P(x0, dt_, hour0)
            data_term = wmse(out, dt_ if project else d, w, t6)
            phys, _ = sequence_residual_loss(guide, x0, out, d, orog, pairs)
            if a.lambda_geo > 0:  # the frames after each pair's step
                f = (x0[:, None] + out[:, pairs]).flatten(0, 1)
                ageo, _ = ageostrophic(guide, f, orog.repeat_interleave(len(pairs), 0))
                phys = phys + a.lambda_geo * ((ageo / b_geo) ** 2).mean().to(phys.dtype)
            return data_term + a.lambda_p * phys, {"data": data_term.item(), "phys": phys.item()}

        # fixed validation windows, perturbations and pairs: comparable across validations
        vr = np.random.default_rng(a.seed + 99)
        vw = [gather_(picks(vr)) for _ in range(0, a.n_val_train, a.batch)] if a.val_every else []

        def val_fn():
            gv = torch.Generator().manual_seed(a.seed + 98)
            parts = [loss_of(*b, gv) for b in vw]
            return {"loss": float(np.mean([q[0].item() for q in parts])),
                    "data": float(np.mean([q[1]["data"] for q in parts])),
                    "phys": float(np.mean([q[1]["phys"] for q in parts]))}

        r = fit(P, P.parameters(), opt, sched, Prefetch(picks, gather_, rng, a.prefetch),
                lambda b: loss_of(*b, g),
                FitConfig(iters=a.iters, ckpt_every=a.ckpt_every, log_every=max(1, a.iters // 20),
                          val_every=a.val_every, patience=a.patience, select=a.select, max_hours=a.max_hours),
                ck, val_fn=val_fn if a.val_every else None, generators={"g": g},
                after_step=P.project_ if a.projector == "balance" and a.bound else None,
                amp=autocast(dev, a.amp), log=log)
        if r["stopped"] == "budget":
            sys.exit(EXIT_BUDGET)
        torch.save(P.state_dict(), ck)
        res.update(train_seconds=r["seconds"], hist=r["hist"][::10], val_curve=r["val"], best_it=r["best_it"],
                   best_val=r["best_val"], stopped=r["stopped"])
    P.eval()

    # -- gates on the validation year, ``--val-chunk`` sequences at a time (66 channels x K frames) ------
    for y in a.train_years:
        del data[y]  # the gates need the validation year only
    vr = np.random.default_rng(123)
    g = torch.Generator().manual_seed(123)
    pairs, alphas, M, n_sp = torch.arange(a.K), (0.25, 0.5, 1.0), 4, min(8, a.n_val)
    ms = {}  # mean squares over the validation sequences, accumulated chunk by chunk

    def add(key, v, wgt):
        ms[key] = ms.get(key, 0.0) + float(v) * wgt

    def rms_all(x0, f, d, orog):
        """residual_rms plus the ageostrophic wind of the frames 1..K over ERA5's (key geo)."""
        r = residual_rms(guide, x0, f, d, orog, pairs)
        fr = (x0[:, None] + f).flatten(0, 1)
        r["geo"] = float(ageostrophic(guide, fr, orog.repeat_interleave(f.shape[1], 0))[0].pow(2).mean().sqrt())
        return r

    def moisture(x0, f):
        """negative q / tcwv fraction, the global moisture mass a clamp would add (relative), the mass (N, 13)."""
        fr = (x0[:, None] + f).flatten(0, 1)
        cl = fr.clone()
        cl[:, pm.idx] = torch.maximum(fr[:, pm.idx], -pm.m / pm.s)
        m, mc = pm.mass(fr), pm.mass(cl)
        return pm.negative_fraction(fr), float(((mc - m) / m.abs().clamp_min(1e-12)).mean()), m

    with torch.no_grad():
        X0, D, OR, H0 = sample(data, [a.val_year], a.K, vr, a.n_val)
        for i in range(0, a.n_val, a.val_chunk):
            j = slice(i, min(a.n_val, i + a.val_chunk))
            x0, d, orog, hour0 = (q[j].to(dev) for q in (X0, D, OR, H0))
            wgt = (j.stop - j.start) / a.n_val
            pd = P(x0, d, hour0)
            add(("identity_change",), wmse(pd, d, w, t6), wgt)
            for name, f in (("base", d), ("identity", pd)):
                for k, v in rms_all(x0, f, d, orog).items():
                    add((name, k), v ** 2, wgt)
            for alpha in alphas:
                dt_ = d + alpha * pert.noise(d.shape, g) * P.keep
                out = P(x0, dt_, hour0)
                masses = {}
                for name, f in (("in", dt_), ("out", out)):
                    for k, v in rms_all(x0, f, d, orog).items():
                        add((alpha, name, k), v ** 2, wgt)
                    neg, clamp_add, masses[name] = moisture(x0, f)
                    add((alpha, "neg_" + name), neg, wgt)
                    add((alpha, "clampmass_" + name), clamp_add, wgt)
                rel = (masses["out"] - masses["in"]) / masses["in"].abs().clamp_min(1e-12)
                add((alpha, "mass_change"), float(rel.abs().mean()), wgt)
                add((alpha, "err_in"), wmse(dt_, d, w, t6), wgt)
                add((alpha, "err_out"), wmse(out, d, w, t6), wgt)
                add((alpha, "change"), wmse(out, dt_, w, t6), wgt)
                if i < n_sp:  # spread: M perturbations of each of the first n_sp truths
                    s = slice(0, min(n_sp - i, j.stop - j.start))
                    e_in = torch.stack([d[s] + alpha * pert.noise(d[s].shape, g) * P.keep for _ in range(M)])
                    e_out = torch.stack([P(x0[s], e, hour0[s]) for e in e_in])
                    for name, e in (("spread_in", e_in), ("spread_out", e_out)):
                        for b, v in band_spread(e, a.k_lat, a.k_lon).items():
                            add((alpha, name, b), v ** 2, s.stop / n_sp)
            log(f"  gates {j.stop}/{a.n_val}")
    rms = {k: math.sqrt(v) for k, v in ms.items()}
    sub = lambda *q: {k[-1]: v for k, v in rms.items() if k[:-1] == q}  # noqa: E731
    base = sub("base")
    val = {"identity": {"change_over_t6": rms[("identity_change",)],
                        "residual_ratio": {k: v / base[k] for k, v in sub("identity").items()}},
           "truth_residual": base}
    for alpha in alphas:
        s_in, s_out = sub(alpha, "spread_in"), sub(alpha, "spread_out")
        val[f"alpha_{alpha}"] = v = {
            "residual_ratio_in": {k: x / base[k] for k, x in sub(alpha, "in").items()},
            "residual_ratio_out": {k: x / base[k] for k, x in sub(alpha, "out").items()},
            "error_ratio": rms[(alpha, "err_out")] / rms[(alpha, "err_in")],
            "change_over_error": rms[(alpha, "change")] / rms[(alpha, "err_in")],
            "spread_ratio": {b: s_out[b] / s_in[b] for b in s_in},
            "negative_fraction_in": ms[(alpha, "neg_in")], "negative_fraction_out": ms[(alpha, "neg_out")],
            "clamp_mass_added_in": ms[(alpha, "clampmass_in")], "clamp_mass_added_out": ms[(alpha, "clampmass_out")],
            "mass_change_by_P": ms[(alpha, "mass_change")]}
        log(f"[val alpha {alpha}] residual in/out " + " ".join(
            f"{k} {v['residual_ratio_in'][k]:.2f}/{v['residual_ratio_out'][k]:.2f}" for k in base)
            + f" | error ratio {v['error_ratio']:.3f} change/err {v['change_over_error']:.3f}"
            + f" | spread low {v['spread_ratio']['low']:.3f} high {v['spread_ratio']['high']:.3f}"
            + f" | q<0 {v['negative_fraction_in']:.3f}->{v['negative_fraction_out']:.3f}"
            + f" mass change {v['mass_change_by_P']:.2e}")
    log(f"[val identity] change {val['identity']['change_over_t6']:.4f} x t6, residual ratio " +
        " ".join(f"{k} {v:.3f}" for k, v in val["identity"]["residual_ratio"].items()))
    if a.projector == "balance":
        val["lipschitz_bound_Pi"] = P.lipschitz_bound()
        val["clip_frac"] = {"first100": float(np.mean(P.net.clip_log[:100])) if P.net.clip_log else None,
                            "last100": float(np.mean(P.net.clip_log[-100:])) if P.net.clip_log else None}
        log(f"[val balance] a priori Lip(Pi) <= {val['lipschitz_bound_Pi']:.3f}, clipped {val['clip_frac']}")
    res["val"] = val
    with open(os.path.join(a.out, f"{tag}{a.json_suffix}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
