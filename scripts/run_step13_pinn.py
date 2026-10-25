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
"""

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import netCDF4 as nc
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm import contract  # noqa: E402
from pgcfm.core import static_features  # noqa: E402
from pgcfm.coupling import PhysicsConfig, PhysicsGuide, estimate_scales  # noqa: E402
from pgcfm.seqdct import (PINNConfig, PINNProjector, SeqDCT, band_spread, residual_rms,  # noqa: E402
                          sequence_residual_loss)
from scripts.run_step11_climode import channel_stats  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402

DYN, OROG = contract.DYNAMIC_CHANNELS, contract.STATIC_CHANNELS


def load_years(kai, years, mean, std):
    """{year: (T, 67, H, W) float32 normalized (KAI: std + eps)}."""
    out = {}
    for y in years:
        with nc.Dataset(os.path.join(kai, f"{y}.nc")) as d:
            f = np.asarray(d["fields"][:], dtype=np.float32)
        f -= mean[:, None, None]
        f /= (std + contract.KAI_STD_EPS)[:, None, None]
        out[y] = torch.from_numpy(f)
    return out


def sample(data, years, K, rng, n):
    """n random sequences -> x0 (n, 66, H, W), d (n, K, 66, H, W), orog (n, 1, H, W), hour0 (n,)."""
    ys = rng.choice(years, n)
    idx = [int(rng.integers(0, len(data[y]) - K)) for y in ys]
    seq = torch.stack([data[y][i:i + K + 1] for y, i in zip(ys, idx)])
    x0 = seq[:, 0, DYN]
    return x0, seq[:, 1:, DYN] - x0[:, None], seq[:, 0, OROG], torch.tensor([6.0 * i for i in idx])


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
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.train_years, a.iters, a.K, a.hidden, a.n_val = a.train_years[:1], 3, 4, 8, 4
        a.val_chunk = 2
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    tag = f"pinn_{a.objective}" + ("_quick" if a.quick else "")
    log = lambda s: print(s, flush=True)  # noqa: E731

    # -- data, normalization, physics guide at a 6 h step ------------------------------------
    stats = os.path.join(a.out, "stats")
    channel_stats(a.kai, a.train_years, stats)
    mean, std = np.load(os.path.join(stats, "global_mean.npy")), np.load(os.path.join(stats, "global_std.npy"))
    data = load_years(a.kai, a.train_years + [a.val_year], mean, std)
    with nc.Dataset(os.path.join(a.kai, f"{a.val_year}.nc")) as d:
        lat_kai = np.asarray(d["latitude"][:])
    contract.register_latitudes(lat_kai)  # 32 rows without poles
    lat, lon, oro, lsm = load_static(a.era5)
    w = torch.from_numpy(np.cos(np.deg2rad(lat)).astype(np.float32))
    w = (w / w.mean()).view(1, 1, 1, -1, 1)
    t6 = torch.sqrt(torch.stack([((data[y][1:, DYN] - data[y][:-1, DYN]) ** 2).mean((0, 2, 3))
                                 for y in a.train_years]).mean(0)).clamp_min(1e-6).view(1, 1, -1, 1, 1)
    guide = PhysicsGuide(32, 64, mean, std, PhysicsConfig(dt_seconds=6 * 3600.0))
    est = []
    for _ in range(10):
        x0, d, orog, _ = sample(data, a.train_years, 1, rng, 16)
        est.append({"x0": x0, "targets": (x0 + d[:, 0])[:, None], "static": orog, "doy": torch.zeros(16, 1)})
    guide.scales = estimate_scales(guide, est)
    for q in guide.parameters():
        q.requires_grad_(False)
    pert = SeqDCT(a.K, contract.N_DYNAMIC, 32, 64, spatial=True)
    pert.fit(sample(data, a.train_years, a.K, rng, 16)[1] for _ in range(16))
    cfg = PINNConfig(K=a.K, k_lat=a.k_lat, k_lon=a.k_lon, hidden=a.hidden)
    P = PINNProjector(cfg, static_features(lat, lon, oro, lsm))
    res = {"args": vars(a), "cfg": dataclasses.asdict(cfg), "scales": vars(guide.scales),
           "n_params": sum(q.numel() for q in P.parameters())}
    log(f"[stage 1 {a.objective}] {res['n_params']} parameters, b_j {res['scales']}")

    # -- training ---------------------------------------------------------------------------------
    ck = os.path.join(a.out, f"{tag}.pt")
    if os.path.exists(ck):
        P.load_state_dict(torch.load(ck))
        res["resumed"] = True
    else:
        opt = torch.optim.Adam(P.parameters(), lr=a.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
        part, hist, start, t0 = ck + ".train", [], 0, time.time()
        if os.path.exists(part):
            s = torch.load(part, weights_only=False)
            P.load_state_dict(s["P"]), opt.load_state_dict(s["opt"]), sched.load_state_dict(s["sched"])
            hist, start, t0 = s["hist"], s["it"], time.time() - s["seconds"]
            rng.bit_generator.state = s["rng"]
            log(f"  resumed at it {start}")
        g = torch.Generator().manual_seed(a.seed + 1)
        for it in range(start, a.iters):
            if it > start and it % 50 == 0:
                torch.save({"P": P.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "hist": hist,
                            "it": it, "seconds": time.time() - t0, "rng": rng.bit_generator.state}, part + ".tmp")
                os.replace(part + ".tmp", part)
            x0, d, orog, hour0 = sample(data, a.train_years, a.K, rng, a.batch)
            alpha = torch.rand(a.batch, generator=g).view(-1, 1, 1, 1, 1) * a.alpha_max
            dt_ = d + alpha * pert.noise(d.shape, g) * P.keep
            out = P(x0, dt_, hour0)
            data_term = wmse(out, dt_ if a.objective == "project" else d, w, t6)
            pairs = torch.from_numpy(rng.choice(a.K, a.pairs, replace=False))
            phys, _ = sequence_residual_loss(guide, x0, out, d, orog, pairs)
            loss = data_term + a.lambda_p * phys
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(P.parameters(), 1.0)
            opt.step()
            sched.step()
            hist.append({"data": data_term.item(), "phys": phys.item()})
            if (it + 1) % max(1, a.iters // 20) == 0:
                h = hist[-50:]
                log(f"  it {it + 1:5d} data {np.mean([q['data'] for q in h]):.4f} "
                    f"phys {np.mean([q['phys'] for q in h]):.3f} {time.time() - t0:.0f}s")
        torch.save(P.state_dict(), ck)
        if os.path.exists(part):
            os.remove(part)
        res.update(train_seconds=time.time() - t0, hist=hist[::10])
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

    with torch.no_grad():
        X0, D, OR, H0 = sample(data, [a.val_year], a.K, vr, a.n_val)
        for i in range(0, a.n_val, a.val_chunk):
            j = slice(i, min(a.n_val, i + a.val_chunk))
            x0, d, orog, hour0, wgt = X0[j], D[j], OR[j], H0[j], (j.stop - j.start) / a.n_val
            pd = P(x0, d, hour0)
            add(("identity_change",), wmse(pd, d, w, t6), wgt)
            for name, f in (("base", d), ("identity", pd)):
                for k, v in residual_rms(guide, x0, f, d, orog, pairs).items():
                    add((name, k), v ** 2, wgt)
            for alpha in alphas:
                dt_ = d + alpha * pert.noise(d.shape, g) * P.keep
                out = P(x0, dt_, hour0)
                for name, f in (("in", dt_), ("out", out)):
                    for k, v in residual_rms(guide, x0, f, d, orog, pairs).items():
                        add((alpha, name, k), v ** 2, wgt)
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
            "spread_ratio": {b: s_out[b] / s_in[b] for b in s_in}}
        log(f"[val alpha {alpha}] residual in/out " + " ".join(
            f"{k} {v['residual_ratio_in'][k]:.2f}/{v['residual_ratio_out'][k]:.2f}" for k in base)
            + f" | error ratio {v['error_ratio']:.3f} change/err {v['change_over_error']:.3f}"
            + f" | spread low {v['spread_ratio']['low']:.3f} high {v['spread_ratio']['high']:.3f}")
    log(f"[val identity] change {val['identity']['change_over_t6']:.4f} x t6, residual ratio " +
        " ".join(f"{k} {v:.3f}" for k, v in val["identity"]["residual_ratio"].items()))
    res["val"] = val
    with open(os.path.join(a.out, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
