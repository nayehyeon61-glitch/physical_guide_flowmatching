"""Step 16: a tendency network integrated by an ODE solver, trained on 6 h rollouts and on the STRONG-form
primitive-equation residual with its own tendency (docs/step16_dynpinn.md, pgcfm/dyncore.py).

    data    6 h KAI sequences (scripts/wb2_to_kai.py --all-hours), the stage-1 PINN's windows
    model   dx/dt = f_theta(x, static, time); x_{k+1} = Heun(f_theta, x_k, 6 h); tisr from the data;
            PositiveMass after every step
    loss    rollout:  |x_k^theta - x_k|^2 (lat weighted, 6 h tendency units) over k = 1..K_cur, with the
                      curriculum K_cur = 1 -> 2 -> 4 -> 8 over the iterations
            tangent:  ``--tangent`` w x |f_theta(x_k) - xdot_k|^2 on the truth frames, xdot_k the centred 12 h
              difference: the data's own tangent vector (owner's "tangent-plane data", 2026-10-27)
    physics:  lambda_p x sum_j gamma_j max(m_j - target, 0), m_j = weighted mean (R_j / b_j)^2 of the
                      strong residuals at ``--phys-states`` random rollout states (the IC included) with the
                      network's tendency; b_j = the RMS of the same residual on ERA5 with centred 12 h
                      differences; target 1 = ERA5's level (not the rest state)

Gates (validation year, K = ``--val-K`` steps from ``--n-val`` ICs):
    skill        z500 / t850 / t2m RMSE by lead against persistence (physical units)
    energy       kinetic energy of the rollout / of the truth by lead (step 3 damping: < 0.95 fails)
    amplitude    RMS 6 h increment of the rollout / of the truth by lead
    residual     strong residual per equation at the rollout states (over ERA5's level), the bridge residual
                 of step 13 (residual_rms) rollout / truth, and the rest state f = 0 over ERA5's level
    projection   members d~ = d + alpha eps: the core's one-step map x~ -> x~ + Delta_core(x~) on the
                 perturbed states (spread ratio out / in in the low and the other DCT modes; step 8
                 contraction: < 0.95 fails) and the blend Pi(d~)_k = (d~_k + x~_{k-1} + Delta(x~_{k-1}) - x0) / 2

    python scripts/run_step16_dynpinn.py --kai /home/claude/era5/kai6h --era5 /home/claude/era5/era5_data \
        --out runs/step16 --lambda-p 0.05 [--quick]
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
from pgcfm.cfm import TISR_DYN  # noqa: E402
from pgcfm.core import static_features  # noqa: E402
from pgcfm.coupling import PhysicsConfig, PhysicsGuide, estimate_scales  # noqa: E402
from pgcfm.dyncore import (DynCore, DynCoreConfig, StrongPhysics, estimate_strong_scales,  # noqa: E402
                           finite_difference_tendency, kinetic_energy)
from pgcfm.runs import (EXIT_BUDGET, FitConfig, Prefetch, autocast, device_from, fit, load_years,  # noqa: E402
                        run_record, tendency_std)
from pgcfm.seqdct import SeqDCT, band_spread, residual_rms  # noqa: E402
from scripts.run_step11_climode import channel_stats  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402
from scripts.run_step13_pinn import gather, pick, sample, wmse  # noqa: E402

DYN = contract.DYNAMIC_CHANNELS
SCORED = {"z500": "z500", "t850": "t850", "t2m": "t2m"}
GAMMA = {"u": 1.0, "v": 1.0, "T": 1.0, "mass": 1.0, "hyd": 1.0, "q": 0.0}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--kai", required=True)
    p.add_argument("--era5", required=True)
    p.add_argument("--out", default="runs/step16")
    p.add_argument("--tag", default=None, help="checkpoint / json name (default dyn_lp<lambda>)")
    p.add_argument("--train-years", type=int, nargs="+", default=[2012, 2013, 2014, 2015])
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--K", type=int, default=8, help="training window (6 h steps)")
    p.add_argument("--curriculum", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="rollout length per equal fraction of the iterations")
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda-p", type=float, default=0.05)
    p.add_argument("--phys-target", type=float, default=1.0, help="hinge target in ERA5 residual units (0: eq. 10)")
    p.add_argument("--phys-states", type=int, default=2, help="rollout states per sample in the physics loss")
    p.add_argument("--tangent", type=float, default=0.0, help="weight of the tangent-vector loss |f_theta(x_k) - "
                   "xdot_k|^2 on the truth states, xdot_k = centred 12 h difference (the data's own tangent)")
    p.add_argument("--dt-hours", type=float, default=6.0)
    p.add_argument("--solver", default="heun", choices=["euler", "heun"])
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--blocks", type=int, nargs="+", default=[2, 2])
    p.add_argument("--no-positive", action="store_true")
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--val-K", type=int, default=24)
    p.add_argument("--val-chunk", type=int, default=4)
    p.add_argument("--proj-K", type=int, default=8, help="projection gate: steps")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-suffix", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cache", default=None)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--val-every", type=int, default=0)
    p.add_argument("--n-val-train", type=int, default=8)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--select", default="best", choices=["best", "last"])
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.train_years, a.iters, a.K, a.hidden, a.blocks = a.train_years[:1], 4, 2, 8, [1, 1]
        a.curriculum, a.n_val, a.val_K, a.val_chunk, a.proj_K, a.n_val_train = [1, 2], 4, 4, 2, 2, 2
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = device_from(a.device)
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or f"dyn_lp{a.lambda_p:g}" + (f"_tan{a.tangent:g}" if a.tangent else "") \
        + ("_nopos" if a.no_positive else "") + ("_quick" if a.quick else "")
    log = lambda s: print(s, flush=True)  # noqa: E731

    # -- data, normalization, physics at a 6 h step --------------------------------------------
    stats = os.path.join(a.out, "stats")
    channel_stats(a.kai, a.train_years, stats)
    mean, std = np.load(os.path.join(stats, "global_mean.npy")), np.load(os.path.join(stats, "global_std.npy"))
    data = load_years(a.kai, a.train_years + [a.val_year], mean, std, cache=a.cache)
    with nc.Dataset(os.path.join(a.kai, f"{a.val_year}.nc")) as d:
        lat_kai = np.asarray(d["latitude"][:])
    contract.register_latitudes(lat_kai)
    lat, lon, oro, lsm = load_static(a.era5)
    w_lat = torch.from_numpy(np.cos(np.deg2rad(lat)).astype(np.float32))
    w_lat = (w_lat / w_lat.mean()).to(dev)
    w = w_lat.view(1, 1, 1, -1, 1)
    t6v = tendency_std(data, a.train_years, DYN).clamp_min(1e-6).to(dev)
    t6 = t6v.view(1, 1, -1, 1, 1)
    guide = PhysicsGuide(32, 64, mean, std, PhysicsConfig(dt_seconds=6 * 3600.0))
    est = []
    for _ in range(10):
        x0, d, orog, _ = sample(data, a.train_years, 1, rng, 16)
        est.append({"x0": x0, "targets": (x0 + d[:, 0])[:, None], "static": orog, "doy": torch.zeros(16, 1)})
    guide.scales = estimate_scales(guide, est)  # the step 13 bridge residual, for the comparable diagnostic
    for q in guide.parameters():
        q.requires_grad_(False)
    guide.to(dev)
    phys = StrongPhysics(guide, t6v)
    triples = []
    for _ in range(4):
        x0, d, orog, _ = (q.to(dev) for q in sample(data, a.train_years, 2, rng, 8))
        triples.append((x0, x0 + d[:, 0], x0 + d[:, 1], orog))
    phys.scales = estimate_strong_scales(phys, triples)
    with torch.no_grad():  # the rest state f = 0 over ERA5's level: how much of b_j is the tendency at all
        rest = {}
        for x0, x1, _, orog in triples:
            for k, v in phys.per_equation(x1, orog, torch.zeros_like(x1)).items():
                rest[k] = rest.get(k, 0.0) + v ** 2 / len(triples)
        rest = {k: math.sqrt(v) for k, v in rest.items()}
        tend = {}  # the centred-difference tendency term alone over b_j: the signal the physics loss can see
        for x0, x1, x2, orog in triples:
            R1, m = phys.residuals(x1, orog, finite_difference_tendency(x0, x2))
            R0, _ = phys.residuals(x1, orog, torch.zeros_like(x1))
            for k, v in phys.mean_squares({q: R1[q] - R0[q] for q in ("u", "v", "T")}, m).items():
                tend[k] = tend.get(k, 0.0) + float(v) / len(triples)
        tend = {k: math.sqrt(v) for k, v in tend.items()}
    pert = SeqDCT(a.K, contract.N_DYNAMIC, 32, 64, spatial=True)
    pert.fit(sample(data, a.train_years, a.K, rng, 16)[1] for _ in range(16))
    pert.to(dev)
    cfg = DynCoreConfig(hidden=a.hidden, blocks=tuple(a.blocks), dt_hours=a.dt_hours, solver=a.solver,
                        positive=not a.no_positive)
    model = DynCore(cfg, static_features(lat, lon, oro, lsm), t6v.cpu(), mean[DYN], std[DYN], lat).to(dev)
    res = {"args": vars(a), "cfg": dataclasses.asdict(cfg), "strong_scales": vars(phys.scales),
           "bridge_scales": vars(guide.scales), "rest_over_era5": rest, "tendency_over_era5": tend,
           "n_params": sum(q.numel() for q in model.parameters()), "run": run_record(dev)}
    log(f"[step 16 {tag}] {res['n_params']} parameters, strong b_j {res['strong_scales']}, rest state / ERA5 "
        + " ".join(f"{k} {v:.3f}" for k, v in rest.items()) + ", tendency term / ERA5 "
        + " ".join(f"{k} {v:.3f}" for k, v in tend.items()) + f", {res['run']['device_name']}")

    def tisr_of(x0, d, K):
        return torch.cat([x0[:, None, TISR_DYN], x0[:, None, TISR_DYN] + d[:, :K, TISR_DYN]], 1)[:, 1:K + 1]

    # -- training ---------------------------------------------------------------------------------
    ck = os.path.join(a.out, f"{tag}.pt")
    if os.path.exists(ck):
        model.load_state_dict(torch.load(ck, map_location=dev))
        res["resumed"] = True
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
        g = torch.Generator().manual_seed(a.seed + 1)  # which rollout states enter the physics loss
        step = {"it": 0}
        if os.path.exists(ck + ".train"):
            step["it"] = torch.load(ck + ".train", weights_only=False)["it"]

        def cur_K():
            return a.curriculum[min(len(a.curriculum) - 1, step["it"] * len(a.curriculum) // a.iters)]

        def after_step():
            step["it"] += 1

        def picks(r):
            return pick(data, a.train_years, a.K, r, a.batch)

        def gather_(p):
            return gather(data, *p, a.K)

        def loss_of(x0, d, orog, hour0, gen, K=None):
            x0, d, orog, hour0 = (q.to(dev) for q in (x0, d, orog, hour0))
            K = K or cur_K()
            y = model.rollout(x0, hour0, K, tisr_of(x0, d, K))
            data_term = wmse(y - x0[:, None], d[:, :K], w, t6)
            parts = {"data": data_term.item(), "K": float(K)}
            loss = data_term
            if a.tangent > 0:  # the truth frames 1..K with both neighbours in the window
                ks = list(range(1, min(K, a.K - 1) + 1))
                xt = torch.stack([x0 + d[:, k - 1] for k in ks], 1).flatten(0, 1)
                xd = torch.stack([finite_difference_tendency(x0 if k == 1 else x0 + d[:, k - 2], x0 + d[:, k])
                                  for k in ks], 1)
                hrs = torch.stack([hour0 + 6.0 * k for k in ks], 1).flatten()
                f = model.tendency(xt, hrs).view(len(x0), len(ks), *xt.shape[1:])
                tan = wmse(f, xd, w, t6)
                loss = loss + a.tangent * tan
                parts["tangent"] = tan.item()
            if a.lambda_p <= 0:
                return loss, parts
            ks = torch.randint(0, K + 1, (a.phys_states,), generator=gen).tolist()
            ms = None
            for k in ks:
                xs = x0 if k == 0 else y[:, k - 1]
                xo = x0 if k == 0 else x0 + d[:, k - 1]
                f = model.tendency(xs, hour0 + 6.0 * k)
                R, _ = phys.residuals(xs, orog, f)
                m = phys.mean_squares(R, guide.mask(xo, xo, orog))
                ms = m if ms is None else {q: ms[q] + m[q] for q in m}
            ms = {q: v / len(ks) for q, v in ms.items()}
            phys_term = phys.loss(ms, GAMMA, a.phys_target).to(data_term.dtype)
            parts.update(phys=phys_term.item(), **{f"m_{q}": v.item() for q, v in ms.items()})
            return loss + a.lambda_p * phys_term, parts

        vr = np.random.default_rng(a.seed + 99)
        vw = [gather_(picks(vr)) for _ in range(0, a.n_val_train, a.batch)] if a.val_every else []

        def val_fn():
            gv = torch.Generator().manual_seed(a.seed + 98)
            parts = [loss_of(*b, gv, K=a.K) for b in vw]
            return {"loss": float(np.mean([q[0].item() for q in parts])),
                    "data": float(np.mean([q[1]["data"] for q in parts]))}

        r = fit(model, model.parameters(), opt, sched, Prefetch(picks, gather_, rng, a.prefetch),
                lambda b: loss_of(*b, g),
                FitConfig(iters=a.iters, ckpt_every=a.ckpt_every, log_every=max(1, a.iters // 20),
                          val_every=a.val_every, patience=a.patience, select=a.select, max_hours=a.max_hours),
                ck, val_fn=val_fn if a.val_every else None, generators={"g": g}, after_step=after_step,
                amp=autocast(dev, a.amp), log=log)
        if r["stopped"] == "budget":
            sys.exit(EXIT_BUDGET)
        torch.save(model.state_dict(), ck)
        res.update(train_seconds=r["seconds"], hist=r["hist"][::10], val_curve=r["val"], best_it=r["best_it"],
                   best_val=r["best_val"], stopped=r["stopped"])
    model.eval()

    # -- gates on the validation year ------------------------------------------------------------------
    for y in a.train_years:
        del data[y]
    vr = np.random.default_rng(123)
    gg = torch.Generator().manual_seed(123)
    sd, mn = torch.from_numpy(std[DYN]).float().to(dev), torch.from_numpy(mean[DYN]).float().to(dev)
    idx = {v: DYN.index(contract.CHANNEL_INDEX[c]) for v, c in SCORED.items()}
    wk = w_lat.view(-1, 1)
    Kv, acc = a.val_K, {}

    def add(key, v, wgt):
        v = v.detach().cpu().double().numpy() if torch.is_tensor(v) else np.asarray(v, dtype=np.float64)
        acc[key] = acc.get(key, 0.0) + v * wgt

    def rmse(y, t, i):  # (B, K, H, W) physical, lat weighted -> (K,) mean square
        return (((y[:, :, i] - t[:, :, i]) * sd[i]) ** 2 * wk).mean((0, -1, -2))

    pairs = torch.arange(min(a.proj_K, Kv))
    with torch.no_grad():
        X0, D, OR, H0 = sample(data, [a.val_year], Kv, vr, a.n_val)
        for i in range(0, a.n_val, a.val_chunk):
            j = slice(i, min(a.n_val, i + a.val_chunk))
            x0, d, orog, hour0 = (q[j].to(dev) for q in (X0, D, OR, H0))
            wgt = (j.stop - j.start) / a.n_val
            y = model.rollout(x0, hour0, Kv, tisr_of(x0, d, Kv))
            t = x0[:, None] + d
            inc = y - x0[:, None]
            for v, ii in idx.items():
                add(("rmse", v), rmse(y, t, ii), wgt)
                add(("persist", v), rmse(x0[:, None].expand_as(t), t, ii), wgt)
            ke_y, ke_t = kinetic_energy(y, sd, mn, wk), kinetic_energy(t, sd, mn, wk)
            add(("ke_y",), ke_y.mean(0), wgt), add(("ke_t",), ke_t.mean(0), wgt)
            st_y = torch.cat([x0[:, None], y], 1)
            st_t = torch.cat([x0[:, None], t], 1)
            add(("amp_y",), (((st_y[:, 1:] - st_y[:, :-1]) / t6) ** 2 * w).mean((0, 2, 3, 4)), wgt)
            add(("amp_t",), (((st_t[:, 1:] - st_t[:, :-1]) / t6) ** 2 * w).mean((0, 2, 3, 4)), wgt)
            # strong residual at the rollout states with the model's tendency, by lead (every 4th state)
            for k in range(0, Kv, 4):
                f = model.tendency(y[:, k], hour0 + 6.0 * (k + 1))
                for q, v in phys.per_equation(y[:, k], orog, f, x_obs=t[:, k]).items():
                    add(("strong", k + 1, q), v ** 2, wgt)
                ft = finite_difference_tendency(st_t[:, k], st_t[:, k + 2]) if k + 2 <= Kv else None
                if ft is not None:
                    for q, v in phys.per_equation(t[:, k], orog, ft).items():
                        add(("strong_truth", k + 1, q), v ** 2, wgt)
            for name, f_ in (("bridge_y", inc), ("bridge_t", d)):
                for q, v in residual_rms(guide, x0, f_, d, orog, pairs).items():
                    add((name, q), v ** 2, wgt)
            # projection gate on the first proj_K steps: the core's one-step map on perturbed states
            # (spread of x~ + Delta(x~) over the spread of x~: step 8 contraction) and the blend beta = 0.5
            Kp, M = len(pairs), 4
            for alpha in (0.5, 1.0):
                e_in = torch.stack([d[:, :Kp] + alpha * pert.noise(d[:, :a.K].shape, gg)[:, :Kp].to(dev)
                                    * model.keep for _ in range(M)])
                e_step, e_blend = [], []
                for m in range(M):
                    prev, out_s, out_b = x0, [], []
                    for k in range(Kp):
                        xk = x0 + e_in[m, :, k]
                        out_s.append(model.step(xk, hour0 + 6.0 * (k + 1), (x0 + d[:, min(k + 1, Kp - 1)])[:, TISR_DYN]) - x0)
                        core = model.step(prev, hour0 + 6.0 * k, (x0 + d[:, k])[:, TISR_DYN]) - x0
                        dk = 0.5 * e_in[m, :, k] + 0.5 * core
                        out_b.append(dk)
                        prev = x0 + dk
                    e_step.append(torch.stack(out_s, 1))
                    e_blend.append(torch.stack(out_b, 1))
                e_step, e_blend = torch.stack(e_step), torch.stack(e_blend)
                for name, e_o, e_i in (("step", e_step, e_in), ("blend", e_blend, e_in)):
                    s_in, s_out = band_spread(e_i, 8, 16), band_spread(e_o, 8, 16)
                    for b in s_in:
                        add((alpha, name, "spread_in", b), s_in[b] ** 2, wgt)
                        add((alpha, name, "spread_out", b), s_out[b] ** 2, wgt)
                    add((alpha, name, "err_in"), wmse(e_i, d[None, :, :Kp], w, t6), wgt)
                    add((alpha, name, "err_out"), wmse(e_o, d[None, :, :Kp], w, t6), wgt)
            log(f"  gates {j.stop}/{a.n_val}")
    leads = [6 * (k + 1) for k in range(Kv)]
    val = {"leads_h": leads,
           "rmse": {v: np.sqrt(acc[("rmse", v)]).tolist() for v in idx},
           "persistence": {v: np.sqrt(acc[("persist", v)]).tolist() for v in idx},
           "ke_ratio": (acc[("ke_y",)] / acc[("ke_t",)]).tolist(),
           "amp_ratio": np.sqrt(acc[("amp_y",)] / acc[("amp_t",)]).tolist(),
           "strong_ratio": {}, "strong_rollout": {}, "strong_truth": {},
           "bridge_ratio": {}, "projection": {}}
    for k in range(0, Kv, 4):
        eqs = [key[2] for key in acc if len(key) == 3 and key[0] == "strong" and key[1] == k + 1]
        val["strong_rollout"][str(6 * (k + 1))] = {q: math.sqrt(acc[("strong", k + 1, q)]) for q in eqs}
        if ("strong_truth", k + 1, eqs[0]) in acc:
            val["strong_truth"][str(6 * (k + 1))] = {q: math.sqrt(acc[("strong_truth", k + 1, q)]) for q in eqs}
            val["strong_ratio"][str(6 * (k + 1))] = {
                q: math.sqrt(acc[("strong", k + 1, q)] / acc[("strong_truth", k + 1, q)]) for q in eqs}
    for alpha in (0.5, 1.0):
        for name in ("step", "blend"):
            val["projection"][f"alpha_{alpha}_{name}"] = {
                "spread_ratio": {b: math.sqrt(acc[(alpha, name, "spread_out", b)] / acc[(alpha, name, "spread_in", b)])
                                 for b in ("low", "high")},
                "error_ratio": math.sqrt(acc[(alpha, name, "err_out")] / acc[(alpha, name, "err_in")])}
    val["bridge_ratio"] = {key[1]: math.sqrt(acc[key] / acc[("bridge_t", key[1])])
                           for key in list(acc) if len(key) == 2 and key[0] == "bridge_y"}
    i24 = min(Kv, 4) - 1
    val["gates"] = {"ke_ratio_24h": val["ke_ratio"][i24], "amp_ratio_24h": val["amp_ratio"][i24],
                    "ke_ratio_last": val["ke_ratio"][-1],
                    "step_spread_ratio_min": min(v["spread_ratio"][b] for k, v in val["projection"].items()
                                                 if k.endswith("step") for b in v["spread_ratio"]),
                    "beats_persistence_z500": [bool(r < pr) for r, pr in
                                               zip(val["rmse"]["z500"], val["persistence"]["z500"])]}
    res["val"] = val
    log("[val] z500 RMSE " + " ".join(f"{h}:{r:.0f}/{pr:.0f}" for h, r, pr in
                                     zip(leads[3::4], val["rmse"]["z500"][3::4], val["persistence"]["z500"][3::4]))
        + " | t850 24h {:.2f}/{:.2f} t2m 24h {:.2f}/{:.2f}".format(
            val["rmse"]["t850"][i24], val["persistence"]["t850"][i24], val["rmse"]["t2m"][i24], val["persistence"]["t2m"][i24]))
    log("[val] KE ratio " + " ".join(f"{h}:{v:.3f}" for h, v in zip(leads[3::4], val["ke_ratio"][3::4]))
        + " | increment amplitude " + " ".join(f"{h}:{v:.3f}" for h, v in zip(leads[3::4], val["amp_ratio"][3::4])))
    log("[val] strong residual rollout / truth " + " ".join(
        f"{h}: " + " ".join(f"{q} {v:.2f}" for q, v in r.items()) for h, r in val["strong_ratio"].items()))
    log("[val] bridge residual rollout / truth " + " ".join(f"{q} {v:.2f}" for q, v in val["bridge_ratio"].items()))
    for k, v in val["projection"].items():
        log(f"[val projection {k}] spread low {v['spread_ratio']['low']:.3f} high {v['spread_ratio']['high']:.3f}"
            f" error ratio {v['error_ratio']:.3f}")
    with open(os.path.join(a.out, f"{tag}{a.json_suffix}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
