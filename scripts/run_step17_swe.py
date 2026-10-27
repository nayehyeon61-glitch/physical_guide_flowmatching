"""Step 17: does "learning the physics directly" give dynamics that data learning does not? (owner 2026-10-27
12:35, docs/step17_swe_pinn.md). A spherical shallow-water test bed where the equations are complete and
the operators exact, with two settings:

    full     the truth lives at the learners' resolution (T21, 64 x 128): the physics is complete except the
             solver's hyperdiffusion
    coarse   the truth is T42 (128 x 256) and the learners see its T21 truncation: the unresolved modes are
             the missing closure, as for ERA5 at 5.625 deg (docs/step16 sec. 4)

arms (grid learners, the same ClimODE ResNet tendency net f_theta):
    data     rollout MSE, Heun substeps (step 16's data-only core)
    node     rollout MSE, torchdiffeq.odeint (the neural-ODE route, as ClimODE)
    pi       data + lambda_p x strong-form residual |f_theta(x) - F(x)|^2 at the rollout states, F = the
             solver's right-hand side at the learners' resolution (step 16's physics term with an exact operator)
    pi_only  the residual alone on data states (no trajectories: "learn the physics directly")
    raissi   maziarraissi/PINNs forward problem per test window: data at t = 0 on the grid + the SWE residual
             by autograd in (lambda, phi, t) on [0, T]; "fit" variant also sees data at every hour (interpolation)

metrics per lead (1..24 h): area-weighted RMSE of Phi and wind vs the truth (persistence as reference), total
energy ratio, and the tangent accuracy |f_theta(x_truth) - F_true| / b with F_true the exact tendency of the
truncated truth, hyperdiffusion included (b = its RMS); ``operator_bias`` = |F_learn(x_truth) - F_true| / b is
the closure gap the physics term cannot see (the hyperdiffusion in ``full``, plus the truncation in ``coarse``).

    python scripts/run_step17_swe.py --out runs/step17 --setting coarse [--quick]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm.swe import (CHANNELS, HOUR, Normalizer, SWEPinn, TendencyNet, World, rollout_heun,  # noqa: E402
                       rollout_node, sample_sphere_time, strong_residual)

GRID_ARMS = ("data", "node", "pi", "pi_only")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/step17")
    p.add_argument("--setting", default="coarse", choices=["full", "coarse"])
    p.add_argument("--arms", nargs="+", default=list(GRID_ARMS) + ["raissi"])
    p.add_argument("--n-train", type=int, default=12, help="perturbed Galewsky trajectories (2 of them held out)")
    p.add_argument("--n-random", type=int, default=4, help="random-IC trajectories in training")
    p.add_argument("--train-hours", type=int, default=72)
    p.add_argument("--spinup", type=int, default=96, help="hours before a trajectory is recorded (the Galewsky "
                   "jet needs about four days to break)")
    p.add_argument("--test-hours", type=int, default=24)
    p.add_argument("--test-t0", type=int, nargs="+", default=[96, 120, 144, 168], help="Galewsky windows")
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--K", type=int, default=3, help="training rollout (h)")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--blocks", type=int, nargs="+", default=[1, 1])
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lambda-p", type=float, default=1.0)
    p.add_argument("--substeps", type=int, default=1)
    p.add_argument("--node-method", default="rk4")
    p.add_argument("--node-step", type=float, default=1.0, help="hours")
    p.add_argument("--pinn-layers", type=int, nargs="+", default=[4, 64, 64, 64, 64, 4])
    p.add_argument("--pinn-adam", type=int, default=3000)
    p.add_argument("--pinn-lbfgs", type=int, default=300)
    p.add_argument("--pinn-colloc", type=int, default=8192)
    p.add_argument("--pinn-T", type=int, nargs="+", default=[6, 24], help="forward horizons (h)")
    p.add_argument("--pinn-windows", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.n_train, a.n_random, a.train_hours, a.test_hours, a.test_t0, a.spinup = 3, 1, 8, 3, [2], 2
        a.iters, a.pinn_adam, a.pinn_lbfgs, a.pinn_colloc, a.pinn_T, a.pinn_windows = 6, 10, 5, 256, [2], 1
        a.pinn_layers = [4, 16, 16, 4]
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731
    tag = a.setting + ("_quick" if a.quick else "")

    # -- worlds and data --------------------------------------------------------------------------------
    learn = World(64, 128, 22, 300.0) if not a.quick else World(32, 64, 11, 600.0)
    truth = learn if a.setting == "full" else (World(128, 256, 43, 150.0) if not a.quick else World(64, 128, 22, 300.0))
    to_learn = lambda us: learn.grid(truth.truncate(us, learn.lmax)).float()  # noqa: E731
    # the exact tendency of the truncated truth, hyperdiffusion included (what the data do)
    F_of = lambda us: learn.grid(truth.truncate(truth.tendency_spec(us, closure=True), learn.lmax)).float()  # noqa: E731

    def trajectory(us0, hours, spinup=0):
        if spinup:
            us0 = truth.step_hours(us0, spinup)[-1]
        specs = [us0] + truth.step_hours(us0, hours)
        return torch.stack([to_learn(s) for s in specs]), torch.stack([F_of(s) for s in specs])

    t0 = time.time()
    train, val = [], []
    for i in range(a.n_train):
        us0 = truth.galewsky(umax=float(rng.uniform(60, 90)), lon0=float(rng.uniform(0, 2 * math.pi)),
                             bump=float(rng.uniform(0.5, 1.5)))
        (val if i >= a.n_train - 2 else train).append(trajectory(us0, a.train_hours, a.spinup))
    for i in range(a.n_random):
        torch.manual_seed(a.seed * 100 + i)
        train.append(trajectory(truth.random_ic(mach=0.1), a.train_hours, max(24, a.spinup // 4)))
    tests = []
    gal_hours = max(a.test_t0) + a.test_hours
    gx, gF = trajectory(truth.galewsky(), gal_hours)
    for t in a.test_t0:
        tests.append(("galewsky", t, gx[t:t + a.test_hours + 1], gF[t:t + a.test_hours + 1]))
    for i in range(2 if not a.quick else 1):
        torch.manual_seed(a.seed * 100 + 50 + i)
        rx, rF = trajectory(truth.random_ic(mach=0.1), a.test_hours, max(24, a.spinup // 4))
        tests.append(("random", 0, rx, rF))
    frames = torch.cat([x for x, _ in train])
    nz = Normalizer(frames)
    b = torch.cat([F for _, F in train]).double().pow(2).mean((0, 2, 3)).sqrt().float()  # tendency RMS, per second
    log(f"[{tag}] data {time.time() - t0:.0f}s: train {len(train)} x {a.train_hours} h, val {len(val)}, tests "
        f"{len(tests)} x {a.test_hours} h; std {nz.std.view(-1).tolist()} t1 {nz.t1.tolist()} b {b.tolist()}")
    res = {"args": vars(a), "setting": a.setting, "leads_h": list(range(1, a.test_hours + 1)),
           "data": {"std": nz.std.view(-1).tolist(), "mean": nz.mean.view(-1).tolist(), "t1": nz.t1.tolist(),
                    "b": b.tolist(), "wind_rms": float(frames[:, 1:].pow(2).mean().sqrt())},
           "tests": [(k, t) for k, t, _, _ in tests], "arms": {}, "raissi": {}}

    # closure gap: the learners' operator vs the exact tendency of the truncated truth, on truth states
    with torch.no_grad():
        xs = torch.cat([x[::6] for x, _, _, _ in [(x, F, 0, 0) for x, F in val]])[:48]
        Fs = torch.cat([F[::6] for _, F in val])[:48]
        Fl = learn.tendency(xs.double()).float()
        res["operator_bias"] = {k: float(((Fl - Fs)[:, i] / b[i]).pow(2).mean().sqrt()) for i, k in enumerate(CHANNELS)}
        res["tendency_over_t1"] = {k: float((Fs[:, i] * HOUR / nz.std.view(-1)[i] / nz.t1[i]).pow(2).mean().sqrt())
                                   for i, k in enumerate(CHANNELS)}
    log(f"  operator bias (closure gap) {res['operator_bias']}")

    w_lat = learn.w.float().view(1, 1, 1, -1, 1)
    wmse = lambda y, t: ((y - t) ** 2 * w_lat * learn.nlat).mean()  # noqa: E731

    def windows(trajs, n, K, r):
        ii = r.integers(0, len(trajs), n)
        tt = [int(r.integers(0, trajs[i][0].shape[0] - K)) for i in ii]
        x = torch.stack([trajs[i][0][t:t + K + 1] for i, t in zip(ii, tt)])
        F = torch.stack([trajs[i][1][t:t + K + 1] for i, t in zip(ii, tt)])
        return nz.norm(x), F

    def rollout(arm, net, z0, hours):
        if arm == "node":
            return rollout_node(net, z0, hours, method=a.node_method, step=a.node_step)
        return rollout_heun(net, z0, hours, substeps=a.substeps)

    def loss_of(arm, net, z, F):
        if arm == "pi_only":
            zf = z.flatten(0, 1)
            return strong_residual(learn, nz, net, zf, b).mean(), {}
        y = rollout(arm, net, z[:, 0], a.K)
        data = wmse(y, z[:, 1:])
        if arm == "pi":
            phys = strong_residual(learn, nz, net, y.flatten(0, 1), b).mean()
            return data + a.lambda_p * phys, {"data": float(data.detach()), "phys": float(phys.detach())}
        return data, {"data": float(data.detach())}

    vr = np.random.default_rng(a.seed + 99)
    vw = [windows(val, a.batch, a.K, vr) for _ in range(2)]
    tr = np.random.default_rng(a.seed + 1)
    for arm in [q for q in a.arms if q in GRID_ARMS]:
        torch.manual_seed(a.seed)
        net = TendencyNet(learn, nz.t1, a.hidden, tuple(a.blocks))
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
        t1, hist, curve = time.time(), [], []
        for it in range(a.iters):
            z, F = windows(train, a.batch, a.K, tr)
            loss, parts = loss_of(arm, net, z, F)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(), sched.step()
            hist.append(float(loss))
            if (it + 1) % max(1, a.iters // 10) == 0 or it == a.iters - 1:
                net.eval()
                with torch.no_grad():
                    vl = float(np.mean([float(loss_of(arm, net, z_, F_)[0]) for z_, F_ in vw]))
                net.train()
                curve.append((it + 1, float(np.mean(hist[-50:])), vl))
                log(f"  [{arm}] it {it + 1} loss {np.mean(hist[-50:]):.4f} val {vl:.4f} {parts} "
                    f"{time.time() - t1:.0f}s ({(time.time() - t1) / (it + 1):.2f} s/it)")
        net.eval()
        torch.save(net.state_dict(), os.path.join(a.out, f"{tag}_{arm}.pt"))
        r = evaluate(arm, net, tests, learn, nz, b, rollout, a.test_hours)
        r.update(train_seconds=time.time() - t1, curve=curve, n_params=sum(q.numel() for q in net.parameters()))
        res["arms"][arm] = r
        log(f"[{arm}] Phi RMSE " + " ".join(f"{h}:{r['rmse']['Phi'][h - 1]:.0f}" for h in (1, 6, 12, a.test_hours)
                                              if h <= a.test_hours)
            + f" | wind {r['rmse']['wind'][a.test_hours - 1]:.2f} | energy {r['energy_ratio'][-1]:.3f} | tangent "
            + " ".join(f"{k} {v:.2f}" for k, v in r["tangent_residual"].items()))
        with open(os.path.join(a.out, f"{tag}.json"), "w") as f:
            json.dump(res, f, indent=1)

    if "raissi" in a.arms:
        res["raissi"] = raissi(a, tests[:a.pinn_windows] if a.quick else
                               [t for t in tests if t[0] == "random"][:1] + [t for t in tests if t[1] == 48][:1]
                               + tests[:1], learn, b, log)
        with open(os.path.join(a.out, f"{tag}.json"), "w") as f:
            json.dump(res, f, indent=1)
    log(f"[{tag}] done {time.time() - t0:.0f}s -> {os.path.join(a.out, tag + '.json')}")
    return res


def evaluate(arm, net, tests, learn, nz, b, rollout, hours):
    """RMSE per lead (Phi, wind, each channel) with persistence, energy ratio, tangent accuracy on truth states."""
    rm, rp, en, tan, opr = [], [], [], [], []
    with torch.no_grad():
        for kind, t0, x, F in tests:
            y = nz.phys(rollout(arm, net, nz.norm(x[:1]), hours)[0])  # (hours, 3, H, W)
            rm.append(learn.wrmse(y.double(), x[1:].double()))  # (hours, 3)
            rp.append(learn.wrmse(x[:1].double().expand_as(x[1:]), x[1:].double()))
            en.append(learn.energy(y.double()) / learn.energy(x[1:].double()))
            tan.append(strong_residual(learn, nz, net, nz.norm(x), b, F_true=F).sqrt())
            opr.append(strong_residual(learn, nz, net, nz.norm(x), b).sqrt())
    wind = lambda r: (r[:, 1] ** 2 + r[:, 2] ** 2).sqrt()  # noqa: E731
    per_window = [{"kind": kind, "t0": t0, "rmse_Phi": r[:, 0].tolist(), "rmse_wind": wind(r).tolist(),
                   "pers_Phi": q[:, 0].tolist(), "pers_wind": wind(q).tolist(), "energy_ratio": e.tolist()}
                  for (kind, t0, _, _), r, q, e in zip(tests, rm, rp, en)]
    rm, rp, en = torch.stack(rm).mean(0), torch.stack(rp).mean(0), torch.stack(en).mean(0)
    out = {"rmse": {k: rm[:, i].tolist() for i, k in enumerate(CHANNELS)}, "rmse_pers": {},
           "energy_ratio": en.tolist(), "windows": per_window,
           "tangent_residual": {k: float(torch.stack(tan)[:, i].mean()) for i, k in enumerate(CHANNELS)},
           "operator_residual": {k: float(torch.stack(opr)[:, i].mean()) for i, k in enumerate(CHANNELS)}}
    out["rmse"]["wind"], out["rmse_pers"]["wind"] = wind(rm).tolist(), wind(rp).tolist()
    for i, k in enumerate(CHANNELS):
        out["rmse_pers"][k] = rp[:, i].tolist()
    return out


def raissi(a, tests, learn, b, log):
    """The coordinate PINN forward problem on each test window and horizon T (hours): data at t = 0 on the
    grid, physics at collocation points in [0, T]; the "fit" variant also gets data at every hour."""
    lat, lon = torch.meshgrid(learn.lat.float(), learn.lon.float(), indexing="ij")
    keep = (lat.abs() < math.radians(89.0)).reshape(-1)
    lam_g, phi_g = lon.reshape(-1)[keep], lat.reshape(-1)[keep]
    bd = {"u": float(b[1:].max()), "v": float(b[1:].max()), "Phi": float(b[0])}
    out = {}
    for variant in ("fwd", "fit"):
        for T in a.pinn_T:
            Ts = T * HOUR
            rm, rp, en, hist = [], [], [], []
            for kind, t0, x, F in tests:
                torch.manual_seed(a.seed)
                x = x[:T + 1]
                fs = {"u": (0.0, max(float(x[0, 1].std()), 1.0)), "v": (0.0, max(float(x[0, 2].std()), 1.0)),
                      "Phi": (float(x[0, 0].mean()), float(x[0, 0].std()))}
                P = SWEPinn(a.pinn_layers, Ts, fs, bd, float(learn.solver.radius), float(learn.solver.omega))
                hours = [0] if variant == "fwd" else list(range(T + 1))
                g = torch.Generator().manual_seed(a.seed + 3)
                X_u, tg = [], {k: [] for k in CHANNELS}
                for h in hours:
                    sel = torch.ones_like(lam_g, dtype=torch.bool) if h == 0 else \
                        torch.rand(len(lam_g), generator=g) < 1024 / len(lam_g)
                    X_u.append((lam_g[sel], phi_g[sel], torch.full((int(sel.sum()),), h * HOUR)))
                    for i, k in enumerate(CHANNELS):
                        tg[k].append(x[h, i].reshape(-1)[keep][sel])
                X_u = tuple(torch.cat([q[j] for q in X_u]) for j in range(3))
                tg = {k: torch.cat(v) for k, v in tg.items()}
                X_f = sample_sphere_time(a.pinn_colloc, Ts, generator=g)
                t1 = time.time()
                h = P.train_model(X_u, tg, X_f, n_adam=a.pinn_adam, lr=1e-3, n_lbfgs=a.pinn_lbfgs)
                y = P.predict_grid(learn, list(range(1, T + 1))).double()
                rm.append(learn.wrmse(y, x[1:].double()))
                rp.append(learn.wrmse(x[:1].double().expand_as(x[1:]), x[1:].double()))
                en.append(learn.energy(y) / learn.energy(x[1:].double()))
                hist.append({"kind": kind, "t0": t0, "data": h[-1]["data"], "phys": h[-1]["phys"],
                             "first": h[0]["loss"], "last": h[-1]["loss"], "seconds": time.time() - t1,
                             "rmse_Phi": rm[-1][:, 0].tolist(), "pers_Phi": rp[-1][:, 0].tolist(),
                             "rmse_wind": (rm[-1][:, 1] ** 2 + rm[-1][:, 2] ** 2).sqrt().tolist(),
                             "pers_wind": (rp[-1][:, 1] ** 2 + rp[-1][:, 2] ** 2).sqrt().tolist(),
                             "energy_ratio": en[-1].tolist()})
                log(f"  [raissi {variant} T{T}] {kind} t0 {t0}: data {h[-1]['data']:.3g} phys {h[-1]['phys']:.3g} "
                    f"Phi RMSE {float(rm[-1][-1, 0]):.0f} (pers {float(rp[-1][-1, 0]):.0f}) wind "
                    f"{float((rm[-1][-1, 1] ** 2 + rm[-1][-1, 2] ** 2).sqrt()):.2f} energy {float(en[-1][-1]):.3f} "
                    f"{time.time() - t1:.0f}s")
            rm, rp, en = torch.stack(rm).mean(0), torch.stack(rp).mean(0), torch.stack(en).mean(0)
            out[f"{variant}_T{T}"] = {
                "leads_h": list(range(1, T + 1)),
                "rmse": {**{k: rm[:, i].tolist() for i, k in enumerate(CHANNELS)},
                         "wind": (rm[:, 1] ** 2 + rm[:, 2] ** 2).sqrt().tolist()},
                "rmse_pers": {**{k: rp[:, i].tolist() for i, k in enumerate(CHANNELS)},
                              "wind": (rp[:, 1] ** 2 + rp[:, 2] ** 2).sqrt().tolist()},
                "energy_ratio": en.tolist(), "fits": hist}
    return out


if __name__ == "__main__":
    main()
