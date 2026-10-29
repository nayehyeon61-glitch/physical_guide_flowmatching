"""Step 12: train and score the deterministic grid cores of pgcfm/core.py on ClimODE's data and protocol.

    data    ERA5 5.625 deg 32x64, z500 t850 t2m u10 v10 every 6 h (the WB1-style files of
            scripts/wb2_to_kai.py, the same files ClimODE is scored on in step 11)
    split   train 2006-2015, validation 2016, test 2017 + 2018
    ICs     ClimODE's: 6 h index 2 + 8 e of each test year (12 UTC every 2nd day), 144 h
    scores  scripts/run_step11_climode.climode_scores (= the ClimODE authors' RMSE / ACC, checked in step 11)

One run = one mode (``--mode jump24 | jump6 | ode | transport``); every mode gets the same network,
batch and iterations. Step 14: ``--net fno [--bound B]`` swaps the ResNet for the norm-bounded FNO of
pgcfm/fno_core.py (files core_<mode>_fno[_b<B>].*). Writes <out>/core_<mode>.pt, core_<mode>.json (training log, validation and test
scores per 24 h lead) and core_<mode>_pred.npz (de-normalized states at 24 h multiples, both test years).

    python scripts/run_step12_core.py --data /home/claude/era5 --out runs/step12 --mode ode [--quick]
"""

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm.core import Core, CoreConfig, area_weights, rollout_loss, static_features  # noqa: E402
from scripts.run_step11_climode import VARS, climode_minmax, climode_scores, climode_truth  # noqa: E402

LEADS_H = (24, 48, 72, 96, 120, 144)


def load_static(era5_data):
    import xarray as xr
    with xr.open_dataset(os.path.join(era5_data, "constants", "constants_5.625deg.nc")) as d:
        oro, lsm = d["orography"].values.astype(np.float32), d["lsm"].values.astype(np.float32)
    with xr.open_dataset(os.path.join(era5_data, "2m_temperature", "2m_temperature_2017_5.625deg.nc")) as d:
        lat, lon = d["lat"].values, d["lon"].values
    return lat, lon, oro, lsm


def windows(n_time, need_after):
    """Valid IC indices of one year: 2 frames of history, ``need_after`` 6 h steps of targets."""
    return np.arange(2, n_time - need_after)


def ic_frames(x, idx):
    """(B, 3, 5, H, W) frames IC-12 h, IC-6 h, IC from one year's (T, 5, H, W) tensor."""
    return torch.stack([x[i - 2:i + 1] for i in idx])


@torch.no_grad()
def forecast(core, x, ics, n6, batch=30):
    """(n_ic, 6, 5, H, W) normalized states at 24 h multiples from ICs ``ics`` of a year tensor."""
    out = []
    for b in range(0, len(ics), batch):
        idx = ics[b:b + batch]
        hours = torch.tensor([6.0 * i for i in idx])
        if core.cfg.mode == "jump24":
            p = core.rollout(ic_frames(x, idx), hours, n6 // 4)
        else:
            p = core.rollout(ic_frames(x, idx), hours, n6)[:, 3::4]
        out.append(p)
    return torch.cat(out)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="output folder of scripts/wb2_to_kai.py (era5_data inside)")
    p.add_argument("--out", default="runs/step12")
    p.add_argument("--mode", default="ode", choices=["jump24", "jump6", "ode", "transport"])
    p.add_argument("--train-years", type=int, nargs="+", default=list(range(2006, 2016)))
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--window", type=int, default=4, help="6 h outputs per training rollout (jump24: 1 day)")
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-ics", type=int, default=None, help="per test year (ClimODE: 180)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--net", default="resnet", choices=["resnet", "fno"], help="fno: step 14 (pgcfm/fno_core.py)")
    p.add_argument("--bound", type=float, default=None, help="fno: operator-norm bound B per block")
    p.add_argument("--fno-width", type=int, default=32)
    p.add_argument("--fno-modes", type=int, default=12)
    p.add_argument("--init", default=None, help="step 14: start from a trained core (.pt of the same config)")
    p.add_argument("--crps-members", type=int, default=0,
                   help="step 14 H-dyn: train on the fair CRPS of this many perturbed rollouts per IC (0 = MSE)")
    p.add_argument("--eps", type=float, default=0.5, help="perturbation size (x 6 h tendency std), as run_step14_hdyn")
    p.add_argument("--corr-cells", type=float, default=1.5)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.train_years, a.iters, a.max_ics, a.hidden = a.train_years[:1], 3, 2, 8
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    tag = a.mode + ("" if a.net == "resnet" else "_fno" + ("" if a.bound is None else f"_b{a.bound:g}")) \
        + ("_crps" if a.crps_members else "") + ("_quick" if a.quick else "")
    era5 = os.path.join(a.data, "era5_data")

    # -- data: physical -> z-score per variable over the training years -------------------------
    raw = {y: climode_truth(era5, y) for y in a.train_years + [a.val_year] + a.test_years}
    tr = np.concatenate([raw[y] for y in a.train_years])
    mean, std = tr.mean((0, 2, 3)), tr.std((0, 2, 3))
    norm = {y: torch.from_numpy((v - mean[:, None, None]) / std[:, None, None]) for y, v in raw.items()}
    tend = {h: torch.from_numpy(np.sqrt(np.mean([((norm[y][h // 6:] - norm[y][:-(h // 6)]) ** 2).mean((0, 2, 3)).numpy()
                                                  for y in a.train_years], 0)).astype(np.float32)) for h in (6, 24)}
    lat, lon, oro, lsm = load_static(era5)
    cfg = CoreConfig(mode=a.mode, hidden=a.hidden, net=a.net, bound=a.bound, fno_width=a.fno_width,
                     fno_modes=a.fno_modes)
    core = Core(cfg, static_features(lat, lon, oro, lsm), tend)
    w = area_weights(lat)
    bounded = a.net == "fno" and a.bound is not None
    res = {"args": vars(a), "cfg": dataclasses.asdict(cfg), "n_params": sum(q.numel() for q in core.parameters()),
           "tend_std": {h: v.tolist() for h, v in tend.items()}}
    print(f"[core {a.mode}] {res['n_params']} parameters", flush=True)

    # -- training: random ICs, rollout loss on every output of the window --------------------------
    ck = os.path.join(a.out, f"core_{tag}.pt")
    need = 4 if a.mode == "jump24" else a.window
    if os.path.exists(ck):
        core.load_state_dict(torch.load(ck))
        res["resumed"] = True
    else:
        if a.init:
            core.load_state_dict(torch.load(a.init))
            res["init"] = a.init
        if a.crps_members:
            from scripts.run_step14_hdyn import smooth_field
            gen = torch.Generator().manual_seed(a.seed + 7)
            eps = a.eps * core.s6.view(-1)
        opt = torch.optim.Adam(core.parameters(), lr=a.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
        hist, t0, start = [], time.time(), 0
        part = os.path.join(a.out, f"core_{tag}_train.pt")  # the container can be reclaimed mid-run
        if os.path.exists(part):
            s = torch.load(part, weights_only=False)
            core.load_state_dict(s["core"]), opt.load_state_dict(s["opt"]), sched.load_state_dict(s["sched"])
            hist, start, t0 = s["hist"], s["it"], time.time() - s["seconds"]
            rng.bit_generator.state = s["rng"]
            print(f"  resumed at it {start}", flush=True)
        core.train()
        for it in range(start, a.iters):
            if it > start and it % 100 == 0:
                torch.save({"core": core.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "hist": hist, "it": it, "seconds": time.time() - t0, "rng": rng.bit_generator.state}, part)
            ys = rng.choice(a.train_years, a.batch)
            idx = [int(rng.choice(windows(len(norm[y]), need))) for y in ys]
            frames = torch.stack([norm[y][i - 2:i + 1] for y, i in zip(ys, idx)])
            if a.mode == "jump24":
                tg = torch.stack([norm[y][i + 4:i + 5] for y, i in zip(ys, idx)])
            else:
                tg = torch.stack([norm[y][i + 1:i + 1 + need] for y, i in zip(ys, idx)])
            hours = torch.tensor([6.0 * i for i in idx])
            if a.crps_members:  # the same perturbations as the H-dyn ensembles, fair CRPS over the members
                M = a.crps_members
                pert = smooth_field((len(idx) * M,) + frames.shape[2:], a.corr_cells, gen) * eps.view(1, -1, 1, 1)
                fm = frames.repeat_interleave(M, 0) + pert[:, None]
                pred = core.rollout(fm, hours.repeat_interleave(M), tg.shape[1]).view(len(idx), M, *tg.shape[1:])
                skill = (pred - tg[:, None]).abs().mean(1)
                spread = (pred[:, :, None] - pred[:, None]).abs().sum((1, 2)) / (2 * M * (M - 1))
                loss = ((skill - spread) * w).mean()
            else:
                loss = rollout_loss(core, frames, hours, tg, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            opt.step()
            if bounded:  # step 14: the hard operator-norm projection after every step
                core.net.project_()
            sched.step()
            hist.append(loss.item())
            if (it + 1) % max(1, a.iters // 20) == 0:
                print(f"  it {it + 1:5d} loss {np.mean(hist[-50:]):.4f} {time.time() - t0:.0f}s", flush=True)
        torch.save(core.state_dict(), ck)
        if os.path.exists(part):
            os.remove(part)
        res.update(train_seconds=time.time() - t0, loss=hist[::10])
        if bounded:  # fraction of the norms the projection reduced, first and last 100 steps of this run
            cl = core.net.clip_log[1:]
            res["clip_frac"] = {"first100": float(np.mean(cl[:100])), "last100": float(np.mean(cl[-100:]))}
    if a.net == "fno":
        res["lipschitz_bound_D"] = core.net.lipschitz_bound()
        print(f"[core {tag}] a priori Lip(D) <= {res['lipschitz_bound_D']:.3g}"
              + (f", projection active on {res['clip_frac']['last100']:.1%} of the norms (last 100 steps)"
                 if "clip_frac" in res else ""), flush=True)
    core.eval()

    # -- validation (2016) and test (2017, 2018): ClimODE scores at 24 h multiples ----------------
    mx, mn = climode_minmax(era5)
    den = lambda t: t.numpy() * std[:, None, None] + mean[:, None, None]  # noqa: E731
    pred_out = {}
    for split, years in (("val", [a.val_year]), ("test", a.test_years)):
        sc = {k: [] for k in ("rmse", "acc", "rmse_pers", "acc_pers")}
        t0 = time.time()
        for y in years:
            x, truth = norm[y], raw[y]
            n_ic = (len(x) - 2 - 24) // 8 + 1
            n_ic = min(n_ic, a.max_ics or n_ic, 30 if split == "val" else n_ic)
            ics = [2 + 8 * e for e in range(n_ic)]
            P = den(forecast(core, x, ics, 24))  # (n_ic, 6, 5, H, W)
            O = np.stack([truth[[i + h // 6 for h in LEADS_H]] for i in ics])
            X0 = np.stack([np.repeat(truth[i:i + 1], len(LEADS_H), 0) for i in ics])
            clim = truth.mean(0)
            s = climode_scores(P, O, np.full_like(P, 1e-6), clim, lat, mx, mn)
            q = climode_scores(X0, O, np.full_like(P, 1e-6), clim, lat, mx, mn)
            sc["rmse"].append(s["rmse"]), sc["acc"].append(s["acc"])
            sc["rmse_pers"].append(q["rmse"]), sc["acc_pers"].append(q["acc"])
            if split == "test":
                pred_out[y] = P.astype(np.float32)
        res[split] = {k: {str(h): {v: float(np.concatenate(sc[k])[:, j, i].mean()) for i, v in enumerate(VARS)}
                          for j, h in enumerate(LEADS_H)} for k in sc}
        res[split]["n_ic"] = int(sum(len(r) for r in sc["rmse"]))
        res[f"{split}_seconds"] = time.time() - t0
        r = res[split]["rmse"]
        print(f"[{split}] z500 RMSE " + " ".join(f"{h}h {r[str(h)]['z']:.0f}" for h in LEADS_H)
              + f" | persistence 24h {res[split]['rmse_pers']['24']['z']:.0f}"
              + f" | t850 24h {r['24']['t']:.2f} t2m 24h {r['24']['t2m']:.2f}", flush=True)
    np.savez_compressed(os.path.join(a.out, f"core_{tag}_pred.npz"), lat=lat, lon=lon,
                        leads=np.array(LEADS_H), **{str(y): v for y, v in pred_out.items()})
    with open(os.path.join(a.out, f"core_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
