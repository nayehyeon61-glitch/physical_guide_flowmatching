"""Step 13, stage 2: sequence flow matching (pgcfm/seqdct.SeqFM) + the frozen stage-1 PINN, scored on
ClimODE's protocol.

    sample   history x(t - 12 h), x(t - 6 h), x(t) -> increments d_k = x(t + 6k h) - x(t), k = 1..K
             (K = 24: the whole 144 h in one flow, no autoregressive feedback)
    basis    dct (the owner's design: flow on standardized 3-D DCT coefficients) or grid (control)
    PINN     the frozen stage-1 projector P on every member (``--pinn``), or none
    test     ClimODE's ICs (6 h index 2 + 8 e, 12 UTC every second day of 2017 and 2018), M members,
             ClimODE scores of the ensemble mean (RMSE, ACC, Gaussian CRPS with the member spread) at
             24 h multiples, ensemble CRPS, spread/skill, the 6 h primitive-equation residual of the
             members, and the member spread before / after P in the low and the other DCT modes
    vars     z500, t850, t2m from the KAI channels; u10/v10 from u1000/v1000 by the step 11
             per-cell linear fit on the validation year

    python scripts/run_step13_seqfm.py --kai /home/claude/era5/kai6h --era5 /home/claude/era5/era5_data \
        --out runs/step13 --basis dct --pinn runs/step13/pinn_project.pt [--quick]
"""

import argparse
import dataclasses
import json
import os
import sys
import time

import netCDF4 as nc
import numpy as np
import properscoring as ps
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm import contract  # noqa: E402
from pgcfm.core import static_features  # noqa: E402
from pgcfm.coupling import PhysicsConfig, PhysicsGuide, estimate_scales  # noqa: E402
from pgcfm.seqdct import (PINNConfig, PINNProjector, SeqFM, SeqFMConfig, band_spread,  # noqa: E402
                          residual_rms)
from scripts.run_step11_climode import VARS, climode_minmax, climode_scores, climode_truth  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402
from scripts.run_step13_pinn import load_years  # noqa: E402

DYN, OROG = contract.DYNAMIC_CHANNELS, contract.STATIC_CHANNELS
SRC = {"z": "z500", "t": "t850", "t2m": "t2m", "u10": "u1000", "v10": "v1000"}
DYN_IDX = {v: DYN.index(contract.CHANNEL_INDEX[c]) for v, c in SRC.items()}
LEADS_H = (24, 48, 72, 96, 120, 144)


def window(x, i, K):
    """hist (3, 67, H, W) = x[i-2..i], increments d (K, 66, H, W), orog (1, H, W), hour of x[i]."""
    hist = x[i - 2:i + 1]
    return hist, x[i + 1:i + 1 + K, DYN] - x[i, DYN][None], x[i, OROG], 6.0 * i


def batch(data, picks, K):
    w = [window(data[y], i, K) for y, i in picks]
    return (torch.stack([q[0][:, DYN] for q in w]), torch.stack([q[1] for q in w]),
            torch.stack([q[2] for q in w]), torch.tensor([q[3] for q in w]))


def random_picks(data, years, K, rng, n):
    ys = rng.choice(years, n)
    return [(y, int(rng.integers(2, len(data[y]) - K))) for y in ys]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--kai", required=True)
    p.add_argument("--era5", required=True)
    p.add_argument("--out", default="runs/step13")
    p.add_argument("--basis", default="dct", choices=["dct", "grid"])
    p.add_argument("--pinn", default=None, help="stage-1 checkpoint (pinn_<objective>.pt); none = no PINN")
    p.add_argument("--train-years", type=int, nargs="+", default=[2012, 2013, 2014, 2015])
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--K", type=int, default=24)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--heun", type=int, default=8)
    p.add_argument("--max-ics", type=int, default=None, help="per test year (ClimODE: 180)")
    p.add_argument("--save-ens", action="store_true",
                   help="save the ensembles (5 ClimODE vars, 24 h multiples) of the validation and test years "
                        "at ClimODE-style ICs, for stage 3")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.train_years, a.iters, a.K, a.channels, a.max_ics, a.members, a.heun = \
            a.train_years[:1], 3, 4, 32, 2, 2, 2
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tag = f"seqfm_{a.basis}" + ("_quick" if a.quick else "")
    log = lambda s: print(s, flush=True)  # noqa: E731
    stats = os.path.join(a.out, "stats")  # written by run_step13_pinn.py on the same training years
    mean, std = np.load(os.path.join(stats, "global_mean.npy")), np.load(os.path.join(stats, "global_std.npy"))
    data = load_years(a.kai, a.train_years + [a.val_year], mean, std)
    with nc.Dataset(os.path.join(a.kai, f"{a.val_year}.nc")) as d:
        contract.register_latitudes(np.asarray(d["latitude"][:]))
    lat, lon, oro, lsm = load_static(a.era5)
    static = static_features(lat, lon, oro, lsm)
    fm = SeqFM(SeqFMConfig(K=a.K, basis=a.basis, model_channels=a.channels), static)
    res = {"args": vars(a), "cfg": dataclasses.asdict(fm.cfg), "n_params": sum(q.numel() for q in fm.parameters())}
    log(f"[stage 2 {a.basis}] {res['n_params']} parameters")

    # -- training: the FM's own loss on its coefficients (never through the PINN) --------------------
    ck = os.path.join(a.out, f"{tag}.pt")
    if os.path.exists(ck):
        fm.load_state_dict(torch.load(ck))
    else:
        fm.fit_basis(batch(data, random_picks(data, a.train_years, a.K, rng, 8), a.K)[1] for _ in range(32))
        opt = torch.optim.Adam(fm.unet.parameters(), lr=a.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
        part, hist, start, t0 = ck + ".train", [], 0, time.time()
        if os.path.exists(part):
            s = torch.load(part, weights_only=False)
            fm.load_state_dict(s["fm"]), opt.load_state_dict(s["opt"]), sched.load_state_dict(s["sched"])
            hist, start, t0 = s["hist"], s["it"], time.time() - s["seconds"]
            rng.bit_generator.state = s["rng"]
            log(f"  resumed at it {start}")
        for it in range(start, a.iters):
            if it > start and it % 100 == 0:
                torch.save({"fm": fm.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "hist": hist,
                            "it": it, "seconds": time.time() - t0, "rng": rng.bit_generator.state}, part + ".tmp")
                os.replace(part + ".tmp", part)
            h, d, _, hour0 = batch(data, random_picks(data, a.train_years, a.K, rng, a.batch), a.K)
            loss = fm.loss(h, d, hour0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.unet.parameters(), 1.0)
            opt.step()
            sched.step()
            hist.append(loss.item())
            if (it + 1) % max(1, a.iters // 30) == 0:
                log(f"  it {it + 1:5d} loss {np.mean(hist[-50:]):.4f} {time.time() - t0:.0f}s")
        torch.save(fm.state_dict(), ck)
        if os.path.exists(part):
            os.remove(part)
        res.update(train_seconds=time.time() - t0, loss=hist[::10])
    fm.eval()

    P = None
    if a.pinn:
        pc = json.load(open(a.pinn.replace(".pt", ".json")))["cfg"]
        P = PINNProjector(PINNConfig(**{**pc, "blocks": tuple(pc["blocks"])}), static)
        P.load_state_dict(torch.load(a.pinn))
        P.eval()
        if P.cfg.K != a.K:
            raise ValueError("PINN and FM need the same K")
    guide = PhysicsGuide(32, 64, mean, std, PhysicsConfig(dt_seconds=6 * 3600.0))
    est = []
    for _ in range(10):
        h, d, orog, _ = batch(data, random_picks(data, a.train_years, 1, rng, 16), 1)
        est.append({"x0": h[:, -1], "targets": (h[:, -1] + d[:, 0])[:, None], "static": orog, "doy": torch.zeros(16, 1)})
    guide.scales = estimate_scales(guide, est)

    # u10/v10 from u1000/v1000 on the validation year (step 11)
    era5 = a.era5
    tv = climode_truth(era5, a.val_year)
    xv = data[a.val_year][:, DYN].numpy()
    unscale = lambda q: q * std[DYN][:, None, None] + mean[DYN][:, None, None]  # noqa: E731
    sfc = {}
    for v in ("u10", "v10"):
        x, y = unscale(xv)[:, DYN_IDX[v]], tv[:, VARS.index(v)]
        b = ((x - x.mean(0)) * (y - y.mean(0))).sum(0) / ((x - x.mean(0)) ** 2).sum(0).clip(1e-6)
        sfc[v] = (y.mean(0) - b * x.mean(0), b)
    del tv, xv
    for y in a.train_years:  # free the training years before the test
        del data[y]

    def to_vars(e):
        """(..., 66, H, W) normalized -> (..., 5, H, W) physical ClimODE variables."""
        e = unscale(e)
        out = np.stack([e[..., DYN_IDX[v], :, :] for v in VARS], -3)
        for v in ("u10", "v10"):
            i = VARS.index(v)
            out[..., i, :, :] = sfc[v][0] + sfc[v][1] * out[..., i, :, :]
        return out

    # -- test on ClimODE's ICs --------------------------------------------------------------------
    mx, mn = climode_minmax(era5)
    wl = np.cos(np.deg2rad(lat))
    wl = (wl / wl.mean())[:, None]
    arms = ("raw",) + (("pinn",) if P is not None else ())
    sc = {arm: {k: [] for k in ("rmse", "acc", "crps_gauss", "crps_ens", "spread", "rmse_pers")} for arm in arms}
    phys = {arm: [] for arm in arms + ("truth",)}
    bands = []
    g = torch.Generator().manual_seed(a.seed + 7)
    t0 = time.time()
    for y in ([a.val_year] if a.save_ens else []) + a.test_years:
        score = y in a.test_years
        x = data[y] if y == a.val_year else load_years(a.kai, [y], mean, std)[y]
        truth = climode_truth(era5, y)
        keep = {"E": [], "O": [], "X0": [], "ics": []}
        clim = truth.mean(0)
        n_ic = (len(x) - 2 - a.K) // 8 + 1
        n_ic = min(n_ic, a.max_ics or n_ic)
        for c0 in range(0, n_ic, 4):
            ics = [2 + 8 * e for e in range(c0, min(n_ic, c0 + 4))]
            h, d, orog, hour0 = batch({y: x}, [(y, i) for i in ics], a.K)
            B = len(ics)
            hm = h.repeat_interleave(a.members, 0)
            ens = {"raw": fm.sample(hm, hour0.repeat_interleave(a.members), a.heun, g)}
            if P is not None:
                with torch.no_grad():
                    ens["pinn"] = P(hm[:, -1], ens["raw"], hour0.repeat_interleave(a.members))
            if P is not None and score:
                e_raw = ens["raw"].view(B, a.members, *ens["raw"].shape[1:]).transpose(0, 1)
                e_p = ens["pinn"].view(B, a.members, *ens["pinn"].shape[1:]).transpose(0, 1)
                s_r, s_p = band_spread(e_raw, P.cfg.k_lat, P.cfg.k_lon), band_spread(e_p, P.cfg.k_lat, P.cfg.k_lon)
                bands.append({b: s_p[b] / s_r[b] for b in s_r})
            sel = torch.tensor([k * 4 - 1 for k in range(1, a.K // 4 + 1)])  # 24 h multiples
            pair_ids = torch.tensor([0, a.K // 2, a.K - 1])
            O = np.stack([truth[[i + hh // 6 for hh in LEADS_H[:len(sel)]]] for i in ics])  # (B, L, 5, H, W)
            X0 = np.repeat(truth[ics][:, None], len(sel), 1)
            if score:
                phys["truth"].append(residual_rms(guide, h[:, -1], d, d, orog, pair_ids))
            for arm, inc in ens.items():
                mem = (hm[:, -1][:, None] + inc[:, sel]).view(B, a.members, len(sel), *inc.shape[2:])
                E = to_vars(mem.numpy())  # (B, M, L, 5, H, W)
                if a.save_ens and arm == arms[-1]:
                    keep["E"].append((E - X0[:, None]).astype(np.float16)), keep["O"].append(O)
                    keep["X0"].append(X0[:, 0]), keep["ics"].extend(ics)
                if not score:
                    continue
                phys[arm].append(residual_rms(guide, hm[:, -1], inc, d.repeat_interleave(a.members, 0),
                                              orog.repeat_interleave(a.members, 0), pair_ids))
                mu, sd = E.mean(1), E.std(1, ddof=1)
                s = climode_scores(mu, O, np.maximum(sd, 1e-6), clim, lat, mx, mn)
                q = climode_scores(X0, O, np.full_like(X0, 1e-6), clim, lat, mx, mn)
                scl = (mx - mn)[:, None, None]
                ce = ps.crps_ensemble((O - mn[:, None, None]) / scl, np.moveaxis((E - mn[:, None, None]) / scl, 1, -1))
                sc[arm]["rmse"].append(s["rmse"]), sc[arm]["acc"].append(s["acc"])
                sc[arm]["crps_gauss"].append(s["crps_gauss"]), sc[arm]["crps_ens"].append(ce.mean((-1, -2)))
                sc[arm]["spread"].append(np.sqrt((sd ** 2 * wl).mean((-1, -2))))
                sc[arm]["rmse_pers"].append(q["rmse"])
        if a.save_ens:  # E - X0 in float16 (physical units), O and X0 in float32
            np.savez(os.path.join(a.out, f"{tag}_{arms[-1]}_ens_{y}.npz"), E_minus_x0=np.concatenate(keep["E"]),
                     O=np.concatenate(keep["O"]), X0=np.concatenate(keep["X0"]), ics=np.array(keep["ics"]),
                     leads=np.array(LEADS_H[:len(sel)]))
        log(f"  {'test' if score else 'val'} {y}: {n_ic} ICs {time.time() - t0:.0f}s")
    res["test"] = {}
    for arm in arms:
        r = {k: np.concatenate(v) for k, v in sc[arm].items()}
        res["test"][arm] = {k: {str(h): {v: float(r[k][:, j, i].mean()) for i, v in enumerate(VARS)}
                                for j, h in enumerate(LEADS_H[:r[k].shape[1]])} for k in r}
        res["test"][arm]["ssr"] = {str(h): {v: float(np.sqrt((r["spread"][:, j, i] ** 2).mean() * (a.members + 1) / a.members)
                                                     / np.sqrt((r["rmse"][:, j, i] ** 2).mean()))
                                            for i, v in enumerate(VARS)} for j, h in enumerate(LEADS_H[:r["rmse"].shape[1]])}
        res["test"][arm]["residual"] = {k: float(np.mean([q[k] for q in phys[arm]])) for k in phys[arm][0]}
        rr = res["test"][arm]["rmse"]
        log(f"[{arm}] z500 RMSE " + " ".join(f"{h}:{rr[h]['z']:.0f}" for h in rr)
            + " | t850 24h {:.2f} t2m 24h {:.2f}".format(rr["24"]["t"], rr["24"]["t2m"])
            + " | SSR z500 " + " ".join(f"{res['test'][arm]['ssr'][h]['z']:.2f}" for h in rr)
            + " | CRPS z500 " + " ".join(f"{res['test'][arm]['crps_ens'][h]['z']:.4f}" for h in rr))
    res["test"]["truth_residual"] = {k: float(np.mean([q[k] for q in phys["truth"]])) for k in phys["truth"][0]}
    if bands:
        res["test"]["pinn_spread_ratio"] = {b: float(np.mean([q[b] for q in bands])) for b in bands[0]}
        log(f"[pinn] spread after/before P: {res['test']['pinn_spread_ratio']}")
    log("[residual] " + " | ".join(f"{arm}: " + " ".join(f"{k} {v:.2f}" for k, v in
                                                          (res['test'][arm]['residual'] if arm != 'truth' else
                                                           res['test']['truth_residual']).items())
                                    for arm in arms + ("truth",)))
    res["test"]["n_ic"] = sum(len(v) for v in sc[arms[0]]["rmse"])
    with open(os.path.join(a.out, f"{tag}{'_pinn' if P is not None else ''}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
