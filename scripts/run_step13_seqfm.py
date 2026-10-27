"""Step 13, stage 2: sequence flow matching (pgcfm/seqdct.SeqFM) + the frozen stage-1 PINN, scored on
ClimODE's protocol, as the owner's four-model comparison (H1-H3, docs/step13_sequence.md):

    --model regress   H1's control: the same U-Net and inputs trained by MSE + statistical errors
                      (Gaussian, diagonal in the 3-D DCT basis of the residual divided by its RMS
                      per lead and channel, fitted on validation-year residuals)
    --model fm        arms  raw          the flow alone
                            pinn         + the frozen PINN on every member, after generation (H2)
                            guided       physics guidance in the late generation ODE (H3, --guide-lam)
                            guided_pinn  both
                            inloop       the frozen PINN inside the late generation ODE (--inloop-mu): its
                                         correction of the endpoint estimate steers the flow, which sees
                                         the corrected state on the later steps
                            smoothed     H3's control (--smooth-lam): the same late guidance with a Laplacian
                                         roughness instead of the physics (no physics, only smoothing)
    every arm then gets the hard constraints q >= 0, tcwv >= 0 (the violated fraction is reported);
    raw, guided, inloop and smoothed start from the same noise, so their differences are paired
    (--resume-ens replays the noise of the skipped years, so arms added later stay paired).

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

Long runs (docs/step15_training_system.md): ``--device cuda [--amp]``, ``--cache DIR`` (memory-mapped
years), ``--val-every N`` (the model's loss on fixed validation windows, plus the z500 RMSE of a small
ensemble with ``--val-members``; the best state is kept and selected), ``--patience``, ``--max-hours``
(saves and exits 75; the same command resumes), ``<tag>.pt_metrics.csv``. The random streams do not
depend on these options, so the defaults reproduce the earlier script; noise draws stay on the CPU, so
the members are the same on every device.
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
from pgcfm.seqdct import (PINNConfig, PINNProjector, SeqFM, SeqFMConfig, SeqRegressor, band_spread,  # noqa: E402
                          hard_constraints, laplacian_roughness, residual_rms, sequence_residual_loss)
from scripts.run_step11_climode import VARS, climode_minmax, climode_scores, climode_truth  # noqa: E402
from pgcfm.runs import (EXIT_BUDGET, FitConfig, Prefetch, autocast, device_from, fit, load_years,  # noqa: E402
                        run_record)
from scripts.run_step12_core import load_static  # noqa: E402

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
    p.add_argument("--model", default="fm", choices=["fm", "regress"])
    p.add_argument("--basis", default="dct", choices=["dct", "grid"], help="fm only")
    p.add_argument("--pinn", default=None, help="stage-1 checkpoint (pinn_<objective>.pt); none = no PINN")
    p.add_argument("--guide-lam", type=float, default=0.0, help="H3 guidance strength (fraction of |v|); 0 = off")
    p.add_argument("--guide-s0", type=float, default=0.75)
    p.add_argument("--inloop-mu", type=float, default=0.0, help="PINN inside the generation ODE (needs --pinn); 0 = off")
    p.add_argument("--smooth-lam", type=float, default=0.0, help="Laplacian smoothing guidance (H3 control); 0 = off")
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
                   help="save the ensembles (5 ClimODE vars, 24 h multiples) of every arm in the test years and of "
                        "pinn (raw without a PINN) in the validation year (stage 3 trains on it)")
    p.add_argument("--resume-ens", action="store_true",
                   help="with --save-ens: skip the years whose ensembles are all saved already (the json then "
                        "scores only the years generated in this run)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N or auto")
    p.add_argument("--amp", action="store_true", help="bf16 autocast on cuda for training (not validated here)")
    p.add_argument("--cache", default=None, help="folder of memory-mapped normalized years (built once)")
    p.add_argument("--prefetch", type=int, default=2, help="batches gathered ahead in a thread (0 = none)")
    p.add_argument("--val-every", type=int, default=0, help="loss on fixed validation windows every N steps; "
                   "the best state is kept (<tag>.pt.best) and selected at the end (0 = off)")
    p.add_argument("--n-val-train", type=int, default=16, help="validation windows of --val-every")
    p.add_argument("--val-members", type=int, default=0,
                   help="with --val-every: members sampled per validation window for the z500 RMSE (0 = none)")
    p.add_argument("--patience", type=int, default=0, help="validations without improvement before stopping")
    p.add_argument("--select", default="best", choices=["best", "last"])
    p.add_argument("--max-hours", type=float, default=0.0, help="wall-clock budget; saves and exits 75")
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--ic-chunk", type=int, default=4,
                   help="ICs generated together (x members); keep it when resuming saved ensembles (noise replay)")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.train_years, a.iters, a.K, a.channels, a.max_ics, a.members, a.heun = \
            a.train_years[:1], 3, 4, 32, 2, 2, 2
        a.n_val_train = 2
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = device_from(a.device)
    tag = (f"seqfm_{a.basis}" if a.model == "fm" else "seqregress") + ("_quick" if a.quick else "")
    log = lambda s: print(s, flush=True)  # noqa: E731
    stats = os.path.join(a.out, "stats")  # written by run_step13_pinn.py on the same training years
    mean, std = np.load(os.path.join(stats, "global_mean.npy")), np.load(os.path.join(stats, "global_std.npy"))
    data = load_years(a.kai, a.train_years + [a.val_year], mean, std, cache=a.cache)
    with nc.Dataset(os.path.join(a.kai, f"{a.val_year}.nc")) as d:
        contract.register_latitudes(np.asarray(d["latitude"][:]))
    lat, lon, oro, lsm = load_static(a.era5)
    static = static_features(lat, lon, oro, lsm)
    w = torch.from_numpy(np.cos(np.deg2rad(lat)).astype(np.float32))
    w = (w / w.mean()).view(1, 1, 1, -1, 1).to(dev)
    cfg = SeqFMConfig(K=a.K, basis=a.basis, model_channels=a.channels)
    model = SeqFM(cfg, static) if a.model == "fm" else SeqRegressor(cfg, static)
    res = {"args": vars(a), "cfg": dataclasses.asdict(cfg), "n_params": sum(q.numel() for q in model.unet.parameters()),
           "run": run_record(dev)}
    log(f"[stage 2 {tag}] {res['n_params']} parameters, {res['run']['device_name']} commit {res['run']['commit']}")
    draws = lambda years, n: (batch(data, random_picks(data, years, a.K, rng, 8), a.K) for _ in range(n))  # noqa: E731
    to_dev = lambda b: tuple(q.to(dev) for q in b)  # noqa: E731

    # -- training: each model's own loss (the FM is never trained through the PINN) -----------------
    ck = os.path.join(a.out, f"{tag}.pt")
    if os.path.exists(ck):
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.to(dev)
    else:
        if a.model == "fm":
            model.fit_basis(q[1] for q in draws(a.train_years, 32))
        else:
            model.fit_scale(q[1] for q in draws(a.train_years, 32))
        model.to(dev)
        opt = torch.optim.Adam(model.unet.parameters(), lr=a.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)

        def loss_of(b):
            h, d, _, hour0 = to_dev(b)
            return model.loss(h, d, hour0) if a.model == "fm" else model.loss(h, d, hour0, w)

        # fixed validation windows; the FM loss draws its own tau and noise from a fixed seed each time
        vr = np.random.default_rng(a.seed + 99)
        vw = [batch(data, random_picks(data, [a.val_year], a.K, vr, a.batch), a.K)
              for _ in range(0, a.n_val_train, a.batch)] if a.val_every else []
        iz = DYN_IDX["z"]

        def val_fn():
            with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
                torch.manual_seed(a.seed + 98)
                out = {"loss": float(np.mean([loss_of(b).item() for b in vw]))}
                if a.val_members:  # z500 RMSE of the member mean at 24 h and the last lead (normalized units)
                    gv = torch.Generator().manual_seed(a.seed + 97)
                    se = {}
                    for b in vw:
                        h, d, _, hour0 = to_dev(b)
                        hm, h0m = h.repeat_interleave(a.val_members, 0), hour0.repeat_interleave(a.val_members)
                        e = model.sample(hm, h0m, a.heun, generator=gv) if a.model == "fm" \
                            else model.sample(hm, h0m, generator=gv)
                        mu = e.view(len(h), a.val_members, *e.shape[1:]).mean(1)
                        for k in {min(3, a.K - 1), a.K - 1}:
                            se.setdefault(k, []).append(((mu[:, k, iz] - d[:, k, iz]) ** 2 * w[0, 0]).mean().item())
                    out.update({f"z500_rmse_{6 * (k + 1)}h": float(np.sqrt(np.mean(v))) for k, v in se.items()})
            return out

        r = fit(model, model.unet.parameters(), opt, sched,
                Prefetch(lambda r_: random_picks(data, a.train_years, a.K, r_, a.batch),
                         lambda p_: batch(data, p_, a.K), rng, a.prefetch),
                loss_of, FitConfig(iters=a.iters, ckpt_every=a.ckpt_every, log_every=max(1, a.iters // 30),
                                   val_every=a.val_every, patience=a.patience, select=a.select,
                                   max_hours=a.max_hours),
                ck, val_fn=val_fn if a.val_every else None, amp=autocast(dev, a.amp), log=log)
        if r["stopped"] == "budget":
            sys.exit(EXIT_BUDGET)
        if a.model == "regress":  # statistical errors: residuals of the regression on the validation year
            with torch.no_grad():
                model.fit_errors(q[1].to(dev) - model(q[0].to(dev), q[3].to(dev)) for q in draws([a.val_year], 16))
        torch.save(model.state_dict(), ck)
        res.update(train_seconds=r["seconds"], loss=[h["loss"] for h in r["hist"]][::10], val_curve=r["val"],
                   best_it=r["best_it"], best_val=r["best_val"], stopped=r["stopped"])
    model.eval()

    P = None
    if a.pinn:
        pc = json.load(open(a.pinn.replace(".pt", ".json")))["cfg"]
        P = PINNProjector(PINNConfig(**{**pc, "blocks": tuple(pc["blocks"])}), static)
        P.load_state_dict(torch.load(a.pinn, map_location="cpu"))
        P.to(dev)
        P.eval()
        if P.cfg.K != a.K:
            raise ValueError("PINN and FM need the same K")
    guide = PhysicsGuide(32, 64, mean, std, PhysicsConfig(dt_seconds=6 * 3600.0))
    est, r_est = [], np.random.default_rng(a.seed)  # its own stream: the scales no longer depend on training/resume
    for _ in range(10):
        h, d, orog, _ = batch(data, random_picks(data, a.train_years, 1, r_est, 16), 1)
        est.append({"x0": h[:, -1], "targets": (h[:, -1] + d[:, 0])[:, None], "static": orog, "doy": torch.zeros(16, 1)})
    guide.scales = estimate_scales(guide, est)
    for q in guide.parameters():
        q.requires_grad_(False)
    guide.to(dev)

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

    # -- ensembles on ClimODE's ICs ---------------------------------------------------------------
    guided = a.model == "fm" and a.guide_lam > 0
    inloop = a.model == "fm" and a.inloop_mu > 0 and P is not None
    smoothed = a.model == "fm" and a.smooth_lam > 0
    arms = ("raw",) + (("pinn",) if P is not None else ()) + (("guided",) if guided else ()) \
        + (("guided_pinn",) if guided and P is not None else ()) + (("inloop",) if inloop else ()) \
        + (("smoothed",) if smoothed else ())
    pair_ids = torch.tensor([0, a.K // 2, a.K - 1])  # residual diagnostics and the H3 guidance
    mx, mn = climode_minmax(era5)
    wl = np.cos(np.deg2rad(lat))
    wl = (wl / wl.mean())[:, None]
    sc = {arm: {k: [] for k in ("rmse", "acc", "crps_gauss", "crps_ens", "spread", "rmse_pers")} for arm in arms}
    phys = {arm: [] for arm in arms + ("truth",)}
    viol = {arm: [] for arm in arms}
    bands = {arm: [] for arm in arms if arm.endswith("pinn") or arm == "inloop"}
    cost = {arm: 0.0 for arm in arms}
    g = torch.Generator().manual_seed(a.seed + 7)
    skipped = []
    saved = lambda arm, y: os.path.join(a.out, f"{tag}_{arm}_ens_{y}.npz")  # noqa: E731
    for y in ([a.val_year] if a.save_ens else []) + a.test_years:
        score = y in a.test_years
        here = list(arms) if score else [q for q in ("raw", "pinn") if q in arms]  # validation: no guidance
        if not score:
            here = here[-1:] if a.resume_ens else here  # only the saved arm matters on a resumed run
        existing = {arm for arm in here if a.resume_ens and os.path.exists(saved(arm, y))}
        with nc.Dataset(os.path.join(a.kai, f"{y}.nc")) as f_:
            n_ic = (len(f_.dimensions["time"]) - 2 - a.K) // 8 + 1
        n_ic = min(n_ic, a.max_ics or n_ic)
        if existing == set(here):  # saved by an earlier run: replay its noise draws (one per chunk) and skip
            for c0 in range(0, n_ic, a.ic_chunk):
                torch.randn((min(a.ic_chunk, n_ic - c0) * a.members, a.K, contract.N_DYNAMIC, 32, 64), generator=g)
            skipped.append(y)
            log(f"  {y}: ensembles exist, skipped (noise replayed)")
            continue
        here = [arm for arm in here if arm not in existing]
        keep = {arm: {"E": [], "O": [], "X0": [], "ics": []} for arm in (here if score else here[-1:])}
        check_raw = "raw" in existing  # the replayed raw members must equal the saved ones
        x = data[y] if y == a.val_year else load_years(a.kai, [y], mean, std, cache=a.cache)[y]
        truth = climode_truth(era5, y)
        clim = truth.mean(0)
        t0 = time.time()
        for c0 in range(0, n_ic, a.ic_chunk):
            if c0 and c0 % 40 == 0:
                log(f"    {y}: {c0}/{n_ic} ICs {time.time() - t0:.0f}s")
            ics = [2 + 8 * e for e in range(c0, min(n_ic, c0 + a.ic_chunk))]
            h, d, orog, hour0 = to_dev(batch({y: x}, [(y, i) for i in ics], a.K))
            B = len(ics)
            hm, h0m, om = h.repeat_interleave(a.members, 0), hour0.repeat_interleave(a.members), \
                orog.repeat_interleave(a.members, 0)
            x0m = hm[:, -1]
            ens, tc = {}, time.time()
            need_guided = "guided" in here or "guided_pinn" in here
            branch = score and (need_guided or "inloop" in here or "smoothed" in here)
            if a.model == "fm":
                eps = model.noise(len(hm), g)
                ens["raw"] = model.sample(hm, h0m, a.heun, eps=eps, keep_at=a.guide_s0 if branch else None)
                if branch:  # the guided arms continue the raw trajectory from s0 (same noise, paired)
                    ens["raw"], state = ens["raw"]
            else:
                ens["raw"] = model.sample(hm, h0m, generator=g)
            cost["raw"] += time.time() - tc
            if check_raw and c0 == 0:
                z_ = np.load(saved("raw", y))
                sel0 = torch.tensor([k * 4 - 1 for k in range(1, a.K // 4 + 1)])
                mem0 = (x0m[:, None] + ens["raw"][:, sel0]).view(B, a.members, len(sel0), *ens["raw"].shape[2:])
                e0 = to_vars(hard_constraints(x0m, ens["raw"], mean, std)[0][:, sel0].add(x0m[:, None])
                             .view_as(mem0).cpu().numpy()) - np.repeat(truth[ics][:, None], len(sel0), 1)[:, None]
                ref = z_["E_minus_x0"][:B].astype(np.float32)
                log(f"    replayed raw vs saved (first ICs): max |diff| {np.abs(e0 - ref).max():.3g}, "
                    f"member scale {np.abs(ref).mean():.3g}")
            if need_guided and score:
                tc = time.time()
                zero = torch.zeros_like(ens["raw"])
                ens["guided"] = model.sample(
                    hm, h0m, a.heun, start=state, lam=a.guide_lam, s0=a.guide_s0, chunk=a.members,
                    physics=lambda q, j: sequence_residual_loss(guide, x0m[j], q, zero[j], om[j], pair_ids)[0])
                cost["guided"] += time.time() - tc
            if "inloop" in here and score:
                tc = time.time()
                ens["inloop"] = model.sample(hm, h0m, a.heun, start=state, s0=a.guide_s0, mu=a.inloop_mu,
                                             project=lambda q: P(x0m, q, h0m))
                cost["inloop"] += time.time() - tc
            if "smoothed" in here and score:
                tc = time.time()
                ens["smoothed"] = model.sample(hm, h0m, a.heun, start=state, lam=a.smooth_lam, s0=a.guide_s0,
                                               chunk=a.members, physics=lambda q, j: laplacian_roughness(q))
                cost["smoothed"] += time.time() - tc
            for src, arm in (("raw", "pinn"), ("guided", "guided_pinn")):
                if arm in here:
                    tc = time.time()
                    with torch.no_grad():
                        ens[arm] = P(x0m, ens[src], h0m)
                    cost[arm] += time.time() - tc
            for arm in here:
                ens[arm], f = hard_constraints(x0m, ens[arm], mean, std)
                if score:
                    viol[arm].append(f)
            if score:
                for arm, src in (("pinn", "raw"), ("guided_pinn", "guided"), ("inloop", "raw")):
                    if arm in ens and arm in here:
                        e0 = ens[src].view(B, a.members, *ens[src].shape[1:]).transpose(0, 1)
                        e1 = ens[arm].view(B, a.members, *ens[arm].shape[1:]).transpose(0, 1)
                        s0_, s1_ = band_spread(e0, P.cfg.k_lat, P.cfg.k_lon), band_spread(e1, P.cfg.k_lat, P.cfg.k_lon)
                        bands[arm].append({k: s1_[k] / s0_[k] for k in s0_})
                phys["truth"].append(residual_rms(guide, h[:, -1], d, d, orog, pair_ids))
            sel = torch.tensor([k * 4 - 1 for k in range(1, a.K // 4 + 1)])  # 24 h multiples
            O = np.stack([truth[[i + hh // 6 for hh in LEADS_H[:len(sel)]]] for i in ics])  # (B, L, 5, H, W)
            X0 = np.repeat(truth[ics][:, None], len(sel), 1)
            for arm in here:
                inc = ens[arm]
                mem = (x0m[:, None] + inc[:, sel]).view(B, a.members, len(sel), *inc.shape[2:])
                E = to_vars(mem.cpu().numpy())  # (B, M, L, 5, H, W)
                if a.save_ens and arm in keep:
                    keep[arm]["E"].append((E - X0[:, None]).astype(np.float16)), keep[arm]["O"].append(O)
                    keep[arm]["X0"].append(X0[:, 0]), keep[arm]["ics"].extend(ics)
                if not score:
                    continue
                phys[arm].append(residual_rms(guide, x0m, inc, d.repeat_interleave(a.members, 0), om, pair_ids))
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
            for arm, kp in keep.items():
                np.savez(os.path.join(a.out, f"{tag}_{arm}_ens_{y}.npz"), E_minus_x0=np.concatenate(kp["E"]),
                         O=np.concatenate(kp["O"]), X0=np.concatenate(kp["X0"]), ics=np.array(kp["ics"]),
                         leads=np.array(LEADS_H[:len(sel)]))
        log(f"  {'test' if score else 'val'} {y}: {n_ic} ICs {time.time() - t0:.0f}s")
    res["skipped_years"] = skipped
    res["test"] = {"truth_residual": {k: float(np.mean([q[k] for q in phys["truth"]])) for k in phys["truth"][0]}}
    for arm in arms:
        if not sc[arm]["rmse"]:  # every test year of this arm was saved by an earlier run
            continue
        n_ic = sum(len(v) for v in sc[arm]["rmse"])
        r = {k: np.concatenate(v) for k, v in sc[arm].items()}
        t = {k: {str(h): {v: float(r[k][:, j, i].mean()) for i, v in enumerate(VARS)}
                 for j, h in enumerate(LEADS_H[:r[k].shape[1]])} for k in r}
        t["ssr"] = {str(h): {v: float(np.sqrt((r["spread"][:, j, i] ** 2).mean() * (a.members + 1) / a.members)
                                      / np.sqrt((r["rmse"][:, j, i] ** 2).mean()))
                             for i, v in enumerate(VARS)} for j, h in enumerate(LEADS_H[:r["rmse"].shape[1]])}
        t["residual"] = {k: float(np.mean([q[k] for q in phys[arm]])) for k in phys[arm][0]}
        t["hard_violation_frac"] = float(np.mean(viol[arm]))
        t["seconds_per_ic"] = cost[arm] / n_ic
        if arm in bands:
            t["spread_ratio_through_pinn"] = {k: float(np.mean([q[k] for q in bands[arm]])) for k in bands[arm][0]}
        t["n_ic"] = n_ic
        res["test"][arm] = t
        log(f"[{arm}] z500 RMSE " + " ".join(f"{h}:{t['rmse'][h]['z']:.0f}" for h in t["rmse"])
            + " | t850 24h {:.2f} t2m 24h {:.2f}".format(t["rmse"]["24"]["t"], t["rmse"]["24"]["t2m"])
            + " | SSR z500 " + " ".join(f"{t['ssr'][h]['z']:.2f}" for h in t["rmse"])
            + " | CRPS z500 " + " ".join(f"{t['crps_ens'][h]['z']:.4f}" for h in t["rmse"])
            + " | residual " + " ".join(f"{k} {v:.2f}" for k, v in t["residual"].items())
            + f" | q/tcwv<0 {t['hard_violation_frac']:.4f} | {t['seconds_per_ic']:.1f} s/IC"
            + (f" | spread through P {t['spread_ratio_through_pinn']}" if arm in bands else ""))
    log("[truth residual] " + " ".join(f"{k} {v:.2f}" for k, v in res["test"]["truth_residual"].items()))
    with open(os.path.join(a.out, f"{tag}{'_pinn' if P is not None else ''}{'_guided' if guided else ''}{'_inloop' if inloop else ''}"
                           f"{'_smooth' if smoothed else ''}{'_resumed' if a.resume_ens else ''}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
