"""Step 9 experiment: classical channel-wise spread calibration, then an operator for the spatial
and inter-variable structure a spread calibration cannot change (owner's classical1-4 notes,
2026-10-24; maps in ``pgcfm/structure.py``).

Same synthetic data, b_j and frozen ``runs/step7/serial.pt`` as steps 7-8. Three periods the
serial chain never saw: operators are trained on 2005 (step 8's calibration year), every
calibration of every arm is fitted on a new validation year 2006 (16 ten-day rollouts), scores
are on 2020 (4 ICs) and 2021 (16 ICs). 8 members, Heun 8 steps.

Operator arms (base transition):
    base           the step 7 serial chain
    base_noQ       the same chain with the PINN Q bypassed at inference (flow matching alone)
    base_nophys    the step 7 serial chain trained without the physics term (``serial_nophys.pt``)
    op             StructureOperator (member + ensemble mean + log spread + condition), free
    op_sp          the same, deviations rescaled to the input's channel spread (spread kept)
    op_sp_member   op_sp without the ensemble statistics in the input (member-wise operator)
    op_sp_nophys   op_sp trained without the physics term
    op_sp_full     op_sp without the DCT band limit on the correction (may add grid-scale detail)
    op_sfno, op_sp_sfno  op / op_sp with the spherical SFNO (torch-harmonics) instead of the FNO
                   (found by the search-first pass: the sphere-aware operator in neuraloperator)
    loop_op, loop_op_sp  classical first, operator on the rest: the operator is trained and run on
                   the in-loop inflated chain (alpha of ``base/loop``), serial -> inflate -> operator
Calibrations, each fitted on 2006 for that arm (classical4's four comparisons = base / base+cal /
op / op+cal):
    raw            none
    cal_c          post-hoc s_c per channel (classical1), empirical CRPS
    cal_ck         post-hoc s_kc per lead and channel, empirical CRPS
    cal_c_fair     cal_c selected with the fair CRPS instead (for the criterion comparison)
    cal_c_tan      cal_c's scales, increment (s_c - 1) a projected on the hydrostatic tangent space at
                   each lead's mean (classical1's "physics projection after the spread change")
    loop           in-loop scalar inflation (step 8 ``inflate``), alpha by empirical CRPS
    mos_ck         classical mean correction a_kc + b_kc mu (least squares) then s_kc refitted:
                   the control for an operator that only damps the mean (= MBM of Van Schaeybroeck and
                   Vannitsem 2015 with a least-squares mean part)
    mos_ck_tan     mos_ck with its change (mean shift and spread) projected on the hydrostatic tangent space

    python scripts/run_step9_structure.py --out runs/step9 --step7 runs/step7 [--arms ...] [--quick]
Finished (arm, calibration) entries are skipped on a rerun; trained operators are kept in
``<out>/<arm>.pt``; ``--force`` reruns everything.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm.cfm import rollout  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset, kai_data_config  # noqa: E402
from pgcfm.ens_operator import train_operator  # noqa: E402
from pgcfm.latent import LatentConfig, LatentPGCFM  # noqa: E402
from pgcfm.physics.primitive import ResidualScales  # noqa: E402
from pgcfm.projection import freeze  # noqa: E402
from pgcfm.spread import KEEP, DCTBands, EnsembleCalibrated, ensemble_scores  # noqa: E402
from pgcfm.structure import (LinearMeanCalibrator, LoopInflated, SpreadCalibrator, StructureCorrected,  # noqa: E402
                             StructureOperator, lead_crps, structure_scores, tangent_calibrated)
from scripts.run_step4_pgcfm import member_residuals  # noqa: E402
from scripts.train_pgcfm import DEFAULTS, build  # noqa: E402

OPS = {  # name: (use_stats, preserve_spread, band limit, physics, network)
    "op": (True, False, True, True, "fno"),
    "op_sp": (True, True, True, True, "fno"),
    "op_sp_member": (False, True, True, True, "fno"),
    "op_sp_nophys": (True, True, True, False, "fno"),
    "op_sp_full": (True, True, False, True, "fno"),
    "op_sfno": (True, False, True, True, "sfno"),
    "op_sp_sfno": (True, True, True, True, "sfno"),
    "loop_op": (True, False, True, True, "fno"),
    "loop_op_sp": (True, True, True, True, "fno"),
}
LOOP_ALPHAS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
CALS = ("raw", "cal_c", "cal_ck", "cal_c_fair", "cal_c_tan", "loop", "mos_ck", "mos_ck_tan")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/step9")
    p.add_argument("--step7", default="runs/step7")
    p.add_argument("--test16", default="runs/step8/data/test16", help="step 8's 2021 test (made if missing)")
    p.add_argument("--arms", nargs="+", default=["base", *OPS])
    p.add_argument("--cals", nargs="+", default=list(CALS))
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--n-test16", type=int, default=16)
    p.add_argument("--iters-op", type=int, default=300)
    p.add_argument("--curriculum", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lambda-p", type=float, default=0.1)
    p.add_argument("--lambda-corr", type=float, default=1.0)
    p.add_argument("--lambda-grad", type=float, default=1.0)
    p.add_argument("--grad-ratio", type=float, default=0.5)
    p.add_argument("--force", action="store_true")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.members, a.steps, a.horizon, a.n_val, a.n_test16, a.iters_op, a.hidden = 3, 2, 2, 2, 2, 2, 8
        a.curriculum = [1, 2]
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    root7 = os.path.join(a.step7, "data")
    tr, te, cal = (os.path.join(root7, n) for n in ("train", "test", "calib"))
    mean_p, std_p = os.path.join(tr, "stats", "global_mean.npy"), os.path.join(tr, "stats", "global_std.npy")
    dcfg = kai_data_config(mean_p, std_p)
    pair_ds = PGCFMSequenceDataset(dcfg, tr, horizon=1)
    n_lat, n_lon = pair_ds.img_shape_x, pair_ds.img_shape_y
    # in child processes: reading a file written in this process segfaults in netCDF4/HDF5
    gen = os.path.join(os.path.dirname(__file__), "make_balanced_era5.py")
    va = os.path.join(a.out, "data", "valid")
    te16 = a.test16
    for path, year, n, seed in ((va, 2006, a.n_val, 600), (te16, 2021, a.n_test16, 2000)):
        if not os.path.exists(os.path.join(path, f"{year}.nc")):
            subprocess.run([sys.executable, gen, "--out", path, "--years", str(year), "--days",
                            str(a.horizon + 2 + 5 * n), "--n-lat", str(n_lat), "--n-lon", str(n_lon),
                            "--seed", str(seed)], check=True)

    def starts(path, n):
        ds = PGCFMSequenceDataset(dcfg, path, horizon=a.horizon, train=False)
        return torch.utils.data.default_collate([ds[5 * i] for i in range(min(n, len(ds) // 5))])

    sets = {"test": starts(te, 4), "test16": starts(te16, a.n_test16)}
    val = starts(va, a.n_val)
    cal_loader = torch.utils.data.DataLoader(PGCFMSequenceDataset(dcfg, cal, horizon=max(a.curriculum)),
                                             batch_size=4, shuffle=True, drop_last=True)

    res7 = json.load(open(os.path.join(a.step7, "results.json")))
    a7 = next(v for k, v in res7.items() if k.startswith("args_"))
    lcfg = LatentConfig(k_lat=a7.get("k_lat", 8), k_lon=a7.get("k_lon", 16), latent=a7.get("latent", 64),
                        enc_hidden=tuple(a7.get("enc_hidden", (256, 128, 64))),
                        dec_hidden=tuple(reversed(a7.get("enc_hidden", (256, 128, 64)))),
                        carry_highpass=a7.get("carry_highpass", False), coupling="serial")
    base = build(dict(DEFAULTS, model_channels=32, channel_mult=[1, 2, 2], num_res_blocks=1), n_lat, n_lon,
                 np.load(mean_p), np.load(std_p))
    guide = base.guide
    guide.scales = ResidualScales(**res7["scales"])
    freeze(guide)
    gm = SimpleNamespace(guide=guide)
    serial = LatentPGCFM(lcfg, n_lat, n_lon)
    serial.load_state_dict(torch.load(os.path.join(a.step7, "serial.pt"))["latent"])
    freeze(serial)
    serial.eval()
    bands = DCTBands(n_lat, n_lon, lcfg.k_lat, lcfg.k_lon)

    res_path = os.path.join(a.out, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    res["args"] = vars(a)

    def save():
        with open(res_path + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(res_path + ".tmp", res_path)

    def roll(tr_, batch, seed=1):
        tr_.layout(a.members, "interleave")
        with torch.no_grad():
            return rollout(tr_, batch["x0"], batch["static"], batch["doy"], a.members, steps=a.steps, seed=seed)

    def crps_both(ens, batch):
        """(empirical, fair) CRPS per lead, (K, 2)."""
        e, o = ens.transpose(0, 1)[..., KEEP, :, :], batch["targets"][..., KEEP, :, :]
        return list(zip(lead_crps(e, o), lead_crps(e, o, "fair")))

    def scores(ens, batch):
        e, o = ens.transpose(0, 1)[..., KEEP, :, :], batch["targets"][..., KEEP, :, :]
        r = ensemble_scores(e, o, bands)
        r["lead_crps_emp"] = [c[0] for c in crps_both(ens, batch)]
        r.update(structure_scores(e, o, bands))
        with torch.no_grad():
            r["lead_member_residual"] = member_residuals(gm, batch["x0"], ens, batch["static"], batch["doy"])
        return r

    def report(name, r):
        s, s16 = r["test"], r["test16"]
        K = len(s16["lead_crps"])
        pick = lambda v: " / ".join(f"{v[i]:.3f}" for i in sorted({0, min(4, K - 1), K - 1}))  # noqa: E731
        print(f"[{name}] test16 CRPS {pick(s16['lead_crps'])} (emp {pick(s16['lead_crps_emp'])}) "
              f"rmse {pick(s16['lead_rmse'])} SSR {pick(s16['lead_ssr'])} ES {pick(s16['lead_es'])} "
              f"gradCRPS {pick(s16['lead_grad_crps'])} xcorr {pick(s16['lead_xcorr_err'])} "
              f"hyd {pick(s16['lead_member_residual']['hyd'])} | test CRPS {pick(s['lead_crps'])}", flush=True)

    # truth's own residuals, the reference for every lead_member_residual
    if "truth_residual" not in res:
        res["truth_residual"] = {n: member_residuals(gm, b["x0"], b["targets"][:, None], b["static"], b["doy"])
                                 for n, b in sets.items()}
        save()

    def transition(name):
        if name == "base":
            return EnsembleCalibrated(serial, "serial"), {}
        if name == "base_noQ":  # flow matching alone: the PINN Q bypassed at inference
            return EnsembleCalibrated(serial, "noQ"), {}
        if name == "base_nophys":  # the step 7 serial chain trained without the physics term
            m0 = LatentPGCFM(lcfg, n_lat, n_lon)
            m0.load_state_dict(torch.load(os.path.join(a.step7, "serial_nophys.pt"))["latent"])
            freeze(m0)
            return EnsembleCalibrated(m0.eval(), "serial"), {}
        use_stats, keep_spread, band, phys, net = OPS[name]
        torch.manual_seed(9)
        lm = min(12, n_lat // 2)
        op = StructureOperator(n_modes=(lm, 2 * lm) if net == "sfno" else (lm, min(12, n_lon // 2)),
                               hidden=a.hidden, use_stats=use_stats, net=net)
        under = EnsembleCalibrated(serial, "serial")
        if name.startswith("loop_"):  # needs base/loop in results.json
            under = LoopInflated(under, res["base/loop"]["alpha"])
        w = StructureCorrected(under, op, band=serial.dct_x if band else None, preserve_spread=keep_spread)
        path = os.path.join(a.out, f"{name}.pt")
        if os.path.exists(path) and not a.force:
            ck = torch.load(path, weights_only=False)
            op.load_state_dict({k: v for k, v in ck["op"].items() if not k.endswith("_metadata")})
            return w, ck["info"]
        t = time.time()
        ck = os.path.join(a.out, f"{name}_train.pt")
        hist = train_operator(w, guide, cal_loader, a.iters_op, members=4, steps=4, grad_ratio=a.grad_ratio,
                              lambda_p=a.lambda_p if phys else 0.0, curriculum=tuple(a.curriculum),
                              lambda_corr=a.lambda_corr, lambda_grad=a.lambda_grad, bptt=False,
                              log_every=max(1, a.iters_op // 6), ckpt=ck)
        info = {"train_seconds": time.time() - t,
                "op_log_last": {k: float(np.mean([h[k] for h in hist[-30:] if k in h])) for k in hist[-1]}}
        torch.save({"op": op.state_dict(), "info": info}, path)
        if os.path.exists(ck):
            os.remove(ck)
        return w, info

    for name in a.arms:
        todo = [c for c in a.cals if a.force or f"{name}/{c}" not in res]
        if not todo:
            continue
        tr_, info = transition(name)
        res.setdefault(f"{name}/info", info)
        t = time.time()
        raw = {sn: roll(tr_, b) for sn, b in sets.items()}
        need_val = any(c.startswith(("cal_", "mos_")) for c in todo)
        vens = roll(tr_, val, seed=3) if need_val else None
        for c in todo:
            extra = {}
            if c == "raw":
                ens = raw
            elif c.startswith("cal_"):
                C = vens.shape[3]
                calib = SpreadCalibrator(C, per_lead=c == "cal_ck", leads=vens.shape[2])  # cal_c_tan: cal_c's fit
                crit = "fair" if c.endswith("fair") else "empirical"
                fit = calib.fit_validation(vens, val["targets"], criterion=crit)
                extra = {"scales": calib.scales.tolist(), "edge": fit["edge"]}
                if c == "cal_c_tan":
                    ens = {sn: tangent_calibrated(e, calib(e), guide, sets[sn]["x0"], sets[sn]["static"],
                                                  sets[sn]["doy"]) for sn, e in raw.items()}
                else:
                    ens = {sn: calib(e) for sn, e in raw.items()}
            elif c in ("mos_ck", "mos_ck_tan"):
                mos = LinearMeanCalibrator(vens.shape[3], vens.shape[2]).fit_validation(vens, val["targets"])
                calib = SpreadCalibrator(vens.shape[3], per_lead=True, leads=vens.shape[2])
                fit = calib.fit_validation(mos(vens), val["targets"])
                extra = {"scales": calib.scales.tolist(), "edge": fit["edge"], "mos_a": mos.a.tolist(),
                         "mos_b": mos.b.tolist()}
                ens = {sn: calib(mos(e)) for sn, e in raw.items()}
                if c == "mos_ck_tan":
                    ens = {sn: tangent_calibrated(raw[sn], e, guide, sets[sn]["x0"], sets[sn]["static"],
                                                  sets[sn]["doy"]) for sn, e in ens.items()}
            else:  # loop: in-loop scalar inflation fitted on the validation year
                grid = {al: crps_both(roll(LoopInflated(tr_, al), val, seed=3), val) for al in LOOP_ALPHAS}
                mean = {al: np.mean(v, axis=0).tolist() for al, v in grid.items()}  # (emp, fair)
                al_emp = min(mean, key=lambda k: mean[k][0])
                al_fair = min(mean, key=lambda k: mean[k][1])
                extra = {"alpha": al_emp, "alpha_fair": al_fair, "grid": {f"{k:g}": v for k, v in mean.items()}}
                print(f"[fit] {name} loop alpha emp {al_emp} fair {al_fair}", flush=True)
                lw = LoopInflated(tr_, al_emp)
                ens = {sn: roll(lw, b) for sn, b in sets.items()}
            r = {sn: scores(ens[sn], b) for sn, b in sets.items()}
            r.update(extra)
            res[f"{name}/{c}"] = r
            report(f"{name}/{c}", r)
            save()
        res[f"{name}/info"]["eval_seconds"] = time.time() - t
        save()
    return res


if __name__ == "__main__":
    main()
