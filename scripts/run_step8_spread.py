"""Step 8 experiment: calibrate the ensemble distribution of the frozen step 7 serial chain
(owner's hypotheses H1/H2, 2026-10-24; maps in ``pgcfm/spread.py``).

Same synthetic data, b_j and frozen ``runs/step7/serial.pt`` as step 7. Every free parameter is fit
on the calibration year (2005, never seen by the chain) from 16 ten-day rollouts; scores are on the
step 7 test (2020, 4 initial states, comparable with docs/step7_latent.md) and on a new
independent test (2021, 16 initial states), 8 members, Heun 8 steps, calibrated members fed back.

    serial            step 7 serial chain, unchanged (reproduces its row)
    inflate           A: mu + alpha (x_m - mu), alpha by CRPS grid search on the calibration year
    posthoc           A without feedback: alpha_k per lead applied after the rollout (statistical
                      post-processing; an upper bound for what a scalar can do at each lead)
    band              4: alpha_b per DCT band (3 bands of the kept block), coordinate search from alpha
    noQ               3: Q bypassed at inference, D(z_cfm)
    noQ_hyd           3: D(z_cfm) with the hard hydrostatic projection T <- Phi instead of Q
    qmean             Q corrects the mean only: mu_post + (pre-Q deviations)
    tangent           C: mu_post + P_T (pre-Q deviations), P_T = hydrostatic tangent space
    tangent_inflate   C + A: alpha on the tangent part of the post-Q deviations only
    covop / covop_nophys  B: ``CovarianceOperator`` (FNO log-scale field + linear SpectralConv mixing),
                      trained on the calibration year (fair energy score, 1->2->3 day curriculum,
                      +/- capped physics); cannot move the ensemble mean
    contraction       H2: variance before/after Q, split into hydrostatic tangent / normal parts
    nophys_contraction  same for the step 7 serial chain trained without physics (``serial_nophys.pt``,
                      written by ``run_step7_latent.py --arms lat_serial_nophys``)

    python scripts/run_step8_spread.py --out runs/step8 --step7 runs/step7 [--arms ...] [--quick]
Finished arms are skipped on a rerun (results.json); ``--force`` reruns them.
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
from pgcfm.ens_operator import OperatorCorrected, train_operator  # noqa: E402
from pgcfm.latent import LatentConfig, LatentPGCFM  # noqa: E402
from pgcfm.physics.primitive import ResidualScales  # noqa: E402
from pgcfm.projection import freeze  # noqa: E402
from pgcfm.spread import (KEEP, CovarianceOperator, DCTBands, EnsembleCalibrated,  # noqa: E402
                          TangentOperatorCorrected, contraction_report, ensemble_scores, fit_bands, fit_posthoc,
                          fit_scalar, posthoc_scale, rollout_contraction)
from pgcfm.train import take  # noqa: E402
from scripts.run_step4_pgcfm import member_residuals  # noqa: E402
from scripts.train_pgcfm import DEFAULTS, build  # noqa: E402

ALPHAS = (1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0)
BAND_FACTORS = (0.6, 0.8, 1.25, 1.6)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/step8")
    p.add_argument("--step7", default="runs/step7")
    p.add_argument("--arms", nargs="+", default=["serial", "inflate", "posthoc", "band", "noQ", "noQ_hyd", "qmean",
                                                  "tangent", "tangent_inflate", "contraction", "covop",
                                                  "covop_nophys", "covop_tangent", "nophys_contraction"])
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--n-cal", type=int, default=16)
    p.add_argument("--n-test16", type=int, default=16)
    p.add_argument("--iters-covop", type=int, default=150)
    p.add_argument("--lambda-p", type=float, default=0.1)
    p.add_argument("--grad-ratio", type=float, default=0.5)
    p.add_argument("--force", action="store_true")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.members, a.steps, a.horizon, a.n_cal, a.n_test16, a.iters_covop = 3, 2, 2, 2, 2, 2
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    root7 = os.path.join(a.step7, "data")
    tr, te, cal = (os.path.join(root7, n) for n in ("train", "test", "calib"))
    mean_p, std_p = os.path.join(tr, "stats", "global_mean.npy"), os.path.join(tr, "stats", "global_std.npy")
    dcfg = kai_data_config(mean_p, std_p)
    pair_ds = PGCFMSequenceDataset(dcfg, tr, horizon=1)
    n_lat, n_lon = pair_ds.img_shape_x, pair_ds.img_shape_y
    te16 = os.path.join(a.out, "data", "test16")
    if not os.path.exists(os.path.join(te16, "2021.nc")):  # independent, larger test
        # in a child process: reading a file written in this process segfaults in netCDF4/HDF5
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "make_balanced_era5.py"),
                        "--out", te16, "--years", "2021", "--days", str(a.horizon + 2 + 5 * a.n_test16),
                        "--n-lat", str(n_lat), "--n-lon", str(n_lon), "--seed", "2000"], check=True)

    def starts(path, n, train=False):
        ds = PGCFMSequenceDataset(dcfg, path, horizon=a.horizon, train=train)
        return torch.utils.data.default_collate([ds[5 * i] for i in range(min(n, len(ds) // 5))])

    sets = {"test": starts(te, 4), "test16": starts(te16, a.n_test16)}
    cal_batch = starts(cal, a.n_cal, train=True)
    val_batches = take(torch.utils.data.DataLoader(PGCFMSequenceDataset(dcfg, te, horizon=1, train=False),
                                                   batch_size=8), 2)
    cal_loader = torch.utils.data.DataLoader(PGCFMSequenceDataset(dcfg, cal, horizon=3), batch_size=4,
                                             shuffle=True, drop_last=True)

    res7 = json.load(open(os.path.join(a.step7, "results.json")))
    a7 = next(v for k, v in res7.items() if k.startswith("args_"))  # the step 7 latent settings
    lcfg = LatentConfig(k_lat=a7.get("k_lat", 8), k_lon=a7.get("k_lon", 16), latent=a7.get("latent", 64),
                        enc_hidden=tuple(a7.get("enc_hidden", (256, 128, 64))),
                        dec_hidden=tuple(reversed(a7.get("enc_hidden", (256, 128, 64)))),
                        carry_highpass=a7.get("carry_highpass", False), coupling="serial")
    base = build(dict(DEFAULTS, model_channels=32, channel_mult=[1, 2, 2], num_res_blocks=1), n_lat, n_lon,
                 np.load(mean_p), np.load(std_p))
    guide = base.guide
    guide.scales = ResidualScales(**res7["scales"])  # the step 7 b_j
    freeze(guide)
    gm = SimpleNamespace(guide=guide)

    def load_serial(name):
        m = LatentPGCFM(lcfg, n_lat, n_lon)
        m.load_state_dict(torch.load(os.path.join(a.step7, name))["latent"])
        freeze(m)
        return m.eval()

    serial = load_serial("serial.pt")
    bands = DCTBands(n_lat, n_lon, lcfg.k_lat, lcfg.k_lon)

    res_path = os.path.join(a.out, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    res["args_" + "_".join(a.arms)] = vars(a)

    def save():
        with open(res_path + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(res_path + ".tmp", res_path)

    def roll(tr_, batch, seed=1):
        getattr(tr_, "layout", lambda *_: None)(a.members, "interleave")
        with torch.no_grad():
            ens = rollout(tr_, batch["x0"], batch["static"], batch["doy"], a.members, steps=a.steps, seed=seed)
        return ens  # (B, M, K, 66, H, W)

    def scores(ens, batch):
        e, o = ens.transpose(0, 1)[..., KEEP, :, :], batch["targets"][..., KEEP, :, :]
        r = ensemble_scores(e, o, bands)
        with torch.no_grad():
            r["lead_member_residual"] = member_residuals(gm, batch["x0"], ens, batch["static"], batch["doy"])
        return r

    def mean_crps(tr_, batch):
        ens = roll(tr_, batch)
        e, o = ens.transpose(0, 1)[..., KEEP, :, :], batch["targets"][..., KEEP, :, :]
        return float(np.mean(ensemble_scores(e, o)["lead_crps"]))

    def score(name, tr_, extra=None, ens_map=None):
        t = time.time()
        r = {}
        for sname, b in sets.items():
            ens = roll(tr_, b)
            if ens_map is not None:
                ens = ens_map(ens)
            r[sname] = scores(ens, b)
        r["eval_seconds"] = time.time() - t
        r.update(extra or {})
        res[name] = r
        s, s16 = r["test"], r["test16"]
        K = len(s["lead_crps"])
        pick = lambda v: " / ".join(f"{v[i]:.3f}" for i in sorted({0, min(4, K - 1), K - 1}))  # noqa: E731
        print(f"[{name}] test CRPS {pick(s['lead_crps'])} SSR {pick(s['lead_ssr'])} | test16 CRPS "
              f"{pick(s16['lead_crps'])} SSR {pick(s16['lead_ssr'])} cov {pick(s16['lead_coverage'])} "
              f"hyd {pick(s16['lead_member_residual']['hyd'])}", flush=True)
        save()

    def todo(name):
        return name in a.arms and (a.force or name not in res)

    def wrap(mode, **kw):
        return EnsembleCalibrated(serial, mode, guide=guide, bands=bands, **kw)

    for mode in ("serial", "noQ", "noQ_hyd", "qmean", "tangent"):
        if todo(mode):
            score(mode, wrap(mode))

    # A: scalar inflation fitted on the calibration year
    if todo("inflate") or todo("tangent_inflate") or todo("band"):
        if "fit_inflate" not in res or a.force:
            alpha, grid = fit_scalar(lambda al: mean_crps(wrap("inflate", alpha=al), cal_batch), ALPHAS)
            res["fit_inflate"] = {"grid": grid, "alpha": alpha}
            print("[fit] inflate", json.dumps({f"{k:g}": round(v, 4) for k, v in grid.items()}), flush=True)
            save()
        alpha = res["fit_inflate"]["alpha"]
        if todo("inflate"):
            score("inflate", wrap("inflate", alpha=alpha), {"alpha": alpha})
        if todo("tangent_inflate"):
            score("tangent_inflate", wrap("tangent_inflate", alpha=alpha), {"alpha": alpha})
        if todo("band"):
            ba, trace = fit_bands(lambda al: mean_crps(wrap("band", band_alpha=al), cal_batch), alpha, bands.n,
                                  BAND_FACTORS)
            print("[fit] band", [round(v, 3) for v in ba], round(min(t[1] for t in trace), 4), flush=True)
            score("band", wrap("band", band_alpha=ba), {"band_alpha": ba, "fit_trace": trace})

    # A without feedback: per-lead scale of the finished serial rollout
    if todo("posthoc"):
        ens = roll(wrap("serial"), cal_batch)
        alphas = fit_posthoc(ens[..., KEEP, :, :], cal_batch["targets"][..., KEEP, :, :])
        print("[fit] posthoc alpha per lead", [round(v, 2) for v in alphas], flush=True)
        score("posthoc", wrap("serial"), {"alpha_per_lead": alphas}, ens_map=lambda e: posthoc_scale(e, alphas))

    # H2: does Q / the physics term remove admissible (tangent) or inadmissible (normal) variance?
    if todo("contraction"):
        res["contraction"] = contraction_report(serial, guide, val_batches, a.members, a.steps)
        print("[contraction]", json.dumps({k: round(v, 5) for k, v in res["contraction"].items()}), flush=True)
        save()
    if todo("contraction") or ("contraction" in a.arms and "rollout_contraction" not in res):
        lead_set = (1, 3, 6) if a.horizon > 6 else (1,)
        res["rollout_contraction"] = rollout_contraction(serial, guide, sets["test16"], a.members, a.steps, lead_set)
        print("[rollout_contraction]", json.dumps(res["rollout_contraction"]), flush=True)
        save()
    if todo("nophys_contraction"):
        path = os.path.join(a.step7, "serial_nophys.pt")
        if os.path.exists(path):
            m0 = load_serial("serial_nophys.pt")
            res["nophys_contraction"] = contraction_report(m0, guide, val_batches, a.members, a.steps)
            res["nophys_rollout_contraction"] = rollout_contraction(m0, guide, sets["test16"], a.members, a.steps,
                                                                    (1, 3, 6) if a.horizon > 6 else (1,))
            print("[nophys_contraction]", json.dumps({k: round(v, 5) for k, v in res["nophys_contraction"].items()}),
                  flush=True)
            score("serial_nophys", EnsembleCalibrated(m0, "serial", guide=guide))
        else:
            print(f"[nophys_contraction] skipped: {path} missing (run_step7_latent.py --arms lat_serial_nophys)")

    # B: covariance operator on the calibration year (covop_tangent: on the tangent map, output kept
    # in the hydrostatic tangent space)
    for name in ("covop", "covop_nophys", "covop_tangent"):
        if not todo(name):
            continue
        torch.manual_seed(7)
        op = CovarianceOperator(n_modes=(min(12, n_lat // 2), min(12, n_lon // 2)), hidden=8 if a.quick else 32)
        if name == "covop_tangent":
            w = TangentOperatorCorrected(wrap("tangent"), op, guide, band=serial.dct_x)
        else:
            w = OperatorCorrected(serial, op, band=serial.dct_x)
        ck = os.path.join(a.out, f"{name}_train.pt")
        t = time.time()
        hist = train_operator(w, guide, cal_loader, a.iters_covop, members=4, steps=4, grad_ratio=a.grad_ratio,
                              lambda_p=0.0 if name.endswith("nophys") else a.lambda_p, curriculum=(1, 2, 3),
                              lambda_corr=0.0, bptt=False, log_every=max(1, a.iters_covop // 6), ckpt=ck)
        last = {k: float(np.mean([h[k] for h in hist[-30:] if k in h])) for k in hist[-1]}
        score(name, w, {"train_seconds": time.time() - t, "op_log_last": last,
                        "gate_mix": float(op.gate_mix)})
        if os.path.exists(ck):
            os.remove(ck)
    return res


if __name__ == "__main__":
    main()
