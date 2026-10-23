"""Step 7 experiment: DCT latent PG-CFM, and whether to freeze the encoder in stage 2.

Same balanced synthetic data, grid, b_j estimate and scoring as step 6
(``run_step4_pgcfm.evaluate``), so the numbers line up with docs/step6.

    floors        DCT truncation / autoencoder floors vs the 24 h tendency (climate_diffusion check)
    grid arms     cfm           grid CFM of step 2/6 (+ the step 6 extra CFM iterations)
                  cfm_dct       same members projected onto the kept DCT modes (non-learned control)
                  cfm_smooth    step 6 Laplacian gradient flow (non-physical smoothing control)
    latent arms   pinn_det      stage 1 only: D(P(E(x))), no noise
                  lat_frozen    stage 2, encoder/decoder frozen, latent CFM + frozen PINN
                  lat_retrain   stage 2, encoder/decoder retrained (CFM + reconstruction)
                  lat_finetune  lat_frozen + low-LR decoder fine-tune on decoded CRPS
                  lat_nophys    stage 1 without the physics term, then lat_frozen (what the PINN term buys)
                  lat_serial    serial coupling encoder -> CFM members -> PINN Q per member -> decoder;
                                CFM (target z_{k+1} - z_k) and Q trained together on L_CFM + decoded
                                member CRPS + capped physics, gradient through the sampler
                  lat_serial_nophys  same without the physics term (what the PINN's physics adds)
                  lat_serial_op      lat_serial (frozen) -> EnsembleOperator (neuraloperator FNO, member-shared,
                                     ensemble-mean pooling), fair energy score + capped physics
                  lat_serial_op_nophys  same operator without the physics term
                  lat_serial_op_band    lat_serial_op with the output projected onto the kept DCT modes
                  lat_serial_op2        second operator version: trained on a calibration year the base
                                        never saw, 1->2->3 day rollout curriculum with BPTT, fair energy
                                        score + lambda_corr ||X_hat - X_tilde||^2 + capped physics,
                                        latitude padding, DCT band limit
                  lat_serial_op2_nophys  same without the physics term

    python scripts/run_step7_latent.py --out runs/step7 [--quick]
"""

import argparse
import copy
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm.coupling import PGCFM, estimate_scales  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset, kai_data_config  # noqa: E402
from pgcfm.latent import (DCTFiltered, DeterministicPINN, LatentConfig, LatentPGCFM, encoder_drift,  # noqa: E402
                          finetune_decoder, floors, member_structure, range_coverage, snapshot, train_serial,
                          train_stage1, train_stage2)
from pgcfm.ens_operator import EnsembleOperator, OperatorCorrected, train_operator  # noqa: E402
from pgcfm.projection import GradientFlowProjection, freeze  # noqa: E402
from pgcfm.train import run_phase, take  # noqa: E402
from scripts.make_balanced_era5 import main as make_balanced  # noqa: E402
from scripts.run_step4_pgcfm import KEEP, evaluate, member_residuals  # noqa: E402
from scripts.run_step6_projection import print_table  # noqa: E402
from scripts.train_pgcfm import DEFAULTS, build  # noqa: E402
from pgcfm import metrics  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/step7")
    p.add_argument("--n-lat", type=int, default=25)
    p.add_argument("--n-lon", type=int, default=48)
    p.add_argument("--train-years", type=int, nargs="+", default=[2001, 2002, 2003, 2004])
    p.add_argument("--train-days", type=int, default=90)
    p.add_argument("--iters-cfm", type=int, default=1200)
    p.add_argument("--iters-cfm-more", type=int, default=240, help="step 6 'cfm' arm extra iterations")
    p.add_argument("--k-lat", type=int, default=8)
    p.add_argument("--k-lon", type=int, default=16)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--enc-hidden", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--iters-stage1", type=int, default=1500)
    p.add_argument("--iters-stage2", type=int, default=2000)
    p.add_argument("--iters-dec-ft", type=int, default=200)
    p.add_argument("--iters-serial", type=int, default=600)
    p.add_argument("--iters-op", type=int, default=300)
    p.add_argument("--lambda-p", type=float, default=0.1)
    p.add_argument("--grad-ratio", type=float, default=0.5)
    p.add_argument("--arms", nargs="+", default=["cfm", "cfm_dct", "cfm_smooth", "pinn_det", "lat_frozen",
                                                  "lat_retrain", "lat_finetune", "lat_nophys", "lat_serial",
                                                  "lat_serial_nophys", "lat_serial_op", "lat_serial_op_nophys",
                                                  "lat_serial_op_band", "lat_serial_op2", "lat_serial_op2_nophys"])
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--n-starts", type=int, default=4)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--carry-highpass", action="store_true",
                   help="decoder output + (x_k - DCT filter(x_k)): unresolved scales persist (W1 fix)")
    p.add_argument("--iters-op2", type=int, default=200)
    p.add_argument("--lambda-corr", type=float, default=1.0)
    p.add_argument("--op2-curriculum", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.train_years, a.train_days, a.iters_cfm, a.iters_cfm_more = [2001], 12, 2, 1
        a.iters_stage1, a.iters_stage2, a.iters_dec_ft, a.iters_serial, a.iters_op, a.iters_op2 = 3, 3, 2, 2, 2, 2
        a.op2_curriculum = [1, 2]
        a.horizon, a.n_starts, a.members, a.steps, a.n_lat, a.n_lon = 2, 1, 2, 2, 17, 32
        a.k_lat, a.k_lon, a.latent = 4, 8, 8
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    root = os.path.join(a.out, "data")
    tr, te = os.path.join(root, "train"), os.path.join(root, "test")
    if not os.path.exists(os.path.join(tr, "stats", "global_std.npy")):  # same data as step 6
        make_balanced(["--out", tr, "--years", *map(str, a.train_years), "--days", str(a.train_days),
                       "--n-lat", str(a.n_lat), "--n-lon", str(a.n_lon), "--seed", "0"])
        test_days = a.horizon + 2 + 5 * a.n_starts
        make_balanced(["--out", te, "--years", "2020", "--days", str(test_days),
                       "--n-lat", str(a.n_lat), "--n-lon", str(a.n_lon), "--seed", "1000"])
    mean_p, std_p = os.path.join(tr, "stats", "global_mean.npy"), os.path.join(tr, "stats", "global_std.npy")
    dcfg = kai_data_config(mean_p, std_p)
    pair_ds = PGCFMSequenceDataset(dcfg, tr, horizon=1)
    test_ds = PGCFMSequenceDataset(dcfg, te, horizon=a.horizon, train=False)
    tpair_ds = PGCFMSequenceDataset(dcfg, te, horizon=1, train=False)
    loader = torch.utils.data.DataLoader(pair_ds, batch_size=8, shuffle=True, drop_last=True)
    test_batch = torch.utils.data.default_collate([test_ds[5 * i] for i in range(a.n_starts)])
    val_batches = take(torch.utils.data.DataLoader(tpair_ds, batch_size=8), 2)
    all_train = torch.utils.data.default_collate([pair_ds[i] for i in range(len(pair_ds))])
    # calibration period for the second operator version: never seen by the base chain
    cal = os.path.join(root, "calib")
    if not os.path.exists(os.path.join(cal, "2005.nc")):
        make_balanced(["--out", cal, "--years", "2005", "--days", str(a.train_days), "--n-lat", str(a.n_lat),
                       "--n-lon", str(a.n_lon), "--seed", "500"])
    cal_loader = torch.utils.data.DataLoader(
        PGCFMSequenceDataset(dcfg, cal, horizon=max(a.op2_curriculum)), batch_size=4, shuffle=True, drop_last=True)

    cfg = dict(DEFAULTS, model_channels=32, channel_mult=[1, 2, 2], num_res_blocks=1)
    torch.manual_seed(0)
    base = build(cfg, pair_ds.img_shape_x, pair_ds.img_shape_y, np.load(mean_p), np.load(std_p))
    base.guide.scales = estimate_scales(base.guide, take(loader, 10))
    freeze(base.guide)
    guide = base.guide
    res_path = os.path.join(a.out, "results.json")
    res = {}
    if os.path.exists(res_path):  # later invocations (--arms split) add to the same file
        with open(res_path) as f:
            res = json.load(f)
    res.update({"args_" + "_".join(a.arms): vars(a), "scales": vars(guide.scales), "n_train_pairs": len(pair_ds)})

    def save():
        with open(res_path, "w") as f:
            json.dump(res, f, indent=1)

    x_k = test_batch["x0"][:, None][..., KEEP, :, :]
    o = test_batch["targets"][..., KEEP, :, :]
    res["persistence_lead_rmse"] = metrics._wmean((x_k - o) ** 2).mean(dim=(0, 2)).sqrt().tolist()
    res["truth_lead_amp"] = metrics._wmean(o ** 2).mean(dim=(0, 2)).sqrt().tolist()
    res["truth_lead_residual"] = member_residuals(
        base, test_batch["x0"], test_batch["targets"][:, None], test_batch["static"], test_batch["doy"])

    lcfg = LatentConfig(k_lat=a.k_lat, k_lon=a.k_lon, latent=a.latent, enc_hidden=tuple(a.enc_hidden),
                        dec_hidden=tuple(reversed(a.enc_hidden)), carry_highpass=a.carry_highpass)
    n_lat, n_lon = pair_ds.img_shape_x, pair_ds.img_shape_y

    def score(name, transition, secs, extra=None):
        r = evaluate(PGCFM(transition, guide), test_batch, val_batches, a.members, a.steps)
        r["train_seconds"] = secs
        getattr(transition, "layout", lambda *_: None)(a.members, "repeat")
        r["structure"] = member_structure(transition, val_batches, a.members, a.steps)
        r["range_coverage"] = range_coverage(transition, val_batches, a.members, a.steps)
        r.update(extra or {})
        res[name] = r
        print(f"[{name}] {secs:.0f}s 1-day", json.dumps({k: round(v, 4) for k, v in r["one_step"].items()
                                                          if not k.startswith("res_obs")}), flush=True)
        print(f"[{name}] structure", json.dumps(r["structure"]), flush=True)
        save()

    arms = set(a.arms)
    # -- grid baselines (step 6) ---------------------------------------------
    if arms & {"cfm", "cfm_dct", "cfm_smooth"}:
        t = time.time()
        run_phase(base, "cfm", loader, a.iters_cfm, 2e-4, log_every=max(1, a.iters_cfm // 5))
        torch.manual_seed(1)
        run_phase(base, "cfm", loader, a.iters_cfm_more, 5e-5, log_every=max(1, a.iters_cfm_more // 2))
        secs = time.time() - t
        freeze(base.cfm)
        dct_ref = LatentPGCFM(lcfg, n_lat, n_lon).dct_x
        if "cfm" in arms:
            score("cfm", base.cfm, secs)
        if "cfm_dct" in arms:
            score("cfm_dct", DCTFiltered(base.cfm, dct_ref), 0.0)
        if "cfm_smooth" in arms:
            score("cfm_smooth", GradientFlowProjection(base.cfm, guide, control="smooth"), 0.0)

    # -- stage 1 ----------------------------------------------------------------
    def stage1(lambda_p):
        ck = os.path.join(a.out, f"stage1_lp{lambda_p:g}.pt")
        torch.manual_seed(2)
        m = LatentPGCFM(lcfg, n_lat, n_lon)
        if os.path.exists(ck):  # reuse across invocations (--arms split)
            st = torch.load(ck)
            m.load_state_dict(st["latent"])
            freeze(m.pinn)
            return m, st["secs"], st["log"]
        m.fit_dct(all_train["x0"], all_train["static"])
        t = time.time()
        hist = train_stage1(m, guide, loader, a.iters_stage1, lambda_p=lambda_p, grad_ratio=a.grad_ratio,
                            log_every=max(1, a.iters_stage1 // 10))
        secs = time.time() - t
        m.fit_residual_scale(take(torch.utils.data.DataLoader(pair_ds, batch_size=32), 100))
        last = hist[-max(1, len(hist) // 10):]
        lg = {k: float(np.mean([h[k] for h in last if k in h])) for k in hist[-1]}
        torch.save({"latent": m.state_dict(), "secs": secs, "log": lg}, ck)
        return m, secs, lg

    latent_arms = arms & {"pinn_det", "lat_frozen", "lat_retrain", "lat_finetune", "lat_serial", "lat_serial_nophys",
                          "lat_serial_op", "lat_serial_op_nophys", "lat_serial_op_band",
                          "lat_serial_op2", "lat_serial_op2_nophys"}
    if latent_arms:
        s1, s1_secs, s1_log = stage1(a.lambda_p)
        res["stage1_seconds"], res["stage1_log_last"] = s1_secs, s1_log
        res["floors"] = floors(s1, val_batches)
        res["floors_train"] = floors(s1, take(loader, 4))
        print("floors (test)", json.dumps({k: round(v, 4) for k, v in res["floors"].items()}), flush=True)
        print("floors (train)", json.dumps({k: round(v, 4) for k, v in res["floors_train"].items()}), flush=True)
        save()
        if "pinn_det" in arms:
            score("pinn_det", DeterministicPINN(s1), s1_secs)

        frozen = None
        for name in ("lat_frozen", "lat_retrain"):
            if name not in arms and not (name == "lat_frozen" and "lat_finetune" in arms):
                continue
            torch.manual_seed(3)
            m = copy.deepcopy(s1)
            t = time.time()
            hist = train_stage2(m, loader, a.iters_stage2, mode=name.split("_")[1],
                                log_every=max(1, a.iters_stage2 // 10))
            secs = time.time() - t
            extra = {"stage2_log_last": {k: float(np.mean([h[k] for h in hist[-50:]])) for k in hist[-1]},
                     "floors_after": floors(m, val_batches), "drift": encoder_drift(m, s1, val_batches)}
            score(name, m, secs, extra)
            if name == "lat_frozen":
                frozen = (m, secs)
        if "lat_finetune" in arms and frozen is not None:
            torch.manual_seed(4)
            m = copy.deepcopy(frozen[0])
            ref = snapshot(m)
            t = time.time()
            finetune_decoder(m, loader, a.iters_dec_ft, steps=a.steps, log_every=max(1, a.iters_dec_ft // 4))
            secs = frozen[1] + time.time() - t
            score("lat_finetune", m, secs, {"floors_after": floors(m, val_batches),
                                             "drift": encoder_drift(m, ref, val_batches)})

        serial = None
        op_arms = {"lat_serial_op", "lat_serial_op_nophys", "lat_serial_op_band", "lat_serial_op2",
                   "lat_serial_op2_nophys"}
        serial_ck = os.path.join(a.out, "serial.pt")
        if "lat_serial" not in arms and arms & op_arms and os.path.exists(serial_ck):
            m = LatentPGCFM(dataclasses.replace(lcfg, coupling="serial"), n_lat, n_lon)
            st_ = torch.load(serial_ck)
            m.load_state_dict(st_["latent"])
            freeze(m)
            serial = (m, st_["secs"])
        for name in ("lat_serial", "lat_serial_nophys"):
            if name not in arms and not (name == "lat_serial" and arms & op_arms and serial is None):
                continue
            torch.manual_seed(3)
            m = LatentPGCFM(dataclasses.replace(lcfg, coupling="serial"), n_lat, n_lon)
            m.load_state_dict(s1.state_dict(), strict=False)  # encoder/decoder/DCT from stage 1
            m.fit_residual_scale(take(torch.utils.data.DataLoader(pair_ds, batch_size=32), 100))
            t = time.time()
            hist = train_serial(m, guide, loader, a.iters_serial, grad_ratio=a.grad_ratio,
                                lambda_p=0.0 if name.endswith("nophys") else a.lambda_p,
                                log_every=max(1, a.iters_serial // 10))
            secs = s1_secs + time.time() - t
            score(name, m, secs,
                  {"stage2_log_last": {k: float(np.mean([h[k] for h in hist[-50:] if k in h])) for k in hist[-1]}})
            if name == "lat_serial":
                serial = (m, secs)
                torch.save({"latent": m.state_dict(), "secs": secs}, serial_ck)
        for name in ("lat_serial_op", "lat_serial_op_nophys", "lat_serial_op_band"):
            if name not in arms or serial is None:
                continue
            torch.manual_seed(5)
            w = OperatorCorrected(serial[0], EnsembleOperator(n_modes=(min(12, n_lat // 2), min(12, n_lon // 2)),
                                                              hidden=16 if a.quick else 64),
                                  band=serial[0].dct_x if name.endswith("band") else None)
            t = time.time()
            hist = train_operator(w, guide, loader, a.iters_op, steps=a.steps, grad_ratio=a.grad_ratio,
                                  lambda_p=0.0 if name.endswith("nophys") else a.lambda_p,
                                  log_every=max(1, a.iters_op // 6))
            secs = serial[1] + time.time() - t
            score(name, w, secs,
                  {"op_log_last": {k: float(np.mean([h[k] for h in hist[-30:] if k in h])) for k in hist[-1]}})
        # second version (owner's possibility_operator + design note): calibration period, rollout
        # curriculum with BPTT, correction-size penalty, latitude padding, DCT band limit
        for name in ("lat_serial_op2", "lat_serial_op2_nophys"):
            if name not in arms or serial is None:
                continue
            torch.manual_seed(5)
            w = OperatorCorrected(serial[0], EnsembleOperator(n_modes=(min(12, n_lat // 2), min(12, n_lon // 2)),
                                                              hidden=16 if a.quick else 64, lat_padding=0.25),
                                  band=serial[0].dct_x)
            t = time.time()
            hist = train_operator(w, guide, cal_loader, a.iters_op2, steps=4, grad_ratio=a.grad_ratio,
                                  lambda_p=0.0 if name.endswith("nophys") else a.lambda_p,
                                  curriculum=tuple(a.op2_curriculum), lambda_corr=a.lambda_corr,
                                  log_every=max(1, a.iters_op2 // 6))
            secs = serial[1] + time.time() - t
            score(name, w, secs,
                  {"op_log_last": {k: float(np.mean([h[k] for h in hist[-30:] if k in h])) for k in hist[-1]}})

    if "lat_nophys" in arms:
        s0, secs0, log0 = stage1(0.0)
        res["floors_nophys"] = floors(s0, val_batches)
        torch.manual_seed(3)
        t = time.time()
        train_stage2(s0, loader, a.iters_stage2, mode="frozen", log_every=max(1, a.iters_stage2 // 10))
        score("lat_nophys", s0, secs0 + time.time() - t, {"stage1_log_last": log0})

    names = [n for n, v in res.items() if isinstance(v, dict) and "lead_rmse" in v]
    print_table(res, names, a.horizon)
    return res


if __name__ == "__main__":
    main()
