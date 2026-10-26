"""Step 10 experiment: comparison models from the owner's references against the PG-CFM chain
(owner, 2026-10-24: U-Cast paper + sipposip/ensemble-neural-network-weather-forecasts + cbuelt/dduq).

Same synthetic data, training years (2001-2004, the step 7 chain's), validation year (2006, every
tuned perturbation size / dropout rate / calibration is chosen there by empirical CRPS, as Scher and
Messori choose theirs with properscoring) and test sets (2021 test16, 2020 test) as step 9. 8 members,
10-day rollouts. The PG-CFM rows are read from ``runs/step9/results.json`` (same test sets).

Arms (``pgcfm/baselines.py``; one U-Cast stage 1 net = the deterministic backbone):
    det          U-Cast stage 1 alone (weighted MAE, dropout off): a single forecast (CRPS = MAE)
    ic_gauss     Scher & Messori random initial perturbations (re-standardized; = dduq Gaussian noise)
    ic_svd       Scher & Messori singular-vector perturbations (16 leading vectors of the 1-day Jacobian)
    ic_rfp       dduq random field perturbations (same-season state differences / total energy)
    mc_dropout   Scher & Messori dropout ensemble of the stage 1 net, rate tuned
    multitrain   Scher & Messori "multitrain": 8 stage 1 nets with different seeds
    ucast        U-Cast stage 2: fair CRPS fine-tuning with M = 2 MC dropout members, then 8 dropout members
    ucast_de     U-Cast stage 3 (DeepEns): 4 stage 2 runs (seeds) x 2 dropout members
    ucast_long, ucast_de_long  the same with a 4x longer stage 2 (sensitivity: 2.7 -> 11 epochs)
    drn          dduq DRN on the stage 1 forecast (Gaussian per point -> 8 quantile members, marginals only)
Calibrations (step 9, fitted on 2006 per arm): raw, mos_ck (MBM), mos_ck_tan (MBM + hydrostatic tangent).

    python scripts/run_step10_baselines.py --out runs/step10 [--arms ...] [--quick]
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import baselines as bl  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset, kai_data_config  # noqa: E402
from pgcfm.physics.primitive import ResidualScales  # noqa: E402
from pgcfm.projection import freeze  # noqa: E402
from pgcfm.spread import KEEP, DCTBands, ensemble_scores  # noqa: E402
from pgcfm.structure import (LinearMeanCalibrator, SpreadCalibrator, lead_crps, structure_scores,  # noqa: E402
                             tangent_calibrated)
from pgcfm.train import _cycle  # noqa: E402
from scripts.run_step4_pgcfm import member_residuals  # noqa: E402
from scripts.train_pgcfm import DEFAULTS, build  # noqa: E402

ARMS = ("det", "ic_gauss", "ic_svd", "ic_rfp", "mc_dropout", "ucast", "ucast_de", "drn", "multitrain",
        "ucast_long", "ucast_de_long")
CALS = ("raw", "mos_ck", "mos_ck_tan")
GRIDS = {"ic_gauss": (0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0),
         "ic_svd": (1.0, 3.0, 10.0, 30.0, 100.0, 300.0),
         "ic_rfp": (0.03, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0),
         "mc_dropout": (0.01, 0.03, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7)}
PGCFM_ROWS = ("base/raw", "base/loop", "base/mos_ck", "base/mos_ck_tan", "base_noQ/mos_ck_tan")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/step10")
    p.add_argument("--step7", default="runs/step7")
    p.add_argument("--step9", default="runs/step9")
    p.add_argument("--test16", default="runs/step8/data/test16")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--n-test16", type=int, default=16)
    p.add_argument("--iters1", type=int, default=1500, help="U-Cast stage 1 (upstream 100 epochs)")
    p.add_argument("--iters2", type=int, default=120, help="U-Cast stage 2 (upstream 8 epochs = 8%% of stage 1)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n-svs", type=int, default=16)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.members, a.horizon, a.n_val, a.n_test16, a.iters1, a.iters2, a.n_svs = 4, 2, 4, 2, 4, 2, 2
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    M = a.members

    root7 = os.path.join(a.step7, "data")
    tr, te = os.path.join(root7, "train"), os.path.join(root7, "test")
    mean_p, std_p = os.path.join(tr, "stats", "global_mean.npy"), os.path.join(tr, "stats", "global_std.npy")
    dcfg = kai_data_config(mean_p, std_p)
    pair_ds = PGCFMSequenceDataset(dcfg, tr, horizon=1)
    n_lat, n_lon = pair_ds.img_shape_x, pair_ds.img_shape_y
    loader = torch.utils.data.DataLoader(pair_ds, batch_size=a.batch, shuffle=True, drop_last=True)
    va = os.path.join(a.step9, "data", "valid")

    def starts(path, n):
        ds = PGCFMSequenceDataset(dcfg, path, horizon=a.horizon, train=False)
        return torch.utils.data.default_collate([ds[5 * i] for i in range(min(n, len(ds) // 5))])

    sets = {"test": starts(te, 4), "test16": starts(a.test16, a.n_test16)}
    val = starts(va, a.n_val)

    res9 = json.load(open(os.path.join(a.step9, "results.json")))
    res7 = json.load(open(os.path.join(a.step7, "results.json")))
    a7 = next(v for k, v in res7.items() if k.startswith("args_"))
    base = build(dict(DEFAULTS, model_channels=32, channel_mult=[1, 2, 2], num_res_blocks=1), n_lat, n_lon,
                 np.load(mean_p), np.load(std_p))
    guide = base.guide
    guide.scales = ResidualScales(**res7["scales"])
    freeze(guide)
    gm = SimpleNamespace(guide=guide)
    bands = DCTBands(n_lat, n_lon, a7.get("k_lat", 8), a7.get("k_lon", 16))

    res_path = os.path.join(a.out, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    res["args"] = vars(a)
    for k in PGCFM_ROWS:
        if k in res9:
            res[f"pgcfm:{k}"] = {sn: res9[k][sn] for sn in ("test", "test16")}

    def save():
        with open(res_path + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(res_path + ".tmp", res_path)

    def mean_crps(ens, batch):
        e, o = ens.transpose(0, 1)[..., KEEP, :, :], batch["targets"][..., KEEP, :, :]
        return float(np.mean(lead_crps(e, o)))

    def scores(ens, batch):
        e, o = ens.transpose(0, 1)[..., KEEP, :, :], batch["targets"][..., KEEP, :, :]
        r = ensemble_scores(e, o, bands)
        r["lead_crps_emp"] = lead_crps(e, o)
        r.update(structure_scores(e, o, bands))
        with torch.no_grad():
            r["lead_member_residual"] = member_residuals(gm, batch["x0"], ens, batch["static"], batch["doy"])
        return r

    def report(name, r):
        s16 = r["test16"]
        K = len(s16["lead_crps_emp"])
        pick = lambda v: " / ".join(f"{v[i]:.3f}" for i in sorted({0, min(4, K - 1), K - 1}))  # noqa: E731
        print(f"[{name}] test16 emp CRPS {pick(s16['lead_crps_emp'])} mean {np.mean(s16['lead_crps_emp']):.3f} "
              f"rmse {pick(s16['lead_rmse'])} SSR {pick(s16['lead_ssr'])} xcorr {pick(s16['lead_xcorr_err'])} "
              f"hyd {pick(s16['lead_member_residual']['hyd'])} | test mean {np.mean(r['test']['lead_crps_emp']):.3f}",
              flush=True)

    # -- U-Cast nets ------------------------------------------------------------------------
    rstd = bl.residual_std(loader)

    def stage1(seed):
        path = os.path.join(a.out, f"ucast_det_s{seed}.pt")
        step = bl.UCastStep(n_lat, n_lon, rstd)
        if os.path.exists(path):
            step.load_state_dict(torch.load(path)["step"])
            return step.eval()
        torch.manual_seed(seed)
        step = bl.UCastStep(n_lat, n_lon, rstd)
        t = time.time()
        hist = bl.train_ucast(step, _cycle(loader), a.iters1, "wmae", lr=3e-4, weight_decay=0.1, muon_lr=3e-3,
                              muon_wd=0.03, ckpt=path + ".ckpt", log_every=max(1, a.iters1 // 5))
        torch.save({"step": step.state_dict(), "hist": hist, "seconds": time.time() - t}, path)
        return step.eval()

    def stage2(seed, iters=None):
        iters = iters or a.iters2
        path = os.path.join(a.out, f"ucast_prob_s{seed}" + ("" if iters == a.iters2 else f"_i{iters}") + ".pt")
        step = bl.UCastStep(n_lat, n_lon, rstd, loss_function="wcrps")
        if os.path.exists(path):
            step.load_state_dict(torch.load(path)["step"])
            return step.eval()
        step.load_state_dict(stage1(0).state_dict())
        torch.manual_seed(seed)
        t = time.time()
        hist = bl.train_ucast(step, _cycle(loader), iters, "wcrps", lr=7e-5, weight_decay=0.1, muon_lr=7e-3,
                              muon_wd=0.0, ckpt=path + ".ckpt", log_every=max(1, iters // 3))
        torch.save({"step": step.state_dict(), "hist": hist, "seconds": time.time() - t}, path)
        return step.eval()

    def det_forecast(step, batch):
        return bl.member_rollout([step], batch["x0"][:, None], batch["static"], batch["doy"], dropout_p=0.0)

    svec_cache = {}

    def svecs(step, name, batch):
        if name not in svec_cache:
            path = os.path.join(a.out, f"svecs_{name}.pt")
            if os.path.exists(path):
                svec_cache[name] = torch.load(path)
            else:
                t = time.time()
                svec_cache[name] = torch.stack([bl.jacobian_svecs(step, batch["x0"][i], batch["static"][i],
                                                                  batch["doy"][i, 1:2], a.n_svs, seed=i)
                                                for i in range(len(batch["x0"]))])
                torch.save(svec_cache[name], path)
                print(f"  [svd] {name}: {time.time() - t:.0f} s", flush=True)
        return svec_cache[name]

    rfp = None

    def generator(name):
        """name -> (ensemble function of (batch, set name, tuning value, seed), tuning grid)."""
        nonlocal rfp
        if name == "ic_gauss":
            s = stage1(0)
            return (lambda b, sn, v, seed: bl.member_rollout(
                [s], bl.scher_rand(b["x0"], M, v, torch.Generator().manual_seed(seed)), b["static"], b["doy"],
                dropout_p=0.0)), GRIDS[name]
        if name == "ic_svd":
            s = stage1(0)
            return (lambda b, sn, v, seed: bl.member_rollout(
                [s], bl.scher_svd(b["x0"], svecs(s, sn, b), M, v, torch.Generator().manual_seed(seed)),
                b["static"], b["doy"], dropout_p=0.0)), GRIDS[name]
        if name == "ic_rfp":
            s = stage1(0)
            if rfp is None:
                full = torch.utils.data.default_collate([pair_ds[i] for i in range(len(pair_ds))])
                doys = full["doy"][:, 0].numpy()
                years = np.cumsum(np.r_[0, np.diff(doys) < 0])  # one file (year) after another
                rfp = bl.RandomFieldPerturber(full["x0"], years, doys,
                                              full["static"], np.load(mean_p), np.load(std_p))
                rng = np.random.default_rng(0)
                ref = np.median([rfp.unit(int(d), rng).pow(2).mean().sqrt().item() for d in rng.choice(doys, 30)])
                rfp.ref = 1.0 / ref  # alpha in units of a raw state difference's rms
            return (lambda b, sn, v, seed: bl.member_rollout(
                [s], rfp(b["x0"], b["doy"][:, 0], M, v * rfp.ref, seed), b["static"], b["doy"], dropout_p=0.0)), \
                GRIDS[name]
        if name == "mc_dropout":
            s = stage1(0)
            return (lambda b, sn, v, seed: (torch.manual_seed(seed), bl.member_rollout(
                [s], b["x0"][:, None].repeat(1, M, 1, 1, 1), b["static"], b["doy"], dropout_p=v))[1]), GRIDS[name]
        if name == "multitrain":
            nets = [stage1(sd) for sd in range(M)]
            return (lambda b, sn, v, seed: bl.member_rollout(
                nets, b["x0"][:, None].repeat(1, M, 1, 1, 1), b["static"], b["doy"], dropout_p=0.0)), (None,)
        if name.startswith("ucast"):
            it2 = 4 * a.iters2 if name.endswith("_long") else a.iters2
            nets = [stage2(100, it2)] if name in ("ucast", "ucast_long") else [stage2(100 + i, it2) for i in range(4)]
            return (lambda b, sn, v, seed: (torch.manual_seed(seed), bl.member_rollout(
                nets, b["x0"][:, None].repeat(1, M, 1, 1, 1), b["static"], b["doy"], dropout_p=0.1))[1]), (None,)
        raise ValueError(name)

    for name in a.arms:
        if name == "det":
            if "det/raw" in res:
                continue
            s = stage1(0)
            r = {}
            for sn, b in sets.items():
                f = det_forecast(s, b)
                e, o = f.transpose(0, 1)[..., KEEP, :, :], b["targets"][..., KEEP, :, :]
                r[sn] = {"lead_crps_emp": lead_crps(e, o),
                         "lead_rmse": ensemble_scores(e.repeat(2, 1, 1, 1, 1, 1), o, bands)["lead_rmse"],
                         "lead_member_residual": member_residuals(gm, b["x0"], f, b["static"], b["doy"])}
            res["det/raw"] = r
            print(f"[det/raw] test16 MAE {np.mean(r['test16']['lead_crps_emp']):.3f} "
                  f"rmse {np.mean(r['test16']['lead_rmse']):.3f}", flush=True)
            save()
            continue
        if name == "drn":
            if "drn/raw" in res:
                continue
            s = stage1(0)
            fv = det_forecast(s, val)[:, 0]
            models = bl.fit_drn(fv, val["targets"], n_valid=max(1, len(fv) // 4))
            r = {sn: scores(bl.drn_members(models, det_forecast(s, b)[:, 0], M), b) for sn, b in sets.items()}
            res["drn/raw"] = r
            report("drn/raw", r)
            save()
            continue
        todo = [c for c in CALS if f"{name}/{c}" not in res]
        if not todo:
            continue
        t = time.time()
        gen, grid = generator(name)
        if grid != (None,):
            tune = {v: mean_crps(gen(val, "val", v, 3), val) for v in grid}
            best = min(tune, key=tune.get)
            print(f"[fit] {name} " + " ".join(f"{v:g}:{c:.3f}" for v, c in tune.items()) + f" -> {best:g}", flush=True)
        else:
            tune, best = {}, None
        raw = {sn: gen(b, sn, best, 1) for sn, b in sets.items()}
        vens = gen(val, "val", best, 3)
        for c in todo:
            extra = {"tuned": best, "tune_grid": {f"{k:g}": v for k, v in tune.items()}}
            if c == "raw":
                ens = raw
            else:
                mos = LinearMeanCalibrator(vens.shape[3], vens.shape[2]).fit_validation(vens, val["targets"])
                calib = SpreadCalibrator(vens.shape[3], per_lead=True, leads=vens.shape[2])
                calib.fit_validation(mos(vens), val["targets"])
                ens = {sn: calib(mos(e)) for sn, e in raw.items()}
                if c == "mos_ck_tan":
                    ens = {sn: tangent_calibrated(raw[sn], e, guide, sets[sn]["x0"], sets[sn]["static"],
                                                  sets[sn]["doy"]) for sn, e in ens.items()}
            r = {sn: scores(ens[sn], b) for sn, b in sets.items()}
            r.update(extra)
            res[f"{name}/{c}"] = r
            report(f"{name}/{c}", r)
            save()
        res[f"{name}/info"] = {"eval_seconds": time.time() - t}
        save()
    return res


if __name__ == "__main__":
    main()
