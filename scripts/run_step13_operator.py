"""Step 13, stage 3: frozen FM + frozen PINN ensembles -> neural operator at each lead -> deterministic
forecast, against the classical regression that beat every learned corrector in step 10.

    input    the stage-2 ensembles saved by run_step13_seqfm.py --save-ens (5 ClimODE vars, 24 h
             multiples, members after the frozen PINN): the validation year trains the operator
             (first ICs) and picks its checkpoint (last ``--n-valid`` ICs); the test years score it
    operator the owner's EnsembleToDeterministic (pgcfm/ens_to_det.py, official neuraloperator FNO:
             shared member encoder, member pooling, global FNO, zero head = the ensemble mean), one
             operator for all leads, context = x(t) and the lead
    arms     mean (ensemble mean), mbmx (a + b mu + c x0 per lead and variable, step 10), operator,
             operator_mean_only (the same model fed the mean only: does the ensemble add anything?)

    python scripts/run_step13_operator.py --ens runs/step13/seqfm_dct_pinn --era5 /home/claude/era5/era5_data \
        --out runs/step13 [--quick]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm import contract  # noqa: E402
from pgcfm.ens_to_det import EnsembleToDeterministic, fit  # noqa: E402
from scripts.run_step10_deterministic import PersistenceMBM  # noqa: E402
from scripts.run_step11_climode import VARS, climode_minmax, climode_scores, climode_truth  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402


def load(prefix, year):
    z = np.load(f"{prefix}_ens_{year}.npz")
    E = z["E_minus_x0"].astype(np.float32) + z["X0"][:, None, None]
    return E, z["O"], z["X0"], z["ics"], z["leads"]


def samples(E, O, X0, leads, m, s):
    """Normalized (n_ic * L) operator samples: members (n, M, 5, H, W), target, context [x0, lead]."""
    n, M, L = E.shape[:3]
    nz = lambda q: (q - m) / s  # noqa: E731
    mem = torch.from_numpy(nz(E).transpose(0, 2, 1, 3, 4, 5).reshape(n * L, M, *E.shape[3:]))
    tgt = torch.from_numpy(nz(O).reshape(n * L, *O.shape[2:]))
    x0 = np.repeat(nz(X0)[:, None], L, 1).reshape(n * L, *X0.shape[1:])
    lead = np.broadcast_to((np.asarray(leads, np.float32) / 144.0)[None, :, None, None, None],
                           (n, L, 1) + X0.shape[-2:]).reshape(n * L, 1, *X0.shape[-2:])
    return mem, tgt, torch.from_numpy(np.concatenate([x0, lead], 1).astype(np.float32))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ens", required=True, help="prefix of the saved ensembles (<prefix>_ens_<year>.npz)")
    p.add_argument("--era5", required=True)
    p.add_argument("--out", default="runs/step13")
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--n-valid", type=int, default=33)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.n_valid, a.width, a.modes, a.epochs = 1, 8, 3, 2
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    lat, lon, _, _ = load_static(a.era5)
    contract.register_latitudes(lat)
    area = torch.from_numpy(np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (len(lat), len(lon))).astype(np.float32).copy())
    tv = climode_truth(a.era5, a.val_year)
    m, s = tv.mean((0, 2, 3))[:, None, None], tv.std((0, 2, 3))[:, None, None]
    E, O, X0, _, leads = load(a.ens, a.val_year)
    L = len(leads)
    tr, va = slice(0, len(E) - a.n_valid), slice(len(E) - a.n_valid, len(E))
    # classical control fitted on the whole validation year (it has no checkpoint to pick)
    mbmx = PersistenceMBM(5, L).fit(torch.from_numpy(E), torch.from_numpy(O), torch.from_numpy(X0))
    res = {"args": vars(a), "mbmx_coef": mbmx.coef.tolist()}
    models = {}
    for arm, use_ens in (("operator", True), ("operator_mean_only", False)):
        torch.manual_seed(0)
        op = EnsembleToDeterministic(5, context_channels=6, width=a.width, modes=a.modes)
        info = fit(op, samples(E[tr], O[tr], X0[tr], leads, m, s), samples(E[va], O[va], X0[va], leads, m, s),
                   area, epochs=a.epochs, use_ensemble=use_ens)
        res[arm] = {"best_val_mse": info["best_val_mse"], "val_history": info["val_mse_history"]}
        models[arm] = (op, use_ens)
        print(f"[{arm}] best val mse {info['best_val_mse']:.4f} (epoch {int(np.argmin(info['val_mse_history']))})",
              flush=True)
    mx, mn = climode_minmax(a.era5)
    sc = {k: {"rmse": [], "acc": []} for k in ("mean", "mbmx", "operator", "operator_mean_only", "persistence")}
    for y in a.test_years:
        E, O, X0, _, leads = load(a.ens, y)
        clim = climode_truth(a.era5, y).mean(0)
        n = len(E)
        preds = {"mean": E.mean(1), "persistence": np.repeat(X0[:, None], L, 1),
                 "mbmx": mbmx(torch.from_numpy(E), torch.from_numpy(X0)).mean(1).numpy()}
        mem, _, ctx = samples(E, O, X0, leads, m, s)
        with torch.no_grad():
            for arm, (op, use_ens) in models.items():
                x = mem if use_ens else mem.mean(1, keepdim=True).expand_as(mem)
                out = torch.cat([op(x[i:i + 64], ctx[i:i + 64]) for i in range(0, len(x), 64)]).numpy()
                preds[arm] = out.reshape(n, L, *out.shape[1:]) * s + m
        for arm, P in preds.items():
            q = climode_scores(P, O, np.full_like(P, 1e-6), clim, lat, mx, mn)
            sc[arm]["rmse"].append(q["rmse"]), sc[arm]["acc"].append(q["acc"])
    res["test"] = {arm: {k: {str(h): {v: float(np.concatenate(r[k])[:, j, i].mean()) for i, v in enumerate(VARS)}
                             for j, h in enumerate(leads)} for k in r} for arm, r in sc.items()}
    for arm, r in res["test"].items():
        print(f"[{arm:18s}] z500 RMSE " + " ".join(f"{h}:{r['rmse'][h]['z']:.0f}" for h in r["rmse"])
              + " | t850 " + " ".join(f"{r['rmse'][h]['t']:.2f}" for h in r["rmse"]), flush=True)
    tag = os.path.basename(a.ens) + ("_quick" if a.quick else "")
    with open(os.path.join(a.out, f"operator_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
