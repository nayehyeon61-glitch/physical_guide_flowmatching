"""Step 13, the owner's third conclusion (2026-10-26), test 1: a diagnostic-closure operator on top of the
reported mean.

The operator's one win in 4.2 was u10/v10 (144 h 3.68 -> 3.46 against a + b mu + c x(t)), fields the KAI model
diagnoses from u/v1000. The text's claim: a field-to-field nonlinear map with a controlled operator norm is the
operator's own job, so an operator that only corrects the diagnostic variables, starting from the regression
mean, should keep that win and carry it to t2m.

    base     mu* = a + b mu_reg + c x(t) (PersistenceMBM on the regression ensemble, the regm mean of 4.2)
    model    y = mu* + mask * correction: the correction only on ``--diag-vars`` (default t2m, u10, v10), zero
             head (every arm starts at mu*), one model for all leads, MSE to the truth on the validation year
             (the last ``--n-valid`` ICs pick the checkpoint), scored once on the test years
    arms     diag_op<B>  the owner's EnsembleToDeterministic (official neuraloperator FNO) with the hard
                         operator-norm projection B after every step (pgcfm/ens_relation.py project_fno_)
             diag_op     the same without the bound
             diag_cnn    ClimODE's Climate_ResNet_2D on [member mean, member std, context]: the control (is
                         the closure an operator's job, or does a convolution do it as well?)
             all_op<B>   the bounded operator correcting all five variables (does mu* gain on z and t too?)
    context  every arm sees the same fields: x(t), the lead, mu*, and the static fields of ClimODE (position,
             orography, land-sea mask)
    output   <out>/diag_<ens name>.json (RMSE / ACC per arm, lead and variable, validation history);
             ``--save-mean`` also writes each test year's corrected mean of the given arms to
             <out>/diag_<arm>_mean_<year>.npy, and ``--shift`` writes saved ensembles moved onto each of
             those means (members + mu_new - mu*, so the spread is unchanged) as <shift>_<arm>_ens_<year>.npz

    python scripts/run_step13_diag.py --ens runs/step13/seqregress_pinn --era5 /home/claude/era5/era5_data \
        --out runs/step13 [--quick]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "climode"))
from model_function import Climate_ResNet_2D  # noqa: E402

from pgcfm.core import static_features  # noqa: E402
from pgcfm.ens_relation import fno_norms, project_fno_  # noqa: E402
from pgcfm.ens_to_det import EnsembleToDeterministic, channel_mse, fit  # noqa: E402
from scripts.run_step10_deterministic import PersistenceMBM  # noqa: E402
from scripts.run_step11_climode import VARS, climode_minmax, climode_scores, climode_truth  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402
from scripts.run_step13_operator import load, samples  # noqa: E402

C = len(VARS)


class CNNClosure(nn.Module):
    """The control: Climate_ResNet_2D on [member mean, member std, context], zero head (output = the mean)."""

    def __init__(self, context_channels: int, layers=(3, 2, 2), hidden=(64, 32, 32)):
        super().__init__()
        self.net = Climate_ResNet_2D(2 * C + context_channels, list(layers), list(hidden))
        self.head = nn.Conv2d(hidden[-1], C, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, members, context):
        mean = members.mean(1)
        return mean + self.head(self.net(torch.cat([mean, members.std(1), context], 1)))


class OnBase(nn.Module):
    """y = mu* + mask * (model(members, context) - member mean); mu* is context[:, base]."""

    def __init__(self, model: nn.Module, base: slice, mask: torch.Tensor):
        super().__init__()
        self.model, self.base = model, base
        self.register_buffer("mask", mask.view(1, C, 1, 1))

    def forward(self, members, context):
        return context[:, self.base] + self.mask * (self.model(members, context) - members.mean(1))


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
    p.add_argument("--bound", type=float, default=1.0)
    p.add_argument("--diag-vars", nargs="+", default=["t2m", "u10", "v10"])
    p.add_argument("--arms", nargs="+", default=["diag_op_b", "diag_op", "diag_cnn", "all_op_b"],
                   help="_b = with the norm bound --bound")
    p.add_argument("--save-mean", nargs="*", default=[], help="arms whose corrected test mean is saved (and used by --shift)")
    p.add_argument("--shift", nargs="*", default=[], help="saved ensembles (prefixes) to move onto each saved mean")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--eval-only", action="store_true", help="load the saved diag_<arm>_<tag>.pt instead of training")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.n_valid, a.width, a.modes, a.epochs = 1, 8, 3, 2
    torch.set_num_threads(a.threads)
    lat, lon, oro, lsm = load_static(a.era5)
    area = torch.from_numpy(np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (len(lat), len(lon))).astype(np.float32).copy())
    static = static_features(lat, lon, oro, lsm)
    tv = climode_truth(a.era5, a.val_year)
    m, s = tv.mean((0, 2, 3))[:, None, None], tv.std((0, 2, 3))[:, None, None]
    E, O, X0, _, leads = load(a.ens, a.val_year)
    if a.quick:
        E, O, X0 = E[:8], O[:8], X0[:8]
    L = len(leads)
    mbmx = PersistenceMBM(C, L).fit(torch.from_numpy(E), torch.from_numpy(O), torch.from_numpy(X0))

    def data(E, O, X0):
        mem, tgt, ctx = samples(E, O, X0, leads, m, s)  # ctx = [x0, lead]
        mu = mbmx(torch.from_numpy(E), torch.from_numpy(X0)).mean(1).numpy()  # (n, L, C, H, W)
        mu_n = torch.from_numpy(((mu - m) / s).reshape(len(E) * L, C, *mu.shape[-2:]).astype(np.float32))
        return mem, tgt, torch.cat([ctx, mu_n, static.expand(len(mu_n), -1, -1, -1)], 1), mu

    base = slice(C + 1, 2 * C + 1)
    n_ctx = C + 1 + C + static.shape[0]
    tr, va = slice(0, len(E) - a.n_valid), slice(len(E) - a.n_valid, len(E))
    dtr, dva = data(E[tr], O[tr], X0[tr])[:3], data(E[va], O[va], X0[va])[:3]
    res = {"args": vars(a)}
    tag = os.path.basename(a.ens) + ("_quick" if a.quick else "")
    models = {}
    for arm in a.arms:
        bounded = arm.endswith("_b")
        name = arm[:-2] + f"{a.bound:g}" if bounded else arm
        diag = [VARS.index(v) for v in (VARS if arm.startswith("all") else a.diag_vars)]
        mask = torch.zeros(C)
        mask[diag] = 1.0
        torch.manual_seed(0)
        inner = CNNClosure(n_ctx) if "cnn" in arm else \
            EnsembleToDeterministic(C, context_channels=n_ctx, width=a.width, modes=a.modes)
        model = OnBase(inner, base, mask)
        post = None
        if bounded:
            project_fno_(inner, a.bound)
            post = lambda mdl, b=a.bound: project_fno_(mdl, b)  # noqa: E731
        ck = os.path.join(a.out, f"diag_{name}_{tag}.pt")
        if a.eval_only:
            model.load_state_dict(torch.load(ck, weights_only=False)["state_dict"])
            model.eval()
            with torch.no_grad():
                va_mse = float(channel_mse(model(dva[0], dva[2]), dva[1], area).mean())
            info = {"best_val_mse": va_mse, "val_mse_history": [va_mse]}
        else:
            info = fit(model, dtr, dva, area, epochs=a.epochs, post_step=post)
        res[name] = {"best_val_mse": info["best_val_mse"], "val_history": info["val_mse_history"],
                     "vars": [VARS[i] for i in diag], "n_params": sum(q.numel() for q in inner.parameters())}
        if "cnn" not in arm:
            res[name]["fno_norms"] = fno_norms(inner)
        if not a.eval_only:
            torch.save({"state_dict": model.state_dict(), "norm_mean": m, "norm_std": s, "arm": arm,
                        "bound": a.bound if bounded else None}, ck)
        print(f"[{name}] best val mse {info['best_val_mse']:.4f} (epoch {int(np.argmin(info['val_mse_history']))}), "
              f"{res[name]['n_params']} parameters", flush=True)
        models[name] = model
    mx, mn = climode_minmax(a.era5)
    sc = {k: {"rmse": [], "acc": []} for k in ["mean", "mbmx", *models, "persistence"]}
    for y in a.test_years:
        E, O, X0, _, leads = load(a.ens, y)
        if a.quick:
            E, O, X0 = E[:8], O[:8], X0[:8]
        clim = climode_truth(a.era5, y).mean(0)
        n = len(E)
        mem, _, ctx, mu = data(E, O, X0)
        preds = {"mean": E.mean(1), "mbmx": mu, "persistence": np.repeat(X0[:, None], L, 1)}
        with torch.no_grad():
            for name, model in models.items():
                out = torch.cat([model(mem[i:i + 64], ctx[i:i + 64]) for i in range(0, len(mem), 64)]).numpy()
                preds[name] = out.reshape(n, L, C, *out.shape[-2:]) * s + m
        for k, P in preds.items():
            q = climode_scores(P, O, np.full_like(P, 1e-6), clim, lat, mx, mn)
            sc[k]["rmse"].append(q["rmse"]), sc[k]["acc"].append(q["acc"])
        for arm in a.save_mean:
            new = preds[arm].astype(np.float32)
            np.save(os.path.join(a.out, f"diag_{arm}_mean_{y}.npy"), new)
            for prefix in a.shift:  # members + (mu_new - mu*): the arm's mean must be mu* (all regm arms)
                z = np.load(f"{prefix}_ens_{y}.npz")
                Em = z["E_minus_x0"].astype(np.float32)
                gap = (Em.mean(1) + z["X0"][:, None] - mu) / s  # the ensemble's mean is mu* up to its own fit
                print(f"[shift] {prefix} {y}: ensemble mean - mu* (normalized) RMS "
                      f"{np.sqrt((gap ** 2).mean((0, 1, 3, 4))).round(4).tolist()}, max {np.abs(gap).max():.3f}; "
                      "members moved by the correction mu_new - mu*", flush=True)
                np.savez(f"{prefix}_{arm}_ens_{y}.npz",
                         E_minus_x0=(Em + (new - mu)[:, None]).astype(np.float16), O=z["O"], X0=z["X0"],
                         ics=z["ics"], leads=z["leads"])
                print(f"[shift] {prefix}_{arm}_ens_{y}.npz", flush=True)
    res["test"] = {k: {q: {str(h): {v: float(np.concatenate(r[q])[:, j, i].mean()) for i, v in enumerate(VARS)}
                           for j, h in enumerate(leads)} for q in r} for k, r in sc.items()}
    for k, r in res["test"].items():
        print(f"[{k:12s}] RMSE 24/72/144 h " + " | ".join(
            f"{v} " + "/".join(f"{r['rmse'][h][v]:.0f}" if v == "z" else f"{r['rmse'][h][v]:.2f}"
                               for h in ("24", "72", "144") if h in r["rmse"]) for v in VARS), flush=True)
    with open(os.path.join(a.out, f"diag_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
