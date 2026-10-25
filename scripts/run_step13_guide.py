"""Step 13, stage 3 variant: the owner's calibration guide (pgcfm/ens_guide.py) on the saved frozen
FM + PINN ensembles, against classical calibration.

    input   <prefix>_ens_<year>.npz from run_step13_seqfm.py --save-ens (5 ClimODE vars, 24 h
            multiples, members after the frozen PINN): the validation year fits everything, the
            test years score it once
    mu*     the owner's EnsembleToDeterministic, cross-fitted on the two halves of the validation
            year (each fold picks its checkpoint on the last ``--n-valid`` ICs of the other half),
            so the Gamma fits see out-of-fold errors; the test years use the mean of the two folds
    arms    raw         the ensemble as it is
            mbm_global  mu* = a + b mu + c x0 (step 10), one Gamma per (lead, var)        closed form
            mbm_band    the same mu*, Gamma per (lead, var, DCT band)                     closed form
            op_band     operator mu*, Gamma per (lead, var, band)                         closed form
            op_mono     op_band with the expected member energy made non-decreasing in the lead
            guide       operator mu*, Gamma from BandGuide (starts at op_mono; fair CRPS + a penalty
                        on member energy that falls from one lead to the next, whole trajectories
                        per batch; first 80 % of the validation ICs, checkpoint on the last 20 %)
    energy  Doob-type control (owner): E[max_k |e_k|^2] / E|e_K|^2 of the member deviations and of
            the error of the mean (<= 4 if the energy is a submartingale in the lead), the energy by
            lead and the fraction of falling lead steps, for every arm
    output  <out>/<tag>_<arm>_ens_<year>.npz in the stage-2 format, for scripts/score_step13.py

    python scripts/run_step13_guide.py --ens runs/step13/seqfm_dct_pinn --era5 /home/claude/era5/era5_data \
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
from pgcfm.ens_guide import (BandGuide, apply_gamma, band_counts, band_error2, band_index, band_spread2,  # noqa: E402
                             calibrated_energy, doob_stats, fair_crps, member_energy, monotone_gamma,
                             spread_skill_gamma)
from pgcfm.ens_to_det import EnsembleToDeterministic, fit  # noqa: E402
from scripts.run_step10_deterministic import PersistenceMBM  # noqa: E402
from scripts.run_step11_climode import VARS, climode_truth  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402
from scripts.run_step13_operator import load, samples  # noqa: E402


def predict(models, mem, ctx, chunk=64):
    with torch.no_grad():
        return torch.stack([torch.cat([op(mem[i:i + chunk], ctx[i:i + chunk]) for i in range(0, len(mem), chunk)])
                            for op in models]).mean(0)


def per_lead_gamma(err2, spread2, L, M):
    """(N = n_ic * L, C, nb) -> closed-form Gamma (L, C, nb)."""
    return torch.stack([spread_skill_gamma(err2[j::L], spread2[j::L], M) for j in range(L)])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ens", required=True, help="prefix of the saved ensembles (<prefix>_ens_<year>.npz)")
    p.add_argument("--era5", required=True)
    p.add_argument("--out", default="runs/step13")
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--n-valid", type=int, default=20, help="checkpoint ICs per fold")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--guide-epochs", type=int, default=30)
    p.add_argument("--lambda-mono", type=float, default=1.0, help="penalty on energy falling with the lead")
    p.add_argument("--gamma-max", type=float, default=4.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.n_valid, a.width, a.modes, a.epochs, a.guide_epochs = 2, 8, 3, 1, 2
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    lat, lon, _, _ = load_static(a.era5)
    contract.register_latitudes(lat)
    area = torch.from_numpy(np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (len(lat), len(lon))).astype(np.float32).copy())
    w = (area / area.mean())[:, :1]
    tv = climode_truth(a.era5, a.val_year)
    m, s = tv.mean((0, 2, 3))[:, None, None], tv.std((0, 2, 3))[:, None, None]
    del tv
    tag = os.path.basename(a.ens) + ("_quick" if a.quick else "")
    log = lambda q: print(q, flush=True)  # noqa: E731

    # -- validation year: cross-fitted operator, classical mean, out-of-fold errors ---------------
    E, O, X0, _, leads = load(a.ens, a.val_year)
    if a.quick:
        E, O, X0 = E[:12], O[:12], X0[:12]
    n, M, L = E.shape[:3]
    mem, tgt, ctx = samples(E, O, X0, leads, m, s)  # (n * L, ...) normalized, index = ic * L + lead
    ic_of = torch.arange(n * L) // L
    folds = [torch.arange(n * L)[ic_of < n // 2], torch.arange(n * L)[ic_of >= n // 2]]
    mu_op, op_models, res = torch.zeros_like(tgt), [], {"args": vars(a)}
    for f, held in enumerate(folds):
        other = folds[1 - f]
        tr, va = other[:-a.n_valid * L], other[-a.n_valid * L:]
        torch.manual_seed(f)
        op = EnsembleToDeterministic(5, context_channels=6, width=a.width, modes=a.modes)
        info = fit(op, (mem[tr], tgt[tr], ctx[tr]), (mem[va], tgt[va], ctx[va]), area, epochs=a.epochs)
        mu_op[held] = predict([op], mem[held], ctx[held])
        op_models.append(op)
        res[f"fold{f}"] = {"best_val_mse": info["best_val_mse"], "val_history": info["val_mse_history"]}
        log(f"[fold {f}] operator best val mse {info['best_val_mse']:.4f}")
    mbmx = PersistenceMBM(5, L).fit(torch.from_numpy(E), torch.from_numpy(O), torch.from_numpy(X0))
    nz = lambda q: (q - torch.from_numpy(m).float()) / torch.from_numpy(s).float()  # noqa: E731

    def mbm_mean(E_, X0_):
        mu = mbmx(torch.from_numpy(E_), torch.from_numpy(X0_)).mean(1)  # (n, L, 5, H, W) physical
        return nz(mu).reshape(-1, *mu.shape[2:])

    mu_mbm = mbm_mean(E, X0)
    H, W = tgt.shape[-2:]
    bidx, one = band_index(H, W), torch.zeros(H, W, dtype=torch.long)
    nb = int(bidx.max()) + 1
    sp_b, sp_1 = band_spread2(mem, bidx, nb), band_spread2(mem, one, 1)
    gam = {"mbm_global": per_lead_gamma(band_error2(mu_mbm, tgt, one, 1), sp_1, L, M),
           "mbm_band": per_lead_gamma(band_error2(mu_mbm, tgt, bidx, nb), sp_b, L, M),
           "op_band": per_lead_gamma(band_error2(mu_op, tgt, bidx, nb), sp_b, L, M)}
    counts = band_counts(bidx, nb)
    sp_lead = torch.stack([sp_b[j::L].mean(0) for j in range(L)])  # (L, C, nb)
    gam["op_mono"] = monotone_gamma(gam["op_band"], sp_lead, counts)
    for k, g in gam.items():
        log(f"[{k}] Gamma z500 by lead x band: " + " | ".join(
            " ".join(f"{x:.2f}" for x in g[j, VARS.index('z')].tolist()) for j in range(L)))

    # -- the learned guide: Gamma from the ensemble's own band energies -----------------------------
    guide = BandGuide(5, nb, L, gamma_max=a.gamma_max)
    guide.base.copy_(gam["op_mono"].clamp(1e-6, a.gamma_max).log())
    guide.fit_inputs(sp_b)
    lead_idx = torch.arange(n * L) % L
    lead_frac = torch.as_tensor(leads, dtype=torch.float32)[lead_idx] / 144.0
    cut = int(0.8 * n)
    tr_ic, va_ic = torch.arange(cut), torch.arange(cut, n)
    rows = lambda ics: (ics[:, None] * L + torch.arange(L)).flatten()  # noqa: E731  whole trajectories

    def guide_loss(ics):
        j = rows(ics)
        g = guide(sp_b[j], lead_idx[j], lead_frac[j])
        crps = fair_crps(apply_gamma(mu_op[j], mem[j], g, bidx), tgt[j], w)
        S = calibrated_energy(g, sp_b[j], counts).view(len(ics), L, -1)  # (ics, L, C)
        fall = torch.relu(S[:, :-1] - S[:, 1:]).mean() / S.mean().detach()
        return crps, fall

    def guide_val():
        with torch.no_grad():
            q = [guide_loss(i) for i in va_ic.split(8)]
            return float(np.mean([c for c, _ in q])), float(np.mean([f for _, f in q]))

    opt = torch.optim.Adam(guide.net.parameters(), lr=3e-3)
    (best, fall0), best_state, hist = guide_val(), {k: v.clone() for k, v in guide.state_dict().items()}, []
    res["guide_val_crps_start"], res["guide_val_fall_start"] = best, fall0
    best_obj = best + a.lambda_mono * fall0
    for ep in range(a.guide_epochs):
        for i in tr_ic[torch.randperm(len(tr_ic))].split(4):
            crps, fall = guide_loss(i)
            opt.zero_grad(set_to_none=True)
            (crps + a.lambda_mono * fall).backward()
            opt.step()
        v, f_ = guide_val()
        hist.append({"crps": v, "fall": f_})
        if v + a.lambda_mono * f_ < best_obj:
            best_obj, best, best_state = v + a.lambda_mono * f_, v, {k: q.clone() for k, q in guide.state_dict().items()}
    guide.load_state_dict(best_state)
    res.update(guide_val=hist, guide_best=best,
               gamma={k: g.tolist() for k, g in gam.items()}, bands=list(range(nb)), band_counts=counts.tolist())
    log(f"[guide] val fair CRPS start (= op_mono) {res['guide_val_crps_start']:.5f} best {best:.5f}"
        f" | falling energy start {res['guide_val_fall_start']:.4f}")

    # -- test years -----------------------------------------------------------------------------------
    unz = lambda q: q * torch.from_numpy(s).float() + torch.from_numpy(m).float()  # noqa: E731
    summary, energy = {}, {}
    for y in ([a.val_year] if a.quick else a.test_years):
        E, O, X0, ics, leads = load(a.ens, y)
        if a.quick:
            E, O, X0, ics = E[:12], O[:12], X0[:12], ics[:12]
        n = len(E)
        mem, tgt, ctx = samples(E, O, X0, leads, m, s)
        lead_idx = torch.arange(n * L) % L
        lead_frac = torch.as_tensor(leads, dtype=torch.float32)[lead_idx] / 144.0
        mu_op, mu_mbm = predict(op_models, mem, ctx), mbm_mean(E, X0)
        sp = band_spread2(mem, bidx, nb)
        with torch.no_grad():
            arms = {"raw": mem,
                    "mbm_global": apply_gamma(mu_mbm, mem, gam["mbm_global"][lead_idx], one),
                    "mbm_band": apply_gamma(mu_mbm, mem, gam["mbm_band"][lead_idx], bidx),
                    "op_band": apply_gamma(mu_op, mem, gam["op_band"][lead_idx], bidx),
                    "op_mono": apply_gamma(mu_op, mem, gam["op_mono"][lead_idx], bidx),
                    "guide": torch.cat([apply_gamma(mu_op[j], mem[j], guide(sp[j], lead_idx[j], lead_frac[j]), bidx)
                                        for j in torch.arange(n * L).split(64)])}
        for arm, ens in arms.items():
            crps = float(np.mean([float(fair_crps(ens[j], tgt[j], w)) for j in torch.arange(n * L).split(64)]))
            summary.setdefault(arm, []).append(crps)
            traj = ens.view(n, L, M, 5, H, W).transpose(1, 2)  # (n, M, L, 5, H, W) normalized
            err = ((traj.mean(1) - tgt.view(n, L, 5, H, W)) ** 2 * w).mean((-1, -2))[:, None]  # (n, 1, L, 5)
            energy.setdefault(arm, []).append((member_energy(traj, w), err))
            phys = unz(ens).view(n, L, M, 5, H, W).transpose(1, 2).numpy()  # (n, M, L, 5, H, W)
            np.savez(os.path.join(a.out, f"{tag}_{arm}_ens_{y}.npz"),
                     E_minus_x0=(phys - X0[:, None, None]).astype(np.float16), O=O, X0=X0, ics=np.asarray(ics),
                     leads=np.asarray(leads))
        log(f"  {y}: " + " ".join(f"{k} {v[-1]:.5f}" for k, v in summary.items()))
    res["test_fair_crps_normalized"] = {k: float(np.mean(v)) for k, v in summary.items()}
    res["energy"] = {}
    iz = VARS.index("z")
    for arm, q in energy.items():
        mem_e, err_e = torch.cat([e for e, _ in q]), torch.cat([e for _, e in q])
        res["energy"][arm] = {"members": doob_stats(mem_e), "error_of_mean": doob_stats(err_e)}
        d = res["energy"][arm]
        log(f"[energy {arm}] z500 member Doob ratio {d['members']['doob_ratio'][iz]:.2f}, falling steps "
            f"{d['members']['falling_steps'][iz]:.2f}, energy by lead "
            + " ".join(f"{r[iz]:.3f}" for r in d["members"]["energy_by_lead"])
            + f" | error of mean: ratio {d['error_of_mean']['doob_ratio'][iz]:.2f}")
    with open(os.path.join(a.out, f"guide_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("[test] fair CRPS (normalized, all vars and leads) " +
        " ".join(f"{k} {v:.5f}" for k, v in res["test_fair_crps_normalized"].items()))
    return res


if __name__ == "__main__":
    main()
