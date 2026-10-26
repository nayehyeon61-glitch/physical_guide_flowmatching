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
            core_band   ``--core``: mu* = the step 12 deterministic core (jump24, trained 2006-2015, so
            core_mono   2016 is out of sample), Gamma closed form per band / with monotone energy, and
            core_guide  the learned guide on top: the core's mean with the FM members' deviations
            reg_band    ``--reg``: mu* = the member mean of another saved ensemble (step 13's sequence
            reg_guide   regression, runs/step13/seqregress_pinn: trained 2012-2015), closed form / guide
            regm_band   ``--reg --reg-mbm``: mu* = a + b mu_reg + c x0 (step 10's PersistenceMBM fitted on
            regm_guide  the validation year's regression ensemble), closed form / guide
            regm_flow   ``--flow-sigma``: regm_band with a flow-dependent spread factor per case and cell
                        (pgcfm/flow_sigma.py: ClimODE's uncertainty network and NLL on the error of mu*,
                        every fifth validation IC held out for the checkpoint)
            regm_flowop ``--flow-op``: the same with the owner's operator as the sigma network, pretrained
                        on the regm_band members of the validation year (MSE to the truth), then
                        fine-tuned with the NLL (head with the operator frozen, then all at lr 1e-4)
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
from pgcfm.core import static_features, time_features  # noqa: E402
from pgcfm.ens_to_det import EnsembleToDeterministic, fit  # noqa: E402
from pgcfm.flow_sigma import (FlowSigma, HeadOnly, OperatorSigma, fit_flow_sigma, flow_features,  # noqa: E402
                              predict_sigma)
from scripts.run_step10_deterministic import PersistenceMBM  # noqa: E402
from scripts.run_step11_climode import VARS, climode_truth  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402
from scripts.run_step13_operator import load, samples  # noqa: E402

LEADS_H = (24, 48, 72, 96, 120, 144)


def core_predictions(prefix, era5, year, ics):
    """(len(ics), 6, 5, H, W) physical forecasts of the step 12 core from the ClimODE ICs ``ics``
    (2 + 8 e): the full year from <prefix>_pred.npz (test years) or computed once and cached in
    <prefix>_pred_<year>.npz (scripts/run_step12_core.py's normalization), then the rows of ``ics``."""
    rows = (np.asarray(ics) - 2) // 8
    assert np.all(2 + 8 * rows == np.asarray(ics)), "core predictions exist only for the ClimODE ICs"
    with np.load(prefix + "_pred.npz") as z:
        if str(year) in z.files:
            return z[str(year)][rows]
    cache = f"{prefix}_pred_{year}.npz"
    if not os.path.exists(cache):
        from pgcfm.core import Core, CoreConfig, static_features
        from scripts.run_step12_core import forecast
        info = json.load(open(prefix + ".json"))
        cfg = CoreConfig(**{**info["cfg"], "blocks": tuple(info["cfg"]["blocks"])})
        tr = np.concatenate([climode_truth(era5, y) for y in info["args"]["train_years"]])
        mean, std = tr.mean((0, 2, 3)), tr.std((0, 2, 3))
        del tr
        lat, lon, oro, lsm = load_static(era5)
        core = Core(cfg, static_features(lat, lon, oro, lsm),
                    {int(h): torch.tensor(v, dtype=torch.float32) for h, v in info["tend_std"].items()})
        core.load_state_dict(torch.load(prefix + ".pt"))
        core.eval()
        truth = climode_truth(era5, year)
        full = [2 + 8 * e for e in range((len(truth) - 2 - 144 // 6 - 1) // 8 + 1)]
        x = torch.from_numpy((truth - mean[:, None, None]) / std[:, None, None])
        with torch.no_grad():
            pred = forecast(core, x, full, 24).numpy() * std[:, None, None] + mean[:, None, None]
        np.savez(cache, pred=pred.astype(np.float32), ics=np.asarray(full))
    return np.load(cache)["pred"][rows]


def ens_mean(prefix, year, ics):
    """(len(ics), L, 5, H, W) physical member mean of a saved stage-2 ensemble at the ICs ``ics``."""
    z = np.load(f"{prefix}_ens_{year}.npz")
    rows = [int(np.flatnonzero(z["ics"] == i)[0]) for i in ics]
    d = np.stack([z["E_minus_x0"][r].astype(np.float32).mean(0) for r in rows])
    return d + z["X0"][rows][:, None]


def reg_mbm_mean(prefix, year, ics, mbm=None):
    """PersistenceMBM a + b mu + c x0 on a saved ensemble at the ICs ``ics``: fitted on this year when
    ``mbm`` is None. Returns (mbm, (len(ics), L, 5, H, W) physical corrected mean)."""
    E, O, X0, ics_, _ = load(prefix, year)
    rows = [int(np.flatnonzero(ics_ == i)[0]) for i in ics]
    E, O, X0 = (torch.from_numpy(np.ascontiguousarray(q[rows])) for q in (E, O, X0))
    if mbm is None:
        mbm = PersistenceMBM(5, E.shape[2]).fit(E, O, X0)
    return mbm, mbm(E, X0).mean(1).numpy()


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
    p.add_argument("--core", default=None, help="step 12 core prefix (runs/step12/core_jump24): adds the core_* arms")
    p.add_argument("--reg", default=None,
                   help="saved ensemble prefix whose member mean is mu* (runs/step13/seqregress_pinn): adds reg_*")
    p.add_argument("--reg-mbm", action="store_true", help="with --reg: also mu* = a + b mu_reg + c x0 (regm_*)")
    p.add_argument("--flow-sigma", action="store_true", help="with --reg --reg-mbm: adds regm_flow")
    p.add_argument("--flow-op", action="store_true",
                   help="with --reg --reg-mbm: adds regm_flowop (operator pretrained on the regm_band members)")
    p.add_argument("--sigma-epochs", type=int, default=30)
    p.add_argument("--sigma-full-epochs", type=int, default=None,
                   help="--flow-op: epochs of the stage that trains the whole operator (default sigma_epochs // 3)")
    p.add_argument("--save-arms", nargs="*", default=None, help="score and save only these arms (default all)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.n_valid, a.width, a.modes, a.epochs, a.guide_epochs, a.sigma_epochs = 2, 8, 3, 1, 2, 1
    if (a.flow_sigma or a.flow_op) and not (a.reg and a.reg_mbm):
        p.error("--flow-sigma and --flow-op need --reg and --reg-mbm")
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
    lead_idx = torch.arange(n * L) % L
    lead_frac = torch.as_tensor(leads, dtype=torch.float32)[lead_idx] / 144.0
    cut = int(0.8 * n)
    tr_ic, va_ic = torch.arange(cut), torch.arange(cut, n)
    rows = lambda ics: (ics[:, None] * L + torch.arange(L)).flatten()  # noqa: E731  whole trajectories

    def train_guide(mu_v, base, name):
        guide = BandGuide(5, nb, L, gamma_max=a.gamma_max)
        guide.base.copy_(base.clamp(1e-6, a.gamma_max).log())
        guide.fit_inputs(sp_b)

        def guide_loss(ics):
            j = rows(ics)
            g = guide(sp_b[j], lead_idx[j], lead_frac[j])
            crps = fair_crps(apply_gamma(mu_v[j], mem[j], g, bidx), tgt[j], w)
            S = calibrated_energy(g, sp_b[j], counts).view(len(ics), L, -1)  # (ics, L, C)
            fall = torch.relu(S[:, :-1] - S[:, 1:]).mean() / S.mean().detach()
            return crps, fall

        def guide_val():
            with torch.no_grad():
                q = [guide_loss(i) for i in va_ic.split(8)]
                return float(np.mean([c for c, _ in q])), float(np.mean([f for _, f in q]))

        opt = torch.optim.Adam(guide.net.parameters(), lr=3e-3)
        (best, fall0), best_state, hist = guide_val(), {k: v.clone() for k, v in guide.state_dict().items()}, []
        best_obj, start = best + a.lambda_mono * fall0, best
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
        res[name] = {"val_crps_start": start, "val_fall_start": fall0, "val": hist, "best": best}
        log(f"[{name}] val fair CRPS start {start:.5f} best {best:.5f} | falling energy start {fall0:.4f}")
        return guide

    guides = {"guide": (train_guide(mu_op, gam["op_mono"], "guide"), "op")}
    if a.core:
        mu_core = nz(torch.from_numpy(core_predictions(a.core, a.era5, a.val_year, load(a.ens, a.val_year)[3][:n])))
        mu_core = mu_core.reshape(-1, 5, H, W)
        gam["core_band"] = per_lead_gamma(band_error2(mu_core, tgt, bidx, nb), sp_b, L, M)
        gam["core_mono"] = monotone_gamma(gam["core_band"], sp_lead, counts)
        guides["core_guide"] = (train_guide(mu_core, gam["core_mono"], "core_guide"), "core")
        log("[core] val z500 RMSE of the core mean by lead (normalized) "
            + " ".join(f"{float(((mu_core[j::L] - tgt[j::L])[:, VARS.index('z')] ** 2 * w).mean().sqrt()):.3f}"
                       for j in range(L)))
    if a.reg:
        mu_reg = nz(torch.from_numpy(ens_mean(a.reg, a.val_year, load(a.ens, a.val_year)[3][:n]))).reshape(-1, 5, H, W)
        gam["reg_band"] = per_lead_gamma(band_error2(mu_reg, tgt, bidx, nb), sp_b, L, M)
        guides["reg_guide"] = (train_guide(mu_reg, monotone_gamma(gam["reg_band"], sp_lead, counts), "reg_guide"), "reg")
        log("[reg] val z500 RMSE of the regression mean by lead (normalized) "
            + " ".join(f"{float(((mu_reg[j::L] - tgt[j::L])[:, VARS.index('z')] ** 2 * w).mean().sqrt()):.3f}"
                       for j in range(L)))
        if a.reg_mbm:
            reg_mbm, mu_regm = reg_mbm_mean(a.reg, a.val_year, load(a.ens, a.val_year)[3][:n])
            mu_regm = nz(torch.from_numpy(mu_regm)).reshape(-1, 5, H, W)
            gam["regm_band"] = per_lead_gamma(band_error2(mu_regm, tgt, bidx, nb), sp_b, L, M)
            guides["regm_guide"] = (train_guide(mu_regm, monotone_gamma(gam["regm_band"], sp_lead, counts),
                                                "regm_guide"), "regm")
            log("[regm] val z500 RMSE of a + b mu_reg + c x0 by lead (normalized) "
                + " ".join(f"{float(((mu_regm[j::L] - tgt[j::L])[:, VARS.index('z')] ** 2 * w).mean().sqrt()):.3f}"
                           for j in range(L)))
    if a.flow_sigma or a.flow_op:  # flow-dependent spread factors on top of regm_band (pgcfm/flow_sigma.py)
        lat_, lon_, oro_, lsm_ = load_static(a.era5)
        static = static_features(lat_, lon_, oro_, lsm_)
        leads_t = torch.as_tensor(leads, dtype=torch.float32)

        def hours(ics_):  # (n * L,) IC hours since Jan 1 and lead hours
            n_ = len(ics_)
            return (6.0 * torch.as_tensor(np.asarray(ics_), dtype=torch.float32)[torch.arange(n_ * L) // L],
                    leads_t[torch.arange(n_ * L) % L])

        def sigma_inputs(ctx_, mu_, ics_):
            ic_hour, lead_h = hours(ics_)
            return flow_features(ctx_[:, :5], mu_, lead_h, ic_hour, static)

        def op_context(ctx_, ics_):  # the operator's context: x0, lead, and the valid time (pgcfm.core)
            ic_hour, lead_h = hours(ics_)
            return torch.cat([ctx_, time_features(ic_hour + lead_h, H, W)], 1)

        def case_corr(sig, absr):  # per lead: corr over ICs of sigma and |error|, area-weighted over cells
            q = []
            for j in range(L):
                s_, e_ = sig[j::L] - sig[j::L].mean(0), absr[j::L] - absr[j::L].mean(0)
                c = (s_ * e_).mean(0) / (s_.std(0, unbiased=False) * e_.std(0, unbiased=False) + 1e-12)
                q.append(((c * area).sum((-1, -2)) / area.sum()).tolist())
            return q  # (L, 5)

        ics_v = load(a.ens, a.val_year)[3][:n]
        held = ic_of % 5 == 4
        tr_rows, va_rows = torch.arange(n * L)[~held], torch.arange(n * L)[held]
        r2 = (tgt - mu_regm) ** 2
        sigma_clim = torch.stack([r2[tr_rows[tr_rows % L == j]].mean(0).sqrt() for j in range(L)])
        rho = {}

        def report(name, sig_v):
            rho[name] = torch.stack([(sig_v[j::L] ** 2).mean(0).sqrt() for j in range(L)])  # (L, 5, H, W)
            res[name]["val_case_corr"] = case_corr(sig_v[va_rows], (tgt - mu_regm).abs()[va_rows])
            log(f"[{name}] held-out 2016 case corr of sigma and |error|, z500 by lead "
                + " ".join(f"{q[VARS.index('z')]:.3f}" for q in res[name]["val_case_corr"]))

    if a.flow_sigma:  # ClimODE's uncertainty network on hand-made flow features
        feats_v = sigma_inputs(ctx, mu_regm, ics_v)
        torch.manual_seed(2)
        fsig = FlowSigma(feats_v.shape[1], sigma_clim)
        res["flow_sigma"] = fit_flow_sigma(fsig, feats_v, mu_regm, tgt, lead_idx, tr_rows, va_rows,
                                           epochs=a.sigma_epochs, log=log)
        report("flow_sigma", predict_sigma(fsig, feats_v, lead_idx))
    if a.flow_op:  # the owner's operator: pretrained on the energy-refined members, fine-tuned into sigma
        with torch.no_grad():
            cal_v = torch.cat([apply_gamma(mu_regm[j], mem[j], gam["regm_band"][lead_idx[j]], bidx)
                               for j in torch.arange(n * L).split(64)])
        ctx_op = op_context(ctx, ics_v)
        torch.manual_seed(3)
        op_s = EnsembleToDeterministic(5, context_channels=ctx_op.shape[1], width=a.width, modes=a.modes)
        info = fit(op_s, (cal_v[tr_rows], tgt[tr_rows], ctx_op[tr_rows]), (cal_v[va_rows], tgt[va_rows], ctx_op[va_rows]),
                   area, epochs=a.epochs)
        log(f"[flow_op] pretrained operator best val mse {info['best_val_mse']:.4f}")
        osig = OperatorSigma(op_s, sigma_clim)
        op_s.requires_grad_(False)
        with torch.no_grad():
            f_v = torch.cat([osig.features(cal_v[j], ctx_op[j]) for j in torch.arange(n * L).split(64)])
        head = fit_flow_sigma(HeadOnly(osig), f_v, mu_regm, tgt, lead_idx, tr_rows, va_rows, epochs=a.sigma_epochs,
                              params=osig.head.parameters(), log=log)
        del f_v
        op_s.requires_grad_(True)
        full = fit_flow_sigma(osig, (cal_v, ctx_op), mu_regm, tgt, lead_idx, tr_rows, va_rows,
                              epochs=a.sigma_full_epochs or max(1, a.sigma_epochs // 3), lr=1e-4, log=log)
        res["flow_op"] = {"pretrain_best_val_mse": info["best_val_mse"], "pretrain_val_mse": info["val_mse_history"],
                          "head": head, "full": full}
        report("flow_op", predict_sigma(osig, (cal_v, ctx_op), lead_idx))
        del cal_v
    res.update(gamma={k: g.tolist() for k, g in gam.items()}, bands=list(range(nb)), band_counts=counts.tolist())

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
        mus = {"op": mu_op}
        if a.core:
            mus["core"] = nz(torch.from_numpy(core_predictions(a.core, a.era5, y, ics))).reshape(-1, 5, H, W)
        if a.reg:
            mus["reg"] = nz(torch.from_numpy(ens_mean(a.reg, y, ics))).reshape(-1, 5, H, W)
        if a.reg and a.reg_mbm:
            mus["regm"] = nz(torch.from_numpy(reg_mbm_mean(a.reg, y, ics, reg_mbm)[1])).reshape(-1, 5, H, W)
        sp = band_spread2(mem, bidx, nb)
        with torch.no_grad():
            arms = {"raw": mem,
                    "mbm_global": apply_gamma(mu_mbm, mem, gam["mbm_global"][lead_idx], one),
                    "mbm_band": apply_gamma(mu_mbm, mem, gam["mbm_band"][lead_idx], bidx),
                    "op_band": apply_gamma(mu_op, mem, gam["op_band"][lead_idx], bidx),
                    "op_mono": apply_gamma(mu_op, mem, gam["op_mono"][lead_idx], bidx)}
            for k in ("core_band", "core_mono", "reg_band", "regm_band"):
                if k in gam:
                    arms[k] = apply_gamma(mus[k.split("_")[0]], mem, gam[k][lead_idx], bidx)
            for k, (gd, src) in guides.items():
                arms[k] = torch.cat([apply_gamma(mus[src][j], mem[j], gd(sp[j], lead_idx[j], lead_frac[j]), bidx)
                                     for j in torch.arange(n * L).split(64)])
            sigmas = {}
            if a.flow_sigma:
                sigmas["flow_sigma", "regm_flow"] = predict_sigma(fsig, sigma_inputs(ctx, mus["regm"], ics), lead_idx)
            if a.flow_op:
                sigmas["flow_op", "regm_flowop"] = predict_sigma(osig, (arms["regm_band"], op_context(ctx, ics)), lead_idx)
            for (name, arm), sig in sigmas.items():
                fac = sig / rho[name][lead_idx]
                arms[arm] = mus["regm"][:, None] + fac[:, None] * (arms["regm_band"] - mus["regm"][:, None])
                res[name].setdefault("test_case_corr", {})[str(y)] = case_corr(sig, (tgt - mus["regm"]).abs())
                res[name].setdefault("factor_cv", {})[str(y)] = float(fac.std() / fac.mean())
                log(f"[{name}] {y} case corr of sigma and |error|, z500 by lead "
                    + " ".join(f"{q[VARS.index('z')]:.3f}" for q in res[name]["test_case_corr"][str(y)])
                    + f" | factor CV {res[name]['factor_cv'][str(y)]:.3f}")
        if a.save_arms is not None:
            arms = {k: v for k, v in arms.items() if k in a.save_arms}
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
    name = f"guide_{tag}" + ("_" + "_".join(a.save_arms) if a.save_arms else "")
    with open(os.path.join(a.out, name + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    log("[test] fair CRPS (normalized, all vars and leads) " +
        " ".join(f"{k} {v:.5f}" for k, v in res["test_fair_crps_normalized"].items()))
    return res


if __name__ == "__main__":
    main()
