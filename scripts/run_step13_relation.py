"""Step 13: the owner's ensemble-relation proposal (pgcfm/ens_relation.py) on the real FM + PINN ensembles,
connected to the ERA5 truth.

    train   G(x_{-k}) -> x_k on the validation year's members (first 80 % of the ICs; checkpoint on the
            rest by the LOO reconstruction):
                loo_mean  the LOO member mean (no training; the shrinkage operator the text derives)
                plain     the relation FNO with the LOO reconstruction only
                owner     the text's recipe: + mean consistency, augmented-Lagrangian E_dual <= alpha E(x),
                          hard operator-norm bound on every FNO block
    truth   test years, every G: is the truth exchangeable with the members under G?
                ratio  mean |o - G(x_{-k})|^2 / mean |x_k - G(x_{-k})|^2 by lead, variable and DCT band
                       (1 = the truth sits in the members' relation like one more member)
                rank   rank of |o - G(x_{-j})|^2 among the members' |x_k - G(x_{-k})|^2, j = IC mod N
                case   corr over ICs of the members' residual energy with the error of regm's mean mu*,
                       against the same corr for the member spread: per cell, area-weighted (the
                       conditional field; regm_flow's sigma reaches 0.15-0.21 for z500) and for the
                       domain mean, plus the partial corr given the spread
    arms    the text's generation penalty on the members (relation_refine with the owner G), then mu*:
                regm_rel       consistency + over-dispersion (E_ref from the validation year's error of
                               mu*), no Gamma: the proposal end to end
                regm_rel_band  consistency only, then the closed-form band Gamma refitted on the refined
                               validation members: does the relation structure add anything to regm_band?
            tau from a grid by the fair CRPS on the validation ICs held out from G
    output  <out>/relation_<tag>.json, relation_<tag>_<G>.pt, <tag>_<arm>_ens_<year>.npz (score_step13.py)

    python scripts/run_step13_relation.py --ens runs/step13/seqfm_dct_pinn --reg runs/step13/seqregress_pinn \
        --era5 /home/claude/era5/era5_data --out runs/step13 [--quick]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm import contract  # noqa: E402
from pgcfm.ens_guide import apply_gamma, band_error2, band_index, band_mean, band_spread2, fair_crps  # noqa: E402
from pgcfm.ens_relation import (EnergyConstraint, EnsembleRelationFNO, fit_relation, l2_sq,  # noqa: E402
                                relation_refine)
from pgcfm.seqdct import dct_2d  # noqa: E402
from scripts.run_step11_climode import VARS, climode_truth  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402
from scripts.run_step13_guide import per_lead_gamma, reg_mbm_mean  # noqa: E402
from scripts.run_step13_operator import load, samples  # noqa: E402


class LOOMean(nn.Module):
    def forward(self, x):
        return (x.sum(1, keepdim=True) - x) / (x.shape[1] - 1)


def corr(a, b, dim=0):
    a, b = a - a.mean(dim), b - b.mean(dim)
    return (a * b).mean(dim) / (a.std(dim, unbiased=False) * b.std(dim, unbiased=False) + 1e-12)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ens", required=True)
    p.add_argument("--reg", required=True, help="regression ensemble prefix: mu* = a + b mu_reg + c x0 (regm)")
    p.add_argument("--era5", required=True)
    p.add_argument("--out", default="runs/step13")
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, nargs=2, default=[12, 12])
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.8, help="E_dual <= alpha E(x) (the text's value)")
    p.add_argument("--bound", type=float, default=2.0, help="operator-norm bound per FNO block (the text's value)")
    p.add_argument("--taus", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.45, 0.6])
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.width, a.modes, a.layers, a.epochs, a.taus = 8, [4, 4], 2, 1, [0.2, 0.6]
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log = lambda q: print(f"{q}  ({(time.time() - t0) / 60:.1f} min)", flush=True)  # noqa: E731
    lat, lon, _, _ = load_static(a.era5)
    contract.register_latitudes(lat)
    area = torch.from_numpy(np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (len(lat), len(lon))).astype(np.float32).copy())
    wn = area / area.sum()
    w = (area / area.mean())[:, :1]  # fair_crps weights, as in run_step13_guide.py
    tv = climode_truth(a.era5, a.val_year)
    m, s = tv.mean((0, 2, 3))[:, None, None], tv.std((0, 2, 3))[:, None, None]
    del tv
    tag = os.path.basename(a.ens) + ("_quick" if a.quick else "")
    nz = lambda q: (q - torch.from_numpy(m).float()) / torch.from_numpy(s).float()  # noqa: E731
    unz = lambda q: q * torch.from_numpy(s).float() + torch.from_numpy(m).float()  # noqa: E731

    # -- validation year: G training, mu*, E_ref ----------------------------------------------------
    E, O, X0, ics, leads = load(a.ens, a.val_year)
    if a.quick:
        E, O, X0, ics = E[:10], O[:10], X0[:10], ics[:10]
    n, M, L = E.shape[:3]
    mem, tgt, _ = samples(E, O, X0, leads, m, s)
    del E
    H, W = tgt.shape[-2:]
    ic_of, lead_idx = torch.arange(n * L) // L, torch.arange(n * L) % L
    cut = int(0.8 * n)
    tr_rows, va_rows = torch.arange(n * L)[ic_of < cut], torch.arange(n * L)[ic_of >= cut]
    reg_mbm, mu_v = reg_mbm_mean(a.reg, a.val_year, ics)
    mu_v = nz(torch.from_numpy(mu_v)).reshape(-1, 5, H, W)
    err_e = l2_sq(tgt - mu_v, area)  # (n L,)
    E_ref = torch.stack([err_e[lead_idx == j].mean() for j in range(L)]) * (M - 1) / (M + 1)
    res = {"args": vars(a), "E_ref": E_ref.tolist()}
    log("[relation] E_ref by lead (normalized, summed over vars) " + " ".join(f"{q:.3f}" for q in E_ref.tolist()))

    Gs = {"loo_mean": LOOMean()}
    for name in ("plain", "owner"):
        torch.manual_seed(1)
        G = EnsembleRelationFNO(5, a.width, tuple(a.modes), a.layers)
        con = EnergyConstraint(alpha=a.alpha, rho=10.0, beta=1.0) if name == "owner" else None
        info = fit_relation(G, mem[tr_rows], mem[va_rows], area, epochs=a.epochs, lr=a.lr, constraint=con,
                            bound=a.bound if name == "owner" else None, log=lambda q: log(f"[{name}] {q}"))
        with torch.no_grad():
            info["op_norms"] = G.l2_operator_norms().tolist()
        res[f"train_{name}"] = info
        torch.save({"state_dict": G.state_dict(), "norm_mean": m, "norm_std": s,
                    "config": {"in_ch": 5, "width": a.width, "modes": tuple(a.modes), "n_layers": a.layers}},
                   os.path.join(a.out, f"relation_{tag}_{name}.pt"))
        Gs[name] = G
    with torch.no_grad():
        res["val_recon"] = {k: float(np.mean([float(l2_sq(G(mem[i]) - mem[i], area).mean())
                                               for i in va_rows.split(32)])) for k, G in Gs.items()}
    log("[relation] held-out 2016 LOO recon " + " ".join(f"{k} {v:.4f}" for k, v in res["val_recon"].items()))

    # -- the generation penalty: tau on the held-out validation ICs, Gamma refitted on the refined members
    Go = Gs["owner"]
    bidx = band_index(H, W)
    nb = int(bidx.max()) + 1

    def refine(mem_, rows, tau, ge):  # rows = ic * L + lead in every year
        return torch.cat([relation_refine(Go, mem_[j], E_ref[j % L], area, tau, a.steps, 1.0, ge)
                          for j in rows.split(16)])

    def crps_of(ens, rows_):
        return float(np.mean([float(fair_crps(ens[i], tgt[rows_][i], w)) for i in torch.arange(len(rows_)).split(64)]))

    base_gam = per_lead_gamma(band_error2(mu_v, tgt, bidx, nb), band_spread2(mem, bidx, nb), L, M)
    sel = {"regm_band": crps_of(apply_gamma(mu_v[va_rows], mem[va_rows], base_gam[lead_idx[va_rows]], bidx), va_rows)}
    log(f"[arms] held-out 2016 fair CRPS regm_band {sel['regm_band']:.5f}")
    best = {}
    for tau in a.taus:
        y = refine(mem, va_rows, tau, 1.0)
        sel[f"regm_rel_tau{tau:g}"] = c1 = crps_of(mu_v[va_rows][:, None] + y - y.mean(1, keepdim=True), va_rows)
        y_all = refine(mem, torch.arange(n * L), tau, 0.0)
        gam = per_lead_gamma(band_error2(mu_v, tgt, bidx, nb), band_spread2(y_all, bidx, nb), L, M)
        sel[f"regm_rel_band_tau{tau:g}"] = c2 = crps_of(apply_gamma(mu_v[va_rows], y_all[va_rows],
                                                                    gam[lead_idx[va_rows]], bidx), va_rows)
        if c1 < best.get("regm_rel", (None, np.inf))[1]:
            best["regm_rel"] = (tau, c1, None)
        if c2 < best.get("regm_rel_band", (None, np.inf))[1]:
            best["regm_rel_band"] = (tau, c2, gam)
        log(f"[arms] tau {tau:g}: held-out 2016 fair CRPS regm_rel {c1:.5f} regm_rel_band {c2:.5f}")
        del y_all
    res["tau_selection"] = sel
    res["tau"] = {k: v[0] for k, v in best.items()}
    log("[arms] chosen tau " + " ".join(f"{k} {v[0]:g}" for k, v in best.items()))
    gam_rel = best["regm_rel_band"][2]
    res["gamma_rel_band"], res["gamma_regm_band"] = gam_rel.tolist(), base_gam.tolist()

    # -- test years: truth diagnostics for every G, then the arms --------------------------------------
    diag = {k: {"eM": [], "eT": [], "bM": [], "bT": [], "loc_r": [], "rank": [], "dual": []} for k in Gs}
    common = {"err": [], "spread": [], "loc_err": [], "loc_sd": [], "lead": []}
    summary = {}
    for y in ([a.val_year] if a.quick else a.test_years):
        E, O, X0, ics, leads = load(a.ens, y)
        if a.quick:
            E, O, X0, ics = E[:10], O[:10], X0[:10], ics[:10]
        n = len(E)
        mem, tgt, _ = samples(E, O, X0, leads, m, s)
        del E
        li = torch.arange(n * L) % L
        mu = nz(torch.from_numpy(reg_mbm_mean(a.reg, y, ics, reg_mbm)[1])).reshape(-1, 5, H, W)
        dev = mem - mem.mean(1, keepdim=True)
        common["err"].append(((tgt - mu) ** 2 * wn).sum((-1, -2)))  # (n L, 5)
        common["spread"].append(((dev ** 2).sum(1) / (M - 1) * wn).sum((-1, -2)))
        common["loc_err"].append((tgt - mu).abs())
        common["loc_sd"].append(((dev ** 2).sum(1) / (M - 1)).sqrt())
        common["lead"].append(li)
        jr = (torch.arange(n * L) // L) % M  # the member the truth stands in for
        with torch.no_grad():
            for k, G in Gs.items():
                for i in torch.arange(n * L).split(32):
                    xh = G(mem[i])
                    r, t = mem[i] - xh, tgt[i][:, None] - xh
                    eM, eT = ((r ** 2) * wn).sum((-1, -2)), ((t ** 2) * wn).sum((-1, -2))  # (b, M, 5)
                    d = diag[k]
                    d["eM"].append(eM), d["eT"].append(eT)
                    d["bM"].append(band_mean((dct_2d(r) ** 2).mean(1), bidx, nb))
                    d["bT"].append(band_mean((dct_2d(t) ** 2).mean(1), bidx, nb))
                    d["loc_r"].append((r ** 2).mean(1).sqrt())
                    eTj = eT[torch.arange(len(i)), jr[i]]  # (b, 5)
                    d["rank"].append((eM < eTj[:, None]).sum(1))  # 0..M
                    xbar = mem[i].mean(1, keepdim=True)
                    d["dual"].append(torch.stack([l2_sq(xh - xbar, area).mean(1), l2_sq(mem[i] - xbar, area).mean(1)], 1))
        # arms
        y_rel = refine(mem, torch.arange(n * L), best["regm_rel"][0], 1.0)
        y_relb = refine(mem, torch.arange(n * L), best["regm_rel_band"][0], 0.0)
        arms = {"regm_rel": mu[:, None] + y_rel - y_rel.mean(1, keepdim=True),
                "regm_rel_band": apply_gamma(mu, y_relb, gam_rel[li], bidx),
                "regm_band_check": apply_gamma(mu, mem, base_gam[li], bidx)}
        del y_rel, y_relb
        for arm, ens in arms.items():
            crps = float(np.mean([float(fair_crps(ens[j], tgt[j], w)) for j in torch.arange(n * L).split(64)]))
            summary.setdefault(arm, []).append(crps)
            if arm == "regm_band_check" or a.quick:
                continue
            phys = unz(ens).view(n, L, M, 5, H, W).transpose(1, 2).numpy()
            np.savez(os.path.join(a.out, f"{tag}_{arm}_ens_{y}.npz"),
                     E_minus_x0=(phys - X0[:, None, None]).astype(np.float16), O=O, X0=X0, ics=np.asarray(ics),
                     leads=np.asarray(leads))
            del phys
        del arms
        log(f"  {y}: " + " ".join(f"{k} {v[-1]:.5f}" for k, v in summary.items()))
    res["test_fair_crps_normalized"] = {k: float(np.mean(v)) for k, v in summary.items()}

    # -- truth diagnostics, by lead and variable -------------------------------------------------------
    cm = {k: torch.cat(v) for k, v in common.items()}
    lead_all = cm["lead"]
    res["truth"] = {}
    for k, d in diag.items():
        d = {q: torch.cat(v) for q, v in d.items()}
        out = {"ratio": [], "ratio_band": [], "classic_ratio": [], "rank_hist": [], "rank_tv": [],
               "case_corr_domain": [], "case_corr_domain_spread": [], "partial_corr_domain": [],
               "case_corr_local": [], "case_corr_local_spread": [], "dual_over_E": []}
        for j in range(L):
            sel_ = lead_all == j
            eM, eT = d["eM"][sel_].mean(1), d["eT"][sel_].mean(1)  # (n, 5)
            err, spr = cm["err"][sel_], cm["spread"][sel_]
            out["ratio"].append((eT.mean(0) / eM.mean(0)).tolist())
            out["ratio_band"].append((d["bT"][sel_].mean(0) / d["bM"][sel_].mean(0)).tolist())  # (5, nb)
            out["classic_ratio"].append((err.mean(0) / (spr.mean(0) * (M + 1) / M)).tolist())
            hist = torch.stack([(d["rank"][sel_] == r).float().mean(0) for r in range(M + 1)])  # (M+1, 5)
            out["rank_hist"].append(hist.T.tolist())
            out["rank_tv"].append((0.5 * (hist - 1 / (M + 1)).abs().sum(0)).tolist())
            c_me, c_se, c_ms = corr(eM, err), corr(spr, err), corr(eM, spr)
            out["case_corr_domain"].append(c_me.tolist())
            out["case_corr_domain_spread"].append(c_se.tolist())
            out["partial_corr_domain"].append(((c_me - c_ms * c_se) / ((1 - c_ms ** 2) * (1 - c_se ** 2)).clamp_min(1e-12).sqrt()).tolist())
            le = cm["loc_err"][sel_]
            out["case_corr_local"].append(((corr(d["loc_r"][sel_], le) * area).sum((-1, -2)) / area.sum()).tolist())
            out["case_corr_local_spread"].append(((corr(cm["loc_sd"][sel_], le) * area).sum((-1, -2)) / area.sum()).tolist())
            out["dual_over_E"].append(float(d["dual"][sel_][:, 0].mean() / d["dual"][sel_][:, 1].mean()))
        res["truth"][k] = out
        iz, it = VARS.index("z"), VARS.index("t")
        log(f"[truth {k}] ratio z500 by lead " + " ".join(f"{q[iz]:.2f}" for q in out["ratio"])
            + " | t850 " + " ".join(f"{q[it]:.2f}" for q in out["ratio"])
            + f" | classic z500 {out['classic_ratio'][0][iz]:.2f} | rank TV z500 24 h {out['rank_tv'][0][iz]:.3f}")
        log(f"[truth {k}] case corr z500 by lead: local residual " + " ".join(f"{q[iz]:.3f}" for q in out["case_corr_local"])
            + " | local spread " + " ".join(f"{q[iz]:.3f}" for q in out["case_corr_local_spread"])
            + " | domain residual " + " ".join(f"{q[iz]:.2f}" for q in out["case_corr_domain"])
            + " | partial " + " ".join(f"{q[iz]:.2f}" for q in out["partial_corr_domain"])
            + f" | E_dual/E {out['dual_over_E'][0]:.4f}")
    with open(os.path.join(a.out, f"relation_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("[test] fair CRPS (normalized, all vars and leads) "
        + " ".join(f"{k} {v:.5f}" for k, v in res["test_fair_crps_normalized"].items()))
    return res


if __name__ == "__main__":
    main()
