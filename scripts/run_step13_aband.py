"""Step 13: the owner's revised hypothesis, a case-adaptive band Gamma (pgcfm/adaptive_band.py), on the real
FM + PINN ensembles and ERA5.

    mean     mu* = a + b mu_reg + c x0 (regm, as regm_band)
    fit      validation year: log s_b^2(c) per variable and DCT band from the conditioning
             [x0, mu*, mu* - x0, lead, valid time, static] and the scalars [lead, valid time], chi^2 NLL on the
             band energy of the truth's error r = o - mu*; every fifth IC held out for the checkpoint
             (as regm_flow)
    arms     regm_aband       T_c with the realized member band spread, Gamma_max = 1 (the text's default)
             regm_aband_g4    the same, Gamma_max = 4 (regm_band's cap)
             regm_aband_marg  the lead's climatological band spread instead of the realized one (ablation)
             regm_aband_clim  log s^2 = the climatological start (no case adaptation), realized spread: what
                              the realized normalization alone does
             regm_sband<n>    the owner's second revision (pgcfm/space_band.py): band x cell multipliers
                              T_c = sum_b P_b M_b(c) P_b, about n coefficients per cell (``--cell-nmin``),
                              realized cell spread, Gamma_max = 1; the n with the lower held-out 2016 fair
                              CRPS is also saved as regm_sband
    report   per band: held-out / test corr over cases of s_b^2 and Ebar_b, its ceiling and the ratio;
             n_eff (lead-mean reference and fitted residual ratio); per arm: fair CRPS, the fraction of
             Gamma at Gamma_max, ||T_c|| = max_b Gamma_b, the case spread/skill CV and the corr over cases
             of the domain spread and the domain RMSE (z500, t850), against regm_band; per cell for the
             band x cell nets; and every model's predicted local variance (band scalar, band x cell, the
             regm_flow sigma from ``--flow-sigma-pt``) pooled to common cell grids against the pooled
             squared error: corr over cases per cell, mean over cells (1 x 1 up to the 32 x 64 grid)
    output   <out>/aband_<tag>.json, aband_<tag>.pt, <tag>_<arm>_ens_<year>.npz for ``--save-arms``

    python scripts/run_step13_aband.py --ens runs/step13/seqfm_dct_pinn --reg runs/step13/seqregress_pinn \
        --era5 /home/claude/era5/era5_data --out runs/step13 [--quick]
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pgcfm import contract  # noqa: E402
from pgcfm.adaptive_band import (AdaptiveBandOperator, BandPool, CondBandVariance, OrthoDCT2,  # noqa: E402
                                 band_corr_with_ceiling, case_ssr, chi2_band_nll, estimate_n_eff, radial_bands)
from pgcfm.core import static_features, time_features  # noqa: E402
from pgcfm.ens_guide import apply_gamma, band_error2, band_index, band_spread2, fair_crps  # noqa: E402
from pgcfm.flow_sigma import FlowSigma, flow_features, predict_sigma  # noqa: E402
from pgcfm.space_band import (BandProjector, CondSpaceBandVariance, SpaceBandOperator,  # noqa: E402
                              band_cell_grids, cell_corr_and_ceiling, estimate_cell_n_eff, local_error_energy,
                              space_band_nll, upsample_sphere)
from scripts.run_step11_climode import VARS, climode_truth  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402
from scripts.run_step13_guide import per_lead_gamma, reg_mbm_mean  # noqa: E402
from scripts.run_step13_operator import load, samples  # noqa: E402

ARMS = {  # name: (adaptive, normalize, Gamma_max)
    "regm_aband": (True, "realized", 1.0),
    "regm_aband_g4": (True, "realized", 4.0),
    "regm_aband_marg": (True, "marginal", 1.0),
    "regm_aband_clim": (False, "realized", 1.0),
}


def corr(a, b, dim=0):
    a, b = a - a.mean(dim), b - b.mean(dim)
    return (a * b).mean(dim) / (a.std(dim, unbiased=False) * b.std(dim, unbiased=False) + 1e-12)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ens", required=True)
    p.add_argument("--reg", required=True)
    p.add_argument("--era5", required=True)
    p.add_argument("--out", default="runs/step13")
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--bands", default="regm", help="regm (band_index, 5 bands) or radial<n> (the text's split)")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--cell-nmin", type=int, nargs="*", default=[16, 32], help="band x cell nets (empty: none)")
    p.add_argument("--flow-sigma-pt", default=None,
                   help="regm_flow's sigma net for the comparison (default <out>/flow_sigma_<ens name>.pt if present)")
    p.add_argument("--save-arms", nargs="*", default=["regm_aband", "regm_aband_clim", "regm_sband"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.width, a.epochs = 8, 2
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log = lambda q: print(f"{q}  ({(time.time() - t0) / 60:.1f} min)", flush=True)  # noqa: E731
    lat, lon, oro, lsm = load_static(a.era5)
    contract.register_latitudes(lat)
    static = static_features(lat, lon, oro, lsm)
    area = torch.from_numpy(np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (len(lat), len(lon))).astype(np.float32).copy())
    wn = area / area.sum()
    w = (area / area.mean())[:, :1]
    tv = climode_truth(a.era5, a.val_year)
    m, s = tv.mean((0, 2, 3))[:, None, None], tv.std((0, 2, 3))[:, None, None]
    del tv
    tag = os.path.basename(a.ens) + ("_quick" if a.quick else "")
    nz = lambda q: (q - torch.from_numpy(m).float()) / torch.from_numpy(s).float()  # noqa: E731
    unz = lambda q: q * torch.from_numpy(s).float() + torch.from_numpy(m).float()  # noqa: E731
    H, W = len(lat), len(lon)
    dct = OrthoDCT2(H, W)
    bidx_r = band_index(H, W)  # regm_band's split
    nb_r = int(bidx_r.max()) + 1
    bidx = bidx_r if a.bands == "regm" else radial_bands(H, W, int(a.bands[len("radial"):]))
    nb = int(bidx.max()) + 1
    pool = BandPool(bidx, nb)
    iz, it = VARS.index("z"), VARS.index("t")

    def inputs(ctx, mu, ics, L, leads):
        n_ = len(ics)
        ic_hour = 6.0 * torch.as_tensor(np.asarray(ics), dtype=torch.float32)[torch.arange(n_ * L) // L]
        lead_h = torch.as_tensor(leads, dtype=torch.float32)[torch.arange(n_ * L) % L]
        feats = flow_features(ctx[:, :5], mu, lead_h, ic_hour, static)
        scal = torch.cat([(lead_h / 144.0)[:, None], time_features(ic_hour + lead_h, 1, 1)[:, :, 0, 0]], 1)
        return feats, scal

    # -- validation year ------------------------------------------------------------------------------
    E, O, X0, ics, leads = load(a.ens, a.val_year)
    if a.quick:
        E, O, X0, ics = E[:10], O[:10], X0[:10], ics[:10]
    n, M, L = E.shape[:3]
    mem, tgt, ctx = samples(E, O, X0, leads, m, s)
    del E
    ic_of, lead_idx = torch.arange(n * L) // L, torch.arange(n * L) % L
    held = ic_of % 5 == 4
    tr, va = torch.arange(n * L)[~held], torch.arange(n * L)[held]
    reg_mbm, mu_v = reg_mbm_mean(a.reg, a.val_year, ics)
    mu_v = nz(torch.from_numpy(mu_v)).reshape(-1, 5, H, W)
    feats, scal = inputs(ctx, mu_v, ics, L, leads)
    Ebar = pool.mean_energy(dct(tgt - mu_v))  # (n L, C, nb)
    clim = torch.stack([Ebar[tr][lead_idx[tr] == j].mean(0) for j in range(L)])  # (L, C, nb)
    n_eff0 = estimate_n_eff(Ebar[tr], clim[lead_idx[tr]], pool.n_b)
    res = {"args": vars(a), "n_b": pool.n_b.tolist(), "n_eff_lead_ref": n_eff0.tolist()}
    log(f"[aband] bands n_b {pool.n_b.long().tolist()} | n_eff (lead-mean reference) "
        + " ".join(f"{q:.1f}" for q in n_eff0.tolist()))
    def fit_net(net, loss_of, name):  # AdamW, checkpoint on the held-out ICs
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr)

        def val():
            net.eval()
            with torch.no_grad():
                return sum(float(loss_of(i)) * len(i) for i in va.split(256)) / len(va)

        best = start = val()
        best_state, hist = copy.deepcopy(net.state_dict()), []
        for ep in range(a.epochs):
            net.train()
            for i in tr[torch.randperm(len(tr))].split(a.batch):
                loss = loss_of(i)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            v = val()
            hist.append(v)
            if v < best:
                best, best_state = v, copy.deepcopy(net.state_dict())
            if ep % 10 == 9 or ep == a.epochs - 1:
                log(f"[{name}] epoch {ep} held-out NLL {v:.4f} (start {start:.4f}, best {best:.4f})")
        net.load_state_dict(best_state)
        net.eval()
        return {"val_nll_start": start, "val_nll": hist, "val_nll_best": best}

    torch.manual_seed(1)
    net = CondBandVariance(feats.shape[1], scal.shape[1], 5, nb, clim.log(), width=a.width)
    res.update(fit_net(net, lambda i: chi2_band_nll(net(feats[i], scal[i], lead_idx[i]), Ebar[i], n_eff0), "aband"))
    with torch.no_grad():
        ls_v = torch.cat([net(feats[i], scal[i], lead_idx[i]) for i in torch.arange(n * L).split(256)])
    n_eff = estimate_n_eff(Ebar[tr], ls_v[tr].exp(), pool.n_b)
    res["n_eff_fitted"] = n_eff.tolist()
    log("[aband] n_eff (fitted residual ratio) " + " ".join(f"{q:.1f}" for q in n_eff.tolist()))

    def corr_report(ls, Eb, rows):
        cs, ce = [], []
        for j in range(L):
            r_ = rows[(rows % L) == j]  # rows = ic * L + lead
            c, ceil = band_corr_with_ceiling(ls[r_], Eb[r_], n_eff)
            cs.append(c), ce.append(ceil)
        c, ce = torch.stack(cs), torch.stack(ce)  # (L, C, nb)
        return {"corr": c.tolist(), "ceiling": ce.tolist(), "corr_mean": c.mean(0).tolist(),
                "ceiling_mean": ce.mean(0).tolist()}

    res["val_band_corr"] = corr_report(ls_v, Ebar, va)
    q = res["val_band_corr"]
    log("[aband] held-out corr(s2, Ebar) z500 by band " + " ".join(f"{x:.3f}" for x in q["corr_mean"][iz])
        + " | ceiling " + " ".join(f"{x:.3f}" for x in q["ceiling_mean"][iz]))
    torch.save({"state_dict": net.state_dict(), "n_eff": n_eff, "bands": a.bands, "norm_mean": m, "norm_std": s,
                "config": {"n_feats": feats.shape[1], "n_scalars": scal.shape[1], "C": 5, "n_bands": nb,
                           "width": a.width}}, os.path.join(a.out, f"aband_{tag}.pt"))

    # -- band x cell nets (second revision) -----------------------------------------------------------
    proj = BandProjector(dct, pool)
    r_v = tgt - mu_v
    sband = {}
    for nmin in a.cell_nmin:
        grids = band_cell_grids(pool.n_b, H, W, nmin)
        Ev = [torch.cat(q) for q in zip(*[local_error_energy(proj, r_v[i], grids) for i in torch.arange(n * L).split(256)])]
        clim_c = [torch.stack([E[tr][lead_idx[tr] == j].mean(0) for j in range(L)]) for E in Ev]
        ne = estimate_cell_n_eff([E[tr] for E in Ev], clim_c, lead_idx[tr], pool.n_b, grids)
        torch.manual_seed(1)
        net_s = CondSpaceBandVariance(feats.shape[1], scal.shape[1], 5, grids, [c.log() for c in clim_c], width=a.width)
        info = fit_net(net_s, lambda i, net_s=net_s, Ev=Ev, ne=ne: space_band_nll(
            net_s(feats[i], scal[i], lead_idx[i]), [E[i] for E in Ev], ne), f"sband{nmin}")
        with torch.no_grad():
            ls_s = [torch.cat(q) for q in zip(*[net_s(feats[i], scal[i], lead_idx[i]) for i in va.split(256)])]
        cc = [cell_corr_and_ceiling([q[(va % L) == j] for q in ls_s], [E[va][(va % L) == j] for E in Ev], ne)
              for j in range(L)]  # [lead][band] -> (corr (C,), ceiling (C,))
        T = SpaceBandOperator(proj, grids, gamma_max=1.0)
        crps_v = 0.0
        for k0 in range(0, len(va), 64):
            i = va[k0:k0 + 64]
            ens_, _ = T(mem[i], mu_v[i], [q[k0:k0 + 64] for q in ls_s])
            crps_v += float(fair_crps(ens_, tgt[i], w)) * len(i)
        info.update(grids=grids, n_eff=ne.tolist(), held_out_fair_crps=crps_v / len(va),
                    held_out_cell_corr=[[c.tolist() for c, _ in q] for q in cc],
                    held_out_cell_ceiling=[[e.tolist() for _, e in q] for q in cc])
        res[f"sband{nmin}"] = info
        log(f"[sband{nmin}] cells {grids} | n_eff " + " ".join(f"{q:.1f}" for q in ne.tolist())
            + " | held-out corr / ceiling z500 by band " + " ".join(
                f"{np.mean([cc[j][b][0][iz] for j in range(L)]):.3f}/{np.mean([cc[j][b][1][iz] for j in range(L)]):.3f}"
                for b in range(nb)) + f" | held-out fair CRPS {crps_v / len(va):.5f}")
        torch.save({"state_dict": net_s.state_dict(), "grids": grids, "n_eff": ne, "norm_mean": m, "norm_std": s,
                    "config": {"n_feats": feats.shape[1], "n_scalars": scal.shape[1], "C": 5, "width": a.width}},
                   os.path.join(a.out, f"aband_cells{nmin}_{tag}.pt"))
        sband[nmin] = (net_s, grids)
    best_nmin = min(sband, key=lambda q: res[f"sband{q}"]["held_out_fair_crps"]) if sband else None
    res["sband_saved_nmin"] = best_nmin
    fsig = None
    a.flow_sigma_pt = a.flow_sigma_pt or os.path.join(a.out, f"flow_sigma_{os.path.basename(a.ens)}.pt")
    if os.path.exists(a.flow_sigma_pt):
        d_ = torch.load(a.flow_sigma_pt, weights_only=False)
        fsig = FlowSigma(d_["in_channels"], torch.zeros(L, 5, H, W))
        fsig.load_state_dict(d_["state_dict"])
        fsig.eval()

    # regm_band (baseline) and the marginal spread on the validation year
    base_gam = per_lead_gamma(band_error2(mu_v, tgt, bidx_r, nb_r), band_spread2(mem, bidx_r, nb_r), L, M)
    T_r = AdaptiveBandOperator(dct, pool)
    sp_v = torch.cat([T_r.ensemble_band_spread(dct(mem[i] - mem[i].mean(1, keepdim=True)))
                      for i in torch.arange(n * L).split(128)])
    sig_marg = torch.stack([sp_v[lead_idx == j].mean(0) for j in range(L)])  # (L, C, nb)
    del mem, ctx, sp_v

    # -- test years -----------------------------------------------------------------------------------
    names = ["regm_band"] + list(ARMS) + [f"regm_sband{q}" for q in sband]
    GRIDS = [(1, 1), (2, 4), (5, 10), (H, W)]
    pooled = {}  # model -> per grid: chunks of (pooled predicted variance, pooled squared error, lead)
    acc = {k: {"crps": [], "clip": [], "opn": [], "ssr": [], "dsp": [], "derr": [], "lead": []} for k in names}
    tcorr = {"ls": [], "Eb": []}
    for y in ([a.val_year] if a.quick else a.test_years):
        E, O, X0, ics, leads = load(a.ens, y)
        if a.quick:
            E, O, X0, ics = E[:10], O[:10], X0[:10], ics[:10]
        n = len(E)
        mem, tgt, ctx = samples(E, O, X0, leads, m, s)
        del E
        li = torch.arange(n * L) % L
        mu = nz(torch.from_numpy(reg_mbm_mean(a.reg, y, ics, reg_mbm)[1])).reshape(-1, 5, H, W)
        feats, scal = inputs(ctx, mu, ics, L, leads)
        with torch.no_grad():
            ls = torch.cat([net(feats[i], scal[i], li[i]) for i in torch.arange(n * L).split(256)])
        Eb = pool.mean_energy(dct(tgt - mu))
        tcorr["ls"].append(ls), tcorr["Eb"].append(Eb)
        with torch.no_grad():
            ls_sb = {q: [torch.cat(z) for z in zip(*[net_s(feats[i], scal[i], li[i]) for i in torch.arange(n * L).split(256)])]
                     for q, (net_s, _) in sband.items()}
        sig_f = predict_sigma(fsig, feats, li) if fsig is not None else None
        # every model's predicted local variance against the squared error, pooled to common cell grids
        r2 = (tgt - mu) ** 2
        models = {"aband": (ls.exp() * pool.n_b / (H * W)).sum(-1)[..., None, None].expand_as(r2)}
        for q in sband:
            models[f"sband{q}"] = sum(upsample_sphere(z.exp(), H, W) for z in ls_sb[q])
        if sig_f is not None:
            models["regm_flow"] = sig_f ** 2
        for k, v in models.items():
            pooled.setdefault(k, [[] for _ in GRIDS])
            for gi, g in enumerate(GRIDS):
                pooled[k][gi].append((F.adaptive_avg_pool2d(v, g), F.adaptive_avg_pool2d(r2, g), li))
        del models, sig_f, r2
        keep = {k: [] for k in a.save_arms}
        for i in torch.arange(n * L).split(64):
            out = {"regm_band": (apply_gamma(mu[i], mem[i], base_gam[li[i]], bidx_r), None)}
            for k, (adapt, norm, gmax) in ARMS.items():
                T = AdaptiveBandOperator(dct, pool, gamma_max=gmax, normalize=norm)
                ens, g, info = T(mem[i], mu[i], ls[i] if adapt else clim.log()[li[i]], sig_marg[li[i]])
                out[k] = (ens, (g, info))
            for q, (_, grids) in sband.items():
                ens, info = SpaceBandOperator(proj, grids, gamma_max=1.0)(mem[i], mu[i], [z[i] for z in ls_sb[q]])
                out[f"regm_sband{q}"] = (ens, (None, {"clip_frac": info["clip_frac"], "op_norm": info["op_norm_bound"]}))
            for k, (ens, gi) in out.items():
                d = acc[k]
                d["crps"].append(float(fair_crps(ens, tgt[i], w)) * len(i))
                if gi is not None:
                    d["clip"].append(float(gi[1]["clip_frac"]) * len(i)), d["opn"].append(gi[1]["op_norm"])
                d["ssr"].append(case_ssr(ens, tgt[i], mu[i]))
                dev = ens - ens.mean(1, keepdim=True)
                d["dsp"].append((((dev ** 2).sum(1) / (M - 1)) * wn).sum((-1, -2)).sqrt())  # (b, C) domain spread
                d["derr"].append((((tgt[i] - mu[i]) ** 2) * wn).sum((-1, -2)).sqrt())
                d["lead"].append(li[i])
                save_as = "regm_sband" if k == f"regm_sband{best_nmin}" else k
                if save_as in keep:
                    keep[save_as].append(unz(ens))
        for k, v in keep.items():
            phys = torch.cat(v).view(n, L, M, 5, H, W).transpose(1, 2).numpy()
            if not a.quick:
                np.savez(os.path.join(a.out, f"{tag}_{k}_ens_{y}.npz"),
                         E_minus_x0=(phys - X0[:, None, None]).astype(np.float16), O=O, X0=X0,
                         ics=np.asarray(ics), leads=np.asarray(leads))
            del phys
        del keep, mem, ctx, ls_sb
        log(f"  {y}: fair CRPS so far " + " ".join(
            f"{k} {sum(acc[k]['crps']) / sum(len(q_) for q_ in acc[k]['lead']):.5f}" for k in names))
    ls, Eb = torch.cat(tcorr["ls"]), torch.cat(tcorr["Eb"])
    res["test_band_corr"] = corr_report(ls, Eb, torch.arange(len(ls)))
    q = res["test_band_corr"]
    log("[aband] test corr(s2, Ebar) z500 by band " + " ".join(f"{x:.3f}" for x in q["corr_mean"][iz])
        + " | ceiling " + " ".join(f"{x:.3f}" for x in q["ceiling_mean"][iz])
        + " | t850 " + " ".join(f"{x:.3f}" for x in q["corr_mean"][it]))
    res["cell_grid_corr"] = {}
    for k, per_grid in pooled.items():
        res["cell_grid_corr"][k] = {}
        for gi, g in enumerate(GRIDS):
            v, e, ld = (torch.cat(z) for z in zip(*per_grid[gi]))
            res["cell_grid_corr"][k][f"{g[0]}x{g[1]}"] = [corr(v[ld == j], e[ld == j]).mean((-1, -2)).tolist()
                                                         for j in range(L)]  # (L, C)
        log(f"[cells {k:9s}] corr over cases of predicted local variance and squared error, z500 mean over leads: "
            + " ".join(f"{g}: {np.mean([q_[iz] for q_ in res['cell_grid_corr'][k][g]]):.3f}"
                       for g in res["cell_grid_corr"][k]))
    res["test"] = {}
    for k in names:
        d = acc[k]
        lead = torch.cat(d["lead"])
        ssr, dsp, derr = torch.cat(d["ssr"]), torch.cat(d["dsp"]), torch.cat(d["derr"])
        r = {"fair_crps": sum(d["crps"]) / len(lead),
             "case_ssr_cv": [float(ssr[lead == j].std() / ssr[lead == j].mean()) for j in range(L)],
             "domain_spread_cv": [(dsp[lead == j].std(0) / dsp[lead == j].mean(0)).tolist() for j in range(L)],
             "domain_rmse_cv": [(derr[lead == j].std(0) / derr[lead == j].mean(0)).tolist() for j in range(L)],
             "domain_case_corr": [corr(dsp[lead == j], derr[lead == j]).tolist() for j in range(L)]}
        if d["clip"]:
            opn = torch.cat(d["opn"])
            r.update(clip_frac=sum(d["clip"]) / len(lead), op_norm_mean=float(opn.mean()),
                     op_norm_max=float(opn.max()), op_norm_frac_above_1=float((opn > 1 + 1e-6).float().mean()))
        res["test"][k] = r
        log(f"[{k:16s}] fair CRPS {r['fair_crps']:.5f} | domain case corr z500 by lead "
            + " ".join(f"{q_[iz]:.2f}" for q_ in r["domain_case_corr"])
            + f" | spread CV z500 24 h {r['domain_spread_cv'][0][iz]:.3f} (RMSE CV {r['domain_rmse_cv'][0][iz]:.3f})"
            + (f" | clip {r['clip_frac']:.3f}, ||T|| mean {r['op_norm_mean']:.2f}" if "clip_frac" in r else ""))
    with open(os.path.join(a.out, f"aband_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("[test] fair CRPS (normalized, all vars and leads) "
        + " ".join(f"{k} {res['test'][k]['fair_crps']:.5f}" for k in names))
    return res


if __name__ == "__main__":
    main()
