"""Step 13 flow-dependence check: does the ensemble spread vary from case to case, and where the error does?

For each arm, test years 2017-2018 (ClimODE's 360 ICs), z500 and t850 at 24 / 72 / 144 h (``--leads``), per
grid point over the ICs:
    cv_spread    std / mean of the member spread (M = 8). A spread that is the same for every case still
                 shows about 1 / sqrt(2 (M - 1)) = 0.27 from sampling; seqregress_pinn (statistical errors
                 with a fixed covariance) is that baseline
    cv_abs_err   the same for |ensemble mean - truth| (0.76 for a Gaussian error of fixed size)
    case_corr    correlation of spread and |error| over the ICs
    bin_*        spread normalized by its grid-point mean, pooled, cut into quintiles; mean normalized
                 spread and RMS normalized error per quintile (a flow-dependent ensemble has rising error)
    cv_spread_true  the case-to-case CV of the spread without the sampling noise (step 14, the owner's H-dyn
                 criterion): the spreads of two disjoint half-ensembles have independent sampling noise, so
                 their covariance over the ICs is the variance of the true spread. Per grid point clipped at 0
                 (biased up: ~0.04 for a spread that does not vary); cv_spread_true_pooled averages the
                 relative covariance over the grid first (RMS of the true CV, unbiased before the root)
    dom_*        per IC the domain spread sqrt(area mean of the member variance) and the domain RMSE of the
                 ensemble mean: CV over the ICs (dom_spread_cv_true by the same split-half covariance) and
                 their correlation
Area-weighted means over grid points. Writes <runs>/flowdep.json (``--out``).

    python scripts/diag_step13_flowdep.py [--runs runs/step13] [--arms seqregress_pinn seqfm_dct_pinn ...]
        [--leads 24 48 72 96 120 144] [--out runs/step13/flowdep_all.json]
"""

import argparse
import json
import os

import numpy as np

STEP_H = 24  # the saved ensembles hold leads 24, 48, ..., 144 h


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/step13")
    p.add_argument("--arms", nargs="+",
                   default=["seqregress_pinn", "seqfm_dct_pinn", "seqfm_grid_pinn", "seqfm_dct_pinn_regm_band"])
    p.add_argument("--years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--leads", type=int, nargs="+", default=[24, 72, 144], help="hours")
    p.add_argument("--out", default=None, help="default <runs>/flowdep.json")
    a = p.parse_args(argv)
    lat = np.linspace(-87.1875, 87.1875, 32)
    wl = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 64))
    wl = wl / wl.sum()
    out = {}
    for arm in a.arms:
        S, Eabs, H1, H2 = [], [], [], []
        for y in a.years:
            z = np.load(os.path.join(a.runs, f"{arm}_ens_{y}.npz"))
            d = z["E_minus_x0"][:, :, :, :2].astype(np.float32)  # (n, M, L, z500 / t850, H, W)
            h = d.shape[1] // 2
            S.append(d.std(1, ddof=1))
            H1.append(d[:, :h].var(1, ddof=1)), H2.append(d[:, h:2 * h].var(1, ddof=1))
            Eabs.append(np.abs(d.mean(1) + z["X0"][:, None, :2] - z["O"][:, :, :2]))
        S, Eabs, H1, H2 = np.concatenate(S), np.concatenate(Eabs), np.concatenate(H1), np.concatenate(H2)
        res = {}
        for lead in a.leads:
            li = lead // STEP_H - 1
            for vi, v in enumerate(("z500", "t850")):
                s, e = S[:, li, vi], Eabs[:, li, vi]
                sc, ec = s - s.mean(0), e - e.mean(0)
                corr = ((sc * ec).mean(0) / (sc.std(0) * ec.std(0) + 1e-12) * wl).sum()
                sn, e2n = (s / s.mean(0)).ravel(), (e ** 2 / (e ** 2).mean(0)).ravel()
                b = np.clip(np.searchsorted(np.quantile(sn, [0, .2, .4, .6, .8, 1]), sn, side="right") - 1, 0, 4)
                r = dict(cv_spread=float((s.std(0) / s.mean(0) * wl).sum()),
                         cv_abs_err=float((e.std(0) / e.mean(0) * wl).sum()), case_corr=float(corr),
                         bin_spread=[float(sn[b == k].mean()) for k in range(5)],
                         bin_rmse=[float(np.sqrt(e2n[b == k].mean())) for k in range(5)])
                s1, s2 = np.sqrt(H1[:, li, vi]), np.sqrt(H2[:, li, vi])
                cov = ((s1 - s1.mean(0)) * (s2 - s2.mean(0))).mean(0)
                m12 = 0.5 * (s1 + s2).mean(0)
                r["cv_spread_true"] = float((np.sqrt(np.clip(cov, 0, None)) / m12 * wl).sum())
                # clipping each grid point at 0 leaves a positive floor (~0.04 for 360 ICs, M = 4 + 4);
                # pooling the relative covariance over the grid before the square root does not
                r["cv_spread_true_pooled"] = float(np.sqrt(max((cov / m12 ** 2 * wl).sum(), 0.0)))
                ds1, ds2 = np.sqrt((H1[:, li, vi] * wl).sum((-2, -1))), np.sqrt((H2[:, li, vi] * wl).sum((-2, -1)))
                ds, de = np.sqrt((S[:, li, vi] ** 2 * wl).sum((-2, -1))), np.sqrt((e ** 2 * wl).sum((-2, -1)))
                dcov = ((ds1 - ds1.mean()) * (ds2 - ds2.mean())).mean()
                r.update(dom_spread_cv=float(ds.std() / ds.mean()),
                         dom_spread_cv_true=float(np.sqrt(max(dcov, 0.0)) / (0.5 * (ds1 + ds2).mean())),
                         dom_rmse_cv=float(de.std() / de.mean()), dom_corr=float(np.corrcoef(ds, de)[0, 1]))
                res[f"{v}_{lead}"] = r
                print(f"{arm:28s} {v} {lead:3d}h  CV spread {r['cv_spread']:.3f} (|err| {r['cv_abs_err']:.2f})  "
                      f"corr {r['case_corr']:.3f}  bins spread {r['bin_spread'][0]:.2f}->{r['bin_spread'][4]:.2f}  "
                      f"rmse {r['bin_rmse'][0]:.2f}->{r['bin_rmse'][4]:.2f}  CV true {r['cv_spread_true']:.3f} "
                      f"(pooled {r['cv_spread_true_pooled']:.3f})  "
                      f"domain CV {r['dom_spread_cv']:.3f} (true {r['dom_spread_cv_true']:.3f}, RMSE "
                      f"{r['dom_rmse_cv']:.3f}) corr {r['dom_corr']:.2f}", flush=True)
        out[arm] = res
    with open(a.out or os.path.join(a.runs, "flowdep.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
