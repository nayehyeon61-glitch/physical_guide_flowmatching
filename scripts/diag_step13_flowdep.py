"""Step 13 flow-dependence check: does the ensemble spread vary from case to case, and where the error does?

For each arm, test years 2017-2018 (ClimODE's 360 ICs), z500 and t850 at 24 / 72 / 144 h, per grid point
over the ICs:
    cv_spread    std / mean of the member spread (M = 8). A spread that is the same for every case still
                 shows about 1 / sqrt(2 (M - 1)) = 0.27 from sampling; seqregress_pinn (statistical errors
                 with a fixed covariance) is that baseline
    cv_abs_err   the same for |ensemble mean - truth| (0.76 for a Gaussian error of fixed size)
    case_corr    correlation of spread and |error| over the ICs
    bin_*        spread normalized by its grid-point mean, pooled, cut into quintiles; mean normalized
                 spread and RMS normalized error per quintile (a flow-dependent ensemble has rising error)
Area-weighted means over grid points. Writes <runs>/flowdep.json.

    python scripts/diag_step13_flowdep.py [--runs runs/step13] [--arms seqregress_pinn seqfm_dct_pinn ...]
"""

import argparse
import json
import os

import numpy as np

LEADS = ((0, 24), (2, 72), (5, 144))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/step13")
    p.add_argument("--arms", nargs="+",
                   default=["seqregress_pinn", "seqfm_dct_pinn", "seqfm_grid_pinn", "seqfm_dct_pinn_regm_band"])
    p.add_argument("--years", type=int, nargs="+", default=[2017, 2018])
    a = p.parse_args(argv)
    lat = np.linspace(-87.1875, 87.1875, 32)
    wl = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 64))
    wl = wl / wl.sum()
    out = {}
    for arm in a.arms:
        S, Eabs = [], []
        for y in a.years:
            z = np.load(os.path.join(a.runs, f"{arm}_ens_{y}.npz"))
            d = z["E_minus_x0"][:, :, :, :2].astype(np.float32)  # (n, M, L, z500 / t850, H, W)
            S.append(d.std(1, ddof=1))
            Eabs.append(np.abs(d.mean(1) + z["X0"][:, None, :2] - z["O"][:, :, :2]))
        S, Eabs = np.concatenate(S), np.concatenate(Eabs)
        res = {}
        for li, lead in LEADS:
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
                res[f"{v}_{lead}"] = r
                print(f"{arm:28s} {v} {lead:3d}h  CV spread {r['cv_spread']:.3f} (|err| {r['cv_abs_err']:.2f})  "
                      f"corr {r['case_corr']:.3f}  bins spread {r['bin_spread'][0]:.2f}->{r['bin_spread'][4]:.2f}  "
                      f"rmse {r['bin_rmse'][0]:.2f}->{r['bin_rmse'][4]:.2f}", flush=True)
        out[arm] = res
    with open(os.path.join(a.runs, "flowdep.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
