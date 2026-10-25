"""Step 13: the owner's four evaluation axes on the saved stage-2 ensembles (run_step13_seqfm.py --save-ens).

    probabilistic   ensemble CRPS (ClimODE's min-max units), tercile Brier scores of the lower / upper
                    tercile events (thresholds per grid cell and calendar month from the 12 UTC
                    analyses of the training years 2006-2015)
    reliability     spread/skill ratio, rank histogram (fraction of the truth outside the ensemble range;
                    perfect: 2 / (M + 1)), its total-variation distance from flat
    structure       zonal power of member anomalies / truth anomalies in three wavenumber bands
                    (blur or noise), z500-t850 anomaly correlation inside a member vs the truth, lag-24 h
                    anomaly autocorrelation along the member trajectory vs the truth (persistence)
    energy          Doob-type control along the lead (owner): E[max_k |e_k|^2] / E|e_K|^2 for the member
                    deviations and for the error of the mean (<= 4 if the energy is a submartingale in
                    the lead), and the fraction of lead steps where the mean member energy falls
    uncertainty     paired bootstrap over ICs (members are never resampled as cases) of the CRPS and RMSE
                    difference of every arm to ``--ref``, 95 % intervals

The per-member physics residuals of the 66 channels are in the stage-2 json (they need all levels).

    python scripts/score_step13.py --era5 /home/claude/era5/era5_data --out runs/step13/score.json \
        --arms runs/step13/seqregress_raw runs/step13/seqfm_dct_raw runs/step13/seqfm_dct_pinn ... --ref ...
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import properscoring as ps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from scripts.run_step11_climode import VARS, climode_minmax, climode_truth  # noqa: E402

BANDS = {"k1-4": (1, 5), "k5-12": (5, 13), "k13-32": (13, 33)}


def tercile_thresholds(era5, years):
    """(12, 2, 5, H, W) lower / upper tercile per calendar month of the 12 UTC analyses."""
    per_month = {m: [] for m in range(1, 13)}
    for y in years:
        t = climode_truth(era5, y)
        times = pd.date_range(f"{y}-01-01", periods=len(t), freq="6h")
        for m in range(1, 13):
            per_month[m].append(t[(times.month == m) & (times.hour == 12)])
    return np.stack([np.quantile(np.concatenate(per_month[m]), [1 / 3, 2 / 3], axis=0) for m in range(1, 13)])


def zonal_power(a):
    """Zonal power spectrum (..., H, W) -> (..., W // 2 + 1), averaged over latitude rows."""
    return (np.abs(np.fft.rfft(a, axis=-1)) ** 2).mean(-2)


def corr(a, b, axes):
    a, b = a - a.mean(axes, keepdims=True), b - b.mean(axes, keepdims=True)
    return (a * b).sum(axes) / np.sqrt((a ** 2).sum(axes) * (b ** 2).sum(axes))


def score_year(prefix, year, era5, terc, lat, mx, mn, chunk=30):
    """All axes for one year, ``chunk`` ICs at a time (a whole year of 8 members in float32 with the
    tercile, rank and FFT temporaries took 13 GB)."""
    z = np.load(f"{prefix}_ens_{year}.npz")
    E16, O_all, X0_all, ics_all, leads = z["E_minus_x0"], z["O"], z["X0"], z["ics"], z["leads"]
    n, M, L = E16.shape[:3]
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None]
    truth = climode_truth(era5, year)
    clim = truth.mean(0)  # ClimODE's ACC climatology (the test year's mean)
    scl = (mx - mn)[:, None, None]
    times = pd.date_range(f"{year}-01-01", periods=len(truth), freq="6h")
    rng = np.random.default_rng(0)
    iz, it = VARS.index("z"), VARS.index("t")
    parts, hist, pe, po = {}, 0.0, 0.0, 0.0
    for c in range(0, n, chunk):
        sl = slice(c, min(n, c + chunk))
        O, X0, ics = O_all[sl], X0_all[sl], ics_all[sl]
        E = E16[sl].astype(np.float32) + X0[:, None, None]  # (b, M, L, 5, H, W)
        out = {}
        mu = E.mean(1)
        out["rmse"] = np.sqrt(((mu - O) ** 2 * w).mean((-1, -2)))  # (b, L, 5)
        out["spread2"] = ((E.var(1, ddof=1)) * w).mean((-1, -2))
        out["crps"] = ps.crps_ensemble((O - mn[:, None, None]) / scl,
                                       np.moveaxis((E - mn[:, None, None]) / scl, 1, -1)).mean((-1, -2))
        # tercile Brier scores
        month = np.array([[times[i + h // 6].month for h in leads] for i in ics])  # (b, L)
        th = terc[month - 1]  # (b, L, 2, 5, H, W)
        lo_p, hi_p = (E < th[:, None, :, 0]).mean(1), (E > th[:, None, :, 1]).mean(1)
        lo_o, hi_o = (O < th[:, :, 0]).astype(np.float32), (O > th[:, :, 1]).astype(np.float32)
        out["brier_lower"] = ((lo_p - lo_o) ** 2 * w).mean((-1, -2))
        out["brier_upper"] = ((hi_p - hi_o) ** 2 * w).mean((-1, -2))
        out["obs_freq_lower"], out["obs_freq_upper"] = (lo_o * w).mean((-1, -2)), (hi_o * w).mean((-1, -2))
        # rank histogram (ties broken at random)
        below = (E < O[:, None]).sum(1) + rng.integers(0, 1 + (E == O[:, None]).sum(1))
        hist = hist + np.stack([(below == r).sum((0, -1, -2)) for r in range(M + 1)], -1)  # (L, 5, M + 1) counts
        # structure: zonal power of anomalies, cross-variable correlation, lag-24 h persistence
        ea, oa = E - clim, O - clim
        pe, po = pe + zonal_power(ea).sum((0, 1)) / M, po + zonal_power(oa).sum(0)  # (L, 5, W/2+1)
        out["zt_corr_member"] = corr(ea[:, :, :, iz], ea[:, :, :, it], (-1, -2)).mean(1)  # (b, L)
        out["zt_corr_truth"] = corr(oa[:, :, iz], oa[:, :, it], (-1, -2))
        out["lag24_member"] = corr(ea[:, :, :-1], ea[:, :, 1:], (-1, -2)).mean(1)  # (b, L - 1, 5)
        out["lag24_truth"] = corr(oa[:, :-1], oa[:, 1:], (-1, -2))
        # energy along the lead (24 h multiples): per member deviation energy and error energy of the mean
        out["e_member"] = (((E - mu[:, None]) ** 2) * w).mean((-1, -2))  # (b, M, L, 5)
        out["e_error"] = (((mu - O) ** 2) * w).mean((-1, -2))[:, None]  # (b, 1, L, 5)
        for k, v in out.items():
            parts.setdefault(k, []).append(v)
        del E, ea, oa, th, lo_p, hi_p, below
    res = {k: np.concatenate(v) for k, v in parts.items()}
    res["rank_hist"] = hist / hist.sum(-1, keepdims=True)
    res["power_ratio"] = {k: pe[..., i:j].sum(-1) / po[..., i:j].sum(-1) for k, (i, j) in BANDS.items()}
    res["M"], res["leads"] = M, leads
    return res


def doob(e):
    """(n, M, L, 5) energies -> Doob ratio E[max_k e_k] / E[e_K] and falling-step fraction, per variable."""
    mk = e.mean((0, 1))
    return e.max(2).mean((0, 1)) / e[:, :, -1].mean((0, 1)), (mk[1:] < mk[:-1]).mean(0)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--era5", required=True)
    p.add_argument("--arms", nargs="+", required=True, help="prefixes <prefix>_ens_<year>.npz")
    p.add_argument("--ref", default=None, help="the arm the bootstrap differences are taken against")
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--clim-years", type=int, nargs="+", default=list(range(2006, 2016)))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    import xarray as xr
    with xr.open_dataset(os.path.join(a.era5, "2m_temperature", f"2m_temperature_{a.test_years[0]}_5.625deg.nc")) as d:
        lat = d["lat"].values
    mx, mn = climode_minmax(a.era5)
    terc = tercile_thresholds(a.era5, a.clim_years)
    per = {}
    for arm in a.arms:
        ys = [score_year(arm, y, a.era5, terc, lat, mx, mn) for y in a.test_years]
        cat = lambda k: np.concatenate([q[k] for q in ys])  # noqa: E731
        per[arm] = {k: cat(k) for k in ("rmse", "spread2", "crps", "brier_lower", "brier_upper", "obs_freq_lower",
                                        "obs_freq_upper", "zt_corr_member", "zt_corr_truth", "lag24_member",
                                        "lag24_truth")}
        per[arm]["rank_hist"] = np.mean([q["rank_hist"] for q in ys], 0)
        per[arm]["power_ratio"] = {b: np.mean([q["power_ratio"][b] for q in ys], 0) for b in BANDS}
        per[arm]["M"], per[arm]["leads"] = ys[0]["M"], ys[0]["leads"]
        per[arm]["doob_member"], per[arm]["falling_member"] = doob(np.concatenate([q["e_member"] for q in ys]))
        per[arm]["doob_error"], _ = doob(np.concatenate([q["e_error"] for q in ys]))
    res = {"arms": {}, "vars": VARS}
    for arm, r in per.items():
        M, leads = r["M"], r["leads"]
        rh = r["rank_hist"]
        bs_clim_lo = r["obs_freq_lower"].mean(0) * (1 - r["obs_freq_lower"].mean(0))
        bs_clim_hi = r["obs_freq_upper"].mean(0) * (1 - r["obs_freq_upper"].mean(0))
        d = {"crps": r["crps"].mean(0), "rmse": r["rmse"].mean(0),
             "ssr": np.sqrt(r["spread2"].mean(0) * (M + 1) / M) / np.sqrt((r["rmse"] ** 2).mean(0)),
             "brier_lower": r["brier_lower"].mean(0), "brier_upper": r["brier_upper"].mean(0),
             "bss_lower": 1 - r["brier_lower"].mean(0) / bs_clim_lo, "bss_upper": 1 - r["brier_upper"].mean(0) / bs_clim_hi,
             "outside_range": rh[..., 0] + rh[..., -1], "outside_range_perfect": 2 / (M + 1),
             "rank_tv_from_flat": 0.5 * np.abs(rh - 1 / (M + 1)).sum(-1),
             "zt_corr_member": r["zt_corr_member"].mean(0), "zt_corr_truth": r["zt_corr_truth"].mean(0),
             "lag24_member": r["lag24_member"].mean(0), "lag24_truth": r["lag24_truth"].mean(0),
             "doob_ratio_member": r["doob_member"], "falling_steps_member": r["falling_member"],
             "doob_ratio_error": r["doob_error"]}
        d.update({f"power_ratio_{b}": v for b, v in r["power_ratio"].items()})
        res["arms"][os.path.basename(arm)] = {k: np.asarray(v).tolist() for k, v in d.items()}
        res["arms"][os.path.basename(arm)]["leads"] = np.asarray(leads).tolist()
    if a.ref:
        rng = np.random.default_rng(1)
        ref = per[a.ref]
        n = len(ref["crps"])
        idx = rng.integers(0, n, (a.n_boot, n))
        res["bootstrap_vs_" + os.path.basename(a.ref)] = {}
        for arm, r in per.items():
            if arm == a.ref:
                continue
            out = {}
            for k in ("crps", "rmse"):
                diff = r[k] - ref[k]  # (n, L, 5), paired by IC
                boot = diff[idx].mean(1)  # (n_boot, L, 5)
                out[k] = {"mean": diff.mean(0).tolist(), "lo": np.quantile(boot, 0.025, 0).tolist(),
                          "hi": np.quantile(boot, 0.975, 0).tolist()}
            res["bootstrap_vs_" + os.path.basename(a.ref)][os.path.basename(arm)] = out
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for arm, d in res["arms"].items():
        j = VARS.index("z")
        print(f"[{arm}] z500 CRPS " + " ".join(f"{v[j]:.4f}" for v in d["crps"]) + " | SSR " +
              " ".join(f"{v[j]:.2f}" for v in d["ssr"]) + " | BSS lo/hi d1 " +
              f"{d['bss_lower'][0][j]:.2f}/{d['bss_upper'][0][j]:.2f} d6 {d['bss_lower'][-1][j]:.2f}/{d['bss_upper'][-1][j]:.2f}"
              + " | outside " + " ".join(f"{v[j]:.2f}" for v in d["outside_range"]) + f" (perfect {d['outside_range_perfect']:.2f})"
              + " | power k13-32 " + " ".join(f"{v[j]:.2f}" for v in d["power_ratio_k13-32"])
              + " | z-t corr " + " ".join(f"{m:.2f}/{t:.2f}" for m, t in zip(d["zt_corr_member"], d["zt_corr_truth"]))
              + f" | Doob member {d['doob_ratio_member'][j]:.2f} error {d['doob_ratio_error'][j]:.2f}",
              flush=True)
    return res


if __name__ == "__main__":
    main()
