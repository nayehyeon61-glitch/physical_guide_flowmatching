"""Step 22b: probabilistic quality and member realism on the ICs of a reference run (ClimODE's checkpoint).

The paired comparison (compare_step13_climode.py) gives RMSE / CRPS. This adds, on exactly the same ICs,
what a deterministic model cannot offer and what ClimODE's Gaussian sigma offers badly:

  - spread-skill ratio (latitude-weighted RMS spread / RMSE of the mean) per lead and variable;
  - tercile Brier scores and skill (events "below the lower / above the upper climatological tercile of the
    calendar month", thresholds from --clim-years as in score_step13.py) with paired bootstrap CIs vs the
    reference; the reference's Gaussian probabilities come from its predicted sigma;
  - rank histograms (ensembles, M + 1 bins) and PIT histograms (Gaussian reference, same number of bins),
    pooled over ICs and grid cells, summarised by the reliability index (sum |f_i - 1/(M+1)|);
  - zonal power spectra of z500 at --spec-leads: truth, members (mean spectrum), ensemble mean, reference
    mean; the ratio member / truth at wavenumbers 10-20 says whether members keep the weather-scale variance.

    python scripts/step22_prob_quality.py --era5 $E --climode runs/step22/climode_pred_34.npz \
        --arms runs/step20/seqfm_dct_pinn_blend_band runs/step13/seqfm_dct_pinn_regm_band runs/step13/seqregress_pinn \
        --out runs/step22/prob_quality_34.json --fig docs/step22/fig_prob_quality.png
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.report_step11 import boot_ci  # noqa: E402
from scripts.run_step11_climode import VARS, climode_truth  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402
from scripts.score_step13 import tercile_thresholds, zonal_power  # noqa: E402

LABELS = {"seqfm_dct_pinn_blend_band": "blend_band", "seqfm_dct_pinn_regm_band": "regm_band",
          "seqregress_pinn": "regression ensemble", "seqfm_dct_pinn_blend_lp0_band": "blend (data-only core)"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--era5", required=True)
    p.add_argument("--climode", default="runs/step22/climode_pred_34.npz")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--clim-years", type=int, nargs="+", default=list(range(2006, 2016)))
    p.add_argument("--spec-leads", type=int, nargs="+", default=[72, 144])
    p.add_argument("--ref-name", default="ClimODE")
    p.add_argument("--out", default="runs/step22/prob_quality.json")
    p.add_argument("--fig", default=None)
    a = p.parse_args(argv)

    cp = dict(np.load(a.climode))
    ics, leads = [int(i) for i in cp["ics"]], [int(h) for h in cp["lead_hours"]]
    years = [int(y) for y in cp["years"]] if "years" in cp else [2017, 2018]
    ref_det = bool(np.nanmax(cp["std"]) < 1e-3)  # deterministic reference (NeuralGCM): no sigma
    lat = load_static(a.era5)[0]
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None]
    terc = tercile_thresholds(a.era5, a.clim_years)  # (12, 2, 5, H, W)
    iz = VARS.index("z")
    rng = np.random.default_rng(0)

    per = {}  # name -> metric -> list over (year, ic) of (L, 5) arrays; hist -> (L, 5, bins)
    spec = {}  # name -> lead -> list of spectra (W // 2 + 1,) of z500
    spec_truth = {h: [] for h in a.spec_leads}

    def add(name, met, v):
        per.setdefault(name, {}).setdefault(met, []).append(np.asarray(v, np.float64))

    def brier(prob_lo, prob_hi, obs, th):
        lo_o, hi_o = (obs < th[:, 0]).astype(np.float64), (obs > th[:, 1]).astype(np.float64)
        return (((prob_lo - lo_o) ** 2) * w).mean((-1, -2)), (((prob_hi - hi_o) ** 2) * w).mean((-1, -2)), \
            (((1 / 3 - lo_o) ** 2) * w).mean((-1, -2)), (((1 / 3 - hi_o) ** 2) * w).mean((-1, -2))

    for yi, y in enumerate(years):
        truth = climode_truth(a.era5, y)
        times = pd.date_range(f"{y}-01-01", periods=len(truth), freq="6h")
        ens = {}
        for prefix in a.arms:
            f = f"{prefix}_ens_{y}.npz"
            if not os.path.exists(f):
                print(f"  {y}: {f} missing, skipped")
                continue
            z = np.load(f)
            rows = [int(np.flatnonzero(z["ics"] == i)[0]) for i in ics]
            lz = [int(h) for h in z["leads"]]
            E = z["E_minus_x0"][rows].astype(np.float32) + z["X0"][rows][:, None, None]  # (n, M, L, 5, H, W)
            ens[os.path.basename(prefix)] = (E, lz)
        for e, i0 in enumerate(ics):
            th = np.stack([terc[times[i0 + h // 6].month - 1] for h in leads])  # (L, 2, 5, H, W)
            obs = np.stack([truth[i0 + h // 6] for h in leads])  # (L, 5, H, W)
            mu, sd = cp["mean"][yi, e], cp["std"][yi, e]  # (L, 5, H, W)
            add("ref", "rmse2", (((mu - obs) ** 2) * w).mean((-1, -2)))
            add("ref", "mae", (np.abs(mu - obs) * w).mean((-1, -2)))
            if not ref_det:
                add("ref", "spread2", ((sd ** 2) * w).mean((-1, -2)))
                lo = norm.cdf((th[:, 0] - mu) / sd)
                hi = 1 - norm.cdf((th[:, 1] - mu) / sd)
            else:  # deterministic reference: its tercile "probability" is 0 or 1
                lo, hi = (mu < th[:, 0]).astype(np.float64), (mu > th[:, 1]).astype(np.float64)
            b_lo, b_hi, c_lo, c_hi = brier(lo, hi, obs, th)
            add("ref", "brier_lower", b_lo), add("ref", "brier_upper", b_hi)
            add("ref", "brier_clim_lower", c_lo), add("ref", "brier_clim_upper", c_hi)
            if not ref_det:
                pit = norm.cdf((obs - mu) / sd)  # (L, 5, H, W)
                per.setdefault("ref", {}).setdefault("pit", []).append(pit)
            for h in a.spec_leads:
                j = leads.index(h)
                spec.setdefault("ref", {}).setdefault(h, []).append(zonal_power(mu[j, iz]))
                spec_truth[h].append(zonal_power(obs[j, iz]))
            for name, (E, lz) in ens.items():
                ks = [lz.index(h) for h in leads]
                Em = E[e][:, ks]  # (M, L, 5, H, W)
                M = Em.shape[0]
                m = Em.mean(0)
                add(name, "rmse2", (((m - obs) ** 2) * w).mean((-1, -2)))
                add(name, "mae", (np.abs(m - obs) * w).mean((-1, -2)))
                add(name, "spread2", (Em.var(0, ddof=1) * w).mean((-1, -2)))
                lo = (Em < th[None, :, 0]).mean(0)
                hi = (Em > th[None, :, 1]).mean(0)
                b_lo, b_hi, c_lo, c_hi = brier(lo, hi, obs, th)
                add(name, "brier_lower", b_lo), add(name, "brier_upper", b_hi)
                add(name, "brier_clim_lower", c_lo), add(name, "brier_clim_upper", c_hi)
                add(f"{name}-ref", "brier_lower", b_lo - per["ref"]["brier_lower"][-1])
                add(f"{name}-ref", "brier_upper", b_hi - per["ref"]["brier_upper"][-1])
                below = (Em < obs[None]).sum(0) + rng.integers(0, 1 + (Em == obs[None]).sum(0))  # (L, 5, H, W)
                per.setdefault(name, {}).setdefault("rank", []).append((below, M))
                for h in a.spec_leads:
                    j = leads.index(h)
                    spec.setdefault(name, {}).setdefault(h, []).append(zonal_power(Em[:, j, iz]).mean(0))
                    spec.setdefault(name + " mean", {}).setdefault(h, []).append(zonal_power(m[j, iz]))
        print(f"{y}: {len(ics)} ICs done", flush=True)

    out = {"reference": a.ref_name, "reference_deterministic": ref_det, "ics": ics, "years": years, "leads": leads,
           "vars": VARS, "clim_years": a.clim_years, "n_ic": len(ics) * len(years), "scores": {}, "hist": {}, "spectra": {}}
    for name, d in per.items():
        s = {}
        if "rmse2" in d:
            r2 = np.stack(d["rmse2"])  # (n, L, 5)
            s["rmse"] = np.sqrt(r2.mean(0)).tolist()
            s["mae"] = np.stack(d["mae"]).mean(0).tolist()
        if "spread2" in d:
            s["spread"] = np.sqrt(np.stack(d["spread2"]).mean(0)).tolist()
            s["ssr"] = (np.sqrt(np.stack(d["spread2"]).mean(0)) / np.sqrt(r2.mean(0))).tolist()
        for k in ("brier_lower", "brier_upper"):
            if k in d:
                b = np.stack(d[k])
                s[k] = b.mean(0).tolist()
                if name.endswith("-ref"):
                    s[k + "_ci95"] = [[boot_ci(b[:, j, i]) for i in range(5)] for j in range(len(leads))]
                else:
                    c = np.stack(d[k.replace("brier", "brier_clim")]).mean(0)
                    s[k.replace("brier", "bss")] = (1 - b.mean(0) / c).tolist()
        out["scores"][name] = s
        if "rank" in d:
            M = d["rank"][0][1]
            hist = sum(np.stack([(b == r).sum((-1, -2)) for r in range(M + 1)], -1) for b, _ in d["rank"])  # (L, 5, M+1)
            f = hist / hist.sum(-1, keepdims=True)
            out["hist"][name] = {"bins": M + 1, "freq": f.tolist(), "reliability_index": np.abs(f - 1 / (M + 1)).sum(-1).tolist()}
        if "pit" in d:
            nb = next((d2["rank"][0][1] + 1 for d2 in per.values() if "rank" in d2), 9)
            pit = np.stack(d["pit"])  # (n, L, 5, H, W)
            hist = np.stack([np.stack([np.histogram(pit[:, j, i], bins=nb, range=(0, 1))[0] for i in range(5)])
                             for j in range(len(leads))])
            f = hist / hist.sum(-1, keepdims=True)
            out["hist"][name] = {"bins": nb, "freq": f.tolist(), "reliability_index": np.abs(f - 1 / nb).sum(-1).tolist()}
    for h in a.spec_leads:
        out["spectra"][str(h)] = {"truth": np.mean(spec_truth[h], 0).tolist()}
        for name, by_h in spec.items():
            out["spectra"][str(h)][name] = np.mean(by_h[h], 0).tolist()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)

    # -- console summary (z500, t850) ------------------------------------------------------------
    jl = [leads.index(h) for h in (24, 72, 144)]
    for name, s in out["scores"].items():
        if name.endswith("-ref"):
            continue
        row = []
        for j in jl:
            row.append(f"rmse {s['rmse'][j][iz]:.0f}" + (f" ssr {s['ssr'][j][iz]:.2f}" if "ssr" in s else "")
                       + (f" bss_up {s['bss_upper'][j][iz]:+.3f}" if "bss_upper" in s else ""))
        ri = out["hist"].get(name, {}).get("reliability_index")
        print(f"[{name}] z500 at 24/72/144 h: " + " | ".join(row)
              + (f" | reliability index {ri[jl[1]][iz]:.2f} (72 h)" if ri else ""))
    for h in a.spec_leads:
        sp = out["spectra"][str(h)]
        tr = np.array(sp["truth"])
        print(f"[spectra {h} h] member/truth power ratio at wavenumbers 10-20: "
              + ", ".join(f"{name} {np.array(v)[10:21].sum() / tr[10:21].sum():.2f}" for name, v in sp.items() if name != "truth"))

    if a.fig:
        plot(out, a.fig)
    return out


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.plot_step13 import GRID, INK, MUTED, SLOTS

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    leads = out["leads"]
    days = np.array(leads) / 24
    iz = out["vars"].index("z")
    arms = [n for n in out["scores"] if n not in ("ref",) and not n.endswith("-ref")]
    ref = out["reference"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    # (a) spread-skill
    a_ = ax[0, 0]
    for k, n in enumerate(arms):
        a_.plot(days, np.array(out["scores"][n]["ssr"])[:, iz], color=SLOTS[k], lw=1.8, label=LABELS.get(n, n))
    if "ssr" in out["scores"]["ref"]:
        a_.plot(days, np.array(out["scores"]["ref"]["ssr"])[:, iz], "-", color=INK, lw=1.2, marker="s", ms=5, label=f"{ref} (predicted σ)")
    a_.axhline(1, color=MUTED, ls=":", lw=1.5, label="perfect (1)")
    a_.set_title("z500 spread-skill ratio"), a_.set_xlabel("Forecast lead (days)"), a_.set_ylabel("RMS spread / RMSE")
    # (b) BSS upper tercile
    a_ = ax[0, 1]
    for k, n in enumerate(arms):
        a_.plot(days, np.array(out["scores"][n]["bss_upper"])[:, iz], color=SLOTS[k], lw=1.8, label=LABELS.get(n, n))
    if "bss_upper" in out["scores"]["ref"]:
        a_.plot(days, np.array(out["scores"]["ref"]["bss_upper"])[:, iz], "-", color=INK, lw=1.2, marker="s", ms=5,
                label=ref + (" (0/1 forecast)" if out.get("reference_deterministic") else ""))
    a_.axhline(0, color=MUTED, ls=":", lw=1.5, label="climatology (0)")
    a_.set_title("z500 Brier skill, event: above the monthly upper tercile"), a_.set_xlabel("Forecast lead (days)")
    a_.set_ylabel("BSS (1 = perfect)")
    # (c) rank / PIT histograms at 72 h
    a_ = ax[1, 0]
    j = leads.index(72) if 72 in leads else 0
    names = [arms[0]] + (["ref"] if "ref" in out["hist"] else [])
    width = 0.8 / len(names)
    for k, n in enumerate(names):
        f = np.array(out["hist"][n]["freq"])[j, iz]
        x = np.arange(len(f))
        a_.bar(x + (k - (len(names) - 1) / 2) * width, f, width, color=SLOTS[0] if n != "ref" else INK,
               label=(LABELS.get(n, n) + " rank histogram") if n != "ref" else f"{ref} PIT histogram")
    a_.axhline(1 / len(f), color=MUTED, ls=":", lw=1.5, label="uniform")
    a_.set_title(f"z500 at {leads[j]} h: where the observation falls"), a_.set_xlabel("bin (members + 1, or PIT decile)")
    a_.set_ylabel("frequency")
    # (d) spectra at the last spec lead
    a_ = ax[1, 1]
    h = list(out["spectra"])[-1]
    sp = out["spectra"][h]
    kk = np.arange(len(sp["truth"]))
    a_.loglog(kk[1:], np.array(sp["truth"])[1:], color=MUTED, lw=2.2, label="ERA5")
    for k, n in enumerate(arms):
        if n in sp:
            a_.loglog(kk[1:], np.array(sp[n])[1:], color=SLOTS[k], lw=1.8, label=f"{LABELS.get(n, n)} members")
        if n + " mean" in sp and k == 0:
            a_.loglog(kk[1:], np.array(sp[n + " mean"])[1:], "--", color=SLOTS[k], lw=1.5, label=f"{LABELS.get(n, n)} mean")
    if "ref" in sp:
        a_.loglog(kk[1:], np.array(sp["ref"])[1:], "-", color=INK, lw=1.2, marker="s", ms=3, label=f"{ref} mean")
    a_.set_title(f"z500 zonal power spectrum at {h} h"), a_.set_xlabel("zonal wavenumber"), a_.set_ylabel("power (m²/s²)²")
    for a_ in ax.flat:
        a_.legend(fontsize=8.5, frameon=False)
    fig.suptitle(f"Probabilistic quality on the same {out['n_ic']} ICs as the {ref} comparison "
                 f"({', '.join(str(y) for y in out['years'])})", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"[plot] {path}")


if __name__ == "__main__":
    main()
