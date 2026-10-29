"""Step 14 figures (the owner's H-dyn, H-cl, H-res; docs/step14_hdyn.md), 2 x 2 panels by lead, persistence dotted.

    fig_hdyn.png          rollout ensembles from isotropic IC perturbations (scripts/run_step14_hdyn.py) against the
                          step 13 baseline: z500 / t850 ensemble-mean RMSE, z500 CRPS, z500 spread-error ratio
    fig_hdyn_flowdep.png  the owner's flow-dependence criteria (scripts/diag_step13_flowdep.py): noise-removed
                          case-to-case variation of the grid-point spread (pooled over the grid, so without the
                          floor of the per-point clipped estimate), of the domain spread, the RMS error of
                          the largest-spread fifth over the smallest, for z500 and t850; the owner's pass and
                          rejection thresholds in grey
    fig_hcl.png           diagnostic-closure operators (scripts/run_step13_diag.py): t2m, u10, v10 RMSE and the
                          change against the regression mean mu*
    fig_hres.png          the cores at 32 x 64, zero-shot at 64 x 128 and interpolated (scripts/run_step14_hres.py)
    fig_hbal.png          balance projections (scripts/run_step13_pinn.py gates on 2016) against the size of the
                          perturbation they correct instead of the lead: geostrophic and hydrostatic residual over
                          ERA5's, the spread they keep, the fraction of negative moisture; the input dotted

    python scripts/plot_step14.py --era5 /home/claude/era5/era5_data --out docs/step14
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.plot_step13 import GRID, INK, LEADS_H, MUTED, SLOTS, persistence  # noqa: E402
from scripts.run_step11_climode import VARS, climode_minmax  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402

S13 = "seqfm_dct_pinn_"
ARMS = [  # (score key, label, line style, slot): color = core, dashed = fine-tuned on the fair CRPS of perturbed rollouts
    ("core_jump6", "ResNet core (MSE), isotropic IC perturbations, no Γ", "-", 0),
    ("core_jump6_crps", "ResNet core + fair-CRPS fine-tune", "--", 0),
    ("core_jump6_fno", "FNO core, unconstrained", "-", 1),
    ("core_jump6_fno_crps", "FNO core, unconstrained + fair-CRPS fine-tune", "--", 1),
    ("core_jump6_fno_b2", "FNO core, norm-projected B = 2", "-", 3),
    ("core_jump6_fno_b2_crps", "FNO core, B = 2 + fair-CRPS fine-tune", "--", 3),
    ("core_jump6_fno_b1", "FNO core, norm-projected B = 1", "-", 2),
    (S13 + "regm_sband_clim_flow", "Step 13 best: regm_sband_clim_flow (FM + Γ + σ field)", "-.", 6),
]
DAYS = np.array(LEADS_H) / 24


def style(a_, title, ylab):
    a_.set_title(title, fontsize=12)
    a_.set_xlabel("Forecast lead (days)")
    a_.set_ylabel(ylab)
    a_.set_xlim(1, 6)
    a_.grid(True, color=GRID, lw=0.8)
    for s_ in ("top", "right"):
        a_.spines[s_].set_visible(False)


def finish(fig, ax, title, path, ncol=3):
    h, l_ = [], []
    for a_ in ax.ravel():
        for hh, ll in zip(*a_.get_legend_handles_labels()):
            if ll not in l_:
                h.append(hh), l_.append(ll)
    fig.legend(h, l_, loc="outside lower center", ncol=ncol, fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")
    return path


def fig_hdyn(arms, flow, p_rmse, p_crps, mx, mn, out):
    iz, it = VARS.index("z"), VARS.index("t")
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5), layout="constrained")
    panels = [(ax[0, 0], "Geopotential at 500 hPa", "Ensemble-mean RMSE (m²/s²)",
               lambda r: np.asarray(r["rmse"])[:, iz], p_rmse[:, iz]),
              (ax[0, 1], "Temperature at 850 hPa", "Ensemble-mean RMSE (K)",
               lambda r: np.asarray(r["rmse"])[:, it], p_rmse[:, it]),
              (ax[1, 0], "Geopotential at 500 hPa", "Ensemble CRPS (m²/s²)",
               lambda r: np.asarray(r["crps"])[:, iz] * (mx - mn)[iz], p_crps[:, iz] * (mx - mn)[iz]),
              (ax[1, 1], "Geopotential at 500 hPa", "Spread / RMSE (SSR)", lambda r: np.asarray(r["ssr"])[:, iz],
               None)]
    for a_, title, ylab, get, pers in panels:
        for k, lab, ls, slot in ARMS:
            if k in arms:
                a_.plot(DAYS, get(arms[k]), ls, color=SLOTS[slot], lw=1.8, label=lab)
        if pers is not None:
            a_.plot(DAYS, pers, ":", color=MUTED, lw=2, label="Persistence")
        style(a_, title, ylab)
    ax[1, 1].axhspan(0.9, 1.2, color=GRID, alpha=0.6, lw=0, label="Owner's H-dyn range 0.9-1.2 (no Γ)")
    ax[1, 1].set_ylim(0, 1.4)
    return finish(fig, ax, "Step 14 H-dyn: spread made by the rollout of a learned vector field\n"
                  "Real ERA5 5.625°, test 2017-2018 (ClimODE's 360 ICs), 8 members", os.path.join(out, "fig_hdyn.png"))


def fig_flowdep(flow, out):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5), layout="constrained")
    ratio = lambda r: r["bin_rmse"][4] / r["bin_rmse"][0]  # noqa: E731
    panels = [(ax[0, 0], "z500: case-to-case variation of the grid-point spread", "Noise-removed CV",
               "z500", lambda r: r["cv_spread_true_pooled"], (0.08, 0.05)),
              (ax[0, 1], "z500: case-to-case variation of the domain spread", "Noise-removed CV",
               "z500", lambda r: r["dom_spread_cv_true"], (0.06, 0.05)),
              (ax[1, 0], "z500: error of the largest-spread fifth / smallest", "RMS error ratio",
               "z500", ratio, (1.2, 1.1)),
              (ax[1, 1], "t850: error of the largest-spread fifth / smallest", "RMS error ratio",
               "t850", ratio, (1.2, 1.1))]
    for a_, title, ylab, v, get, (ok, rej) in panels:
        for k, lab, ls, slot in ARMS:
            if k in flow:
                a_.plot(DAYS, [get(flow[k][f"{v}_{h}"]) for h in LEADS_H], ls, color=SLOTS[slot], lw=1.8,
                        marker="o", ms=4, label=lab)
        a_.axhline(ok, color=MUTED, lw=1.2, ls="--", label="Owner's pass level")
        a_.axhline(rej, color=MUTED, lw=1.2, ls="-.", label="Owner's rejection level")
        style(a_, title, ylab)
    for k, lab, ls, slot in ARMS[:1]:  # the error's own case-to-case variation, for scale
        if k in flow:
            ax[0, 1].plot(DAYS, [flow[k][f"z500_{h}"]["dom_rmse_cv"] for h in LEADS_H], "-", color=INK, lw=1,
                          label="Domain RMSE CV of the ResNet core (scale)")
    return finish(fig, ax, "Step 14 H-dyn: is the spread flow dependent? (grey: the owner's criteria)\n"
                  "Real ERA5 5.625°, test 2017-2018, 8 members split 4 + 4 to remove the sampling noise",
                  os.path.join(out, "fig_hdyn_flowdep.png"))


def fig_hcl(diag, out):
    test = diag["test"]
    lines = [("persistence", "Persistence", ":", None), ("mean", "Regression ensemble mean", "-", 4),
             ("mbmx", "μ* = a + b·mean + c·x(t) (MBM)", "-", 7), ("diag_op1", "Diagnostic operator on μ*, B ≤ 1", "-", 2),
             ("diag_op", "Diagnostic operator, unconstrained", "--", 2), ("diag_cnn", "Diagnostic CNN on μ*", "-", 0),
             ("all_op1", "Operator on all 5 variables, B ≤ 1", "--", 1)]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5), layout="constrained")
    for a_, v, unit in ((ax[0, 0], "t2m", "K"), (ax[0, 1], "u10", "m/s"), (ax[1, 0], "v10", "m/s")):
        for k, lab, ls, slot in lines:
            if k in test:
                a_.plot(DAYS, [test[k]["rmse"][str(h)][v] for h in LEADS_H], ls,
                        color=MUTED if slot is None else SLOTS[slot], lw=2 if slot is None else 1.8, label=lab)
        style(a_, {"t2m": "2 m temperature", "u10": "10 m u wind", "v10": "10 m v wind"}[v], f"RMSE ({unit})")
    a_ = ax[1, 1]
    for k, lab, ls, slot in lines[3:]:
        if k in test:
            for v, mk in (("t2m", "o"), ("u10", "s"), ("v10", "^")):
                a_.plot(DAYS, [100 * (test[k]["rmse"][str(h)][v] / test["mbmx"]["rmse"][str(h)][v] - 1)
                               for h in LEADS_H], ls, color=SLOTS[slot], lw=1.4, marker=mk, ms=4)
    a_.axhline(-5, color=MUTED, lw=1.2, ls="--", label="Owner's pass level (−5 %)")
    a_.axhline(-2, color=MUTED, lw=1.2, ls="-.", label="Owner's rejection level (−2 %)")
    a_.axhline(0, color=INK, lw=0.8)
    style(a_, "RMSE change against μ* (o t2m, □ u10, △ v10)", "%")
    return finish(fig, ax, "Step 14 H-cl: diagnostic-closure operator on the regression mean μ*\n"
                  "Real ERA5 5.625°, trained on 2016 ICs, test 2017-2018", os.path.join(out, "fig_hcl.png"))


def fig_hres(hres, out):
    tags = [("core_jump6", "ResNet", 0), ("core_jump6_fno", "FNO", 1), ("core_jump6_fno_b2", "FNO B = 2", 3)]
    kinds = [("coarse", "32 × 64 (training grid)", "-"), ("native", "64 × 128 zero-shot", "--"),
             ("interp", "32 × 64 run, interpolated", "-.")]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5), layout="constrained")
    first = next(iter(t for t, _, _ in tags if t in hres))
    for a_, v, q, ylab, title in ((ax[0, 0], "z", "rmse", "RMSE (m²/s²)", "Geopotential at 500 hPa"),
                                  (ax[0, 1], "t", "rmse", "RMSE (K)", "Temperature at 850 hPa"),
                                  (ax[1, 0], "z", "skill", "1 − RMSE / RMSE persistence", "z500 skill"),
                                  (ax[1, 1], "t2m", "skill", "1 − RMSE / RMSE persistence", "t2m skill")):
        for t, lab, slot in tags:
            if t not in hres:
                continue
            for k, klab, ls in kinds:
                a_.plot(DAYS, [hres[t][k][q][str(h)][v] for h in LEADS_H], ls, color=SLOTS[slot], lw=1.8,
                        label=f"{lab}: {klab}")
        if q == "rmse":
            for k, ls in (("persist_c", ":"), ("persist_f", (0, (1, 3)))):
                a_.plot(DAYS, [hres[first][k]["rmse"][str(h)][v] for h in LEADS_H], ls=ls, color=MUTED, lw=2,
                        label="Persistence " + ("32 × 64" if k == "persist_c" else "64 × 128"))
        else:
            a_.axhline(0, color=MUTED, lw=2, ls=":", label="Persistence")
        style(a_, title, ylab)
    return finish(fig, ax, "Step 14 H-res: cores trained at 32 × 64 run zero-shot on 64 × 128 ERA5\n"
                  "Test 2017-2018 (ClimODE's 360 ICs), each against the truth at its own resolution",
                  os.path.join(out, "fig_hres.png"))


HBAL = [  # (gate json in runs/step13, label, slot)
    ("pinn_project_geo", "PINN projector (step 13, 8 × 16 DCT modes)", 0),
    ("pibal_project_b0.8_g1_inmass", "Π = pos(x + N), full-mode FNO, B = 0.8, geostrophic loss", 1),
    ("pibal_project_b0.8_g0", "Π, B = 0.8, no geostrophic loss", 3),
    ("pibal_project_b1.4_g1", "Π, B = 1.4, geostrophic loss", 4),
    ("pibal_project_b0_g1", "Π, no norm bound, geostrophic loss", 2),
]
ALPHAS = (0.25, 0.5, 1.0)


def fig_hbal(gates, out):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5), layout="constrained")
    first = next(iter(gates.values()))  # the inputs are the same perturbations for every projector
    for a_, c, title in ((ax[0, 0], "geo", "Geostrophic residual (ageostrophic wind, 200-850 hPa, |lat| ≥ 20°)"),
                         (ax[0, 1], "hyd", "Hydrostatic residual")):
        a_.axhline(1, color=MUTED, lw=2, ls=":", label="Not projected")
        for k, lab, slot in HBAL:
            if k in gates:
                v = gates[k]
                a_.plot((0,) + ALPHAS, [v["identity"]["residual_ratio"][c]]
                        + [v[f"alpha_{x}"]["residual_ratio_out"][c] / v[f"alpha_{x}"]["residual_ratio_in"][c]
                           for x in ALPHAS], "-o", color=SLOTS[slot], lw=1.8, ms=5, label=lab)
        a_.set_title(title, fontsize=12), a_.set_ylabel("RMS residual after Π / before")
    a_ = ax[1, 0]
    for k, lab, slot in HBAL:
        if k in gates:
            for b, ls in (("high", "-o"), ("low", "--o")):
                a_.plot(ALPHAS, [gates[k][f"alpha_{x}"]["spread_ratio"][b] for x in ALPHAS], ls, color=SLOTS[slot],
                        lw=1.8, ms=5)
    for y, lab in ((0.95, "Owner's pass (≥ 0.95)"), (0.9, "Owner's rejection (< 0.9)")):
        a_.axhline(y, color=MUTED, lw=1.5, ls="-." if y == 0.9 else "--", label=lab)
    a_.set_title("Spread kept (solid: DCT modes above 8 × 16, dashed: the low modes)", fontsize=12)
    a_.set_ylabel("Spread of Π(perturbed) / spread of the perturbations")
    a_ = ax[1, 1]
    a_.plot(ALPHAS, [first[f"alpha_{x}"]["negative_fraction_in"] for x in ALPHAS], ":", color=MUTED, lw=2.5, zorder=5)
    for k, lab, slot in HBAL:
        if k in gates:
            a_.plot(ALPHAS, [gates[k][f"alpha_{x}"]["negative_fraction_out"] for x in ALPHAS], "-o", color=SLOTS[slot],
                    lw=1.8, ms=5)
    a_.set_title("Negative moisture (q at 12 levels, tcwv)", fontsize=12), a_.set_ylabel("Fraction of grid values < 0")
    for a_ in ax.ravel():
        a_.set_xlabel("Perturbation size α (0 = ERA5 truth)")
        a_.grid(True, color=GRID, lw=0.8)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    return finish(fig, ax, "Step 14 H-bal: full-mode norm-projected FNO projection Π against the PINN projector\n"
                  "Validation 2016, 32 truths; α scales the FM-like perturbation of the 6 h sequence (α = 0: ERA5 itself)",
                  os.path.join(out, "fig_hbal.png"), ncol=2)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--era5", required=True)
    p.add_argument("--runs", default="runs/step14")
    p.add_argument("--step13", default="runs/step13")
    p.add_argument("--out", default="docs/step14")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    j = lambda *q: json.load(open(os.path.join(*q))) if os.path.exists(os.path.join(*q)) else None  # noqa: E731
    lat = load_static(a.era5)[0]
    mx, mn = climode_minmax(a.era5)
    written = []
    arms = {}
    for s in (j(a.step13, "score_aband_relation.json"), j(a.runs, "score_hdyn.json"), j(a.runs, "score_hdyn_b.json"),
              j(a.runs, "score_hdyn_crps.json")):
        arms.update(s["arms"] if s else {})
    flow = {}
    for s in (j(a.runs, "flowdep_step13_base.json"), j(a.runs, "flowdep_hdyn.json"), j(a.runs, "flowdep_hdyn_b.json"),
              j(a.runs, "flowdep_hdyn_crps.json")):
        flow.update(s or {})
    if any(k in arms for k, *_ in ARMS[:-1]):
        p_rmse, p_crps = persistence(a.era5, [2017, 2018], lat, mx, mn)
        written.append(fig_hdyn(arms, flow, p_rmse, p_crps, mx, mn, a.out))
    if any(k in flow for k, *_ in ARMS[:-1]):
        written.append(fig_flowdep(flow, a.out))
    diag = j(a.step13, "diag_seqregress_pinn.json")
    if diag:
        written.append(fig_hcl(diag, a.out))
    hres = j(a.runs, "hres.json")
    if hres:
        written.append(fig_hres(hres, a.out))
    gates = {k: g["val"] for k, *_ in HBAL if (g := j(a.step13, k + ".json"))}
    if gates:
        written.append(fig_hbal(gates, a.out))
    return written


if __name__ == "__main__":
    main()
