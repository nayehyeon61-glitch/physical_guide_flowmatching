"""Step 13 figure for the flow-dependent spread field (docs/step13_sequence.md 4.3), 2 x 2 panels by lead:

    top      ensemble CRPS change against regm_band (%, bootstrap 95 % interval over ICs), z500 and t850
    bottom   RMS error in the largest-spread fifth of cases / in the smallest-spread fifth (1 = the spread
             says nothing about the error), from scripts/diag_step13_flowdep.py --leads 24 48 72 96 120 144

    python scripts/plot_step13_flowdep.py --score runs/step13/score_flowop30.json \
        --flowdep runs/step13/flowdep_all.json --out docs/step13
    python scripts/plot_step13_flowdep.py --set proposals --score runs/step13/score_aband_relation.json \
        --flowdep runs/step13/flowdep_aband_relation.json --out docs/step13      # fig_proposals.png

``--set proposals`` draws the owner's 2026-10-26 proposals instead (adaptive band Gamma, band x cell
multipliers with and without case adaptation, the cell climatology times regm_flow's case factor, the LOO
relation penalty), with regm_flow for reference.
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
from scripts.plot_step13 import GRID, INK, LEADS_H, MUTED, SLOTS  # noqa: E402

REF = "seqfm_dct_pinn_regm_band"
LINES = [  # (arm, label, categorical slot)
    ("seqfm_dct_pinn_regm_band", "regm_band: band Γ only", 7),
    ("seqfm_dct_pinn_regm_flow", "regm_flow: + σ field, ClimODE uncertainty net", 0),
    ("seqfm_dct_pinn_regm_flowop", "regm_flowop: + σ field, pretrained operator fine-tuned (30 epochs)", 2),
    ("seqfm_dct_pinn_regm_flowop10", "regm_flowop, 10-epoch fine-tune", 2),
    ("seqregress_pinn", "Regression + statistical errors (fixed covariance)", 6),
]
DASHED = ("seqfm_dct_pinn_regm_flowop10", "seqregress_pinn")
SETS = {
    "sigma": (LINES, DASHED, "fig_flowdep.png",
              "Step 13: flow-dependent spread field on top of regm_band (same mean, same band Γ)", None),
    "proposals": ([
        ("seqfm_dct_pinn_regm_band", "regm_band: band Γ only", 7),
        ("seqfm_dct_pinn_regm_flow", "regm_flow: grid-point σ field (ClimODE uncertainty net)", 0),
        ("seqfm_dct_pinn_regm_aband", "regm_aband: case-adaptive band Γ_b(c)", 3),
        ("seqfm_dct_pinn_regm_sband", "regm_sband: band × cell multipliers, case-adaptive", 2),
        ("seqfm_dct_pinn_regm_sband_clim", "regm_sband_clim: band × cell climatology only", 2),
        ("seqfm_dct_pinn_regm_sband_clim_flow", "regm_sband_clim_flow: band × cell climatology × regm_flow factor", 6),
        ("seqfm_dct_pinn_regm_rel", "regm_rel: LOO relation penalty on the members", 4),
    ], ("seqfm_dct_pinn_regm_sband_clim",), "fig_proposals.png",
        "Step 13: the owner's conditional-field proposals on top of regm_band (same mean)", (-1.8, 1.2)),
}  # (lines, dashed arms, file, title, CRPS-panel y-limits: points above are marked at the edge)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--score", required=True, help="score_step13.py json with --ref " + REF)
    p.add_argument("--flowdep", required=True, help="diag_step13_flowdep.py json")
    p.add_argument("--out", default="docs/step13")
    p.add_argument("--set", default="sigma", choices=list(SETS))
    a = p.parse_args(argv)
    lines, dashed, fname, title, ylim = SETS[a.set]
    sc, fd = json.load(open(a.score)), json.load(open(a.flowdep))
    V = sc["vars"]
    boot = sc[f"bootstrap_vs_{REF}"]
    base = np.asarray(sc["arms"][REF]["crps"])
    days = np.array(LEADS_H) / 24
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.6), layout="constrained")
    for col, (v, name) in enumerate((("z", "Geopotential at 500 hPa"), ("t", "Temperature at 850 hPa"))):
        vi = V.index(v)
        a_ = ax[0, col]
        a_.axhline(0, color=SLOTS[7], lw=1.8, label=lines[0][1])
        for arm, lab, slot in lines[1:]:
            if arm in boot:
                b = {k: 100 * np.asarray(boot[arm]["crps"][k])[:, vi] / base[:, vi] for k in ("mean", "lo", "hi")}
                if arm not in dashed:
                    a_.fill_between(days, b["lo"], b["hi"], color=SLOTS[slot], alpha=0.18, lw=0)
                a_.plot(days, b["mean"], "--" if arm in dashed else "-", color=SLOTS[slot], lw=1.8, marker="o", ms=4,
                        label=lab)
                if ylim is not None and b["mean"].max() > ylim[1]:
                    j = int(np.argmax(b["mean"]))
                    a_.annotate(f"{lab.split(':')[0]} off scale: {b['mean'][j]:+.1f} % at day {days[j]:g}",
                                (days[j], ylim[1]), xytext=(6, -14), textcoords="offset points", fontsize=9,
                                color=INK)
        if ylim is not None:
            a_.set_ylim(*ylim)
        a_.set_title(name, fontsize=13)
        a_.set_ylabel("Ensemble CRPS change vs regm_band (%)")
        a_ = ax[1, col]
        key = "z500" if v == "z" else "t850"
        for arm, lab, slot in lines:
            if arm in fd:
                leads = [h for h in LEADS_H if f"{key}_{h}" in fd[arm]]
                r = [fd[arm][f"{key}_{h}"]["bin_rmse"] for h in leads]
                a_.plot(np.array(leads) / 24, [q[4] / q[0] for q in r], "--" if arm in dashed else "-",
                        color=SLOTS[slot], lw=1.8, marker="o", ms=4, label=lab)
        a_.axhline(1, color=MUTED, ls=":", lw=2, label="No flow dependence")
        a_.set_title(name, fontsize=13)
        a_.set_ylabel("RMS error, largest-spread / smallest-spread fifth")
    for a_ in ax.flat:
        a_.set_xlabel("Forecast lead (days)")
        a_.set_xlim(1, 6)
        a_.grid(True, color=GRID, lw=0.8)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.legend(*ax[1, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"{title}\n"
                 "Real ERA5 5.625°, test 2017-2018 (ClimODE's 360 ICs), 8 members; σ fitted on 2016", fontsize=13)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


if __name__ == "__main__":
    main()
