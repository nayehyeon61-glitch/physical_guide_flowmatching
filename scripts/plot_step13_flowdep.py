"""Step 13 figure for the flow-dependent spread field (docs/step13_sequence.md 4.3), 2 x 2 panels by lead:

    top      ensemble CRPS change against regm_band (%, bootstrap 95 % interval over ICs), z500 and t850
    bottom   RMS error in the largest-spread fifth of cases / in the smallest-spread fifth (1 = the spread
             says nothing about the error), from scripts/diag_step13_flowdep.py --leads 24 48 72 96 120 144

    python scripts/plot_step13_flowdep.py --score runs/step13/score_flowop.json \
        --flowdep runs/step13/flowdep_all.json --out docs/step13
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
from scripts.plot_step13 import GRID, LEADS_H, MUTED, SLOTS  # noqa: E402

REF = "seqfm_dct_pinn_regm_band"
LINES = [  # (arm, label, categorical slot)
    ("seqfm_dct_pinn_regm_band", "regm_band: band Γ only (reported model)", 7),
    ("seqfm_dct_pinn_regm_flow", "regm_flow: + σ field, ClimODE uncertainty net", 0),
    ("seqfm_dct_pinn_regm_flowop", "regm_flowop: + σ field, pretrained operator fine-tuned", 2),
    ("seqregress_pinn", "Regression + statistical errors (fixed covariance)", 5),
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--score", required=True, help="score_step13.py json with --ref " + REF)
    p.add_argument("--flowdep", required=True, help="diag_step13_flowdep.py json")
    p.add_argument("--out", default="docs/step13")
    a = p.parse_args(argv)
    sc, fd = json.load(open(a.score)), json.load(open(a.flowdep))
    V = sc["vars"]
    boot = sc[f"bootstrap_vs_{REF}"]
    base = np.asarray(sc["arms"][REF]["crps"])
    days = np.array(LEADS_H) / 24
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.6), layout="constrained")
    for col, (v, name) in enumerate((("z", "Geopotential at 500 hPa"), ("t", "Temperature at 850 hPa"))):
        vi = V.index(v)
        a_ = ax[0, col]
        a_.axhline(0, color=SLOTS[7], lw=1.8, label=LINES[0][1])
        for arm, lab, slot in LINES[1:3]:
            if arm in boot:
                b = {k: 100 * np.asarray(boot[arm]["crps"][k])[:, vi] / base[:, vi] for k in ("mean", "lo", "hi")}
                a_.fill_between(days, b["lo"], b["hi"], color=SLOTS[slot], alpha=0.18, lw=0)
                a_.plot(days, b["mean"], "-", color=SLOTS[slot], lw=1.8, marker="o", ms=4, label=lab)
        a_.set_title(name, fontsize=13)
        a_.set_ylabel("Ensemble CRPS change vs regm_band (%)")
        a_ = ax[1, col]
        key = "z500" if v == "z" else "t850"
        for arm, lab, slot in LINES:
            if arm in fd:
                leads = [h for h in LEADS_H if f"{key}_{h}" in fd[arm]]
                r = [fd[arm][f"{key}_{h}"]["bin_rmse"] for h in leads]
                a_.plot(np.array(leads) / 24, [q[4] / q[0] for q in r], "--" if arm == "seqregress_pinn" else "-",
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
    fig.suptitle("Step 13: flow-dependent spread field on top of the reported model (same mean, same band Γ)\n"
                 "Real ERA5 5.625°, test 2017-2018 (ClimODE's 360 ICs), 8 members; σ fitted on 2016", fontsize=13)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "fig_flowdep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


if __name__ == "__main__":
    main()
