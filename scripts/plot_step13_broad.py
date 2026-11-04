"""Step 13 broad run (train 1979-2015) against ClimODE's paper and the 4-year step 13 run, same 360 test ICs.

Grouped bars of the deterministic RMSE at 24 / 72 / 144 h for z500, t850 and t2m (144 h only for t2m,
the only lead the 4-year table reports). ClimODE: paper Table 7 (docs/step11/climode_paper_tables.json).
4-year run: regm_band row of docs/step13_sequence.md 5.1 (train 2012-2015). Broad run: score_step13.py
output (regm_band, its mean is shared by regm_flow and regm_flowop).

    python scripts/plot_step13_broad.py --scores runs/step13_broad/score_best.json --out docs/step13_broad
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LEADS_H = (24, 48, 72, 96, 120, 144)
SHOW_H = (24, 72, 144)
VARS = ("z", "t", "t2m", "u10", "v10")
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# docs/step13_sequence.md 5.1, regm_band (a + b·μ_reg + c·x(t)), train 2012-2015
FOUR_YEAR = {"z": {24: 244, 72: 522, 144: 778}, "t": {24: 1.76, 72: 2.66, 144: 3.45}, "t2m": {144: 2.61}}
ARM = "seqfm_dct_pinn_regm_band"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--out", default="docs/step13_broad")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    r = np.asarray(json.load(open(a.scores))["arms"][ARM]["rmse"])  # (lead, var)
    paper = json.load(open(os.path.join(os.path.dirname(__file__), "..", "docs", "step11",
                                        "climode_paper_tables.json")))["rmse"]["ClimODE"]
    rows = [("ClimODE (paper)", INK, lambda v, h: paper[str(h)][v]),
            ("Step 13 regm_band, train 2012-2015", SLOTS[0], lambda v, h: FOUR_YEAR[v].get(h, np.nan)),
            ("Step 13 regm_band, train 1979-2015", SLOTS[7], lambda v, h: r[LEADS_H.index(h), VARS.index(v)])]
    panels = [("z", "Geopotential at 500 hPa", "RMSE (m²/s²)", SHOW_H, "{:.0f}"),
              ("t", "Temperature at 850 hPa", "RMSE (K)", SHOW_H, "{:.2f}"),
              ("t2m", "2 m temperature", "RMSE (K)", (144,), "{:.2f}")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6), layout="constrained", width_ratios=[3, 3, 1.4])
    w = 0.26
    for a_, (v, sub, ylab, hs, fmt) in zip(ax, panels):
        x = np.arange(len(hs))
        for i, (lab, col, get) in enumerate(rows):
            y = np.array([get(v, h) for h in hs], float)
            bars = a_.bar(x + (i - 1) * w, y, w - 0.02, color=col, label=lab, edgecolor="white", linewidth=1)
            for b, yv in zip(bars, y):
                if np.isfinite(yv):
                    a_.text(b.get_x() + b.get_width() / 2, yv, fmt.format(yv), ha="center", va="bottom",
                            fontsize=8, color=MUTED)
        a_.set_xticks(x, [f"{h} h" for h in hs])
        a_.set_title(sub, fontsize=13)
        a_.set_ylabel(ylab)
        a_.grid(True, axis="y", color=GRID, lw=0.8)
        a_.set_axisbelow(True)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=10, frameon=False)
    fig.suptitle("Deterministic RMSE on ClimODE's 360 test ICs (2017-2018), lower is better", fontsize=13)
    path = os.path.join(a.out, "fig_broad_vs_climode.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")
    return path


if __name__ == "__main__":
    main()
