"""Lead-time figure of blends and loops of one score file against a base model (owner 2026-11-16: KAI + the best
two-stage core, blended).

Four panels over the lead day, every change written as a gain over ``--base`` (+ = better):

  nRMSE gain            100 (1 - A / base) of the 65-channel normalized RMSE
  information gain      the same for the RMSE that is left at the best amplitude, sqrt(bias + s_o^2 (1 - r^2))
                        (scripts/eval_kai_breakdown.py terms2): a blend that only damps the field gains nothing here
  ACC difference        A - base (score files' definition: per forecast, then the mean)
  activity              pattern amplitude of the forecast / the truth's (1 = the truth's)

    python scripts/plot_stage_blend.py --out runs/kaitend --score score_st2.json --raw score_st2_raw.npz \
        --lines 'KAI + two-stage core (blend)=blend_st2c' 'two-stage core alone=st2c' ... --refs 'KAI + core_gb=blend_core_gb'

``--lines`` take the categorical colours in order, ``--refs`` are drawn in gray (dashed, dotted). The base itself is
the zero line of the first three panels and a gray line in the activity panel.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from eval_kai_breakdown import terms2  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CTRL = "#6f6e69"
REF_STYLES = [(0, (5, 2)), (0, (1, 1.5)), (0, (5, 1.5, 1, 1.5))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--score", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--stats", default="stats.npz")
    p.add_argument("--base", default="kai")
    p.add_argument("--base-name", default="KAI")
    p.add_argument("--lines", nargs="+", required=True, help="LABEL=ARM, coloured")
    p.add_argument("--refs", nargs="*", default=[], help="LABEL=ARM, gray")
    p.add_argument("--island", type=int, default=5, help="0 = no marker of the second stage")
    p.add_argument("--title", default=None)
    p.add_argument("--png", default="fig_stage_blend.png")
    p.add_argument("--json", default=None, help="also write the plotted numbers")
    a = p.parse_args()
    S = json.load(open(os.path.join(a.out, a.score)))
    raw = np.load(os.path.join(a.out, a.raw))
    std = np.load(os.path.join(a.out, a.stats))["std"].astype(np.float64) + 1e-8
    names = S["channels"]
    sc = np.array([i for i, n in enumerate(names) if n not in ("orog", "tisr")])
    L = int(S["leads"])
    days = np.arange(1, L + 1)

    def row(k):
        mse = raw[f"{k}__mse"].astype(np.float64).mean(0)                     # (L, 67)
        t = terms2(raw, k)                                                    # bias, information, amplitude cost
        return dict(nrmse=(np.sqrt(mse)[:, sc] / std[sc]).mean(1),
                    info=(np.sqrt(t[..., 0] + t[..., 1])[:, sc] / std[sc]).mean(1),
                    acc=raw[f"{k}__acc"].astype(np.float64).mean(0)[:, sc].mean(1),
                    act=raw[f"{k}__act"].astype(np.float64).mean(0)[:, sc].mean(1))

    base = row(a.base)
    spec = [(s.split("=", 1) + [COLORS[i], "-"]) for i, s in enumerate(a.lines)]
    spec += [(s.split("=", 1) + [CTRL, REF_STYLES[i]]) for i, s in enumerate(a.refs)]
    R = {arm: row(arm) for _, arm, _, _ in spec}
    panels = [("nRMSE gain over %s (%%, + = better)" % a.base_name, lambda r: 100 * (1 - r["nrmse"] / base["nrmse"]), True),
              ("information gain over %s (%%, + = better)" % a.base_name, lambda r: 100 * (1 - r["info"] / base["info"]), True),
              ("ACC minus %s (+ = better)" % a.base_name, lambda r: r["acc"] - base["acc"], True),
              ("activity (1 = truth)", lambda r: r["act"], False)]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), facecolor=SURFACE)
    handles = []
    for ax, (ttl, f, zero) in zip(axes.ravel(), panels):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if zero:
            ax.axhline(0, color=CTRL, lw=1.2)
        else:
            ax.plot(days, f(base), color=CTRL, lw=1.6)
            ax.annotate(a.base_name, (days[-1], f(base)[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=MUTED)
        for label, arm, col, ls in spec:
            h, = ax.plot(days, f(R[arm]), color=col, lw=2.0 if ls == "-" else 1.6, ls=ls,
                         marker="o" if ls == "-" else None, ms=4.5, mec=SURFACE, mew=1.0, label=label)
            if ax is axes[0, 0]:
                handles.append(h)
        if a.island:
            ax.axvline(a.island + 0.5, color=MUTED, lw=0.9, ls=(0, (2, 3)))
        ax.set_title(ttl, loc="left", fontsize=10.5, color=INK, pad=8)
        ax.set_xticks(days)
        ax.set_xlim(0.6, L + (0.9 if not zero else 0.4))
        ax.set_xlabel("lead (days)")
    if a.island:
        y0, y1 = axes[0, 0].get_ylim()
        axes[0, 0].text(a.island + 0.62, y1 - 0.03 * (y1 - y0), "second stage starts", fontsize=8.5, color=MUTED, va="top")
    fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)), frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(a.title or f"Blends with {a.base_name}: gains over {a.base_name} alone", x=0.06, y=0.985, ha="left",
                 fontsize=12.5, color=INK)
    rows_leg = (len(handles) + 2) // 3
    fig.tight_layout(rect=(0, 0.035 + 0.03 * rows_leg, 1, 0.97))
    fig.savefig(os.path.join(a.out, a.png), dpi=170, facecolor=SURFACE)
    print("wrote", os.path.join(a.out, a.png))
    out = {a.base: {k: [round(float(x), 4) for x in v] for k, v in base.items()}}
    for label, arm, _, _ in spec:
        out[arm] = dict(label=label, **{k: [round(float(x), 4) for x in v] for k, v in R[arm].items()})
        g = 100 * (1 - R[arm]["nrmse"] / base["nrmse"])
        gi = 100 * (1 - R[arm]["info"] / base["info"])
        print(f"{label:44s} gain {' '.join(f'{x:+5.1f}' for x in g)} | info {' '.join(f'{x:+5.1f}' for x in gi)}"
              f" | act {' '.join(f'{x:.2f}' for x in R[arm]['act'])}")
    if a.json:
        json.dump(out, open(os.path.join(a.out, a.json), "w"))


if __name__ == "__main__":
    main()
