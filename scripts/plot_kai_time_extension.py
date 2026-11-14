"""Figures of the time-extension test (docs/kai_time_extension.md).

    python scripts/plot_kai_time_extension.py runs/kaitend docs/kai_time_extension

fig_gap.png    the generalization gap of the 4-year models (scripts/kai_gap_diag.py: diag_gap.json, diag_gap5.json):
               the one-step loss by year (trained years shaded) and the gap by lead of the models' own rollouts.
fig_clim.png   the climatology's trend (scripts/kai_clim_trend.py: diag_clim_trend.json): the global mean of t2m
               and t50 by year, the climatologies as forecasts of 2016 / 2017, the ACC of stored forecasts with the
               4-year and the 32-year climatology as reference.
fig_T<n>.png   (when score_T<n>.json is there) 2 x 2 by lead: gains of the time-extended models, + = better.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from eval_kai_breakdown import CAT, GRID, INK, MUTED, SURFACE  # noqa: E402


def style():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def lst(v):
    return v if isinstance(v, list) else [v]


def fig_gap(run, out):
    g1 = json.load(open(os.path.join(run, "diag_gap.json")))
    g5 = json.load(open(os.path.join(run, "diag_gap5.json")))
    name = {"core_gngmz8": "baseline core (0.26 M)", "core_gngmz8w": "wide core (0.52 M)", "kai": "KAI (0.43 M)",
            "kai_ro": "fine-tuned KAI (0.43 M)"}
    col = {"core_gngmz8": CAT[0], "core_gngmz8w": CAT[1], "kai": CAT[2], "kai_ro": CAT[3]}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    p = ax[0]
    src = {"core_gngmz8": g1, "core_gngmz8w": g1, "kai": g5}
    ys = sorted(int(y) for y in g1["core_gngmz8"]["years"])
    seen = [y for y in ys if g1["core_gngmz8"]["years"][str(y)]["trained"]]
    p.axvspan(min(seen) - 0.5, max(seen) + 0.5, color=GRID, alpha=0.6, lw=0)
    for k, s in src.items():
        v = [lst(s[k]["years"][str(y)]["loss"])[0] for y in ys]
        p.plot(ys, v, color=col[k], lw=2, marker="o", ms=5, label=name[k])
        tr = np.mean([q for q, y in zip(v, ys) if y in seen])
        un = np.mean([q for q, y in zip(v, ys) if y not in seen])
        p.text(ys[-1] + 0.25, v[-1], f"{name[k]}\nunseen / trained - 1: {100 * (un / tr - 1):+.1f} %", color=INK,
               va="center", fontsize=8.5)
    p.text(np.mean(seen), p.get_ylim()[1], "trained years", ha="center", va="top", color=MUTED, fontsize=9)
    p.set_xticks(ys)
    p.set_xlim(ys[0] - 0.6, ys[-1] + 4.6)
    p.set_xlabel("year the loss is measured on")
    p.set_title("one-step loss by year (relative L2)", fontsize=10.5, loc="left")
    p.grid(axis="y", color=GRID, lw=0.8)
    p = ax[1]
    for k in ("core_gngmz8w", "kai", "kai_ro"):
        gap = 100 * np.array(g5[k]["gap"])
        p.plot(np.arange(1, len(gap) + 1), gap, color=col[k], lw=2, marker="o", ms=5, label=name[k])
    p.plot([1], [100 * np.array(lst(g1["core_gngmz8"]["gap"]))[0]], color=col["core_gngmz8"], marker="o", ms=6, ls="",
           label=name["core_gngmz8"] + ", day 1")
    p.axhline(0, color=INK, lw=1)
    p.set_ylim(0, None)
    p.set_xticks(range(1, 6))
    p.set_xlabel("day of the model's own rollout")
    p.set_title("generalization gap: loss on unseen years / on trained years - 1 (%)", fontsize=10.5, loc="left")
    p.grid(axis="y", color=GRID, lw=0.8)
    p.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Are the 4-year models short of data? Trained on 2012-2015, measured on 2008-2017 (5.625 deg)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "fig_gap.png"), dpi=150)
    plt.close(fig)


def fig_clim(run, out):
    o = json.load(open(os.path.join(run, "diag_clim_trend.json")))
    names = o["channels"]
    gm = {int(y): np.array(v) for y, v in o["trend"]["global_mean"].items()}
    ys = np.array(sorted(gm))
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.0))
    per = [("1980-2011", CAT[0]), ("2002-2011", CAT[2]), ("2012-2015", CAT[1])]
    for p, ch, unit in ((ax[0, 0], "t2m", "K"), (ax[0, 1], "t50", "K")):
        i = names.index(ch)
        v = np.array([gm[y][i] for y in ys])
        p.plot(ys, v, color=INK, lw=1.6, marker="o", ms=3)
        for s, c in per:
            a, b = (int(q) for q in s.split("-"))
            m = v[(ys >= a) & (ys <= b)].mean()
            p.plot([a, b], [m, m], color=c, lw=3, solid_capstyle="butt", label=f"mean of {s}")
        a, b = 1980, 2011
        k = (ys >= a) & (ys <= b)
        sl, ic = np.polyfit(ys[k], v[k], 1)
        p.plot([a, 2017], [sl * a + ic, sl * 2017 + ic], color=CAT[0], lw=1.4, ls="--", label="linear trend of 1980-2011, to 2017")
        p.plot([2017], [v[-1]], color=CAT[5], marker="o", ms=8, ls="", label="2017 (test year)")
        p.set_title(f"global mean of {ch} by year ({unit})", fontsize=10.5, loc="left")
        p.grid(axis="y", color=GRID, lw=0.8)
        p.legend(frameon=False, fontsize=8.5, loc="upper left" if ch == "t2m" else "upper right")
    p = ax[1, 0]
    order = [("2012-2015", "2012-2015\n(4 y, used now)"), ("2002-2011", "2002-2011\n(10 y)"), ("1980-2011", "1980-2011\n(32 y)"),
             ("1980-2011+g", "32 y + trend\n(global mean)"), ("1980-2011+z", "32 y + trend\n(zonal mean)"),
             ("1980-2011+p", "32 y + trend\n(grid point)")]
    x = np.arange(len(order))
    for y, c in (("2016", CAT[3]), ("2017", CAT[5])):
        v = [o["clim"][y][k]["nrmse"] for k, _ in order]
        p.plot(x, v, color=c, ls="", marker="o", ms=9, label=f"against {y}")
        for xi, q in zip(x, v):
            p.text(xi + 0.14, q, f"{q:.4f}", ha="left", va="center", fontsize=8, color=INK)
    p.set_xlim(-0.5, len(order) - 0.3)
    p.set_xticks(x)
    p.set_xticklabels([q for _, q in order], fontsize=8)
    p.set_title("the climatology as a forecast: normalized RMSE (65 channels, lower = better)", fontsize=10.5, loc="left")
    p.grid(axis="y", color=GRID, lw=0.8)
    p.legend(frameon=False, fontsize=8.5, loc="center", bbox_to_anchor=(0.72, 0.5))
    p = ax[1, 1]
    lab = {"kai_ro": "fine-tuned KAI", "core_gngmz8w": "wide core", "persistence": "persistence"}
    for tg, c in (("core_gngmz8w", CAT[1]), ("kai_ro", CAT[3]), ("persistence", MUTED)):
        if tg not in o["forecasts"]:
            continue
        r = o["forecasts"][tg]
        L = np.arange(1, len(r["run"]["acc"]) + 1)
        p.plot(L, r["run"]["acc"], color=c, lw=2, marker="o", ms=4, label=f"{lab[tg]}, 4-year climatology (as scored)")
        p.plot(L, r["1980-2011"]["acc"], color=c, lw=2, ls="--", marker="o", ms=4, mfc=SURFACE, label=f"{lab[tg]}, 32-year climatology")
    p.set_xticks(L)
    p.set_ylim(0, 1)
    p.set_xlabel("lead (days)")
    p.set_title("ACC of the same forecasts with either climatology as reference", fontsize=10.5, loc="left")
    p.grid(axis="y", color=GRID, lw=0.8)
    p.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("The climatology's trend: climatologies of several periods against 2016 / 2017 (5.625 deg, daily means)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out, "fig_clim.png"), dpi=150)
    plt.close(fig)


def main(run, out):
    style()
    os.makedirs(out, exist_ok=True)
    fig_gap(run, out)
    fig_clim(run, out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
