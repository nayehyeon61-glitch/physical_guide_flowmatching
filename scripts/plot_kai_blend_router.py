"""Figure of the blend router test (scripts/kai_blend_router.py, docs/kai_blend_router.md).

    python scripts/plot_kai_blend_router.py runs/kaitend/router_A runs/kaitend/stats.npz docs/kai_blend_router/fig_router.png

Four panels by lead: the RMSE gain of every router over the static blend R0, the information part of that gain
(scripts/eval_kai_breakdown.py: the pattern term at the best amplitude), the activity of the blended forecast, and
the held-out loss of the gating networks by epoch (the static blend = the dashed line). + = better than R0.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from eval_kai_breakdown import CAT, GRID, INK, MUTED, SURFACE, breakdown  # noqa: E402

SERIES = [("R1g", "static, per grid point", CAT[0]), ("R2s", "gate, place only", CAT[1]),
          ("R2", "gate, sees forecasts (MoE)", CAT[2]), ("R2b", "MoE + bias map", CAT[3]),
          ("R0c", "static + climatology (damping)", CAT[4])]


def main(run, stats, out):
    S = json.load(open(os.path.join(run, "score_router.json")))
    raw = np.load(os.path.join(run, "score_router_raw.npz"))
    info = json.load(open(os.path.join(run, "router_info.json")))
    std = np.load(stats)["std"]
    L = S["leads"]
    x = np.arange(1, L + 1)
    sc = np.array([i for i, n in enumerate(S["channels"]) if n not in ("orog", "tisr")])
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.6))
    have = [s for s in SERIES if f"{s[0]}__mse" in raw]
    B = {k: breakdown(S, raw, std, k, "R0") for k, _, _ in have}

    def ends(p, ys):  # direct labels at the right end; lines that end together share one label
        lo, hi = p.get_ylim()
        gap = (hi - lo) * 0.055
        groups = []
        for name, y, _ in sorted(ys, key=lambda q: q[1]):
            if groups and y - groups[-1][1][-1] < gap * 0.6:
                groups[-1][0].append(name)
                groups[-1][1].append(y)
            else:
                groups.append(([name], [y]))
        pos = []
        for names, vals in groups:
            y = float(np.mean(vals))
            y = y if not pos or y - pos[-1] >= gap else pos[-1] + gap
            pos.append(y)
            p.text(L + 0.15, y, ", ".join(names), color=INK, fontsize=8.5, va="center")
        p.set_xlim(0.7, L + 3.4)

    val = {k: (B[k]["dterm2_pct"].sum(1), B[k]["dterm2_pct"][:, 1]) for k, _, _ in have}
    lim = (min(-0.3, min(v[i].min() for v in val.values() for i in (0, 1)) - 0.2),
           max(v[0].max() for v in val.values()) * 1.06)  # one scale: most of the long-lead gain is not information
    for i, (p, ttl) in enumerate(((ax[0, 0], "RMSE gain over the static blend R0 (%)"),
                                  (ax[0, 1], "information part of that gain (%)"))):
        ys = []
        for k, lab, c in have:
            p.plot(x, val[k][i], color=c, lw=2, marker="o", ms=4, label=f"{k}: {lab}")
            ys.append((k, val[k][i][-1], c))
        p.axhline(0, color=INK, lw=1)
        p.set_title(ttl, fontsize=10.5, loc="left")
        p.set_xlabel("lead (days)")
        p.grid(axis="y", color=GRID, lw=0.8)
        p.set_xticks(x)
        p.set_ylim(lim)
        ends(p, ys)
    ax[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")

    p, ys = ax[1, 0], []
    for k, lab, c in [("R0", "static blend R0", MUTED)] + have:
        y = raw[f"{k}__act"].mean(0)[:, sc].mean(1)
        p.plot(x, y, color=c, lw=2, marker="o", ms=4, ls="--" if k == "R0" else "-", label=f"{k}: {lab}")
        ys.append((k, y[-1], c))
    p.set_title("activity: pattern amplitude of the blended forecast / truth's", fontsize=10.5, loc="left")
    p.set_xlabel("lead (days)")
    p.set_ylim(0, 1.05)
    p.grid(axis="y", color=GRID, lw=0.8)
    p.set_xticks(x)
    ends(p, ys)

    p = ax[1, 1]
    col = dict((k, c) for k, _, c in SERIES)
    for k, v in info["variants"].items():
        if "holdout_curve" not in v or k.rstrip("c").split("_")[0] not in col or "c" in k.split("_")[0][2:]:
            continue
        base = k.split("_")[0]
        cv = np.array(v["holdout_curve"])
        p.plot(np.arange(1, len(cv) + 1), cv, color=col[base], lw=2, marker="o", ms=4,
               ls="-" if k == base else ":", label=f"{k} (used {v['epochs_used']} epochs)")
        st = v["holdout_static"]
    p.axhline(st, color=INK, lw=1, ls="--", label="static blend R0 on the held-out blocks")
    p.set_title("held-out loss of the gating networks (relative MSE) by epoch", fontsize=10.5, loc="left")
    p.set_xlabel("epoch")
    p.grid(axis="y", color=GRID, lw=0.8)
    p.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Blend router test, 5.625 deg: experts " + ", ".join(info["experts"])
                 + f" (fitted on 2016, {S['ics']} ICs of 2017)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:4])
