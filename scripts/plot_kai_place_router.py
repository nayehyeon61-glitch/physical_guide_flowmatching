"""Figure of the placement router test (scripts/kai_place_router.py, docs/kai_place_router.md).

    python scripts/plot_kai_place_router.py runs/kaitend/place_A runs/kaitend/stats.npz docs/kai_place_router/fig_place.png

Four panels by lead: the RMSE gain of every variant over the static blend R0, the information part of that gain
(scripts/eval_kai_breakdown.py), the activity, and how often each placement names the expert that really was
closer (the label accuracy). Placed (hard) variants are dashed, weighted (soft) ones solid. + = better than R0.
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

# tag, label, colour, dashed
SERIES = [("R1g", "static, per grid point", CAT[0], False), ("R1k", "static, per scale and latitude band", CAT[1], False),
          ("Pmap_soft", "label map, weighted", CAT[2], False), ("Pmap_hard", "label map, placed", CAT[2], True),
          ("NO_soft", "SFNO label router, weighted", CAT[5], False), ("NO_hard", "SFNO label router, placed", CAT[5], True),
          ("NOm", "SFNO router trained by MSE", CAT[3], False)]
ACC = [("Pmap", "label map (no network)", CAT[2]), ("NOs", "SFNO, place and lead only", CAT[0]),
       ("NO", "SFNO, sees the forecasts", CAT[5]), ("CN", "CNN, sees the forecasts", CAT[1]),
       ("Pclim", "rule: the one closer to climatology", CAT[4])]


def main(run, stats, out):
    S = json.load(open(os.path.join(run, "score_place.json")))
    raw = np.load(os.path.join(run, "score_place_raw.npz"))
    info = json.load(open(os.path.join(run, "place_info.json")))
    std = np.load(stats)["std"]
    L = S["leads"]
    x = np.arange(1, L + 1)
    sc = np.array([i for i, n in enumerate(S["channels"]) if n not in ("orog", "tisr")])
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.0))
    have = [s for s in SERIES if f"{s[0]}__mse" in raw]
    B = {s[0]: breakdown(S, raw, std, s[0], "R0") for s in have}
    val = {k: (B[k]["dterm2_pct"].sum(1), B[k]["dterm2_pct"][:, 1]) for k in B}
    lo = min(v[i].min() for v in val.values() for i in (0, 1))
    hi = max(v[i].max() for v in val.values() for i in (0, 1))
    lim = (lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))
    for i, (p, ttl) in enumerate(((ax[0, 0], "RMSE gain over the static blend R0 (%)"),
                                  (ax[0, 1], "information part of that gain (%)"))):
        for k, lab, c, dash in have:
            p.plot(x, val[k][i], color=c, lw=2, marker="o", ms=4, ls="--" if dash else "-", label=f"{k}: {lab}")
        p.axhline(0, color=INK, lw=1)
        p.set_title(ttl, fontsize=10.5, loc="left")
        p.set_xlabel("lead (days)")
        p.grid(axis="y", color=GRID, lw=0.8)
        p.set_xticks(x)
        p.set_ylim(lim)
    ax[0, 0].legend(frameon=False, fontsize=8.5, loc="lower left", ncol=1)

    p = ax[1, 0]
    extra = [("R0", "static blend R0", MUTED, False), (info["kai"], "KAI alone", INK, True), (info["core"], "core alone", INK, False)]
    for k, lab, c, dash in extra + have:
        y = raw[f"{k}__act"].mean(0)[:, sc].mean(1)
        p.plot(x, y, color=c, lw=1.4 if c in (INK, MUTED) else 2, marker="o", ms=3 if c in (INK, MUTED) else 4,
               ls="--" if dash else "-", label=lab if c in (INK, MUTED) else None)
    p.set_title("activity: pattern amplitude of the forecast / truth's", fontsize=10.5, loc="left")
    p.set_xlabel("lead (days)")
    p.set_ylim(0, 1.05)
    p.grid(axis="y", color=GRID, lw=0.8)
    p.set_xticks(x)
    p.legend(frameon=False, fontsize=8.5, loc="lower left")

    p = ax[1, 1]
    for k, lab, c in ACC:
        v = info["variants"].get(k, {})
        if "label_accuracy" in v:
            p.plot(x, 100 * np.array(v["label_accuracy"]), color=c, lw=2, marker="o", ms=4, label=f"{k}: {lab}")
    base = np.array(info["labels"]["core_closer_by_lead"])
    p.plot(x, 100 * np.maximum(base, 1 - base), color=MUTED, lw=1.4, ls=":", label="always the more frequent label")
    p.axhline(50, color=INK, lw=1)
    p.set_title("label accuracy: the placement names the expert that was closer (%)", fontsize=10.5, loc="left")
    p.set_xlabel("lead (days)")
    p.grid(axis="y", color=GRID, lw=0.8)
    p.set_xticks(x)
    p.legend(frameon=False, fontsize=8.5)
    fig.suptitle(f"Placement router test, 5.625 deg: {info['kai']} or {info['core']} (labels and routers fitted on 2016, "
                 f"{S['ics']} ICs of 2017)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:4])
