"""Figures of the ratio routers (scripts/kai_place_router.py, docs/kai_place_router.md section 13).

    python scripts/plot_kai_ratio_router.py runs/kaitend/place_C runs/kaitend/stats.npz docs/kai_place_router/fig_ratio.png

fig_ratio.png, four panels by lead: the RMSE gain of every variant over the static blend R0, the information part of
that gain (scripts/eval_kai_breakdown.py), the activity, and the core's ratio per zonal-wavenumber group (static
R1k dashed, the SFNO ratio router NOk solid).  fig_ratio_lambda.png: the label as soft constraint, from the pure
ratio router (lam = 0, NOm) to the pure label router (NO used as weight): gain by lam, and the ratio by lead.
Static variants are dashed, learned ones solid; the colour follows the domain the ratio is given for. + = better.
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

GROUPS = ["k = 0", "k = 1-3", "k = 4-9", "k >= 10"]


def series(res, lam_best):
    # tag, label, colour, line style
    S = [("R1g", "static ratio per variable x grid point", CAT[0], "--"),
         ("NOm", "SFNO ratio per variable x grid point", CAT[0], "-"),
         (f"NOc{lam_best}", f"  + label soft constraint (lam {lam_best})", CAT[2], "-"),
         ("NO_soft", "SFNO label router used as weight", CAT[5], "-"),
         ("R1k", "static ratio per variable x scale x latitude band", CAT[1], "--"),
         ("NOk", "SFNO ratio per variable x scale x grid point", CAT[1], "-"),
         ("NOks", "  the same, sees place and lead only", CAT[4], "-"),
         ("NOkc0.3", "  + scale label soft constraint (lam 0.3)", CAT[3], "-")]
    return [s for s in S if f"{s[0]}__mse" in res]


def main(run, stats, out):
    S = json.load(open(os.path.join(run, "score_place.json")))
    raw = np.load(os.path.join(run, "score_place_raw.npz"))
    info = json.load(open(os.path.join(run, "place_info.json")))
    sw = np.load(os.path.join(run, "static_weights.npz"))
    std = np.load(stats)["std"]
    L = S["leads"]
    x = np.arange(1, L + 1)
    sc = np.array([i for i, n in enumerate(S["channels"]) if n not in ("orog", "tisr")])
    wl = np.cos(np.deg2rad(np.linspace(-87.1875, 87.1875, sw["R1k"].shape[2])))
    wl = wl / wl.mean()
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    lams = sorted(float(k[3:]) for k in info["variants"] if k.startswith("NOc"))
    gain = lambda k: breakdown(S, raw, std, k, "R0")["dterm2_pct"]  # noqa: E731
    lam_best = max(lams, key=lambda q: gain(f"NOc{q:g}").sum(1).mean()) if lams else 0
    have = series(raw, f"{lam_best:g}")
    val = {s[0]: (gain(s[0]).sum(1), gain(s[0])[:, 1]) for s in have}

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.0))
    lo = min(v[i].min() for v in val.values() for i in (0, 1))
    hi = max(v[i].max() for v in val.values() for i in (0, 1))
    lim = (lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))
    for i, (p, ttl) in enumerate(((ax[0, 0], "RMSE gain over the static blend R0 (%)"),
                                  (ax[0, 1], "information part of that gain (%)"))):
        for k, lab, c, ls in have:
            p.plot(x, val[k][i], color=c, lw=2, marker="o", ms=4, ls=ls, label=f"{k}: {lab}")
        p.axhline(0, color=INK, lw=1)
        p.set_title(ttl, fontsize=10.5, loc="left")
        p.set_xlabel("lead (days)")
        p.grid(axis="y", color=GRID, lw=0.8)
        p.set_xticks(x)
        p.set_ylim(lim)
    ax[0, 1].legend(*ax[0, 0].get_legend_handles_labels(), frameon=False, fontsize=8.2, loc="upper right")

    p = ax[1, 0]
    extra = [("R0", "static blend R0", MUTED, "-"), (info["kai"], "KAI alone", INK, "--"), (info["core"], "core alone", INK, "-")]
    for k, lab, c, ls in extra + have:
        y = raw[f"{k}__act"].mean(0)[:, sc].mean(1)
        thin = c in (INK, MUTED)
        p.plot(x, y, color=c, lw=1.4 if thin else 2, marker="o", ms=3 if thin else 4, ls=ls, label=lab if thin else None)
    p.set_title("activity: pattern amplitude of the forecast / truth's", fontsize=10.5, loc="left")
    p.set_xlabel("lead (days)")
    p.set_ylim(0, 1.05)
    p.grid(axis="y", color=GRID, lw=0.8)
    p.set_xticks(x)
    p.legend(frameon=False, fontsize=8.5, loc="lower left")

    p = ax[1, 1]
    wk = sw["R1k"]                                      # (L, C, H, M): constant inside a wavenumber group
    first = [0, 1, 4, 10]
    for gi, (name, c) in enumerate(zip(GROUPS, (CAT[0], CAT[2], CAT[3], CAT[5]))):
        p.plot(x, (wk[:, sc][:, :, :, first[gi]] * wl).mean((1, 2)), color=c, lw=1.6, ls="--")
        if "NOk" in info["variants"]:
            p.plot(x, info["variants"]["NOk"]["core_weight_by_group"][gi], color=c, lw=2, marker="o", ms=4, label=name)
    p.plot([], [], color=MUTED, lw=1.6, ls="--", label="static R1k")
    p.axhline(0.5, color=INK, lw=1)
    p.set_title("core's ratio by zonal wavenumber k (rest = KAI): SFNO ratio router NOk", fontsize=10.5, loc="left")
    p.set_xlabel("lead (days)")
    p.set_ylim(0, 1)
    p.grid(axis="y", color=GRID, lw=0.8)
    p.set_xticks(x)
    p.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    fig.suptitle(f"Ratio routers, 5.625 deg: {info['kai']} + ratio x ({info['core']} - {info['kai']}) (fitted on 2016, "
                 f"{S['ics']} ICs of 2017)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # ---- the label as soft constraint: lam 0 (NOm) ... lam -> infinity (NO used as weight)
    ends = [("NOm", "0\n(ratio only)")] + [(f"NOc{q:g}", f"{q:g}") for q in lams] + [("NO_soft", "inf\n(label only)")]
    ends = [e for e in ends if f"{e[0]}__mse" in raw]
    if len(ends) < 3:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    xs = np.arange(len(ends))
    g = np.array([gain(k).sum(1) for k, _ in ends])     # (variants, L)
    p = ax[0]
    for q, c in zip((0, 2, 4, 9), (CAT[0], CAT[2], CAT[3], CAT[5])):
        if q < L:
            p.plot(xs, g[:, q], color=c, lw=2, marker="o", ms=5, label=f"day {q + 1}")
    p.axhline(0, color=INK, lw=1)
    p.set_xticks(xs)
    p.set_xticklabels([e[1] for e in ends])
    p.set_xlabel("lam: weight of the label constraint on the ratio")
    p.set_title("RMSE gain over the static blend R0 (%)", fontsize=10.5, loc="left")
    p.grid(axis="y", color=GRID, lw=0.8)
    p.legend(frameon=False, fontsize=8.5)
    p = ax[1]
    p.plot(x, (sw["R0"][:, sc, 0, 0]).mean(1), color=MUTED, lw=1.6, ls="--", label="static blend R0")
    cols = [CAT[0], CAT[1], CAT[2], CAT[3], CAT[4], CAT[5]]
    for (k, lab), c in zip(ends, cols):
        v = info["variants"].get(k, {}).get("core_weight")
        if v is not None:
            p.plot(x, v, color=c, lw=2, marker="o", ms=4, label=f"lam {lab.splitlines()[0]}")
    base = info["labels"]["core_closer_by_lead"]
    p.plot(x, base, color=INK, lw=1.4, ls=":", label="share of points with the core closer (label)")
    p.axhline(0.5, color=INK, lw=1)
    p.set_ylim(0.3, 1)
    p.set_xticks(x)
    p.set_xlabel("lead (days)")
    p.set_title("core's ratio, mean over the variables and the globe", fontsize=10.5, loc="left")
    p.grid(axis="y", color=GRID, lw=0.8)
    p.legend(frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out.replace(".png", "_lambda.png"), dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:4])
