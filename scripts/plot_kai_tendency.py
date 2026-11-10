"""Figures of the KAI vs KAI + tendency core comparison (scripts/run_kai_tendency.py --score).

    python scripts/plot_kai_tendency.py runs/kaitend docs/kai_tendency
fig_kai_lead.png   lead-time lines (normalized RMSE, z500, t850, t2m, q700 RMSE, z500 ACC), persistence dotted
fig_kai_where.png  relative RMSE change per channel x lead for each step of the ladder (x = not significant)
fig_kai_scale.png  error power and prediction/truth power by zonal wavenumber band (z500, t850, u850, q700)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STYLE = {"persistence": dict(color="k", ls=":", label="persistence"),
         "climatology": dict(color="0.6", ls="--", label="climatology"),
         "kai": dict(color="#1f77b4", label="KAI original (2000 it)"),
         "kai_long": dict(color="#1f77b4", ls="-.", label="KAI original (8000 it, same time as tend)"),
         "res": dict(color="#2ca02c", label="+ residual"),
         "tend1": dict(color="#ff7f0e", label="+ Heun/season/tisr/q>=0"),
         "tend": dict(color="#d62728", lw=2, label="+ rollout = tendency core")}


def main(run, out):
    os.makedirs(out, exist_ok=True)
    S = json.load(open(os.path.join(run, "score.json")))
    raw = np.load(os.path.join(run, "score_raw.npz"))
    names = S["channels"]
    L = S["leads"]
    days = np.arange(1, L + 1)
    arms = [k for k in STYLE if k in S["arms"]]

    # ---- 1. lead lines
    panels = [("nrmse", None, "normalized RMSE (65 ch)"), ("rmse", "z500", "z500 RMSE (m$^2$/s$^2$)"),
              ("rmse", "t850", "t850 RMSE (K)"), ("rmse", "t2m", "t2m RMSE (K)"),
              ("rmse", "q700", "q700 RMSE (kg/kg)"), ("acc", "z500", "z500 ACC")]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    for p, (key, ch, title) in zip(ax.flat, panels):
        for k in arms:
            v = np.array(S["arms"][k][key])
            y = v if ch is None else v[:, names.index(ch)]
            p.plot(days, y, **STYLE[k])
        p.set_title(title)
        p.set_xlabel("lead (days)")
        p.grid(alpha=0.3)
    ax.flat[0].legend(fontsize=7)
    fig.suptitle(f"KAI-alpha vs KAI + tendency core, ERA5 5.625 deg daily, test 2017 ({S['ics']} ICs), "
                 "train 2012-15, same backbone (0.43 M)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_kai_lead.png"), dpi=130)
    plt.close(fig)

    # ---- 2. where: ladder steps, relative RMSE change per channel x lead, paired bootstrap over ICs
    mse = {k: raw[f"{k}__mse"] for k in arms if f"{k}__mse" in raw}
    steps = [("res", "kai"), ("tend1", "res"), ("tend", "tend1"), ("tend", "kai"), ("tend", "kai_long")]
    steps = [s for s in steps if s[0] in mse and s[1] in mse]
    rows = [i for i, n in enumerate(names) if n not in ("orog", "tisr")]
    rng = np.random.default_rng(0)
    N = mse[arms[0]].shape[0] if arms else 0
    idx = rng.integers(0, N, (300, N))
    fig, ax = plt.subplots(1, len(steps), figsize=(3.3 * len(steps), 12), sharey=True)
    ax = np.atleast_1d(ax)
    summary = {}
    for p, (a, b) in zip(ax, steps):
        r = np.sqrt(mse[a].mean(0)) / np.sqrt(mse[b].mean(0)) - 1  # (L, 67)
        bs = np.stack([np.sqrt(mse[a][i].mean(0)) / np.sqrt(mse[b][i].mean(0)) - 1 for i in idx])
        lo, hi = np.percentile(bs, [2.5, 97.5], 0)
        sig = (lo > 0) | (hi < 0)
        im = p.imshow(100 * r[:, rows].T, cmap="RdBu_r", vmin=-30, vmax=30, aspect="auto")
        yy, xx = np.where(~sig[:, rows].T)
        p.scatter(xx, yy, marker="x", s=6, c="k", lw=0.5)
        p.set_title(f"{a} vs {b}\n(RMSE change %)", fontsize=9)
        p.set_xticks(range(L))
        p.set_xticklabels(days)
        p.set_xlabel("lead (days)")
        summary[f"{a}/{b}"] = {"better_cells": int(((r < 0) & sig)[:, rows].sum()),
                               "worse_cells": int(((r > 0) & sig)[:, rows].sum()), "cells": int(len(rows) * L)}
    ax[0].set_yticks(range(len(rows)))
    ax[0].set_yticklabels([names[i] for i in rows], fontsize=5.5)
    fig.colorbar(im, ax=ax, shrink=0.4, label="RMSE change % (blue = better)")
    fig.suptitle("Where each step changes the error (x = 95 % bootstrap CI contains 0)")
    fig.savefig(os.path.join(out, "fig_kai_where.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    json.dump(summary, open(os.path.join(out, "where_summary.json"), "w"), indent=1)

    # ---- 3. scales, from the full zonal spectra: band error power relative to KAI, and prediction / truth power
    bands = [(0, 0), (1, 3), (4, 9), (10, 20), (21, 32)]
    bcol = ["0.4", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    chans = ["z500", "t850", "u850", "q700", "t2m"]
    sp = {k: {q: raw[f"{k}__{q}"] for q in ("sp_pred", "sp_true", "sp_err")} for k in arms + ["persistence"]
          if f"{k}__sp_err" in raw}
    comp = [k for k in ("kai_long", "tend1", "tend") if k in sp]
    ls = {"kai_long": "-.", "tend1": "--", "tend": "-", "kai": ":"}
    fig, ax = plt.subplots(2, len(chans), figsize=(3.6 * len(chans), 7.5))
    scale_sum = {}
    for j, ch in enumerate(chans):
        c = names.index(ch)
        for bi, (lo, hi) in enumerate(bands):
            bl = f"k{lo}" if lo == hi else f"k{lo}-{hi}"
            base = sp["kai"]["sp_err"][:, c, lo:hi + 1].sum(-1)
            for k in comp:
                r = sp[k]["sp_err"][:, c, lo:hi + 1].sum(-1) / base
                ax[0, j].plot(days, r, color=bcol[bi], ls=ls[k], label=f"{k} {bl}" if j == 0 else None)
                scale_sum[f"{k}/{ch}/{bl}"] = [float(r[0]), float(r[min(4, L - 1)]), float(r[-1])]
            if lo == 0:
                continue
            for k in ["kai"] + comp:
                r = sp[k]["sp_pred"][:, c, lo:hi + 1].sum(-1) / sp[k]["sp_true"][:, c, lo:hi + 1].sum(-1)
                ax[1, j].plot(days, r, color=bcol[bi], ls=ls[k], label=f"{k} {bl}" if j == 0 else None)
        ax[0, j].axhline(1, color="k", lw=0.6)
        ax[0, j].set_yscale("log")
        ax[0, j].set_title(f"{ch}: error power / KAI's, by band", fontsize=9)
        ax[1, j].axhline(1, color="k", lw=0.6)
        ax[1, j].set_title(f"{ch}: prediction / truth power", fontsize=9)
        for p in ax[:, j]:
            p.set_xlabel("lead (days)")
            p.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=5.5, ncol=2)
    ax[1, 0].legend(fontsize=5.5, ncol=2)
    fig.suptitle("Scales (zonal wavenumber bands; colour = band, line = arm; below 1 in the top row = less error than KAI)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_kai_scale.png"), dpi=130)
    plt.close(fig)
    summary["scale_err_ratio_vs_kai_lead1_5_10"] = scale_sum
    json.dump(summary, open(os.path.join(out, "where_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if "/" in k and "scale" not in k}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
