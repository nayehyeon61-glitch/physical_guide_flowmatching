"""Three-model comparison figures of the tendency core ++ run (docs/kai_tendency_plusplus.md, section 7):
KAI, fine-tuned KAI and the best blend (fine-tuned KAI blended with the two extended cores).

    python scripts/plot_kai_pp_compare.py runs/kaiplus_1p5_daily docs/kai_tendency_plusplus_1p5_daily/figures

Reads <run>/score_pp.json; writes

  cmp1_rmse_by_variable.png     RMSE in physical units of nine headline variables (t2m, z850, z500, ...), three models
  cmp2_skill_vs_lead.png        65-channel nRMSE and ACC of the three models, and the gain over KAI with 95 % intervals
  cmp3_heatmap_headline.png     RMSE gain of the headline variables x lead, with the numbers (three comparisons)
  cmp4_heatmap_all.png          RMSE gain of every variable x level x lead (three comparisons)
  cmp5_heatmap_by_group.png     RMSE gain by variable group x lead, with the numbers (three comparisons)

Sign convention as everywhere in this project: a GAIN, + = better.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.plot_kai_plus_summary import (DIV, INK, LEVELS, LEVVARS, MUTED, SFC, SURFACE, base, end_labels,  # noqa: E402
                                           groups_of, plt, save, setup)

BEST = "blend_core_plus_u64+core_plus_u64_s1@kai_ro"
COL = {"kai": "#2a78d6", "kai_ro": "#eb6834", BEST: "#6250d6", "climatology": MUTED}
NAME = {"kai": "KAI", "kai_ro": "fine-tuned KAI", BEST: "best blend (fine-tuned KAI + two U-Net cores)",
        "climatology": "climatology"}
SHORT = {"kai": "KAI", "kai_ro": "fine-tuned KAI", BEST: "best blend"}
MODELS = ["kai", "kai_ro", BEST]
# (a, b, title): the RMSE gain of a over b
CMP = [("kai_ro", "kai", "fine-tuned KAI  against KAI"), (BEST, "kai", "best blend  against KAI"),
       (BEST, "kai_ro", "best blend  against fine-tuned KAI")]
HEAD = [("t2m", "2 m temperature (K)", 1.0), ("z850", "geopotential 850 hPa (m²/s²)", 1.0),
        ("z500", "geopotential 500 hPa (m²/s²)", 1.0), ("t850", "temperature 850 hPa (K)", 1.0),
        ("u850", "zonal wind 850 hPa (m/s)", 1.0), ("v850", "meridional wind 850 hPa (m/s)", 1.0),
        ("msl", "mean sea level pressure (hPa)", 0.01), ("sp", "surface pressure (hPa)", 0.01),
        ("q700", "specific humidity 700 hPa (g/kg)", 1000.0)]
HEAT = ["t2m", "skt", "msl", "sp", "tcwv", "z850", "z500", "z250", "t850", "t500", "u850", "u250", "v850", "q700"]


def pair(S, a, b):
    """(gain %, significant) of a over b, both (leads, channels)."""
    pr = S["paired_vs_kai"][a] if b == "kai" and a in S["paired_vs_kai"] and f"{a}/{b}" not in S["pairs"] \
        else S["pairs"][f"{a}/{b}"]
    sig = (np.array(pr["lo"]) > 0) | (np.array(pr["hi"]) < 0)
    return -100 * np.array(pr["rel"]), sig, pr


def line(p, days, y, m, **kw):
    p.plot(days, y, color=COL[m], lw=2.2 if m == BEST else 2, ls=(0, (5, 3)) if m == "climatology" else "-",
           label=NAME[m], **kw)


def cmp1(S, out, sub):
    days = np.arange(1, S["leads"] + 1)
    ch, A = S["channels"], S["arms"]
    fig, ax = plt.subplots(3, 3, figsize=(17, 12.5), gridspec_kw=dict(wspace=0.2, hspace=0.32, left=0.045, right=0.985,
                                                                       top=0.9, bottom=0.05))
    for j, (p, (c, ttl, f)) in enumerate(zip(ax.ravel(), HEAD)):
        base(p, "RMSE of " + ttl, days, xlabel=j >= 6)
        for m in ["climatology"] + MODELS:
            line(p, days, f * np.array(A[m]["rmse"])[:, ch.index(c)], m, marker=None if m == "climatology" else "o",
                 ms=3.5)
        p.set_ylim(0, None)
        y = {m: f * np.array(A[m]["rmse"])[-1, ch.index(c)] for m in MODELS}
        p.text(0.02, 0.97, f"day {S['leads']}:  " + "   ".join(f"{SHORT[m]} {y[m]:.3g}" for m in MODELS),
               transform=p.transAxes, fontsize=8.5, color=MUTED, va="top")
        if j == 0:
            p.legend(fontsize=9, loc="lower right")
    fig.suptitle("KAI, fine-tuned KAI and the best blend: RMSE in physical units of nine headline variables "
                 "(lower = better)\n" + sub, fontsize=11.5, x=0.045, ha="left", y=0.975)
    save(fig, out, "cmp1_rmse_by_variable.png")


def cmp2(S, out, sub):
    days = np.arange(1, S["leads"] + 1)
    A = S["arms"]
    fig, ax = plt.subplots(1, 4, figsize=(21, 5.4), gridspec_kw=dict(wspace=0.3, left=0.035, right=0.93, top=0.8,
                                                                      bottom=0.11))
    p = ax[0]
    base(p, "65-channel normalized RMSE (lower = better)", days)
    for m in ["climatology"] + MODELS:
        line(p, days, A[m]["nrmse"], m)
    p.set_ylim(0, None)
    p.legend(fontsize=9, loc="lower right")
    p = ax[1]
    base(p, "mean anomaly correlation (higher = better)", days)
    for m in MODELS:
        line(p, days, A[m]["acc_mean"], m)
    p.legend(fontsize=9, loc="lower left")
    p = ax[2]
    base(p, "nRMSE gain over KAI (%; + = better; band = 95 % interval)", days, zero=0.0)
    ys = {}
    for m in MODELS[1:]:
        pr = pair(S, m, "kai")[2]
        g, lo, hi = (-100 * np.array(pr[k]) for k in ("nrmse_rel", "nrmse_hi", "nrmse_lo"))
        p.fill_between(days, lo, hi, color=COL[m], alpha=0.18, lw=0)
        p.plot(days, g, color=COL[m], lw=2, label=SHORT[m])
        ys[m] = g
    p.legend(fontsize=9, loc="upper left")
    p = ax[3]
    base(p, "anomaly correlation gain over KAI (ACC difference; + = better)", days, zero=0.0)
    for m in MODELS[1:]:
        p.plot(days, np.array(A[m]["acc_mean"]) - np.array(A["kai"]["acc_mean"]), color=COL[m], lw=2, label=SHORT[m])
    p.legend(fontsize=9, loc="upper left")
    fig.suptitle("KAI, fine-tuned KAI and the best blend: skill against lead time\n" + sub, fontsize=11.5, x=0.035,
                 ha="left", y=0.97)
    save(fig, out, "cmp2_skill_vs_lead.png")


def numbered(p, M, rows, L, vmax, first, fs=8.5):
    im = p.pcolormesh(np.arange(L + 1) + 0.5, np.arange(len(rows) + 1), M, cmap=DIV, vmin=-vmax, vmax=vmax,
                      edgecolors=SURFACE, linewidth=2)
    for r in range(len(rows)):
        for d in range(L):
            v = M[r, d]
            p.text(d + 1, r + 0.5, f"{v:+.1f}", ha="center", va="center", fontsize=fs,
                   color="white" if abs(v) > 0.62 * vmax else INK)
    p.set_ylim(len(rows), 0)
    p.set_yticks(np.arange(len(rows)) + 0.5)
    p.set_yticklabels(rows if first else [])
    p.set_xticks(np.arange(1, L + 1))
    p.set_xlabel("lead (days)")
    p.tick_params(length=0)
    for s in p.spines.values():
        s.set_visible(False)
    return im


def cmp3(S, out, sub):
    L, ch = S["leads"], S["channels"]
    idx = [ch.index(c) for c in HEAT]
    vmax = 20.0
    fig, ax = plt.subplots(1, 3, figsize=(20, 7.6), gridspec_kw=dict(wspace=0.06, left=0.04, right=0.93, top=0.84,
                                                                      bottom=0.08))
    for i, (p, (a, b, ttl)) in enumerate(zip(ax, CMP)):
        im = numbered(p, pair(S, a, b)[0][:, idx].T, HEAT, L, vmax, i == 0)
        p.set_title(ttl, fontsize=10.5, loc="left")
    cb = fig.colorbar(im, cax=fig.add_axes([0.94, 0.08, 0.008, 0.76]), extend="both")
    cb.set_label(f"RMSE gain (%), colour clipped at +-{vmax:.0f}:  + (blue) = better,  - (red) = worse", color=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("RMSE gain of the headline variables by lead (%; + = better)\n" + sub, fontsize=11.5, x=0.04,
                 ha="left", y=0.97)
    save(fig, out, "cmp3_heatmap_headline.png")


def cmp4(S, out, sub):
    L = S["leads"]
    days = np.arange(1, L + 1)
    G = groups_of(S)
    vmax = 30.0
    fig = plt.figure(figsize=(19, 15.5))
    gs = fig.add_gridspec(3, 60, wspace=0.0, hspace=0.34, left=0.085, right=0.955, top=0.9, bottom=0.045)
    for r, (a, b, ttl) in enumerate(CMP):
        gain, sig, _ = pair(S, a, b)
        for i, g in enumerate(LEVVARS + ["surface"]):
            p = fig.add_subplot(gs[r, i * 9 + (1 if i else 0):i * 9 + 8 + (1 if i else 0)] if i < 5 else gs[r, 47:55])
            c = G[g]
            im = p.pcolormesh(np.arange(L + 1) + 0.5, np.arange(len(c) + 1), gain[:, c].T, cmap=DIV, vmin=-vmax,
                              vmax=vmax, edgecolors=SURFACE, linewidth=1.2)
            yy, xx = np.nonzero(~sig[:, c].T)
            p.scatter(xx + 1, yy + 0.5, s=6, color=MUTED, linewidths=0)
            p.set_ylim(len(c), 0)
            p.set_yticks(np.arange(len(c)) + 0.5)
            p.set_yticklabels([str(q) for q in LEVELS] if g != "surface" else SFC)
            if i not in (0, 5):
                p.set_yticklabels([])
            if i == 0:
                p.set_ylabel(ttl + "\n\nlevel (hPa)", color=INK)
            p.set_xticks(days)
            if r == 2:
                p.set_xlabel("lead (days)")
            nb, nw = int((sig[:, c] & (gain[:, c] > 0)).sum()), int((sig[:, c] & (gain[:, c] < 0)).sum())
            p.set_title(f"{g}   {nb} better / {nw} worse of {len(c) * L}", fontsize=10, loc="left")
            p.tick_params(length=0)
            for s in p.spines.values():
                s.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, 58:59]), extend="both")
    cb.set_label(f"RMSE gain (%), colour clipped at +-{vmax:.0f}:  + (blue) = better,  - (red) = worse", color=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("RMSE gain of every variable, level and lead: fine-tuned KAI against KAI, the best blend against KAI, "
                 "the best blend against fine-tuned KAI\n(a dot = the 95 % interval contains zero)   " + sub,
                 fontsize=11.5, x=0.04, ha="left", y=0.975)
    save(fig, out, "cmp4_heatmap_all.png")


def cmp5(S, out, sub):
    L = S["leads"]
    G = groups_of(S)
    rows = LEVVARS + ["surface"]
    vmax = 15.0
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.6), gridspec_kw=dict(wspace=0.08, left=0.05, right=0.93, top=0.76,
                                                                      bottom=0.12))
    for i, (p, (a, b, ttl)) in enumerate(zip(ax, CMP)):
        gain = pair(S, a, b)[0]
        im = numbered(p, np.array([[gain[d, G[g]].mean() for d in range(L)] for g in rows]), rows, L, vmax, i == 0, 9)
        p.set_title(ttl, fontsize=10.5, loc="left")
    cb = fig.colorbar(im, cax=fig.add_axes([0.94, 0.12, 0.008, 0.64]), extend="both")
    cb.set_label("RMSE gain (%)", color=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("RMSE gain by variable group and lead (%; + = better; mean over the channels of the group: "
                 "12 levels, or the 5 surface fields t2m, msl, sp, skt, tcwv)\n" + sub,
                 fontsize=11.5, x=0.05, ha="left", y=0.97)
    save(fig, out, "cmp5_heatmap_by_group.png")


def main():
    run, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    S = json.load(open(os.path.join(run, "score_pp.json")))
    sub = (f"ERA5 daily means at 1.5 deg, trained on 2012-15, {S['ics']} initial conditions of 2017; "
           "best blend = fine-tuned KAI blended with two U-Net cores (u64, seeds 0 and 1)")
    setup()
    cmp1(S, out, sub)
    cmp2(S, out, sub)
    cmp3(S, out, sub)
    cmp4(S, out, sub)
    cmp5(S, out, sub)


if __name__ == "__main__":
    main()
