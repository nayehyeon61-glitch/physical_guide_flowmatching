"""Summary figures of one tendency core + run (docs/kai_tendency_plus_1p5_daily.md): the whole run on six pages,
next to the per-pair figures of scripts/eval_kai_breakdown.py.

    python scripts/plot_kai_plus_summary.py runs/kaiplus_1p5_daily docs/kai_tendency_plus_1p5_daily/figures

Reads <run>/score_plus.json, <run>/breakdown/breakdown.json and <run>/{kai,kai_ro,core_plus}.log; writes

  fig1_skill_vs_lead.png        65-channel nRMSE of every model, and the nRMSE / ACC gain over KAI with 95 % intervals
  fig2_heatmap_best_vs_kai.png  RMSE gain of the best model over KAI, every variable x level x lead
  fig3_heatmap_by_group.png     RMSE gain over KAI by variable group x lead, three models, with the numbers
  fig4_rmse_by_variable.png     RMSE in physical units of six headline variables
  fig5_information_amplitude.png  the gain split into pattern (information) and amplitude cost, and the activity
  fig6_training_curves.png      validation loss of the KAI and core + trainings

Sign convention as everywhere in this project: a GAIN, + = better. One model has one colour in every figure.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# the palette of eval_kai_breakdown.py (fixed order); a model keeps its colour in every figure
COL = {"kai": "#2a78d6", "kai_ro": "#eb6834", "loop_core_plus": "#1baf7a", "loop_core_plus@kai_ro": "#6250d6",
       "core_plus": "#e87ba4", "persistence": MUTED, "climatology": MUTED}
NAME = {"kai": "KAI", "kai_ro": "fine-tuned KAI", "loop_core_plus": "KAI + core +",
        "loop_core_plus@kai_ro": "fine-tuned KAI + core +", "core_plus": "core + alone",
        "persistence": "persistence", "climatology": "climatology"}
STYLE = {"persistence": (0, (1, 2)), "climatology": (0, (5, 3))}
DIV = LinearSegmentedColormap.from_list("lossgain", ["#8a1f1e", "#e34948", "#f0efec", "#2a78d6", "#12407a"])
LEVELS = [50, 100, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
LEVVARS = ["z", "q", "t", "u", "v"]
SFC = ["t2m", "msl", "sp", "skt", "tcwv"]
BEST = "loop_core_plus@kai_ro"
VS_KAI = ["kai_ro", "loop_core_plus", BEST]


def setup():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                         "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "legend.frameon": False})


def base(p, title, days, zero=None, xlabel=True):
    p.set_title(title, fontsize=10, loc="left")
    p.set_xticks(days)
    if xlabel:
        p.set_xlabel("lead (days)")
    p.grid(color=GRID, lw=0.8)
    p.set_axisbelow(True)
    p.tick_params(length=0)
    if zero is not None:
        p.axhline(zero, color=INK, lw=0.8)
    for s in ("top", "right"):
        p.spines[s].set_visible(False)


def line(p, days, y, m, **kw):
    p.plot(days, y, color=COL[m], lw=2, ls=STYLE.get(m, "-"), label=NAME[m], **kw)


def end_labels(p, days, ys, gap):
    """Direct labels at the line ends, pushed apart to at least ``gap`` (data units) so they do not collide."""
    order = sorted(ys, key=lambda m: ys[m][-1])
    pos = []
    for m in order:
        y = ys[m][-1]
        pos.append(y if not pos else max(y, pos[-1] + gap))
    for m, y in zip(order, pos):
        p.annotate(NAME[m], (days[-1], ys[m][-1]), xytext=(days[-1] + 0.25, y), fontsize=9, color=INK, va="center",
                   arrowprops=dict(arrowstyle="-", color=COL[m], lw=1.2, shrinkA=0, shrinkB=0))


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.join(out, name))


def fig1(S, out, sub):
    days = np.arange(1, S["leads"] + 1)
    A, P = S["arms"], S["paired_vs_kai"]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4), gridspec_kw=dict(wspace=0.6, left=0.04, right=0.885, top=0.8,
                                                                      bottom=0.11, width_ratios=[1, 1, 1]))
    p = ax[0]
    base(p, "65-channel normalized RMSE (lower = better)", days)
    for m in ["persistence", "climatology", "core_plus", "kai", "kai_ro", "loop_core_plus", BEST]:
        line(p, days, A[m]["nrmse"], m)
    p.set_ylim(0, None)
    p.legend(fontsize=9, loc="lower right")
    p = ax[1]
    base(p, "nRMSE gain over KAI (%; + = better; band = 95 % interval over the ICs)", days, zero=0.0)
    ys = {}
    for m in VS_KAI:
        g, lo, hi = (-100 * np.array(P[m][k]) for k in ("nrmse_rel", "nrmse_hi", "nrmse_lo"))
        p.fill_between(days, lo, hi, color=COL[m], alpha=0.18, lw=0)
        line(p, days, g, m)
        ys[m] = g
    end_labels(p, days, ys, 0.45)
    p.legend(fontsize=9, loc="lower right")
    p = ax[2]
    base(p, "anomaly correlation gain over KAI (ACC difference; + = better)", days, zero=0.0)
    ys = {m: np.array(A[m]["acc_mean"]) - np.array(A["kai"]["acc_mean"]) for m in VS_KAI}
    for m in VS_KAI:
        line(p, days, ys[m], m)
    end_labels(p, days, ys, 0.0028)
    p.legend(fontsize=9, loc="upper left")
    for p in ax[1:]:
        p.set_xlim(0.6, S["leads"] + 0.3)
    fig.suptitle("Tendency core + at 1.5 deg: skill against lead time\n" + sub, fontsize=11.5, x=0.045, ha="left", y=0.97)
    save(fig, out, "fig1_skill_vs_lead.png")


def groups_of(S):
    ch = S["channels"]
    g = {v: [ch.index(f"{v}{q}") for q in LEVELS] for v in LEVVARS}
    g["surface"] = [ch.index(c) for c in SFC]
    return g


def fig2(S, out, sub):
    L = S["leads"]
    days = np.arange(1, L + 1)
    pr = S["pairs"][f"{BEST}/kai"]
    gain = -100 * np.array(pr["rel"])
    sig = (np.array(pr["lo"]) > 0) | (np.array(pr["hi"]) < 0)
    G = groups_of(S)
    allc = sum(G.values(), [])
    vmax = float(np.clip(5 * np.ceil(np.percentile(np.abs(gain[:, allc]), 98) / 5), 5, 60))
    fig = plt.figure(figsize=(19, 5.6))
    gs = fig.add_gridspec(1, 60, wspace=0.0, left=0.04, right=0.955, top=0.78, bottom=0.11)
    for i, g in enumerate(LEVVARS + ["surface"]):
        p = fig.add_subplot(gs[0, i * 9 + (1 if i else 0):i * 9 + 8 + (1 if i else 0)] if i < 5 else gs[0, 47:55])
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
            p.set_ylabel("level (hPa)")
        p.set_xticks(days)
        p.set_xlabel("lead (days)")
        nb, nw = int((sig[:, c] & (gain[:, c] > 0)).sum()), int((sig[:, c] & (gain[:, c] < 0)).sum())
        p.set_title(f"{g}   {nb} better / {nw} worse of {len(c) * L}", fontsize=10, loc="left")
        p.tick_params(length=0)
        for s in p.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 58:59]), extend="both")
    cb.set_label(f"RMSE gain over KAI (%), colour clipped at +-{vmax:.0f}:  + (blue) = better,  - (red) = worse", color=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("Best model (fine-tuned KAI + core +, coupled loop) against KAI: RMSE gain of every variable, level "
                 "and lead  (a dot = the 95 % interval contains zero)\n" + sub, fontsize=11.5, x=0.04, ha="left", y=0.97)
    save(fig, out, "fig2_heatmap_best_vs_kai.png")


def fig3(S, out, sub):
    L = S["leads"]
    G = groups_of(S)
    rows = LEVVARS + ["surface"]
    M = {m: np.array([[-100 * np.mean(np.array(S["paired_vs_kai"][m]["rel"])[d, G[g]]) for d in range(L)]
                      for g in rows]) for m in VS_KAI}
    vmax = 15.0
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.6), gridspec_kw=dict(wspace=0.08, left=0.05, right=0.93, top=0.76,
                                                                      bottom=0.12))
    for i, (p, m) in enumerate(zip(ax, VS_KAI)):
        im = p.pcolormesh(np.arange(L + 1) + 0.5, np.arange(len(rows) + 1), M[m], cmap=DIV, vmin=-vmax, vmax=vmax,
                          edgecolors=SURFACE, linewidth=2)
        for r in range(len(rows)):
            for d in range(L):
                v = M[m][r, d]
                p.text(d + 1, r + 0.5, f"{v:+.1f}", ha="center", va="center", fontsize=9,
                       color="white" if abs(v) > 0.62 * vmax else INK)
        p.set_ylim(len(rows), 0)
        p.set_yticks(np.arange(len(rows)) + 0.5)
        p.set_yticklabels(rows if i == 0 else [])
        p.set_xticks(np.arange(1, L + 1))
        p.set_xlabel("lead (days)")
        p.set_title(f"{NAME[m]}  against KAI", fontsize=10.5, loc="left")
        p.tick_params(length=0)
        for s in p.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_axes([0.94, 0.12, 0.008, 0.64]))
    cb.set_label("RMSE gain over KAI (%)", color=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("RMSE gain over KAI by variable group and lead (%; + = better; mean over the channels of the group: "
                 "12 levels, or the 5 surface fields t2m, msl, sp, skt, tcwv)\n" + sub,
                 fontsize=11.5, x=0.05, ha="left", y=0.97)
    save(fig, out, "fig3_heatmap_by_group.png")


def fig4(S, out, sub):
    days = np.arange(1, S["leads"] + 1)
    ch, A = S["channels"], S["arms"]
    V = [("t2m", "2 m temperature (K)", 1.0), ("t850", "temperature 850 hPa (K)", 1.0),
         ("z500", "geopotential 500 hPa (m²/s²)", 1.0), ("u850", "zonal wind 850 hPa (m/s)", 1.0),
         ("msl", "mean sea level pressure (hPa)", 0.01), ("q700", "specific humidity 700 hPa (g/kg)", 1000.0)]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9), gridspec_kw=dict(wspace=0.2, hspace=0.3, left=0.045, right=0.985,
                                                                    top=0.87, bottom=0.07))
    for j, (p, (c, ttl, f)) in enumerate(zip(ax.ravel(), V)):
        base(p, "RMSE of " + ttl, days, xlabel=j >= 3)
        for m in ["climatology", "core_plus", "kai", "kai_ro", "loop_core_plus", BEST]:
            line(p, days, f * np.array(A[m]["rmse"])[:, ch.index(c)], m)
        p.set_ylim(0, None)
        if j == 0:
            p.legend(fontsize=9, loc="lower right")
    fig.suptitle("RMSE in physical units of six headline variables (lower = better)\n" + sub, fontsize=11.5, x=0.045,
                 ha="left", y=0.965)
    save(fig, out, "fig4_rmse_by_variable.png")


def fig5(S, B, out, sub):
    days = np.arange(1, S["leads"] + 1)
    pairs = [("kai_ro:kai", "fine-tuned KAI  against KAI"), ("loop_core_plus:kai", "KAI + core +  against KAI"),
             (f"{BEST}:kai_ro", "fine-tuned KAI + core +  against fine-tuned KAI")]
    fig, ax = plt.subplots(1, 4, figsize=(19, 5.2), gridspec_kw=dict(wspace=0.24, left=0.04, right=0.885, top=0.78,
                                                                      bottom=0.11))
    lo = min(np.array(B[k]["dterm2_pct"]).min() for k, _ in pairs)
    hi = max(np.array(B[k]["dterm2_pct"]).sum(1).max() for k, _ in pairs)
    for i, (p, (k, ttl)) in enumerate(zip(ax, pairs)):
        T = np.array(B[k]["dterm2_pct"])  # (leads, 3): bias, pattern (information), amplitude cost
        base(p, ttl + "\ncontribution to the nRMSE gain (%; + = better)", days, zero=0.0)
        w = 0.36
        p.bar(days - w / 2, T[:, 1], w * 0.92, color="#2a78d6", label="pattern (information)")
        p.bar(days + w / 2, T[:, 2], w * 0.92, color="#eda100", label="amplitude cost (smoother forecast)")
        p.plot(days, T.sum(1), color=INK, lw=1.6, marker="o", ms=4.5, label="total gain")
        p.set_ylim(lo - 0.3, hi + 0.5)
        if i == 0:
            p.legend(fontsize=9, loc="upper left")
    p = ax[3]
    base(p, "activity: amplitude of the forecast anomaly\n/ the truth's (1 = full strength)", days, zero=1.0)
    act = {"kai": B["loop_core_plus:kai"]["act"]["kai"], "kai_ro": B["kai_ro:kai"]["act"]["kai_ro"],
           "loop_core_plus": B["loop_core_plus:kai"]["act"]["loop_core_plus"], BEST: B[f"{BEST}:kai_ro"]["act"][BEST]}
    for m in act:
        line(p, days, act[m], m)
    end_labels(p, days, {m: np.array(v) for m, v in act.items()}, 0.011)
    p.set_xlim(0.6, S["leads"] + 0.3)
    p.set_ylim(0.84, 1.01)
    p.legend(fontsize=9, loc="lower left")
    fig.suptitle("Where the gain comes from: a better pattern (information) or a smoother forecast (amplitude cost)\n"
                 + sub, fontsize=11.5, x=0.04, ha="left", y=0.97)
    save(fig, out, "fig5_information_amplitude.png")


def read_log(path):
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith("{") and '"it"' in ln:
            rows.append(json.loads(ln))
    last = {}
    for r in rows:  # a resumed training repeats nothing, but keep the last line of an iteration to be safe
        last[r["it"]] = r
    return [last[k] for k in sorted(last)]


def fig6(run, out, sub):
    K, C = read_log(os.path.join(run, "kai.log")), read_log(os.path.join(run, "core_plus.log"))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw=dict(wspace=0.2, left=0.055, right=0.975, top=0.8,
                                                                      bottom=0.12))
    for p, R, ttl, m in ((ax[0], K, "KAI: loss against iteration (1-day step, batch 8)", "kai"),
                         (ax[1], C, "core +: loss against iteration (8-day sequences, batch 8)", "core_plus")):
        it = np.array([r["it"] for r in R])
        p.set_title(ttl, fontsize=10, loc="left")
        p.plot(it, [r["loss"] for r in R], color=MUTED, lw=1.2, ls=(0, (4, 2)), label="training loss (one batch)")
        p.plot(it, [r["val"] for r in R], color=COL[m], lw=2, marker="o", ms=4, label="validation loss (2016)")
        p.annotate(f"{R[-1]['val']:.4f}", (it[-1], R[-1]["val"]), xytext=(-4, 9), textcoords="offset points",
                   ha="right", fontsize=9, color=INK)
        p.set_xlabel("iteration")
        p.grid(color=GRID, lw=0.8)
        p.set_axisbelow(True)
        p.tick_params(length=0)
        for s in ("top", "right"):
            p.spines[s].set_visible(False)
        if m == "kai":
            p.legend(fontsize=9, loc="upper right")
        else:  # between the curriculum labels at the top and the training loss
            p.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 0.95))
    ks = [r.get("K", 1) for r in C]
    for a, b in zip(C[:-1], C[1:]):  # the curriculum of the core: the number of rollout days K of the loss
        if b.get("K", 1) != a.get("K", 1):
            ax[1].axvline((a["it"] + b["it"]) / 2, color=GRID, lw=1.2)
            ax[1].text((a["it"] + b["it"]) / 2, ax[1].get_ylim()[1], f" K = {b['K']}", fontsize=8.5, color=MUTED, va="top")
    if len(set(ks)) > 1:
        ax[1].set_title(ax[1].get_title(loc="left") + "\nK = rollout days in the training loss (the validation loss "
                        "is always the 1-day loss)", fontsize=10, loc="left")
    fig.suptitle("Training curves\n" + sub, fontsize=11.5, x=0.055, ha="left", y=0.97)
    save(fig, out, "fig6_training_curves.png")


def main():
    run, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    S = json.load(open(os.path.join(run, "score_plus.json")))
    B = json.load(open(os.path.join(run, "breakdown", "breakdown.json")))
    sub = f"ERA5 daily means at 1.5 deg, trained on 2012-15, {S['ics']} initial conditions of 2017, one core (seed 0)"
    setup()
    fig1(S, out, sub)
    fig2(S, out, sub)
    fig3(S, out, sub)
    fig4(S, out, sub)
    fig5(S, B, out, sub)
    fig6(run, out, sub)


if __name__ == "__main__":
    main()
