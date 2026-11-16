"""Figures of the core scaling test (docs/kai_core_scaling.md).

    python scripts/plot_kai_core_scaling.py runs/kaitend docs/kai_core_scaling

fig_size.png   (score_S2, breakdown_S2, diag_gap_size.json) 2 x 2: the error of the core alone against its parameter
               count, log-log, with the exponent alpha = -d ln E / d ln N of the ladder; the same inside the coupled
               loop on the 32-year KAI; the core's share on that KAI by lead for every size; the generalization gap
               (unseen / trained years - 1) against the size. Solid = nRMSE, dashed = its information part.
fig_data.png   (score_S3 / score_S4 and their gap files, when they exist) the same along the training years.
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
from kai_scaling_fit import errors, exponents  # noqa: E402

K = "kai_y32x8ro"
SEQ = ["#8fbbee", "#4f93e0", "#1f63bd", "#0d3266"]  # one hue, light -> dark: a ladder is a magnitude
DAYS = (1, 3, 5)


def style():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def gap_of(g, tag, leads=3):
    """unseen / trained - 1 of the one-step .. 3-day loss of a core's own rollout, (leads,)."""
    v = g[tag]["years"]
    tr = np.mean([np.atleast_1d(y["loss"])[:leads] for y in v.values() if y["trained"]], 0)
    un = np.mean([np.atleast_1d(y["loss"])[:leads] for y in v.values() if not y["trained"]], 0)
    return un / tr - 1


def ladder_panel(p, x, E, Ei, title, xlabel, extra=None, base=None, split=0):
    """log-log lines of the error against x for DAYS, with the slope of the ladder written at the end. ``split`` = k:
    two exponents instead, the one of the first k intervals and the slope over the rest (a ladder with a bend)."""
    local, slope = exponents(x, E)
    _, slope_i = exponents(x, Ei)
    if split:
        _, s1 = exponents(x[:split + 1], E[:split + 1])
        _, s2 = exponents(x[split:], E[split:])
    for j, d in enumerate(DAYS):
        c = CAT[j]
        p.plot(x, E[:, d - 1], color=c, lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.2, zorder=3)
        p.plot(x, Ei[:, d - 1], color=c, lw=1.4, ls="--", zorder=2)
        txt = f"day {d}: α = {slope[d - 1]:.3f}\n(information {slope_i[d - 1]:.3f})"
        if split:
            txt = (f"day {d}: α = {s1[d - 1]:.3f} ({x[0]:g} to {x[split]:g})\n"
                   f"then {s2[d - 1]:.3f} ({x[split]:g} to {x[-1]:g})")
        p.annotate(txt, (x[-1], E[-1, d - 1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5)
        if base is not None:
            p.plot([x[0], x[-1]], [base[d - 1]] * 2, color=c, lw=1, ls=":")
        if extra is not None:
            p.plot([extra[0]], [extra[1][d - 1]], ls="none", marker="s", ms=6.5, mfc=SURFACE, mec=c, mew=1.6, zorder=4)
    p.set_xscale("log")
    p.set_yscale("log")
    p.set_xlabel(xlabel)
    p.set_title(title, fontsize=10.5, loc="left")
    p.grid(axis="y", which="both", color=GRID, lw=0.8)
    p.set_xlim(x[0] / 1.25, x[-1] * (4.6 if split else 3.4))
    p.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    p.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    p.tick_params(axis="y", which="minor", labelsize=8)


def fig_size(run, out):
    S = json.load(open(os.path.join(run, "score_S2.json")))
    raw = np.load(os.path.join(run, "score_S2_raw.npz"))
    std = np.load(os.path.join(run, "stats.npz"))["std"].astype(np.float64) + 1e-8
    B = json.load(open(os.path.join(run, "breakdown_S2", "breakdown.json")))
    tags = ["core_gngmz8h24", "core_gngmz8", "core_gngmz8w", "core_gngmz8h96"]
    x = np.array([S["arms"][t]["params"] for t in tags], float)
    name = [f"{v / 1e6:.2f} M" for v in x]
    E = lambda k: errors(S, raw, std, k)  # noqa: E731
    fig, ax = plt.subplots(2, 2, figsize=(13.0, 9.4))

    A = [E(t) for t in tags]
    ladder_panel(ax[0, 0], x, np.stack([a[0] for a in A]), np.stack([a[1] for a in A]),
                 "core alone: nRMSE against the parameter count (4 years, 3000 it)", "parameters of the core",
                 extra=(x[1], E("core_gngmz8_x2")[0]))
    ax[0, 0].plot([], [], ls="none", marker="s", ms=6.5, mfc=SURFACE, mec=MUTED, mew=1.6, label="0.26 M core with 6000 it")
    ax[0, 0].plot([], [], color=MUTED, lw=2, marker="o", ms=5, label="nRMSE")
    ax[0, 0].plot([], [], color=MUTED, lw=1.4, ls="--", label="information part")
    ax[0, 0].legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.5))

    A = [E(f"loop_{t}@{K}") for t in tags]
    ladder_panel(ax[0, 1], x, np.stack([a[0] for a in A]), np.stack([a[1] for a in A]),
                 "coupled loop on the 32-year KAI: nRMSE against the core's parameter count", "parameters of the core",
                 base=E(K)[0])
    ax[0, 1].plot([], [], color=MUTED, lw=1, ls=":", label="the 32-year fine-tuned KAI alone")
    ax[0, 1].legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.5))

    p = ax[1, 0]
    L = np.arange(1, S["leads"] + 1)
    for t, c, n in zip(tags, SEQ, name):
        b = B[f"loop_{t}@{K}:{K}"]
        g = 100 * (1 - np.array(b["nrmse"][f"loop_{t}@{K}"]) / np.array(b["nrmse"][K]))
        p.plot(L, g, color=c, lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1)
        p.plot(L, np.array(b["dterm2_pct"])[:, 1], color=c, lw=1.4, ls="--")
        p.annotate(n, (L[0], g[0]), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5)
    p.plot([], [], color=MUTED, lw=2, marker="o", ms=4.5, label="total gain")
    p.plot([], [], color=MUTED, lw=1.4, ls="--", label="information part")
    p.axhline(0, color=INK, lw=1)
    p.set_xticks(L)
    p.set_xlim(-0.25, L[-1] + 0.3)
    p.set_xlabel("lead (days)")
    p.set_title("the core's share: loop over the 32-year fine-tuned KAI alone (%)", fontsize=10.5, loc="left")
    p.grid(axis="y", color=GRID, lw=0.8)
    p.legend(frameon=False, fontsize=8.5, loc="upper right")

    p = ax[1, 1]
    G = {}
    for f in ("diag_gap_core32.json", "diag_gap_size.json"):
        G.update(json.load(open(os.path.join(run, f))))
    gp = 100 * np.stack([gap_of(G, t) for t in tags])
    p.axhspan(5, 7, color=GRID, alpha=0.6, lw=0)
    p.text(x[0] / 1.2, 5.15, "above 5 %: short of data", fontsize=8.5, color=MUTED, va="bottom")
    p.axhline(3, color=MUTED, lw=1, ls=":")
    p.text(x[0] / 1.2, 3.1, "below 3 %: short of capacity or steps", fontsize=8.5, color=MUTED, va="bottom")
    for j in range(3):
        p.plot(x, gp[:, j], color=CAT[j + 3], lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.2)
        p.annotate(f"rollout day {j + 1}", (x[-1], gp[-1, j]), xytext=(8, -7 if j == 1 else 0),
                   textcoords="offset points", va="center", fontsize=8.5)
    p.plot([x[1]], [100 * gap_of(json.load(open(os.path.join(run, "diag_gap_core_x2.json"))), "core_gngmz8_x2")[2]],
           ls="none", marker="s", ms=6.5, mfc=SURFACE, mec=CAT[5], mew=1.6, label="0.26 M core with 6000 it, day 3")
    p.axhline(0, color=INK, lw=1)
    p.set_xscale("log")
    p.set_xlim(x[0] / 1.25, x[-1] * 2.2)
    p.set_ylim(-0.6, 7)
    p.set_xlabel("parameters of the core")
    p.set_title("generalization gap: loss on unseen years / on the 4 trained years - 1 (%)", fontsize=10.5, loc="left")
    p.grid(axis="y", color=GRID, lw=0.8)
    p.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.62))
    for q in (ax[0, 0], ax[0, 1], ax[1, 1]):
        q.set_xticks(x)
        q.set_xticklabels(name)
        q.minorticks_off() if q is ax[1, 1] else q.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    fig.suptitle("Core scaling test S2: the size ladder of the tendency core, 0.08 to 0.90 M parameters "
                 "(5.625 deg, 119 forecasts of 2017)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out, "fig_size.png"), dpi=150)
    plt.close(fig)


def fig_data(run, out):
    S = json.load(open(os.path.join(run, "score_S3.json")))
    raw = np.load(os.path.join(run, "score_S3_raw.npz"))
    std = np.load(os.path.join(run, "stats.npz"))["std"].astype(np.float64) + 1e-8
    tags = ["core_gngmz8_y1", "core_gngmz8_y2", "core_gngmz8", "core_gngmz8_y10", "core_gngmz8_y32"]
    x = np.array([1, 2, 4, 10, 32], float)
    name = [f"{int(v)}" for v in x]
    E = lambda k: errors(S, raw, std, k)  # noqa: E731
    fig, ax = plt.subplots(2, 2, figsize=(13.0, 9.4))

    A = [E(t) for t in tags]
    Et = np.stack([a[0] for a in A])
    ladder_panel(ax[0, 0], x, Et, np.stack([a[1] for a in A]),
                 "core alone (0.26 M, 3000 it): nRMSE against the training years", "training years", split=1)
    ax[0, 0].plot([], [], color=MUTED, lw=2, marker="o", ms=5, label="nRMSE")
    ax[0, 0].plot([], [], color=MUTED, lw=1.4, ls="--", label="information part")
    ax[0, 0].legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.5))

    A = [E(f"loop_{t}@{K}") for t in tags]
    ladder_panel(ax[0, 1], x, np.stack([a[0] for a in A]), np.stack([a[1] for a in A]),
                 "coupled loop on the 32-year KAI: nRMSE against the core's training years", "training years of the core",
                 base=E(K)[0], split=1)
    ax[0, 1].plot([], [], color=MUTED, lw=1, ls=":", label="the 32-year fine-tuned KAI alone")
    ax[0, 1].legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.5))

    p = ax[1, 0]
    G = {}
    for f in ("diag_gap_core32.json", "diag_gap_data.json", "diag_gap_size.json", "diag_gap_core_x2.json", "diag_gap_h96y32.json"):
        if os.path.exists(os.path.join(run, f)):
            G.update(json.load(open(os.path.join(run, f))))
    gp = 100 * np.stack([gap_of(G, t) for t in tags])
    p.axhspan(5, 12, color=GRID, alpha=0.6, lw=0)
    p.text(x[-1] * 1.9, 5.2, "above 5 %: short of data", fontsize=8.5, color=MUTED, va="bottom", ha="right")
    p.axhline(3, color=MUTED, lw=1, ls=":")
    for j in range(3):
        p.plot(x, gp[:, j], color=CAT[j + 3], lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.2, label=f"rollout day {j + 1}")
    p.legend(frameon=False, fontsize=8.5, loc="upper right")
    p.axhline(0, color=INK, lw=1)
    p.set_xscale("log")
    p.set_xlim(x[0] / 1.25, x[-1] * 2.0)
    p.set_ylim(-0.6, 12)
    p.set_xlabel("training years")
    p.set_title("generalization gap: loss on unseen years / on trained years - 1 (%)", fontsize=10.5, loc="left")
    p.grid(axis="y", color=GRID, lw=0.8)

    # what the years still have to give, against the gap: the gain of the 32-year core over each smaller-data core
    p = ax[1, 1]
    n = lambda tag, S_=S: np.array(S_["arms"][tag]["nrmse"])  # noqa: E731
    pts = [(100 * gap_of(G, t)[0], 100 * (1 - n("core_gngmz8_y32") / n(t)), "o", f"{int(v)} y") for t, v in zip(tags[:4], x)]
    f3 = os.path.join(run, "score_T3.json")
    if os.path.exists(f3) and "core_gngmz8_x2" in G:
        S3 = json.load(open(f3))
        pts.append((100 * gap_of(G, "core_gngmz8_x2")[0],
                    100 * (1 - n("core_gngmz8_y32x2", S3) / n("core_gngmz8_x2", S3)), "s", "4 y, 6000 it"))
    f4 = os.path.join(run, "score_S4.json")
    if os.path.exists(f4) and "core_gngmz8h96" in G:
        S4 = json.load(open(f4))
        pts.append((100 * gap_of(G, "core_gngmz8h96")[0],
                    100 * (1 - n("core_gngmz8h96_y32", S4) / n("core_gngmz8h96", S4)), "D", "4 y, 0.90 M"))
    for gx, gy, mk, lab in pts:
        for j, d in enumerate(DAYS):
            p.plot([gx], [gy[d - 1]], ls="none", marker=mk, ms=7, mfc=CAT[j] if mk == "o" else SURFACE, mec=CAT[j],
                   mew=1.6, zorder=3)
        p.annotate(lab, (gx, max(gy[d - 1] for d in DAYS)), xytext=(0, 7), textcoords="offset points", ha="center",
                   fontsize=8.5)
    for j, d in enumerate(DAYS):
        p.plot([], [], ls="none", marker="o", ms=7, color=CAT[j], label=f"day {d}")
    p.plot([], [], ls="none", marker="s", ms=7, mfc=SURFACE, mec=MUTED, mew=1.6, label="0.26 M core with 6000 it")
    if any(q[2] == "D" for q in pts):
        p.plot([], [], ls="none", marker="D", ms=7, mfc=SURFACE, mec=MUTED, mew=1.6, label="0.90 M core (width 96)")
    p.axhline(0, color=INK, lw=1)
    p.set_xlabel("generalization gap of the core at rollout day 1 (%)")
    p.set_title("what 32 years add: gain of the 32-year core over the core of fewer years (%)", fontsize=10.5, loc="left")
    p.grid(axis="y", color=GRID, lw=0.8)
    p.legend(frameon=False, fontsize=8.5, loc="upper left")
    for q in (ax[0, 0], ax[0, 1], ax[1, 0]):
        q.set_xticks(x)
        q.set_xticklabels(name)
        q.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    fig.suptitle("Core scaling test S3 / S4: the data ladder of the baseline core, 1 to 32 training years "
                 "(5.625 deg, 119 forecasts of 2017)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out, "fig_data.png"), dpi=150)
    plt.close(fig)


def main(run, out):
    os.makedirs(out, exist_ok=True)
    style()
    if os.path.exists(os.path.join(run, "breakdown_S2", "breakdown.json")):
        fig_size(run, out)
        print("wrote", os.path.join(out, "fig_size.png"))
    if os.path.exists(os.path.join(run, "score_S3.json")):
        fig_data(run, out)
        print("wrote", os.path.join(out, "fig_data.png"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
