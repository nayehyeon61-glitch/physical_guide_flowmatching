"""Lead-time figure of a two-stage forecast against the single-stage cores, with the control at equal smoothing.

Four panels over the lead day: ACC (65 scored channels), RMSE of one variable (physical units), the ACC difference
from the control, and the activity. The dashed line is the control filtered to the two-stage forecast's zonal
spectrum (scripts/diag_stage_filter.py, row ``match``): what the smoothing alone gives.

    python scripts/plot_stage_filter.py --out runs/kaitend --diag diag_stage_filter.json \
        --arm 'core_gbv>core_s2v@5' --name 'two-stage: core_gbv to day 5, then core_s2v' \
        --lines 'core_gbv (entropy state as input)=diag:core_gbv' \
                'core_gbn (entropy in the loss)=ens_eval_gbn.json:core_gbn~det' --png fig_twostage_acc_t2m.png
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed order: the two-stage forecast first
CTRL = "#6f6e69"
UNIT = {"t2m": "K", "t850": "K", "z500": "m2/s2", "u850": "m/s", "msl": "Pa"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--diag", default="diag_stage_filter.json")
    p.add_argument("--arm", required=True, help="the two-stage forecast (an arm of the diagnostic)")
    p.add_argument("--name", default=None)
    p.add_argument("--lines", nargs="*", default=[],
                   help="LABEL=diag:ARM (another arm of the diagnostic) or LABEL=FILE.json:KEY (an arm of eval_kai_ensemble.py)")
    p.add_argument("--var", default="t2m")
    p.add_argument("--island", type=int, default=5)
    p.add_argument("--title", default=None)
    p.add_argument("--png", default="fig_stage_filter.png")
    a = p.parse_args()
    D = json.load(open(os.path.join(a.out, a.diag)))
    rows = D["arms"][a.arm]["rows"]
    get = lambda r: dict(acc=np.array(r["acc"]), act=np.array(r["act"]), rmse=np.array(r["vars"][a.var]["rmse"]))  # noqa: E731
    S = [(a.name or a.arm, get(rows["arm"]), dict(color=HUES[0], lw=2.2, marker="o", ms=5.5, zorder=5))]
    for k, spec in enumerate(a.lines):
        label, src = spec.split("=", 1)
        f, key = src.split(":", 1)
        if f == "diag":
            d = get(D["arms"][key]["rows"]["arm"])
        else:
            e = json.load(open(os.path.join(a.out, f)))["arms"][key]
            d = dict(acc=np.array(e["acc_mean"]), act=np.array(e["activity_mean"]), rmse=np.array(e["vars"][a.var]["rmse_mean"]))
        S.append((label, d, dict(color=HUES[k + 1], lw=2, marker="o", ms=5, zorder=4)))
    ctrl = get(rows["ctrl"])
    S.append((f"{D['ctrl']} (control, no entropy)", ctrl, dict(color=CTRL, lw=2, marker="o", ms=5, zorder=3)))
    S.append((f"{D['ctrl']} filtered to the two-stage forecast's smoothness", get(rows["match"]),
              dict(color=CTRL, lw=1.8, ls=(0, (4, 2.5)), zorder=2)))
    x = np.arange(1, len(ctrl["acc"]) + 1)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "lines.solid_capstyle": "round",
                         "lines.markeredgecolor": SURFACE, "lines.markeredgewidth": 1.2})
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2))
    unit = UNIT.get(a.var, "")
    scale = 0.01 if a.var == "msl" else 1.0
    unit = "hPa" if a.var == "msl" else unit
    panels = [
        ("acc", "ACC, mean of 65 channels (higher = better)", lambda d: d["acc"]),
        ("rmse", f"{a.var} RMSE in {unit} (lower = better)", lambda d: d["rmse"] * scale),
        ("dacc", f"ACC minus the control {D['ctrl']} (+ = better)", lambda d: d["acc"] - ctrl["acc"]),
        ("act", "activity: forecast anomaly size / observed (1 = as active as nature)", lambda d: d["act"]),
    ]
    for q, (key, ttl, fn) in zip(ax.ravel(), panels):
        q.axvline(a.island + 0.5, color=GRID, lw=1, zorder=1)
        for label, d, st in S:
            if key == "dacc" and d is ctrl:
                q.axhline(0, color=CTRL, lw=1.2, zorder=1)
                continue
            q.plot(x, fn(d), label=label, **st)
        q.set_title(ttl, fontsize=10.5, loc="left", color=INK)
        q.set_xticks(x)
        q.set_xlabel("lead (days)")
        q.grid(axis="y", color=GRID, lw=0.8)
        q.set_axisbelow(True)
        for s in ("top", "right"):
            q.spines[s].set_visible(False)
        q.tick_params(length=0)
        lo, hi = q.get_ylim()
        q.text(a.island + 0.58, hi - 0.02 * (hi - lo), "second stage starts", color=MUTED, fontsize=8.5, va="top")
    # direct labels at the last day (not in the first panel, where the lines lie on each other); labels that would
    # touch are moved apart as a group, equal ones are written once
    for q, (key, _, fn) in zip(ax.ravel(), panels):
        q.set_xlim(0.6, x[-1] + 1.0)
        if key == "acc":
            continue
        fmt = "{:+.3f}" if key == "dacc" else "{:.2f}"
        ends = {}
        for label, d, _ in S:
            if not (key == "dacc" and d is ctrl):
                ends.setdefault(fmt.format(fn(d)[-1]), fn(d)[-1])
        txt = sorted(ends, key=ends.get)
        pos = np.array([ends[t] for t in txt], float)
        lo, hi = q.get_ylim()
        sep = 0.052 * (hi - lo)
        for _ in range(20):  # clusters of touching labels: spaced by sep about their mean
            groups, g = [], [0]
            for i in range(1, len(pos)):
                if pos[i] - pos[i - 1] < sep * 0.999:
                    g.append(i)
                else:
                    groups.append(g)
                    g = [i]
            groups.append(g)
            if all(len(g) == 1 for g in groups):
                break
            for g in groups:
                pos[g] = pos[g].mean() + sep * (np.arange(len(g)) - (len(g) - 1) / 2)
        for t, y in zip(txt, pos):
            q.text(x[-1] + 0.2, y, t, va="center", fontsize=8.5, color=MUTED)
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=3.2, columnspacing=2.5)
    fig.suptitle(a.title or f"Two-stage forecast vs single-stage cores, {D['ics']} ICs, one deterministic run each",
                 fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.085, 1, 0.965), h_pad=2.2, w_pad=2.0)
    out = os.path.join(a.out, a.png)
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
