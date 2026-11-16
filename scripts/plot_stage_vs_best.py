"""Lead-time figure of a two-stage forecast against the best models measured before.

Six panels over the lead day: nRMSE (65 scored channels), its gain over the control, the ACC difference from the
control, RMSE of two variables (physical units) and the activity. Sources:

* the two-stage forecast, the control and the control at equal smoothing: scripts/diag_stage_filter.py (ACC and
  activity in the score files' definition, ``acc_fc`` / ``act_fc``: per forecast, then the mean);
* models of a score file (run_kai_tendency.py --score: all ten leads; activity from its raw file);
* models of a variable table (scripts/kai_variable_table.py, day 1 / 3 / 5 / 7 / 10: RMSE of the variables; nRMSE given
  on the command line; no ACC / activity).

    python scripts/plot_stage_vs_best.py --out runs/kaitend --arm 'core_gbv>core_s2v@5' --name 'two-stage core' \
        --score score_mem.json --raw score_mem_raw.npz --score-lines 'best core alone=core_gndmz8' \
        --table variable_table_S4.md --table-lines 'best model, loop=COLUMN NAME=0.145,0.261,0.358,0.433,0.509'
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
BLUE, ORANGE, AQUA, CTRL = "#2a78d6", "#eb6834", "#1baf7a", "#6f6e69"
UNIT = {"t2m": "K", "t850": "K", "z500": "m2/s2", "u850": "m/s", "msl": "hPa"}
TDAYS = np.array([1, 3, 5, 7, 10])


def table_column(path, column, var):
    """RMSE of ``var`` at day 1 / 3 / 5 / 7 / 10 from the first table of a variable table (markdown)."""
    head = None
    for line in open(path, encoding="utf-8"):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if head is None and c and c[0] == "변수":
            head = c
        elif head is not None and c and c[0] == var:
            return np.array([float(x) for x in c[head.index(column)].split("/")])
    raise KeyError((column, var))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--diag", default="diag_stage_filter.json")
    p.add_argument("--arm", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--score", default=None)
    p.add_argument("--raw", default=None)
    p.add_argument("--score-lines", nargs="*", default=[], help="LABEL=ARM of the score file")
    p.add_argument("--table", default=None)
    p.add_argument("--table-lines", nargs="*", default=[], help="LABEL=COLUMN=nRMSE at day 1,3,5,7,10")
    p.add_argument("--vars", nargs=2, default=["z500", "t2m"])
    p.add_argument("--island", type=int, default=5)
    p.add_argument("--title", default=None)
    p.add_argument("--png", default="fig_stage_vs_best.png")
    a = p.parse_args()
    D = json.load(open(os.path.join(a.out, a.diag)))
    rows = D["arms"][a.arm]["rows"]
    x = np.arange(1, len(rows["ctrl"]["nrmse"]) + 1)

    def get(r):
        return dict(x=x, nrmse=np.array(r["nrmse"]), acc=np.array(r["acc_fc"]), act=np.array(r["act_fc"]),
                    **{v: np.array(r["vars"][v]["rmse"]) * (0.01 if v == "msl" else 1) for v in a.vars})

    ctrl = get(rows["ctrl"])
    S = [(a.name or a.arm, get(rows["arm"]), dict(color=BLUE, lw=2.3, marker="o", ms=5.5, zorder=6))]
    if a.score:
        sj = json.load(open(os.path.join(a.out, a.score)))
        ch = sj["channels"]
        raw = np.load(os.path.join(a.out, a.raw)) if a.raw else None
        scored = [i for i, c in enumerate(ch) if c not in ("orog", "orography", "tisr")]
        for k, spec in enumerate(a.score_lines):
            label, arm = spec.split("=", 1)
            e = sj["arms"][arm]
            d = dict(x=x, nrmse=np.array(e["nrmse"]), acc=np.array(e["acc_mean"]),
                     **{v: np.array(e["rmse"])[:, ch.index(v)] * (0.01 if v == "msl" else 1) for v in a.vars})
            if raw is not None:
                d["act"] = raw[f"{arm}__act"].mean(0)[:, scored].mean(1)
            S.append((label, d, dict(color=ORANGE, lw=2, marker="o", ms=5, zorder=5, ls=["-", (0, (1, 1.6))][k % 2])))
    for k, spec in enumerate(a.table_lines):
        label, col, nr = spec.split("=", 2)
        d = dict(x=TDAYS, nrmse=np.array([float(v) for v in nr.split(",")]),
                 **{v: table_column(a.table, col, v) for v in a.vars})
        S.append((label, d, dict(color=AQUA, lw=2, marker="o", ms=5, zorder=4, ls=["-", (0, (1, 1.6))][k % 2])))
    S.append((f"{D['ctrl']} (control: same inputs, one stage)", ctrl, dict(color=CTRL, lw=2, marker="o", ms=5, zorder=3)))
    S.append((f"{D['ctrl']} filtered to the two-stage forecast's smoothness", get(rows["match"]),
              dict(color=CTRL, lw=1.8, ls=(0, (4, 2.5)), zorder=2)))
    at = lambda d, key: np.interp(d["x"], x, ctrl[key])  # noqa: E731  the control at the days of a series
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "lines.solid_capstyle": "round",
                         "lines.markeredgecolor": SURFACE, "lines.markeredgewidth": 1.2})
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.6))
    v1, v2 = a.vars
    note = " (best model: not saved)" if a.table_lines else ""
    panels = [
        ("nrmse", "nRMSE, mean of 65 channels (lower = better)", lambda d: d["nrmse"], "{:.3f}"),
        ("gain", f"nRMSE gain over {D['ctrl']} in % (+ = better)", lambda d: 100 * (1 - d["nrmse"] / at(d, "nrmse")), "{:+.1f}"),
        ("dacc", f"ACC minus {D['ctrl']} (+ = better){note}", lambda d: d["acc"] - at(d, "acc"), "{:+.3f}"),
        (v1, f"{v1} RMSE in {UNIT.get(v1, '')} (lower = better)", lambda d: d[v1], "{:.0f}" if v1 == "z500" else "{:.2f}"),
        (v2, f"{v2} RMSE in {UNIT.get(v2, '')} (lower = better)", lambda d: d[v2], "{:.0f}" if v2 == "z500" else "{:.2f}"),
        ("act", f"activity: forecast anomaly / observed{note}", lambda d: d["act"], "{:.2f}"),
    ]
    for q, (key, ttl, fn, fmt) in zip(ax.ravel(), panels):
        need = {"dacc": "acc", "act": "act"}.get(key)
        q.axvline(a.island + 0.5, color=GRID, lw=1, zorder=1)
        ends = []
        for n, (label, d, st) in enumerate(S):
            if need and need not in d:
                continue
            if key in ("gain", "dacc") and d is ctrl:
                q.axhline(0, color=CTRL, lw=1.2, zorder=1)
                continue
            y = fn(d)
            q.plot(d["x"], y, label=label, **st)
            ends.append((0 if n == 0 else 1 if d is ctrl else 2 + n, y[-1], fmt.format(y[-1])))
        q.set_title(ttl, fontsize=10.5, loc="left", color=INK)
        q.set_xticks(x)
        q.set_xlabel("lead (days)")
        q.grid(axis="y", color=GRID, lw=0.8)
        q.set_axisbelow(True)
        for s in ("top", "right"):
            q.spines[s].set_visible(False)
        q.tick_params(length=0)
        q.set_xlim(0.6, x[-1] + 1.15)
        lo, hi = q.get_ylim()
        if key == "nrmse":
            q.text(a.island + 0.58, lo + 0.02 * (hi - lo), "second stage starts", color=MUTED, fontsize=8.5, va="bottom")
        # values at the last day, selectively: the two-stage forecast first, the control next, another series only
        # where its label does not touch one already written
        put = []
        for _, y, t in sorted(ends):
            if all(abs(y - z) >= 0.055 * (hi - lo) for z in put):
                q.text(x[-1] + 0.2, y, t, va="center", fontsize=8.5, color=MUTED)
                put.append(y)
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=3.4, columnspacing=2.5)
    fig.suptitle(a.title or f"Two-stage forecast vs the best models measured before, {D['ics']} ICs", fontsize=12.5,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.075, 1, 0.965), h_pad=2.2, w_pad=2.0)
    out = os.path.join(a.out, a.png)
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
