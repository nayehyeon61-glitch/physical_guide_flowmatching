"""Lead-time figure of rows of scripts/diag_stage_filter_light.py jsons (cached forecasts scored against one
control): ACC, nRMSE, the nRMSE gain at EQUAL smoothing against the files' control and the activity.

    python scripts/plot_filter_rows.py --out runs/kaitend --ref 'REF LABEL=FILE.json' \
        --series 'LABEL=FILE.json:ARM' ... [--extra 'LABEL=FILE.json:ctrl' ...] [--points 'LABEL=1:.145,3:.261'] --png X.png

``--ref`` = the control row of a file (gray line, the zero of the gain panel); ``--series`` = arms of files with
that control (one hue each, fixed order; their ``arm_vs_match`` gain with its 95 % interval goes to the third
panel); ``--extra`` = other rows without a gain line (dashed gray), ``--points`` = published nRMSE values (markers).
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed order of the series
CTRL, AUX, PTS = "#6f6e69", "#9c9a92", "#7a4fd0"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ref", required=True, help="LABEL=FILE.json (its control row)")
    p.add_argument("--series", nargs="+", required=True, help="LABEL=FILE.json:ARM")
    p.add_argument("--extra", nargs="*", default=[], help="LABEL=FILE.json:ARM or :ctrl")
    p.add_argument("--points", nargs="*", default=[], help="LABEL=day:nrmse,day:nrmse")
    p.add_argument("--island", type=int, default=5)
    p.add_argument("--note", default="Kalman island")
    p.add_argument("--gain-title", default="nRMSE gain at equal smoothing against the control, % (+ = better)")
    p.add_argument("--title", default=None)
    p.add_argument("--png", default="fig_filter_rows.png")
    a = p.parse_args()
    load = lambda f: json.load(open(os.path.join(a.out, f)))  # noqa: E731
    get = lambda r: dict(acc=np.array(r["acc"]), act=np.array(r["act"]), nrmse=np.array(r["nrmse"]))  # noqa: E731
    lab, f = a.ref.split("=", 1)
    ref = get(load(f)["ctrl_rows"]["ctrl"])
    S = [(lab, ref, dict(color=CTRL, lw=2, marker="o", ms=5, zorder=3), None)]
    for spec in a.extra:
        lab, src = spec.split("=", 1)
        f, key = src.rsplit(":", 1)
        J = load(f)
        r = J["ctrl_rows"]["ctrl"] if key == "ctrl" else J["arms"][key]["rows"]["arm"]
        S.append((lab, get(r), dict(color=AUX, lw=1.8, ls=(0, (4, 2.5)), zorder=2), None))
    for k, spec in enumerate(a.series):
        lab, src = spec.split("=", 1)
        f, key = src.rsplit(":", 1)
        A = load(f)["arms"][key]
        g = A["arm_vs_match"]
        S.append((lab, get(A["rows"]["arm"]), dict(color=HUES[k], lw=2.2, marker="o", ms=5.5, zorder=6 - k),
                  (np.array(g["gain_pct"]), np.array(g["gain_lo"]), np.array(g["gain_hi"]))))
    x = np.arange(1, len(ref["acc"]) + 1)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "lines.solid_capstyle": "round",
                         "lines.markeredgecolor": SURFACE, "lines.markeredgewidth": 1.2})
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
    panels = [("acc", "ACC, mean of 65 channels (higher = better)"),
              ("nrmse", "normalized RMSE, mean of 65 channels (lower = better)"),
              ("gain", a.gain_title),
              ("act", "activity: forecast anomaly size / observed (1 = as active as nature)")]
    for q, (key, ttl) in zip(ax.ravel(), panels):
        q.axvline(a.island + 0.5, color=GRID, lw=1, zorder=1)
        ends = []
        for lab, d, st, g in S:
            if key == "gain":
                if g is None:
                    continue
                q.fill_between(x, g[1], g[2], color=st["color"], alpha=0.16, lw=0, zorder=1)
                q.plot(x, g[0], label=lab, **st)
                ends.append((g[0][-1], "{:+.1f}".format(g[0][-1]), st["color"]))
            else:
                q.plot(x, d[key], label=lab, **st)
                ends.append((d[key][-1], ("{:.2f}" if key == "act" else "{:.3f}").format(d[key][-1]), st["color"]))
        if key == "gain":
            q.axhline(0, color=CTRL, lw=1.2, zorder=1)
        if key == "nrmse":
            for spec in a.points:
                lab, src = spec.split("=", 1)
                xy = np.array([[float(v) for v in t.split(":")] for t in src.split(",")])
                q.plot(xy[:, 0], xy[:, 1], ls="none", marker="D", ms=6.5, color=PTS, zorder=7, label=lab)
        q.set_title(ttl, fontsize=10.5, loc="left", color=INK)
        q.set_xticks(x)
        q.set_xlim(0.6, x[-1] + 1.0)
        q.set_xlabel("lead (days)")
        q.grid(axis="y", color=GRID, lw=0.8)
        q.set_axisbelow(True)
        for s in ("top", "right"):
            q.spines[s].set_visible(False)
        q.tick_params(length=0)
        lo, hi = q.get_ylim()
        q.text(a.island + 0.58, hi - 0.02 * (hi - lo), a.note, color=MUTED, fontsize=8.5, va="top")
        ends.sort(key=lambda e: e[0])                      # direct labels at the last day, touching ones moved apart
        val, sep = [e[0] for e in ends], 0.05 * (hi - lo)
        groups = [[i] for i in range(len(val))]            # touching labels form a group centred on its values
        place = lambda g: np.mean([val[i] for i in g]) + (np.arange(len(g)) - (len(g) - 1) / 2) * sep  # noqa: E731
        merged = True
        while merged:
            merged = False
            for j in range(len(groups) - 1):
                if place(groups[j + 1])[0] - place(groups[j])[-1] < sep * 0.999:
                    groups[j:j + 2] = [groups[j] + groups[j + 1]]
                    merged = True
                    break
        for g in groups:
            for i, y in zip(g, place(g)):
                q.text(x[-1] + 0.18, y, ends[i][1], color=INK, fontsize=8.5, va="center")
    h, l = ax[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    if a.title:
        fig.suptitle(a.title, fontsize=12.5, x=0.012, ha="left", color=INK)
    n_rows = (len(l) + 1) // 2
    fig.tight_layout(rect=(0, 0.03 + 0.028 * n_rows, 1, 0.965 if a.title else 1))
    path = os.path.join(a.out, a.png)
    fig.savefig(path, dpi=170)
    print("saved", path)


if __name__ == "__main__":
    main()
