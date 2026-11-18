"""Figure of scripts/diag_island_shock.py (part 2): by quartile of a sensor at the island, the later error variance,
the best damping factor and the correlation with the truth; and the gain of a sensor-dependent damping.

    python scripts/plot_island_shock.py --json runs/kaitend/diag_island_shock.json --png runs/kaitend/fig_island_shock.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
RAMP = ["#a9cbf2", "#6aa3e6", "#2a78d6", "#143f7a"]            # quartile 1 (lowest sensor) .. 4, one hue light to dark
SENS = {"gradient": ("#eb6834", "gradient of the island state"), "disagree": ("#1baf7a", "disagreement of the core and KAI")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True); p.add_argument("--png", required=True)
    p.add_argument("--sensor", default="gradient"); p.add_argument("--title", default=None)
    a = p.parse_args()
    J = json.load(open(a.json)); S = J["sensor"]; x = np.array(S["leads"]); Q = S["quartile"][a.sensor]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "lines.solid_capstyle": "round",
                         "lines.markeredgecolor": SURFACE, "lines.markeredgewidth": 1.2})
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
    name = SENS[a.sensor][1]
    panels = [("err", "error variance by quartile of the sensor (1 = the mean)"),
              ("beta", "best damping factor of the forecast, by quartile"),
              ("skill", "correlation of the forecast with the truth, by quartile"),
              ("gain", "gain of a sensor-dependent damping, nRMSE % (+ = better)")]
    for q, (key, ttl) in zip(ax.ravel(), panels):
        ends = []
        if key == "gain":
            for s, (col, lab) in SENS.items():
                d = S["damping"][s]["region"]
                q.fill_between(x, d["gain_lo"], d["gain_hi"], color=col, alpha=0.16, lw=0, zorder=1)
                q.plot(x, d["gain_pct"], color=col, lw=2.2, marker="o", ms=5.5, label="sensor: " + lab, zorder=3)
                ends.append((d["gain_pct"][-1], "{:+.2f}".format(d["gain_pct"][-1])))
            q.axhline(0, color="#6f6e69", lw=1.2, zorder=1); q.set_ylim(-1.0, 1.0)
        else:
            v = np.array(Q[key])
            for j in range(4):
                q.plot(x, v[:, j], color=RAMP[j], lw=2.2, marker="o", ms=5.5, zorder=3 + j,
                       label="quartile {} of the sensor ({}){}".format(j + 1, name, ", lowest" if j == 0 else ", highest" if j == 3 else ""))
                ends.append((v[-1, j], "{:.2f}".format(v[-1, j])))
        q.set_title(ttl, fontsize=10.5, loc="left", color=INK)
        q.set_xticks(x); q.set_xlim(x[0] - 0.4, x[-1] + 0.9); q.set_xlabel("lead (days); the sensor is read at the island, day {}".format(J["island"]))
        q.grid(axis="y", color=GRID, lw=0.8); q.set_axisbelow(True)
        for s in ("top", "right"):
            q.spines[s].set_visible(False)
        q.tick_params(length=0)
        lo, hi = q.get_ylim(); sep = 0.055 * (hi - lo); ends.sort(key=lambda e: e[0]); ys = [e[0] for e in ends]
        for i in range(1, len(ys)):                       # labels at the last day, pushed apart where they touch
            ys[i] = max(ys[i], ys[i - 1] + sep)
        sh = (np.mean([e[0] for e in ends]) - np.mean(ys)) if len(ys) > 1 else 0
        for (v0, t), y in zip(ends, ys):
            q.text(x[-1] + 0.15, y + sh, t, color=INK, fontsize=8.5, va="center")
    h, l = ax[0, 0].get_legend_handles_labels(); h2, l2 = ax[1, 1].get_legend_handles_labels()
    fig.legend(h + h2, ["panels 1-3: " + t for t in l] + ["panel 4: " + t for t in l2], loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    if a.title:
        fig.suptitle(a.title, fontsize=12.5, x=0.012, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.965 if a.title else 1))
    fig.savefig(a.png, dpi=170); print("saved", a.png)


if __name__ == "__main__":
    main()
