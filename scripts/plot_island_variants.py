"""Figure of scripts/diag_island_ensemble.py: the gain of every island variant at EQUAL smoothing with its 95 %
interval, against the scalar island (left) and, for the variants fitted on top of a saved increment (--base), against
that base (right).

    python scripts/plot_island_variants.py --out runs/kaitend --rows 'LABEL=FILE.json:VARIANT:GROUP' ... \
        --rows-base 'LABEL=FILE.json:VARIANT:GROUP' ... --groups 'members' 'continuous operator' --png X.png
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed order of the groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--png", required=True)
    p.add_argument("--rows", nargs="+", required=True, help="LABEL=FILE.json:VARIANT:GROUP (group = index into --groups)")
    p.add_argument("--rows-base", nargs="*", default=[]); p.add_argument("--groups", nargs="+", required=True)
    p.add_argument("--key", default="vs_match"); p.add_argument("--title", default=None)
    p.add_argument("--left-title", default="gain over the scalar island, nRMSE % at equal smoothing (+ = better)")
    p.add_argument("--right-title", default="what the variant adds to the matrix island, % (+ = better)")
    a = p.parse_args()
    cache = {}

    def get(spec, base):
        lab, rest = spec.split("=", 1); f, v, g = rest.split(":")
        J = cache.setdefault(f, json.load(open(os.path.join(a.out, f)))); o = J["variants"][v]
        d = (o["vs_base"] if base else o)[a.key]
        return lab, d["gain_pct"], d["gain_lo"], d["gain_hi"], int(g)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK, "xtick.color": MUTED,
                         "ytick.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "lines.solid_capstyle": "round"})
    panels = [([get(s, False) for s in a.rows], a.left_title)] + ([([get(s, True) for s in a.rows_base], a.right_title)] if a.rows_base else [])
    n = max(len(r) for r, _ in panels)
    fig, ax = plt.subplots(1, len(panels), figsize=(6.4 * len(panels) + 0.6, 0.42 * n + 2.3), squeeze=False)
    lo = min(r[2] for rows, _ in panels for r in rows); hi = max(r[3] for rows, _ in panels for r in rows); pad = 0.12 * (hi - lo)
    for q, (rows, ttl) in zip(ax[0], panels):
        for i, (lab, gp, gl, gh, grp) in enumerate(rows):
            y = -i; c = HUES[grp]
            q.plot([gl, gh], [y, y], color=c, lw=3.2, alpha=0.35, zorder=2)
            q.plot([gp], [y], marker="o", ms=8, color=c, markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
            q.text(gh + 0.25 * pad, y, "{:+.2f}".format(gp), va="center", ha="left", fontsize=9, color=INK)
        q.axvline(0, color="#6f6e69", lw=1.2, zorder=1)
        q.set_yticks([-i for i in range(len(rows))]); q.set_yticklabels([r[0] for r in rows]); q.set_ylim(-n + 0.4, 0.6)
        q.set_xlim(lo - pad, hi + 2.2 * pad); q.set_title(ttl, fontsize=10.5, loc="left", color=INK)
        q.grid(axis="x", color=GRID, lw=0.8); q.set_axisbelow(True); q.tick_params(length=0)
        for s in ("top", "right", "left"):
            q.spines[s].set_visible(False)
    used = sorted({r[4] for rows, _ in panels for r in rows})
    h = [plt.Line2D([], [], marker="o", ms=8, lw=3.2, color=HUES[g], markeredgecolor=SURFACE, label=a.groups[g] + " (dot = gain, bar = 95 % interval)") for g in used]
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    if a.title:
        fig.suptitle(a.title, fontsize=12, x=0.012, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94 if a.title else 1))
    fig.savefig(a.png, dpi=170); print("saved", a.png)


if __name__ == "__main__":
    main()
