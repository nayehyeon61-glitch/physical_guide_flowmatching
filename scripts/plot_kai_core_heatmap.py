"""Heat maps of one model against several others, from the csv of scripts/kai_variable_table.py (--csv --stats --raw).

    python scripts/plot_kai_core_heatmap.py variable_rmse.csv outprefix --base core_gngmz8 \
        "core_gn=basic core" "core_gngmz=final core" ... [--base-label "8-day core"] [--lim 15] [--lim-var 20]

Gains in % (+ = the base model is better, blue; - = worse, red). Three views:
  total        100 (1 - RMSE_A / RMSE_B)
  information  the part of it that is left when both forecasts have their best amplitude (csv columns i1..iN)
  amplitude    the rest (what damping or keeping the strength of the field does to the RMSE)
Files: <outprefix>_overview.png  rows = the other models, columns = lead; all scored variables (mean of RMSE / norm,
                                 the nRMSE of the score), one panel per view
       <outprefix>_variables_total.png / _variables_information.png
                                 one panel per other model, rows = every variable, columns = lead
A model without the i columns (no raw metrics) has empty information / amplitude cells.
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CMAP = LinearSegmentedColormap.from_list("gain", ["#b93535", "#e66767", "#f4abab", "#f0efec", "#9ec5f4", "#5598e7", "#256abf"])
CMAP.set_bad("#ffffff")


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    nl = sum(1 for k in rows[0] if k[0] == "d" and k[1:].isdigit())
    num = lambda r, p: np.array([float(r[f"{p}{i + 1}"]) if r.get(f"{p}{i + 1}", "") != "" else np.nan for i in range(nl)])  # noqa: E731
    M = {}
    for r in rows:
        M.setdefault(r["model"], {})[r["variable"]] = (num(r, "d"), num(r, "i"), float(r["norm"]), r["group"])
    return M, nl


def gains(M, a, b, names):
    """(total, information, amplitude) per variable (V, L) and for all variables (L,), in %."""
    A = np.stack([M[a][n][0] for n in names])
    B = np.stack([M[b][n][0] for n in names])
    ia = np.stack([M[a][n][1] for n in names])
    ib = np.stack([M[b][n][1] for n in names])
    s = np.array([M[a][n][2] for n in names])[:, None]
    tot, inf = 100 * (1 - A / B), 100 * (ib ** 2 - ia ** 2) / (B * (A + B))
    den = (B / s).mean(0)
    at, ai = 100 * (1 - (A / s).mean(0) / den), 100 * ((ib ** 2 - ia ** 2) / ((A + B) * s)).mean(0) / den
    return (tot, inf, tot - inf), (at, ai, at - ai)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("out")
    p.add_argument("others", nargs="+", help="tag=label")
    p.add_argument("--base", default="core_gngmz8")
    p.add_argument("--base-label", default="8-day core")
    p.add_argument("--lim", type=float, default=15.0, help="color range of the overview (+-%)")
    p.add_argument("--lim-var", type=float, default=20.0, help="color range of the per-variable panels (+-%)")
    a = p.parse_args(argv)
    M, nl = load(a.csv)
    others = [tuple(o.split("=", 1)) if "=" in o else (o, o) for o in a.others]
    others = [(k, lab) for k, lab in others if k in M]
    names = list(M[a.base])
    G = {k: gains(M, a.base, k, names) for k, _ in others}
    leads = np.arange(1, nl + 1)

    # overview: other models x lead, three views
    fig, axs = plt.subplots(1, 3, figsize=(17, 0.36 * len(others) + 1.9), sharey=True)
    for ax, j, title in zip(axs, range(3), ["total RMSE gain", "information gain (amplitude effect removed)", "amplitude effect"]):
        Z = np.stack([G[k][1][j] for k, _ in others])
        ax.imshow(np.ma.masked_invalid(Z), cmap=CMAP, vmin=-a.lim, vmax=a.lim, aspect="auto")
        for i in range(Z.shape[0]):
            for d in range(nl):
                if np.isfinite(Z[i, d]):
                    ax.text(d, i, f"{Z[i, d]:+.1f}" if abs(Z[i, d]) < 9.95 else f"{Z[i, d]:+.0f}", ha="center", va="center", fontsize=7.5,
                            color="white" if abs(Z[i, d]) > 0.62 * a.lim else "#1a1a1a")
            if not np.isfinite(Z[i]).any():
                ax.text((nl - 1) / 2, i, "no term data", ha="center", va="center", fontsize=7.5, color="#888888")
        ax.set_xticks(range(nl), leads)
        ax.set_yticks(range(len(others)), [f"vs {lab}" for _, lab in others])
        ax.set_xlabel("lead (days)")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(-0.5, nl), minor=True)
        ax.set_yticks(np.arange(-0.5, len(others)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"{a.base_label} against the other cores: gain in % over all {len(names)} variables (+ blue = {a.base_label} better, "
                 f"- red = worse; colors clipped at +-{a.lim:g} %)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.out + "_overview.png", dpi=150)
    plt.close(fig)

    # per variable: one panel per other model
    grp = [M[a.base][n][3] for n in names]
    cuts = [i for i in range(1, len(names)) if grp[i] != grp[i - 1]]
    for j, view in ((0, "total"), (1, "information")):
        ks = [(k, lab) for k, lab in others if np.isfinite(G[k][0][j]).any()]
        nc = min(6, len(ks))
        nr = int(np.ceil(len(ks) / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(2.75 * nc + 0.9, 8.6 * nr + 0.7), squeeze=False)
        for q, ax in enumerate(axs.ravel()):
            if q >= len(ks):
                ax.axis("off")
                continue
            k, lab = ks[q]
            im = ax.imshow(np.ma.masked_invalid(G[k][0][j]), cmap=CMAP, vmin=-a.lim_var, vmax=a.lim_var, aspect="auto", interpolation="nearest")
            ax.set_title(f"vs {lab}", fontsize=9)
            ax.set_xticks(range(nl), leads, fontsize=7)
            ax.set_yticks(range(len(names)), names if q % nc == 0 else [""] * len(names), fontsize=5.5)
            ax.tick_params(length=0)
            for c in cuts:
                ax.axhline(c - 0.5, color="white", linewidth=2)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if q // nc == nr - 1 or q + nc >= len(ks):
                ax.set_xlabel("lead (days)", fontsize=8)
        fig.suptitle(f"{a.base_label} against the other cores, {view} gain in % per variable and lead "
                     f"(+ blue = {a.base_label} better, - red = worse)", fontsize=11)
        fig.tight_layout(rect=(0, 0.035, 1, 0.975))
        cb = fig.colorbar(im, cax=fig.add_axes((0.3, 0.012, 0.4, 0.008)), orientation="horizontal", extend="both")
        cb.ax.tick_params(labelsize=7)
        fig.savefig(f"{a.out}_variables_{view}.png", dpi=130)
        plt.close(fig)
    print("overview, all variables (total | information), lead 1 / 3 / 5 / 7 / 10")
    L = [x for x in (0, 2, 4, 6, 9) if x < nl]
    for k, lab in others:
        t, i, _ = G[k][1]
        print(f"  {lab:28s}", " / ".join(f"{t[x]:+.1f}" for x in L), "|", " / ".join(f"{i[x]:+.1f}" if np.isfinite(i[x]) else "-" for x in L))


if __name__ == "__main__":
    main()
