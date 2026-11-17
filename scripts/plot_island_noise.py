"""Lead-time figure of the island experiments: one core with noise or a correction at the island against the
two-core pair.

Four panels over the lead day: ACC (65 scored channels), nRMSE, the ACC difference from the single core and the
activity. Lines: runs of scripts/diag_island_noise.py / diag_island_kalman.py (``--runs LABEL=FILE.json:RUN``; RUN =
a key of the file's "runs", "central" or "limit" = the many-member limit of the file's first run from its sums),
the single core, the two-core pair and the single core filtered to the pair's smoothness (rows of a
scripts/diag_stage_filter.py json: ``--pair-diag``, ``--pair-arm``).

    python scripts/plot_island_noise.py --out runs/kaitend --pair-diag diag_stage_filter_s2c.json \
        --pair-arm 'core_gb>core_s2c@5' --runs 'one core + island noise, 16 members=diag_island_noise_gb_err1.json:isn-core_gb-err1-m16' \
        'same, many members (estimate)=diag_island_noise_gb_err1.json:limit' --png fig_island_noise_gb.png
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed order: the island runs first, the pair last
CTRL = "#6f6e69"
N_SC = 65


def limit(path):
    """The many-member limit of the first run of a diag_island_noise.py json: the power of a mean of k pairs is
    A + B / k per lead and channel (least squares over k), its covariance with the truth does not depend on k."""
    J = json.load(open(path))
    name = next(iter(J["runs"]))
    z = np.load(path[:-5] + "_sums.npz")
    ks = np.arange(1, J["members"] // 2 + 1)
    S = np.stack([z[f"{name}__m{2 * k}"] for k in ks])           # (pairs, 4, L, 66): mse, pp, tt, pt
    pp, tt, pt = S[:, 1], S[0, 2], S[:, 3].mean(0)
    A = np.linalg.lstsq(np.stack([np.ones(len(ks)), 1 / ks], 1), pp.reshape(len(ks), -1), rcond=None)[0][0].reshape(pp.shape[1:])
    A = np.where(ks.size and J["n1"] > 0, A, A)
    A[:J["n1"]] = pp[-1, :J["n1"]]                                 # before the island every member is the central run
    ok = tt.mean(0) > 0
    sc = np.flatnonzero(ok)[:66]
    sc = [i for i in sc if not np.allclose(pp[-1, :, i], tt[:, i])]  # the forcing channel is the truth itself
    mse = A - 2 * pt + tt
    return dict(acc=(pt / np.sqrt(A * tt))[:, sc].mean(1), act=np.sqrt(A / tt)[:, sc].mean(1),
                nrmse=np.sqrt(np.maximum(mse, 0))[:, sc].mean(1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="LABEL=FILE.json:RUN")
    p.add_argument("--pair-diag", required=True)
    p.add_argument("--pair-arm", required=True)
    p.add_argument("--pair-name", default="two cores: a second core trained for day 6-10")
    p.add_argument("--core-name", default=None)
    p.add_argument("--island", type=int, default=5)
    p.add_argument("--note", default="noise / correction here")
    p.add_argument("--title", default=None)
    p.add_argument("--png", default="fig_island_noise.png")
    a = p.parse_args()
    D = json.load(open(os.path.join(a.out, a.pair_diag)))
    rows = D["arms"][a.pair_arm]["rows"]
    get = lambda r: dict(acc=np.array(r["acc"]), act=np.array(r["act"]), nrmse=np.array(r["nrmse"]))  # noqa: E731
    S, dashes = [], [None, (0, (1.2, 1.8)), (0, (5, 2)), (0, (3, 1.5, 1, 1.5))]
    files = []
    for spec in a.runs:
        label, src = spec.split("=", 1)
        f, key = src.rsplit(":", 1)
        if f not in files:
            files.append(f)
        k = files.index(f)                      # one hue per file (= per experiment), dashes for its variants
        v = sum(1 for s in S if s[2]["color"] == HUES[k])
        path = os.path.join(a.out, f)
        if key == "limit":
            d = limit(path)
        else:
            J = json.load(open(path))
            d = get(J["central"] if key == "central" else J["runs"][key]["mean"])
        st = dict(color=HUES[k], lw=2.2 if v == 0 else 1.9, zorder=6 - v)
        if v == 0:
            st.update(marker="o", ms=5.5)
        else:
            st.update(ls=dashes[min(v, 3)])
        S.append((label, d, st))
    S.append((a.pair_name, get(rows["arm"]), dict(color=HUES[len(files)], lw=2, marker="o", ms=5, zorder=4)))
    ctrl = get(rows["ctrl"])
    S.append((a.core_name or f"{D['ctrl']}: one core, no noise", ctrl, dict(color=CTRL, lw=2, marker="o", ms=5, zorder=3)))
    S.append((f"{D['ctrl']} filtered to the two-core smoothness", get(rows["match"]),
              dict(color=CTRL, lw=1.8, ls=(0, (4, 2.5)), zorder=2)))
    x = np.arange(1, len(ctrl["acc"]) + 1)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "lines.solid_capstyle": "round",
                         "lines.markeredgecolor": SURFACE, "lines.markeredgewidth": 1.2})
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.4))
    panels = [
        ("acc", "ACC, mean of 65 channels (higher = better)", lambda d: d["acc"]),
        ("nrmse", "normalized RMSE, mean of 65 channels (lower = better)", lambda d: d["nrmse"]),
        ("dacc", "ACC minus the single core (+ = better)", lambda d: d["acc"] - ctrl["acc"]),
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
        if key in ("dacc", "act"):   # the zoom on the days after the island lives in these two panels
            pass
        lo, hi = q.get_ylim()
        q.text(a.island + 0.58, hi - 0.02 * (hi - lo), a.note, color=MUTED, fontsize=8.5, va="top")
    for q, (key, _, fn) in zip(ax.ravel(), panels):   # direct labels at the last day, touching ones moved apart
        q.set_xlim(0.6, x[-1] + 1.0)
        fmt = "{:+.3f}" if key == "dacc" else "{:.3f}" if key in ("acc", "nrmse") else "{:.2f}"
        ends = {}
        for label, d, _ in S:
            if not (key == "dacc" and d is ctrl):
                ends.setdefault(fmt.format(fn(d)[-1]), fn(d)[-1])
        txt = sorted(ends, key=ends.get)
        pos = np.array([ends[t] for t in txt], float)
        lo, hi = q.get_ylim()
        sep = 0.052 * (hi - lo)
        for _ in range(30):
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
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=9.5, handlelength=3.2,
               bbox_to_anchor=(0.5, 0.0))
    if a.title:
        fig.suptitle(a.title, x=0.012, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.045 + 0.022 * ((len(S) + 1) // 2), 1, 0.965 if a.title else 1))
    fig.savefig(os.path.join(a.out, a.png), dpi=150)
    print("wrote", os.path.join(a.out, a.png))


if __name__ == "__main__":
    main()
