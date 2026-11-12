"""Lead-time figure of the 1.5 deg tendency core + run next to the 5.625 deg final model (2 x 2 panels).

(a) z500 RMSE, (b) t850 RMSE, (c) ACC averaged over the scored channels, (d) the gain of ONE core coupled on the
rollout fine-tuned KAI over that KAI (total and information part, + = better) at both grids. Each run is scored
on its own grid, so the physical RMSEs of the two grids are only roughly comparable (see
docs/kai_tendency_plus_1p5_daily.md section 8).

    python scripts/plot_kai_plus_vs_5p625.py \
        --s5 runs/kaitend/score_Lq.json --b5 runs/kaitend/breakdown_Ln/breakdown.json \
        --s15 docs/kai_tendency_plus_1p5_daily/score_plus.json --b15 docs/kai_tendency_plus_1p5_daily/breakdown.json \
        --out docs/kai_tendency_plus_1p5_daily/figures/fig7_vs_5p625.png
"""
import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--s5", required=True, help="5.625 deg score json (with the two-core loop)")
p.add_argument("--b5", required=True, help="5.625 deg breakdown json with the pair <loop5>:kai_ro")
p.add_argument("--s15", required=True)
p.add_argument("--b15", required=True)
p.add_argument("--best5", default="loop_core_gngmz8s1+core_gngmz8@kai_ro")
p.add_argument("--loop5", default="loop_core_gngmz8@kai_ro")
p.add_argument("--best15", default="loop_core_plus@kai_ro")
p.add_argument("--out", required=True)
a = p.parse_args()

S5, S15 = json.load(open(a.s5)), json.load(open(a.s15))
L = np.arange(1, int(S15["leads"]) + 1)
BLUE, ORANGE, AQUA, VIOLET, GRAY, LGRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#555555", "#9a9a9a"
INK, MUTED = "#222222", "#666666"
series = [(S15, a.best15, "1.5° best (fine-tuned KAI + core loop)", BLUE, "-"),
          (S15, "kai", "1.5° KAI (%.2f M)" % (S15["arms"]["kai"]["params"] / 1e6), AQUA, "--"),
          (S5, a.best5, "5.625° best (fine-tuned KAI + two cores)", ORANGE, "-"),
          (S5, "kai", "5.625° KAI (%.2f M)" % (S5["arms"]["kai"]["params"] / 1e6), VIOLET, "--")]
pers = [(S15, "persistence (1.5°)", GRAY), (S5, "persistence (5.625°)", LGRAY)]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#bbbbbb", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlesize": 10, "axes.titleweight": "bold"})
fig, ax = plt.subplots(2, 2, figsize=(10, 7.6))


def style(x, title, ylabel):
    x.set_title(title, loc="left", color=INK)
    x.set_xlabel("lead (days)")
    x.set_ylabel(ylabel)
    x.set_xticks(L)
    x.grid(True, color="#e6e6e6", lw=0.6)
    for s in ("top", "right"):
        x.spines[s].set_visible(False)


def curve(S, arm, key, v=None):
    y = np.array(S["arms"][arm][key])
    return y[:, S["channels"].index(v)] if v else y


for x, v, title, unit in [(ax[0, 0], "z500", "(a) z500 RMSE (lower is better)", "m²/s²"),
                          (ax[0, 1], "t850", "(b) t850 RMSE (lower is better)", "K")]:
    for S, lab, col in pers:
        x.plot(L, curve(S, "persistence", "rmse", v), ":", color=col, lw=1.6, label=lab)
    for S, arm, lab, col, ls in series:
        x.plot(L, curve(S, arm, "rmse", v), ls, color=col, lw=2, marker="o", ms=3.5, label=lab)
    style(x, title, f"RMSE ({unit})")
x = ax[1, 0]
for S, lab, col in pers:
    x.plot(L, curve(S, "persistence", "acc_mean"), ":", color=col, lw=1.6, label=lab)
for S, arm, lab, col, ls in series:
    x.plot(L, curve(S, arm, "acc_mean"), ls, color=col, lw=2, marker="o", ms=3.5, label=lab)
style(x, "(c) ACC, mean of the scored channels (higher is better)", "ACC")


def terms(path, pair):
    d = np.array(json.load(open(path))[pair]["dterm2_pct"])  # (L, 3): bias, pattern, amplitude gains in %
    return d.sum(1), d[:, 0] + d[:, 1]


x = ax[1, 1]
x.axhline(0, color="#999999", lw=0.8)
for path, pair, g, col, dy in [(a.b5, f"{a.loop5}:kai_ro", "5.625°", ORANGE, 0), (a.b15, f"{a.best15}:kai_ro", "1.5°", BLUE, -6)]:
    t, i = terms(path, pair)
    x.plot(L, t, "-", color=col, lw=2, marker="o", ms=3.5, label=f"{g}: total gain")
    x.plot(L, i, "--", color=col, lw=2, marker="s", ms=3.5, label=f"{g}: information (pattern) part")
    x.annotate(f"{t[0]:+.1f}", (1, t[0]), xytext=(6, dy), textcoords="offset points", color=INK, va="center")
    print(g, "core gain", np.round(t, 1), "information", np.round(i, 1))
style(x, "(d) one core's gain over its own fine-tuned KAI (%, + = better)", "nRMSE gain (%)")
x.legend(frameon=False, fontsize=8, loc="upper right")

h, lab = ax[0, 0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Tendency core +: 1.5° GPU run vs 5.625° CPU run (2017, each on its own grid)", x=0.01, ha="left",
             fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
fig.savefig(a.out, dpi=150, bbox_inches="tight")
print("saved", a.out)
