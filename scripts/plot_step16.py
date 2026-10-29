"""Step 16 figure: 2 x 2 lead-time lines from the json files of scripts/run_step16_dynpinn.py.

    z500 RMSE (m^2/s^2)     t850 RMSE (K)         persistence dotted (the same ICs)
    kinetic energy ratio    strong residual ratio rollout / truth (momentum u, v and T)

    python scripts/plot_step16.py --runs "data only=runs/step16/dyn_lp0.json" "λ_p 0.05=..." --out docs/step16/fig_step16.png
"""

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

for f in ("WenQuanYi Zen Hei", "Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"):
    if any(f == x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="label=json")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    runs = [(s.split("=", 1)[0], json.load(open(s.split("=", 1)[1]))) for s in a.runs]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    leads = np.array(runs[0][1]["val"]["leads_h"])
    for i, (label, r) in enumerate(runs):
        v, c = r["val"], COLORS[i % len(COLORS)]
        ax[0, 0].plot(leads, v["rmse"]["z500"], color=c, lw=2, label=label)
        ax[0, 1].plot(leads, v["rmse"]["t850"], color=c, lw=2, label=label)
        ax[1, 0].plot(leads, v["ke_ratio"], color=c, lw=2, label=label)
        hs = sorted(int(h) for h in v["strong_ratio"])
        for q, ls in (("u", "-"), ("T", "--")):
            ax[1, 1].plot(hs, [v["strong_ratio"][str(h)][q] for h in hs], color=c, lw=2, ls=ls, marker="o", ms=4,
                          label=f"{label} ({q})")
    v0 = runs[0][1]["val"]
    ax[0, 0].plot(leads, v0["persistence"]["z500"], "k:", lw=1.5, label="persistence")
    ax[0, 1].plot(leads, v0["persistence"]["t850"], "k:", lw=1.5, label="persistence")
    ax[1, 0].axhline(1.0, color="k", ls=":", lw=1), ax[1, 0].axhline(0.95, color="grey", ls=":", lw=1)
    ax[1, 1].axhline(1.0, color="k", ls=":", lw=1)
    ax[0, 0].set_title("z500 RMSE (m²/s²), 2016 16 IC"), ax[0, 1].set_title("t850 RMSE (K)")
    ax[1, 0].set_title("운동에너지 비 rollout / 진실 (gate 0.95)")
    ax[1, 1].set_title("강형식 잔차 비 rollout / 진실 (u 실선, T 점선)")
    for q in ax.flat:
        q.set_xlabel("lead (h)"), q.grid(alpha=0.3), q.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(a.out)


if __name__ == "__main__":
    main()
