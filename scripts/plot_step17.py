"""Step 17 figure: one row per setting (coarse / full), 2 x 2 per row from scripts/run_step17_swe.py json:

    Phi RMSE vs lead (grid arms lines, persistence dotted, Raissi markers)   wind RMSE vs lead
    total energy ratio vs lead                                               tangent residual per arm and channel

    python scripts/plot_step17.py --runs runs/step17/coarse.json runs/step17/full.json --out docs/step17/fig_step17.png
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
COLORS = {"data": "#4477AA", "node": "#66CCEE", "pi": "#EE6677", "pi_only": "#AA3377"}
MARK = {"fwd": "x", "fit": "o"}
LABEL = {"data": "자료 (Heun)", "node": "자료 (neural ODE, torchdiffeq)", "pi": "자료 + 물리 잔차", "pi_only": "물리 잔차만",
         "fwd": "Raissi PINN (t=0 자료 + 물리)", "fit": "Raissi PINN (매시간 자료 + 물리)"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="json files (one row each)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    runs = [json.load(open(f)) for f in a.runs]
    fig, axes = plt.subplots(len(runs), 4, figsize=(18, 4.2 * len(runs)), squeeze=False)
    for row, r in enumerate(runs):
        ax = axes[row]
        leads = np.array(r["leads_h"])
        arms = r["arms"]
        for arm, v in arms.items():
            c = COLORS.get(arm, "k")
            ax[0].plot(leads, v["rmse"]["Phi"], color=c, lw=2, label=LABEL.get(arm, arm))
            ax[1].plot(leads, v["rmse"]["wind"], color=c, lw=2, label=LABEL.get(arm, arm))
            ax[2].plot(leads, v["energy_ratio"], color=c, lw=2, label=LABEL.get(arm, arm))
        first = next(iter(arms.values()), None)
        if first is not None:
            ax[0].plot(leads, first["rmse_pers"]["Phi"], "k:", lw=1.5, label="persistence")
            ax[1].plot(leads, first["rmse_pers"]["wind"], "k:", lw=1.5, label="persistence")
        for key, v in r.get("raissi", {}).items():
            variant, T = key.split("_T")
            lh = np.array(v["leads_h"])
            ax[0].plot(lh, v["rmse"]["Phi"], ls="--", marker=MARK[variant], ms=5, lw=1, color="#228833" if variant == "fwd" else "#CCBB44",
                       label=f"{LABEL[variant]} T={T} h")
            ax[1].plot(lh, v["rmse"]["wind"], ls="--", marker=MARK[variant], ms=5, lw=1, color="#228833" if variant == "fwd" else "#CCBB44")
            ax[2].plot(lh, v["energy_ratio"], ls="--", marker=MARK[variant], ms=5, lw=1, color="#228833" if variant == "fwd" else "#CCBB44")
        ch = ("Phi", "u", "v")
        xs = np.arange(len(ch))
        n = max(1, len(arms))
        for i, (arm, v) in enumerate(arms.items()):
            ax[3].bar(xs + (i - n / 2 + 0.5) * 0.8 / n, [v["tangent_residual"][k] for k in ch], 0.8 / n,
                      color=COLORS.get(arm, "k"), label=LABEL.get(arm, arm))
        ax[3].plot(xs, [r["operator_bias"][k] for k in ch], "k_", ms=25, mew=2, label="물리 연산자의 닫힘 결손")
        ax[3].set_xticks(xs), ax[3].set_xticklabels(ch)
        ax[3].axhline(1.0, color="grey", ls=":", lw=1)
        s = r["setting"]
        ax[0].set_title(f"[{s}] Φ RMSE (m²/s²)"), ax[1].set_title(f"[{s}] 바람 RMSE (m/s)")
        ax[2].set_title(f"[{s}] 에너지 비 예측 / 진실"), ax[3].set_title(f"[{s}] 접벡터 잔차 |f − F_true| / b (진실 상태)")
        ax[2].axhline(1.0, color="k", ls=":", lw=1)
        for q in ax[:3]:
            q.set_xlabel("lead (h)"), q.grid(alpha=0.3)
        ax[0].legend(fontsize=7), ax[3].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print(a.out)


if __name__ == "__main__":
    main()
