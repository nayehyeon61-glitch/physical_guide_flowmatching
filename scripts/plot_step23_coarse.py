"""Step 23 figure: NeuralGCM from the full ERA5 IC vs from our 5.625 deg / 12-level IC, with our blend_band, on the
same 9 ICs of 2020 (2x2 lead-time panels, persistence dotted).

(a) z500 RMSE, (b) t850 RMSE, (c) z500 Gaussian CRPS (ClimODE protocol, deterministic NeuralGCM = MAE-like),
(d) the share of our gap to full-IC NeuralGCM that the IC information explains: (coarse - full) / (ours - full).

    python scripts/plot_step23_coarse.py --era5 /home/claude/era5/era5_data --coarse-json runs/step23/coarse_ic.json \
        --paired runs/step23/neuralgcm_coarse_paired.json --full-paired runs/step23/neuralgcm_sub9_paired.json \
        --out docs/step23/fig_neuralgcm_coarse_ic.png
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.plot_step13 import GRID, INK, MUTED, SLOTS  # noqa: E402
from scripts.run_step11_climode import VARS, climode_minmax  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--era5", required=True)
    p.add_argument("--coarse-json", default="runs/step23/coarse_ic.json")
    p.add_argument("--paired", default="runs/step23/neuralgcm_coarse_paired.json")
    p.add_argument("--full-paired", default="runs/step23/neuralgcm_sub9_paired.json",
                   help="compare_step13_climode.py output for the full-IC run restricted to the same ICs")
    p.add_argument("--arm", default="seqfm_dct_pinn_blend_band")
    p.add_argument("--out", default="docs/step23/fig_neuralgcm_coarse_ic.png")
    a = p.parse_args(argv)
    C = json.load(open(a.coarse_json))
    P = json.load(open(a.paired))
    F = json.load(open(a.full_paired))
    leads = C["leads"]
    days = np.array(leads) / 24
    S = P["scores"]
    get = lambda d, name, met, v: np.array([d[name][str(h)][met][v]["mean"] for h in leads])  # noqa: E731
    mx, mn = climode_minmax(a.era5)
    zs = (mx - mn)[VARS.index("z")]
    assert F["ics"] == C["ics"] == P["ics"], "the three inputs must hold the same ICs"

    fig, ax = plt.subplots(2, 2, figsize=(13, 9.6), layout="constrained")
    lines = [("full", "NeuralGCM 2.8° from the full ERA5 IC (0.25°, 37 levels)", INK, "-", "s"),
             ("coarse", "NeuralGCM 2.8° from our IC (5.625°, 12 levels, no cloud water)", INK, "--", "o")]
    for a_, v, ylab, title in [(ax[0, 0], "z", "RMSE (m²/s²)", "Geopotential at 500 hPa"),
                               (ax[0, 1], "t", "RMSE (K)", "Temperature at 850 hPa")]:
        for key, lab, col, ls, mk in lines:
            a_.plot(days, C[key][v], ls, color=col, lw=1.4, marker=mk, ms=5, mfc="white" if key == "coarse" else col,
                    label=lab)
        a_.plot(days, get(S, a.arm, "rmse", v), "-", color=SLOTS[2], lw=1.8, label="blend_band (ours, ensemble mean)")
        a_.plot(days, get(S, "persistence", "rmse", v), ":", color=MUTED, lw=1.8, label="Persistence")
        a_.set_title(title, fontsize=13), a_.set_ylabel(ylab)
    a_ = ax[1, 0]
    a_.plot(days, get(F["scores"], "climode", "crps_gauss", "z") * zs, "-", color=INK, lw=1.4, marker="s", ms=5)
    a_.plot(days, get(S, "climode", "crps_gauss", "z") * zs, "--", color=INK, lw=1.4, marker="o", ms=5, mfc="white")
    a_.plot(days, get(S, a.arm, "crps_gauss", "z") * zs, "-", color=SLOTS[2], lw=1.8)
    a_.plot(days, get(S, "persistence", "crps_gauss", "z") * zs, ":", color=MUTED, lw=1.8)
    a_.set_title("Geopotential at 500 hPa", fontsize=13), a_.set_ylabel("Gaussian CRPS (m²/s², ClimODE protocol)")
    a_ = ax[1, 1]
    shares = {}
    for k, v, lab in [(0, "z", "z500"), (1, "t", "t850")]:
        ours = get(S, a.arm, "rmse", v)
        shares[v] = 100 * (np.array(C["coarse"][v]) - np.array(C["full"][v])) / (ours - np.array(C["full"][v]))
        a_.plot(days, shares[v], "-", color=SLOTS[k], lw=1.8, marker="o", ms=5, label=f"{lab}: IC-information share")
    for v, other in (("z", "t"), ("t", "z")):  # label the upper line above, the lower one below
        for x, y, y2 in zip(days, shares[v], shares[other]):
            a_.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 7 if y >= y2 else -14), ha="center",
                        fontsize=8, color=MUTED)
    a_.axhline(100, color=MUTED, ls=":", lw=1.2)
    a_.set_ylim(0, 110)
    a_.set_title("Share of our RMSE gap to NeuralGCM explained by the IC", fontsize=13)
    a_.set_ylabel("(coarse-IC − full-IC) / (ours − full-IC), %")
    a_.legend(fontsize=9, frameon=False, loc="lower left")
    for a_ in ax.flat:
        a_.set_xlabel("Forecast lead (days)"), a_.grid(color=GRID, lw=0.8), a_.set_xlim(days[0], days[-1])
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(f"Equal-information test: NeuralGCM 2.8° deterministic on {len(C['ics'])} ICs of 2020, "
                 "full vs our IC\nsame checkpoint, same forcings (SST, sea ice); scored on our 5.625° grid", fontsize=13)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"[plot] {a.out}")


if __name__ == "__main__":
    main()
