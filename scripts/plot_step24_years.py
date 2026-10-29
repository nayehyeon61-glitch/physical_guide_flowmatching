"""Step 24 figure: data-efficiency curve of the step 13 reported model (regm_band) on ClimODE's 360 test ICs.

Runs with 1, 2 and 4 training years on CPU (runs/step24/y1, y2, runs/step13) plus the owner's 37-year GPU run, whose
score json is not in this repository: its numbers are the ones in docs/step15_training_system.md (z500 RMSE
170 / 423 / 741 at 24 / 72 / 144 h, t850 144 h 3.21 K, z500 144 h CRPS 0.0310) and are drawn as markers only.
2x2 panels: (a) z500 RMSE by lead, (b) t850 RMSE by lead, (c) z500 CRPS by lead, (d) z500 RMSE against training
years (log axis) at 24 / 72 / 144 h. Persistence dotted.

    python scripts/plot_step24_years.py --era5 /home/claude/era5/era5_data \
        --runs 1:runs/step24/y1/score_best.json 2:runs/step24/y2/score_best.json 4:runs/step13/score_regm.json \
        --out docs/step24/fig_data_efficiency.png
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
from scripts.plot_step13 import GRID, INK, LEADS_H, MUTED, SLOTS, persistence  # noqa: E402
from scripts.run_step11_climode import VARS, climode_minmax  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402

ARM = "seqfm_dct_pinn_regm_band"
BROAD = {"years": 37, "z_rmse": {24: 170, 72: 423, 144: 741}, "t_rmse": {144: 3.21}, "z_crps": {144: 0.0310}}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--era5", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="<training years>:<score json with arm regm_band>")
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--out", default="docs/step24/fig_data_efficiency.png")
    p.add_argument("--summary", default=None, help="optional json with the plotted numbers")
    a = p.parse_args(argv)
    runs = []
    for r in a.runs:
        n, path = r.split(":", 1)
        s = json.load(open(path))["arms"][ARM]
        runs.append((int(n), np.array(s["rmse"]), np.array(s["crps"])))
    runs.sort(key=lambda t: t[0])
    lat = load_static(a.era5)[0]
    mx, mn = climode_minmax(a.era5)
    p_rmse, p_crps = persistence(a.era5, a.test_years, lat, mx, mn)
    iz, it = VARS.index("z"), VARS.index("t")
    zs = (mx - mn)[iz]
    days = np.array(LEADS_H) / 24
    cols = [SLOTS[k] for k in (0, 1, 2)]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9.6), layout="constrained")
    panels = [(ax[0, 0], "Geopotential at 500 hPa", "RMSE (m²/s²)", 1, iz, 1.0, "z_rmse"),
              (ax[0, 1], "Temperature at 850 hPa", "RMSE (K)", 1, it, 1.0, "t_rmse"),
              (ax[1, 0], "Geopotential at 500 hPa", "CRPS (m²/s², ClimODE protocol)", 2, iz, zs, "z_crps")]
    for a_, title, ylab, m, v, scale, bkey in panels:
        for (n, r, c), col in zip(runs, cols):
            y = (r if m == 1 else c)[:, v] * scale
            a_.plot(days, y, "-", color=col, lw=1.8, label=f"{n} training year{'s' if n > 1 else ''} (CPU)")
        bx = [h / 24 for h in BROAD[bkey]]
        a_.plot(bx, [BROAD[bkey][h] * scale for h in BROAD[bkey]], "s", color=INK, ms=7,
                label="37 training years (owner's GPU run, reported leads only)")
        pers = (p_rmse if m == 1 else p_crps)[:, v] * scale
        a_.plot(days, pers, ":", color=MUTED, lw=1.8, label="Persistence")
        a_.set_title(title, fontsize=13), a_.set_ylabel(ylab), a_.set_xlabel("Forecast lead (days)")
        a_.set_xlim(days[0] - 0.1, days[-1] + 0.1)
    a_ = ax[1, 1]
    for k, h in enumerate((24, 72, 144)):
        j = LEADS_H.index(h)
        xs = [n for n, _, _ in runs] + [BROAD["years"]]
        ys = [r[j, iz] for _, r, _ in runs] + [BROAD["z_rmse"][h]]
        a_.plot(xs, ys, "-", color=SLOTS[k + 3], lw=1.8, marker="o", ms=6, label=f"{h} h")
        for x, y in zip(xs, ys):
            a_.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=MUTED)
    a_.set_xscale("log"), a_.set_xticks([1, 2, 4, 10, 37]), a_.set_xticklabels(["1", "2", "4", "10", "37"])
    a_.set_title("z500 RMSE against training years", fontsize=13)
    a_.set_xlabel("Training years (log scale; 1-4 CPU, 37 GPU with more iterations)"), a_.set_ylabel("RMSE (m²/s²)")
    a_.legend(fontsize=9, frameon=False, title="lead", title_fontsize=9, loc="center right")
    for a_ in ax.flat:
        a_.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Step 24: data efficiency of regm_band (regression mean + FM deviations + band Γ)\n"
                 "ClimODE's 360 test ICs of 2017-2018; training years 2015 / 2014-15 / 2012-15 (CPU) and 1979-2015 (GPU),"
                 " calibration 2016", fontsize=13)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"[plot] {a.out}")
    for n, r, c in runs:
        print(f"[{n} y] z500 RMSE " + " ".join(f"{x:.0f}" for x in r[:, iz]) + " | t850 " + " ".join(f"{x:.2f}" for x in r[:, it])
              + " | z500 CRPS " + " ".join(f"{x:.4f}" for x in c[:, iz]))
    if a.summary:
        out = {"leads": list(LEADS_H), "runs": {str(n): {"rmse_z500": r[:, iz].tolist(), "rmse_t850": r[:, it].tolist(),
                                                          "crps_z500": c[:, iz].tolist()} for n, r, c in runs},
               "broad_37y": BROAD, "persistence": {"rmse_z500": p_rmse[:, iz].tolist(), "rmse_t850": p_rmse[:, it].tolist()}}
        with open(a.summary, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
