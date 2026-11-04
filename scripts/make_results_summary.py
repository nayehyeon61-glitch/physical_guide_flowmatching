"""Collect the scores of the finished GPU runs into docs/results_2026_11 (tidy CSVs, raw JSONs, figures).

Reads only the small score JSONs under runs/; forecasts, checkpoints and ERA5 stay out of git.

    python scripts/make_results_summary.py [--runs runs] [--out docs/results_2026_11]
"""

import argparse
import csv
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VARS = ["z", "t", "t2m", "u10", "v10"]
NAME = {"z": "z500", "t": "t850", "t2m": "t2m", "u10": "u10", "v10": "v10"}
UNIT = {"z": "m²/s²", "t": "K", "t2m": "K", "u10": "m/s", "v10": "m/s"}
BLEND = "seqfm_dct_pinn_blend_band"
# run id, score file, training years, test years, ICs, source run
RUNS = [
    ("step13_broad", "step13_broad/score_best.json", "1979-2015", "2017-2018", 360),
    ("step15_climode", "step15_climode/score_best.json", "2006-2015", "2017-2018", 360),
    ("step15_broad_long", "step15_broad_long/score_best.json", "1979-2015", "2017-2018", 360),
    ("blend_4y", "blend_4y/score_blend.json", "2012-2015", "2017-2018", 360),
    ("blend_climode", "blend_climode/score_blend.json", "2006-2015", "2017-2018", 360),
    ("blend_ngcm", "blend_ngcm/score_blend.json", "1979-2015", "2017-2018", 360),
    ("blend_ngcm_2020", "blend_ngcm/ngcm2020/score_2020.json", "1979-2015", "2020", 180),
]
PAIRED = [
    ("blend_4y", "blend_4y/climode_paired_34.json", "2012-2015"),
    ("blend_climode", "blend_climode/climode_paired_34.json", "2006-2015"),
    ("blend_ngcm", "blend_ngcm/climode_paired_34.json", "1979-2015"),
    ("blend_ngcm_2020", "blend_ngcm/ngcm2020/climode2020_paired.json", "1979-2015"),
]
SERIES = {  # color follows the entity in every figure
    "ours": ("#2a78d6", "o"), "ClimODE": ("#eb6834", "s"), "10y": ("#1baf7a", "^"), "4y": ("#eda100", "D"),
}
INK, INK2, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": AXIS,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": .6, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def line(ax, x, y, key, label=None, ls="-"):
    c, m = SERIES[key]
    ax.plot(x, y, ls, color=c, marker=m, ms=4.5, lw=1.6, mec=SURF, mew=.8, label=label or key)


def by_lead(paired, arm, metric, var):
    return [paired["scores"][arm][str(h)][metric][var]["mean"] for h in paired["leads"]]


def fig_rmse_2020(p, days, out):
    fig, axs = plt.subplots(1, 5, figsize=(13, 2.9), constrained_layout=True)
    for ax, v in zip(axs, VARS):
        o, c = by_lead(p, BLEND, "rmse", v), by_lead(p, "climode", "rmse", v)
        line(ax, days, c, "ClimODE")
        line(ax, days, o, "ours", "Ours (blend_band)")
        ax.set_title(NAME[v]); ax.set_ylabel(f"RMSE ({UNIT[v]})"); ax.set_xlabel("lead (days)"); ax.set_ylim(bottom=0); ax.set_xticks(days)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper right", ncol=2, frameon=False)
    fig.suptitle(f"2020 hold-out RMSE, same {p['n_ic']['climode']} ICs: ours is lower from day 2 in every variable", x=.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(out, "fig1_rmse_2020.png"), dpi=200)
    plt.close(fig)


def fig_change(p, days, out):
    a = np.array([[100 * m / c for m, c in zip(by_lead(p, BLEND + "-climode", "rmse", v), by_lead(p, "climode", "rmse", v))] for v in VARS])
    fig, ax = plt.subplots(figsize=(6.4, 3.1), constrained_layout=True)
    ax.grid(False)
    cm = matplotlib.colors.LinearSegmentedColormap.from_list("d", ["#2a78d6", "#f0efec", "#e34948"])
    im = ax.imshow(a, cmap=cm, vmin=-65, vmax=65, aspect="auto")
    ax.set_xticks(range(6), [f"{d} d" for d in days]); ax.set_yticks(range(5), [NAME[v] for v in VARS])
    for s in ax.spines.values():
        s.set_visible(False)
    for (i, j), x in np.ndenumerate(a):
        ax.text(j, i, f"{x:+.0f}%", ha="center", va="center", color="white" if abs(x) > 40 else INK, fontsize=8.5)
    fig.colorbar(im, ax=ax, label="blue = ours better, red = ClimODE better", shrink=.9)
    ax.set_title("RMSE change of ours vs ClimODE, 2020 (paired, 179 ICs)")
    fig.savefig(os.path.join(out, "fig2_rmse_change_2020.png"), dpi=200)
    plt.close(fig)
    return a


def fig_spread(p, days, out):
    fig, axs = plt.subplots(1, 5, figsize=(13, 2.7), sharey=True, constrained_layout=True)
    for ax, v in zip(axs, VARS):
        for arm, key, lab in (("climode", "ClimODE", "ClimODE"), (BLEND, "ours", "Ours (blend_band)")):
            s, r = by_lead(p, arm, "spread", v), by_lead(p, arm, "rmse", v)
            line(ax, days, np.array(s) / np.array(r), key, lab)
        ax.axhline(1, color=INK2, lw=.9, ls="--")
        ax.set_title(NAME[v]); ax.set_xlabel("lead (days)"); ax.set_xticks(days); ax.set_ylim(0, 1.4)
    axs[0].set_ylabel("spread / RMSE (dashed: 1)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper right", ncol=2, frameon=False)
    fig.suptitle("2020 calibration: ours stays near 1, ClimODE's uncertainty swings with a 48 h period", x=.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(out, "fig3_spread_skill_2020.png"), dpi=200)
    plt.close(fig)


def fig_budget(pairs, days, out):
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3), constrained_layout=True)
    for ax, v in zip(axs, ("z", "t")):
        line(ax, days, by_lead(pairs["blend_4y"], "climode", "rmse", v), "ClimODE", "ClimODE (2006-2016)")
        line(ax, days, by_lead(pairs["blend_4y"], BLEND, "rmse", v), "4y", "Ours, 4 y (2012-2015)")
        line(ax, days, by_lead(pairs["blend_climode"], BLEND, "rmse", v), "10y", "Ours, 10 y (2006-2015)")
        line(ax, days, by_lead(pairs["blend_ngcm"], BLEND, "rmse", v), "ours", "Ours, 37 y (1979-2015)")
        ax.set_title(NAME[v]); ax.set_ylabel(f"RMSE ({UNIT[v]})"); ax.set_xlabel("lead (days)"); ax.set_ylim(bottom=0); ax.set_xticks(days)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("blend_band by training years, 2017-18, same 34 ICs (iterations also differ)", x=.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(out, "fig4_training_years_2017_18.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="docs/results_2026_11")
    a = ap.parse_args()
    data, figs, raw = (os.path.join(a.out, d) for d in ("scores", "figures", "scores/raw"))
    for d in (figs, raw):
        os.makedirs(d, exist_ok=True)

    # 1. every arm of every run, ensemble scores on all ICs
    rows, summary = [], []
    for run, f, tr, te, n in RUNS:
        src = os.path.join(a.runs, f)
        s = json.load(open(src))
        shutil.copy(src, os.path.join(raw, f"{run}_score.json"))
        for arm, d in s["arms"].items():
            leads = d["leads"]
            for met in ("rmse", "crps", "spread", "ssr"):
                rows += [[run, tr, te, n, arm, h, v, met, d[met][j][i]] for j, h in enumerate(leads) for i, v in enumerate(VARS)]
            z = np.array(d["rmse"])[:, 0]; t = np.array(d["rmse"])[:, 1]
            summary.append([run, tr, te, n, arm, round(float(np.mean(d["crps"])), 5)] + [round(float(z[k]), 1) for k in (0, 2, 5)] + [round(float(t[k]), 3) for k in (0, 2, 5)])
    write_csv(os.path.join(data, "scores_long.csv"), ["run", "train_years", "test_years", "n_ic", "arm", "lead_h", "var", "metric", "value"], rows)
    write_csv(os.path.join(data, "run_summary.csv"), ["run", "train_years", "test_years", "n_ic", "arm", "crps_5var_mean", "z500_rmse_24h", "z500_rmse_72h", "z500_rmse_144h", "t850_rmse_24h", "t850_rmse_72h", "t850_rmse_144h"], summary)

    # 2. paired comparisons with ClimODE (bootstrap intervals on the difference)
    rows, pairs = [], {}
    for run, f, tr in PAIRED:
        src = os.path.join(a.runs, f)
        p = pairs[run] = json.load(open(src))
        shutil.copy(src, os.path.join(raw, f"{run}_climode_paired.json"))
        for arm in ("climode", BLEND, BLEND + "-climode"):
            for h in p["leads"]:
                for met, by_v in p["scores"][arm][str(h)].items():
                    for v, r in by_v.items():
                        ci = r["ci95"] or [None, None]
                        rows.append([run, tr, "-".join(map(str, p["years"])), p["n_ic"]["climode"], arm, h, v, met, r["mean"], ci[0], ci[1]])
    write_csv(os.path.join(data, "paired_vs_climode.csv"), ["run", "train_years", "test_years", "n_ic", "arm", "lead_h", "var", "metric", "mean", "ci95_low", "ci95_high"], rows)

    # 3. figures
    p = pairs["blend_ngcm_2020"]
    days = [h // 24 for h in p["leads"]]
    fig_rmse_2020(p, days, figs)
    change = fig_change(p, days, figs)
    fig_spread(p, days, figs)
    fig_budget(pairs, days, figs)
    print("RMSE change % (rows z t t2m u10 v10)\n", np.round(change, 1))
    for run in ("blend_4y", "blend_climode", "blend_ngcm"):
        q = pairs[run]
        print(run, "n", q["n_ic"]["climode"], "z", np.round(by_lead(q, BLEND, "rmse", "z")), "t", np.round(by_lead(q, BLEND, "rmse", "t"), 2),
              "diff z", [round(q["scores"][BLEND + "-climode"][str(h)]["rmse"]["z"]["mean"]) for h in (24, 72, 144)],
              "ci24", q["scores"][BLEND + "-climode"]["24"]["rmse"]["t"]["ci95"], "t24", round(q["scores"][BLEND + "-climode"]["24"]["rmse"]["t"]["mean"], 3))
    q = pairs["blend_4y"]; print("climode34 z", np.round(by_lead(q, "climode", "rmse", "z")), "t", np.round(by_lead(q, "climode", "rmse", "t"), 2))
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
