"""Step 11 report: ClimODE paper models + reproduced ClimODE + PG-CFM on the same ICs.

Reads docs/step11/climode_paper_tables.json (ClimODE paper Tables 6-7), <run>/climode/climode_scores.json +
climode_pred.npz (scripts/run_climode_eval.py) and <run>/results*.json + pgcfm*_pred_<year>.npz
(scripts/run_step11_climode.py). The paired part scores both models on the ICs both were run on
(ClimODE's first n ICs of each test year) with the vectorized ClimODE scores.

    python scripts/report_step11.py --run runs/step11 --out docs/step11
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.run_step11_climode import VARS, climode_minmax, climode_scores, climode_truth  # noqa: E402

LEADS = ["24", "72", "144"]


def boot_ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/step11")
    p.add_argument("--data", default="data/era5_5p625")
    p.add_argument("--out", default="docs/step11")
    p.add_argument("--tags", nargs="+", default=[""])
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    paper = json.load(open(os.path.join(os.path.dirname(__file__), "..", "docs", "step11", "climode_paper_tables.json")))
    cl = json.load(open(os.path.join(a.run, "climode", "climode_scores.json")))
    runs = {t: json.load(open(os.path.join(a.run, f"results{t}.json"))) for t in a.tags}
    rows = {}  # name -> metric -> lead -> var
    for m in paper["rmse"]:
        rows[f"{m} (paper)"] = {"rmse": paper["rmse"][m], "acc": paper["acc"][m]}
    rows[f"ClimODE (reproduced, {2 * cl['n_ic_per_year']} ICs)"] = cl["climode"]
    rows[f"persistence ({2 * cl['n_ic_per_year']} ICs)"] = cl["persistence"]
    for t, r in runs.items():
        for arm in ("raw", "noQ", "mbm_tan", "mbmx_tan", "persistence", "climatology"):
            if arm in r:
                rows[f"PG-CFM{t} {arm} ({r['n_ic']} ICs)" if arm not in ("persistence", "climatology")
                     else f"{arm} ({r['n_ic']} ICs)"] = r[arm]

    # -- paired: same ICs ---------------------------------------------------------------------
    cp = dict(np.load(os.path.join(a.run, "climode", "climode_pred.npz")))  # decompress once
    n = cp["mean"].shape[1]
    mx, mn = climode_minmax(os.path.join(a.data, "era5_data"))
    lat = None
    paired = {}
    for t in a.tags:
        for yi, y in enumerate((2017, 2018)):
            f = os.path.join(a.run, f"pgcfm{t}_pred_{y}.npz")
            if not os.path.exists(f):
                continue
            pg = dict(np.load(f))
            lat = pg["lat"]
            truth = climode_truth(os.path.join(a.data, "era5_data"), y)
            clim = truth.mean(0)
            for j, h in enumerate(int(q) for q in cp["lead_hours"]):
                k = h // 24 - 1
                for e in range(min(n, pg["raw"].shape[0])):
                    obs = truth[2 + 8 * e + h // 6]
                    sc_c = climode_scores(cp["mean"][yi, e, j], obs, cp["std"][yi, e, j], clim, lat, mx, mn)
                    for arm in ("raw", "mbm_tan", "mbmx_tan"):
                        sd = pg["spread"][e, k] if arm == "mbm_tan" else np.full_like(obs, 1e-6)
                        sc_p = climode_scores(pg[arm][e, k], obs, sd, clim, lat, mx, mn)
                        d = paired.setdefault(f"{t}{arm}", {}).setdefault(str(h), {})
                        for met in ("rmse", "acc", "crps_gauss"):
                            d.setdefault(met, []).append(sc_p[met] - sc_c[met])
                    if t != a.tags[0]:  # ClimODE's own scores once, not once per PG-CFM variant
                        continue
                    d = paired.setdefault("climode", {}).setdefault(str(h), {})
                    d.setdefault("rmse", []).append(sc_c["rmse"])
                    d.setdefault("spread", []).append(np.sqrt((cp["std"][yi, e, j] ** 2 *
                                                               (np.cos(np.deg2rad(lat)) /
                                                                np.cos(np.deg2rad(lat)).mean())[:, None]).mean((-1, -2))))
                    d.setdefault("crps_gauss", []).append(sc_c["crps_gauss"])
    summary = {}
    for arm, by_h in paired.items():
        for h, d in by_h.items():
            for met, v in d.items():
                v = np.array(v)
                summary.setdefault(arm, {}).setdefault(h, {})[met] = {
                    var: {"mean": float(v[:, i].mean()), "ci95": boot_ci(v[:, i]) if arm != "climode" else None}
                    for i, var in enumerate(VARS)}
    rep = {"rows": rows, "paired_minus_climode": summary,
           "n_paired_ics": int(len(next(iter(paired.get("climode", {"": {"rmse": []}}).values()))["rmse"]))
           if paired else 0}
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)

    # -- markdown table ------------------------------------------------------------------------
    lines = ["| model | " + " | ".join(f"{v} {h}h" for h in LEADS for v in ("z500", "t850", "t2m")) + " |",
             "|---|" + "---|" * (3 * len(LEADS))]
    for name, r in rows.items():
        cells = []
        for h in LEADS:
            for v in ("z", "t", "t2m"):
                x = r.get("rmse", {}).get(h, {}).get(v)
                cells.append("" if x is None or (isinstance(x, float) and np.isnan(x)) else
                             (f"{x:.0f}" if v == "z" else f"{x:.2f}"))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    with open(os.path.join(a.out, "rmse_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # -- figure ----------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(20, 3.8))
    keep = [n for n in rows if n.split(" (")[0] in ("IFS", "FCN", "ClimaX", "ClimODE", "persistence")
            or n.split(" (")[0].endswith(("raw", "mbmx_tan"))]
    keep = [n for n in keep if not (n.startswith("persistence") and "16 ICs" in n)]
    colors = plt.cm.tab20(np.arange(20))
    for i, v in enumerate(VARS):
        for c, name in enumerate(keep):
            r = rows[name]
            hs = sorted((int(h) for h in r.get("rmse", {}) if v in r["rmse"][h]
                         and not np.isnan(r["rmse"][h][v])))
            hs = [h for h in hs if h % 24 == 0 or "paper" in name]  # 6 h ClimODE output: diurnal zig-zag
            if not hs:
                continue
            ls = "--" if "paper" in name else "-"
            ax[i].plot(hs, [r["rmse"][str(h)][v] for h in hs], ls, marker="o", ms=3, color=colors[c % 20],
                       label=name)
        ax[i].set_title(v + (" (PG-CFM: from u/v1000)" if v in ("u10", "v10") else ""))
        ax[i].set_xlabel("lead (h)")
    ax[0].set_ylabel("lat-weighted RMSE")
    ax[-1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "rmse_vs_lead.png"), dpi=120)
    return rep


if __name__ == "__main__":
    main()
