"""Step 13 figures: scores by lead for each arm (2 x 2 panels), from scripts/score_step13.py output.

    normalized RMSE   ensemble-mean RMSE / the variable's std (2017-2018 truth), mean over the 5 vars
    t850 RMSE (K)     ClimODE's latitude-weighted RMSE
    z500 CRPS         ensemble CRPS (properscoring, unweighted grid mean) in m^2/s^2
    z500 spread       sqrt of the mean member variance (latitude weighted), m^2/s^2

Persistence is computed here on the same ICs; ClimODE's paper RMSE (Table 7, the same 360 test ICs) is
drawn where it exists (24, 36, 72, 144 h). ClimODE has no ensemble CRPS or spread in the paper.

    python scripts/plot_step13.py --era5 /path/era5_data --scores runs/step13/score_a.json \
        runs/step13/score_guide.json --out docs/step13
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
from scripts.run_step11_climode import VARS, climode_minmax, climode_truth  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402

LEADS_H = (24, 48, 72, 96, 120, 144)
# reference categorical order (dataviz skill palette, light mode), fixed per figure
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FIGURES = {  # (score key, label, line style, categorical slot); calibration: slot = source of the mean
    "fig_stage2": ("Stage 2: sequence FM ensembles and controls", [
        ("seqfm_dct_raw", "FM (DCT sequence)", "-", 0),
        ("seqfm_dct_pinn", "FM + frozen PINN (H2)", "--", 1),
        ("seqfm_dct_inloop", "FM + PINN inside the ODE", "-.", 2),
        ("seqfm_dct_guided", "FM + physics guidance (H3)", "-", 3),
        ("seqfm_dct_smoothed", "FM + smoothing guidance (H3 control)", "--", 4),
        ("seqregress_raw", "Regression + statistical errors (H1)", "-", 5),
    ]),
    "fig_calibration": ("Stage 3: calibration guide on FM + PINN members (color = mean, dashed = learned guide)", [
        ("seqfm_dct_pinn_raw", "FM + PINN members", "-", 0),
        ("seqfm_dct_pinn_mbm_band", "MBM mean + band spread", "-", 1),
        ("seqfm_dct_pinn_op_mono", "Operator mean + band spread", "-", 2),
        ("seqfm_dct_pinn_guide", "Operator mean + learned band guide", "--", 2),
        ("seqfm_dct_pinn_core_band", "Step 12 core mean + band spread", "-", 3),
        ("seqfm_dct_pinn_core_guide", "Core mean + learned band guide", "--", 3),
        ("seqfm_dct_pinn_reg_band", "Sequence regression mean + band spread", "-", 5),
        ("seqfm_dct_pinn_reg_guide", "Regression mean + learned band guide", "--", 5),
        ("seqfm_dct_pinn_regm_band", "a + b·regression mean + c·x(t) + band spread", "-", 7),
        ("seqfm_dct_pinn_regm_guide", "a + b·regression mean + c·x(t) + learned guide", "--", 7),
    ]),
}


def persistence(era5, years, lat, mx, mn):
    """RMSE (L, 5), CRPS (L, 5) in min-max units, on ClimODE's ICs 2 + 8 e of each year."""
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None]
    rmse, crps = [], []
    for y in years:
        t = climode_truth(era5, y)
        ics = np.arange(2, len(t) - max(LEADS_H) // 6, 8)[:180]
        for h in LEADS_H:
            d = t[ics + h // 6] - t[ics]
            rmse.append(np.sqrt((d ** 2 * w).mean((-1, -2))))
            crps.append((np.abs(d) / (mx - mn)[:, None, None]).mean((-1, -2)))
    L = len(LEADS_H)
    r = np.stack(rmse).reshape(len(years), L, -1, 5).mean((0, 2))
    c = np.stack(crps).reshape(len(years), L, -1, 5).mean((0, 2))
    return r, c


PAIRED = [  # (score key, label, line style, categorical slot)
    ("seqfm_dct_raw", "FM (DCT sequence)", "-", 0),
    ("seqregress_raw", "Regression + statistical errors", "-", 6),
    ("seqfm_dct_pinn_core_guide", "Core mean + FM deviations + band guide", "-", 3),
    ("seqfm_dct_pinn_reg_band", "Regression mean + FM deviations + band Γ", "-", 5),
    ("seqregress_pinn_reg_band", "Regression mean + its own deviations + band Γ", "--", 5),
    ("seqfm_dct_pinn_regm_band", "a + b·regression mean + c·x(t) + FM deviations + band Γ", "-", 7),
]


def plot_paired(paths, out, mx, mn):
    """z500 / t850 on ClimODE's 16 reproduced ICs (the first 8 of each test year), the same ICs for every line.
    Several compare_step13_climode.py files are merged (same ICs, same ClimODE and persistence)."""
    d = json.load(open(paths[0]))
    for q in paths[1:]:
        e = json.load(open(q))
        assert e["leads"] == d["leads"] and e["n_ic"]["climode"] == d["n_ic"]["climode"]
        d["scores"].update(e["scores"])
    S, leads = d["scores"], d["leads"]
    days = np.array(leads) / 24
    iz = VARS.index("z")
    get = lambda name, met, v: np.array([S[name][str(h)][met][v]["mean"] for h in leads])  # noqa: E731
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.6), layout="constrained")
    panels = [(ax[0, 0], "Geopotential at 500 hPa", "RMSE (m²/s²)", "rmse", "z", 1.0),
              (ax[0, 1], "Temperature at 850 hPa", "RMSE (K)", "rmse", "t", 1.0),
              (ax[1, 0], "Geopotential at 500 hPa", "Gaussian CRPS (m²/s², ClimODE protocol)", "crps_gauss", "z",
               (mx - mn)[iz]),
              (ax[1, 1], "Geopotential at 500 hPa", "Ensemble spread (m²/s²)", "spread", "z", 1.0)]
    for a_, sub, ylab, met, v, scale in panels:
        for k, lab, ls, slot in PAIRED:
            if k in S:
                a_.plot(days, get(k, met, v) * scale, ls, color=SLOTS[slot], lw=1.8, label=lab)
        pers = get("persistence", met, v) * scale if met != "spread" else np.zeros(len(leads))
        a_.plot(days, pers, ":", color=MUTED, lw=2, label="Persistence")
        a_.plot(days, get("climode", met, v) * scale, "-", color=INK, lw=1.2, marker="s", ms=5,
                label="ClimODE (public checkpoint, reproduced)")
        a_.set_title(sub, fontsize=13)
        a_.set_xlabel("Forecast lead (days)")
        a_.set_ylabel(ylab)
        a_.set_xlim(1, 6)
        a_.grid(True, color=GRID, lw=0.8)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=9, frameon=False)
    n = d["n_ic"].get("climode") if isinstance(d["n_ic"], dict) else d["n_ic"]
    fig.suptitle(f"Step 13 vs ClimODE on the same {n} ICs (ClimODE's first 8 of 2017 and of 2018, January)\n"
                 "ClimODE: authors' checkpoint and code; ours: CPU budget, 8 members", fontsize=13)
    p = os.path.join(out, "fig_climode_paired.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}: " + ", ".join(k for k, *_ in PAIRED if k in S))
    return p


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--era5", required=True)
    p.add_argument("--scores", nargs="+", required=True, help="score_step13.py json files")
    p.add_argument("--years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--out", default="docs/step13")
    p.add_argument("--paired", nargs="*", default=None, help="climode_paired*.json (scripts/compare_step13_climode.py)")
    p.add_argument("--subtitle", default="Real ERA5 5.625°, test 2017-2018 (ClimODE's 360 ICs), 8 members; "
                                         "CPU budget: FM 3000 it, PINN 1000 it")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    arms = {}
    for f in a.scores:
        arms.update(json.load(open(f))["arms"])
    lat = load_static(a.era5)[0]
    mx, mn = climode_minmax(a.era5)
    std = np.concatenate([climode_truth(a.era5, y) for y in a.years]).std((0, 2, 3))
    p_rmse, p_crps = persistence(a.era5, a.years, lat, mx, mn)
    paper = json.load(open(os.path.join(os.path.dirname(__file__), "..", "docs", "step11",
                                        "climode_paper_tables.json")))["rmse"]["ClimODE"]
    ph = [h for h in (24, 36, 72, 144) if str(h) in paper]
    paper_rmse = np.array([[paper[str(h)][v] for v in VARS] for h in ph])
    iz, it = VARS.index("z"), VARS.index("t")
    days, pdays = np.array(LEADS_H) / 24, np.array(ph) / 24
    written = []
    for name, (title, lines) in FIGURES.items():
        present = [q for q in lines if q[0] in arms]
        if not present:
            print(f"[plot] {name}: no arms in the score files, skipped")
            continue
        fig, ax = plt.subplots(2, 3, figsize=(18, 9.6), layout="constrained")
        zs, ts = (mx - mn)[iz], (mx - mn)[it]
        panels = [
            (ax[0, 0], "Mean of 5 variables", "Normalized RMSE (RMSE / std)",
             lambda r: (np.asarray(r["rmse"]) / std).mean(1), (p_rmse / std).mean(1), (paper_rmse / std).mean(1)),
            (ax[0, 1], "Geopotential at 500 hPa", "RMSE (m²/s²)",
             lambda r: np.asarray(r["rmse"])[:, iz], p_rmse[:, iz], paper_rmse[:, iz]),
            (ax[0, 2], "Temperature at 850 hPa", "RMSE (K)",
             lambda r: np.asarray(r["rmse"])[:, it], p_rmse[:, it], paper_rmse[:, it]),
            (ax[1, 0], "Geopotential at 500 hPa", "Ensemble CRPS (m²/s²)",
             lambda r: np.asarray(r["crps"])[:, iz] * zs, p_crps[:, iz] * zs, None),
            (ax[1, 1], "Geopotential at 500 hPa", "Ensemble spread (m²/s²)",
             lambda r: np.asarray(r["spread"])[:, iz] if "spread" in r else None, np.zeros(len(LEADS_H)), None),
            (ax[1, 2], "Temperature at 850 hPa", "Ensemble CRPS (K)",
             lambda r: np.asarray(r["crps"])[:, it] * ts, p_crps[:, it] * ts, None),
        ]
        for a_, sub, ylab, get, pers, cl in panels:
            for k, lab, ls, slot in present:
                yv = get(arms[k])
                if yv is not None:
                    a_.plot(days, yv, ls, color=SLOTS[slot], lw=1.8, label=lab)
            a_.plot(days, pers, ":", color=MUTED, lw=2, label="Persistence")
            if cl is not None:
                a_.plot(pdays, cl, "-", color=INK, lw=1, marker="s", ms=5, label="ClimODE (paper, same ICs)")
            a_.set_title(sub, fontsize=13)
            a_.set_xlabel("Forecast lead (days)")
            a_.set_ylabel(ylab)
            a_.set_xlim(1, 6)
            a_.grid(True, color=GRID, lw=0.8)
            for s_ in ("top", "right"):
                a_.spines[s_].set_visible(False)
        fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=9,
                   frameon=False)
        fig.suptitle(f"Step 13: {title}\n{a.subtitle}", fontsize=13)
        path = os.path.join(a.out, name + ".png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
        print(f"[plot] {path}: " + ", ".join(q[0] for q in present))
    paired = [q for q in (a.paired or []) if os.path.exists(q)]
    if paired:
        written.append(plot_paired(paired, a.out, mx, mn))
    return written


if __name__ == "__main__":
    main()
