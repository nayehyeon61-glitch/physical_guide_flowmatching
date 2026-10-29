"""Step 15: several run_step13_best.sh output folders side by side (same test ICs, same arm).

For each run, reads <dir>/score_best.json (the arm's scores), the stage json files (training years,
iterations, best iteration, how it stopped, device) and <dir>/<tag>.pt_metrics.csv (validation curves).
Writes to --out:
    fig_compare.png           2 x 2 by lead: z500 RMSE, t850 RMSE, z500 CRPS, z500 spread; one color per run,
                              persistence dotted (--era5), ClimODE paper RMSE as black markers
    fig_compare_training.png  validation loss (solid) and train loss (thin) of FM and regression per run
    compare.md                table: training setup, z500 / t850 RMSE 24 / 72 / 144 h, 5-variable mean CRPS,
                              z500 SSR, plus the differences to the first run

    python scripts/compare_step15_runs.py --runs climode=runs/step15_climode broad=runs/step13_broad \\
        broad_long=runs/step15_broad_long --era5 data/era5_5p625/era5_data --out docs/step15
"""

import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ARM = "seqfm_dct_pinn_regm_band"
STAGES = (("pinn_project", "PINN"), ("seqfm_dct", "FM"), ("seqregress", "regression"))


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def stage_json(d, tag):
    """<tag>.json (stage 1), <tag>_pinn.json or <tag>_pinn_resumed.json (stage 2 with --resume-ens)."""
    for suffix in ("", "_pinn", "_pinn_resumed"):
        r = load(os.path.join(d, f"{tag}{suffix}.json"))
        if r is not None:
            return r
    return None


def years_text(ys):
    ys = sorted(int(y) for y in ys)
    return f"{ys[0]}-{ys[-1]}" if len(ys) > 1 and ys[-1] - ys[0] == len(ys) - 1 else " ".join(map(str, ys))


def curve(d, tag):
    path = os.path.join(d, f"{tag}.pt_metrics.csv")
    if not os.path.exists(path):
        return [], []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    tr = [(int(r["it"]), float(r["train_loss"])) for r in rows if r["train_loss"]]
    va = [(int(r["it"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
    return tr, va


def collect(label, d, arm):
    sc = load(os.path.join(d, "score_best.json"))
    if sc is None or arm not in sc["arms"]:
        raise SystemExit(f"{label}: {d}/score_best.json has no arm {arm}")
    a = sc["arms"][arm]
    st = {tag: stage_json(d, tag) for tag, _ in STAGES}
    return {"label": label, "dir": d, "vars": sc["vars"], "leads": list(a.get("leads", (24, 48, 72, 96, 120, 144))),
            "rmse": np.asarray(a["rmse"]), "crps": np.asarray(a["crps"]), "ssr": np.asarray(a["ssr"]),
            "spread": np.asarray(a["spread"]), "stages": st}


def setup_cell(r, tag):
    s = r["stages"].get(tag)
    if s is None:
        return "-"
    ar = s.get("args", {})
    best, stop = s.get("best_it"), s.get("stopped")
    txt = f"{ar.get('iters', '-')} it x {ar.get('batch', '-')}"
    if best is not None and best >= 0:
        txt += f", best {best}"
    if stop:
        txt += f", {stop}"
    return txt


def table(runs):
    leads = runs[0]["leads"]
    show = [h for h in (24, 72, 144) if h in leads]
    iz, it = runs[0]["vars"].index("z"), runs[0]["vars"].index("t")
    idx = [leads.index(h) for h in show]
    hs = " / ".join(map(str, show)) + " h"
    lines = ["| run | train years | device | FM | regression | z500 RMSE " + hs + " | t850 RMSE " + hs
             + " | 5-var CRPS | z500 SSR " + hs + " |", "| --- " * 9 + "|"]
    for r in runs:
        s = r["stages"].get("seqfm_dct") or r["stages"].get("seqregress") or {}
        yrs = years_text(s.get("args", {}).get("train_years", [])) if s else "-"
        dev = (s.get("run") or {}).get("device_name", "not recorded")
        lines.append(f"| {r['label']} | {yrs} | {dev} | {setup_cell(r, 'seqfm_dct')} | {setup_cell(r, 'seqregress')} | "
                     + " / ".join(f"{r['rmse'][i, iz]:.0f}" for i in idx) + " | "
                     + " / ".join(f"{r['rmse'][i, it]:.2f}" for i in idx) + f" | {r['crps'].mean():.4f} | "
                     + " / ".join(f"{r['ssr'][i, iz]:.2f}" for i in idx) + " |")
    if len(runs) > 1:
        b = runs[0]
        lines += ["", f"Differences to {b['label']} (negative = better):", "",
                  "| run | z500 RMSE " + hs + " | t850 RMSE " + hs + " | 5-var CRPS |", "| --- " * 4 + "|"]
        for r in runs[1:]:
            lines.append(f"| {r['label']} | " + " / ".join(f"{r['rmse'][i, iz] - b['rmse'][i, iz]:+.0f}" for i in idx)
                         + " | " + " / ".join(f"{r['rmse'][i, it] - b['rmse'][i, it]:+.2f}" for i in idx)
                         + f" | {100 * (r['crps'].mean() / b['crps'].mean() - 1):+.2f} % |")
    return "\n".join(lines) + "\n"


def style(ax, title, ylab):
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylab)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def fig_scores(runs, pers, paper, out):
    leads = np.asarray(runs[0]["leads"])
    iz, it = runs[0]["vars"].index("z"), runs[0]["vars"].index("t")
    panels = [("rmse", iz, "z500 RMSE", "m²/s²"), ("rmse", it, "t850 RMSE", "K"),
              ("crps", iz, "z500 CRPS", "normalized"), ("spread", iz, "z500 spread", "m²/s²")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    for ax, (m, v, title, unit) in zip(axes.flat, panels):
        for k, r in enumerate(runs):
            ax.plot(leads, r[m][:, v], color=SLOTS[k % len(SLOTS)], lw=2, marker="o", ms=5, label=r["label"])
        if pers is not None and m in ("rmse", "crps"):
            ax.plot(leads, pers[m][:, v], color=MUTED, lw=1.5, ls=":", label="persistence")
        if paper and m == "rmse":
            name = runs[0]["vars"][v]
            hp = [h for h in leads if str(h) in paper and name in paper[str(h)]]
            if hp:
                ax.plot(hp, [paper[str(h)][name] for h in hp], ls="none", marker="s", ms=6, color=INK,
                        label="ClimODE (paper)")
        style(ax, title, unit)
        ax.set_xticks(leads, [f"{h} h" for h in leads])
    axes[0, 0].legend(fontsize=9, frameon=False)
    fig.suptitle(f"{ARM} on the same test ICs, by run", fontsize=13)
    path = os.path.join(out, "fig_compare.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_training(runs, out):
    tags = [("seqfm_dct", "FM"), ("seqregress", "regression")]
    if not any(curve(r["dir"], t)[0] for r in runs for t, _ in tags):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for ax, (tag, name) in zip(axes, tags):
        for k, r in enumerate(runs):
            tr, va = curve(r["dir"], tag)
            c = SLOTS[k % len(SLOTS)]
            if tr:
                ax.plot(*zip(*tr), color=c, lw=0.8, alpha=0.5)
            if va:
                ax.plot(*zip(*va), color=c, lw=2, label=f"{r['label']} (validation)")
            best = (r["stages"].get(tag) or {}).get("best_it")
            if best is not None and best >= 0:
                ax.axvline(best, color=c, lw=1, ls=":")
        style(ax, f"{name}: train (thin) and validation (thick) loss", "loss")
        ax.set_xlabel("iteration")
        ax.legend(fontsize=8, frameon=False)
    path = os.path.join(out, "fig_compare_training.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    global ARM
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="label=folder, the first is the reference")
    p.add_argument("--arm", default=ARM)
    p.add_argument("--era5", default=None, help="WB1-style era5_data folder, for the persistence lines")
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--out", default="docs/step15")
    a = p.parse_args(argv)
    ARM = a.arm
    os.makedirs(a.out, exist_ok=True)
    runs = [collect(*q.split("=", 1), a.arm) for q in a.runs]
    pers = None
    if a.era5:
        from scripts.plot_step13 import persistence
        from scripts.run_step11_climode import climode_minmax
        from scripts.run_step12_core import load_static
        lat = load_static(a.era5)[0]
        mx, mn = climode_minmax(a.era5)
        r_, c_ = persistence(a.era5, a.test_years, lat, mx, mn)
        pers = {"rmse": r_, "crps": c_}
    paper = load(os.path.join(os.path.dirname(__file__), "..", "docs", "step11", "climode_paper_tables.json"))
    paper = paper["rmse"]["ClimODE"] if paper else None
    md = table(runs)
    with open(os.path.join(a.out, "compare.md"), "w") as f:
        f.write(f"# Run comparison ({a.arm})\n\n" + md)
    print(md)
    print(fig_scores(runs, pers, paper, a.out))
    t = fig_training(runs, a.out)
    if t:
        print(t)


if __name__ == "__main__":
    main()
