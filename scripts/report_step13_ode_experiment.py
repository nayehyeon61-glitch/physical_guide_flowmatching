"""Export an inspectable scorecard/figure from a completed Step13 ODE run."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARMS = ("raw", "pinn", "ode", "ode_no_transport", "persistence")
LABELS = {"raw": "FM", "pinn": "FM + PINN", "ode": "FM + transport ODE",
          "ode_no_transport": "FM + nudging only", "persistence": "Persistence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--tag", default="comparison")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, target = Path(args.run), Path(args.out)
    report = json.loads((source / f"{args.tag}.json").read_text())
    with (source / f"{args.tag}_scores.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["arm"], int(row["lead_hours"]), row["channel"]): row for row in rows}
    leads = report["daily_leads_hours"]
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / f"{args.tag}.json", target / "summary.json")
    shutil.copy2(source / f"{args.tag}_scores.csv", target / "scores.csv")
    days = [lead / 24 for lead in leads]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
    colors = ("#565656", "#d47a17", "#1976ba", "#24854b", "#ad4258")
    styles = ("-", "--", "-", "--", ":")
    for arm, color, style in zip(ARMS, colors, styles):
        if arm not in report["normalized_rmse_all_channels_by_6h_lead"]:
            continue
        all_rmse = report["normalized_rmse_all_channels_by_6h_lead"][arm]
        axes[0, 0].plot(days, [all_rmse[lead // 6 - 1] for lead in leads], style, color=color, label=LABELS[arm])
        for ax, channel, metric, factor in (
            (axes[0, 1], "t850", "rmse", 1),
            (axes[1, 0], "q850", "crps", 1000),
            (axes[1, 1], "q850", "spread", 1000),
        ):
            ax.plot(days, [float(lookup[arm, lead, channel][metric]) * factor for lead in leads],
                    style, color=color, label=LABELS[arm])
    for ax, title, units in zip(axes.flat,
        ("All 66 channels", "Temperature at 850 hPa", "Specific humidity at 850 hPa", "Specific humidity at 850 hPa"),
        ("Normalized RMSE", "RMSE (K)", "Empirical CRPS (g/kg)", "Population ensemble spread (g/kg)")):
        ax.set(title=title, xlabel="Forecast lead (days)", ylabel=units, xlim=(1, max(days)))
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    synthetic = report["data_kind"] == "synthetic"
    heading = "SYNTHETIC PIPELINE TEST — no ERA5 skill claim" if synthetic else "Exploratory ERA5 comparison"
    fig.suptitle(f"{heading}\nICs={len(report['test_initial_indices'])}, members={report['members']}; "
                 f"PINN {report['training']['pinn']['iterations']} / FM {report['training']['fm']['iterations']} iterations",
                 fontsize=12)
    fig.savefig(target / "scorecard.png", dpi=180)
    plt.close(fig)
    last = max(leads)
    text = ["# Step13 ODE experiment scorecard", "", heading, "",
            f"Final lead: {last} h. Initial conditions: {len(report['test_initial_indices'])}; members: {report['members']}.",
            "", "| Arm | All-channel normalized RMSE | T850 RMSE (K) | Q850 RMSE (g/kg) | Q850 CRPS (g/kg) |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        nrmse = report["normalized_rmse_all_channels_by_6h_lead"][arm][last // 6 - 1]
        t = lookup[arm, last, "t850"]
        q = lookup[arm, last, "q850"]
        text.append(f"| {LABELS[arm]} | {nrmse:.6f} | {float(t['rmse']):.6f} | "
                    f"{1000*float(q['rmse']):.6f} | {1000*float(q['crps']):.6f} |")
    text += ["", "ODE changes q/T at the 12 pressure levels only. Other channels, including TISR, pass through.",
             "Nudging-only is a required control: reduction against FM alone is not evidence of a transport benefit.",
             "", "![Scorecard](scorecard.png)", "", "[All scores](scores.csv) · [Full metadata](summary.json)", ""]
    (target / "README.md").write_text("\n".join(text))
    print(target / "README.md")


if __name__ == "__main__":
    main()
