"""The result table of the Kalman island + version (scripts/run_kai_island_plus.sh): from one json of
scripts/diag_stage_filter_light.py (every arm against the same control, the fine-tuned KAI), a Markdown table of the
65-channel nRMSE, ACC and activity, the five variables in physical units, and every arm's gain over the control at
EQUAL smoothing (the control given the arm's zonal spectrum, scripts/diag_stage_filter.py match) with its 95 % interval.
--gains adds the same gain table of further jsons of the light check (other controls: D3 against D2 = the matrix
island's share, D3 and D2 against the simple blend = the island structure's share). Signs: + = better.

    python scripts/island_plus_table.py --json OUT/score_island_vs_kai.json --ctrl-label 'KAI (fine-tuned)' \
        --rows 'D3 = ...=ARM' ... [--gains 'TITLE=OUT/score_island_d3.json' ...] --md OUT/table_island_plus.md
"""
import argparse
import json

DAYS = [1, 3, 5, 7, 10]
def sg(x, n=1):
    """Signed number; a rounded zero is +0.0, never -0.0."""
    t = f"{x:+.{n}f}"
    return "+" + t[1:] if float(t) == 0 else t


UNIT = {"t2m": ("K", 1.0), "t850": ("K", 1.0), "z500": ("m²/s²", 1.0), "u850": ("m/s", 1.0), "msl": ("hPa", 0.01)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--ctrl-label", default="control")
    p.add_argument("--rows", nargs="+", required=True, help="LABEL=ARM (an arm of the json)")
    p.add_argument("--gains", nargs="*", default=[], help="TITLE=JSON: gain table of every arm of that json against its control")
    p.add_argument("--days", type=int, nargs="+", default=DAYS)
    p.add_argument("--title", default="Kalman island + version")
    p.add_argument("--md", required=True)
    a = p.parse_args()
    J = json.load(open(a.json))
    rows = [lab_arm.rsplit("=", 1) for lab_arm in a.rows]
    L = len(J["ctrl_rows"]["ctrl"]["nrmse"])
    days = [d for d in a.days if d <= L]
    hd = " | ".join(f"day {d}" for d in days)
    sep = "|".join(["---"] * (len(days) + 1))
    o = [f"# {a.title}", "", f"Test year, {J['ics']} start dates; control = {a.ctrl_label}. + = better.", ""]
    rr = [(a.ctrl_label, J["ctrl_rows"]["ctrl"])] + [(lab, J["arms"][arm]["rows"]["arm"]) for lab, arm in rows]
    for key, name, fmt in (("nrmse", "nRMSE, mean of 65 channels (lower = better)", "{:.3f}"),
                           ("acc", "ACC, mean of 65 channels (higher = better)", "{:.3f}"),
                           ("act", "activity, forecast anomaly size / observed (1 = as active as nature)", "{:.2f}")):
        o += [f"## {name}", "", f"| model | {hd} |", f"|{sep}|"]
        o += [f"| {lab} | " + " | ".join(fmt.format(r[key][d - 1]) for d in days) + " |" for lab, r in rr]
        o.append("")
    o += ["## Gain over the control at equal smoothing, nRMSE % (95 % interval), ACC difference", "",
          f"| model | {hd} |", f"|{sep}|"]
    for lab, arm in rows:
        g = J["arms"][arm]["arm_vs_match"]
        o.append(f"| {lab} | " + " | ".join(f"{sg(g['gain_pct'][d - 1])} ({sg(g['gain_lo'][d - 1])}..{sg(g['gain_hi'][d - 1])})"
                                            for d in days) + " |")
        o.append(f"| {lab}, ACC | " + " | ".join(f"{sg(g['dacc'][d - 1], 3)}" for d in days) + " |")
    o.append("")
    lab_of = {arm: lab for lab, arm in rows}
    for spec in a.gains:  # the same gain at equal smoothing against other controls
        ttl, f = spec.rsplit("=", 1)
        try:
            G = json.load(open(f))
        except (OSError, ValueError) as e:
            o += [f"## {ttl}", "", f"(not available: {e})", ""]
            continue
        o += [f"## {ttl}", "", "nRMSE gain at equal smoothing, % (95 % interval); ACC difference; then the five variables, %", "",
              f"| model | {hd} |", f"|{sep}|"]
        for arm, ga in G["arms"].items():
            g = ga["arm_vs_match"]; lab = lab_of.get(arm, arm)
            o.append(f"| {lab} | " + " | ".join(f"{sg(g['gain_pct'][d - 1])} ({sg(g['gain_lo'][d - 1])}..{sg(g['gain_hi'][d - 1])})"
                                                for d in days) + " |")
            o.append(f"| {lab}, ACC | " + " | ".join(f"{sg(g['dacc'][d - 1], 3)}" for d in days) + " |")
            for v in UNIT:
                o.append(f"| {lab}, {v} | " + " | ".join(f"{sg(g['vars'][v]['gain_pct'][d - 1])}" for d in days) + " |")
        o.append("")
    for v, (unit, k) in UNIT.items():
        o += [f"## {v} ({unit}): RMSE / ACC", "", f"| model | {hd} |", f"|{sep}|"]
        for lab, r in rr:
            q = r["vars"][v]
            o.append(f"| {lab} | " + " | ".join(f"{q['rmse'][d - 1] * k:.2f} / {q['acc'][d - 1]:.3f}" for d in days) + " |")
        for lab, arm in rows:
            g = J["arms"][arm]["arm_vs_match"]["vars"][v]
            o.append(f"| {lab}: gain at equal smoothing | " + " | ".join(f"{sg(g['gain_pct'][d - 1])} %" for d in days) + " |")
        o.append("")
    open(a.md, "w").write("\n".join(o))
    print("\n".join(o))


if __name__ == "__main__":
    main()
