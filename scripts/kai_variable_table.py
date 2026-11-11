"""Per-variable table of a KAI / tendency-core score: RMSE in physical units by lead, and the gains.

    python scripts/kai_variable_table.py runs/kaitend/score_Lk.json out.md \
        "kai=KAI" "core_gngmz=final core" "loop_core_gngmz=final model" ... [--base kai] [--pairs A:B ...]
        [--leads 1 3 5 7 10] [--head t2m t850 z500 u850 msl] [--csv out.csv]
        [--extra score_Lg.json score_Lh.json ...] [--stats runs/kaitend/stats.npz] [--raw score_Li_raw.npz ...]

Owner (2026-11-11): the comparison is read as a weather-benchmark table, per variable (t2m, t850, z500, ...) in
physical units with persistence and climatology as references, next to the normalized 65-channel gains.
Writes a Markdown file with (1) the RMSE of every scored variable for every listed model, (2) the gain of every
model over ``--base`` in % (+ = better), (3) the gain of A over B for every ``--pairs A:B``; prints the head table
(the ``--head`` variables) and up to which lead each model beats climatology.
Units: z m2/s2, t / t2m / skt K, u / v m/s, q g/kg, msl / sp hPa, tcwv kg/m2.
``--csv`` also writes the RMSE of every listed model at every lead as one row per model and variable
(model, label, variable, group, level, unit, d1 .. dN): the data file of the heat map / line graph page.
``--extra`` reads models that the first score file does not hold from other score files of the same test set
(checked: the persistence RMSE has to be identical), so cores of different scores go into one table.
``--stats`` adds the column ``norm`` to the csv: the training std of the variable in the unit of the table, so that
RMSE / norm is the normalized RMSE of the score and a total over variables can be formed (mean of RMSE / norm).
``--raw`` (score_*_raw.npz files) adds the columns i1 .. iN and a1 .. aN to the csv: the information RMSE (what is
left of the RMSE at the best amplitude: sqrt(bias^2 + s_o^2 (1 - r^2)), the split of eval_kai_breakdown.py) and the
activity (pattern amplitude of the forecast / truth's). With them the gain of A over B in one cell splits exactly:
100 (1 - A / B) = information part 100 (i_B^2 - i_A^2) / (B (A + B)) + amplitude part (the rest).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract  # noqa: E402

NAMES = contract.CHANNEL_NAMES


def unit(n):
    if n in ("msl", "sp"):
        return 0.01, "hPa"
    if n == "tcwv":
        return 1.0, "kg/m²"
    if n[0] == "q":
        return 1000.0, "g/kg"
    if n[0] == "z":
        return 1.0, "m²/s²"
    if n[0] in "uv":
        return 1.0, "m/s"
    return 1.0, "K"  # t levels, t2m, skt


def fmt(v):
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}" if v >= 0.1 else f"{v:.3f}"


def terms(raws, tags):
    """tag -> (information RMSE (L, C), activity (L, C)) from the raw files that hold the tag's dec / act metrics."""
    out = {}
    for f in raws:
        z = np.load(f)
        if "climatology__dec" not in z.files:
            continue
        so2 = z["climatology__dec"][..., 1].astype(np.float64)  # the truth's pattern variance
        for k in tags:
            if k in out or f"{k}__dec" not in z.files or f"{k}__act" not in z.files:
                continue
            d, am = z[f"{k}__dec"].astype(np.float64), z[f"{k}__act"].astype(np.float64)
            cov = am * so2 - d[..., 2] / 2
            info = so2 - np.where(am > 1e-6, cov ** 2 / (am ** 2 * so2 + 1e-30), 0.0)
            out[k] = (np.sqrt(np.maximum((d[..., 0] + info).mean(0), 0.0)), am.mean(0))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("score")
    p.add_argument("out")
    p.add_argument("models", nargs="+", help="tag=label; persistence and climatology are added in front")
    p.add_argument("--base", default="kai")
    p.add_argument("--pairs", nargs="*", default=[], help="A:B = gain of A over B per variable")
    p.add_argument("--leads", type=int, nargs="+", default=[1, 3, 5, 7, 10])
    p.add_argument("--head", nargs="+", default=["t2m", "t850", "z500", "u850", "msl"])
    p.add_argument("--head-leads", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--csv", default="", help="also write model,label,variable,group,level,unit,d1..dN (physical units)")
    p.add_argument("--extra", nargs="*", default=[], help="more score.json files of the same test set (missing models)")
    p.add_argument("--stats", default="", help="stats.npz of the run: adds the column norm (training std) to the csv")
    p.add_argument("--raw", nargs="*", default=[], help="score_*_raw.npz: adds information RMSE i1..iN and activity a1..aN")
    a = p.parse_args(argv)
    d = json.load(open(a.score))
    arms = dict(d["arms"])
    for f in a.extra:
        e = json.load(open(f))["arms"]
        if not np.array_equal(np.array(e["persistence"]["rmse"]), np.array(arms["persistence"]["rmse"]), equal_nan=True):
            sys.exit(f"{f}: the persistence RMSE differs, not the same test set")
        for k, v in e.items():
            arms.setdefault(k, v)
    models = [("persistence", "persistence"), ("climatology", "climatology")]
    models += [tuple(m.split("=", 1)) if "=" in m else (m, m) for m in a.models]
    models = [(k, lab) for k, lab in models if k in arms]
    lab = dict(models)
    R = {k: np.array(arms[k]["rmse"]) for k, _ in models}
    sc = [i for i, n in enumerate(NAMES) if n not in ("tisr", "orog")]
    L = [x - 1 for x in a.leads]
    ls = " / ".join(str(x) for x in a.leads)

    def cell(k, c, leads=L):
        return " / ".join(fmt(R[k][i, c] * unit(NAMES[c])[0]) for i in leads)

    def gain(ka, kb, c):
        return " / ".join(f"{100 * (1 - R[ka][i, c] / R[kb][i, c]):+.1f}" for i in L)

    n_ic = d.get("ics", 0)
    n_ic = n_ic if isinstance(n_ic, int) else len(n_ic)
    out = [f"# 변수별 RMSE 비교 ({os.path.basename(a.score)}, IC {n_ic}개, 일평균 5.625°)", "",
           f"RMSE는 물리 단위이고 낮을수록 좋습니다. 선행 {ls}일 순서입니다.",
           "단위: z m²/s², t / t2m / skt K, u / v m/s, q g/kg, msl / sp hPa, tcwv kg/m².", "",
           "모델: " + "; ".join(f"{b} = `{k}`" for k, b in models), "",
           f"## 1. 변수별 RMSE (선행 {ls}일)", "",
           "| 변수 | 단위 | " + " | ".join(b for _, b in models) + " |", "|---|---|" + "---|" * len(models)]
    out += [f"| {NAMES[c]} | {unit(NAMES[c])[1]} | " + " | ".join(cell(k, c) for k, _ in models) + " |" for c in sc]
    cmp_ = [(k, b) for k, b in models if k not in ("persistence", "climatology", a.base)]
    if a.base in R and cmp_:
        out += ["", f"## 2. {lab[a.base]} 대비 RMSE 이득 % (+ = 좋아짐, 선행 {ls}일)", "",
                "| 변수 | " + " | ".join(b for _, b in cmp_) + " |", "|---|" + "---|" * len(cmp_)]
        out += [f"| {NAMES[c]} | " + " | ".join(gain(k, a.base, c) for k, _ in cmp_) + " |" for c in sc]
    pairs = [pr.split(":") for pr in a.pairs]
    pairs = [(x, y) for x, y in pairs if x in R and y in R]
    if pairs:
        out += ["", f"## 3. 짝 비교: RMSE 이득 % (+ = 앞의 모델이 좋음, 선행 {ls}일)", "",
                "| 변수 | " + " | ".join(f"{lab[x]} 대 {lab[y]}" for x, y in pairs) + " |", "|---|" + "---|" * len(pairs)]
        out += [f"| {NAMES[c]} | " + " | ".join(gain(x, y, c) for x, y in pairs) + " |" for c in sc]
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")

    if a.csv:
        nl = R[models[0][0]].shape[0]
        std = np.load(a.stats)["std"] if a.stats else None
        T = terms(a.raw, [k for k, _ in models])
        with open(a.csv, "w", encoding="utf-8") as fh:
            fh.write("model,label,variable,group,level,unit," + ("norm," if a.stats else "")
                     + ",".join(f"d{i + 1}" for i in range(nl))
                     + ("," + ",".join(f"{x}{i + 1}" for x in "ia" for i in range(nl)) if a.raw else "") + "\n")
            for k, b in models:
                for c in sc:
                    n = NAMES[c]
                    lev = n[1:] if n[0] in "zqtuv" and n[1:].isdigit() else ""
                    row = [k, b.replace(",", " "), n, n[0] if lev else "sfc", lev, unit(n)[1]]
                    row += [f"{std[c] * unit(n)[0]:.6g}"] if a.stats else []
                    row += [f"{R[k][i, c] * unit(n)[0]:.5g}" for i in range(nl)]
                    if a.raw:  # empty cells for a model without raw metrics
                        row += [f"{T[k][0][i, c] * unit(n)[0]:.5g}" for i in range(nl)] if k in T else [""] * nl
                        row += [f"{T[k][1][i, c]:.3f}" for i in range(nl)] if k in T else [""] * nl
                    fh.write(",".join(row) + "\n")

    hl = [x - 1 for x in a.head_leads]
    head = [h for h in a.head if h in NAMES]
    print(f"RMSE, 선행 {' / '.join(str(x) for x in a.head_leads)}일")
    print("| 모델 | " + " | ".join(f"{h} ({unit(h)[1]})" for h in head) + " |")
    print("|---|" + "---|" * len(head))
    for k, b in models:
        print(f"| {b} | " + " | ".join(cell(k, NAMES.index(h), hl) for h in head) + " |")
    if a.base in R:
        for x, y in [(k, a.base) for k, _ in cmp_] + pairs:
            print(f"\nRMSE 이득 %, {lab[x]} 대 {lab[y]}, 선행 {ls}일")
            for h in head:
                print(f"| {h} | {gain(x, y, NAMES.index(h))} |")
    if "climatology" in R:
        print("\nclimatology를 이기는 마지막 선행일")
        for k, b in models:
            if k in ("persistence", "climatology"):
                continue
            row = []
            for h in head:
                c = NAMES.index(h)
                days = [i + 1 for i in range(R[k].shape[0]) if R[k][i, c] < R["climatology"][i, c]]
                row.append(f"{h} {max(days) if days else 0}")
            print(f"  {b}: " + ", ".join(row))


if __name__ == "__main__":
    main()
