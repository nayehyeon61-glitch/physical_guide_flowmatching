"""Effective scaling exponents of a ladder of models from one score (docs/kai_core_scaling.md).

A ladder is a list of models that differ along one axis x: parameters, training years or iterations. For the error
E(x) the script prints the local exponent between neighbours, alpha = -d ln E / d ln x, and the least-squares slope
of ln E on ln x over the ladder. Reading: a clear alpha on the parameter (or iteration) axis with alpha near 0 on
the year axis = the model is short of capacity (or of steps) and more years cannot count yet; the reverse = short
of data.

Two errors per lead: the score's nRMSE (mean over the 65 scored channels of RMSE / std) and its information part,
sqrt(s_o^2 (1 - r^2)) / std: the error that is left after the best amplitude scaling of each forecast. The second
does not move when a model only damps its forecast, which is how RMSE is gained at long leads without information.

    python scripts/kai_scaling_fit.py runs/kaitend/score_S2.json runs/kaitend/score_S2_raw.npz \\
        --points core_gngmz8h24 core_gngmz8 core_gngmz8w core_gngmz8h96 --json runs/kaitend/scaling_size.json
    ... --axis years --points core_gngmz8_y1=1 core_gngmz8_y2=2 core_gngmz8=4 core_gngmz8_y10=10 core_gngmz8_y32=32
    ... --wrap "loop_{}@kai_y32x8ro"      the same ladder inside the coupled loop (x is still the core's)

``tag=x`` gives x by hand; without it x is the parameter count of the tag in the score. A two-point ladder gives
one local exponent and no error bar: a slope over three or more points is the least that deserves the word "law".
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_kai_breakdown import terms2  # noqa: E402

DAYS = (1, 3, 5, 7, 10)


def errors(S, raw, std, tag):
    """nRMSE and information nRMSE per lead of one arm, (L,) each."""
    sc = np.array([i for i, n in enumerate(S["channels"]) if n not in ("orog", "tisr")])
    mse = raw[f"{tag}__mse"].astype(np.float64).mean(0)  # (L, 67)
    tot = (np.sqrt(mse)[:, sc] / std[sc]).mean(1)
    info = None
    if f"{tag}__dec" in raw and f"{tag}__act" in raw and "climatology__dec" in raw:
        info = (np.sqrt(np.clip(terms2(raw, tag)[..., 1], 0, None))[:, sc] / std[sc]).mean(1)
    return tot, info


def exponents(x, E):
    """Local exponents between neighbours (n - 1, L) and the least-squares slope over the ladder (L,)."""
    lx, lE = np.log(np.asarray(x, float)), np.log(np.asarray(E, float))  # (n,), (n, L)
    local = -(lE[1:] - lE[:-1]) / (lx[1:] - lx[:-1])[:, None]
    slope = -np.polyfit(lx, lE, 1)[0] if len(x) > 2 else local[0]
    return local, slope


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("score")
    p.add_argument("raw")
    p.add_argument("--points", nargs="+", required=True, help="tags of the ladder in the order of x, or tag=x")
    p.add_argument("--axis", default="parameters")
    p.add_argument("--wrap", default="{}", help='arm name pattern, e.g. "loop_{}@kai_y32x8ro"')
    p.add_argument("--stats", default="runs/kaitend/stats.npz")
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    S, raw = json.load(open(a.score)), np.load(a.raw)
    std = np.load(a.stats)["std"].astype(np.float64) + 1e-8
    tags, xs = [], []
    for spec in a.points:
        t, x = (spec.split("=", 1) + [None])[:2]
        tags.append(t)
        xs.append(float(x) if x is not None else float(S["arms"][t]["params"]))
    arms = [a.wrap.format(t) for t in tags]
    E = [errors(S, raw, std, k) for k in arms]
    idx = [d - 1 for d in DAYS if d <= S["leads"]]
    f = lambda v, n=3: " / ".join(f"{u:.{n}f}" for u in np.asarray(v)[idx])  # noqa: E731
    out = {"axis": a.axis, "x": xs, "arms": arms, "days": [i + 1 for i in idx]}
    print(f"ladder along {a.axis}; day " + " / ".join(str(i + 1) for i in idx))
    for name, j in (("nRMSE", 0), ("information nRMSE", 1)):
        if any(e[j] is None for e in E):
            continue
        Ej = np.stack([e[j] for e in E])
        local, slope = exponents(xs, Ej)
        print(f"\n{name}")
        for k, x, e in zip(arms, xs, Ej):
            print(f"  {k:44s} x = {x:<10g} {f(e)}")
        for i in range(len(xs) - 1):
            gain = 100 * (1 - Ej[i + 1] / Ej[i])
            print(f"  x {xs[i]:g} -> {xs[i + 1]:g} (x {xs[i + 1] / xs[i]:.2f}): gain % {f(gain, 1)}   alpha {f(local[i])}")
        if len(xs) > 2:
            print(f"  slope over the ladder: alpha {f(slope)}")
        out[name] = {"E": Ej[:, idx].tolist(), "alpha_local": local[:, idx].tolist(), "alpha_slope": slope[idx].tolist()}
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh)


if __name__ == "__main__":
    main()
