"""Pairs of one score file (run_kai_tendency.py --score): nRMSE gain, information gain and ACC difference of A over
B per lead with a paired bootstrap over the ICs (+ = A better), and the rows of the listed arms.

    python scripts/score_pair_info.py runs/kaitend score_st2.json score_st2_raw.npz A:B [C:D ...] [--arms X Y ...]
        [--leads 1 3 5 6 7 8 9 10] [--json out.json]

Information = the RMSE left at the best amplitude, sqrt(bias + s_o^2 (1 - r^2)) (scripts/eval_kai_breakdown.py):
a forecast that is only damped gains nothing there. ACC and activity in the score files' definition (per forecast,
then the mean over the ICs).
"""
import argparse
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("score")
    p.add_argument("raw")
    p.add_argument("pairs", nargs="*")
    p.add_argument("--arms", nargs="*", default=[])
    p.add_argument("--leads", type=int, nargs="+", default=[1, 3, 5, 6, 7, 8, 9, 10])
    p.add_argument("--boot", type=int, default=400)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    S = json.load(open(os.path.join(a.out, a.score)))
    raw = np.load(os.path.join(a.out, a.raw))
    std = np.load(os.path.join(a.out, "stats.npz"))["std"].astype(np.float64) + 1e-8
    sc = np.array([i for i, n in enumerate(S["channels"]) if n not in ("orog", "tisr")])
    so2 = raw["climatology__dec"][..., 1].astype(np.float64)
    N, li = so2.shape[0], [l - 1 for l in a.leads]

    def parts(k):
        d, am = raw[f"{k}__dec"].astype(np.float64), raw[f"{k}__act"].astype(np.float64)
        cov = am * so2 - d[..., 2] / 2
        info = so2 - np.where(am > 1e-6, cov ** 2 / (am ** 2 * so2 + 1e-30), 0.0)
        return raw[f"{k}__mse"].astype(np.float64), d[..., 0] + info, raw[f"{k}__acc"].astype(np.float64), am

    def stat(P, idx):
        mse, inf, acc, am = P
        return ((np.sqrt(mse[idx].mean(0))[:, sc] / std[sc]).mean(1),
                (np.sqrt(np.maximum(inf[idx].mean(0), 0))[:, sc] / std[sc]).mean(1),
                acc[idx].mean(0)[:, sc].mean(1), am[idx].mean(0)[:, sc].mean(1))

    out, every = {"leads": a.leads, "arms": {}, "pairs": {}}, np.arange(N)
    print("day", " ".join(f"{l:>5d}" for l in a.leads))
    for k in a.arms:
        n, i, c, m = stat(parts(k), every)
        out["arms"][k] = dict(nrmse=n[li].round(4).tolist(), info=i[li].round(4).tolist(), acc=c[li].round(4).tolist(),
                              act=m[li].round(3).tolist())
        print(f"{k:22s} nRMSE {' '.join(f'{x:.3f}' for x in n[li])} | ACC {' '.join(f'{x:.3f}' for x in c[li])}"
              f" | act {' '.join(f'{x:.2f}' for x in m[li])}")
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, N, N) for _ in range(a.boot)]
    gain = lambda x, y: np.stack([100 * (1 - x[0] / y[0]), 100 * (1 - x[1] / y[1]), x[2] - y[2]])  # noqa: E731
    for pr in a.pairs:
        ka, kb = pr.split(":")
        A, B = parts(ka), parts(kb)
        est = gain(stat(A, every), stat(B, every))
        bs = np.stack([gain(stat(A, i), stat(B, i)) for i in boots])
        lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        print(f"\n{ka} vs {kb} (+ = {ka} better)")
        out["pairs"][pr] = {}
        for j, (nm, f) in enumerate([("nRMSE gain %", "{:+.1f}[{:+.1f},{:+.1f}]"), ("information gain %", "{:+.1f}[{:+.1f},{:+.1f}]"),
                                     ("ACC difference", "{:+.3f}[{:+.3f},{:+.3f}]")]):
            print(f"  {nm:20s}", " ".join(f.format(est[j][l], lo[j][l], hi[j][l]) for l in li))
            out["pairs"][pr][nm] = dict(est=est[j][li].round(4).tolist(), lo=lo[j][li].round(4).tolist(), hi=hi[j][li].round(4).tolist())
    if a.json:
        json.dump(out, open(os.path.join(a.out, a.json), "w"))


if __name__ == "__main__":
    main()
