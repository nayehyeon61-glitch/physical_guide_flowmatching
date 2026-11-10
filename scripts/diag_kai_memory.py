"""How much memory is left in the one-step residuals of a tendency core? (Mori-Zwanzig diagnostic)

    python scripts/diag_kai_memory.py --cores core_gn core_gnh [--year 2016] [--lags 8] [--out runs/kaitend]

The exact equation of the resolved state is  dx/dt = Markov term + memory integral + noise, and the noise is
orthogonal to the past of the resolved variables. A core's one-step residual from data,
r_t = x_{t+1} - core(x_t), is therefore (memory it has not captured) + (noise): whatever part of r_t can still
be predicted from the past is uncaptured memory. Measured here, per channel and lag s = 1..S, as the pattern
correlation (cos-lat weighted, mean over the days) of
    r_t with r_{t-s}                     the residual sequence with itself (white = nothing left)
    r_t with the trend x_{t-s+1} - x_{t-s}   the residual with the observed past increments
and as the upper bound of a linear memory of S lags: the fraction of the residual variance explained by a
regression of r_t on r_{t-1..t-S} per channel (pooled over space and days, in sample).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_kai_tendency as R  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.kai_tend import DYN, TISR, core_rollout  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cores", nargs="+", required=True)
    p.add_argument("--year", type=int, default=2016)
    p.add_argument("--lags", type=int, default=8)
    p.add_argument("--threads", type=int, default=1)
    a0, rest = p.parse_known_args(argv)
    a = R.parser().parse_args(rest)
    torch.set_num_threads(a0.threads)
    mean, std, inc, _ = R.stats(a)
    x = R.load_years(a, [a0.year])[a0.year]
    lat = R.lat_of(os.path.join(a.data, f"{a0.year}.nc"))
    xz = torch.from_numpy(R.zscore(x, mean, std))
    dyn = torch.tensor(DYN)
    w = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)
    w = (w / w.mean()).view(1, 1, -1, 1)
    s = torch.as_tensor(inc[DYN], dtype=torch.float32).view(1, -1, 1, 1)
    names = [contract.CHANNEL_NAMES[c] for c in DYN]
    sc = [i for i, n in enumerate(names) if n != "tisr"]
    groups = {v: [i for i, n in enumerate(names) if n[0] == v and n[1:].isdigit()] for v in "zqtuv"}
    groups["surface"] = [i for i, n in enumerate(names) if n in ("t2m", "msl", "sp", "skt", "tcwv")]
    S, T = a0.lags, len(xz)
    trend = ((xz[1:, dyn] - xz[:-1, dyn]) / s)  # trend[t] = x_{t+1} - x_t
    out = {}

    def pc(u, v):  # pattern correlation per (day, channel)
        return (u * v * w).sum((-2, -1)) / torch.sqrt((u ** 2 * w).sum((-2, -1)) * (v ** 2 * w).sum((-2, -1)) + 1e-30)

    for tag in a0.cores:
        core = R.build_core(a, mean, std, inc, lat, tag=tag)
        W = getattr(core, "mem", 0)
        res = []
        with torch.no_grad():
            for i in range(max(W, 1), T - 1, 16):  # one-step forecasts from data, day i -> i + 1
                j = torch.arange(i, min(i + 16, T - 1))
                prev = None
                if W:
                    prev = torch.stack([xz[j - k][:, dyn] for k in range(W, 0, -1)], 1)
                elif getattr(core, "hist", False):
                    prev = xz[j - 1][:, dyn]
                y = core_rollout(core, xz[j][:, dyn], j, 1, xz[j + 1][:, None, TISR], prev=prev)[:, 0]
                res.append((xz[j + 1][:, dyn] - y) / s)
        r = torch.cat(res)  # r[n] = residual of the step from day i0 + n
        i0 = max(W, 1)
        n = len(r)
        rr = np.stack([pc(r[k:], r[:n - k]).mean(0).numpy() for k in range(1, S + 1)])       # (S, 66)
        rt = []
        for k in range(1, S + 1):  # r of the step from day d with the increment observed k days before: trend[d - k]
            n0 = max(0, k - i0)
            rt.append(pc(r[n0:], trend[i0 + n0 - k:i0 + n0 - k + n - n0]).mean(0).numpy())
        rt = np.stack(rt)
        # linear memory of S lags: regression of r_t on r_{t-1..t-S}, one coefficient per lag and channel
        A = np.zeros((66, S, S))
        b = np.zeros((66, S))
        tot = (r[S:] ** 2 * w).sum((0, 2, 3)).numpy()
        Z = [r[S - k:n - k] for k in range(1, S + 1)]
        for p_ in range(S):
            b[:, p_] = (r[S:] * Z[p_] * w).sum((0, 2, 3)).numpy()
            for q in range(p_, S):
                A[:, p_, q] = A[:, q, p_] = (Z[p_] * Z[q] * w).sum((0, 2, 3)).numpy()
        r2 = np.zeros((S, 66))
        for m in range(1, S + 1):  # explained fraction with the first m lags
            for c in sc:
                coef = np.linalg.solve(A[c, :m, :m] + 1e-9 * np.eye(m) * A[c, 0, 0], b[c, :m])
                r2[m - 1, c] = float(coef @ b[c, :m] / tot[c])
        rms = float(torch.sqrt((r[:, sc] ** 2 * w).mean()))
        out[tag] = dict(res_rms=rms, res_res=rr.tolist(), res_trend=rt.tolist(), r2=r2.tolist())
        f = lambda v, c=sc: " ".join(f"{v[k][c].mean():+.3f}" for k in range(S))  # noqa: E731
        print(f"\n{tag}: one-step residual RMS {rms:.4f} (increment-std units), {n} days of {a0.year}")
        print(f"  lag                      " + " ".join(f"{k:6d}" for k in range(1, S + 1)))
        print(f"  corr(r_t, r_t-s)   all   {f(rr)}")
        for g, c in groups.items():
            print(f"                     {g:5.5s} {f(rr, c)}")
        print(f"  corr(r_t, trend_t-s) all {f(rt)}")
        print(f"  explained by s lags all  " + " ".join(f"{100 * r2[k][sc].mean():6.1f}" for k in range(S)) + "  % of the residual variance")
        for g, c in groups.items():
            print(f"                     {g:5.5s} " + " ".join(f"{100 * r2[k][c].mean():6.1f}" for k in range(S)))
    with open(os.path.join(a.out, "diag_memory.json"), "w") as fh:
        json.dump(dict(names=names, lags=S, year=a0.year, cores=out), fh)


if __name__ == "__main__":
    main()
