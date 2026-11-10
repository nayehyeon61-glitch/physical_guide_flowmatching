"""Does the past of 3 to 5 days hold information the core does not use? (diagnostic before the accumulated memory)

    python scripts/diag_kai_longmem.py [--cores core_gngmz] [--year 2016] [--leads 1 3 5 7] [--taus 3 5]
                                       [--ic-stride 2] [--out runs/kaitend]

Owner (2026-11-11): the memory should hold the past of 3 to 5 days for certain. The trained GRU uses one day
(scripts/run_kai_tendency.py --mem-warm). Before a structure is trained on it, this measures what is there to be
used. For a trained core and every IC day d of the year:

  target      E_L(d) = (x_{d+L} - forecast of day d + L started from d) / s      L = the leads (free rollout)
  known at d  a_d      = (x_d - climatology of the day) / s                    (the present state's anomaly)
              r_{d-1}  = the one-step residual of the forecast valid at d (yesterday's innovation)
              b_{d-1}  = (x_d - x_{d-1}) / s                                   (yesterday's trend)
              A_tau, B_tau = the last two ACCUMULATED over the past with the kernel exp(-age / tau):
                             A_tau(d) = lam A_tau(d-1) + (1 - lam) r_{d-1}, lam = exp(-1 / tau)

and the fraction of the variance of E_L that a linear combination of the predictors explains, per channel
(pooled over the points and the ICs, cos-lat weighted, same grid point and channel, in sample), for nested sets:
  by time scale  [state + yesterday]  ->  [+ accumulated over tau_1]  ->  [+ accumulated over tau_2] ...
  by kind        [state + yesterday]  ->  [+ accumulated innovations]  ->  [+ accumulated trends]
The increase from the first set to the last is the (linear, pointwise) information of the accumulated past that
neither the present state nor a one-day memory has. The present state's anomaly is in the first set because the
accumulated trend is the state minus its recent mean: without it a damping error (error proportional to the
anomaly) would be counted as memory. A conv net can use more than this (other channels, neighbours, nonlinear),
so it is a lower bound of what is there, not of what a structure will find.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_kai_tendency as R  # noqa: E402
from diag_kai_physmem import explained  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.kai_tend import DYN, TISR, core_rollout  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cores", nargs="+", default=["core_gngmz"])
    p.add_argument("--year", type=int, default=2016)
    p.add_argument("--leads", type=int, nargs="+", default=[1, 3, 5, 7])
    p.add_argument("--taus", type=float, nargs="+", default=[3.0, 5.0])
    p.add_argument("--ic-stride", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    a0, rest = p.parse_known_args(argv)
    a = R.parser().parse_args(rest)
    torch.set_num_threads(a0.threads)
    mean, std, inc, clim = R.stats(a)
    x = R.load_years(a, [a0.year])[a0.year]
    lat = R.lat_of(os.path.join(a.data, f"{a0.year}.nc"))
    contract.register_latitudes(lat)
    xz = torch.from_numpy(R.zscore(x, mean, std))
    dyn = torch.tensor(DYN)
    names = [contract.CHANNEL_NAMES[c] for c in DYN]
    sc = torch.tensor([i for i, n in enumerate(names) if n != "tisr"])  # the 65 scored channels
    nm = [names[i] for i in sc.tolist()]
    groups = {v: [i for i, n in enumerate(nm) if n[0] == v and n[1:].isdigit()] for v in "zqtuv"}
    groups["surface"] = [i for i, n in enumerate(nm) if n in ("t2m", "msl", "sp", "skt", "tcwv")]
    w = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)
    w = (w / w.mean()).view(1, 1, -1, 1)
    s = torch.as_tensor(inc[DYN], dtype=torch.float32)[sc].view(1, -1, 1, 1)
    T, Lmax = len(xz), max(a0.leads)
    xs = xz[:, dyn][:, sc]
    trend = (xs[1:] - xs[:-1]) / s  # trend[t] = x_{t+1} - x_t
    anom = (xs - torch.from_numpy(R.zscore(clim[:T], mean, std))[:, dyn][:, sc]) / s  # from the training climatology
    out = dict(leads=a0.leads, taus=a0.taus, year=a0.year, cores={})

    def hist(core, j):
        W = getattr(core, "mem", 0)
        if W:
            return torch.stack([xz[j - k][:, dyn] for k in range(W, 0, -1)], 1)
        return xz[j - 1][:, dyn] if getattr(core, "hist", False) else None

    def accumulate(v, i0):
        """v[n] belongs to day i0 + n -> i0 + n + 1 (known at day i0 + n + 1). Returns {tau: A} with A[d] = the
        accumulation known at day d (zeros before the first value)."""
        acc = {}
        for tau in a0.taus:
            lam, A, cur = float(np.exp(-1.0 / tau)), torch.zeros(T, *v.shape[1:]), torch.zeros_like(v[0])
            for n in range(len(v)):
                cur = lam * cur + (1 - lam) * v[n]
                A[i0 + n + 1] = cur
            acc[tau] = A
        return acc

    for tag in a0.cores:
        core = R.build_core(a, mean, std, inc, lat, tag=tag)
        i0 = max(getattr(core, "mem", 0), 1)
        res = []
        with torch.no_grad():
            for i in range(i0, T - 1, 16):  # one-step forecasts from data, day i -> i + 1
                j = torch.arange(i, min(i + 16, T - 1))
                y = core_rollout(core, xz[j][:, dyn], j, 1, xz[j + 1][:, None, TISR], prev=hist(core, j))[:, 0]
                res.append((xz[j + 1][:, dyn] - y)[:, sc] / s)
        r = torch.cat(res)  # r[n] = residual of the step from day i0 + n
        Ar, Bt = accumulate(r, i0), accumulate(trend[i0:], i0)
        d0 = i0 + int(3 * max(a0.taus))  # the accumulators have seen about three time constants
        ics = torch.arange(d0, T - Lmax, a0.ic_stride)
        err = []
        with torch.no_grad():
            for b in range(0, len(ics), 16):
                j = ics[b:b + 16]
                tis = torch.stack([xz[j + k + 1][:, TISR] for k in range(Lmax)], 1)
                y = core_rollout(core, xz[j][:, dyn], j, Lmax, tis, prev=hist(core, j))
                tru = torch.stack([xz[j + k + 1][:, dyn] for k in range(Lmax)], 1)
                err.append((tru - y)[:, :, sc] / s.unsqueeze(1))
        err = torch.cat(err)  # (N, Lmax, 65, H, W)
        base = [anom[ics], r[ics - 1 - i0], trend[ics - 1]]  # the present state, yesterday's innovation and trend
        orders = {"time scale": (sum(([Ar[t][ics], Bt[t][ics]] for t in a0.taus), []), 2,
                                 [f"+ {t:g} d acc." for t in a0.taus]),
                  "kind": ([Ar[t][ics] for t in a0.taus] + [Bt[t][ics] for t in a0.taus], len(a0.taus),
                           ["+ acc. innov.", "+ acc. trend"])}
        ww = w.expand(len(ics), 1, -1, 1)
        rows = {}
        print(f"\n{tag}: {len(ics)} ICs of {a0.year}; explained % of the lead-L forecast error by the past known at the IC")
        for name, (extra, step, cols) in orders.items():
            head = ["state + yday"] + cols
            print(f"  nested by {name}")
            print("    lead  group    " + " ".join(f"{h:>13s}" for h in head) + "   gain of the accumulated past")
            for L in a0.leads:
                r2 = explained(err[:, L - 1], base + extra, ww)  # (n predictors, 65): the first m predictors
                sets = r2[len(base) - 1::step]  # after the base set, then after every group of ``step`` predictors
                rows[f"{name}/{L}"] = sets.tolist()
                for g, c in [("all", list(range(len(nm))))] + list(groups.items()):
                    v = 100 * sets[:, c].mean(1)
                    print(f"    {L:4d}  {g:8s} " + " ".join(f"{q:13.2f}" for q in v) + f"   {v[-1] - v[0]:+.2f}")
        out["cores"][tag] = dict(ics=int(len(ics)), r2=rows, names=nm)
    with open(os.path.join(a.out, "diag_longmem.json"), "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
