"""Does a longer training period need another climatology? The climatology's trend against the test year.

``stats.npz`` of a run (the normalization, and the climatology of the score's ACC / activity / error terms and of
the blend's climatology expert) comes from the training years. With 1980-2011 as training period that is the
climatology of a colder climate than the test year's. Measured here, for the climatologies of several periods:

  1. the climatology as a forecast of a year (every day): the RMSE in units of the run's std (mean over the 65
     scored channels) and the exact split of its MSE into the offset of the global year mean, the stationary
     pattern (the year mean of the error at every grid point, without that offset) and the rest (weather);
  2. the ACC, the activity and the three error terms of stored forecasts (``--dump``: the forecasts written by
     ``scripts/kai_place_router.py``'s dump) with the anomalies taken from each climatology. The forecasts and
     their MSE are the same; only the reference moves;
  3. the shift of the normalization (mean, std per channel) and the trend of the global mean per decade.

Climatologies: ``--periods 1980-2011 2002-2011 2012-2015`` (day-of-year mean with 31-day smoothing, as in
``run_kai_tendency.stats``), and for ``--trend 1980-2011`` three trend-aware ones: the period's climatology moved
to the target year by the linear trend (least squares on the annual means of the period's years) of the global
mean (``+g``), of the zonal mean (``+z``) or of every grid point (``+p``).

    python scripts/kai_clim_trend.py --cache data/era5_5p625/kaiday --stats runs/kaitend/stats.npz \
        --dump runs/kaitend/dump_router --tags kai_ro core_gngmz8w --periods 1980-2011 2002-2011 2012-2015 \
        --trend 1980-2011 --years 2016 2017 --json runs/kaitend/diag_clim_trend.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract  # noqa: E402

NAMES = list(contract.CHANNEL_NAMES)
SCORED = [i for i, n in enumerate(NAMES) if n not in ("orog", "tisr")]
MAIN = ["t2m", "t850", "z500", "u850", "msl", "t50", "tcwv", "skt"]


def unit(n):
    """(factor, unit) of a channel for the tables."""
    if n in ("msl", "sp"):
        return 0.01, "hPa"
    if n[0] == "q":
        return 1000.0, "g/kg"
    if n[0] == "z":
        return 1.0, "m2/s2"
    if n[0] in "uv":
        return 1.0, "m/s"
    return 1.0, "kg/m2" if n == "tcwv" else "K"


def span(s):
    a, b = (int(q) for q in s.split("-"))
    return list(range(a, b + 1))


def smooth(clim, cnt):
    for i in range(366):
        clim[i] = clim[i] / cnt[i] if cnt[i] else clim[i - 1]  # 29 Feb / 31 Dec of a leap year: the neighbour
    out = np.empty(clim.shape, dtype=np.float32)
    for i in range(366):
        out[i] = clim[np.arange(i - 15, i + 16) % 366].mean(0)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--periods", nargs="+", required=True)
    p.add_argument("--trend", default=None, help="period whose climatology is also moved to the target year by its trend")
    p.add_argument("--years", type=int, nargs="+", default=[2016, 2017], help="years the climatologies are compared with")
    p.add_argument("--dump", default=None)
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    st = np.load(a.stats)
    mean, std = st["mean"], st["std"]
    sd = (std + 1e-8)[:, None, None]
    per = {s: span(s) for s in a.periods}
    need = sorted(set(sum(per.values(), [])) | set(a.years))
    year = lambda y: np.load(os.path.join(a.cache, f"{y}.npy"), mmap_mode="r")  # noqa: E731
    C, H, W = year(need[0]).shape[1:]
    lat = np.linspace(-90 + 90 / H, 90 - 90 / H, H)
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None]                                  # (H, 1): mean over the globe = (x * w).mean()

    # ---- one pass over the years: day-of-year sums, annual means, moments
    acc = {s: np.zeros((366, C, H, W)) for s in per}
    cnt = {s: np.zeros(366) for s in per}
    mom = {s: [np.zeros(C), np.zeros(C), 0] for s in per}
    ann = {}
    for y in need:
        x = year(y)
        tot = np.zeros((C, H, W))
        for i in range(0, len(x), 61):
            c = np.asarray(x[i:i + 61], dtype=np.float64)
            tot += c.sum(0)
            for s, ys in per.items():
                if y in ys:
                    acc[s][i:i + len(c)] += c
                    d = c - mean[None, :, None, None]
                    mom[s][0] += d.sum((0, 2, 3))
                    mom[s][1] += (d ** 2).sum((0, 2, 3))
                    mom[s][2] += c.shape[0] * H * W
        for s, ys in per.items():
            if y in ys:
                cnt[s][:len(x)] += 1
        ann[y] = tot / len(x)
        print(f"read {y}", flush=True)
    clim = {s: smooth(acc[s], cnt[s]) for s in per}
    del acc
    clim["run"] = st["clim"].astype(np.float32)                  # the climatology of the run's stats.npz
    out = dict(periods=per, channels=NAMES, norm={}, trend={}, clim={}, forecasts={})

    # ---- normalization shift and the trend of the global mean
    for s in per:
        m1, m2, n = mom[s]
        mu = mean + m1 / n
        sg = np.sqrt(m2 / n - (m1 / n) ** 2)
        out["norm"][s] = dict(mean_shift_in_std=((mu - mean) / (std + 1e-8)).tolist(), std_ratio=(sg / (std + 1e-8)).tolist())
    ys = np.array(sorted(ann))
    g = np.stack([(ann[y] * w).mean((1, 2)) for y in ys])        # (years, C) global annual means
    t = ys - ys.mean()
    slope = (t[:, None] * (g - g.mean(0))).sum(0) / (t ** 2).sum()
    out["trend"] = dict(years=[int(ys[0]), int(ys[-1])], per_decade=(10 * slope).tolist(), global_mean={int(y): g[i].tolist() for i, y in enumerate(ys)})

    # ---- trend-aware climatologies of one period, per target year
    def moved(s, target):
        yy = np.array(per[s])
        A = np.stack([ann[y] for y in yy])                       # (n, C, H, W)
        tt = yy - yy.mean()
        sl = (tt[:, None, None, None] * (A - A.mean(0))).sum(0) / (tt ** 2).sum()   # per grid point, per year
        dt = target - yy.mean()
        gz = (sl * w).mean((1, 2), keepdims=True)
        zz = sl.mean(2, keepdims=True)
        return {f"{s}+g": (clim[s] + dt * gz).astype(np.float32), f"{s}+z": (clim[s] + dt * zz).astype(np.float32),
                f"{s}+p": (clim[s] + dt * sl).astype(np.float32)}

    def clims_for(target):
        c = dict(clim)
        if a.trend:
            c.update(moved(a.trend, target))
        return c

    # ---- 1. the climatology as a forecast of a year
    for y in a.years:
        x = year(y)
        n = len(x)
        res = {}
        for s, cl in clims_for(y).items():
            se = np.zeros((C, H, W))
            me = np.zeros((C, H, W))
            for i in range(0, n, 61):
                e = (np.asarray(x[i:i + 61], dtype=np.float64) - cl[i:i + 61][:len(x[i:i + 61])]) / sd
                se += (e ** 2).sum(0)
                me += e.sum(0)
            me /= n
            mse = (se / n * w).mean((1, 2))
            off = (me * w).mean((1, 2))
            stat = ((me - off[:, None, None]) ** 2 * w).mean((1, 2))
            res[s] = dict(nrmse=float(np.sqrt(mse)[SCORED].mean()), mse=mse.tolist(), offset=off.tolist(),
                          terms=dict(offset=float((off ** 2)[SCORED].mean()), stationary=float(stat[SCORED].mean()),
                                     weather=float((mse - off ** 2 - stat)[SCORED].mean())))
        out["clim"][str(y)] = res

    # ---- 2. the stored forecasts with the anomalies from each climatology
    if a.dump and a.tags:
        meta = np.load(os.path.join(a.dump, "meta.npz"))
        ty, ics, L = int(meta["test_year"]), meta["test_ics"], int(meta["leads"])
        assert np.allclose(meta["mean"], mean) and np.allclose(meta["std"], std)
        x = year(ty)
        z = lambda v: ((v - mean[:, None, None]) / sd).astype(np.float32)  # noqa: E731
        truth = np.stack([z(np.asarray(x[i + 1:i + 1 + L], dtype=np.float64)) for i in ics])    # (N, L, C, H, W)
        x0 = np.stack([z(np.asarray(x[i], dtype=np.float64)) for i in ics])
        preds = {"persistence": np.broadcast_to(x0[:, None], truth.shape)}
        for tg in a.tags:
            preds[tg] = np.load(os.path.join(a.dump, f"{tg}__test.npy"), mmap_mode="r")
        out["forecasts"] = {tg: {} for tg in preds}
        m = lambda q: (q * w).mean((-2, -1))  # noqa: E731
        days = np.minimum(ics[:, None] + 1 + np.arange(L)[None], 365)                           # as in the score
        for s, cl in clims_for(ty).items():
            clz = z(cl.astype(np.float64))
            for tg, pr in preds.items():
                P = dict(acc=[], accc=[], act=[], dec=[])
                for c in range(0, len(ics), 17):
                    cc = clz[days[c:c + 17]].astype(np.float64)
                    pa, ta = np.asarray(pr[c:c + 17], dtype=np.float64) - cc, truth[c:c + 17] - cc
                    mf, mo, pt = m(pa), m(ta), m(pa * ta)
                    vf, vo = m(pa ** 2), m(ta ** 2)
                    sf, so = np.sqrt(np.maximum(vf - mf ** 2, 0)), np.sqrt(np.maximum(vo - mo ** 2, 0))
                    cov = pt - mf * mo
                    P["acc"].append(pt / np.sqrt(vf * vo + 1e-30))
                    P["accc"].append(cov / (sf * so + 1e-30))
                    P["act"].append(sf / (so + 1e-30))
                    P["dec"].append(np.stack([(mf - mo) ** 2, (sf - so) ** 2, 2 * (sf * so - cov)], -1))   # (n, L, C, 3)
                P = {k: np.concatenate(v).mean(0) for k, v in P.items()}                        # (L, C[, 3])
                out["forecasts"][tg][s] = dict(acc=P["acc"][:, SCORED].mean(1).tolist(),
                                               acc_centred=P["accc"][:, SCORED].mean(1).tolist(),
                                               act=P["act"][:, SCORED].mean(1).tolist(),
                                               terms=P["dec"][:, SCORED].mean(1).tolist(),
                                               acc_by_channel=P["acc"].tolist())
            print(f"forecasts scored with the climatology {s}", flush=True)

    # ---- print
    ci = [NAMES.index(n) for n in MAIN]
    print("\ntrend of the global mean per decade (%d-%d):" % tuple(out["trend"]["years"]))
    print("  " + "  ".join(f"{n} {10 * slope[i] * unit(n)[0]:+.3g} {unit(n)[1]}" for n, i in zip(MAIN, ci)))
    print("\nnormalization of a period against the run's (mean shift in std / std ratio):")
    for s in per:
        q = out["norm"][s]
        print(f"  {s:>10s}  " + "  ".join(f"{n} {q['mean_shift_in_std'][i]:+.3f}/{q['std_ratio'][i]:.3f}" for n, i in zip(MAIN, ci)))
    for y in a.years:
        print(f"\nclimatology as a forecast of {y}: nRMSE (65 channels), MSE terms offset / stationary / weather, "
              f"global-mean offset (truth - climatology) of " + ", ".join(MAIN[:5]))
        for s, r in out["clim"][str(y)].items():
            tr = r["terms"]
            print(f"  {s:>12s}  {r['nrmse']:.4f}   {tr['offset']:.4f} / {tr['stationary']:.4f} / {tr['weather']:.4f}   "
                  + "  ".join(f"{r['offset'][i] * std[i] * unit(n)[0]:+.3g}" for n, i in zip(MAIN[:5], ci[:5])))
    for tg, res in out["forecasts"].items():
        print(f"\n{tg}: ACC (as scored) day 1/3/5/7/10 | activity | information term")
        for s, r in res.items():
            d = [0, 2, 4, 6, 9]
            print(f"  {s:>12s}  " + "/".join(f"{r['acc'][k]:.3f}" for k in d) + "  |  " + "/".join(f"{r['act'][k]:.3f}" for k in d)
                  + "  |  " + "/".join(f"{r['terms'][k][2]:.4f}" for k in d))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
