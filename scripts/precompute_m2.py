"""The second-moment hint of a two-stage forecast, made once along the frozen first stage's trajectory.

For nonlinear dynamics the mean of a spread of states does not follow the central forecast:
mean = G(mean) + 1/2 H C + ... (Evensen, Data Assimilation, eq. 4.27). The second stage of a two-stage forecast
(scripts/run_kai_tendency.py --stage1) gets the equations' own estimate of that term as 48 input channels:

    m2(x) = mean_p 1/2 [D(x + e_p) + D(x - e_p)] - D(x)

D = the dycore's 24 h integrated increment of u, v, T, q (the hint), x = the FIRST stage's forecast at day
n1 + l (l = 0 .. leads - 1: the states the second stage starts its days from, to the accuracy that both stages
follow the same trajectory), e_p = a fixed library of real forecast errors of the first stage at the same lead
(``--npairs`` cases of the validation year, so no training or test case meets its own error; symmetric pairs: the
first-order response cancels, the forecast stays deterministic). Diagnostic: scripts/diag_dycore_moment.py.

Writes ``<out>/m2_<stage1>_<year>.npy`` (days, leads, 48, H, W) float16 = m2 / (0.3 x the hint's scale), clamped
at 5; zeros where it was not computed (``<...>_<year>_days.npy`` lists the days done). Restartable per year.

    python scripts/precompute_m2.py --out runs/kaitend --stage1 core_gbv --years 2012 2013 2014 2015
    python scripts/precompute_m2.py --out runs/kaitend --stage1 core_gbv --years 2017 --stride 3
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_kai_tendency as R  # noqa: E402
from pgcfm.kai_tend import DYN, TISR, core_rollout  # noqa: E402

REL = 0.3  # m2 is about 0.3 of the hint's RMS (diagnostic): the stored channels are of order one


def main():
    p = R.parser()
    p.add_argument("--years", type=int, nargs="+", required=True)
    p.add_argument("--stride", type=int, default=1, help="every stride-th day of the year as IC")
    p.add_argument("--m2-leads", type=int, default=5)
    p.add_argument("--npairs", type=int, default=8)
    p.add_argument("--lib-year", type=int, default=2016)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    tr = R.load_years(a, a.train_years)
    mean, std, inc, clim = R.stats(a, tr)
    del tr
    n1, Lm, P, B = a.stage1_days, a.m2_leads, a.npairs, 8
    nr = n1 + Lm - 1                                     # days of the first stage: states at day n1 .. n1 + Lm - 1
    lat = None
    core, dy, lib = None, None, None
    keep = torch.ones(len(DYN)); keep[DYN.index(TISR)] = 0
    keep = keep.view(1, -1, 1, 1)

    def roll(xz, days):
        tg = torch.stack([xz[days + k + 1] for k in range(nr)], 1)
        with torch.no_grad():
            y = core_rollout(core, xz[days][:, DYN], torch.tensor(days), nr, tg[:, :, TISR])
        return y[:, n1 - 1:].float(), tg[:, n1 - 1:][:, :, DYN].float()   # (B, Lm, 66, H, W) forecast, truth

    def integ(x):
        raw, _ = dy._integrate(x)
        return torch.from_numpy(raw).float(), np.asarray(dy.last_bad).copy()

    for year in [a.lib_year] + list(a.years):
        yr = R.load_years(a, [year])[year]
        if core is None:
            lat = R.lat_of(os.path.join(a.data, f"{a.test_year}.nc"), yr.shape[-2], a.era5)
            core = R.build_core(a, mean, std, inc, lat, load=True, tag=a.stage1)
            core.noise_on = False
            dy = core.features.dy
            scale = dy.scale.cpu().float() * REL
        xz = torch.from_numpy(R.zscore(yr, mean, std))
        if lib is None:  # the library: the first stage's errors of P cases of the validation year, per lead
            lp = os.path.join(a.out, f"m2_{a.stage1}_lib.pt")
            if os.path.exists(lp):
                lib = torch.load(lp)
            else:
                days = np.linspace(20, len(xz) - nr - 20, P).astype(int)
                f, t = roll(xz, days)
                lib = ((t - f) * keep.unsqueeze(1)).transpose(0, 1).contiguous()   # (Lm, P, 66, H, W)
                torch.save(lib, lp)
            assert lib.shape[0] >= Lm and lib.shape[1] >= P, lib.shape
            continue
        path = os.path.join(a.out, f"m2_{a.stage1}_{year}.npy")
        dpath = path[:-4] + "_days.npy"
        days_all = np.arange(0, len(xz) - nr, a.stride)
        done = np.load(dpath) if os.path.exists(dpath) else np.zeros(0, int)
        H, W = xz.shape[-2:]
        tab = (np.lib.format.open_memmap(path, mode="r+") if os.path.exists(path) and len(done)
               else np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(xz), Lm, 48, H, W)))
        todo = np.setdiff1d(days_all, done)
        t0, bad_n = time.time(), 0
        for i in range(0, len(todo), B):
            d = todo[i:i + B]
            dd = np.concatenate([d, np.repeat(d[-1:], B - len(d))])  # one batch shape for the dycore
            f, _ = roll(xz, dd)
            for l in range(Lm):
                x = f[:, l]
                d0, b0 = integ(x)
                acc, cnt = torch.zeros_like(d0), torch.zeros(B)
                for q in range(P):
                    e = lib[l, q][None]
                    rp, bp = integ(x + e)
                    rm, bm = integ(x - e)
                    ok = torch.from_numpy(~(bp | bm | b0)).float()
                    acc += (0.5 * (rp + rm) - d0) * ok.view(-1, 1, 1, 1)
                    cnt += ok
                bad_n += int((cnt[:len(d)] < P).sum())
                z = (acc / cnt.clamp_min(1).view(-1, 1, 1, 1) / scale).clamp(-5, 5)
                tab[d, l] = z[:len(d)].numpy().astype(np.float16)
            done = np.concatenate([done, d])
            if (i // B) % 10 == 9 or i + B >= len(todo):
                tab.flush()
                np.save(dpath, done)
                print(json.dumps(dict(year=year, done=int(len(done)), of=int(len(days_all)), short_pairs=bad_n,
                                      sec=round(time.time() - t0))), flush=True)
        del tab


if __name__ == "__main__":
    main()
