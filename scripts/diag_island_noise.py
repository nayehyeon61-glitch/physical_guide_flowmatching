"""The owner's assumption as an experiment (2026-11-17 10:37 KST): the equation stays, the state changes by noise.

    "The equation does not change; what changes is the phenomenon, and it changes by noise. With a noise put in at
     the middle we use the preserved equation from a new state."

ONE frozen core runs all the days. At the island (day ``--n1``) the state is made new by noise: members
x_n1 +- a P_k (symmetric pairs, so the member mean at the island is the central state), every member is
integrated by the SAME core, and the forecast of the days after the island is the member mean. Nothing is trained.
The noise:

  shape err    P_k = the core's own day-n1 forecast errors of OTHER dates of ``--lib-year`` (same season, at least
               ``--gap`` days away when the library year is the forecast year): real error patterns, balanced and
               large-scale; a = 1 is the measured size of the island's error (its entropy in the Gaussian sense)
  shape white  P = white noise smoothed over 3 x 3 cells x the library's error RMS per channel and grid point

A memory core (``--mem-shift own``, the default) takes the new state as its own: its last forecast and its previous
state are moved by the same P, so the memory sees no innovation and no jump of the trend. ``--mem-shift innov``: the
memory sees the noise as an innovation (the state was changed from outside, as in a coupled rollout).

The members share the central run up to the island (one run, the memory state copied to every member). Written per
amplitude: the member-mean forecast in the cache format of scripts/diag_stage_filter.py (``fc_<name>_<year>.npy``,
name = isn-<tag>-<shape><a>-m<M>; that script then takes ``--arms <name>`` and gives the equal-smoothing check
against any control), and a json / log with nRMSE, ACC, activity of the member mean of 2, 4, .. members, of one
member, the spread / error ratio and the five variables.

    python scripts/diag_island_noise.py --out runs/kaitend --tag core_gb --amps 1 --members 8
    python scripts/diag_stage_filter.py --out runs/kaitend --arms isn-core_gb-err1-m8 'core_gb>core_s2c@5' --ctrl core_gb
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_kai_tendency as R  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.kai_tend import DYN, TISR, core_step  # noqa: E402

VARS = ["t2m", "t850", "z500", "u850", "msl"]
MEM_STATE = ("m", "prev", "gprev", "fc", "z", "M", "PD")  # a memory core's tensors of ONE trajectory


def run_days(core, x, doy, ks, tis):
    """The days ``ks`` (indices of tis) of one core from x: (B, len(ks), 66, H, W)."""
    st, out = getattr(core, "stride", 1), []
    for k in ks:
        x = core_step(core, x, doy + k * st, tis[:, k])
        out.append(x)
    return torch.stack(out, 1)


def get_state(core):
    return ({k: getattr(core, k) for k in MEM_STATE if torch.is_tensor(getattr(core, k, None))},
            list(getattr(core, "past", [])))


def set_state(core, state, M, pert=None):
    """Every member gets the central run's memory; ``pert``: its last forecast and previous state move with the
    noise (the new state is the core's own)."""
    st, past = state
    for k, v in st.items():
        v = v.repeat_interleave(M, 0)
        setattr(core, k, v + pert if pert is not None and k in ("prev", "fc") else v)
    if past:
        core.past = [v.repeat_interleave(M, 0) for v in past]


def main():
    p = R.parser()
    p.add_argument("--n1", type=int, default=5, help="the island: the day after which the noise is put in")
    p.add_argument("--amps", type=float, nargs="+", default=[1.0])
    p.add_argument("--shape", default="err", choices=["err", "white"])
    p.add_argument("--members", type=int, default=8, help="even: symmetric pairs")
    p.add_argument("--mem-shift", default="own", choices=["own", "innov"])
    p.add_argument("--lib-year", type=int, default=None, help="year of the error patterns (default: --val-year)")
    p.add_argument("--lib-stride", type=int, default=5)
    p.add_argument("--season", type=int, default=45, help="patterns from dates within this many days of the year")
    p.add_argument("--gap", type=int, default=20, help="... and at least this far away in the same year")
    p.add_argument("--year", type=int, default=None, help="year of the forecasts (default: --test-year)")
    p.add_argument("--chunk", type=int, default=4, help="ICs per batch (x members)")
    p.add_argument("--max-ics", type=int, default=0, help="> 0: only every k-th of the ICs so that about this many are run")
    p.add_argument("--no-save", action="store_true", help="scores only")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    assert a.tag, "--tag: the core (the one equation)"
    assert a.members >= 2 and a.members % 2 == 0, "symmetric pairs"
    torch.manual_seed(a.seed)
    year, lib_year = a.year or a.test_year, a.lib_year or a.val_year
    tr = R.load_years(a, a.train_years)
    mean, std, inc, clim = R.stats(a, tr)
    del tr
    data = R.load_years(a, sorted({year, lib_year}))
    lat = R.lat_of(os.path.join(a.data, f"{year}.nc"), data[year].shape[-2], a.era5)
    Z = {y: torch.from_numpy(R.zscore(data[y], mean, std)) for y in data}
    del data
    cz = torch.from_numpy(R.zscore(clim, mean, std))[:, DYN]
    L, n1, M, B0 = a.leads, a.n1, a.members, a.chunk
    core = R.build_core(a, mean, std, inc, lat, load=True, tag=a.tag)
    core.noise_on = False
    if getattr(core, "nz", 0):
        core.features.noise_on = False
    W1 = getattr(core, "mem", 0)
    assert not getattr(core, "entv", False) and (W1 or not getattr(core, "hist", False)), "plain or memory cores"
    nan = lambda: int(getattr(getattr(core, "features", None), "n_nan", 0))  # noqa: E731
    keep = torch.ones(len(DYN))
    keep[DYN.index(TISR)] = 0
    keep = keep.view(1, -1, 1, 1)
    w = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)
    w = (w / w.mean()).view(1, -1, 1)
    sc = [i for i, c in enumerate(DYN) if c != TISR]
    vi = [DYN.index(contract.CHANNEL_INDEX[v]) for v in VARS]
    vstd = np.asarray(std[[contract.CHANNEL_INDEX[v] for v in VARS]], dtype=np.float64).reshape(-1)
    wm = lambda x: (x.double() * w).mean((-2, -1))  # noqa: E731
    rep = lambda v: v.repeat_interleave(M, 0)  # noqa: E731

    def batch(tez, j, size):
        """The ICs j, filled up to ``size`` with the last one (one batch shape = one compilation of the dycore)."""
        j = np.concatenate([j, np.repeat(j[-1:], size - len(j))])
        x0, doy = tez[j][:, DYN], torch.tensor(j)
        tis = torch.stack([tez[j + k + 1] for k in range(L)], 1)[:, :, TISR]
        pv = torch.stack([tez[np.maximum(j - k, 0)] for k in range(W1, 0, -1)], 1)[:, :, DYN] if W1 else None
        return x0, doy, tis, pv

    def start(x0, doy, pv):
        if W1:
            core.reset(pv, doy.float() * 24.0, x0)

    # ---- the library: the core's day-n1 errors over the library year (normalized states, 66 channels)
    lib_path = os.path.join(a.out, f"isn_lib_{a.tag}_{n1}_{lib_year}_s{a.lib_stride}.npz")
    if os.path.exists(lib_path):
        f = np.load(lib_path)
        lib, lib_d = torch.from_numpy(f["err"].astype(np.float32)), f["day"]
    else:
        tez = Z[lib_year]
        ld = np.arange(0, len(tez) - L, a.lib_stride)
        errs, t0 = [], time.time()
        for i in range(0, len(ld), B0 * M):
            j = ld[i:i + B0 * M]
            x0, doy, tis, pv = batch(tez, j, B0 * M)
            with torch.no_grad():
                start(x0, doy, pv)
                y = run_days(core, x0, doy, range(n1), tis)[:len(j), -1]
            errs.append((y - tez[j + n1][:, DYN]) * keep)
            print(f"library {i + len(j)} / {len(ld)} {time.time() - t0:.0f} s", flush=True)
        lib, lib_d = torch.cat(errs), ld + n1  # the day of the year the error belongs to
        np.savez(lib_path, err=lib.numpy().astype(np.float16), day=lib_d)
    lib_rms = torch.sqrt((lib ** 2).mean(0))  # (66, H, W)
    print(f"library {lib_year}: {len(lib)} patterns, rms {float(lib_rms[sc].mean()):.3f} state std", flush=True)

    tez = Z[year]
    ics = np.arange(0, len(tez) - L, a.ic_stride)
    if a.max_ics and len(ics) > a.max_ics:
        ics = ics[::int(np.ceil(len(ics) / a.max_ics))]
    N = len(ics)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)

    def patterns(j):
        """(B, M / 2, 66, H, W): the noise of the pairs of the ICs j (amplitude 1)."""
        out = []
        for d in j + n1:
            if a.shape == "white":
                xi = torch.randn((M // 2, len(DYN)) + tuple(lib.shape[-2:]), generator=gen)
                xi = F.avg_pool2d(F.pad(F.pad(xi, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="reflect"), 3, 1) * 3.0
                out.append(xi * lib_rms * keep)
                continue
            dd = np.abs(lib_d - d)
            dd = np.minimum(dd, 365 - dd)
            ok = dd <= a.season
            if lib_year == year:
                ok &= np.abs(lib_d - d) >= a.gap
            out.append(lib[rng.choice(np.flatnonzero(ok), M // 2, replace=False)])
        return torch.stack(out)

    def sums(fc, Tb, Cb):
        """A forecast (B, L, 66, H, W) -> the sums over its ICs of the area means (L, 66) the scores are made of."""
        pa, ta = fc - Cb, Tb - Cb
        return torch.stack([wm(v).sum(0) for v in ((pa - ta) ** 2, pa ** 2, ta ** 2, pa * ta)])

    def score(S):
        mse, pp, tt, pt = (v.numpy() for v in S / N)
        acc = pt / np.sqrt(pp * tt)
        return dict(nrmse=np.sqrt(mse)[:, sc].mean(1).tolist(), acc=acc[:, sc].mean(1).tolist(),
                    act=np.sqrt(pp / tt)[:, sc].mean(1).tolist(),
                    vars={v: dict(rmse=(np.sqrt(mse[:, q]) * s).tolist(), acc=acc[:, q].tolist(),
                                  act=np.sqrt(pp[:, q] / tt[:, q]).tolist()) for v, q, s in zip(VARS, vi, vstd)})

    names = {amp: f"isn-{a.tag}-{a.shape}{amp:g}-m{M}" + ("-innov" if W1 and a.mem_shift == "innov" else "")
             for amp in a.amps}
    kinds = [f"m{2 * k}" for k in range(1, M // 2 + 1)] + ["one"]
    acc = {"central": 0.0, **{(amp, k): 0.0 for amp in a.amps for k in kinds}}
    var = {amp: 0.0 for amp in a.amps}
    nans = {"central": 0, **{amp: 0 for amp in a.amps}}
    keepfc = {amp: [] for amp in a.amps}
    cen_fc, t0 = [], time.time()
    for i in range(0, N, B0):
        j = ics[i:i + B0]
        nb = len(j)
        x0, doy, tis, pv = batch(tez, j, B0)
        Tb = torch.stack([tez[j + k + 1] for k in range(L)], 1)[:, :, DYN]
        Cb = torch.stack([cz[(torch.tensor(j) + k + 1) % len(cz)] for k in range(L)], 1)
        P1 = patterns(j)                                                         # (nb, M/2, 66, H, W)
        P1 = torch.cat([P1, P1[-1:].expand(B0 - nb, *P1.shape[1:])])
        with torch.no_grad():
            n0 = nan()
            start(x0, doy, pv)
            ya = run_days(core, x0, doy, range(n1), tis)                         # the central run up to the island
            state = get_state(core)
            cen = torch.cat([ya, run_days(core, ya[:, -1], doy, range(n1, L), tis)], 1)[:nb]
            nans["central"] += nan() - n0
            acc["central"] = acc["central"] + sums(cen, Tb, Cb)
            cen_fc.append(cen.numpy().astype(np.float16))
            for amp in a.amps:
                n0 = nan()
                pert = torch.stack([P1, -P1], 2).reshape(B0 * M, *P1.shape[2:]) * amp   # members +P_1, -P_1, +P_2, ..
                set_state(core, state, M, pert if a.mem_shift == "own" else None)
                y = run_days(core, rep(ya[:, -1]) + pert, rep(doy), range(n1, L), rep(tis))
                y = y.view(B0, M, L - n1, len(DYN), *x0.shape[-2:])[:nb]
                nans[amp] += nan() - n0
                full = lambda v: torch.cat([ya[:nb], v], 1)                      # noqa: E731
                for k in range(1, M // 2 + 1):
                    acc[amp, f"m{2 * k}"] = acc[amp, f"m{2 * k}"] + sums(full(y[:, :2 * k].mean(1)), Tb, Cb)
                acc[amp, "one"] = acc[amp, "one"] + sums(full(y[:, 0]), Tb, Cb)
                var[amp] = var[amp] + wm(y.var(1, unbiased=True)).sum(0)         # (L - n1, 66)
                if not a.no_save and not a.max_ics:
                    keepfc[amp].append(full(y.mean(1)).numpy().astype(np.float16))
        print(f"{i + nb} / {N} {time.time() - t0:.0f} s, failed dycore states {sum(nans.values())}", flush=True)

    out = {"tag": a.tag, "n1": n1, "members": M, "shape": a.shape, "mem_shift": a.mem_shift if W1 else None,
           "year": year, "lib_year": lib_year, "ics": int(N), "lib": int(len(lib)), "lib_rms": float(lib_rms[sc].mean()),
           "central": score(acc["central"]), "nan_central": nans["central"], "runs": {}}
    g = lambda v, d=3: " ".join(f"{x:.{d}f}" for x in v)  # noqa: E731
    c = out["central"]
    path = os.path.join(a.out, f"fc_{a.tag}_{year}.npy")
    if os.path.exists(path) and not a.max_ics:  # the same forecast as the other scripts' (their loop)
        ref = np.load(path).astype(np.float32)
        cen_all = np.concatenate(cen_fc).astype(np.float32)
        out["central_vs_cache"] = float(np.abs(cen_all - ref).max()) if ref.shape == cen_all.shape else None
        print("central run vs the cached forecast, max |difference|:", out["central_vs_cache"])
    print(f"== {a.tag} central run, day 1..{L} ({N} ICs of {year})\n nRMSE {g(c['nrmse'])}\n ACC   {g(c['acc'])}"
          f"\n act   {g(c['act'], 2)}", flush=True)
    for amp in a.amps:
        name = names[amp]
        r = {k: score(acc[amp, k]) for k in kinds}
        r["mean"], r["nan_states"] = r[f"m{M}"], nans[amp]
        mse = (acc[amp, f"m{M}"][0] / N)[n1:, sc].mean(1)
        r["ssr"] = torch.sqrt((1 + 1 / M) * (var[amp] / N)[:, sc].mean(1) / mse).tolist()
        out["runs"][name] = r
        m = r["mean"]
        print(f"== {name}: mean of {M} members, noise after day {n1}, {r['nan_states']} failed dycore states")
        print(f" nRMSE {g(m['nrmse'])}\n ACC   {g(m['acc'])}\n act   {g(m['act'], 2)}"
              f"\n spread / error, day {n1 + 1}..{L}: {g(r['ssr'], 2)}")
        print(" gain over the central run (%, + = better) "
              + " ".join(f"{100 * (1 - x / y):+.1f}" for x, y in zip(m["nrmse"], c["nrmse"])))
        print(" ACC - central " + " ".join(f"{x - y:+.3f}" for x, y in zip(m["acc"], c["acc"])))
        for k in kinds[:-2]:
            print(f" {k[1:]} members: nRMSE {g(r[k]['nrmse'][n1:])} | ACC {g(r[k]['acc'][n1:])} | act {g(r[k]['act'][n1:], 2)}")
        print(f" one member: nRMSE {g(r['one']['nrmse'][n1:])} | ACC {g(r['one']['acc'][n1:])} | act {g(r['one']['act'][n1:], 2)}")
        for q in VARS:
            print(f" {q:5s} RMSE {g(m['vars'][q]['rmse'], 2)} | central {g(c['vars'][q]['rmse'], 2)}")
            print(f"       ACC  {g(m['vars'][q]['acc'])} | central {g(c['vars'][q]['acc'])}")
        sys.stdout.flush()
        if keepfc[amp]:
            np.save(os.path.join(a.out, f"fc_{name}_{year}.npy"), np.concatenate(keepfc[amp]))
    js = a.json or os.path.join(a.out, f"diag_island_noise_{a.tag}_{a.shape}" + ("_innov" if W1 and a.mem_shift == "innov" else "") + ".json")
    json.dump(out, open(js, "w"), indent=1)
    # the sums themselves (4 x L x 66: MSE, forecast power, truth power, covariance; / ICs), e.g. for the limit of
    # many members: the power of a mean of k pairs is A + B / k, its covariance with the truth does not depend on k
    np.savez(js[:-5] + "_sums.npz", central=(acc["central"] / N).numpy(),
             **{f"{names[amp]}__{k}": (acc[amp, k] / N).numpy() for amp in a.amps for k in kinds},
             **{f"{names[amp]}__var": (var[amp] / N).numpy() for amp in a.amps})


if __name__ == "__main__":
    main()
