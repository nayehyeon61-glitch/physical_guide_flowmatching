"""The island as a Kalman filter (owner 2026-11-17 11:06 KST): prediction, correction at the island, prediction.

    day 0 .. n1      the first core (prediction: its error variance S at the island = the entropy of the forecast)
    the island       x_a = x_f + K (z - x_f): the measurement z is the day-n1 forecast of KAI (--kalman TAG), its
                     error variance R; the gain per channel and latitude from the library year,
                       --gain corr    K = (S - c) / (S + R - 2 c), c = the covariance of the two errors
                       --gain plain   K = S / (S + R)                    (the two errors taken as independent)
                       --gain isl     K(x): S(x) = the island's entropy read off the island state by the second
                                      stage's island head (a core trained with --joint-kalman / --joint-entv),
                                      R and the error correlation rho from the library year, c = rho sqrt(S R)
                     --robust nu: the innovation of every grid point weighted by (nu + 1) / (nu + d^2 / var_d), the
                     Student-t weight (a long-tailed innovation moves the state less)
    day n1+1 .. L    the second core (--stage2 TAG) or the same core, from the corrected state; --amps a > 0: members
                     x_a +- a sqrt(P_a / S) P_k (what is left of the uncertainty after the correction,
                     P_a = (1 - K)^2 S + K^2 R + 2 K (1 - K) c), their mean is the forecast; a = 0: one run

--kal-days D ...: the same correction on days BEFORE the island as well (a side check: the owner's goal of
2026-11-17 17:33 KST is to keep the early days as they are and improve the late ones, and KAI's forecast is the only
thing measured so far that adds to day 1-4), a Kalman filter cycle along the first core: on each of these days
x_a = x_f + K_d (z_d - x_f), z_d = KAI's forecast of that day, and the corrected state is the start of the next day
(the memory sees the correction as an innovation). The gain of every day comes from the library year IN SEQUENCE: the errors of day d are those of the
run that already has the corrections of the days before it. --kal-open: only the output of these days is corrected,
the core goes on from its own state (= the best linear blend per lead; the comparison).

Under a Gaussian the entropy power of an error is its variance, so K is the owner's entropy form; the entropy taken
out at the island is I = 1/2 ln(S / P_a) nats per degree of freedom (printed). A memory core that goes on after the
island sees the correction as an innovation (the input it was trained with) and the noise as its own state.
A second stage WITH the memory (owner 2026-11-17 17:47 KST; trained by run_kai_tendency.py --joint-freeze with a
memory core as --joint) takes the first stage's memory over at the island and goes on in the same way.
Without --kalman this is scripts/diag_island_noise.py with a second stage. Nothing is trained.

Written per amplitude: the forecast in the cache format of scripts/diag_stage_filter.py (``fc_<name>_<year>.npy``),
a json / log (nRMSE, ACC, activity, five variables, KAI and the corrected state at the island) and the sums.

    python scripts/diag_island_kalman.py --out runs/kaitend --tag core_gndmz8 --stage2 core_s2c --kalman kai --amps 0

A fine grid (1.5 deg, the Kalman island + version, scripts/run_kai_island_plus.sh): --device cuda runs the cores and
KAI on the GPU (the sums stay on the CPU), the years are normalized on access (they stay on disk) and the forecasts are
written into the cache file block by block (one cache of 119 ICs x 10 days is 4.6 GB at 1.5 deg). --table-only stops
after the gain table; --no-central / --no-runs do not write the forecast without / with the correction.
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
from diag_island_noise import VARS, get_state, max_diff, run_days, set_state  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.kai_tend import DYN, TISR, core_step, hand_memory  # noqa: E402


class CacheWriter:
    """A forecast cache (N, L, 66, H, W) float16 written block by block (np.lib.format.open_memmap): at 1.5 deg one
    cache of 119 ICs is 4.6 GB, so the blocks are not kept in memory. The file gets its name when it is complete."""

    def __init__(self, path, shape):
        self.path, self.tmp, self.n = path, path[:-4] + ".part.npy", 0
        self.mm = np.lib.format.open_memmap(self.tmp, mode="w+", dtype=np.float16, shape=tuple(shape))

    def write(self, block):
        self.mm[self.n:self.n + len(block)] = block
        self.n += len(block)

    def close(self):
        assert self.n == len(self.mm), (self.n, self.mm.shape)
        self.mm.flush()
        del self.mm
        os.replace(self.tmp, self.path)


def main():
    p = R.parser()
    p.add_argument("--stage2", default=None, help="the core of the days after the island (default: the same core)")
    p.add_argument("--kalman", default=None, help="KAI checkpoint tag: its day-n1 forecast is the measurement")
    p.add_argument("--gain", default="corr", choices=["corr", "plain", "isl"])
    p.add_argument("--robust", type=float, default=0.0, help="> 0: Student-t weight with this many degrees of freedom")
    p.add_argument("--n1", type=int, default=5, help="the island")
    p.add_argument("--kal-days", type=int, nargs="*", default=[], help="days before the island with the same correction")
    p.add_argument("--kal-open", action="store_true", help="--kal-days: correct the output only, no feedback to the core")
    p.add_argument("--kai-fc", nargs="?", const="save", default=None, choices=["save", "only"],
                   help="also save KAI's forecasts of all leads (fc_<kalman>_<year>.npy); only: nothing else")
    p.add_argument("--isl-inc", default=None, help="npy (ICs, channels, H, W), z-scored: added to the island state after the "
                   "correction (the matrix gain of scripts/diag_island_ensemble.py --save); the name gets +inc<stem>")
    p.add_argument("--amps", type=float, nargs="+", default=[0.0], help="noise after the island (0 = one run)")
    p.add_argument("--shape", default="err", choices=["err", "white"])
    p.add_argument("--members", type=int, default=8, help="even: symmetric pairs")
    p.add_argument("--lib-year", type=int, default=None, help="year of the error statistics (default: --val-year)")
    p.add_argument("--lib-stride", type=int, default=5)
    p.add_argument("--season", type=int, default=45)
    p.add_argument("--gap", type=int, default=20)
    p.add_argument("--year", type=int, default=None, help="year of the forecasts (default: --test-year)")
    p.add_argument("--chunk", type=int, default=4, help="ICs per batch (x members)")
    p.add_argument("--kai-chunk", type=int, default=16, help="ICs per batch of KAI's forecasts")
    p.add_argument("--max-ics", type=int, default=0)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-central", action="store_true", help="do not write the forecast without the correction")
    p.add_argument("--no-runs", action="store_true", help="do not write the forecasts with the correction")
    p.add_argument("--table-only", action="store_true", help="stop after the gain table of the library year")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    assert a.tag, "--tag: the first core"
    assert a.members >= 2 and a.members % 2 == 0, "symmetric pairs"
    torch.manual_seed(a.seed)
    year, lib_year = a.year or a.test_year, a.lib_year or a.val_year
    tr = R.load_years(a, a.train_years)
    mean, std, inc, clim = R.stats(a, tr)
    del tr
    data = R.load_years(a, sorted({year, lib_year}))
    lat = R.lat_of(os.path.join(a.data, f"{year}.nc"), data[year].shape[-2], a.era5)
    # a fine grid stays on disk and is normalized on access (R.ZYear), a coarse one is normalized here
    Z = {y: (R.ZYear(data[y], mean, std, as_torch=True) if isinstance(data[y], np.memmap)
             else torch.from_numpy(R.zscore(data[y], mean, std))) for y in data}
    del data
    dev = torch.device(a.device)
    cz = torch.from_numpy(R.zscore(clim, mean, std))[:, DYN]
    L, n1, M, B0 = a.leads, a.n1, a.members, a.chunk
    core = R.build_core(a, mean, std, inc, lat, load=True, tag=a.tag)
    core2 = R.build_core(a, mean, std, inc, lat, load=True, tag=a.stage2) if a.stage2 else None
    for c in (core, core2):
        if c is not None:
            c.noise_on = False
            assert not getattr(c, "nz", 0), "plain or memory cores"
    assert not getattr(core, "entv", False)
    ent2 = core2 is not None and getattr(core2, "entv", False)   # a second stage with an entropy state: it starts
    assert not ent2 or getattr(core2, "isl_head", None) is not None, "an entropy second stage needs the island head"
    assert a.gain != "isl" or (ent2 and a.kalman), "--gain isl: --kalman and a second stage with the island head"
    sd2 = torch.as_tensor(inc[DYN], dtype=torch.float32).view(1, -1, 1, 1) ** 2   # the entropy state's unit
    W1 = getattr(core, "mem", 0)
    assert W1 or not getattr(core, "hist", False)
    # a second stage WITH the memory (owner 2026-11-17 17:47 KST): it takes over the first stage's memory at the
    # island (hand_memory), so the island's correction is an innovation for it, as for the same core going on
    W2 = getattr(core2, "mem", 0) if core2 is not None else 0
    assert core2 is None or (W2 == W1 and getattr(core2, "mz", False) == getattr(core, "mz", False)) or not (
        W2 or getattr(core2, "hist", False)), "a plain second stage, or one of the first stage's memory form"
    cont = core2 if core2 is not None else core   # the core of the days after the island
    Wc = W1 if (core2 is None or W2) else 0       # ... carries the first stage's memory: the same core, or a memory core
    nan = lambda: sum(int(getattr(getattr(c, "features", None), "n_nan", 0)) for c in (core, core2) if c is not None)  # noqa: E731
    kai = None
    if a.kalman:
        ck = torch.load(os.path.join(a.out, f"{a.kalman}.pt"), map_location="cpu", weights_only=False)
        kai = R.build(a, a.kalman.split("_")[0], inc, None)
        miss = kai.load_state_dict(ck["state"], strict=False)
        assert set(miss.missing_keys) <= {"dyn"} and not miss.unexpected_keys, miss
        kai.eval()
        del ck
    keep = torch.ones(len(DYN))
    keep[DYN.index(TISR)] = 0
    keep = keep.view(1, -1, 1, 1)
    w = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)
    w = (w / w.mean()).view(1, -1, 1)
    sc = [i for i, c in enumerate(DYN) if c != TISR]
    vi = [DYN.index(contract.CHANNEL_INDEX[v]) for v in VARS]
    vstd = np.asarray(std[[contract.CHANNEL_INDEX[v] for v in VARS]], dtype=np.float64).reshape(-1)
    wm = lambda x: (x.double() * w).mean((-2, -1))  # noqa: E731

    def batch(tez, j, size):
        """The ICs j, filled up to ``size`` with the last one (one batch shape = one compilation of the dycore)."""
        j = np.concatenate([j, np.repeat(j[-1:], size - len(j))])
        x0, doy = tez[j][:, DYN].to(dev), torch.tensor(j, device=dev)
        tis = torch.stack([tez[j + k + 1][:, TISR] for k in range(L)], 1).to(dev)
        pv = torch.stack([tez[np.maximum(j - k, 0)][:, DYN] for k in range(W1, 0, -1)], 1).to(dev) if W1 else None
        return j, x0, doy, tis, pv

    def start(x0, doy, pv):
        if W1:
            core.reset(pv, doy.float() * 24.0, x0)

    def kai_days(tez, j, n=None):
        """(len(j), n, 66, H, W): KAI's forecasts of day 1 .. n (default n1) from the ICs j."""
        out, n = [], n or n1
        with torch.no_grad():
            for i in range(0, len(j), a.kai_chunk):
                q = j[i:i + a.kai_chunk]
                tis = torch.stack([tez[q + k + 1][:, TISR] for k in range(n)], 1).to(dev)
                out.append(kai.rollout(tez[q].to(dev), torch.tensor(q, device=dev), n, tis,
                                       prev=tez[np.maximum(q - 1, 0)].to(dev))[:, :, DYN].cpu())
        return torch.cat(out)

    if a.kai_fc and kai is not None and not a.max_ics:   # KAI's own forecasts of every lead, in the cache format
        pk = os.path.join(a.out, f"fc_{a.kalman}_{year}.npy")
        if not os.path.exists(pk):
            jk = np.arange(0, len(Z[year]) - L, a.ic_stride)
            wk = None
            for i in range(0, len(jk), a.kai_chunk):
                b = kai_days(Z[year], jk[i:i + a.kai_chunk], L).numpy().astype(np.float16)
                wk = wk or CacheWriter(pk, (len(jk),) + b.shape[1:])
                wk.write(b)
            wk.close()
            print(f"saved {pk}", flush=True)
        if a.kai_fc == "only":
            return

    early = sorted({d for d in a.kal_days if 0 < d < n1}) if a.kalman else []   # corrected days before the island
    feed, st1, keep5 = not a.kal_open, getattr(core, "stride", 1), keep.unsqueeze(1)
    keep_d = keep.to(dev)

    def stage1(x0, doy, tis, zs, Kd):
        """Day 1 .. n1 of the first core with the corrections Kd (day -> gain) toward zs: (the forecasts as they
        leave the core, the outputs); with ``feed`` the corrected state is the start of the next day."""
        x, raw, outp = x0, [], []
        for k in range(n1):
            x = core_step(core, x, doy + k * st1, tis[:, k])
            raw.append(x)
            if k + 1 in Kd:
                xc = x + Kd[k + 1] * (zs[:, k] - x) * keep_d
                outp.append(xc)
                x = xc if feed else x
            else:
                outp.append(x)
        return torch.stack(raw, 1), torch.stack(outp, 1)

    # ---- the library: the first core's day-n1 errors over the library year (as scripts/diag_island_noise.py)
    lib_path = os.path.join(a.out, f"isn_lib_{a.tag}_{n1}_{lib_year}_s{a.lib_stride}.npz")
    tez = Z[lib_year]
    ld = np.arange(0, len(tez) - L, a.lib_stride)
    if os.path.exists(lib_path):
        f = np.load(lib_path)
        lib, lib_d = torch.from_numpy(f["err"].astype(np.float32)), f["day"]
        assert len(lib) == len(ld)
    else:
        errs, t0 = [], time.time()
        for i in range(0, len(ld), B0 * M):
            j = ld[i:i + B0 * M]
            _, x0, doy, tis, pv = batch(tez, j, B0 * M)
            with torch.no_grad():
                start(x0, doy, pv)
                y = run_days(core, x0, doy, range(n1), tis)[:len(j), -1].cpu()
            errs.append((y - tez[j + n1][:, DYN]) * keep)
            print(f"library {i + len(j)} / {len(ld)} {time.time() - t0:.0f} s", flush=True)
        lib, lib_d = torch.cat(errs), ld + n1
        np.savez(lib_path, err=lib.numpy().astype(np.float16), day=lib_d)
    lib_rms = torch.sqrt((lib ** 2).mean(0))
    print(f"library {lib_year}: {len(lib)} patterns, rms {float(lib_rms[sc].mean()):.3f} state std", flush=True)

    ktag = ("_d" + "".join(str(d) for d in early) + ("o" if a.kal_open else "")) if early else ""
    if a.kalman and a.kalman != "kai":   # another measurement model: its own gain table and run names
        ktag += "_" + a.kalman.replace("_", "")
    out = {"tag": a.tag, "stage2": a.stage2, "kalman": a.kalman, "gain": a.gain, "robust": a.robust, "n1": n1,
           "members": M, "shape": a.shape, "year": year, "lib_year": lib_year, "lib": int(len(lib))}
    g = lambda v, d=3: " ".join(f"{x:.{d}f}" for x in v)  # noqa: E731
    K = var_d = None
    post = torch.ones(1, len(DYN), 1, 1)
    Ke = {}
    if kai is not None:
        # ---- the gain: error statistics of the island in the library year, per channel and latitude
        zl = kai_days(tez, ld)
        ezl = (zl - torch.stack([tez[ld + k + 1][:, DYN] for k in range(n1)], 1)) * keep5
        ez, lib_k = ezl[:, -1], lib
        m3 = lambda v: v.double().mean((0, -1), keepdim=True)  # noqa: E731  (1, 66, H, 1)
        if early:   # the gains of the days before the island, in sequence (every day sees the corrections before it)
            def lib_run(Kd):
                """The first core's errors of day 1 .. n1 over the library year with the corrections Kd."""
                errs = []
                for i in range(0, len(ld), B0 * M):
                    j = ld[i:i + B0 * M]
                    jp, x0, doy, tis, pv = batch(tez, j, B0 * M)
                    zc = zl[i:i + len(j)]
                    zc = torch.cat([zc, zc[-1:].expand(B0 * M - len(j), *zc.shape[1:])]).to(dev)
                    with torch.no_grad():
                        start(x0, doy, pv)
                        raw = stage1(x0, doy, tis, zc, {d: v.to(dev) for d, v in Kd.items()})[0][:len(j)].cpu()
                    errs.append((raw - torch.stack([tez[j + k + 1][:, DYN] for k in range(n1)], 1)) * keep5)
                return torch.cat(errs)

            def gain_of(ef, ek):
                S_, R_, c_ = m3(ef ** 2), m3(ek ** 2), m3(ef * ek)
                Kd = S_ / (S_ + R_).clamp_min(1e-12) if a.gain == "plain" else (S_ - c_) / (S_ + R_ - 2 * c_).clamp_min(1e-12)
                return Kd.clamp(0, 1).float(), float(np.sqrt((S_ * w.view(1, 1, -1, 1).double()).mean((0, 2, 3)).numpy()[sc]).mean())
            t0, el, rms_e = time.time(), lib_run({}), {}
            for d in early:
                if feed and Ke:
                    el = lib_run(Ke)
                Ke[d], rms_e[d] = gain_of(el[:, d - 1], ezl[:, d - 1])
                print(f"gain of day {d}: K {float((Ke[d].double() * w.view(1, 1, -1, 1)).mean((0, 2, 3))[sc].mean()):.3f}, "
                      f"core rms {rms_e[d]:.3f}, {time.time() - t0:.0f} s", flush=True)
            lib_k = (lib_run(Ke) if feed else el)[:, -1]   # the island's errors after the earlier corrections
            out["kal_days"], out["kal_open"] = early, bool(a.kal_open)
            out["early_K"] = {str(d): float((Ke[d].double() * w.view(1, 1, -1, 1)).mean((0, 2, 3))[sc].mean()) for d in early}
        S, Rv, c = m3(lib_k ** 2), m3(ez ** 2), m3(lib_k * ez)
        var_d = (S + Rv - 2 * c).clamp_min(1e-12)
        K = (S / (S + Rv).clamp_min(1e-12) if a.gain == "plain" else (S - c) / var_d).clamp(0, 1)
        Pa = (1 - K) ** 2 * S + K ** 2 * Rv + 2 * K * (1 - K) * c
        post = torch.sqrt((Pa / S.clamp_min(1e-12)).clamp(0, 1)).float()
        wl = w.view(1, 1, -1, 1).double()
        am = lambda v: (v * wl).mean((0, 2, 3)).numpy()  # noqa: E731  area mean per channel
        info = 0.5 * torch.log(S.clamp_min(1e-12) / Pa.clamp_min(1e-12))
        rho = c / torch.sqrt(S * Rv).clamp_min(1e-12)
        # the table of the island (state units): for --joint-kalman of scripts/run_kai_tendency.py and --gain isl
        np.savez(os.path.join(a.out, f"kal_tab_{a.tag}_{n1}_{lib_year}_s{a.lib_stride}{ktag}.npz"), S=S[0, :, :, 0].numpy(),
                 R=Rv[0, :, :, 0].numpy(), c=c[0, :, :, 0].numpy(), rho=rho[0, :, :, 0].numpy())
        out["gain_stats"] = {"K": float(am(K)[sc].mean()), "rho": float(am(rho)[sc].mean()),
                             "S_over_R": float((am(S)[sc] / am(Rv)[sc]).mean()),
                             "Pa_over_S": float((am(Pa)[sc] / am(S)[sc]).mean()), "info_nats": float(am(info)[sc].mean()),
                             "vars": {v: dict(K=float(am(K)[q]), rho=float(am(rho)[q]), rmse_core=float(np.sqrt(am(S)[q]) * s),
                                              rmse_kai=float(np.sqrt(am(Rv)[q]) * s), rmse_post=float(np.sqrt(am(Pa)[q]) * s),
                                              info_nats=float(am(info)[q])) for v, q, s in zip(VARS, vi, vstd)}}
        gs = out["gain_stats"]
        print(f"== the island's gain from {lib_year} ({a.gain}): K {gs['K']:.3f}, error correlation {gs['rho']:.3f}, "
              f"S / R {gs['S_over_R']:.3f}, P_a / S {gs['Pa_over_S']:.3f}, entropy taken out {gs['info_nats']:.4f} nats")
        for v in VARS:
            q = gs["vars"][v]
            print(f" {v:5s} K {q['K']:.3f} rho {q['rho']:.3f} RMSE core {q['rmse_core']:.2f} KAI {q['rmse_kai']:.2f} "
                  f"corrected (expected) {q['rmse_post']:.2f} info {q['info_nats']:.4f}", flush=True)
        K, var_d, Pa_s = K.float().to(dev), var_d.float().to(dev), Pa.float().to(dev)
        Rv_f, rho_f = Rv.float().to(dev), rho.float().clamp(-0.99, 0.99).to(dev)
        Ke = {d: v.to(dev) for d, v in Ke.items()}
        del zl, ezl
    if a.table_only:
        json.dump(out, open(a.json or os.path.join(a.out, f"kal_tab_{a.tag}_{n1}_{lib_year}_s{a.lib_stride}{ktag}.json"), "w"), indent=1)
        print("table only: done", flush=True)
        return
    post = post.to(dev)

    tez = Z[year]
    ics = np.arange(0, len(tez) - L, a.ic_stride)
    if a.max_ics and len(ics) > a.max_ics:
        ics = ics[::int(np.ceil(len(ics) / a.max_ics))]
    N = len(ics)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)

    def patterns(j):
        o = []
        for d in j + n1:
            if a.shape == "white":
                xi = torch.randn((M // 2, len(DYN)) + tuple(lib.shape[-2:]), generator=gen)
                xi = F.avg_pool2d(F.pad(F.pad(xi, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="reflect"), 3, 1) * 3.0
                o.append(xi * lib_rms * keep)
                continue
            dd = np.abs(lib_d - d)
            dd = np.minimum(dd, 365 - dd)
            ok = dd <= a.season
            if lib_year == year:
                ok &= np.abs(lib_d - d) >= a.gap
            o.append(lib[rng.choice(np.flatnonzero(ok), M // 2, replace=False)])
        return torch.stack(o)

    def sums(fc, Tb, Cb):
        pa, ta = fc - Cb, Tb - Cb
        return torch.stack([wm(v).sum(0) for v in ((pa - ta) ** 2, pa ** 2, ta ** 2, pa * ta)])

    def score(S_):
        mse, pp, tt, pt = (v.numpy() for v in S_ / N)
        acc_ = pt / np.sqrt(pp * tt)
        return dict(nrmse=np.sqrt(mse)[:, sc].mean(1).tolist(), acc=acc_[:, sc].mean(1).tolist(),
                    act=np.sqrt(pp / tt)[:, sc].mean(1).tolist(),
                    vars={v: dict(rmse=(np.sqrt(mse[:, q]) * s).tolist(), acc=acc_[:, q].tolist(),
                                  act=np.sqrt(pp[:, q] / tt[:, q]).tolist()) for v, q, s in zip(VARS, vi, vstd)})

    base = a.tag + (f"+{a.stage2}" if a.stage2 else "")
    kal = (f"-K{a.gain}" + (f"t{a.robust:g}" if a.robust else "") + ktag.replace("_", "")) if kai is not None else ""
    INC = None
    if a.isl_inc:
        assert kai is not None
        INC = np.load(a.isl_inc, mmap_mode="r")
        assert INC.shape[0] == N, (INC.shape, N)
        kal += "+inc" + os.path.basename(a.isl_inc).split(".")[0].replace("isl_inc_", "").rsplit("_", 1)[0]
    names = {amp: f"isn-{base}{kal}-{a.shape}{amp:g}-m{M if amp else 1}" for amp in a.amps}
    kinds = [f"m{2 * k}" for k in range(1, M // 2 + 1)] + ["one"]
    acc = {"central": 0.0, "kai": 0.0, **{(amp, k): 0.0 for amp in a.amps for k in kinds}}
    var = {amp: 0.0 for amp in a.amps}
    nans = {"central": 0, **{amp: 0 for amp in a.amps}}
    wsum, ksum, t0 = 0.0, 0.0, time.time()
    H_, W_ = Z[year][0].shape[-2:]
    spec = a.tag + (f"-{a.stage2}_{n1}" if a.stage2 else "")
    path = os.path.join(a.out, f"fc_{spec}_{year}.npy")
    save = not a.no_save and not a.max_ics
    shp = (N, L, len(DYN), H_, W_)
    # the forecast without the correction: written when it is not cached yet (else compared with the cache)
    wc = CacheWriter(path, shp) if save and not a.no_central and not os.path.exists(path) else None
    cmp_blocks = [] if save and not a.no_central and os.path.exists(path) else None
    wr = {amp: CacheWriter(os.path.join(a.out, f"fc_{names[amp]}_{year}.npy"), shp)
          for amp in a.amps if save and not a.no_runs}

    def run2(x, doy_, tis_, fix=None):
        """The days after the island from x; an entropy second stage starts from ``fix`` = (log S, variance) or
        reads the island's entropy off x."""
        if ent2:
            core2.reset_entropy()
            core2.isl_fix = fix
        return run_days(cont, x, doy_, range(n1, L), tis_)
    for i in range(0, N, B0):
        j = ics[i:i + B0]
        nb = len(j)
        jp, x0, doy, tis, pv = batch(tez, j, B0)
        Tb = torch.stack([tez[j + k + 1][:, DYN] for k in range(L)], 1)
        Cb = torch.stack([cz[(torch.tensor(j) + k + 1) % len(cz)] for k in range(L)], 1)
        P1 = None
        if any(a.amps):
            P1 = patterns(j)
            P1 = torch.cat([P1, P1[-1:].expand(B0 - nb, *P1.shape[1:])]).to(dev)
        with torch.no_grad():
            n0 = nan()
            start(x0, doy, pv)
            ya = ya0 = run_days(core, x0, doy, range(n1), tis)                   # the prediction up to the island
            state = get_state(core) if Wc else None
            if W2:
                hand_memory(core, cont)
            cen = torch.cat([ya, run2(ya[:, -1], doy, tis)], 1)[:nb].cpu()
            nans["central"] += nan() - n0
            acc["central"] = acc["central"] + sums(cen, Tb, Cb)
            if wc is not None:
                wc.write(cen.numpy().astype(np.float16))
            elif cmp_blocks is not None and len(cmp_blocks) < 4:   # the first blocks are enough for the comparison
                cmp_blocks.append(cen.numpy().astype(np.float16))
            xa, fix = ya[:, -1], None
            if kai is not None:
                zs = kai_days(tez, jp).to(dev)
                if early:   # again from the IC, with the corrections of the days before the island
                    start(x0, doy, pv)
                    raw, ya = stage1(x0, doy, tis, zs, Ke)
                    state, xa = get_state(core) if Wc else None, raw[:, -1]
                z = zs[:, -1]                                                    # the correction at the island
                d = (z - xa) * keep_d
                Kx, vd, Pa_x, v0 = K, var_d, Pa_s.expand_as(xa), None
                if ent2:
                    v0 = core2.island_entropy(xa, (doy + n1).float() * 24.0)     # log S(x), unit sd2
                if a.gain == "isl":
                    S_ = torch.exp(v0) * sd2
                    c_ = rho_f * torch.sqrt(S_ * Rv_f)
                    vd = (S_ + Rv_f - 2 * c_).clamp_min(1e-12)
                    Kx = ((S_ - c_) / vd).clamp(0, 1) * keep_d
                    Pa_x = ((1 - Kx) ** 2 * S_ + Kx ** 2 * Rv_f + 2 * Kx * (1 - Kx) * c_).clamp_min(1e-12)
                    ksum = ksum + float(wm(Kx[:nb].cpu())[:, sc].mean()) * nb
                if ent2:
                    fix = (v0, (Pa_x / sd2).clamp_min(1e-8))
                wt = (a.robust + 1) / (a.robust + d ** 2 / vd) if a.robust else torch.ones_like(d)
                wsum = wsum + float(wt[:nb].mean()) * nb
                xa = xa + Kx * wt * d
                if INC is not None:
                    xa[:nb] = xa[:nb] + torch.from_numpy(np.asarray(INC[i:i + nb], np.float32)).to(dev) * keep_d
                zz = cen.clone()
                zz[:, :n1] = (zs.cpu() * keep5 + ya0.cpu() * (1 - keep5))[:nb]
                acc["kai"] = acc["kai"] + sums(zz, Tb, Cb)                       # only its days 1 .. n1 are KAI's
            head = torch.cat([ya[:, :n1 - 1], xa[:, None]], 1)[:nb].cpu()        # day 1 .. n1, the island corrected
            for amp in a.amps:
                n0 = nan()
                Me = M if amp else 1
                rep = lambda v, Me=Me: v.repeat_interleave(Me, 0)                # noqa: E731
                pert = (torch.stack([P1, -P1], 2).reshape(B0 * M, *P1.shape[2:]) * amp * post if amp
                        else torch.zeros_like(xa))
                if Wc:  # the correction is an innovation for the memory, the noise is its own state
                    set_state(cont, state, Me, pert)
                y = run2(rep(xa) + pert, rep(doy), rep(tis), None if fix is None else tuple(rep(v) for v in fix))
                y = y.view(B0, Me, L - n1, len(DYN), *x0.shape[-2:])[:nb].cpu()
                nans[amp] += nan() - n0
                full = lambda v: torch.cat([head, v], 1)                         # noqa: E731
                for k in range(1, M // 2 + 1):
                    acc[amp, f"m{2 * k}"] = acc[amp, f"m{2 * k}"] + sums(full(y[:, :min(2 * k, Me)].mean(1)), Tb, Cb)
                acc[amp, "one"] = acc[amp, "one"] + sums(full(y[:, 0]), Tb, Cb)
                if Me > 1:
                    var[amp] = var[amp] + wm(y.var(1, unbiased=True)).sum(0)
                if wr.get(amp) is not None:
                    wr[amp].write(full(y.mean(1)).numpy().astype(np.float16))
        print(f"{i + nb} / {N} {time.time() - t0:.0f} s, failed dycore states {sum(nans.values())}", flush=True)

    out.update(ics=int(N), central=score(acc["central"]), nan_central=nans["central"], runs={})
    c = out["central"]
    if wc is not None:   # the plain forecast in the cache format of diag_stage_filter.py
        wc.close()
    elif cmp_blocks:
        ref = np.load(path, mmap_mode="r")
        nb_ = sum(len(b) for b in cmp_blocks)
        out["central_vs_cache"] = (float(np.abs(np.concatenate(cmp_blocks).astype(np.float32) - np.asarray(ref[:nb_], np.float32)).max())
                                   if ref.shape[1:] == cmp_blocks[0].shape[1:] else None)
        print(f"run without correction vs the cached forecast (first {nb_} ICs), max |difference|:", out["central_vs_cache"])
    print(f"== {base} without correction, day 1..{L} ({N} ICs of {year})\n nRMSE {g(c['nrmse'])}\n ACC   {g(c['acc'])}"
          f"\n act   {g(c['act'], 2)}", flush=True)
    if kai is not None:
        kz = out["kai_island"] = score(acc["kai"])
        out["robust_weight"] = wsum / N
        if a.gain == "isl":
            out["K_isl"] = ksum / N
            print(f"== the gain from the island's entropy: mean K(x) {ksum / N:.3f} (table of {lib_year}: {out['gain_stats']['K']:.3f})")
        print(f"== day {n1} (the island): nRMSE core {c['nrmse'][n1 - 1]:.4f}, KAI {kz['nrmse'][n1 - 1]:.4f}; "
              f"ACC core {c['acc'][n1 - 1]:.4f}, KAI {kz['acc'][n1 - 1]:.4f}; mean robust weight {wsum / N:.3f}")
        print(f"== KAI alone, day 1..{n1}: nRMSE {g(kz['nrmse'][:n1])} | ACC {g(kz['acc'][:n1])}")
        if early:
            print(f"== corrections on day {early} ({'output only' if a.kal_open else 'fed back to the core'}), mean K "
                  + " ".join(f"{out['early_K'][str(d)]:.3f}" for d in early))
    for amp in a.amps:
        name = names[amp]
        r = {k: score(acc[amp, k]) for k in kinds}
        r["mean"], r["nan_states"] = r[f"m{M}"], nans[amp]
        m = r["mean"]
        if amp:
            mse = (acc[amp, f"m{M}"][0] / N)[n1:, sc].mean(1)
            r["ssr"] = torch.sqrt((1 + 1 / M) * (var[amp] / N)[:, sc].mean(1) / mse).tolist()
        out["runs"][name] = r
        print(f"== {name}: noise amplitude {amp:g} after day {n1}, {r['nan_states']} failed dycore states")
        print(f" nRMSE {g(m['nrmse'])}\n ACC   {g(m['acc'])}\n act   {g(m['act'], 2)}")
        if amp:
            print(f" spread / error, day {n1 + 1}..{L}: {g(r['ssr'], 2)}")
        print(" gain over the run without correction (%, + = better) "
              + " ".join(f"{100 * (1 - x / y):+.1f}" for x, y in zip(m["nrmse"], c["nrmse"])))
        print(" ACC difference " + " ".join(f"{x - y:+.3f}" for x, y in zip(m["acc"], c["acc"])))
        if amp:
            for k in kinds[:-2]:
                print(f" {k[1:]} members: nRMSE {g(r[k]['nrmse'][n1:])} | ACC {g(r[k]['acc'][n1:])} | act {g(r[k]['act'][n1:], 2)}")
            print(f" one member: nRMSE {g(r['one']['nrmse'][n1:])} | ACC {g(r['one']['acc'][n1:])} | act {g(r['one']['act'][n1:], 2)}")
        for q in VARS:
            print(f" {q:5s} RMSE {g(m['vars'][q]['rmse'], 2)} | without {g(c['vars'][q]['rmse'], 2)}")
            print(f"       ACC  {g(m['vars'][q]['acc'])} | without {g(c['vars'][q]['acc'])}")
        sys.stdout.flush()
        if wr.get(amp) is not None:
            wr[amp].close()
    js = a.json or os.path.join(a.out, f"diag_island_kalman_{base}{kal}_{a.shape}.json")
    json.dump(out, open(js, "w"), indent=1)
    np.savez(js[:-5] + "_sums.npz", central=(acc["central"] / N).numpy(),
             **({"kai": (acc["kai"] / N).numpy()} if kai is not None else {}),
             **{f"{names[amp]}__{k}": (acc[amp, k] / N).numpy() for amp in a.amps for k in kinds},
             **{f"{names[amp]}__var": (var[amp] / N).numpy() for amp in a.amps if amp})


if __name__ == "__main__":
    main()
