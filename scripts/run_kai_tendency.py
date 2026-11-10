"""Where does the separately trained tendency core go in KAI-alpha? (docs/kai_tendency.md)

Data: KAI's contract (67 channels, daily mean of 00/06/12/18 UTC, z-score with std + 1e-8, no cross-year
pairs), made from our 6-hourly WB2 files (scripts/wb2_to_kai.py --all-hours) at 5.625 deg (32 x 64).

1. ``--arm core``: the tendency core alone (pgcfm.dyncore.DynCore, step 16 recipe: Heun, tisr forcing,
   PositiveMass, rollout curriculum, latitude-weighted MSE in units of the increment std). Saved, then frozen.
2. ``--arm <position>``: KAI-alpha with KAI's own recipe (single step, LpLoss, AdamW 1e-3, cosine, clip 1.0)
   and the frozen core at one position (pgcfm/kai_tend.py ARMS):
       kai    original, no core                      res    x + s net(x), no core (control of ``out``)
       in     tau as extra input channels            inres  res with tau at the input
       s1-s4  tau added after stage 1..4             out    core forecast + s net(x)
       gate   KAI's own output mixed with the core   inout  out with tau at the input
              forecast, one weight per channel (--init kai: the trained KAI is kept, g starts at 0)
   ``--init <tag> --rollout 2 3``: fine-tuning in the model's own rollout (the gradient passes through the
   frozen core's input).
3. ``--score <tags>``: paired rollouts on the test year: RMSE / ACC per channel and lead, zonal spectra,
   bootstrap vs ``kai``; plus ``core`` alone, ``blend`` (lead-wise blend of the kai and core forecasts, weights
   from the validation year) and, for every arm with an injected tau, the same rollout with tau = 0
   (``<arm>~notau``: how much the trained KAI relies on the core at that position) and with the core running
   open loop on its own trajectory (``<arm>~open``: is a diverging rollout the KAI <-> core feedback?).

    python scripts/run_kai_tendency.py --arm core --iters 3000
    python scripts/run_kai_tendency.py --arm s2
    python scripts/run_kai_tendency.py --score kai in s1 s2 s3 s4 res inres out inout
"""

import argparse
import json
import os
import sys
import time

import netCDF4 as nc
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract  # noqa: E402
from pgcfm.kai_tend import (ARMS, DYN, CoreBase, KAIPos, KAIPosConfig, LpLoss, OROG, TISR, config_dict,  # noqa: E402
                            core_rollout, make_core, needs_core)

EPS = contract.KAI_STD_EPS


def daily(path6h: str, cache: str) -> np.ndarray:
    """(D, 67, H, W) daily means (KAI: mean of 00/06/12/18 UTC) of one 6-hourly year file, cached as .npy."""
    if os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    with nc.Dataset(path6h) as d:
        f = d["fields"]
        hours = np.asarray(d["time"][:])
        T = f.shape[0]
        assert hours[0] == 0 and T % 4 == 0 and np.all(np.diff(hours) == 6), path6h
        f.set_auto_mask(False)
        out = np.empty((T // 4,) + f.shape[1:], dtype=np.float32)
        for i in range(0, T, 400):  # read in large slabs (zlib chunks), average every 4 times
            x = np.asarray(f[i:i + 400], dtype=np.float32)
            out[i // 4:(i + len(x)) // 4] = x.reshape(-1, 4, *x.shape[1:]).mean(1)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache + ".tmp.npy", out)
    os.replace(cache + ".tmp.npy", cache)
    return np.load(cache, mmap_mode="r")


def lat_of(path6h):
    with nc.Dataset(path6h) as d:
        for k in ("latitude", "lat"):
            if k in d.variables:
                return np.asarray(d[k][:], dtype=np.float64)
    raise KeyError("latitude")


def load_years(a, years):
    return {y: np.asarray(daily(os.path.join(a.data, f"{y}.nc"), os.path.join(a.cache, f"{y}.npy"))) for y in years}


def stats(a, train=None):
    p = os.path.join(a.out, "stats.npz")
    if os.path.exists(p):
        s = np.load(p)
        return s["mean"], s["std"], s["inc"], s["clim"]
    allx = np.concatenate([train[y] for y in sorted(train)], 0).astype(np.float64)
    mean, std = allx.mean((0, 2, 3)), allx.std((0, 2, 3))
    z = lambda x: (x - mean[None, :, None, None]) / (std[None, :, None, None] + EPS)  # noqa: E731
    inc = np.sqrt(np.mean(np.concatenate([np.mean(np.diff(z(train[y]), axis=0) ** 2, (2, 3)) for y in train], 0), 0))
    inc = np.where(inc > 0, inc, 1.0)
    # daily climatology (physical units) for ACC: mean per day of year over the training years, 31-day smoothing
    clim = np.zeros((366,) + allx.shape[1:], dtype=np.float64)
    cnt = np.zeros(366)
    for y in train:
        n = len(train[y])
        clim[:n] += train[y]
        cnt[:n] += 1
    for i in range(366):
        clim[i] = clim[i] / cnt[i] if cnt[i] else clim[i - 1]  # 29 Feb / 31 Dec of a leap year: neighbour
    ker = np.ones(31) / 31
    pad = np.concatenate([clim[-15:], clim, clim[:15]], 0)
    clim = np.stack([np.tensordot(ker, pad[i:i + 31], 1) for i in range(366)]).astype(np.float32)
    np.savez(p, mean=mean, std=std, inc=inc, clim=clim)
    return mean, std, inc, clim


def zscore(x, mean, std):
    return ((x - mean[None, :, None, None]) / (std[None, :, None, None] + EPS)).astype(np.float32)


def build_core(a, mean, std, inc, lat, load=True, tag=None):
    """The tendency core ``tag`` (default --core-tag): built from the options saved in its checkpoint, frozen.
    ``load=False``: a new core from the command line options (training)."""
    import xarray as xr
    from pgcfm.core import static_features
    with xr.open_dataset(os.path.join(a.era5, "constants", "constants_5.625deg.nc")) as d:
        oro, lsm, lon = d["orography"].values.astype(np.float32), d["lsm"].values.astype(np.float32), d["lon"].values
        assert np.allclose(d["lat"].values, lat)
    contract.register_latitudes(lat)
    opts = dict(hidden=a.core_hidden, blocks=list(a.core_blocks), norm=a.core_norm, substeps=a.core_substeps,
                stride=a.core_stride)
    if a.core_hist:
        opts["hist"] = True
    if a.core_mem:
        opts.update(mem=a.core_mem, mem_ch=a.core_mem_ch)
        if a.core_mz:
            opts["mz"] = True
    ck = None
    if load:
        ck = torch.load(os.path.join(a.out, f"{tag or a.core_tag}.pt"), map_location="cpu", weights_only=False)
        opts = ck.get("core_opts", dict(hidden=a.core_hidden, blocks=list(a.core_blocks), norm="bn", substeps=1))
    core = make_core(static_features(lat, lon, oro, lsm), inc, mean, std, lat, **opts)
    core.opts = opts
    if load:
        core.load_state_dict(ck["state"])
        core.eval()
        for p in core.parameters():
            p.requires_grad_(False)
    return core


def build(a, arm, inc, core):
    """``core``: the frozen DynCore (or None for arms without one). Arm ``stack`` loads the trained arm ``res``
    as its frozen first stage instead (the control: is the gain of ``out`` specific to the tendency core?)."""
    base = None
    if arm == "stack":
        base = build(a, "res", inc, None)
        base.load_state_dict(torch.load(os.path.join(a.out, f"{a.stack_tag}.pt"), map_location="cpu",
                                        weights_only=False)["state"])
        base.eval()
        for p in base.parameters():
            p.requires_grad_(False)
    elif needs_core(arm):
        base = CoreBase(core)
    return KAIPos(KAIPosConfig.arm(arm, a.depths, a.dims), inc, base)


def train(a, arm):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tr = load_years(a, a.train_years)
    va = load_years(a, [a.val_year])[a.val_year]
    mean, std, inc, _ = stats(a, tr)
    lat = lat_of(os.path.join(a.data, f"{a.val_year}.nc"))
    trz = {y: torch.from_numpy(zscore(tr[y], mean, std)) for y in tr}
    vaz = torch.from_numpy(zscore(va, mean, std))
    is_core = arm == "core"
    dyn = torch.tensor(DYN)
    st = a.core_stride if is_core else 1  # days per training step (long-scale core: > 1)
    core = None if is_core or not needs_core(arm) or arm == "stack" else build_core(a, mean, std, inc, lat)
    # does the model see the state one step before (the observed trend)? a history core, alone or inside KAI,
    # or arm kaih
    hist = (a.core_hist or a.core_mem > 0) if is_core else (ARMS[arm][1] == "hist" or getattr(core, "hist", False))
    W = a.core_mem if is_core else 0  # memory core: the W steps before the batch's start warm the memory up
    if is_core:
        if st > 1:  # the tendency unit of a long-scale core: std of the st-day increment
            inc = np.sqrt(np.mean(np.stack([((trz[y][st:] - trz[y][:-st]) ** 2).mean((0, 2, 3)).numpy() for y in trz]), 0))
            inc = np.where(inc > 0, inc, 1.0)
        m = build_core(a, mean, std, inc, lat, load=False)
        opt = torch.optim.AdamW(m.parameters(), lr=a.core_lr, weight_decay=1e-3)  # step 16
        cur = a.curriculum
        s_dyn = torch.as_tensor(inc[DYN], dtype=torch.float32).view(1, 1, -1, 1, 1)
        scd = torch.tensor([i for i, c in enumerate(DYN) if c != TISR])  # the 65 scored channels among the 66
        keep_dyn = torch.ones(len(DYN)).index_fill(0, torch.tensor([DYN.index(TISR)]), 0.0).view(1, 1, -1, 1, 1)
        orth_log = None
        wl = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)
        wl = (wl / wl.mean()).view(1, 1, 1, -1, 1)
    else:
        m = build(a, arm, inc, core)
        if a.init:  # fine-tuning (e.g. --rollout 2 3 from a single-step checkpoint)
            miss = m.load_state_dict(torch.load(os.path.join(a.out, f"{a.init}.pt"), map_location="cpu",
                                                weights_only=False)["state"], strict=False)
            assert set(miss.missing_keys) <= {"dyn", "gate"} and not miss.unexpected_keys, miss
        cur = a.rollout  # [1] = KAI's single-step training
        m.core_grad = max(cur) > 1
        gate = [q for k, q in m.named_parameters() if k == "gate"]
        rest = [q for k, q in m.named_parameters() if k != "gate"]
        opt = torch.optim.AdamW([dict(params=rest)] + ([dict(params=gate, lr=a.gate_lr, weight_decay=0.0)] if gate else []),
                                lr=a.lr)  # KAI's trainer; the 67 gate weights move on their own scale
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    lp = LpLoss(d=2, p=2, size_average=True, reduction=True)
    years = sorted(trz)
    log, t0 = [], time.time()

    def batch(K):
        ys = rng.choice(years, a.batch)
        xs, tg, doy, pv = [], [], [], []
        for y in ys:
            d = int(rng.integers(W * st if W else st if hist else 0, len(trz[y]) - K * st))
            xs.append(trz[y][d])
            tg.append(trz[y][d + st:d + st * K + 1:st])
            pv.append(trz[y][d - W * st:d:st] if W else trz[y][d - st] if hist else None)
            doy.append(d)
        prev.clear()
        prev.append(torch.stack(pv) if hist else None)
        return torch.stack(xs), torch.stack(tg), torch.tensor(doy)

    prev = [None]  # history cores: the state one step before the batch's start (memory cores: W steps, oldest first)

    def pdyn(pv):
        return None if pv is None else pv.index_select(-3, dyn)

    def one_step(x, doy, t, pv=None):
        if is_core:
            return x.index_copy(1, dyn, core_rollout(m, x[:, dyn], doy, 1, t[:, None, TISR], prev=pdyn(pv))[:, 0])
        return m(x, doy, t[:, TISR], prev=pv)

    def val():  # KAI's validation loss (single step, relative L2) on 64 days of the validation year
        m.eval()
        with torch.no_grad():
            idx = np.arange(st if hist else 0, len(vaz) - st, max(1, (len(vaz) - st) // 64))[:64]
            tot = 0.0
            for i in range(0, len(idx), 16):
                j = idx[i:i + 16]
                pv = vaz[j - st] if hist else None
                if W:  # the same days as a history core; before the first day of the year: that day itself
                    pv = torch.stack([vaz[np.maximum(j - k * st, 0)] for k in range(W, 0, -1)], 1)
                tot += float(lp(one_step(vaz[j], torch.tensor(j), vaz[j + st], pv), vaz[j + st])) * len(j)
        m.train()
        return tot / len(idx)

    for it in range(1, a.iters + 1):
        K = cur[min(len(cur) - 1, (it - 1) * len(cur) // a.iters)]
        x, tg, doy = batch(K)
        if is_core:
            y = core_rollout(m, x[:, dyn], doy, K, tg[:, :, TISR], prev=pdyn(prev[0]))
            err = (y - tg[:, :, dyn]) / s_dyn
            if getattr(m, "mz", False):
                # Mori-Zwanzig form on the sequence: the W days before the batch's start are forecast one by one
                # (first from an empty memory = the Markov term, then with the memory integral up to that day)
                # and enter the loss with the rollout; W + K supervised positions
                seq_true = torch.cat([pdyn(prev[0])[:, 1:], x[:, None, dyn]], 1)
                es = (m.seq - seq_true) / s_dyn
                loss = ((torch.cat([es, err], 1) ** 2) * wl).mean()
                # the noise term is orthogonal to the past: consecutive one-step residuals of the sequence (the W
                # forecasts and the first rollout day, all started from data) must not be correlated at lag 1, 2
                r = torch.cat([es, err[:, :1]], 1) * keep_dyn
                orth = []
                for lag in (1, 2):
                    u, v = r[:, lag:], r[:, :-lag]
                    c = (u * v * wl).sum((-2, -1)) / torch.sqrt((u ** 2 * wl).sum((-2, -1)) * (v ** 2 * wl).sum((-2, -1)) + 1e-12)
                    orth.append(c.mean((0, 1)))  # (66,): mean over the batch and the pairs of the sequence
                orth_log = [float(o.detach()[scd].mean()) for o in orth]
                if a.mz_orth > 0:
                    loss = loss + a.mz_orth * sum((o[scd] ** 2).mean() for o in orth)
            else:
                loss = ((err ** 2) * wl).mean()
            if a.core_zm > 0:  # the zonal mean of the error on its own (a drift is small in the full MSE)
                loss = loss + a.core_zm * ((err.mean(-1, keepdim=True) ** 2) * wl).mean()
        elif K == 1:
            loss = lp(m(x, doy, tg[:, 0, TISR], prev=prev[0]), tg[:, 0])
        else:  # the connected model in its own rollout (the frozen core runs on KAI's states)
            y = m.rollout(x, doy, K, tg[:, :, TISR], prev=prev[0])
            loss = sum(lp(y[:, k], tg[:, k]) for k in range(K)) / K
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % a.log_every == 0 or it == a.iters:
            log.append(dict(it=it, K=K, loss=float(loss.detach()), val=val(), sec=time.time() - t0))
            if is_core and orth_log is not None:  # residual correlation at lag 1, 2 in the last batch's sequences
                log[-1]["res_corr"] = [round(o, 4) for o in orth_log]
            print(json.dumps(log[-1]), flush=True)
    torch.save(dict(state=m.state_dict(), cfg=config_dict(m.cfg), log=log, args=vars(a),
                    core_opts=getattr(m, "opts", None),
                    params=sum(p.numel() for p in m.parameters()), train_seconds=time.time() - t0),
               os.path.join(a.out, f"{a.tag or arm}.pt"))


# ---------------------------------------------------------------------------------------------- scoring
def simplex2(b1, b2, g11, g22, g12):
    """argmin over w1, w2 >= 0, w1 + w2 <= 1 of |r - w1 d1 - w2 d2|^2 from the moments b_i = <d_i r>,
    g_ij = <d_i d_j> (arrays): the optimum of a convex problem on a triangle lies inside or on an edge."""
    eps = 1e-30
    det = g11 * g22 - g12 ** 2
    full = np.stack([(b1 * g22 - b2 * g12) / np.maximum(det, eps), (b2 * g11 - b1 * g12) / np.maximum(det, eps)])
    ok = (full >= 0).all(0) & (full.sum(0) <= 1) & (det > eps)
    t = np.clip((b1 - b2 - g12 + g22) / np.maximum(g11 - 2 * g12 + g22, eps), 0, 1)
    z = np.zeros_like(b1)
    cands = [np.stack([z, z]), np.stack([np.clip(b1 / np.maximum(g11, eps), 0, 1), z]),
             np.stack([z, np.clip(b2 / np.maximum(g22, eps), 0, 1)]), np.stack([t, 1 - t]), np.where(ok, full, 0.0)]
    cost = np.stack([-2 * (c[0] * b1 + c[1] * b2) + c[0] ** 2 * g11 + c[1] ** 2 * g22 + 2 * c[0] * c[1] * g12
                     for c in cands])
    return np.take_along_axis(np.stack(cands), cost.argmin(0)[None, None], 0)[0].astype(np.float32)


def score(a, tags):
    mean, std, inc, clim = stats(a)
    te = load_years(a, [a.test_year])[a.test_year]
    lat = lat_of(os.path.join(a.data, f"{a.test_year}.nc"))
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None].astype(np.float32)
    L = a.leads
    regions = [np.abs(lat) < 20, lat >= 20, lat <= -20]
    sd = (std + EPS).astype(np.float32)[None, None, :, None, None]
    mn = mean.astype(np.float32)[None, None, :, None, None]
    cores, dyn = {}, torch.tensor(DYN)

    def get_core(tag):
        if tag not in cores:
            cores[tag] = build_core(a, mean, std, inc, lat, tag=tag)
        return cores[tag]

    def setup(year_x):
        ics = np.arange(0, len(year_x) - L, a.ic_stride)
        return ics, zscore(year_x, mean, std), np.stack([year_x[i + 1:i + 1 + L] for i in ics])

    def rollout(fn, xz, ics):
        out = []
        with torch.no_grad():
            for i in range(0, len(ics), 16):
                j = ics[i:i + 16]
                tis = torch.from_numpy(np.stack([xz[k + 1:k + 1 + L, TISR] for k in j]))
                pv = torch.from_numpy(xz[np.maximum(j - 1, 0)])  # the day before the IC (first day of the year: itself)
                if getattr(fn, "mem", 0):  # a memory core: the fn.mem days before the IC, oldest first
                    pv = torch.from_numpy(np.stack([xz[np.maximum(j - k, 0)] for k in range(fn.mem, 0, -1)], 1))
                out.append(fn(torch.from_numpy(xz[j]), torch.tensor(j), tis, pv).numpy())
        return (np.concatenate(out) * sd + mn).astype(np.float32)

    def core_fn(tag, warm=None, freeze=False):
        """``warm`` (memory cores, diagnostic): only the last ``warm`` days before the IC enter the memory
        (0 = it starts empty); ``freeze``: the memory is not updated after the IC."""
        core = get_core(tag)
        W = getattr(core, "mem", 0)
        W = W if warm is None else min(W, warm)

        def fn(x0, doy, tis, pv=None):
            if getattr(core, "mem", 0):
                core.freeze_mem = freeze
                pv = pv if W else None
            y = core_rollout(core, x0[:, dyn], doy, L, tis, prev=None if pv is None else pv.index_select(-3, dyn))
            if getattr(core, "mem", 0):
                core.freeze_mem = False
            return x0[:, None].repeat(1, L, 1, 1, 1).index_copy(2, dyn, y)
        fn.mem = W
        return fn

    def load(tag):  # the arm is the part before '_'; the core is the one the checkpoint was trained with
        ck = torch.load(os.path.join(a.out, f"{tag}.pt"), map_location="cpu", weights_only=False)
        arm = tag.split("_")[0]
        core = get_core(ck["args"].get("core_tag", "core")) if needs_core(arm) and arm != "stack" else None
        m = build(a, arm, inc, core)
        missing = m.load_state_dict(ck["state"], strict=False)
        assert set(missing.missing_keys) <= {"dyn"} and not missing.unexpected_keys, missing
        m.eval()
        return m, ck

    ics, tez, truth = setup(te)
    cl = np.stack([clim[[min(i + k, 365) for k in range(1, L + 1)]] for i in ics])
    res = {}

    def metrics(pred, spectra=True):
        err = pred - truth
        out = dict(mse=(err ** 2 * w).mean((-2, -1)).astype(np.float32))  # (N, L, 67)
        # systematic part: the IC-mean of the global-mean error, in units of the channel's std (L, 67)
        out["gbias"] = ((err * w).mean((-2, -1)).mean(0) / (std + EPS)).astype(np.float32)
        pa, ta = pred - cl, truth - cl
        out["acc"] = ((pa * ta * w).sum((-2, -1))
                      / np.sqrt((pa ** 2 * w).sum((-2, -1)) * (ta ** 2 * w).sum((-2, -1)) + 1e-30)).astype(np.float32)
        # the MSE of every forecast split exactly into three terms, on the anomalies (scripts/eval_kai_breakdown.py):
        # mean_w(e^2) = (mean_w e)^2 + (s_f - s_o)^2 + 2 s_f s_o (1 - r)  = bias + amplitude + phase (pattern) error
        mf, mo = (pa * w).mean((-2, -1), dtype=np.float64), (ta * w).mean((-2, -1), dtype=np.float64)
        vf = (pa ** 2 * w).mean((-2, -1), dtype=np.float64) - mf ** 2
        vo = (ta ** 2 * w).mean((-2, -1), dtype=np.float64) - mo ** 2
        cov = (pa * ta * w).mean((-2, -1), dtype=np.float64) - mf * mo
        sf, so = np.sqrt(np.maximum(vf, 0)), np.sqrt(np.maximum(vo, 0))
        out["dec"] = np.stack([(mf - mo) ** 2, (sf - so) ** 2, 2 * (sf * so - cov)], -1).astype(np.float32)
        out["act"] = (sf / (so + 1e-30)).astype(np.float32)  # activity: pattern amplitude of the forecast / truth's
        # the same MSE by region: tropics (rows with |lat| < 20 = 22.5S-22.5N), north and south of it (area weights)
        out["mse_reg"] = np.stack([(err[:, :, :, r] ** 2 * w[r]).mean((-2, -1)) / w[r].mean() for r in regions],
                                  -1).astype(np.float32)
        if spectra:  # zonal power (cos-lat weighted over rows) of prediction, truth and error per wavenumber
            P = lambda f: (np.abs(np.fft.rfft(f, axis=-1)) ** 2 * w).mean(-2).mean(0).astype(np.float32)  # noqa: E731
            out.update(sp_pred=P(pred), sp_true=P(truth), sp_err=P(err))
        return out

    res["persistence"] = metrics(np.repeat(te[ics][:, None], L, 1))
    res["climatology"] = metrics(cl.astype(np.float32))
    keep = {}  # test forecasts kept for the blends
    fns, models = {}, {}  # tag -> forecast function (x0, doy, tisr) -> (B, L, 67, H, W); tag -> KAIPos
    core_tags = a.score_cores if a.score_cores is not None else [a.core_tag]
    for ct in core_tags:
        assert get_core(ct).stride == 1, f"{ct}: a long-scale core is scored through --hier"
        fns[ct] = core_fn(ct)
        keep[ct] = rollout(fns[ct], tez, ics)
        res[ct] = metrics(keep[ct])
        if a.blend_cores is not None and ct not in a.blend_cores:
            del keep[ct]
        ckc = torch.load(os.path.join(a.out, f"{ct}.pt"), map_location="cpu", weights_only=False)
        res[ct].update(train_seconds=ckc["train_seconds"], params=ckc["params"])
        print("scored", ct, flush=True)
        if getattr(get_core(ct), "mem", 0):  # how much of the past does the trained memory use?
            for wm in a.mem_warm:
                res[f"{ct}~w{wm}"] = metrics(rollout(core_fn(ct, warm=wm), tez, ics))
            if a.mem_freeze:
                res[f"{ct}~frozen"] = metrics(rollout(core_fn(ct, freeze=True), tez, ics))
            print("scored", ct, "memory diagnostics", flush=True)
    for tag in tags:
        m, ck = load(tag)
        models[tag] = m
        fns[tag] = lambda x, d, t, pv=None, m=m: m.rollout(x, d, L, t, prev=pv)
        pred = rollout(fns[tag], tez, ics)
        res[tag] = metrics(pred)
        res[tag].update(train_seconds=ck["train_seconds"], params=ck["params"])
        if hasattr(m, "gate"):
            res[tag]["gate"] = m.gate.detach().view(-1).numpy().copy()
        if tag in a.blend_bases or tag in a.blend_with:
            keep[tag] = pred
        if m.cfg.inject != "none":  # reliance on the core at this position: the same rollout with tau = 0
            m.ablate = True
            res[f"{tag}~notau"] = metrics(rollout(lambda x, d, t, pv=None: m.rollout(x, d, L, t, prev=pv), tez, ics),
                                          spectra=False)
            m.ablate = False
            if m.cfg.form == "full" and m.uses_core:  # is a diverging rollout the KAI <-> core feedback? the core on its own trajectory
                res[f"{tag}~open"] = metrics(rollout(
                    lambda x, d, t, pv=None: m.rollout(x, d, L, t, open_loop=True, prev=pv), tez, ics))
        del pred
        print("scored", tag, flush=True)

    def hier_fn(jumps, daily):
        """Hierarchical rollout over time scales: every lead is split greedily into the largest steps first
        (strides of the long-scale cores ``jumps``, then single days by ``daily``: a core or a scored position
        tag), e.g. strides 5, 3, 1: day 9 = 5 + 3 + 1. The state at a lead is one step from an earlier lead."""
        strides = sorted({get_core(j).stride: CoreBase(get_core(j)) for j in jumps}.items(), reverse=True)
        strides.append((1, models[daily] if daily in models else CoreBase(get_core(daily))))

        def fn(x0, doy, tis, pv=None):
            state = {0: x0}
            for lead in range(1, L + 1):
                rem, last = lead, None
                for st, step in strides:  # greedy: the last (smallest) step used decides the parent lead
                    if rem >= st:
                        rem, last = rem % st, (st, step)
                st, step = last
                before = pv if lead == st else state.get(lead - st - 1)  # one day before the step's start
                state[lead] = step(state[lead - st], doy + lead - st, tis[:, lead - 1], prev=before)
            return torch.stack([state[k] for k in range(1, L + 1)], 1)
        return fn

    for pair in a.hier:
        jumps, daily = pair.split(":")
        name = f"hier_{jumps}+{daily}"
        fns[name] = hier_fn(jumps.split("+"), daily)
        pred = rollout(fns[name], tez, ics)
        res[name] = metrics(pred)
        if name in a.blend_with:
            keep[name] = pred
        del pred
        print("scored", name, flush=True)
    done = set()

    def run_cascades():
        """--cascade A>B@n: A's forecast up to day n, then B started from A's day-n state (a position in time)."""
        for spec in a.cascade:
            ab, n = spec.rsplit("@", 1)
            ta, tb = ab.split(">")
            if spec in done or ta not in fns or tb not in fns:
                continue
            done.add(spec)

            def cas_fn(x0, doy, tis, pv=None, fa=fns[ta], fb=fns[tb], n=int(n)):
                ya = fa(x0, doy, tis, pv)
                ts = torch.cat([tis[:, n:], tis[:, -1:].expand(-1, n, -1, -1)], 1)  # padded: only L - n days are used
                yb = fb(ya[:, n - 1], doy + n, ts, ya[:, n - 2] if n >= 2 else x0)
                return torch.cat([ya[:, :n], yb[:, :L - n]], 1)
            name = f"cas_{spec}"
            fns[name] = cas_fn
            pred = rollout(cas_fn, tez, ics)
            res[name] = metrics(pred)
            if name in a.blend_with:
                keep[name] = pred
            del pred
            print("scored", name, flush=True)

    run_cascades()
    # output-level position without training: lead-wise blend of a KAI (``--blend-bases``) with a core or another
    # scored tag, weights from the validation year. Names: blend / blend_<other>, with @<base> if base != kai.
    bases = [b for b in a.blend_bases if b in keep]
    if bases:
        va = load_years(a, [a.val_year])[a.val_year]
        vics, vaz, vtruth = setup(va)
    for bt in bases:
        kai_m = models[bt]
        sfx = "" if bt == "kai" else f"@{bt}"
        vk = rollout(fns[bt], vaz, vics)
        for other in [t for t in core_tags if t in keep] + [t for t in a.blend_with
                                                            if t in keep and t not in core_tags and t != bt]:
            vc = rollout(fns[other], vaz, vics)
            num = ((vk - vc) * (vtruth - vc) * w).mean((0, -2, -1))
            den = ((vk - vc) ** 2 * w).mean((0, -2, -1))
            wk = np.clip(num / np.maximum(den, 1e-30), 0.0, 1.0).astype(np.float32)  # (L, 67) weight of KAI
            del vc
            name = ("blend" if other == "core" else f"blend_{other}") + sfx
            wb = wk[None, :, :, None, None]
            res[name] = metrics(wb * keep[bt] + (1 - wb) * keep[other])
            res[name]["kai_weight"] = wk
            print("scored", name, flush=True)
            if other in a.loop_blend:  # the same two models coupled every step: the day-1 weights, fed back
                w1 = torch.from_numpy(wk[0]).view(1, -1, 1, 1)
                base = CoreBase(get_core(other))

                def loop_fn(x0, doy, tis, pv=None, w1=w1, base=base, kai_m=kai_m):
                    x, p, out = x0, pv, []
                    for k in range(L):
                        y = w1 * kai_m(x, doy + k, tis[:, k], prev=p) + (1 - w1) * base(x, doy + k, tis[:, k], prev=p)
                        x, p = y, x
                        out.append(x)
                    return torch.stack(out, 1)
                lname = ("loop" if other == "core" else f"loop_{other}") + sfx
                fns[lname] = loop_fn
                res[lname] = metrics(rollout(loop_fn, tez, ics))
                res[lname]["kai_weight"] = np.repeat(wk[:1], L, 0)
                print("scored", lname, flush=True)
        for combo in a.blend_multi:  # KAI with two others: weights >= 0, sum 1, per lead and channel
            oth = combo.split("+")
            if len(oth) != 2 or not all(o in keep for o in oth):
                continue
            avg = lambda f: (f * w).mean((0, -2, -1)).astype(np.float64)  # noqa: E731  (L, 67)
            r = vtruth - vk
            d1 = rollout(fns[oth[0]], vaz, vics) - vk
            d2 = rollout(fns[oth[1]], vaz, vics) - vk
            b1, b2, g11, g22, g12 = avg(d1 * r), avg(d2 * r), avg(d1 * d1), avg(d2 * d2), avg(d1 * d2)
            del r, d1, d2
            best = simplex2(b1, b2, g11, g22, g12)  # (2, L, 67)
            w1, w2 = (x[None, :, :, None, None] for x in best)
            name = f"blend_{combo}" + sfx
            res[name] = metrics(keep[bt] + w1 * (keep[oth[0]] - keep[bt]) + w2 * (keep[oth[1]] - keep[bt]))
            res[name]["kai_weight"] = 1 - best.sum(0)
            res[name]["other_weights"] = best
            print("scored", name, flush=True)
        del vk
    # --seq-mix A+B[+C]: the members' one-day forecasts mixed EVERY step with weights per lead and channel and fed
    # back (the step-wise coupling with lead-dependent weights). The weights are fitted on the validation year
    # step by step: given the mixed states of day k, the weights of day k + 1 minimize that day's error.
    for combo in a.seq_mix:
        mem = combo.split("+")
        if not (2 <= len(mem) <= 3) or not all(t in models or t in a.seq_cores for t in mem):
            continue
        steps = [models[t] if t in models else CoreBase(get_core(t)) for t in mem]
        va = load_years(a, [a.val_year])[a.val_year]
        vics = np.arange(0, len(va) - L, a.ic_stride)
        vaz = zscore(va, mean, std)
        wl = torch.from_numpy(w)

        def members(x, doy, tk, pv):
            with torch.no_grad():
                return [torch.cat([st(x[i:i + 16], doy[i:i + 16], tk[i:i + 16], prev=pv[i:i + 16])
                                   for i in range(0, len(x), 16)]) for st in steps]

        x = torch.from_numpy(vaz[vics])
        pv = torch.from_numpy(vaz[np.maximum(vics - 1, 0)])
        doy = torch.tensor(vics)
        W = np.zeros((L, len(mem), contract.N_CHANNELS), dtype=np.float32)
        avg = lambda f: (f * wl).mean((0, -2, -1)).double().numpy()  # noqa: E731
        for k in range(L):
            tgt = torch.from_numpy(vaz[vics + k + 1])
            F = members(x, doy + k, tgt[:, TISR], pv)
            r, d = tgt - F[0], [f - F[0] for f in F[1:]]
            if len(d) == 1:
                best = np.clip(avg(d[0] * r) / np.maximum(avg(d[0] * d[0]), 1e-30), 0, 1)[None].astype(np.float32)
            else:
                best = simplex2(avg(d[0] * r), avg(d[1] * r), avg(d[0] * d[0]), avg(d[1] * d[1]), avg(d[0] * d[1]))
            W[k, 1:], W[k, 0] = best, 1 - best.sum(0)
            pv, x = x, sum(torch.from_numpy(W[k, m]).view(1, -1, 1, 1) * F[m] for m in range(len(mem)))
        del va, vaz
        Wt = torch.from_numpy(W)

        def seq_fn(x0, doy, tis, pv=None, steps=steps, Wt=Wt):
            x, p, out = x0, pv, []
            for k in range(L):
                y = sum(Wt[k, m].view(1, -1, 1, 1) * st(x, doy + k, tis[:, k], prev=p) for m, st in enumerate(steps))
                x, p = y, x
                out.append(x)
            return torch.stack(out, 1)
        name = f"seq_{combo}"
        fns[name] = seq_fn
        res[name] = metrics(rollout(seq_fn, tez, ics))
        res[name]["seq_weights"] = W
        print("scored", name, flush=True)
    run_cascades()
    np.savez_compressed(os.path.join(a.out, "score_raw.npz"),
                        **{f"{k}__{q}": v for k, d in res.items() for q, v in d.items() if isinstance(v, np.ndarray)})
    summarize(a, res, std, ics)


def summarize(a, res, std, ics):
    names = contract.CHANNEL_NAMES
    scored = [c for c in range(contract.N_CHANNELS) if c not in (OROG, TISR)]
    rng = np.random.default_rng(0)
    out = {"ics": int(len(ics)), "leads": a.leads, "channels": names, "arms": {}}
    for k, d in res.items():
        rmse = np.sqrt(d["mse"].mean(0))  # (L, 67)
        e = {"rmse": rmse.tolist(), "acc": d["acc"].mean(0).tolist(),
             "nrmse": (rmse[:, scored] / std[scored]).mean(1).tolist(),
             "acc_mean": d["acc"].mean(0)[:, scored].mean(1).tolist()}  # amplitude-free: is the gain damping only?
        for x in ("train_seconds", "params"):
            if x in d:
                e[x] = float(d[x])
        out["arms"][k] = e
    if "blend" in res:
        out["blend_kai_weight"] = res["blend"]["kai_weight"].tolist()
    out["global_bias"] = {k: d["gbias"].tolist() for k, d in res.items() if "gbias" in d}
    out["gates"] = {k: d["gate"].tolist() for k, d in res.items() if "gate" in d}
    # paired bootstrap over ICs of the RMSE change vs kai: per channel and lead, and for the 65-channel mean
    if "kai" in res:
        base = res["kai"]["mse"]
        idx = rng.integers(0, len(ics), (500, len(ics)))
        nr = lambda m: (np.sqrt(m.mean(0))[:, scored] / std[scored]).mean(1)  # noqa: E731
        out["paired_vs_kai"] = {}
        for k in res:
            if k in ("kai", "climatology"):
                continue
            mm = res[k]["mse"]
            bs = np.stack([np.sqrt(mm[i].mean(0)) / np.sqrt(base[i].mean(0)) - 1 for i in idx])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            bn = np.stack([nr(mm[i]) / nr(base[i]) - 1 for i in idx])
            out["paired_vs_kai"][k] = {
                "rel": (np.sqrt(mm.mean(0)) / np.sqrt(base.mean(0)) - 1).tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                "nrmse_rel": (nr(mm) / nr(base) - 1).tolist(),
                "nrmse_lo": np.percentile(bn, 2.5, 0).tolist(), "nrmse_hi": np.percentile(bn, 97.5, 0).tolist()}
    # further paired comparisons a:b (same bootstrap): the effect of the core against its controls
    out["pairs"] = {}
    for pair in a.pairs:
        x, y = pair.split(":")
        if x not in res or y not in res:
            continue
        mx, my = res[x]["mse"], res[y]["mse"]
        idx = rng.integers(0, len(ics), (500, len(ics)))
        nr = lambda m: (np.sqrt(m.mean(0))[:, scored] / std[scored]).mean(1)  # noqa: E731
        bn = np.stack([nr(mx[i]) / nr(my[i]) - 1 for i in idx])
        bs = np.stack([np.sqrt(mx[i].mean(0)) / np.sqrt(my[i].mean(0)) - 1 for i in idx[:300]])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        px = (np.sqrt(mx)[:, :, scored] / std[scored]).mean(2)
        py = (np.sqrt(my)[:, :, scored] / std[scored]).mean(2)
        out["pairs"][f"{x}/{y}"] = {
            "rel": (np.sqrt(mx.mean(0)) / np.sqrt(my.mean(0)) - 1).tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
            "nrmse_rel": (nr(mx) / nr(my) - 1).tolist(),
            "nrmse_lo": np.percentile(bn, 2.5, 0).tolist(), "nrmse_hi": np.percentile(bn, 97.5, 0).tolist(),
            "ic_win_frac": (px < py).mean(0).tolist(),
            "better_cells": int((hi < 0)[:, scored].sum()), "worse_cells": int((lo > 0)[:, scored].sum())}
    for k in list(res):
        if k.startswith("blend") and "kai_weight" in res[k]:
            out.setdefault("blend_weights", {})[k] = res[k]["kai_weight"].tolist()
    with open(os.path.join(a.out, "score.json"), "w") as f:
        json.dump(out, f)
    print("pairs (normalized RMSE of a relative to b, %, lead 1/3/5/10 [95 % CI]; cells better/worse of 650)")
    for k, e in out["pairs"].items():
        print(f"  {k:22s}", " ".join(f"{100 * e['nrmse_rel'][q]:+.1f}[{100 * e['nrmse_lo'][q]:+.1f},{100 * e['nrmse_hi'][q]:+.1f}]"
                                    for q in (0, 2, 4, a.leads - 1) if q < a.leads), e["better_cells"], e["worse_cells"])
    print("ICs", len(ics))
    ls = [q for q in (0, 2, 4, 9) if q < a.leads]
    for k, e in out["arms"].items():
        r = np.array(e["rmse"])
        print(f"{k:12s}", " ".join(f"{n}:" + "/".join(f"{r[q, names.index(n)]:.3g}" for q in ls)
                                    for n in ("z500", "t850", "t2m", "u850")),
              " nrmse", "/".join(f"{e['nrmse'][q]:.3f}" for q in ls),
              " acc", "/".join(f"{e['acc_mean'][q]:.3f}" for q in ls))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/era5_5p625/kai6h")
    p.add_argument("--era5", default="data/era5_5p625/era5_data", help="constants (orography, lsm) for the core")
    p.add_argument("--cache", default="data/era5_5p625/kaiday")
    p.add_argument("--out", default="runs/kaitend")
    p.add_argument("--arm", choices=["core"] + list(ARMS))
    p.add_argument("--tag", default=None, help="checkpoint name (default: the arm); 'kai_long' scores as arm kai")
    p.add_argument("--score", nargs="*", help="tags to score (the arm is the part before '_')")
    p.add_argument("--train-years", type=int, nargs="+", default=[2012, 2013, 2014, 2015])
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--test-year", type=int, default=2017)
    p.add_argument("--depths", type=int, nargs=4, default=[2, 2, 4, 2], help="KAI default: 3 3 15 3")
    p.add_argument("--dims", type=int, nargs=4, default=[24, 48, 64, 96], help="KAI default: 48 96 192 288")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rollout", type=int, nargs="+", default=[1],
                   help="position arms: rollout days of the training loss, a curriculum over the iterations")
    p.add_argument("--init", default=None, help="position arms: start from this checkpoint tag (fine-tuning)")
    p.add_argument("--gate-lr", type=float, default=1e-2, help="arm gate: learning rate of the 67 mixing weights")
    p.add_argument("--blend-cores", nargs="*", default=None, help="cores that are blended (default: all scored)")
    p.add_argument("--blend-multi", nargs="*", default=[],
                   help="A+B: three-way blend of the base KAI with two kept forecasts (e.g. a short- and a long-range core)")
    p.add_argument("--seq-mix", nargs="*", default=[],
                   help="A+B[+C]: one-day forecasts of scored position tags / cores mixed every step with weights per "
                        "lead and channel (fitted step by step on the validation year) and fed back; seq_A+B[+C]")
    p.add_argument("--seq-cores", nargs="*", default=[], help="core tags that --seq-mix may use")
    p.add_argument("--cascade", nargs="*", default=[],
                   help="A>B@n: forecast function A up to day n, then B from A's day-n state; scored as cas_A>B@n "
                        "(A, B: core tags, position tags, loop_<core>)")
    p.add_argument("--blend-bases", nargs="*", default=["kai"],
                   help="scored KAI tags that are blended with the cores / --blend-with tags")
    p.add_argument("--loop-blend", nargs="*", default=[],
                   help="core tags: score the step-wise blend (KAI and core forecasts mixed every day and fed back)")
    p.add_argument("--core-tag", default="core")
    p.add_argument("--stack-tag", default="res", help="arm stack: the trained res checkpoint used as first stage")
    p.add_argument("--blend-with", nargs="*", default=["res", "kai_s1"],
                   help="extra blends of kai with these scored tags (controls of the kai + core blend)")
    p.add_argument("--pairs", nargs="*", default=["out:res", "out_s1:res_s1", "out:res_s1", "out_s1:res", "out:stack",
                                                  "out:kai_long", "out:core", "res:kai", "core_gn:core", "core_gn2:core_gn",
                                                  "core_gnzm:core_gn", "core_gn:kai", "core_gn2:kai", "out_gn:out",
                                                  "out_gn:res", "out_gn:kai", "out_gn2:out_gn", "out_gn2:kai",
                                                  "out_gn:kai_long", "out_gn2:kai_long", "in_gn:kai", "in_gn:in",
                                                  "loop:kai", "loop_core_gn:kai", "loop_core_gn:blend_core_gn",
                                                  "blend_core_gn:kai", "blend_core_gn:blend", "blend_core_gn:blend_kai_s1",
                                                  "loop_core_gn:blend_kai_s1", "blend_out_gn:kai", "blend_out_gn:blend_kai_s1",
                                                  "out_gnro:kai_ro", "out_gnro:out_gn", "kai_ro:kai", "out_gnro:kai",
                                                  "out_gnro:blend_core_gn", "inout_gn:out_gn", "inout_gn:kai", "kai_s1:kai", "out_s1:out",
                                                  "res_s1:res", "blend:blend_res", "blend:blend_kai_s1", "inout:out",
                                                  "inres:res", "stack:res"],
                   help="extra paired comparisons a:b (RMSE of a relative to b)")
    p.add_argument("--core-hidden", type=int, default=48)
    p.add_argument("--core-blocks", type=int, nargs="+", default=[2, 2])
    p.add_argument("--core-lr", type=float, default=4e-4)
    p.add_argument("--core-norm", default="bn", choices=["bn", "gn"],
                   help="bn = ClimODE's block as it is; gn = GroupNorm, no dropout (no train / inference mismatch)")
    p.add_argument("--core-substeps", type=int, default=1, help="Heun steps per day")
    p.add_argument("--core-mem", type=int, default=0,
                   help="memory core (Mori-Zwanzig memory term as a ConvGRU state): days before the IC that warm "
                        "the memory up; 0 = off")
    p.add_argument("--core-mem-ch", type=int, default=24, help="channels of the memory state")
    p.add_argument("--core-mz", action="store_true",
                   help="memory core in the Mori-Zwanzig sequence form: the --core-mem days are forecast one by one "
                        "from an empty memory (sequence loss on all of them + the rollout), the forecast errors "
                        "(innovations) feed the memory")
    p.add_argument("--mz-orth", type=float, default=1.0,
                   help="--core-mz: weight of the orthogonality term (squared lag-1 and lag-2 correlation of the "
                        "sequence's one-step residuals); 0 = off")
    p.add_argument("--mem-warm", type=int, nargs="*", default=[],
                   help="score: memory cores again with only the last N days before the IC in the memory (0 = empty)")
    p.add_argument("--mem-freeze", action="store_true",
                   help="score: memory cores again without updating the memory after the IC")
    p.add_argument("--core-hist", action="store_true",
                   help="the core also gets the state one step before (the observed trend) as input")
    p.add_argument("--core-stride", type=int, default=1,
                   help="days per solver step of the core (long-scale core; --curriculum then counts these steps)")
    p.add_argument("--hier", nargs="*", default=[],
                   help="JUMPS:DAILY, JUMPS = long-scale core tags joined by '+': every lead is made of the largest "
                        "steps first, DAILY (a core tag or a scored position tag) takes the single days; "
                        "scored as hier_JUMPS+DAILY")
    p.add_argument("--core-zm", type=float, default=0.0, help="weight of a zonal-mean error term in the core loss")
    p.add_argument("--score-cores", nargs="*", default=None, help="core tags to score alone (default: --core-tag)")
    p.add_argument("--curriculum", type=int, nargs="+", default=[1, 2, 3, 4], help="core rollout days")
    p.add_argument("--leads", type=int, default=10)
    p.add_argument("--ic-stride", type=int, default=3)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    if a.arm:
        train(a, a.arm)
    if a.score is not None:
        score(a, a.score)


if __name__ == "__main__":
    main()
