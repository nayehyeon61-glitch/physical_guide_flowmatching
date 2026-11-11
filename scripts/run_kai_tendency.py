"""Where does the separately trained tendency core go in KAI-alpha? (docs/kai_tendency.md)

Data: KAI's contract (67 channels, daily mean of 00/06/12/18 UTC, z-score with std + 1e-8, no cross-year
pairs), made from our 6-hourly WB2 files (scripts/wb2_to_kai.py --all-hours) at 5.625 deg (32 x 64).
The tendency core + (docs/kai_tendency_plus.md): the same on KAI's own grid, 1.5 deg (121 x 240, poles included;
``wb2_to_kai.py --grid 240x121`` or KAI's own daily files). A fine grid stays on disk and is normalized on access,
the test forecasts are scored in chunks of ICs, ``--device cuda`` trains and scores on a GPU, ``--core-substeps``
keeps the distance travelled per solver step and ``--core-ckpt`` the memory of a sequence at that of one day.

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
import glob
import json
import os
import shutil
import sys
import time

import netCDF4 as nc
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract  # noqa: E402
from pgcfm.kai_tend import (ARMS, DYN, CachedFeatures, CoreBase, KAIPos, KAIPosConfig, LpLoss, OROG, TISR,  # noqa: E402
                            config_dict, core_rollout, make_core, needs_core, operator_features)

EPS = contract.KAI_STD_EPS
LAZY_BYTES = 2 ** 30  # a year of daily fields larger than this stays on disk (1.5 deg: 2.9 GB; 5.625 deg: 0.2 GB)


def daily(path6h: str, cache: str) -> np.ndarray:
    """(D, 67, H, W) daily means (KAI: mean of 00/06/12/18 UTC) of one 6-hourly year file, cached as .npy.
    A year file that already holds one field a day (KAI's own daily files) is cached as it is."""
    if os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    with nc.Dataset(path6h) as d:
        f = d["fields"]
        hours = np.asarray(d["time"][:]) if "time" in d.variables else None
        T = f.shape[0]
        f.set_auto_mask(False)
        if hours is not None and T > 1 and np.all(np.diff(hours) == 6):
            assert hours[0] == 0 and T % 4 == 0, path6h
            out = np.empty((T // 4,) + f.shape[1:], dtype=np.float32)
            for i in range(0, T, 400):  # read in large slabs (zlib chunks), average every 4 times
                x = np.asarray(f[i:i + 400], dtype=np.float32)
                out[i // 4:(i + len(x)) // 4] = x.reshape(-1, 4, *x.shape[1:]).mean(1)
        else:  # daily fields already
            assert T <= 366, f"{path6h}: neither 6-hourly nor one field a day"
            out = np.empty(f.shape, dtype=np.float32)
            for i in range(0, T, 64):
                out[i:i + 64] = f[i:i + 64]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache + ".tmp.npy", out)
    os.replace(cache + ".tmp.npy", cache)
    return np.load(cache, mmap_mode="r")


def lat_of(path6h, n_lat=None, era5=None):
    """Latitudes of the data's grid: from the year file; without it or without coordinates in it (a cache of
    daily means only, KAI's own files): from the constants file under ``era5`` that has ``n_lat`` rows, else the
    registered grid of ``n_lat`` rows (121: 1.5 deg with poles)."""
    if os.path.exists(path6h):
        with nc.Dataset(path6h) as d:
            for k in ("latitude", "lat"):
                if k in d.variables:
                    return np.asarray(d[k][:], dtype=np.float64)
    if n_lat is None:
        raise KeyError("latitude")
    for p in sorted(glob.glob(os.path.join(era5 or "", "constants", "constants_*deg.nc"))):
        with nc.Dataset(p) as d:
            if "lat" in d.variables and d["lat"].shape[0] == n_lat:
                return np.asarray(d["lat"][:], dtype=np.float64)
    return contract.grid_latitudes(n_lat)


def load_years(a, years):
    """year -> (D, 67, H, W) daily fields, in memory; a fine grid (or --lazy) stays on disk as np.memmap."""
    out = {}
    for y in years:
        x = daily(os.path.join(a.data, f"{y}.nc"), os.path.join(a.cache, f"{y}.npy"))
        out[y] = x if a.lazy or x[0].nbytes * 366 > LAZY_BYTES else np.asarray(x)
    return out


class ZYear:
    """A year of daily fields (D, 67, H, W) normalized on access (the same numbers as ``zscore`` of the whole
    year), so that a fine grid stays on disk. Index = days [, further indices of the (67, H, W) fields]."""

    def __init__(self, x, mean, std, as_torch=False):
        self.x, self.as_torch = x, as_torch
        self.m, self.s = mean[:, None, None], std[:, None, None] + EPS

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        t, rest = (i[0], tuple(i[1:])) if isinstance(i, tuple) else (i, ())
        y = ((np.asarray(self.x[t]) - self.m) / self.s).astype(np.float32)
        if rest:
            y = y[(slice(None),) * (y.ndim - 3) + rest]
        return torch.from_numpy(y) if self.as_torch else y


class Lazy:
    """An (N, ...) array given by a function of a slice of its first axis (the ICs): only chunks of it are ever
    in memory."""

    def __init__(self, fn, n):
        self.fn, self.n = fn, n

    def __len__(self):
        return self.n

    @property
    def shape(self):
        return (self.n,)

    def __getitem__(self, s):
        return self.fn(s)


def stats_stream(train, step=16):
    """``stats`` for years that stay on disk: the same quantities from sums over slabs of days."""
    ys = sorted(train)
    C = train[ys[0]].shape[1]
    off = np.asarray(train[ys[0]][:step], dtype=np.float64).mean((0, 2, 3))[None, :, None, None]
    s1, s2, n = np.zeros(C), np.zeros(C), 0
    for y in ys:
        for i in range(0, len(train[y]), step):
            x = np.asarray(train[y][i:i + step], dtype=np.float64) - off
            s1 += x.sum((0, 2, 3))
            s2 += (x ** 2).sum((0, 2, 3))
            n += x.shape[0] * x.shape[2] * x.shape[3]
    mean = off.reshape(-1) + s1 / n
    std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))
    sq = []
    for y in ys:
        for i in range(0, len(train[y]) - 1, step):
            x = np.asarray(train[y][i:i + step + 1], dtype=np.float64)
            x = (x - mean[None, :, None, None]) / (std[None, :, None, None] + EPS)
            sq.append(np.mean(np.diff(x, axis=0) ** 2, (2, 3)))
    inc = np.sqrt(np.mean(np.concatenate(sq, 0), 0))
    inc = np.where(inc > 0, inc, 1.0)
    # daily climatology: mean per day of year, 31-day smoothing (a year shorter than 365 days is a smoke test)
    nd = max(366 if len(train[y]) >= 365 else len(train[y]) for y in ys)
    clim, cnt = np.zeros((nd,) + train[ys[0]].shape[1:], dtype=np.float32), np.zeros(nd)
    for y in ys:
        for i in range(0, len(train[y]), step):
            clim[i:i + step] += train[y][i:i + step]
        cnt[:len(train[y])] += 1
    for i in range(nd):
        clim[i] = clim[i] / cnt[i] if cnt[i] else clim[i - 1]
    half = min(15, (nd - 1) // 2)
    out = np.empty_like(clim)
    for i in range(nd):
        out[i] = clim[np.arange(i - half, i + half + 1) % nd].mean(0)
    return mean, std, inc, out


def stats(a, train=None):
    p = os.path.join(a.out, "stats.npz")
    if os.path.exists(p):
        s = np.load(p)
        return s["mean"], s["std"], s["inc"], s["clim"]
    if any(isinstance(v, np.memmap) for v in train.values()):
        mean, std, inc, clim = stats_stream(train)
        np.savez(p, mean=mean, std=std, inc=inc, clim=clim)
        return mean, std, inc, clim
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


def constants_of(a, lat):
    """Orography, land-sea mask and longitudes of the data's grid: the constants file under --era5 whose
    latitudes are the data's (scripts/wb2_to_kai.py writes constants_<deg>deg.nc for every grid)."""
    import xarray as xr
    for p in sorted(glob.glob(os.path.join(a.era5, "constants", "constants_*deg.nc"))):
        with xr.open_dataset(p) as d:
            if len(d["lat"]) == len(lat) and np.allclose(d["lat"].values, lat):
                return (d["orography"].values.astype(np.float32), d["lsm"].values.astype(np.float32),
                        d["lon"].values)
    raise FileNotFoundError(f"no constants_*deg.nc with the data's {len(lat)} latitudes under {a.era5}/constants")


def build_core(a, mean, std, inc, lat, load=True, tag=None):
    """The tendency core ``tag`` (default --core-tag): built from the options saved in its checkpoint, frozen.
    ``load=False``: a new core from the command line options (training)."""
    from pgcfm.core import static_features
    oro, lsm, lon = constants_of(a, lat)
    contract.register_latitudes(lat)
    opts = dict(hidden=a.core_hidden, blocks=list(a.core_blocks), norm=a.core_norm, substeps=a.core_substeps,
                stride=a.core_stride)
    if a.core_hist:
        opts["hist"] = True
    if a.core_mem:
        opts.update(mem=a.core_mem, mem_ch=a.core_mem_ch)
        if a.core_mz:
            opts["mz"] = True
        if a.core_nomem:
            opts["nomem"] = True
        if a.core_acc:
            opts.update(acc=list(a.core_acc), acc_ch=a.core_acc_ch)
        if a.core_kern:
            opts.update(kern=a.core_kern, kern_ch=a.core_kern_ch)
        if a.core_pacc:
            opts["pacc"] = list(a.core_pacc)
    if a.core_gfeat:
        opts["gfeat"] = True
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
    if opts.get("gfeat"):
        # the physical terms G(x) (step 18's operator features) on the daily means; their scale (RMS over the last
        # training year, every 4th day) is a buffer of the core, so a loaded core brings its own
        y = a.train_years[-1]
        yx = load_years(a, [y])[y]
        H, W = yx.shape[-2:]
        if load or not isinstance(yx, np.memmap) or H * W <= 8192:
            xz = torch.from_numpy(zscore(yx[:1] if load else yx[::4], mean, std))
            feats = operator_features(mean, std, xz[:1], states=None if load else xz[:, DYN])
        else:  # a fine grid: the states of the scale estimate are read a few at a time
            idx = np.arange(0, len(yx), 4)
            states = Lazy(lambda i: torch.from_numpy(zscore(np.asarray(yx[idx[i]]), mean, std))[:, DYN], len(idx))
            feats = operator_features(mean, std, torch.from_numpy(zscore(yx[:1], mean, std)), states=states,
                                      chunk=max(1, 8 * 8192 // (H * W)))
        if not load:
            core.gscale.copy_(feats.scale)
        core.to(a.device)
        feats.to(a.device)
        feats.scale = core.gscale  # after the move: the buffer the core now holds
        core.features = CachedFeatures(feats)
    return core.to(a.device)


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
    return KAIPos(KAIPosConfig.arm(arm, a.depths, a.dims), inc, base).to(a.device)


def train(a, arm):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tr = load_years(a, a.train_years)
    va = load_years(a, [a.val_year])[a.val_year]
    mean, std, inc, _ = stats(a, tr)
    lat = lat_of(os.path.join(a.data, f"{a.val_year}.nc"), va.shape[-2], a.era5)
    dev = torch.device(a.device)
    if isinstance(va, np.memmap):  # a fine grid (or --lazy): the years stay on disk, normalized per batch
        trz = {y: ZYear(tr[y], mean, std, as_torch=True) for y in tr}
        vaz = ZYear(va, mean, std, as_torch=True)
    else:
        trz = {y: torch.from_numpy(zscore(tr[y], mean, std)) for y in tr}
        vaz = torch.from_numpy(zscore(va, mean, std))
    is_core = arm == "core"
    dyn = torch.tensor(DYN, device=dev)
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
        m.grad_ckpt = bool(a.core_ckpt)  # the activations of a day are recomputed in the backward pass
        opt = torch.optim.AdamW(m.parameters(), lr=a.core_lr, weight_decay=1e-3)  # step 16
        cur = a.curriculum
        s_dyn = torch.as_tensor(inc[DYN], dtype=torch.float32).view(1, 1, -1, 1, 1).to(dev)
        scd = torch.tensor([i for i, c in enumerate(DYN) if c != TISR], device=dev)  # the 65 scored channels among the 66
        keep_dyn = torch.ones(len(DYN)).index_fill(0, torch.tensor([DYN.index(TISR)]), 0.0).view(1, 1, -1, 1, 1).to(dev)
        orth_log, acc_log = None, None
        wl = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)
        wl = (wl / wl.mean()).view(1, 1, 1, -1, 1).to(dev)
    else:
        m = build(a, arm, inc, core)
        if a.init:  # fine-tuning (e.g. --rollout 2 3 from a single-step checkpoint)
            miss = m.load_state_dict(torch.load(os.path.join(a.out, f"{a.init}.pt"), map_location=dev,
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
        prev.append(torch.stack(pv).to(dev) if hist else None)
        return torch.stack(xs).to(dev), torch.stack(tg).to(dev), torch.tensor(doy, device=dev)

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
            for i in range(0, len(idx), a.val_batch):
                j = idx[i:i + a.val_batch]
                pv = vaz[j - st] if hist else None
                if W:  # the same days as a history core; before the first day of the year: that day itself
                    pv = torch.stack([vaz[np.maximum(j - k * st, 0)] for k in range(W, 0, -1)], 1)
                if pv is not None:
                    pv = pv.to(dev)
                tgt = vaz[j + st].to(dev)
                tot += float(lp(one_step(vaz[j].to(dev), torch.tensor(j, device=dev), tgt, pv), tgt)) * len(j)
        m.train()
        return tot / len(idx)

    # --resume: the state of the training is saved at every log step (<tag>.part) and a training that was cut
    # off (the machine stopped) goes on from there
    part, it0 = os.path.join(a.out, f"{a.tag or arm}.part"), 0
    if a.resume and os.path.exists(part):
        ck = torch.load(part, map_location=dev, weights_only=False)
        m.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        log, it0, t0 = ck["log"], ck["it"], time.time() - ck["sec"]
        print(json.dumps(dict(resumed_at=it0)), flush=True)
    for it in range(it0 + 1, a.iters + 1):
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
                # the accumulated memory (--core-acc): the same condition on its time scales. The residual of a day
                # must not be correlated with the residuals ACCUMULATED over the days before it with the kernel of
                # tau days, R_tau(t) = exp(-1/tau) R_tau(t-1) + (1 - exp(-1/tau)) r_(t-1) (the effect of the long
                # memory as a loss: what is left after it is not predictable from the past 3 to 5 days)
                # (--core-pacc: its starting time constants; the memory kernel network, --core-kern, has none of
                # its own: --mz-acc-taus)
                acc_log = []
                for tau in (getattr(m, "acc", ()) or getattr(m, "pacc", ())
                            or (a.mz_acc_taus if getattr(m, "kern", 0) else ())):
                    lam, R, cs = float(np.exp(-1.0 / tau)), None, []
                    for t in range(1, r.shape[1]):
                        R = (1 - lam) * r[:, t - 1] if R is None else lam * R + (1 - lam) * r[:, t - 1]
                        if t >= 2:  # at least two days in the accumulated residual
                            u = r[:, t]
                            cs.append((u * R * wl[0]).sum((-2, -1)) / torch.sqrt(
                                (u ** 2 * wl[0]).sum((-2, -1)) * (R ** 2 * wl[0]).sum((-2, -1)) + 1e-12))
                    c = torch.stack(cs, 1).mean((0, 1))  # (66,)
                    acc_log.append(round(float(c.detach()[scd].mean()), 4))
                    if a.mz_acc_orth > 0:
                        loss = loss + a.mz_acc_orth * (c[scd] ** 2).mean()
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
                if acc_log:  # correlation of the residual with the residuals accumulated over tau days
                    log[-1]["res_corr_acc"] = acc_log
                if getattr(m, "pacc", ()):  # the learned kernel time (days): mean over the levels, per time scale
                    tk = m.pacc_tau().detach().view(len(m.pacc), -1, 12).mean(-1)  # and equation (u, v, T, q)
                    log[-1]["pacc_tau"] = [[round(float(v), 2) for v in row] for row in tk]
                    if m.mz:  # ... of the innovations: mean over the scored channels
                        te = m.pacc_tau(innov=True).detach()[:, scd].mean(-1)
                        log[-1]["pacc_tau_innov"] = [round(float(v), 2) for v in te]
            print(json.dumps(log[-1]), flush=True)
            if a.resume and it < a.iters:
                torch.save(dict(state=m.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), it=it, log=log,
                                rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), sec=time.time() - t0),
                           part + ".tmp")
                os.replace(part + ".tmp", part)
                if it == a.stop_at:
                    sys.exit(3)
    torch.save(dict(state=m.state_dict(), cfg=config_dict(m.cfg), log=log, args=vars(a),
                    core_opts=getattr(m, "opts", None),
                    params=sum(p.numel() for p in m.parameters()), train_seconds=time.time() - t0),
               os.path.join(a.out, f"{a.tag or arm}.pt"))
    if os.path.exists(part):
        os.remove(part)


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
    lat = lat_of(os.path.join(a.data, f"{a.test_year}.nc"), te.shape[-2], a.era5)
    dev = torch.device(a.device)
    # a fine grid: the year stays on disk, the forecasts of a model go to a file under <out>/score_tmp and every
    # metric is computed over chunks of ICs (the per-IC numbers are the same whatever the chunks)
    big = isinstance(te, np.memmap)
    tmp, n_tmp = os.path.join(a.out, "score_tmp"), [0]
    if big:
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None].astype(np.float32)
    L = a.leads
    regions = [np.abs(lat) < 20, lat >= 20, lat <= -20]
    sd = (std + EPS).astype(np.float32)[None, None, :, None, None]
    mn = mean.astype(np.float32)[None, None, :, None, None]
    cores, dyn = {}, torch.tensor(DYN, device=dev)
    CH = a.score_chunk or (max(1, int(1e8 // (L * te[0].size))) if big else 0)  # ICs per chunk; 0 = all at once

    def chunks(n):
        return [slice(None)] if not CH or CH >= n else [slice(i, min(n, i + CH)) for i in range(0, n, CH)]

    def gmean(f, n):
        """Mean over the ICs and the grid of f(IC slice) (n, L, 67, H, W) -> (L, 67)."""
        ch = chunks(n)
        if len(ch) == 1:
            return f(ch[0]).mean((0, -2, -1))
        tot = sum(f(c).mean((0, -2, -1), dtype=np.float64) * (c.stop - c.start) for c in ch)
        return (tot / n).astype(np.float32)

    def get_core(tag):
        if tag not in cores:
            cores[tag] = build_core(a, mean, std, inc, lat, tag=tag)
        return cores[tag]

    def setup(year_x):
        ics = np.arange(0, len(year_x) - L, a.ic_stride)
        if big:
            return ics, ZYear(year_x, mean, std), Lazy(
                lambda c: np.stack([year_x[i + 1:i + 1 + L] for i in ics[c]]), len(ics))
        return ics, zscore(year_x, mean, std), np.stack([year_x[i + 1:i + 1 + L] for i in ics])

    def rollout(fn, xz, ics):
        out, nb = [], a.score_batch
        if big:  # the forecasts of all ICs, on disk
            n_tmp[0] += 1
            out = np.lib.format.open_memmap(os.path.join(tmp, f"{n_tmp[0]}.npy"), mode="w+", dtype=np.float32,
                                            shape=(len(ics), L) + te.shape[1:])
        with torch.no_grad():
            for i in range(0, len(ics), nb):
                j = ics[i:i + nb]
                tis = torch.from_numpy(np.stack([xz[k + 1:k + 1 + L, TISR] for k in j]))
                pv = torch.from_numpy(xz[np.maximum(j - 1, 0)])  # the day before the IC (first day of the year: itself)
                if getattr(fn, "mem", 0):  # a memory core: the fn.mem days before the IC, oldest first
                    pv = torch.from_numpy(np.stack([xz[np.maximum(j - k, 0)] for k in range(fn.mem, 0, -1)], 1))
                y = fn(torch.from_numpy(xz[j]).to(dev), torch.tensor(j, device=dev), tis.to(dev), pv.to(dev)).cpu().numpy()
                if big:
                    out[i:i + len(j)] = (y * sd + mn).astype(np.float32)
                else:
                    out.append(y)
        return out if big else (np.concatenate(out) * sd + mn).astype(np.float32)

    def core_fn(tag, warm=None, freeze=False, noacc=False, k0=False, pacc_off=False):
        """``warm`` (memory cores, diagnostic): only the last ``warm`` days before the IC enter the memory
        (0 = it starts empty); ``freeze``: the memory is not updated after the IC; ``noacc``: the long-memory
        term (--core-acc, --core-kern) is switched off and the accumulated physical tendency (--core-pacc) is
        held at zero; ``k0``: the memory kernel without its network; ``pacc_off`` = "phys" / "innov": only that
        part of the accumulated physical tendency (P and D / the accumulated innovations) is held at zero."""
        core = get_core(tag)
        W = getattr(core, "mem", 0)
        W = W if warm is None else min(W, warm)

        def fn(x0, doy, tis, pv=None):
            if getattr(core, "mem", 0):
                core.freeze_mem, core.acc_off, core.pacc_off = freeze, noacc, "all" if noacc else pacc_off
                pv = pv if W else None
            if getattr(core, "kern", 0):
                core.kernel.static = k0
            y = core_rollout(core, x0[:, dyn], doy, L, tis, prev=None if pv is None else pv.index_select(-3, dyn))
            if getattr(core, "mem", 0):
                core.freeze_mem, core.acc_off, core.pacc_off = False, False, False
            if getattr(core, "kern", 0):
                core.kernel.static = False
            return x0[:, None].repeat(1, L, 1, 1, 1).index_copy(2, dyn, y)
        fn.mem = W
        return fn

    def load(tag):  # the arm is the part before '_'; the core is the one the checkpoint was trained with
        ck = torch.load(os.path.join(a.out, f"{tag}.pt"), map_location=dev, weights_only=False)
        arm = tag.split("_")[0]
        core = get_core(ck["args"].get("core_tag", "core")) if needs_core(arm) and arm != "stack" else None
        m = build(a, arm, inc, core)
        missing = m.load_state_dict(ck["state"], strict=False)
        assert set(missing.missing_keys) <= {"dyn"} and not missing.unexpected_keys, missing
        m.eval()
        return m, ck

    ics, tez, truth = setup(te)
    nc_ = len(clim) - 1  # 365; a smoke test with a short year: its last day
    cl_of = lambda c: np.stack([clim[[min(i + k, nc_) for k in range(1, L + 1)]] for i in ics[c]])  # noqa: E731
    cl = Lazy(cl_of, len(ics)) if big else cl_of(slice(None))
    res = {}

    def metrics(pred, spectra=True):
        """``pred`` (N, L, 67, H, W): an array, a file of forecasts or a Lazy; scored against the test year."""
        parts = [metrics_chunk(pred[c], truth[c], cl[c], spectra) for c in chunks(len(ics))]
        out = {k: np.concatenate([q[k] for q in parts]) for k in ("mse", "acc", "dec", "act", "mse_reg")}
        # systematic part: the IC-mean of the global-mean error, in units of the channel's std (L, 67)
        out["gbias"] = (np.concatenate([q["gm"] for q in parts]).mean(0) / (std + EPS)).astype(np.float32)
        if spectra:
            for k in ("sp_pred", "sp_true", "sp_err"):
                out[k] = np.concatenate([q[k] for q in parts]).mean(0).astype(np.float32)
        return out

    def metrics_chunk(pred, truth, cl, spectra):
        err = pred - truth
        out = dict(mse=(err ** 2 * w).mean((-2, -1)).astype(np.float32))  # (N, L, 67)
        out["gm"] = (err * w).mean((-2, -1))  # the global-mean error of every forecast
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
            P = lambda f: (np.abs(np.fft.rfft(f, axis=-1)) ** 2 * w).mean(-2)  # noqa: E731  (per IC; mean in metrics)
            out.update(sp_pred=P(pred), sp_true=P(truth), sp_err=P(err))
        return out

    def drop(*arrs):
        """Remove the forecast files (fine grid) of arrays that are not kept for the blends."""
        for x in arrs:
            if isinstance(x, np.memmap) and not any(x is k for k in keep.values()):
                os.remove(x.filename)

    def scored(fn, **kw):
        """The metrics of a forecast function whose forecasts are not kept."""
        pred = rollout(fn, tez, ics)
        out = metrics(pred, **kw)
        drop(pred)
        return out

    keep = {}  # test forecasts kept for the blends
    res["persistence"] = metrics(Lazy(lambda c: np.repeat(te[ics[c]][:, None], L, 1), len(ics)))
    res["climatology"] = metrics(Lazy(lambda c: cl[c].astype(np.float32), len(ics)))
    fns, models = {}, {}  # tag -> forecast function (x0, doy, tisr) -> (B, L, 67, H, W); tag -> KAIPos
    core_tags = a.score_cores if a.score_cores is not None else [a.core_tag]
    for ct in core_tags:
        assert get_core(ct).stride == 1, f"{ct}: a long-scale core is scored through --hier"
        fns[ct] = core_fn(ct)
        keep[ct] = rollout(fns[ct], tez, ics)
        res[ct] = metrics(keep[ct])
        if a.blend_cores is not None and ct not in a.blend_cores:
            drop(keep.pop(ct))
        ckc = torch.load(os.path.join(a.out, f"{ct}.pt"), map_location="cpu", weights_only=False)
        res[ct].update(train_seconds=ckc["train_seconds"], params=ckc["params"])
        del ckc
        print("scored", ct, flush=True)
        if getattr(get_core(ct), "mem", 0):  # how much of the past does the trained memory use?
            for wm in a.mem_warm:
                res[f"{ct}~w{wm}"] = scored(core_fn(ct, warm=wm))
            if a.mem_freeze:
                res[f"{ct}~frozen"] = scored(core_fn(ct, freeze=True))
            if any(getattr(get_core(ct), k, 0) for k in ("acc", "kern", "pacc")):
                # the effect of the long-memory term: the same rollout without it
                res[f"{ct}~noacc"] = scored(core_fn(ct, noacc=True))
            if getattr(get_core(ct), "pacc", ()):  # which accumulation? the physical tendency and trend / the
                res[f"{ct}~nophys"] = scored(core_fn(ct, pacc_off="phys"))  # innovations
                if get_core(ct).mz:
                    res[f"{ct}~noinn"] = scored(core_fn(ct, pacc_off="innov"))
            if getattr(get_core(ct), "kern", 0):  # the kernel network's part: the same rollout with the kernel k0
                res[f"{ct}~k0"] = scored(core_fn(ct, k0=True))
            print("scored", ct, "memory diagnostics", flush=True)
    for tag in tags:
        m, ck = load(tag)
        models[tag] = m
        fns[tag] = lambda x, d, t, pv=None, m=m: m.rollout(x, d, L, t, prev=pv)
        pred = rollout(fns[tag], tez, ics)
        res[tag] = metrics(pred)
        res[tag].update(train_seconds=ck["train_seconds"], params=ck["params"])
        if hasattr(m, "gate"):
            res[tag]["gate"] = m.gate.detach().view(-1).cpu().numpy().copy()
        if tag in a.blend_bases or tag in a.blend_with:
            keep[tag] = pred
        if m.cfg.inject != "none":  # reliance on the core at this position: the same rollout with tau = 0
            m.ablate = True
            res[f"{tag}~notau"] = scored(lambda x, d, t, pv=None: m.rollout(x, d, L, t, prev=pv), spectra=False)
            m.ablate = False
            if m.cfg.form == "full" and m.uses_core:  # is a diverging rollout the KAI <-> core feedback? the core on its own trajectory
                res[f"{tag}~open"] = scored(lambda x, d, t, pv=None: m.rollout(x, d, L, t, open_loop=True, prev=pv))
        drop(pred)
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
        drop(pred)
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
            drop(pred)
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
        vk, nv = rollout(fns[bt], vaz, vics), len(vics)
        for other in [t for t in core_tags if t in keep] + [t for t in a.blend_with
                                                            if t in keep and t not in core_tags and t != bt]:
            vc = rollout(fns[other], vaz, vics)
            num = gmean(lambda c: (vk[c] - vc[c]) * (vtruth[c] - vc[c]) * w, nv)
            den = gmean(lambda c: (vk[c] - vc[c]) ** 2 * w, nv)
            wk = np.clip(num / np.maximum(den, 1e-30), 0.0, 1.0).astype(np.float32)  # (L, 67) weight of KAI
            drop(vc)
            vc = None  # frees the validation forecasts (the weights are computed)
            name = ("blend" if other == "core" else f"blend_{other}") + sfx
            wb = wk[None, :, :, None, None]
            res[name] = metrics(Lazy(lambda c, wb=wb, kb=keep[bt], ko=keep[other]: wb * kb[c] + (1 - wb) * ko[c],
                                     len(ics)))
            res[name]["kai_weight"] = wk
            print("scored", name, flush=True)
            if other in a.loop_blend:  # the same two models coupled every step: the day-1 weights, fed back
                w1 = torch.from_numpy(wk[0]).view(1, -1, 1, 1).to(dev)
                base = CoreBase(get_core(other), stateful=True)

                def loop_fn(x0, doy, tis, pv=None, w1=w1, base=base, kai_m=kai_m):
                    x, p, out = x0, pv, []
                    if base.mem:  # a memory core: pv = the days before the IC; its memory follows the mixed states
                        base.start(pv, doy, x0)
                        p = pv[:, -1]
                    for k in range(L):
                        y = w1 * kai_m(x, doy + k, tis[:, k], prev=p) + (1 - w1) * base(x, doy + k, tis[:, k], prev=p)
                        x, p = y, x
                        out.append(x)
                    return torch.stack(out, 1)
                lname = ("loop" if other == "core" else f"loop_{other}") + sfx
                loop_fn.mem = base.mem
                fns[lname] = loop_fn
                res[lname] = scored(loop_fn)
                res[lname]["kai_weight"] = np.repeat(wk[:1], L, 0)
                print("scored", lname, flush=True)
        for combo in a.blend_multi:  # KAI with two others: weights >= 0, sum 1, per lead and channel
            oth = combo.split("+")
            if len(oth) != 2 or not all(o in keep for o in oth):
                continue
            v1, v2 = rollout(fns[oth[0]], vaz, vics), rollout(fns[oth[1]], vaz, vics)
            r = lambda c: vtruth[c] - vk[c]  # noqa: E731
            d1 = lambda c: v1[c] - vk[c]  # noqa: E731
            d2 = lambda c: v2[c] - vk[c]  # noqa: E731
            avg = lambda f, g: gmean(lambda c: f(c) * g(c) * w, nv).astype(np.float64)  # noqa: E731  (L, 67)
            b1, b2, g11, g22, g12 = avg(d1, r), avg(d2, r), avg(d1, d1), avg(d2, d2), avg(d1, d2)
            drop(v1, v2)
            v1 = v2 = None
            best = simplex2(b1, b2, g11, g22, g12)  # (2, L, 67)
            w1, w2 = (x[None, :, :, None, None] for x in best)
            name = f"blend_{combo}" + sfx
            res[name] = metrics(Lazy(lambda c, w1=w1, w2=w2, kb=keep[bt], k1=keep[oth[0]], k2=keep[oth[1]]:
                                     kb[c] + w1 * (k1[c] - kb[c]) + w2 * (k2[c] - kb[c]), len(ics)))
            res[name]["kai_weight"] = 1 - best.sum(0)
            res[name]["other_weights"] = best
            print("scored", name, flush=True)
            if combo in a.loop_multi:  # KAI and both cores coupled every step: the day-1 weights, fed back
                wa, wb2 = (torch.from_numpy(best[i][0].astype(np.float32)).view(1, -1, 1, 1).to(dev) for i in (0, 1))
                pair = [CoreBase(get_core(o), stateful=True) for o in oth]
                mem = max(b.mem for b in pair)

                def loop2_fn(x0, doy, tis, pv=None, wa=wa, wb2=wb2, pair=pair, kai_m=kai_m, mem=mem):
                    x, p, out = x0, pv, []
                    if mem:  # pv = the mem days before the IC, oldest first; every memory core gets its own length
                        for b in pair:
                            if b.mem:
                                b.start(pv[:, -b.mem:], doy, x0)
                        p = pv[:, -1]
                    for k in range(L):
                        yk = kai_m(x, doy + k, tis[:, k], prev=p)
                        y1, y2 = (b(x, doy + k, tis[:, k], prev=p) for b in pair)
                        x, p = yk + wa * (y1 - yk) + wb2 * (y2 - yk), x
                        out.append(x)
                    return torch.stack(out, 1)
                lname = f"loop_{combo}" + sfx
                loop2_fn.mem = mem
                fns[lname] = loop2_fn
                res[lname] = scored(loop2_fn)
                res[lname]["kai_weight"] = np.repeat((1 - best.sum(0))[:1], L, 0)
                res[lname]["other_weights"] = np.repeat(best[:, :1], L, 1)
                print("scored", lname, flush=True)
        drop(vk)
        vk = None
    # --blend-loop LOOP+OTHER: the coupled loop as the base of a lead-wise blend with another scored core (the RMSE
    # readout of the final model: the loop keeps the strong trajectory, the blend adds the long-rollout core's
    # damped field per lead and channel). Name: blend_<other>@<loop>.
    for combo in a.blend_loop:
        lp, other = combo.rsplit("+", 1)  # the loop of two cores has a "+" in its own name
        if lp not in fns or other not in keep or not bases:
            continue
        vk, vc = rollout(fns[lp], vaz, vics), rollout(fns[other], vaz, vics)
        num = gmean(lambda c: (vk[c] - vc[c]) * (vtruth[c] - vc[c]) * w, len(vics))
        den = gmean(lambda c: (vk[c] - vc[c]) ** 2 * w, len(vics))
        wk = np.clip(num / np.maximum(den, 1e-30), 0.0, 1.0).astype(np.float32)  # (L, 67) weight of the loop
        drop(vk, vc)
        vk = vc = None
        wb = wk[None, :, :, None, None]
        name = f"blend_{other}@{lp}"
        tl = rollout(fns[lp], tez, ics)
        res[name] = metrics(Lazy(lambda c, wb=wb, tl=tl, ko=keep[other]: wb * tl[c] + (1 - wb) * ko[c], len(ics)))
        drop(tl)
        del tl
        res[name]["kai_weight"] = wk
        print("scored", name, flush=True)
    # --seq-mix A+B[+C]: the members' one-day forecasts mixed EVERY step with weights per lead and channel and fed
    # back (the step-wise coupling with lead-dependent weights). The weights are fitted on the validation year
    # step by step: given the mixed states of day k, the weights of day k + 1 minimize that day's error.
    for combo in a.seq_mix:
        assert not big and dev.type == "cpu", "--seq-mix: small grids on the CPU only"
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
        res[name] = scored(seq_fn)
        res[name]["seq_weights"] = W
        print("scored", name, flush=True)
    run_cascades()
    if big:
        keep.clear()
        shutil.rmtree(tmp, ignore_errors=True)
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
    p.add_argument("--loop-multi", nargs="*", default=[],
                   help="A+B of --blend-multi (two core tags): also score the coupled loop of the base KAI with both "
                        "cores, the day-1 weights of that blend fed back every day (name loop_A+B)")
    p.add_argument("--blend-loop", nargs="*", default=[],
                   help="LOOP+OTHER: lead-wise blend of a scored coupled loop (loop_<core>) with another scored core, "
                        "weights from the validation year (name blend_<other>@<loop>)")
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
    p.add_argument("--core-nomem", action="store_true",
                   help="control of the memory term: the sequence training of --core-mz with the memory state held at "
                        "zero (a Markov core that gets the same forecasts of the sequence in its loss)")
    p.add_argument("--mz-orth", type=float, default=1.0,
                   help="--core-mz: weight of the orthogonality term (squared lag-1 and lag-2 correlation of the "
                        "sequence's one-step residuals); 0 = off")
    p.add_argument("--core-acc", type=float, nargs="*", default=[],
                   help="--core-mem: the accumulated memory, time constants in days (e.g. 3 5). The memory's inputs "
                        "are also accumulated with fixed exponential kernels of these e-folding times and enter the "
                        "tendency as their own long-memory term; with --core-mz the residuals accumulated over the "
                        "same time scales enter the orthogonality loss (--mz-acc-orth)")
    p.add_argument("--core-acc-ch", type=int, default=12, help="channels of each accumulator of --core-acc")
    p.add_argument("--mz-acc-orth", type=float, default=1.0,
                   help="--core-acc with --core-mz: weight of the squared correlation of a day's one-step residual "
                        "with the residuals accumulated over the days before it; 0 = off")
    p.add_argument("--core-kern", type=int, default=0,
                   help="--core-mem: the memory kernel network over the memory states of the last S days (e.g. 8): "
                        "M = sum over the lags s of K_s(x, m) m(t - s), K_s = a learned kernel k0[s] times a "
                        "state-dependent gain; M enters the tendency as the long-memory term")
    p.add_argument("--core-kern-ch", type=int, default=24, help="hidden channels of the kernel network")
    p.add_argument("--core-pacc", type=float, nargs="*", default=[],
                   help="--core-mem with --core-gfeat: the kernel in the memory term. The physical tendency G(x) and "
                        "the trend of its channels are accumulated in time (starting time constants in days, e.g. "
                        "3 5; the kernel time of every physical term is learned within 2 to 10 days) and enter the "
                        "memory as inputs; with --core-mz the accumulated-residual loss (--mz-acc-orth) uses the "
                        "same time scales")
    p.add_argument("--mz-acc-taus", type=float, nargs="*", default=[3.0, 5.0],
                   help="--core-kern with --core-mz: time constants (days) of the accumulated residuals in the "
                        "orthogonality loss (--mz-acc-orth); a core with --core-acc uses its own")
    p.add_argument("--mem-warm", type=int, nargs="*", default=[],
                   help="score: memory cores again with only the last N days before the IC in the memory (0 = empty)")
    p.add_argument("--mem-freeze", action="store_true",
                   help="score: memory cores again without updating the memory after the IC")
    p.add_argument("--core-gfeat", action="store_true",
                   help="the physical terms: the primitive-equation operator's tendency G(x) of u, v, T, q on the 12 "
                        "levels as 48 input channels of the core (step 18); with --core-mem the memory integrates "
                        "G(x) of the past days as well (memory terms of the physical terms)")
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
    p.add_argument("--resume", action="store_true",
                   help="training: save its state at every log step (<tag>.part) and go on from it if it is there")
    p.add_argument("--stop-at", type=int, default=0, help="test of --resume: stop after the log step of this iteration")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cuda: train and score on a GPU (the data stays in main memory)")
    p.add_argument("--lazy", action="store_true",
                   help="keep the years on disk and normalize per batch (automatic above 1 GB a year: the 1.5 deg grid)")
    p.add_argument("--core-ckpt", action="store_true",
                   help="core training: gradient checkpointing per day, the activations of a day are recomputed in "
                        "the backward pass (a fine grid with --core-substeps: a sequence costs the memory of one day)")
    p.add_argument("--score-batch", type=int, default=16, help="ICs per forward pass of a rollout in the score")
    p.add_argument("--val-batch", type=int, default=16, help="days per forward pass of the validation loss")
    p.add_argument("--score-chunk", type=int, default=0,
                   help="ICs per chunk of the metrics (0 = all at once; a fine grid: about 0.8 GB a chunk)")
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
