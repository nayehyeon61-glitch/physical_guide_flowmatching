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
    if is_core:
        if st > 1:  # the tendency unit of a long-scale core: std of the st-day increment
            inc = np.sqrt(np.mean(np.stack([((trz[y][st:] - trz[y][:-st]) ** 2).mean((0, 2, 3)).numpy() for y in trz]), 0))
            inc = np.where(inc > 0, inc, 1.0)
        m = build_core(a, mean, std, inc, lat, load=False)
        opt = torch.optim.AdamW(m.parameters(), lr=a.core_lr, weight_decay=1e-3)  # step 16
        cur = a.curriculum
        s_dyn = torch.as_tensor(inc[DYN], dtype=torch.float32).view(1, 1, -1, 1, 1)
        wl = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)
        wl = (wl / wl.mean()).view(1, 1, 1, -1, 1)
    else:
        m = build(a, arm, inc, build_core(a, mean, std, inc, lat) if needs_core(arm) and arm != "stack" else None)
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
        xs, tg, doy = [], [], []
        for y in ys:
            d = int(rng.integers(0, len(trz[y]) - K * st))
            xs.append(trz[y][d])
            tg.append(trz[y][d + st:d + st * K + 1:st])
            doy.append(d)
        return torch.stack(xs), torch.stack(tg), torch.tensor(doy)

    def one_step(x, doy, t):
        if is_core:
            return x.index_copy(1, dyn, core_rollout(m, x[:, dyn], doy, 1, t[:, None, TISR])[:, 0])
        return m(x, doy, t[:, TISR])

    def val():  # KAI's validation loss (single step, relative L2) on 64 days of the validation year
        m.eval()
        with torch.no_grad():
            idx = np.arange(0, len(vaz) - st, max(1, (len(vaz) - st) // 64))[:64]
            tot = 0.0
            for i in range(0, len(idx), 16):
                j = idx[i:i + 16]
                tot += float(lp(one_step(vaz[j], torch.tensor(j), vaz[j + st]), vaz[j + st])) * len(j)
        m.train()
        return tot / len(idx)

    for it in range(1, a.iters + 1):
        K = cur[min(len(cur) - 1, (it - 1) * len(cur) // a.iters)]
        x, tg, doy = batch(K)
        if is_core:
            y = core_rollout(m, x[:, dyn], doy, K, tg[:, :, TISR])
            err = (y - tg[:, :, dyn]) / s_dyn
            loss = ((err ** 2) * wl).mean()
            if a.core_zm > 0:  # the zonal mean of the error on its own (a drift is small in the full MSE)
                loss = loss + a.core_zm * ((err.mean(-1, keepdim=True) ** 2) * wl).mean()
        elif K == 1:
            loss = lp(m(x, doy, tg[:, 0, TISR]), tg[:, 0])
        else:  # the connected model in its own rollout (the frozen core runs on KAI's states)
            y = m.rollout(x, doy, K, tg[:, :, TISR])
            loss = sum(lp(y[:, k], tg[:, k]) for k in range(K)) / K
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % a.log_every == 0 or it == a.iters:
            log.append(dict(it=it, K=K, loss=float(loss.detach()), val=val(), sec=time.time() - t0))
            print(json.dumps(log[-1]), flush=True)
    torch.save(dict(state=m.state_dict(), cfg=config_dict(m.cfg), log=log, args=vars(a),
                    core_opts=getattr(m, "opts", None),
                    params=sum(p.numel() for p in m.parameters()), train_seconds=time.time() - t0),
               os.path.join(a.out, f"{a.tag or arm}.pt"))


# ---------------------------------------------------------------------------------------------- scoring
def score(a, tags):
    mean, std, inc, clim = stats(a)
    te = load_years(a, [a.test_year])[a.test_year]
    lat = lat_of(os.path.join(a.data, f"{a.test_year}.nc"))
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None].astype(np.float32)
    L = a.leads
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
                out.append(fn(torch.from_numpy(xz[j]), torch.tensor(j), tis).numpy())
        return (np.concatenate(out) * sd + mn).astype(np.float32)

    def core_fn(tag):
        def fn(x0, doy, tis):
            y = core_rollout(get_core(tag), x0[:, dyn], doy, L, tis)
            return x0[:, None].repeat(1, L, 1, 1, 1).index_copy(2, dyn, y)
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
    for tag in tags:
        m, ck = load(tag)
        models[tag] = m
        fns[tag] = lambda x, d, t, m=m: m.rollout(x, d, L, t)
        pred = rollout(fns[tag], tez, ics)
        res[tag] = metrics(pred)
        res[tag].update(train_seconds=ck["train_seconds"], params=ck["params"])
        if hasattr(m, "gate"):
            res[tag]["gate"] = m.gate.detach().view(-1).numpy().copy()
        if tag in a.blend_bases or tag in a.blend_with:
            keep[tag] = pred
        if m.cfg.inject != "none":  # reliance on the core at this position: the same rollout with tau = 0
            m.ablate = True
            res[f"{tag}~notau"] = metrics(rollout(lambda x, d, t: m.rollout(x, d, L, t), tez, ics), spectra=False)
            m.ablate = False
            if m.cfg.form == "full":  # is a diverging rollout the KAI <-> core feedback? the core on its own trajectory
                res[f"{tag}~open"] = metrics(rollout(lambda x, d, t: m.rollout(x, d, L, t, open_loop=True), tez, ics))
        del pred
        print("scored", tag, flush=True)

    def hier_fn(jump, daily):
        """Hierarchical rollout: lead = a * stride + b; the long-scale core takes the a stride-day steps from x0,
        the daily model (a core or a scored position tag) the b < stride remaining days."""
        jc, st = CoreBase(get_core(jump)), get_core(jump).stride
        dstep = models[daily] if daily in models else CoreBase(get_core(daily))

        def fn(x0, doy, tis):
            out, y = [None] * L, x0
            for j in range(L // st + 1):
                x = y
                for b in range(1, st):
                    lead = j * st + b
                    if lead > L:
                        break
                    x = dstep(x, doy + lead - 1, tis[:, lead - 1])
                    out[lead - 1] = x
                lead = (j + 1) * st
                if lead <= L:
                    y = jc(y, doy + j * st, tis[:, lead - 1])
                    out[lead - 1] = y
            return torch.stack(out, 1)
        return fn

    for pair in a.hier:
        jump, daily = pair.split(":")
        name = f"hier_{jump}+{daily}"
        fns[name] = hier_fn(jump, daily)
        pred = rollout(fns[name], tez, ics)
        res[name] = metrics(pred)
        if name in a.blend_with:
            keep[name] = pred
        del pred
        print("scored", name, flush=True)
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

                def loop_fn(x0, doy, tis, w1=w1, base=base, kai_m=kai_m):
                    x, out = x0, []
                    for k in range(L):
                        x = w1 * kai_m(x, doy + k, tis[:, k]) + (1 - w1) * base(x, doy + k, tis[:, k])
                        out.append(x)
                    return torch.stack(out, 1)
                lname = ("loop" if other == "core" else f"loop_{other}") + sfx
                res[lname] = metrics(rollout(loop_fn, tez, ics))
                res[lname]["kai_weight"] = np.repeat(wk[:1], L, 0)
                print("scored", lname, flush=True)
        del vk
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


def main(argv=None):
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
    p.add_argument("--core-stride", type=int, default=1,
                   help="days per solver step of the core (long-scale core; --curriculum then counts these steps)")
    p.add_argument("--hier", nargs="*", default=[],
                   help="JUMP:DAILY pairs: the long-scale core JUMP takes the stride-day steps, DAILY (a core tag of "
                        "--score-cores or a scored position tag) the remaining days; scored as hier_JUMP+DAILY")
    p.add_argument("--core-zm", type=float, default=0.0, help="weight of a zonal-mean error term in the core loss")
    p.add_argument("--score-cores", nargs="*", default=None, help="core tags to score alone (default: --core-tag)")
    p.add_argument("--curriculum", type=int, nargs="+", default=[1, 2, 3, 4], help="core rollout days")
    p.add_argument("--leads", type=int, default=10)
    p.add_argument("--ic-stride", type=int, default=3)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    if a.arm:
        train(a, a.arm)
    if a.score is not None:
        score(a, a.score)


if __name__ == "__main__":
    main()
