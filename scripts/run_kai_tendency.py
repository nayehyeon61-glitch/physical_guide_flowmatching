"""KAI-alpha original vs KAI-alpha + the step 16 tendency core, on KAI-format daily ERA5 (docs/kai_tendency.md).

Data: KAI's contract (67 channels, daily mean of 00/06/12/18 UTC, z-score with std + 1e-8, no cross-year
pairs), made from our 6-hourly WB2 files (scripts/wb2_to_kai.py --all-hours) at 5.625 deg (32 x 64).

Arms (pgcfm/kai_tend.py), same backbone size, optimizer (AdamW, cosine, clip 1.0 as KAI's trainer), loss
(KAI's LpLoss, relative L2), batch and iterations:
    kai    original KAI: full next-day state, single-step training
    res    + residual output x + s * net(x), zero head
    tend1  + Heun, season input, tisr forcing, PositiveMass (single-step training)
    tend   + rollout training (curriculum K = 1 -> --kmax days)

    python scripts/run_kai_tendency.py --data data/era5_5p625/kai6h --out runs/kaitend --arm kai
    python scripts/run_kai_tendency.py ... --score kai res tend1 tend     # paired scores + figures
"""

import argparse
import json
import math
import os
import sys
import time

import netCDF4 as nc
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract  # noqa: E402
from pgcfm.kai_tend import KAITend, KAITendConfig, LpLoss, TISR, OROG, config_dict  # noqa: E402

EPS = 1e-8  # KAI: std + 1e-8


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


def stats(a, train):
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


def build(a, arm, mean, std, inc, lat):
    cfg = KAITendConfig.arm(arm, a.depths, a.dims)
    return KAITend(cfg, inc, mean, std, lat)


def train(a, arm):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tr = load_years(a, a.train_years)
    va = load_years(a, [a.val_year])[a.val_year]
    mean, std, inc, _ = stats(a, tr)
    lat = lat_of(os.path.join(a.data, f"{a.val_year}.nc"))
    m = build(a, arm, mean, std, inc, lat)
    zs = lambda x: torch.from_numpy(((x - mean[None, :, None, None]) / (std[None, :, None, None] + EPS)).astype(np.float32))  # noqa
    trz = {y: zs(tr[y]) for y in tr}
    vaz = zs(va)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    loss_fn = LpLoss(d=2, p=2, size_average=True, reduction=True)
    kmax = a.kmax if arm == "tend" else 1
    stage = max(1, a.iters // kmax)
    years = sorted(trz)
    log, t0 = [], time.time()

    def batch(K):
        ys = rng.choice(years, a.batch)
        xs, tg, doy = [], [], []
        for y in ys:
            d = int(rng.integers(0, len(trz[y]) - K))
            xs.append(trz[y][d])
            tg.append(trz[y][d + 1:d + 1 + K])
            doy.append(d)
        return torch.stack(xs), torch.stack(tg), torch.tensor(doy)

    def val():
        m.eval()
        with torch.no_grad():
            idx = np.arange(0, len(vaz) - 1, max(1, (len(vaz) - 1) // 64))[:64]
            l = 0.0
            for i in range(0, len(idx), 16):
                j = idx[i:i + 16]
                x, t = vaz[j], vaz[j + 1]
                y = m(x, torch.tensor(j), t[:, TISR])
                l += float(loss_fn(y, t)) * len(j)
        m.train()
        return l / len(idx)

    for it in range(1, a.iters + 1):
        K = min(kmax, 1 + (it - 1) // stage)
        x, tg, doy = batch(K)
        y = m.rollout(x, doy, K, tg[:, :, TISR])
        loss = sum(loss_fn(y[:, k], tg[:, k]) for k in range(K)) / K
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % a.log_every == 0 or it == a.iters:
            v = val()
            log.append(dict(it=it, K=K, loss=float(loss), val=v, sec=time.time() - t0))
            print(json.dumps(log[-1]), flush=True)
    torch.save(dict(state=m.state_dict(), cfg=config_dict(m.cfg), log=log, args=vars(a),
                    params=sum(p.numel() for p in m.parameters()), train_seconds=time.time() - t0),
               os.path.join(a.out, f"{a.tag or arm}.pt"))


# ---------------------------------------------------------------------------------------------- scoring
BANDS = [(0, 3), (4, 9), (10, 20), (21, 32)]


def score(a, arms):
    tr = load_years(a, a.train_years)
    te = load_years(a, [a.test_year])[a.test_year]
    mean, std, inc, clim = stats(a, tr)
    lat = lat_of(os.path.join(a.data, f"{a.test_year}.nc"))
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None].astype(np.float32)
    L = a.leads
    ics = np.arange(0, len(te) - L, a.ic_stride)
    tez = ((te - mean[None, :, None, None]) / (std[None, :, None, None] + EPS)).astype(np.float32)
    truth = np.stack([te[i + 1:i + 1 + L] for i in ics])  # (N, L, 67, H, W) physical
    cl = np.stack([clim[[min(i + k, 365) for k in range(1, L + 1)]] for i in ics])
    res = {}

    def metrics(pred):
        err = pred - truth
        mse = (err ** 2 * w).mean((-2, -1))  # (N, L, 67)
        pa, ta = pred - cl, truth - cl
        acc = (pa * ta * w).sum((-2, -1)) / np.sqrt((pa ** 2 * w).sum((-2, -1)) * (ta ** 2 * w).sum((-2, -1)) + 1e-30)
        # zonal spectra (cos-lat weighted over rows): power of prediction, truth and error per wavenumber
        P = lambda f: (np.abs(np.fft.rfft(f, axis=-1)) ** 2 * w).mean(-2)  # noqa: E731
        return dict(mse=mse.astype(np.float32), acc=acc.astype(np.float32),
                    sp_pred=P(pred).mean(0).astype(np.float32), sp_true=P(truth).mean(0).astype(np.float32),
                    sp_err=P(err).mean(0).astype(np.float32))

    res["persistence"] = metrics(np.repeat(te[ics][:, None], L, 1))
    res["climatology"] = metrics(cl.astype(np.float32))
    for arm in arms:
        ck = torch.load(os.path.join(a.out, f"{arm}.pt"), map_location="cpu", weights_only=False)
        m = build(a, arm.split("_")[0], mean, std, inc, lat)
        m.load_state_dict(ck["state"])
        m.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(ics), 16):
                j = ics[i:i + 16]
                x0 = torch.from_numpy(tez[j])
                tis = torch.from_numpy(np.stack([tez[k + 1:k + 1 + L, TISR] for k in j]))
                out.append(m.rollout(x0, torch.tensor(j), L, tis).numpy())
        pred = np.concatenate(out) * (std[None, None, :, None, None] + EPS) + mean[None, None, :, None, None]
        res[arm] = metrics(pred.astype(np.float32))
        res[arm]["train_seconds"] = ck["train_seconds"]
        res[arm]["params"] = ck["params"]
        del pred
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
        nrmse = (rmse[:, scored] / std[scored]).mean(1)
        e = {"rmse": rmse.tolist(), "acc": d["acc"].mean(0).tolist(), "nrmse": nrmse.tolist()}
        st, se, sp = d["sp_true"], d["sp_err"], d["sp_pred"]  # (L, 67, nk)
        e["band_err_frac"], e["band_power_ratio"] = {}, {}
        for lo, hi in BANDS:
            b = f"k{lo}-{hi}"
            e["band_err_frac"][b] = (se[..., lo:hi + 1].sum(-1) / st.sum(-1)).tolist()  # error power / total truth power
            e["band_power_ratio"][b] = (sp[..., lo:hi + 1].sum(-1) / st[..., lo:hi + 1].sum(-1)).tolist()
        for x in ("train_seconds", "params"):
            if x in d:
                e[x] = float(d[x])
        out["arms"][k] = e
    # paired bootstrap of the RMSE change vs kai, per channel and lead (ICs resampled)
    if "kai" in res:
        base = res["kai"]["mse"]
        out["paired_vs_kai"] = {}
        B = 500
        idx = rng.integers(0, len(ics), (B, len(ics)))
        for k in res:
            if k in ("kai", "climatology"):
                continue
            mm = res[k]["mse"]
            d = np.sqrt(mm.mean(0)) / np.sqrt(base.mean(0)) - 1
            bs = np.stack([np.sqrt(mm[i].mean(0)) / np.sqrt(base[i].mean(0)) - 1 for i in idx])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            out["paired_vs_kai"][k] = {"rel": d.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                                       "win_frac": (mm < base).mean(0).tolist()}
    with open(os.path.join(a.out, "score.json"), "w") as f:
        json.dump(out, f)
    show = ["z500", "t850", "u850", "v850", "q700", "t2m", "msl", "u200", "z50"]
    print("ICs", len(ics))
    for k, e in out["arms"].items():
        r = np.array(e["rmse"])
        print(f"{k:12s}", " ".join(f"{n}:" + "/".join(f"{r[l, names.index(n)]:.3g}" for l in (0, 2, 4, 9) if l < a.leads)
                                    for n in show[:4]), " nrmse", "/".join(f"{e['nrmse'][l]:.3f}" for l in (0, 2, 4, 9) if l < a.leads))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/era5_5p625/kai6h")
    p.add_argument("--cache", default="data/era5_5p625/kaiday")
    p.add_argument("--out", default="runs/kaitend")
    p.add_argument("--arm", choices=["kai", "res", "tend1", "tend"])
    p.add_argument("--tag", default=None, help="checkpoint name (default: the arm); scored as <tag>")
    p.add_argument("--score", nargs="*", help="tags to score; a tag 'kai_long' uses the arm before '_'")
    p.add_argument("--train-years", type=int, nargs="+", default=[2012, 2013, 2014, 2015])
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--test-year", type=int, default=2017)
    p.add_argument("--depths", type=int, nargs=4, default=[2, 2, 4, 2])
    p.add_argument("--dims", type=int, nargs=4, default=[24, 48, 64, 96])
    p.add_argument("--iters", type=int, default=1200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--kmax", type=int, default=3)
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
    if a.score:
        score(a, a.score)


if __name__ == "__main__":
    main()
