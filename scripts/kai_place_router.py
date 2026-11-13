"""A placement router for the blend: KAI's domain or the tendency core's?  docs/kai_place_router.md

Owner's proposal (2026-11-13): the blend does not have to create information, it has to distribute the two
forecasts well. Label where KAI is the better one and where the tendency core is ("KAI label" / "tendency
label"), train a neural-operator router (SFNO of github.com/neuraloperator/neuraloperator) in ONE stage to tell
the two labels apart, and place the experts by its answer.

Label of a forecast (IC, lead), channel and grid point: 1 if the core is closer to the truth than KAI, else 0
(``--label-win r``: the squared errors are first averaged over a (2r+1) x (2r+1) window, a region label).
The router gives one logit per channel and grid point, trained with the binary cross-entropy of that label.
Its answer p is used in two ways: placed (hard: the core where p > 0.5, KAI elsewhere) and weighted (soft:
p as the core's weight).  Everything is fitted on the validation year and scored on the test year with the
metrics of the score, on forecasts written by ``run_kai_tendency.py --score ... --dump DIR``.

Compared:
  R0, R1, R1g   the static blend per lead x channel, + latitude band, + grid point (scripts/kai_blend_router.py)
  R0k, R1k      static blend per lead x channel x zonal-wavenumber group (k 0 / 1-3 / 4-9 / 10+), + latitude band:
                the placement by scale without training (closed form, Parseval along the longitude)
  P0            placed by the measured table: per lead x channel the expert with the lower RMSE of the validation year
  Pmap          placed by the label map: per lead x channel x grid point the label that was more frequent in the
                validation year (hard), or that frequency as the weight (soft). No network.
  Pclim         placed by a rule without any fit: the expert whose forecast is closer to the climatology
  Poracle       placed by the true label of the test forecast: the upper bound of any placement
  NO            SFNO router that sees the forecasts, trained on the labels (the proposal)
  NOs           the same router that sees the place and the lead only (what the forecasts add to the placement)
  CN            a convolutional router with the same inputs and labels (what the neural operator adds)
  NOm           the SFNO router trained by the MSE of the blended forecast, started at R0 (what the labels add)

    python scripts/kai_place_router.py --dump runs/kaitend/dump_router --cache data/era5_5p625/kaiday \
        --stats runs/kaitend/stats.npz --kai kai_ro --core core_gngmz8w --out runs/kaitend/place_A
"""

import argparse
import json
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from kai_blend_router import SCORED, Conv, metrics_chunk  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.kai_tend import OROG  # noqa: E402
from run_kai_tendency import EPS, summarize  # noqa: E402

KGROUPS = [(0, 1), (1, 4), (4, 10), (10, 10 ** 6)]  # zonal wavenumber groups of scripts/eval_kai_breakdown.py


# ------------------------------------------------------------------------------------------------ routers
class Router(nn.Module):
    """One logit per channel and grid point: > 0 = the core's domain. ``body`` maps the input channels to C maps;
    its last layer starts at zero, so the untrained router gives the table b0 (lead, channel)."""

    def __init__(self, body, last, C, L, static, b0, see):
        super().__init__()
        self.body, self.C, self.L, self.see = body, C, L, see
        self.register_buffer("static", static)           # (S, H, W)
        self.b0 = nn.Parameter(torch.as_tensor(b0, dtype=torch.float32).clone())  # (L, C)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    @staticmethod
    def n_in(C, L, S, see):
        return S + L + (4 * C if see else 0)

    def forward(self, kai, core, cl, lead):
        """kai, core, cl (B, C, H, W) z-scored forecasts and climatology, lead (B,) 0-based -> logits (B, C, H, W)."""
        B, _, H, W = kai.shape
        x = [self.static[None].expand(B, -1, -1, -1), Fn.one_hot(lead, self.L).float()[:, :, None, None].expand(-1, -1, H, W)]
        if self.see:
            x = [kai, core, core - kai, cl] + x
        return self.body(torch.cat(x, 1)) + self.b0[lead][:, :, None, None]


def sfno_router(C, L, static, b0, see, hidden, layers, grid):
    """The SFNO of the neuraloperator library. The grid of the data (cell centres, 5.625 deg) is neither of the
    grids of torch-harmonics; the Gauss grid is the closer one (at most 1.4 deg off, at the poles)."""
    from neuralop.layers.spherical_convolution import SphericalConv
    from neuralop.models import SFNO
    H, W = static.shape[-2:]
    body = SFNO(n_modes=(H, W), in_channels=Router.n_in(C, L, static.shape[0], see), out_channels=C,
                hidden_channels=hidden, n_layers=layers, positional_embedding=None,
                conv_module=partial(SphericalConv, sht_grids=grid))
    return Router(body, body.projection.fcs[-1], C, L, static, b0, see)


class CNNBody(nn.Module):
    def __init__(self, cin, C, hidden, layers):
        super().__init__()
        self.inp = Conv(cin, hidden)
        self.blocks = nn.ModuleList(Conv(hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.GroupNorm(8, hidden) for _ in range(layers + 1))
        self.out = nn.Conv2d(hidden, C, 1)

    def forward(self, x):
        h = self.inp(x)
        for i, n in enumerate(self.norms):
            if i:
                h = h + self.blocks[i - 1](h)
            h = Fn.gelu(n(h))
        return self.out(h)


def cnn_router(C, L, static, b0, see, hidden, layers):
    body = CNNBody(Router.n_in(C, L, static.shape[0], see), C, hidden, layers)
    return Router(body, body.out, C, L, static, b0, see)


def box(x, r):
    """Mean over a (2r+1) x (2r+1) window: periodic in longitude, the edge row repeated at the poles."""
    if r == 0:
        return x
    x = Fn.pad(Fn.pad(x, (r, r, 0, 0), mode="circular"), (0, 0, r, r), mode="replicate")
    return Fn.avg_pool2d(x, 2 * r + 1, stride=1)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True, help="folder written by run_kai_tendency.py --score ... --dump")
    p.add_argument("--cache", required=True, help="the daily means <year>.npy (the truth)")
    p.add_argument("--stats", required=True, help="stats.npz of the run (mean, std, climatology)")
    p.add_argument("--kai", default="kai_ro", help="tag of the KAI forecast in the dump")
    p.add_argument("--core", default="core_gngmz8w", help="tag of the tendency core's forecast in the dump")
    p.add_argument("--out", required=True)
    p.add_argument("--band", type=int, default=4, help="latitude rows per band")
    p.add_argument("--label-win", type=int, default=0, help="r: labels from squared errors averaged over (2r+1)^2 points")
    p.add_argument("--routers", nargs="*", default=["NO", "NOs", "CN", "NOm"])
    p.add_argument("--hidden", type=int, default=48, help="width of the SFNO")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--cnn-hidden", type=int, default=96)
    p.add_argument("--cnn-layers", type=int, default=3)
    p.add_argument("--sht-grid", default="legendre-gauss")
    p.add_argument("--epochs", type=int, default=8, help="most epochs; the number used is chosen on held-out blocks")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--block", type=int, default=15, help="ICs per block of the split of the validation year")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--pairs", nargs="*", default=None)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    meta, st = np.load(os.path.join(a.dump, "meta.npz")), np.load(a.stats)
    lat, mean, std, L = meta["lat"], meta["mean"], meta["std"], int(meta["leads"])
    clim = st["clim"]
    sd = (std + EPS).astype(np.float32)[None, None, :, None, None]
    mn = mean.astype(np.float32)[None, None, :, None, None]
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None].astype(np.float32)
    wl = w[:, 0].astype(np.float64)
    regions = [np.abs(lat) < 20, lat >= 20, lat <= -20]
    nc_ = len(clim) - 1
    H, W = len(lat), clim.shape[-1]
    C = contract.N_CHANNELS
    z = lambda x: ((x - mn) / sd).astype(np.float16)  # noqa: E731

    def year(tagy, ics):
        x = np.load(os.path.join(a.cache, f"{int(meta[tagy])}.npy"), mmap_mode="r")
        truth = np.stack([x[i + 1:i + 1 + L] for i in ics]).astype(np.float32)
        cl = np.stack([clim[[min(i + k, nc_) for k in range(1, L + 1)]] for i in ics]).astype(np.float32)
        return truth, cl

    ics_te, ics_va = meta["test_ics"], meta["val_ics"]
    truth_te, cl_te = year("test_year", ics_te)
    truth_va, cl_va = year("val_year", ics_va)
    Tva, Cva, Tte, Cte = z(truth_va), z(cl_va), z(truth_te), z(cl_te)
    del truth_va, cl_va
    load = lambda t, part: np.load(os.path.join(a.dump, f"{t}__{part}.npy"))  # noqa: E731  float16 (N, L, C, H, W)
    Kva, Ova, Kte, Ote = load(a.kai, "val"), load(a.core, "val"), load(a.kai, "test"), load(a.core, "test")
    Nv, Nt = len(ics_va), len(ics_te)
    f32 = lambda x: x.astype(np.float32)  # noqa: E731
    print(f"KAI {a.kai}, core {a.core}; validation ICs {Nv}, test ICs {Nt}, grid {H} x {W}, label window {a.label_win}",
          flush=True)

    res, info, t0 = {}, {"kai": a.kai, "core": a.core, "label_win": a.label_win, "variants": {}}, time.time()

    def score(name, pz):
        """pz (Nt, L, C, H, W) z-scored forecast of the test year."""
        parts = []
        for i in range(0, Nt, 40):
            s = slice(i, i + 40)
            parts.append(metrics_chunk(f32(pz[s]) * sd + mn, truth_te[s], cl_te[s], w, regions))
        out = {k: np.concatenate([q[k] for q in parts]) for k in ("mse", "acc", "dec", "act", "mse_reg")}
        out["gbias"] = (np.concatenate([q["gm"] for q in parts]).mean(0) / (std + EPS)).astype(np.float32)
        for k in ("sp_pred", "sp_true", "sp_err"):
            out[k] = np.concatenate([q[k] for q in parts]).mean(0).astype(np.float32)
        res[name] = out
        nr = (np.sqrt(out["mse"].mean(0))[:, SCORED] / std[SCORED]).mean(1)
        print(f"scored {name:12s} nrmse " + "/".join(f"{nr[q]:.4f}" for q in (0, 2, 4, 6, 9) if q < L)
              + "  act " + "/".join(f"{out['act'].mean(0)[q, SCORED].mean():.2f}" for q in (0, 4, 9) if q < L)
              + f"  [{time.time() - t0:.0f} s]", flush=True)

    def mix(wgt):  # weight of the core, broadcast to (Nt, L, C, H, W): KAI + w (core - KAI) on the test year
        out = np.empty((Nt, L, C, H, W), dtype=np.float32)
        for i in range(0, Nt, 16):
            s = slice(i, i + 16)
            k = f32(Kte[s])
            out[s] = k + (wgt[s] if getattr(wgt, "shape", (0,))[0] == Nt else wgt) * (f32(Ote[s]) - k)
        return out

    score("persistence", z(np.repeat(np.load(os.path.join(a.cache, f"{int(meta['test_year'])}.npy"),
                                             mmap_mode="r")[ics_te][:, None], L, 1).astype(np.float32)))
    score("climatology", Cte)
    score(a.kai, Kte)
    score(a.core, Ote)

    # ---- moments of the validation year (two halves of alternating blocks): with r = truth - KAI, d = core - KAI
    #      the core's best weight is clip(E[r d] / E[d d], 0, 1) and the MSE of a weight w is E[r r] - 2 w B + w^2 A
    blk = np.arange(Nv) // a.block
    half = blk % 2
    hold = blk % 4 == 3                                    # held-out blocks: epochs of the routers are chosen there
    M = W // 2 + 1
    fac = np.full(M, 2.0)
    fac[0] = 1.0
    if W % 2 == 0:
        fac[-1] = 1.0
    S = {k: np.zeros((2, L, C, H, W)) for k in ("rr", "A", "B")}
    Sm = {k: np.zeros((2, L, C, H, M)) for k in ("A", "B")}
    lab = np.zeros((2, L, C, H, W))                        # label counts: [train blocks, held-out blocks]
    cnt, cnth = np.zeros(2), np.zeros(2)
    for i in range(0, Nv, 8):
        j = np.arange(i, min(Nv, i + 8))
        r, d = f32(Tva[j]) - f32(Kva[j]), f32(Ova[j]) - f32(Kva[j])
        y = (box(torch.from_numpy((r - d) ** 2).flatten(0, 1), a.label_win)
             < box(torch.from_numpy(r ** 2).flatten(0, 1), a.label_win)).view(r.shape).numpy()
        rh, dh = np.fft.rfft(r, axis=-1), np.fft.rfft(d, axis=-1)
        for h in (0, 1):
            m = half[j] == h
            if m.any():
                S["rr"][h] += (r[m] ** 2).sum(0)
                S["A"][h] += (d[m] ** 2).sum(0)
                S["B"][h] += (r[m] * d[m]).sum(0)
                Sm["A"][h] += (np.abs(dh[m]) ** 2).sum(0) * fac / W ** 2
                Sm["B"][h] += (rh[m] * np.conj(dh[m])).real.sum(0) * fac / W ** 2
                cnt[h] += m.sum()
            m = hold[j] == bool(h)
            if m.any():
                lab[h] += y[m].sum(0)
                cnth[h] += m.sum()
    del rh, dh
    nrm = lambda X: X / cnt.reshape(2, *([1] * (X.ndim - 1)))  # noqa: E731
    S = {k: nrm(v) for k, v in S.items()}
    Sm = {k: nrm(v) for k, v in Sm.items()}
    tot = lambda X: (X * cnt.reshape(2, *([1] * (X.ndim - 1)))).sum(0) / cnt.sum()  # noqa: E731
    bands = [np.arange(i, min(H, i + a.band)) for i in range(0, H, a.band)]
    groups = [np.arange(lo, min(hi, M)) for lo, hi in KGROUPS if lo < M]
    amean = lambda X: (X * wl[:, None]).mean((-2, -1))     # noqa: E731  area mean over (H, W) or sum over m per row

    def band_of(X):  # (L, C, H, W) -> the band's area mean at every point of the band
        out = np.empty_like(X)
        for rws in bands:
            out[:, :, rws] = ((X[:, :, rws] * wl[rws, None]).mean((2, 3)) / wl[rws].mean())[:, :, None, None]
        return out

    def fit(Sx, kind, alpha=0.0):
        A, B = Sx["A"], Sx["B"]
        if kind == "R0":
            return np.clip(amean(B) / amean(A), 0, 1)[:, :, None, None]
        if kind == "R1":
            return np.clip(band_of(B)[..., :1] / band_of(A)[..., :1], 0, 1)
        return np.clip(((1 - alpha) * B + alpha * band_of(B)) / ((1 - alpha) * A + alpha * band_of(A)), 0, 1)  # R1g

    def cost(wgt, Sx):  # (L, C) area-mean MSE of a weight map on the moments Sx
        return amean(Sx["rr"] - 2 * wgt * Sx["B"] + wgt ** 2 * Sx["A"])

    def fit_k(Sx, by_band):  # weight per lead, channel, (band,) wavenumber group -> (L, C, H, M)
        out = np.empty(Sx["A"].shape)
        for rws in (bands if by_band else [np.arange(H)]):
            for g in groups:
                num = (Sx["B"][:, :, rws][..., g] * wl[rws, None]).sum((2, 3))
                den = (Sx["A"][:, :, rws][..., g] * wl[rws, None]).sum((2, 3))
                out[np.ix_(range(L), range(C), rws, g)] = np.clip(num / np.maximum(den, 1e-30), 0, 1)[:, :, None, None]
        return out

    def cost_k(wk, Sx, Sg):  # the same cost for a spectral weight (the sum over m is the mean over the longitude)
        return amean(Sg["rr"]) + ((-2 * wk * Sx["B"] + wk ** 2 * Sx["A"]).sum(-1) * wl).mean(-1)

    full, fullm = {k: tot(v) for k, v in S.items()}, {k: tot(v) for k, v in Sm.items()}
    hf = lambda D, h: {k: v[h] for k, v in D.items()}      # noqa: E731
    s0 = cost(fit(full, "R0"), full)                       # (L, C) MSE of the static blend on the validation year
    cv = {}
    for kind, alphas in (("R0", [0.0]), ("R1", [0.0]), ("R1g", [0.0, 0.3, 0.6, 0.8, 0.9, 0.95])):
        for al in alphas:
            c = sum(cost(fit(hf(S, h), kind, al), hf(S, 1 - h)) * cnt[1 - h] for h in (0, 1)) / cnt.sum()
            cv[f"{kind}@{al}"] = float((c[:, SCORED] / s0[:, SCORED]).mean())
    for kind, bb in (("R0k", False), ("R1k", True)):
        c = sum(cost_k(fit_k(hf(Sm, h), bb), hf(Sm, 1 - h), hf(S, 1 - h)) * cnt[1 - h] for h in (0, 1)) / cnt.sum()
        cv[kind] = float((c[:, SCORED] / s0[:, SCORED]).mean())
    best_al = min((0.0, 0.3, 0.6, 0.8, 0.9, 0.95), key=lambda al: cv[f"R1g@{al}"])
    info["parseval_check"] = float(np.abs(cost_k(np.broadcast_to(fit(full, "R0"), fullm["A"].shape), fullm, full)[:, SCORED]
                                          / s0[:, SCORED] - 1).max())
    info["cv"] = cv
    print("two-fold CV inside the validation year (relative MSE, R0 fitted on all = 1):",
          {k: round(v, 4) for k, v in cv.items()}, "-> R1g alpha", best_al,
          "; spectral cost of R0 / grid cost - 1:", info["parseval_check"], flush=True)

    w0 = fit(full, "R0")
    stat = {"R0": w0, "R1": fit(full, "R1"), "R1g": fit(full, "R1g", best_al)}
    for k, wgt in stat.items():
        score(k, mix(wgt[None]))
        info["variants"][k] = {"core_weight": amean(np.broadcast_to(wgt, (L, C, H, W)))[:, SCORED].mean(1).tolist()}
    for kind, bb in (("R0k", False), ("R1k", True)):
        wk = fit_k(fullm, bb).astype(np.float32)
        pz = np.empty((Nt, L, C, H, W), dtype=np.float32)
        for i in range(0, Nt, 16):
            s = slice(i, i + 16)
            k = f32(Kte[s])
            pz[s] = k + np.fft.irfft(wk[None] * np.fft.rfft(f32(Ote[s]) - k, axis=-1), n=W, axis=-1)
        score(kind, pz)
        del pz
        info["variants"][kind] = {"core_weight_by_group": [
            (wk[:, SCORED][:, :, :, g[0]] * wl[None, None]).mean((1, 2)).tolist() for g in groups]}
        stat[kind] = wk
    np.savez_compressed(os.path.join(a.out, "static_weights.npz"), **stat)

    # ---- the labels: where is the core closer to the truth than KAI?
    def labels(K_, O_, T_, n, l):  # (B, C, H, W) float 0 / 1
        k, t = torch.from_numpy(f32(K_[n, l])), torch.from_numpy(f32(T_[n, l]))
        return (box((torch.from_numpy(f32(O_[n, l])) - t) ** 2, a.label_win) < box((k - t) ** 2, a.label_win)).float()

    Yte = np.empty((Nt, L, C, H, W), dtype=bool)
    for l in range(L):
        Yte[:, l] = labels(Kte, Ote, Tte, np.arange(Nt), l).numpy() > 0.5
    freq = lab.sum(0) / cnth.sum()                         # (L, C, H, W) how often the core was closer in the validation year
    freq_tr = lab[0] / cnth[0]                             # the same on the training blocks only
    names = contract.CHANNEL_NAMES
    grp = {g: [c for c in SCORED if (names[c][0] == g and names[c][1:].isdigit())] for g in "zqtuv"}
    grp["surface"] = [c for c in SCORED if not any(c in v for v in grp.values())]
    ytf = lambda X, cs, rw=slice(None): float((X[:, cs][:, :, rw] * w[rw]).mean() / w[rw].mean())  # noqa: E731
    info["labels"] = {
        "core_closer_by_lead": [ytf(Yte[:, l], SCORED) for l in range(L)],
        "core_closer_by_group": {g: [ytf(Yte[:, l], cs) for l in range(L)] for g, cs in grp.items()},
        "core_closer_by_region": {r: [ytf(Yte[:, l], SCORED, rw) for l in range(L)]
                                  for r, rw in zip(("tropics", "north", "south"), regions)},
        "channels_core_wins": [int((w0[l, SCORED, 0, 0] > 0.5).sum()) for l in range(L)],
    }
    print("labels of the test year, share with the core closer, day 1/3/5/7/10:",
          "/".join(f"{info['labels']['core_closer_by_lead'][q]:.3f}" for q in (0, 2, 4, 6, 9) if q < L),
          {g: "/".join(f"{v[q]:.2f}" for q in (0, 4, 9) if q < L) for g, v in info["labels"]["core_closer_by_group"].items()},
          flush=True)

    def placed(name, wgt, hard=True, soft=False):
        """wgt: the core's weight or probability, (Nt, L, C, H, W) or broadcastable (1, L, C, h, w)."""
        v = {}
        if hard:
            pick = wgt > 0.5
            v["label_accuracy"] = [ytf(np.broadcast_to(pick, Yte.shape)[:, l] == Yte[:, l], SCORED) for l in range(L)]
            v["core_share"] = [ytf(np.broadcast_to(pick, Yte.shape)[:, l], SCORED) for l in range(L)]
            score(name + ("_hard" if soft else ""), mix(pick.astype(np.float32)))
        if soft:
            score(name + "_soft", mix(wgt.astype(np.float32)))
        info["variants"].setdefault(name, {}).update(v)
        if hard:
            print(f"  {name}: label accuracy day 1/3/5/7/10 " + "/".join(f"{v['label_accuracy'][q]:.3f}" for q in (0, 2, 4, 6, 9) if q < L)
                  + "; core share " + "/".join(f"{v['core_share'][q]:.2f}" for q in (0, 4, 9) if q < L), flush=True)
        return v

    placed("P0", (cost(np.ones((1, 1, 1, 1)), full) < cost(np.zeros((1, 1, 1, 1)), full)).astype(np.float32)[None, :, :, None, None])
    placed("Pmap", freq[None].astype(np.float32), soft=True)
    pc = np.empty((Nt, L, C, H, W), dtype=bool)
    for i in range(0, Nt, 16):
        s = slice(i, i + 16)
        pc[s] = np.abs(f32(Ote[s]) - f32(Cte[s])) < np.abs(f32(Kte[s]) - f32(Cte[s]))
    placed("Pclim", pc)
    placed("Poracle", Yte)

    # ---- the routers
    static = torch.tensor(np.stack(
        [np.sin(np.deg2rad(lat))[:, None].repeat(W, 1), np.cos(np.deg2rad(lat))[:, None].repeat(W, 1),
         np.sin(2 * np.pi * np.arange(W) / W)[None].repeat(H, 0), np.cos(2 * np.pi * np.arange(W) / W)[None].repeat(H, 0),
         f32(Kva[0, 0, OROG])]), dtype=torch.float32)
    wt = torch.tensor(w)[None, None]                       # (1, 1, H, 1)
    sc = torch.tensor(SCORED)
    s0t = torch.tensor(s0, dtype=torch.float32)
    logit = lambda q, e=0.02: np.log(np.clip(q, e, 1 - e) / (1 - np.clip(q, e, 1 - e)))  # noqa: E731
    qtr = torch.tensor(np.clip(freq_tr, 0.02, 0.98), dtype=torch.float32)

    def router(name, arch, see, target):
        """target 'label': cross-entropy of the label; 'mse': the MSE of the blended forecast, started at R0."""
        b0 = logit(amean(freq)) if target == "label" else logit(w0[:, :, 0, 0], 1e-3)

        def make():
            if arch == "sfno":
                return sfno_router(C, L, static, b0, see, a.hidden, a.layers, a.sht_grid)
            return cnn_router(C, L, static, b0, see, a.cnn_hidden, a.cnn_layers)

        def loss_of(m, n, l):
            g = lambda X: torch.from_numpy(f32(X[n, l]))   # noqa: E731
            k, o, ld = g(Kva), g(Ova), torch.as_tensor(l)
            lg = m(k, o, g(Cva), ld)
            if target == "label":
                y = labels(Kva, Ova, Tva, n, l)
                return (Fn.binary_cross_entropy_with_logits(lg, y, reduction="none") * wt).mean((-2, -1))[:, sc].mean()
            mse = (((k + torch.sigmoid(lg) * (o - k) - g(Tva)) ** 2) * wt).mean((-2, -1))
            return (mse[:, sc] / s0t[ld][:, sc]).mean()

        def ev_loss(m, ics):
            ev = np.array([(n, l) for n in ics for l in range(L)])
            with torch.no_grad():
                return sum(float(loss_of(m, ev[i:i + 64, 0], ev[i:i + 64, 1])) * len(ev[i:i + 64])
                           for i in range(0, len(ev), 64)) / len(ev)

        def run(train_ics, epochs, eval_ics=None):
            torch.manual_seed(a.seed)
            rng = np.random.default_rng(a.seed)
            m = make()
            opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
            pairs = np.array([(n, l) for n in train_ics for l in range(L)])
            curve = []
            for ep in range(epochs):
                m.train()
                rng.shuffle(pairs)
                for i in range(0, len(pairs), a.batch):
                    q = pairs[i:i + a.batch]
                    loss = loss_of(m, q[:, 0], q[:, 1])
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                if eval_ics is not None:
                    m.eval()
                    curve.append(ev_loss(m, eval_ics))
            return m, curve

        tr_ics, ho_ics = np.where(~hold)[0], np.where(hold)[0]
        m, curve = run(tr_ics, a.epochs, ho_ics)
        if target == "label":  # reference: the cross-entropy of the label map of the training blocks on the held-out blocks
            ev = np.array([(n, l) for n in ho_ics for l in range(L)])
            base = sum(float((Fn.binary_cross_entropy(qtr[ev[i:i + 64, 1]], labels(Kva, Ova, Tva, ev[i:i + 64, 0], ev[i:i + 64, 1]),
                                                      reduction="none") * wt).mean((-2, -1))[:, sc].mean()) * len(ev[i:i + 64])
                       for i in range(0, len(ev), 64)) / len(ev)
        else:                  # reference: the untrained router = the static blend R0
            base = ev_loss(make().eval(), ho_ics)
        best = int(np.argmin(curve)) + 1
        use = best if (target == "label" or curve[best - 1] < base) else 0
        m, _ = run(np.arange(Nv), use)
        m.eval()
        P = np.empty((Nt, L, C, H, W), dtype=np.float16)
        with torch.no_grad():
            for l in range(L):
                for i in range(0, Nt, 32):
                    n = np.arange(i, min(Nt, i + 32))
                    g = lambda X: torch.from_numpy(f32(X[n, l]))  # noqa: E731
                    P[n, l] = torch.sigmoid(m(g(Kte), g(Ote), g(Cte), torch.full((len(n),), l))).numpy()
        Ps = f32(P[:, :, SCORED])
        v = {"arch": arch, "sees_forecasts": see, "target": target, "holdout_curve": curve, "holdout_reference": base,
             "epochs_used": use, "params": sum(q.numel() for q in m.parameters()),
             "spread_space": float(Ps.mean(0).std((-2, -1)).mean()), "spread_time": float(Ps.std(0).mean()),
             "agrees_with_Pclim": [ytf((P[:, l] > 0.5) == pc[:, l], SCORED) for l in range(L)]}
        info["variants"][name] = v
        print(f"  {name}: held-out {'cross-entropy' if target == 'label' else 'relative MSE'} reference {base:.4f}, best "
              f"{min(curve):.4f} at epoch {best}, used {use}; {v['params']} parameters; spread of p over the grid "
              f"{v['spread_space']:.3f}, over the forecasts {v['spread_time']:.3f}", flush=True)
        del Ps
        if target == "label":
            placed(name, P, soft=True)
        else:
            score(name, mix(f32(P)))

    for name, arch, see, target in (("NOs", "sfno", False, "label"), ("NO", "sfno", True, "label"),
                                    ("CN", "cnn", True, "label"), ("NOm", "sfno", True, "mse")):
        if name in a.routers:
            router(name, arch, see, target)

    np.savez_compressed(os.path.join(a.out, "score_place_raw.npz"),
                        **{f"{k}__{q}": v for k, d in res.items() for q, v in d.items() if isinstance(v, np.ndarray)})
    if a.pairs is None:
        a.pairs = [f"{x}:{y}" for x, y in (
            ("R0", a.kai), ("R0", a.core), ("R1g", "R0"), ("R0k", "R0"), ("R1k", "R0"), ("R1k", "R1"), ("P0", "R0"),
            ("Pmap_hard", "R0"), ("Pmap_soft", "R0"), ("Pclim", "R0"), ("Poracle", "R0"),
            ("NO_hard", "R0"), ("NO_soft", "R0"), ("NO_hard", "Pmap_hard"), ("NO_soft", "Pmap_soft"),
            ("NO_hard", "NOs_hard"), ("NO_soft", "NOs_soft"), ("NO_hard", "CN_hard"), ("NO_soft", "CN_soft"),
            ("NO_hard", a.kai), ("NO_hard", a.core), ("NOm", "R0"), ("NO_soft", "NOm"))
            if x in res and y in res]
    ns = argparse.Namespace(out=a.out, leads=L, pairs=a.pairs)
    summarize(ns, res, std, ics_te)
    os.replace(os.path.join(a.out, "score.json"), os.path.join(a.out, "score_place.json"))
    with open(os.path.join(a.out, "place_info.json"), "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    main()
