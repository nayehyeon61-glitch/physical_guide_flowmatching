"""Does the lead-wise blend need a learned router (a mixture of experts)?  docs/kai_blend_router.md

The blend of scripts/run_kai_tendency.py mixes the forecasts of KAI and the cores with one weight per lead and
channel, fitted on the validation year: the same weight everywhere on the globe and in every weather situation.
A mixture-of-experts router replaces it by a gating network that sees the experts' forecasts and gives a weight
map per grid point and channel. This script compares both, and the steps between them, on forecasts written by
``run_kai_tendency.py --score ... --dump DIR``:

  R0    static weights per lead and channel, on the simplex (the current blend, for any number of experts)
  R1    static weights per lead, channel and latitude band (--band rows a band)
  R1g   static weights per lead, channel and grid point, shrunk toward R1 (the shrinkage is chosen by a two-fold
        cross-validation inside the validation year)
  R2s   gating network that sees the place and the lead only: learned static weight maps
  R2    gating network that also sees the experts' forecasts: the mixture of experts (weights per grid point,
        channel and situation)
  R2b   R2 with an additive bias map (the form of MoWE, arXiv 2509.09052)
  suffix c: the same with the daily climatology as one more expert. The damping control: a router whose gain is
        only a stronger damping must not beat a static blend that is allowed to damp as well.

Every variant is fitted on the validation year only and scored on the test year with the metrics of the score
(the same formulas); score_router.json / score_router_raw.npz have the format of the score, so
scripts/eval_kai_breakdown.py splits every pair into bias / information / amplitude.

    python scripts/kai_blend_router.py --dump runs/kaitend/dump_router --cache data/era5_5p625/kaiday \
        --stats runs/kaitend/stats.npz --experts kai_ro core_gngmz8 core_gngmz8s1 --out runs/kaitend/router_A
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from pgcfm import contract  # noqa: E402
from pgcfm.kai_tend import OROG, TISR  # noqa: E402
from run_kai_tendency import EPS, summarize  # noqa: E402

SCORED = [c for c in range(contract.N_CHANNELS) if c not in (OROG, TISR)]


# ------------------------------------------------------------------------------------------- static weights
def simplex_qp(E):
    """argmin_w w'Ew with w >= 0, sum w = 1, for stacked error moment matrices E (..., K, K) (Bates & Granger
    with the sign constraint). K is small: every subset S of experts is tried in closed form,
    w_S = E_S^-1 1 / (1' E_S^-1 1) with cost 1 / (1' E_S^-1 1); the best subset whose weights are >= 0 is the
    optimum of the convex problem."""
    K = E.shape[-1]
    best_w = np.zeros(E.shape[:-1])
    best_c = np.full(E.shape[:-2], np.inf)
    tr = np.trace(E, axis1=-2, axis2=-1)[..., None, None] / K
    for n in range(1, K + 1):
        for S in itertools.combinations(range(K), n):
            S = list(S)
            Es = E[..., S, :][..., :, S] + 1e-9 * tr * np.eye(n)
            x = np.linalg.solve(Es, np.ones(E.shape[:-2] + (n, 1)))[..., 0]
            sx = x.sum(-1)
            ok = (sx > 0) & (x / np.where(sx > 0, sx, 1.0)[..., None] >= -1e-9).all(-1)
            cost = np.where(ok, 1.0 / np.where(sx > 0, sx, 1.0), np.inf)
            take = cost < best_c
            w = np.zeros(E.shape[:-1])
            w[..., S] = np.clip(x / np.where(sx > 0, sx, 1.0)[..., None], 0.0, None)
            best_w = np.where(take[..., None], w, best_w)
            best_c = np.where(take, cost, best_c)
    return (best_w / best_w.sum(-1, keepdims=True)).astype(np.float32)


def cost_of(wgt, E):
    """w'Ew for weights (..., K) and moments (..., K, K)."""
    return np.einsum("...k,...kj,...j->...", wgt.astype(np.float64), E, wgt.astype(np.float64))


# ------------------------------------------------------------------------------------------- gating network
class Conv(nn.Module):
    """3 x 3 convolution on the sphere's grid: periodic in longitude, the edge row repeated at the poles."""

    def __init__(self, cin, cout):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, 3)

    def forward(self, x):
        return self.c(Fn.pad(Fn.pad(x, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="replicate"))


class Gate(nn.Module):
    """Mixture-of-experts router: softmax weights over K experts per channel and grid point (+ optional bias map).
    The logits start at the static weights w0 (lead, channel, K) and the last layer starts at zero, so the
    untrained router IS the static blend R0; training can only move away from it."""

    def __init__(self, K, C, L, static, w0, hidden=96, layers=3, see=True, bias=False):
        super().__init__()
        self.K, self.C, self.see = K, C, see
        self.register_buffer("static", static)  # (S, H, W)
        self.register_buffer("logw0", torch.log(torch.as_tensor(w0, dtype=torch.float32) + 1e-3))  # (L, C, K)
        cin = static.shape[0] + (K * C if see else 0)
        self.inp = Conv(cin, hidden)
        self.blocks = nn.ModuleList(Conv(hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.GroupNorm(8, hidden) for _ in range(layers + 1))
        self.film = nn.Embedding(L, 2 * hidden * (layers + 1))  # scale and shift of every layer from the lead
        nn.init.zeros_(self.film.weight)
        self.out = nn.Conv2d(hidden, K * C, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.bias = None
        if bias:
            self.bias = nn.Conv2d(hidden, C, 1)
            nn.init.zeros_(self.bias.weight)
            nn.init.zeros_(self.bias.bias)

    def forward(self, F, lead):
        """F (B, K, C, H, W) z-scored forecasts, lead (B,) 0-based -> blended forecast (B, C, H, W), weights."""
        B = F.shape[0]
        x = self.static[None].expand(B, -1, -1, -1)
        if self.see:
            x = torch.cat([F.flatten(1, 2), x], 1)
        film = self.film(lead).view(B, len(self.norms), 2, -1, 1, 1)
        h = self.inp(x)
        for i, n in enumerate(self.norms):
            if i:
                h = h + self.blocks[i - 1](h)
            h = Fn.gelu(n(h) * (1 + film[:, i, 0]) + film[:, i, 1])
        logits = self.out(h).view(B, self.K, self.C, *h.shape[-2:]) + self.logw0[lead].permute(0, 2, 1)[..., None, None]
        w = torch.softmax(logits, 1)
        y = (w * F).sum(1)
        if self.bias is not None:
            y = y + self.bias(h)
        return y, w


# ------------------------------------------------------------------------------------------- metrics (the score's)
def metrics_chunk(pred, truth, cl, w, regions):
    """The metrics of scripts/run_kai_tendency.py (score.metrics_chunk), physical units, (N, L, 67, H, W)."""
    err = pred - truth
    out = dict(mse=(err ** 2 * w).mean((-2, -1)).astype(np.float32))
    out["gm"] = (err * w).mean((-2, -1))
    pa, ta = pred - cl, truth - cl
    out["acc"] = ((pa * ta * w).sum((-2, -1))
                  / np.sqrt((pa ** 2 * w).sum((-2, -1)) * (ta ** 2 * w).sum((-2, -1)) + 1e-30)).astype(np.float32)
    mf, mo = (pa * w).mean((-2, -1), dtype=np.float64), (ta * w).mean((-2, -1), dtype=np.float64)
    vf = (pa ** 2 * w).mean((-2, -1), dtype=np.float64) - mf ** 2
    vo = (ta ** 2 * w).mean((-2, -1), dtype=np.float64) - mo ** 2
    cov = (pa * ta * w).mean((-2, -1), dtype=np.float64) - mf * mo
    sf, so = np.sqrt(np.maximum(vf, 0)), np.sqrt(np.maximum(vo, 0))
    out["dec"] = np.stack([(mf - mo) ** 2, (sf - so) ** 2, 2 * (sf * so - cov)], -1).astype(np.float32)
    out["act"] = (sf / (so + 1e-30)).astype(np.float32)
    out["mse_reg"] = np.stack([(err[:, :, :, r] ** 2 * w[r]).mean((-2, -1)) / w[r].mean() for r in regions],
                              -1).astype(np.float32)
    P = lambda f: (np.abs(np.fft.rfft(f, axis=-1)) ** 2 * w).mean(-2)  # noqa: E731
    out.update(sp_pred=P(pred), sp_true=P(truth), sp_err=P(err))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True, help="folder written by run_kai_tendency.py --score ... --dump")
    p.add_argument("--cache", required=True, help="the daily means <year>.npy (the truth)")
    p.add_argument("--stats", required=True, help="stats.npz of the run (mean, std, climatology)")
    p.add_argument("--experts", nargs="+", required=True, help="tags of the dump; the first is the base (KAI)")
    p.add_argument("--out", required=True)
    p.add_argument("--band", type=int, default=4, help="R1: latitude rows per band")
    p.add_argument("--variants", nargs="+", default=["R0", "R1", "R1g", "R2s", "R2", "R2b", "R0c", "R1c", "R2c"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="seeds of the gating networks (R2 gets all)")
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30, help="most epochs; the number used is chosen on held-out blocks")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--block", type=int, default=15, help="ICs per block of the split of the validation year")
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
    regions = [np.abs(lat) < 20, lat >= 20, lat <= -20]
    nc_ = len(clim) - 1
    H, W = len(lat), clim.shape[-1]
    C = contract.N_CHANNELS

    def year(tagy, ics):
        x = np.load(os.path.join(a.cache, f"{int(meta[tagy])}.npy"), mmap_mode="r")
        truth = np.stack([x[i + 1:i + 1 + L] for i in ics]).astype(np.float32)
        cl = np.stack([clim[[min(i + k, nc_) for k in range(1, L + 1)]] for i in ics]).astype(np.float32)
        return truth, cl

    ics_te, ics_va = meta["test_ics"], meta["val_ics"]
    truth_te, cl_te = year("test_year", ics_te)
    truth_va, cl_va = year("val_year", ics_va)
    z = lambda x: ((x - mn) / sd).astype(np.float16)  # noqa: E731
    Tz_va = z(truth_va)
    names = list(a.experts)
    Fva = [np.load(os.path.join(a.dump, f"{t}__val.npy")) for t in names]   # float16 (N, L, C, H, W)
    Fte = [np.load(os.path.join(a.dump, f"{t}__test.npy")) for t in names]
    Fva_c, Fte_c = Fva + [z(cl_va)], Fte + [z(cl_te)]                       # with the climatology as an expert
    del truth_va
    Nv, Nt = len(ics_va), len(ics_te)
    print(f"experts {names}; validation ICs {Nv}, test ICs {Nt}, grid {H} x {W}", flush=True)

    # ---- the error moments of the validation year per grid point, in two halves of alternating blocks
    blk = (np.arange(Nv) // a.block)
    half = blk % 2

    def moments(F):
        K = len(F)
        Eg = np.zeros((2, L, C, H, W, K, K))
        cnt = np.zeros(2)
        for i in range(0, Nv, 8):
            j = np.arange(i, min(Nv, i + 8))
            e = np.stack([f[j].astype(np.float32) for f in F]) - Tz_va[j].astype(np.float32)[None]  # (K, n, L, C, H, W)
            for h in (0, 1):
                m = half[j] == h
                if m.any():
                    Eg[h] += np.einsum("knlcyx,jnlcyx->lcyxkj", e[:, m], e[:, m], optimize=True)
                    cnt[h] += m.sum()
        return Eg / cnt[:, None, None, None, None, None, None], cnt

    bands = [np.arange(i, min(H, i + a.band)) for i in range(0, H, a.band)]
    wl = w[:, 0].astype(np.float64)

    def pool(Eg):  # per grid point -> global (L, C, K, K) and per band (L, C, B, K, K), area weights
        g = (Eg * wl[None, None, :, None, None, None]).mean((2, 3))
        b = np.stack([(Eg[:, :, r] * wl[None, None, r, None, None, None]).mean((2, 3)) / wl[r].mean() for r in bands], 2)
        return g, b

    def band_map(x):  # (L, C, B, ...) -> (L, C, H, 1, ...)
        out = np.zeros(x.shape[:2] + (H, 1) + x.shape[3:], dtype=x.dtype)
        for i, r in enumerate(bands):
            out[:, :, r, 0] = x[:, :, i][:, :, None]
        return out

    def static_fit(Eg, kind, alpha=0.0):
        """Weights (L, C, H or 1, W or 1, K) from the grid-point moments Eg (L, C, H, W, K, K)."""
        g, b = pool(Eg)
        if kind == "R0":
            return simplex_qp(g)[:, :, None, None]
        if kind == "R1":
            return band_map(simplex_qp(b))
        Es = (1 - alpha) * Eg + alpha * band_map(b)  # R1g: grid-point moments shrunk toward their band's
        return simplex_qp(Es)

    def gcost(wgt, Eg):  # area-mean cost (L, C) of weights (broadcast over the grid) on grid-point moments
        return (cost_of(np.broadcast_to(wgt, Eg.shape[:-1]), Eg) * wl[None, None, :, None]).mean((2, 3))

    res, info, preds = {}, {"experts": names, "variants": {}}, {}
    t0 = time.time()

    def score(name, pz):
        """pz (Nt, L, C, H, W) z-scored blended forecast of the test year."""
        parts = []
        for i in range(0, Nt, 40):
            s = slice(i, i + 40)
            parts.append(metrics_chunk(pz[s].astype(np.float32) * sd + mn, truth_te[s], cl_te[s], w, regions))
        out = {k: np.concatenate([q[k] for q in parts]) for k in ("mse", "acc", "dec", "act", "mse_reg")}
        out["gbias"] = (np.concatenate([q["gm"] for q in parts]).mean(0) / (std + EPS)).astype(np.float32)
        for k in ("sp_pred", "sp_true", "sp_err"):
            out[k] = np.concatenate([q[k] for q in parts]).mean(0).astype(np.float32)
        res[name] = out
        nr = (np.sqrt(out["mse"].mean(0))[:, SCORED] / std[SCORED]).mean(1)
        print(f"scored {name:14s} nrmse " + "/".join(f"{nr[q]:.4f}" for q in (0, 2, 4, 6, 9) if q < L)
              + "  act " + "/".join(f"{out['act'].mean(0)[q, SCORED].mean():.2f}" for q in (0, 4, 9) if q < L)
              + f"  [{time.time() - t0:.0f} s]", flush=True)

    def apply_static(wgt, F):  # weights (L, C, h, w, K) on forecasts F[k] (N, L, C, H, W)
        out = np.zeros(F[0].shape, dtype=np.float32)
        for k, f in enumerate(F):
            out += wgt[None, ..., k] * f.astype(np.float32)
        return out

    score("persistence", z(np.repeat(np.load(os.path.join(a.cache, f"{int(meta['test_year'])}.npy"),
                                             mmap_mode="r")[ics_te][:, None], L, 1).astype(np.float32)))
    score("climatology", Fte_c[-1])
    for k, t in enumerate(names):
        score(t, Fte[k])
    if "nrmse" in meta.files:  # the same numbers as the score that wrote the dump?
        mine = np.stack([(np.sqrt(res[t]["mse"].mean(0))[:, SCORED] / std[SCORED]).mean(1) for t in names])
        tags = [str(x) for x in meta["tags"]]
        ref = np.stack([meta["nrmse"][tags.index(t)] for t in names])
        info["max_abs_diff_to_score"] = float(np.abs(mine - ref).max())
        print("largest difference to the score's nRMSE of the experts:", info["max_abs_diff_to_score"], flush=True)

    # ---- static variants
    stat_w, mom = {}, {}
    for sfx, F, Ft in (("", Fva, Fte), ("c", Fva_c, Fte_c)):
        if not any(v in a.variants for v in (f"R0{sfx}", f"R1{sfx}", f"R1g{sfx}", f"R2s{sfx}", f"R2{sfx}", f"R2b{sfx}")):
            continue
        Eh, cnt = moments(F)
        Eg = (Eh * cnt[:, None, None, None, None, None, None]).sum(0) / cnt.sum()
        mom[sfx] = Eg
        s0 = gcost(static_fit(Eg, "R0"), Eg)  # (L, C) the static blend's MSE on the validation year
        cv = {}
        for kind, alphas in (("R0", [0.0]), ("R1", [0.0]), ("R1g", [0.0, 0.3, 0.6, 0.8, 0.9, 0.95])):
            for al in alphas:  # two-fold: fitted on one half of the blocks, cost on the other
                c = sum(gcost(static_fit(Eh[h], kind, al), Eh[1 - h]) * cnt[1 - h] for h in (0, 1)) / cnt.sum()
                cv[(kind, al)] = float((c / s0)[:, SCORED].mean())
        best_al = min((al for k, al in cv if k == "R1g"), key=lambda al: cv[("R1g", al)])
        info["variants"][f"cv{sfx}"] = {f"{k}@{al}": v for (k, al), v in cv.items()}
        print(f"two-fold CV inside the validation year (relative MSE, R0 fitted on all = 1){' +clim' if sfx else ''}:",
              {f"{k}@{al}": round(v, 4) for (k, al), v in cv.items()}, "-> R1g alpha", best_al, flush=True)
        for kind in ("R0", "R1", "R1g"):
            wgt = static_fit(Eg, kind, best_al)
            stat_w[kind + sfx] = wgt
            if kind + sfx in a.variants:
                pz = apply_static(wgt, Ft)
                score(kind + sfx, pz)
                info["variants"][kind + sfx] = {
                    "mean_weight": (np.broadcast_to(wgt, (L, C, H, W, len(F))) * w[None, None, :, :, None]
                                    ).mean((2, 3))[:, SCORED].mean(1).tolist(),
                    "alpha": best_al if kind == "R1g" else None}
                del pz
    np.savez_compressed(os.path.join(a.out, "static_weights.npz"), **stat_w)

    # ---- gating networks
    static = torch.tensor(np.stack(
        [np.sin(np.deg2rad(lat))[:, None].repeat(W, 1), np.cos(np.deg2rad(lat))[:, None].repeat(W, 1),
         np.sin(2 * np.pi * np.arange(W) / W)[None].repeat(H, 0), np.cos(2 * np.pi * np.arange(W) / W)[None].repeat(H, 0),
         Fva[0][0, 0, OROG].astype(np.float32)]), dtype=torch.float32)
    wt = torch.tensor(w)[None, None]                       # (1, 1, H, 1)
    sc = torch.tensor(SCORED)
    hold = blk % 4 == 3                                    # held-out blocks: the number of epochs is chosen there
    Tva = torch.from_numpy(Tz_va.astype(np.float32))

    def gate(name, F, Ft, sfx, see, bias, seed):
        K = len(F)
        w0 = stat_w["R0" + sfx][:, :, 0, 0]                # (L, C, K)
        s0 = torch.tensor(gcost(stat_w["R0" + sfx], mom[sfx]), dtype=torch.float32)  # (L, C)

        def batch_of(n, l):
            x = torch.from_numpy(np.stack([f[n, l] for f in F], 1).astype(np.float32))
            return x, Tva[n, l], torch.as_tensor(l)

        def loss_of(m, n, l):
            x, t, ld = batch_of(n, l)
            y, _ = m(x, ld)
            mse = (((y - t) ** 2) * wt).mean((-2, -1))     # (B, C)
            return (mse / s0[ld])[:, sc].mean()

        def run(train_ics, epochs, eval_ics=None):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            m = Gate(K, C, L, static, w0, a.hidden, a.layers, see, bias)
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
                    ev = np.array([(n, l) for n in eval_ics for l in range(L)])
                    with torch.no_grad():
                        tot = sum(float(loss_of(m, ev[i:i + 64, 0], ev[i:i + 64, 1])) * len(ev[i:i + 64])
                                  for i in range(0, len(ev), 64)) / len(ev)
                    curve.append(tot)
            return m, curve

        tr_ics, ho_ics = np.where(~hold)[0], np.where(hold)[0]
        m, curve = run(tr_ics, a.epochs, ho_ics)
        with torch.no_grad():  # the held-out loss of the untrained router = the static blend on those blocks
            m0 = Gate(K, C, L, static, w0, a.hidden, a.layers, see, bias).eval()
            ev = np.array([(n, l) for n in ho_ics for l in range(L)])
            base = sum(float(loss_of(m0, ev[i:i + 64, 0], ev[i:i + 64, 1])) * len(ev[i:i + 64])
                       for i in range(0, len(ev), 64)) / len(ev)
        best = int(np.argmin(curve)) + 1
        use = best if curve[best - 1] < base else 0        # no epoch better than the static blend: keep the static blend
        m, _ = run(np.arange(Nv), use)                     # the router that is scored: all blocks, that many epochs
        m.eval()
        pz = np.zeros((Nt, L, C, H, W), dtype=np.float32)
        wsum, wsq, wtime = np.zeros((L, K)), np.zeros((L, K)), np.zeros((L, K))
        with torch.no_grad():
            for l in range(L):
                ws = []
                for i in range(0, Nt, 32):
                    n = np.arange(i, min(Nt, i + 32))
                    x = torch.from_numpy(np.stack([f[n, l] for f in Ft], 1).astype(np.float32))
                    y, wg = m(x, torch.full((len(n),), l))
                    pz[n, l] = y.numpy()
                    ws.append(wg[:, :, sc].numpy())
                ws = np.concatenate(ws)                    # (Nt, K, 65, H, W)
                wsum[l] = (ws * w[None, None, None]).mean((0, 2, 3, 4))
                wsq[l] = ws.mean(0).std((-2, -1)).mean(-1)     # spread of the time-mean weight over the grid
                wtime[l] = ws.std(0).mean((1, 2, 3))           # spread over the forecasts (the situations) at a place
        score(name, pz)
        info["variants"][name] = {
            "holdout_curve": curve, "holdout_static": base, "epochs_used": use, "seed": seed,
            "params": sum(q.numel() for q in m.parameters()), "mean_weight": wsum.tolist(),
            "weight_spread_space": wsq.tolist(), "weight_spread_time": wtime.tolist()}
        print(f"  {name}: held-out loss static {base:.4f}, best {min(curve):.4f} at epoch {best}, used {use} epochs; "
              f"weight spread over the grid {wsq.mean():.3f}, over the forecasts {wtime.mean():.3f}", flush=True)

    for sfx, F, Ft in (("", Fva, Fte), ("c", Fva_c, Fte_c)):
        for v, see, bias in (("R2s", False, False), ("R2", True, False), ("R2b", True, True)):
            if v + sfx not in a.variants:
                continue
            for seed in (a.seeds if v == "R2" else a.seeds[:1]):
                gate(v + sfx + ("" if seed == a.seeds[0] else f"_s{seed}"), F, Ft, sfx, see, bias, seed)

    np.savez_compressed(os.path.join(a.out, "score_router_raw.npz"),
                        **{f"{k}__{q}": v for k, d in res.items() for q, v in d.items() if isinstance(v, np.ndarray)})
    if a.pairs is None:
        a.pairs = [f"{x}:{y}" for x, y in (
            ("R0", names[0]), ("R1", "R0"), ("R1g", "R0"), ("R2s", "R0"), ("R2", "R0"), ("R2b", "R0"), ("R2", "R1"),
            ("R2", "R2s"), ("R2b", "R2"), ("R0c", "R0"), ("R1c", "R0c"), ("R2c", "R0c"), ("R2", "R0c"), ("R2c", "R2"))]
    ns = argparse.Namespace(out=a.out, leads=L, pairs=a.pairs)
    summarize(ns, res, std, ics_te)
    os.replace(os.path.join(a.out, "score.json"), os.path.join(a.out, "score_router.json"))
    with open(os.path.join(a.out, "router_info.json"), "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    main()
