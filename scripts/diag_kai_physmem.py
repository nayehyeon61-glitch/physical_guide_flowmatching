"""Do the physical terms G(x) need memory terms next to them? (Mori-Zwanzig diagnostic on the daily means)

    python scripts/diag_kai_physmem.py [--cores core_gn core_gnh] [--year 2016] [--lags 6] [--out runs/kaitend]

G(x) = -R0(x) is the discrete primitive-equation operator's own tendency of (u, v, T, q) on the 12 levels
(48 channels, pgcfm.dyncore.OperatorFeatures, step 18). On daily means it is the Markov term of the resolved
state evaluated with the resolved state alone: what the sub-daily and sub-grid motion adds is, in the
Mori-Zwanzig equation, the memory integral + the noise. Measured here per channel (pooled over the valid points
and the days, cos-lat weighted, G as an anomaly from its annual mean field):

  A  the observed daily increment d_t = x_{t+1} - x_t against the physical terms
       corr(d_t, G_{t-k})                         k = 0 .. S
       explained fraction of d_t by G_t .. G_{t-m}  m = 0 .. S   (m = 0: the physical term alone; m > 0: with
                                                                 a linear memory of the physical term)
  B  the one-step residual of a trained core r_t = x_{t+1} - core(x_t) against the same terms: what a core
     that sees x_t alone has left for the physical term (m = 0), for its memory (m > 0), for the memory of its
     own residuals (r_{t-1}, r_{t-2}) and for both together.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_kai_tendency as R  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.coupling import LEVEL_DYN  # noqa: E402
from pgcfm.dyncore import G_EQS  # noqa: E402
from pgcfm.kai_tend import DYN, TISR, core_rollout, operator_features  # noqa: E402

EQ_VAR = {"u": "u", "v": "v", "T": "t", "q": "q"}


def explained(y, Z, w):
    """Fraction of the weighted variance of y (N, C, H, W) explained by the first m of the predictors Z (list of
    (N, C, H, W)), per channel, for m = 1 .. len(Z): (len(Z), C)."""
    P, C = len(Z), y.shape[1]
    A, b = np.zeros((C, P, P)), np.zeros((C, P))
    tot = (y ** 2 * w).sum((0, 2, 3)).numpy()
    for p_ in range(P):
        b[:, p_] = (y * Z[p_] * w).sum((0, 2, 3)).numpy()
        for q in range(p_, P):
            A[:, p_, q] = A[:, q, p_] = (Z[p_] * Z[q] * w).sum((0, 2, 3)).numpy()
    r2 = np.zeros((P, C))
    for m in range(1, P + 1):
        for c in range(C):
            coef = np.linalg.solve(A[c, :m, :m] + 1e-9 * np.eye(m) * (A[c, 0, 0] + 1e-30), b[c, :m])
            r2[m - 1, c] = float(coef @ b[c, :m] / tot[c])
    return r2


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cores", nargs="*", default=[])
    p.add_argument("--year", type=int, default=2016)
    p.add_argument("--lags", type=int, default=6)
    p.add_argument("--threads", type=int, default=1)
    a0, rest = p.parse_known_args(argv)
    a = R.parser().parse_args(rest)
    torch.set_num_threads(a0.threads)
    mean, std, inc, _ = R.stats(a)
    x = R.load_years(a, [a0.year])[a0.year]
    lat = R.lat_of(os.path.join(a.data, f"{a0.year}.nc"))
    contract.register_latitudes(lat)
    xz = torch.from_numpy(R.zscore(x, mean, std))
    dyn = torch.tensor(DYN)
    S, T = a0.lags, len(xz)
    idx = torch.tensor(sum((LEVEL_DYN[EQ_VAR[e]] for e in G_EQS), []))  # the 48 channels with an equation
    names = [f"{e}{lv}" for e in G_EQS for lv in contract.LEVELS_HPA]
    groups = {e: list(range(i * 12, (i + 1) * 12)) for i, e in enumerate(G_EQS)}
    wl = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)
    wl = (wl / wl.mean()).view(1, 1, -1, 1)
    s = torch.as_tensor(inc[DYN], dtype=torch.float32)[idx].view(1, -1, 1, 1)

    feats = operator_features(mean, std, xz[:1], states=xz[::4, dyn])
    with torch.no_grad():
        g, mask = [], []
        for i in range(0, T, 32):
            raw, m = feats.raw(xz[i:i + 32, dyn])
            g.append((raw / feats.scale.double()).clamp(-feats.clamp, feats.clamp).float() * m)
            mask.append(m.float())
    g, mask = torch.cat(g), torch.cat(mask)
    g = (g - (g * mask).sum(0, keepdim=True) / mask.sum(0, keepdim=True).clamp_min(1.0)) * mask  # anomaly
    d = (xz[1:, dyn][:, idx] - xz[:-1, dyn][:, idx]) / s  # d[t] = x_{t+1} - x_t
    w = (wl * mask[:-1])  # valid points of the day the step starts from

    def corr(u, v, ww):
        return ((u * v * ww).sum((0, 2, 3)) / torch.sqrt((u ** 2 * ww).sum((0, 2, 3)) * (v ** 2 * ww).sum((0, 2, 3))
                                                         + 1e-30)).numpy()

    def lagged(y, t0):  # y[n] belongs to the step from day t0 + n; predictors G_{t-k}, k = 0 .. S
        n0 = max(0, S - t0)  # first usable n
        yy, ww = y[n0:], w[t0 + n0:t0 + n0 + len(y) - n0]
        Z = [g[t0 + n0 - k:t0 + n0 - k + len(yy)] for k in range(S + 1)]
        return yy, Z, ww

    def table(title, v, fmt="{:+.3f}"):
        print(f"  {title}")
        print("    " + " " * 6 + " ".join(f"{k:7d}" for k in range(v.shape[0])))
        print("    all   " + " ".join(fmt.format(v[k].mean()).rjust(7) for k in range(v.shape[0])))
        for e, c in groups.items():
            print(f"    {e:5s} " + " ".join(fmt.format(v[k][c].mean()).rjust(7) for k in range(v.shape[0])))

    out = dict(names=names, lags=S, year=a0.year, scale=feats.scale.view(-1).tolist())
    yy, Z, ww = lagged(d, 0)
    cg = np.stack([corr(yy, z, ww) for z in Z])
    r2 = explained(yy, Z, ww)
    out["increment"] = dict(corr=cg.tolist(), r2=r2.tolist())
    print(f"A  observed daily increment d_t against the physical terms G (48 channels, {len(yy)} days of {a0.year})")
    table("corr(d_t, G_t-k), k =", cg)
    table("explained % of d_t by G_t .. G_t-m, m =", 100 * r2, "{:.1f}")

    out["cores"] = {}
    for tag in a0.cores:
        core = R.build_core(a, mean, std, inc, lat, tag=tag)
        W = getattr(core, "mem", 0)
        i0 = max(W, 1)
        res = []
        with torch.no_grad():
            for i in range(i0, T - 1, 16):
                j = torch.arange(i, min(i + 16, T - 1))
                prev = None
                if W:
                    prev = torch.stack([xz[j - k][:, dyn] for k in range(W, 0, -1)], 1)
                elif getattr(core, "hist", False):
                    prev = xz[j - 1][:, dyn]
                y = core_rollout(core, xz[j][:, dyn], j, 1, xz[j + 1][:, None, TISR], prev=prev)[:, 0]
                res.append((xz[j + 1][:, dyn] - y)[:, idx] / s)
        r = torch.cat(res)  # r[n] = residual of the step from day i0 + n
        yy, Z, ww = lagged(r, i0)
        n0 = len(r) - len(yy)
        cg = np.stack([corr(yy, z, ww) for z in Z])
        r2g = explained(yy, Z, ww)
        # the memory of the core's own residuals (2 lags) and both memories together; the first two steps of the
        # residual sequence have no two predecessors
        k0 = max(2 - n0, 0)
        y2, w2 = yy[k0:], ww[k0:]
        Zr = [r[n0 + k0 - k:n0 + k0 - k + len(y2)] for k in (1, 2)]
        r2r = explained(y2, Zr, w2)[-1]
        r2b = explained(y2, Zr + [z[k0:] for z in Z], w2)[-1]
        rms = float(torch.sqrt((r ** 2 * w[i0:i0 + len(r)]).sum() / w[i0:i0 + len(r)].sum()))
        out["cores"][tag] = dict(res_rms=rms, corr=cg.tolist(), r2_g=r2g.tolist(), r2_res=r2r.tolist(),
                                 r2_both=r2b.tolist())
        print(f"\nB  one-step residual of {tag} (RMS {rms:.4f} increment-std units on the 48 channels) against G")
        table("corr(r_t, G_t-k), k =", cg)
        table("explained % of r_t by G_t .. G_t-m, m =", 100 * r2g, "{:.1f}")
        f = lambda v: f"{100 * v.mean():.1f} (" + ", ".join(f"{e} {100 * v[c].mean():.1f}" for e, c in groups.items()) + ")"  # noqa: E731
        print(f"  explained % of r_t by r_t-1, r_t-2 (memory of the residuals)       {f(r2r)}")
        print(f"  explained % of r_t by both (G_t .. G_t-{S} and r_t-1, r_t-2)          {f(r2b)}")
    with open(os.path.join(a.out, "diag_physmem.json"), "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
