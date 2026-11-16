"""Is the ACC gain of a smoother forecast information, or the effect of the smoothing itself?

A forecast trained far into the unpredictable range is smoother (lower activity), and a smoother forecast has a higher
ACC without knowing more: damping the scales that carry no skill raises the pattern correlation. Each arm (a core TAG
or a two-stage forecast ``A>B@n``, as in scripts/eval_kai_ensemble.py --stages) is compared with the control at EQUAL
smoothing, on the anomalies from the climatology:

* ``match``   the control filtered so that its zonal power spectrum (per lead, channel and latitude) equals the arm's:
              real gains sqrt(P_arm / P_ctrl) per zonal wavenumber, no phase change. arm - match > 0: the arm is
              better than the control made as smooth as the arm.
* ``wiener``  the best real gain per (lead, channel, latitude, zonal wavenumber) for the control AND for the arm,
              g = Re <conj(F) T> / <|F|^2> >= 0, estimated on one half of the ICs (alternating blocks of ``--fold``
              ICs) and applied to the other half. wiener(arm) - wiener(ctrl): what the arm knows more than the control
              when each is damped as well as a zonal scale filter can.

Gains are pooled over ``--pool`` latitude rows on each side. Limits: the filters see zonal scales only (meridional
smoothing is matched in variance per latitude, not in shape). Per lead: nRMSE, ACC, activity (the sums of
eval_kai_ensemble.py) of ctrl / arm / match / wiener(ctrl) / wiener(arm), paired differences with 95 % bootstrap
intervals over the ICs, and the split bias / pattern / amplitude (scripts/eval_kai_breakdown.py) of arm vs ctrl and of
arm vs match; each row also with RMSE (physical units), ACC and activity of t2m, t850, z500, u850, msl. Forecasts are cached in ``<out>/fc_<arm>_<year>.npy`` (float16).

    python scripts/diag_stage_filter.py --out runs/kaitend --arms 'core_gbv>core_s2v@5' core_gbv --ctrl core_gb
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_kai_tendency as R  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.kai_tend import DYN, TISR, core_rollout  # noqa: E402

VARS = ["t2m", "t850", "z500", "u850", "msl"]


def terms(pa, ta, w):
    """Anomalies (N, L, C, H, W) -> per forecast (N, L, C): MSE, its three terms and the sums of ACC / activity."""
    wm = lambda x: (x.double() * w).mean((-2, -1))  # noqa: E731
    mse = wm((pa - ta) ** 2)
    mf, mo = wm(pa), wm(ta)
    pp, tt, pt = wm(pa ** 2), wm(ta ** 2), wm(pa * ta)
    vf, vo, cov = pp - mf ** 2, tt - mo ** 2, pt - mf * mo
    info = vo - cov ** 2 / vf.clamp_min(1e-30)
    bias = (mf - mo) ** 2
    return {k: v.numpy() for k, v in dict(mse=mse, bias=bias, info=info, amp=mse - bias - info, pp=pp, tt=tt, pt=pt).items()}


def pool(x, k):
    """Running mean over the latitude rows (dim -2) with k rows on each side, shorter at the two ends."""
    if k <= 0:
        return x
    H = x.shape[-2]
    cs = torch.cat([torch.zeros_like(x[..., :1, :]), x.cumsum(-2)], -2)
    lo, hi = (torch.arange(H) - k).clamp(min=0), (torch.arange(H) + k + 1).clamp(max=H)
    return (cs[..., hi, :] - cs[..., lo, :]) / (hi - lo).view(-1, 1)


def match(ctrl, arm, k):
    """The control with the arm's zonal power spectrum (per lead, channel, latitude)."""
    out = torch.empty_like(ctrl)
    for l in range(ctrl.shape[1]):
        Fc, Fa = torch.fft.rfft(ctrl[:, l], dim=-1), torch.fft.rfft(arm[:, l], dim=-1)
        g = torch.sqrt(pool((Fa.abs() ** 2).mean(0), k) / pool((Fc.abs() ** 2).mean(0), k).clamp_min(1e-20))
        out[:, l] = torch.fft.irfft(g * Fc, n=ctrl.shape[-1], dim=-1)
    return out


def wiener(f, t, fold, k):
    """The forecast with the best real zonal gains, estimated on the other fold of the ICs."""
    out = torch.empty_like(f)
    for l in range(f.shape[1]):
        F, T = torch.fft.rfft(f[:, l], dim=-1), torch.fft.rfft(t[:, l], dim=-1)
        for q in (0, 1):
            fit, use = fold != q, fold == q
            num, den = (F[fit].conj() * T[fit]).real.mean(0), (F[fit].abs() ** 2).mean(0)
            g = (pool(num, k) / pool(den, k).clamp_min(1e-20)).clamp(0, 2)
            out[use, l] = torch.fft.irfft(g * F[use], n=f.shape[-1], dim=-1)
    return out


def main():
    p = R.parser()
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--arms", nargs="+", required=True, help="core TAG or two-stage A>B@n")
    p.add_argument("--ctrl", required=True)
    p.add_argument("--m2", default=None, help="tables of scripts/precompute_m2.py for a second stage with 48 channels")
    p.add_argument("--pool", type=int, default=2)
    p.add_argument("--fold", type=int, default=10)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    tr = R.load_years(a, a.train_years)
    mean, std, inc, clim = R.stats(a, tr)
    del tr
    te = R.load_years(a, [a.test_year])[a.test_year]
    lat = R.lat_of(os.path.join(a.data, f"{a.test_year}.nc"), te.shape[-2], a.era5)
    tez = torch.from_numpy(R.zscore(te, mean, std))
    cz = torch.from_numpy(R.zscore(clim, mean, std))[:, DYN]
    L = a.leads
    ics = np.arange(0, len(tez) - L, a.ic_stride)
    N = len(ics)
    w = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)
    w = (w / w.mean()).view(1, -1, 1)
    sc = [i for i, c in enumerate(DYN) if c != TISR]
    vi = [DYN.index(contract.CHANNEL_INDEX[v]) for v in VARS]
    vstd = np.asarray(std[[contract.CHANNEL_INDEX[v] for v in VARS]], dtype=np.float64).reshape(-1)
    C = torch.stack([cz[(torch.tensor(ics) + k + 1) % len(cz)] for k in range(L)], 1)   # (N, L, 66, H, W)
    TA = torch.stack([tez[ics + k + 1] for k in range(L)], 1)[:, :, DYN] - C            # the truth's anomalies
    fold = torch.from_numpy((np.arange(N) // a.fold) % 2)
    m2_tab = None

    def forecast(spec):
        nonlocal m2_tab
        path = os.path.join(a.out, f"fc_{spec.replace('>', '-').replace('@', '_')}_{a.test_year}.npy")
        if os.path.exists(path):
            y = np.load(path)
            if y.shape[:2] == (N, L):
                return torch.from_numpy(y.astype(np.float32))
        core2, n1 = None, 0
        if ">" in spec:
            ta, rest = spec.split(">")
            tb, n1 = rest.rsplit("@", 1)
            n1 = int(n1)
            core = R.build_core(a, mean, std, inc, lat, load=True, tag=ta)
            core2 = R.build_core(a, mean, std, inc, lat, load=True, tag=tb)
            core2.noise_on = False
        else:
            core = R.build_core(a, mean, std, inc, lat, load=True, tag=spec)
        core.noise_on = False
        if getattr(core, "nz", 0):
            core.features.noise_on = False
        ys = []
        for i in range(0, N, a.chunk):
            j = ics[i:i + a.chunk]
            x0, doy = tez[j][:, DYN], torch.tensor(j)
            tis = torch.stack([tez[j + k + 1] for k in range(L)], 1)[:, :, TISR]
            with torch.no_grad():
                if core2 is None:
                    y = core_rollout(core, x0, doy, L, tis)
                else:
                    ya = core_rollout(core, x0, doy, n1, tis[:, :n1])
                    both = getattr(core, "entv", False) and getattr(core2, "entv", False)
                    core2.ent_init = core.ent_s2.clone() if both else None
                    if a.m2 and getattr(core2, "nz", 0) == 48:
                        if m2_tab is None:
                            m2_tab = np.load(f"{a.m2}_{a.test_year}.npy", mmap_mode="r")
                        core2.features.ext = [torch.from_numpy(np.asarray(m2_tab[j, l], np.float32))
                                              for l in range(L - n1)]
                    y = torch.cat([ya, core_rollout(core2, ya[:, -1], doy + n1, L - n1, tis[:, n1:])], 1)
            ys.append(y.float())
            print(spec, i + len(j), "/", N, flush=True)
        y = torch.cat(ys)
        np.save(path, y.numpy().astype(np.float16))
        return y

    def stat(X, ix=None):
        """(L,) nRMSE, ACC, activity over the scored channels of the ICs ix (sums as in eval_kai_ensemble.py)."""
        s = (lambda k: X[k].mean(0)) if ix is None else (lambda k: X[k][ix].mean(0))
        return (np.sqrt(s("mse"))[:, sc].mean(1), (s("pt") / np.sqrt(s("pp") * s("tt")))[:, sc].mean(1),
                np.sqrt(s("pp") / s("tt"))[:, sc].mean(1))

    def stat_var(X, ix=None):
        """(L, 5) RMSE (physical units) and ACC of the five variables."""
        s = (lambda k: X[k].mean(0)[:, vi]) if ix is None else (lambda k: X[k][ix].mean(0)[:, vi])
        return np.sqrt(s("mse")) * vstd, s("pt") / np.sqrt(s("pp") * s("tt"))

    def diff(A, B, nb=500):
        """A over B: nRMSE gain (%) and ACC difference, each with a 95 % bootstrap interval over the ICs; the same
        for the five variables (``vars``)."""
        (ra, ca, _), (rb, cb, _) = stat(A), stat(B)
        (va, wa), (vb, wb) = stat_var(A), stat_var(B)
        rng, bs, bv = np.random.default_rng(0), [], []
        for _ in range(nb):
            ix = rng.integers(0, N, N)
            (r1, c1, _), (r2, c2, _) = stat(A, ix), stat(B, ix)
            bs.append(np.stack([100 * (1 - r1 / r2), c1 - c2]))
            (v1, w1), (v2, w2) = stat_var(A, ix), stat_var(B, ix)
            bv.append(np.stack([100 * (1 - v1 / v2), w1 - w2]))
        lo, hi = np.percentile(np.stack(bs), [2.5, 97.5], axis=0)
        vlo, vhi = np.percentile(np.stack(bv), [2.5, 97.5], axis=0)  # (2, L, 5)
        o = {"gain_pct": (100 * (1 - ra / rb)).tolist(), "gain_lo": lo[0].tolist(), "gain_hi": hi[0].tolist(),
             "dacc": (ca - cb).tolist(), "dacc_lo": lo[1].tolist(), "dacc_hi": hi[1].tolist(), "vars": {}}
        for q, v in enumerate(VARS):
            o["vars"][v] = {"gain_pct": (100 * (1 - va / vb))[:, q].tolist(), "gain_lo": vlo[0][:, q].tolist(),
                            "gain_hi": vhi[0][:, q].tolist(), "dacc": (wa - wb)[:, q].tolist(),
                            "dacc_lo": vlo[1][:, q].tolist(), "dacc_hi": vhi[1][:, q].tolist()}
        return o

    def split(A, B):
        ra, rb = np.sqrt(A["mse"].mean(0)), np.sqrt(B["mse"].mean(0))  # (L, C)
        con = lambda k: 100 * ((B[k].mean(0) - A[k].mean(0))[:, sc] / (ra + rb)[:, sc]).mean(1) / rb[:, sc].mean(1)  # noqa: E731
        return {"total": (100 * (1 - ra[:, sc].mean(1) / rb[:, sc].mean(1))).tolist(), "info": con("info").tolist(),
                "amp": con("amp").tolist(), "bias": con("bias").tolist()}

    g = lambda v, d=3: " ".join(f"{x:.{d}f}" for x in v)  # noqa: E731
    f = lambda v, d=1: " ".join(f"{x:+.{d}f}" for x in v)  # noqa: E731
    out = {"ctrl": a.ctrl, "ics": int(N), "pool": a.pool, "fold": a.fold, "arms": {}}
    ctrl = forecast(a.ctrl) - C
    Tc = terms(ctrl, TA, w)
    Twc = terms(wiener(ctrl, TA, fold, a.pool), TA, w)

    def row(X):
        """The three scores over the scored channels, and RMSE (physical units) / ACC / activity of the five variables."""
        o = dict(zip(("nrmse", "acc", "act"), (v.tolist() for v in stat(X))))
        m = lambda k: X[k].mean(0)  # noqa: E731
        o["vars"] = {v: {"rmse": (np.sqrt(m("mse")[:, vi[q]]) * vstd[q]).tolist(),
                         "acc": (m("pt") / np.sqrt(m("pp") * m("tt")))[:, vi[q]].tolist(),
                         "act": np.sqrt(m("pp") / m("tt"))[:, vi[q]].tolist()} for q, v in enumerate(VARS)}
        return o

    out["ctrl_rows"] = {"ctrl": row(Tc), "wiener_ctrl": row(Twc)}
    json_path = a.json or os.path.join(a.out, "diag_stage_filter.json")
    for spec in a.arms:
        arm = forecast(spec) - C
        Ta = terms(arm, TA, w)
        Tm = terms(match(ctrl, arm, a.pool), TA, w)
        Twa = terms(wiener(arm, TA, fold, a.pool), TA, w)
        del arm
        rows = {"ctrl": Tc, "arm": Ta, "match": Tm, "wiener_ctrl": Twc, "wiener_arm": Twa}
        o = {"rows": {k: row(v) for k, v in rows.items()},
             "arm_vs_ctrl": diff(Ta, Tc), "arm_vs_match": diff(Ta, Tm), "wiener_arm_vs_wiener_ctrl": diff(Twa, Twc),
             "split_arm_vs_ctrl": split(Ta, Tc), "split_arm_vs_match": split(Ta, Tm),
             "split_wiener": split(Twa, Twc)}
        out["arms"][spec] = o
        print(f"== {spec} vs {a.ctrl}, day 1..{L}")
        for k, v in o["rows"].items():
            print(f"  {k:12s} nRMSE {g(v['nrmse'])}\n  {'':12s} ACC   {g(v['acc'])}\n  {'':12s} act   {g(v['act'], 2)}")
        for k in ("arm_vs_ctrl", "arm_vs_match", "wiener_arm_vs_wiener_ctrl"):
            d = o[k]
            print(f"  {k}: nRMSE gain % {f(d['gain_pct'])}\n      95 % low {f(d['gain_lo'])}\n      95 % high {f(d['gain_hi'])}"
                  f"\n    ACC diff {f(d['dacc'], 3)}\n      95 % low {f(d['dacc_lo'], 3)}\n      95 % high {f(d['dacc_hi'], 3)}")
        for k in ("arm_vs_ctrl", "arm_vs_match", "wiener_arm_vs_wiener_ctrl"):
            for v in VARS:
                d = o[k]["vars"][v]
                print(f"  {k} {v}: RMSE gain % {f(d['gain_pct'])}\n      95 % low {f(d['gain_lo'])}  high {f(d['gain_hi'])}"
                      f"\n    ACC diff {f(d['dacc'], 3)}\n      95 % low {f(d['dacc_lo'], 3)}  high {f(d['dacc_hi'], 3)}")
        for k in ("split_arm_vs_ctrl", "split_arm_vs_match", "split_wiener"):
            s = o[k]
            print(f"  {k}: total {f(s['total'])}\n    pattern {f(s['info'])}\n    amplitude {f(s['amp'])}\n    bias {f(s['bias'])}")
        sys.stdout.flush()
        json.dump(out, open(json_path, "w"), indent=1)


if __name__ == "__main__":
    main()
