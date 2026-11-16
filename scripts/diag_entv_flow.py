"""Two checks of a deterministic entropy core (``--core-entv``), one 10-day run per IC of the test year.

1. ``--ctrl TAG``: the gain of the core over the control split into bias / pattern (information) / amplitude cost,
   the same split and the same contribution formula as scripts/eval_kai_breakdown.py (per forecast, on the
   anomalies from the climatology, area weights; contributions add up to the gain of the 65-channel nRMSE).
2. Does the entropy state know more than the lead time? s^2 = the entropy state (z-score units), e^2 = the squared
   error of the same run. Per lead:
   * ``ssr``      sqrt(mean s^2 / mean e^2) over the scored channels (1 = calibrated on average),
   * ``static``   correlation over (channel, grid point) of log(mean_ic s^2) with log(mean_ic e^2): does the state
                  know WHERE the error is large (a fixed map would do that as well),
   * ``case_sd``  std over ICs of log(s^2 / mean_ic s^2): how much the state changes from case to case at all,
   * ``bins``     the case part: points sorted by s^2 / mean_ic s^2 into 5 bins; mean of that ratio and mean of
                  e^2 / mean_ic e^2 in each bin. A state that knows the case has rising error ratios (slope 1 = exact),
   * ``slope``    least-squares slope of the 5 bin means of the error ratio on those of the state ratio.

    python scripts/diag_entv_flow.py --out runs/kaitend --tag core_gbv --ctrl core_gb
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_kai_tendency as R  # noqa: E402
from pgcfm.kai_tend import DYN, TISR, core_rollout  # noqa: E402


def terms(y, t, c, w):
    """Per forecast (N, L, C): MSE and its three terms bias, s_o^2 (1 - r^2), s_o^2 (a - r)^2; activity, ACC."""
    wm = lambda x: (x.double() * w).mean((-2, -1))  # noqa: E731
    pa, ta = y - c, t - c
    mse = wm((y - t) ** 2)
    mf, mo = wm(pa), wm(ta)
    vf, vo = wm(pa ** 2) - mf ** 2, wm(ta ** 2) - mo ** 2
    cov = wm(pa * ta) - mf * mo
    info = vo - cov ** 2 / vf.clamp_min(1e-30)
    bias = (mf - mo) ** 2
    acc = wm(pa * ta) / torch.sqrt(wm(pa ** 2) * wm(ta ** 2) + 1e-30)
    return dict(mse=mse, bias=bias, info=info, amp=mse - bias - info, act=torch.sqrt(vf / vo.clamp_min(1e-30)), acc=acc)


def main():
    p = R.parser()
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--ctrl", default=None, help="control core of the split")
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
    w = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32)
    w = (w / w.mean()).view(1, -1, 1)
    sc = [i for i, c in enumerate(DYN) if c != TISR]
    out = {"tag": a.tag, "ctrl": a.ctrl, "ics": int(len(ics)), "leads": {}}
    T, S, E = {}, [], []
    for tag in [a.tag] + ([a.ctrl] if a.ctrl else []):
        core = R.build_core(a, mean, std, inc, lat, load=True, tag=tag)
        core.noise_on = False
        entv = getattr(core, "entv", False)
        parts = []
        for i in range(0, len(ics), a.chunk):
            j = ics[i:i + a.chunk]
            x0 = tez[j][:, DYN]
            tgt = torch.stack([tez[j + k + 1] for k in range(L)], 1)
            c = torch.stack([cz[(torch.tensor(j) + k + 1) % len(cz)] for k in range(L)], 1)
            with torch.no_grad():
                y = core_rollout(core, x0, torch.tensor(j), L, tgt[:, :, TISR])
            parts.append(terms(y, tgt[:, :, DYN], c, w))
            if entv and tag == a.tag:
                s2 = torch.exp(torch.stack(list(core.v_log), 1)) * core.t6.cpu().view(1, 1, -1, 1, 1) ** 2
                S.append(s2[:, :, sc].float())
                E.append(((y - tgt[:, :, DYN]) ** 2)[:, :, sc].float())
            print(tag, i + len(j), "/", len(ics), flush=True)
        T[tag] = {k: torch.cat([q[k] for q in parts]).numpy() for k in parts[0]}
    D = [0, 2, 4, 6, 9]
    f = lambda v, d=1: "/".join(f"{x:+.{d}f}" for x in np.asarray(v)[D])  # noqa: E731
    g = lambda v, d=3: "/".join(f"{x:.{d}f}" for x in np.asarray(v)[D])  # noqa: E731
    for tag, t in T.items():
        r = np.sqrt(t["mse"].mean(0))[:, sc].mean(1)
        out[f"nrmse_{tag}"], out[f"acc_{tag}"] = r.tolist(), t["acc"].mean(0)[:, sc].mean(1).tolist()
        out[f"act_{tag}"] = t["act"].mean(0)[:, sc].mean(1).tolist()
        print(f"{tag}: nRMSE {g(r)}  ACC {g(out[f'acc_{tag}'])}  activity {g(out[f'act_{tag}'], 2)}")
    if a.ctrl:
        A, B = T[a.tag], T[a.ctrl]
        ra, rb = np.sqrt(A["mse"].mean(0)), np.sqrt(B["mse"].mean(0))  # (L, C)
        con = lambda k: 100 * ((B[k].mean(0) - A[k].mean(0))[:, sc] / (ra + rb)[:, sc]).mean(1) / rb[:, sc].mean(1)  # noqa: E731
        sp = {k: con(k) for k in ("bias", "info", "amp")}
        sp["total"] = 100 * (1 - ra[:, sc].mean(1) / rb[:, sc].mean(1))
        out["split_pct"] = {k: v.tolist() for k, v in sp.items()}
        print(f"gain of {a.tag} over {a.ctrl}, % of the control's nRMSE (day 1/3/5/7/10):")
        for k, n in (("total", "total"), ("info", "pattern (information)"), ("amp", "amplitude cost"), ("bias", "bias")):
            print(f"  {n:22s} {f(sp[k])}")
        # paired bootstrap over the ICs of the information contribution
        rng = np.random.default_rng(0)
        N = len(ics)
        bs = []
        for _ in range(500):
            ix = rng.integers(0, N, N)
            ra_, rb_ = np.sqrt(A["mse"][ix].mean(0)), np.sqrt(B["mse"][ix].mean(0))
            bs.append(100 * ((B["info"][ix].mean(0) - A["info"][ix].mean(0))[:, sc] / (ra_ + rb_)[:, sc]).mean(1)
                      / rb_[:, sc].mean(1))
        lo, hi = np.percentile(np.stack(bs), [2.5, 97.5], axis=0)
        out["info_ci"] = [lo.tolist(), hi.tolist()]
        print(f"  information, 95 % interval over the ICs: low {f(lo)}  high {f(hi)}")
    if S:
        S, E = torch.cat(S), torch.cat(E)                       # (N, L, C, H, W)
        ww = w.expand(S.shape[2], -1, S.shape[-1]).reshape(-1)  # weights of (C, H, W)
        for l in range(L):
            s, e = S[:, l], E[:, l]
            sm, em = s.mean(0), e.mean(0)                       # static maps (C, H, W)
            ssr = float(torch.sqrt((sm * w).mean() / (em * w).mean()))
            ls, le = torch.log(sm).reshape(-1), torch.log(em).reshape(-1)
            mu_s, mu_e = (ls * ww).sum() / ww.sum(), (le * ww).sum() / ww.sum()
            cov = ((ls - mu_s) * (le - mu_e) * ww).sum()
            static = float(cov / torch.sqrt((((ls - mu_s) ** 2) * ww).sum() * (((le - mu_e) ** 2) * ww).sum()))
            rs, re = (s / sm).reshape(-1), (e / em).reshape(-1)
            wt = ww.repeat(s.shape[0])
            case_sd = float(torch.log(s / sm).std(0).mean())
            q = torch.quantile(rs[::101], torch.tensor([0.2, 0.4, 0.6, 0.8]))
            b = torch.bucketize(rs, q)
            bs_, be_ = [], []
            for k in range(5):
                m = b == k
                bs_.append(float((rs[m] * wt[m]).sum() / wt[m].sum()))
                be_.append(float((re[m] * wt[m]).sum() / wt[m].sum()))
            slope = float(np.polyfit(bs_, be_, 1)[0])
            out["leads"][l + 1] = {"ssr": ssr, "static": static, "case_sd": case_sd, "bin_state": bs_,
                                   "bin_error": be_, "slope": slope}
            print(f"day {l + 1:2d} ssr {ssr:.2f} static {static:.2f} case_sd {case_sd:.3f} slope {slope:+.2f} state",
                  [round(v, 2) for v in bs_], "error", [round(v, 2) for v in be_], flush=True)
    json.dump(out, open(a.json or os.path.join(a.out, "diag_entv_flow.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
