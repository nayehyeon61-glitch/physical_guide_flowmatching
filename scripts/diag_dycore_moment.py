"""Does the noise change the equations' own forecast? The second-moment term of the dycore day at a forecast state.

For nonlinear dynamics the central forecast is not the expected value: mean = G(mean) + 1/2 H C + ... (Evensen,
Data Assimilation, eq. 4.27; the ensemble Kalman filter gets the term by integrating every member, eq. 4.43).
Here, at the day-``--day`` forecast x of a deterministic entropy core (the "island" of the two-stage forecast), the
dycore D integrates x + d and x - d and

    m2 = 1/2 [D(x + d) + D(x - d)] - D(x)          (the equations' own second-order term along d)

for two kinds of d: ``w`` = smoothed white noise with the entropy state's amplitude s (capped at ``--amax``
increment std, m2 rescaled by 1 / kappa^2), ``e`` = the real forecast error of ANOTHER case at the same lead
(balanced, the climatological error covariance; fixed patterns, so the forecast stays deterministic).

Printed per variable (u, v, T, q on 12 levels, area weights):
* size       rms(m2) / rms(D(x))
* split      correlation of m2 from pairs 1-2 with m2 from pairs 3-4 (how noisy one pair is)
* quad       correlation of m2 at half amplitude x 4 with m2 at full amplitude (is it second order?)
* damp       m2 regressed on D(x): coefficient and R^2 (the part that is a plain damping of the central hint)
* res_gain   R^2 (%) of the TRAINED core's own next-day residual (truth(day + 1) - the core's step from x) explained
             by m2: what the core has not already learned; one pair, placebo, and with D(x) beside it
* R2         of the next day's increment (truth(day + 1) - x) explained by D(x) alone, by D(x) + m2, by D(x) + one
             pair's m2, by D(x) + the m2 of another case (placebo); gain = MSE left, %.

    python scripts/diag_dycore_moment.py --out runs/kaitend --tag core_gbv --year 2016
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

HV = ("u", "v", "t", "q")


def smooth_noise(shape, gen):
    F = torch.nn.functional
    xi = torch.randn(shape, generator=gen)
    return F.avg_pool2d(F.pad(F.pad(xi, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="reflect"), 3, 1) * 3.0


def main():
    p = R.parser()
    p.add_argument("--year", type=int, default=2016)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--day", type=int, default=5)
    p.add_argument("--npairs", type=int, default=4)
    p.add_argument("--amax", type=float, default=0.5)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    tr = R.load_years(a, a.train_years)
    mean, std, inc, clim = R.stats(a, tr)
    del tr
    yr = R.load_years(a, [a.year])[a.year]
    lat = R.lat_of(os.path.join(a.data, f"{a.test_year}.nc"), yr.shape[-2], a.era5)
    xz = torch.from_numpy(R.zscore(yr, mean, std))
    core = R.build_core(a, mean, std, inc, lat, load=True, tag=a.tag)
    core.noise_on = False
    dy = core.features.dy
    n1, n, P, B = a.day, a.n, a.npairs, 8
    ics = np.linspace(0, len(xz) - n1 - 2, n).astype(int)
    w = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float64)
    w = (w / w.mean()).view(1, 1, -1, 1)
    t6 = core.t6.cpu().view(1, -1, 1, 1)
    hch = [DYN.index(f) for v in HV for f in range(*dy.full_of[v].indices(len(mean)))]
    hstd = torch.as_tensor(std[[DYN[c] for c in hch]], dtype=torch.float64).view(1, -1, 1, 1)
    X, S, Y, E, FC = [], [], [], [], []
    for i in range(0, n, B):
        j = ics[i:i + B]
        tgt = torch.stack([xz[j + k + 1] for k in range(n1 + 1)], 1)
        with torch.no_grad():
            y = core_rollout(core, xz[j][:, DYN], torch.tensor(j), n1 + 1, tgt[:, :n1 + 1, TISR])
        x = y[:, n1 - 1]
        X.append(x)
        FC.append(((y[:, n1] - x)[:, hch]).double() * hstd)        # the trained core's own next-day increment
        S.append(torch.sqrt(torch.exp(core.v_log[n1 - 1])) * t6 if getattr(core, "entv", False)
                 else torch.ones_like(x) * t6)
        E.append(tgt[:, n1 - 1][:, DYN] - x)                       # this case's own error at the island
        Y.append(((tgt[:, n1][:, DYN] - x)[:, hch]).double() * hstd)  # the next day's increment, SI per day
        print("rollout", i + len(j), "/", n, flush=True)
    X, S, Y, E = torch.cat(X).float(), torch.cat(S).float(), torch.cat(Y), torch.cat(E).float()
    FC = torch.cat(FC)
    keep = torch.ones(len(DYN)); keep[DYN.index(TISR)] = 0
    keep = keep.view(1, -1, 1, 1)

    def integ(x):  # (n, 66, H, W) -> raw48 (n, 48, H, W) float64, bad (n,)
        out, bad = [], []
        for i in range(0, len(x), B):
            raw, _ = dy._integrate(x[i:i + B])
            out.append(torch.from_numpy(raw)); bad.append(np.asarray(dy.last_bad).copy())
        return torch.cat(out).double(), np.concatenate(bad)

    D0, bad0 = integ(X)
    print("central integrations done, failed", int(bad0.sum()), flush=True)
    gen = torch.Generator().manual_seed(a.seed)
    offs = [5 + 3 * k for k in range(max(P, 2))]
    M2, NB, KAP = {}, {}, {}

    def pair(d, kappa):
        rp, bp = integ(X + kappa.view(-1, 1, 1, 1) * d)
        rm, bm = integ(X - kappa.view(-1, 1, 1, 1) * d)
        m2 = (0.5 * (rp + rm) - D0) / (kappa.view(-1, 1, 1, 1).double() ** 2)
        b = bp | bm | bad0
        m2[torch.from_numpy(b)] = 0.0
        return m2, b

    for kind in ("e", "w", "e_half"):
        ms, nb, kp = [], 0, []
        for k in range(P if kind != "e_half" else 2):
            if kind == "w":
                d = S * smooth_noise(X.shape, gen) * keep
                rms = torch.sqrt(((d / t6) ** 2 * keep).mean((1, 2, 3)) * len(DYN) / (len(DYN) - 1))
                kappa = torch.clamp(a.amax / rms, max=1.0)
            else:
                d = E[(np.arange(n) + offs[k]) % n] * keep
                kappa = torch.full((n,), 0.5 if kind == "e_half" else 1.0)
            m2, b = pair(d, kappa)
            ms.append(m2); nb += int(b.sum()); kp.append(float(kappa.mean()))
            print(kind, "pair", k + 1, "failed", int(b.sum()), "kappa", round(kp[-1], 3), flush=True)
        M2[kind], NB[kind], KAP[kind] = torch.stack(ms), nb, kp

    def wmean(x):  # area mean over cases and grid -> (C,)
        return (x * w).mean((0, 2, 3))

    def corr(p_, q_):
        pa, qa = p_ - wmean(p_).view(1, -1, 1, 1), q_ - wmean(q_).view(1, -1, 1, 1)
        return wmean(pa * qa) / torch.sqrt(wmean(pa ** 2) * wmean(qa ** 2) + 1e-300)

    def r2(yv, regs):  # per channel least squares with intercept -> R^2 (C,)
        C = yv.shape[1]
        out = torch.zeros(C, dtype=torch.float64)
        ww = w.expand(yv.shape[0], 1, -1, yv.shape[-1]).reshape(-1)
        for c in range(C):
            A = torch.stack([torch.ones_like(ww)] + [r[:, c].reshape(-1) for r in regs], 1)
            t = yv[:, c].reshape(-1)
            Aw, tw = A * ww.sqrt()[:, None], t * ww.sqrt()
            beta = torch.linalg.lstsq(Aw, tw[:, None]).solution[:, 0]
            res = tw - Aw @ beta
            tm = (t * ww).sum() / ww.sum()
            out[c] = 1 - (res ** 2).sum() / (((t - tm) ** 2) * ww).sum()
        return out

    grp = {v: slice(12 * i, 12 * (i + 1)) for i, v in enumerate(HV)}
    f3 = lambda t: {v: round(float(t[s].mean()), 4) for v, s in grp.items()}  # noqa: E731
    out = {"tag": a.tag, "year": a.year, "n": n, "day": n1, "pairs": P, "failed": NB, "kappa": KAP, "kinds": {}}
    r0 = r2(Y, [D0])
    out["r2_central"] = f3(r0)
    print("R2 of the next day's increment by the central hint D(x):", out["r2_central"])
    RES = Y - FC                                                 # what the trained core leaves at day + 1
    out["skill_core"] = f3(1 - wmean(RES ** 2) / wmean((Y - wmean(Y).view(1, -1, 1, 1)) ** 2))
    out["r2_res_central"] = f3(r2(RES, [D0]))
    print("skill of the trained core's own step (1 - MSE / variance):", out["skill_core"])
    print("R2 of the core's residual by the central hint (should be ~0):", out["r2_res_central"])
    for kind in ("e", "w"):
        m = M2[kind]
        mm, h1, h2 = m.mean(0), m[:P // 2].mean(0), m[P // 2:].mean(0)
        size = torch.sqrt(wmean(mm ** 2) / wmean(D0 ** 2))
        split = corr(h1, h2)
        num, den = wmean(mm * D0), wmean(D0 ** 2)
        dcoef = num / den
        dr2 = corr(mm, D0) ** 2
        r1, r1p = r2(Y, [D0, mm]), r2(Y, [D0, m[0]])
        rpl = r2(Y, [D0, mm[(np.arange(n) + 3) % n]])
        gain = lambda rr: 100 * (rr - r0) / (1 - r0)  # noqa: E731
        res = {"size": f3(size), "split": f3(split), "damp_coef": f3(dcoef), "damp_r2": f3(dr2),
               "r2": f3(r1), "gain_mse_pct": f3(gain(r1)), "gain_one_pair": f3(gain(r1p)), "gain_placebo": f3(gain(rpl))}
        res["res_gain_pct"] = f3(100 * r2(RES, [mm]))            # of the trained core's residual MSE
        res["res_gain_one_pair"] = f3(100 * r2(RES, [m[0]]))
        res["res_gain_placebo"] = f3(100 * r2(RES, [mm[(np.arange(n) + 3) % n]]))
        res["res_gain_with_hint"] = f3(100 * r2(RES, [D0, mm]))
        if kind == "e":
            res["quad"] = f3(corr(M2["e_half"].mean(0), m[:2].mean(0)))
        out["kinds"][kind] = res
        print(f"--- perturbation kind {kind}: failed integrations {NB[kind]}, kappa {KAP[kind]}")
        for k_, v_ in res.items():
            print(f"  {k_:14s}", v_)
    np.savez_compressed(os.path.join(a.out, "diag_dycore_moment.npz"), D0=D0.float().numpy(), Y=Y.float().numpy(),
                        FC=FC.float().numpy(), me=M2["e"].mean(0).float().numpy(), mw=M2["w"].mean(0).float().numpy(),
                        me1=M2["e"][0].float().numpy(), S=S.numpy(), ics=ics)
    json.dump(out, open(a.json or os.path.join(a.out, "diag_dycore_moment.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
