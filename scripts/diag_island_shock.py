"""Two no-training checks for the question "shock / entropy structure at the Kalman island" (2026-11-18), from cached
forecasts only (fc_<spec>_<year>.npy of scripts/diag_island_kalman.py / make_full_blend.py):

1. POSITION vs AMPLITUDE at the island (day ``--island``): the linear Kalman blend x_f + K (z - x_f) of the core and the
   measurement averages amplitudes, so two troughs at different longitudes cancel partly. Against it: a blend that keeps
   the position information per zonal wavenumber (amplitude = blend of the two amplitudes, phase moved along the circle
   by K times the phase difference; ``restore`` = the linear blend's phase with the blended amplitude). Rows: nRMSE, ACC,
   activity, and each variant against the linear blend raw, at equal smoothing (the linear blend with the variant's
   zonal spectrum) and Wiener against Wiener (each with its best zonal gains from the other fold).
2. SENSORS at the island as predictors of WHERE the later error of ``--target`` is: the squared core-measurement
   disagreement and the squared gradient of the island state, each divided by its mean over the ICs (so the fixed
   lead / channel / place table is removed), against the equally normalized squared error at day island..L:
   correlation over ICs (global means), regions (boxes) and points; and what a sensor-dependent damping (one factor per
   sensor quartile, fitted on the other fold) gains over one damping per lead and channel.

    python scripts/diag_island_shock.py --out runs/kaitend --core core_gndmz8 --meas kai_x8ro \
        --table runs/kaitend/kal_tab_core_gndmz8_5_2016_s5_kaix8ro.npz --target fb10-m8_s2memx-kaix8ro --json OUT.json
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_kai_tendency as R
import diag_stage_filter as D
from pgcfm import contract
from pgcfm.kai_tend import DYN, TISR
VARS = D.VARS
p = R.parser()
p.add_argument("--core", required=True); p.add_argument("--meas", required=True); p.add_argument("--table", required=True)
p.add_argument("--target", required=True, help="forecast whose later error the sensors should locate")
p.add_argument("--island", type=int, default=5); p.add_argument("--pool", type=int, default=2); p.add_argument("--fold", type=int, default=10)
p.add_argument("--boot", type=int, default=300); p.add_argument("--json", required=True)
a = p.parse_args()
mean, std, inc, clim = R.stats(a, None if os.path.exists(os.path.join(a.out, "stats.npz")) else R.load_years(a, a.train_years))
te = R.load_years(a, [a.test_year])[a.test_year]
lat = R.lat_of(os.path.join(a.data, f"{a.test_year}.nc"), te.shape[-2], a.era5)
tez = torch.from_numpy(R.zscore(te, mean, std)); cz = torch.from_numpy(R.zscore(clim, mean, std))[:, DYN]
L = a.leads; ics = np.arange(0, len(tez) - L, a.ic_stride); N = len(ics)
coslat = np.cos(np.deg2rad(lat)); w = torch.as_tensor(coslat, dtype=torch.float32); w = (w / w.mean()).view(1, -1, 1)
sc = [i for i, c in enumerate(DYN) if c != TISR]
vi = [DYN.index(contract.CHANNEL_INDEX[v]) for v in VARS]
vstd = np.asarray(std[[contract.CHANNEL_INDEX[v] for v in VARS]], dtype=np.float64).reshape(-1)
fold = torch.from_numpy((np.arange(N) // a.fold) % 2)
rng = np.random.default_rng(0); BOOT = [rng.integers(0, N, N) for _ in range(a.boot)]
def fc(spec):
    y = np.load(os.path.join(a.out, f"fc_{spec.replace('>', '-').replace('@', '_')}_{a.test_year}.npy"), mmap_mode="r")
    assert y.shape[:2] == (N, L), (spec, y.shape); return y
def anom(F, l):          # lead index l (0 = day 1) -> anomalies (N, 1, C, H, W), truth anomalies, climatology
    C = cz[(torch.tensor(ics) + l + 1) % len(cz)][:, None]
    return torch.from_numpy(np.asarray(F[:, l:l + 1], np.float32)) - C, tez[ics + l + 1][:, None][:, :, DYN] - C, C
def stat(X, ix=None):
    s = (lambda k: X[k].mean(0)) if ix is None else (lambda k: X[k][ix].mean(0))
    return (np.sqrt(s("mse"))[:, sc].mean(1), (s("pt") / np.sqrt(s("pp") * s("tt")))[:, sc].mean(1), np.sqrt(s("pp") / s("tt"))[:, sc].mean(1))
def row(X):
    r, c, s = stat(X); m = lambda k: X[k].mean(0)  # noqa: E731
    return {"nrmse": float(r[0]), "acc": float(c[0]), "act": float(s[0]),
            "vars": {v: {"rmse": float(np.sqrt(m("mse")[0, vi[q]]) * vstd[q]), "acc": float((m("pt") / np.sqrt(m("pp") * m("tt")))[0, vi[q]]),
                         "act": float(np.sqrt(m("pp") / m("tt"))[0, vi[q]])} for q, v in enumerate(VARS)}}
def diff(A, B):          # A against B: nRMSE gain % (+ = A better) and ACC difference with 95 % intervals over the ICs
    (ra, ca, _), (rb, cb, _) = stat(A), stat(B); bs = []
    for ix in BOOT:
        (r1, c1, _), (r2, c2, _) = stat(A, ix), stat(B, ix); bs.append([100 * (1 - r1[0] / r2[0]), c1[0] - c2[0]])
    lo, hi = np.percentile(np.asarray(bs), [2.5, 97.5], axis=0)
    return {"gain_pct": float(100 * (1 - ra[0] / rb[0])), "gain_lo": float(lo[0]), "gain_hi": float(hi[0]),
            "dacc": float(ca[0] - cb[0]), "dacc_lo": float(lo[1]), "dacc_hi": float(hi[1])}
out = {"core": a.core, "meas": a.meas, "target": a.target, "ics": int(N), "island": a.island, "year": a.test_year}

# ---- 1. position vs amplitude at the island
li = a.island - 1
f, T, C = anom(fc(a.core), li); z, _, _ = anom(fc(a.meas), li)
tab = np.load(a.table); K = np.clip((tab["S"] - tab["c"]) / np.maximum(tab["S"] + tab["R"] - 2 * tab["c"], 1e-12), 0, 1)
Kt = torch.as_tensor(K, dtype=torch.float32).view(1, 1, K.shape[0], K.shape[1], 1)
lin = f + Kt * (z - f)
def blends(ff, zz):      # fields (anomalies or full) -> position-preserving blend, amplitude-restored linear blend
    F, Z = torch.fft.rfft(ff, dim=-1), torch.fft.rfft(zz, dim=-1); Kc = Kt.to(F.real.dtype)
    Lk = F + Kc * (Z - F); amp = (1 - Kc) * F.abs() + Kc * Z.abs()
    ph = torch.angle(F) + Kc * torch.angle(Z * F.conj())
    P = torch.polar(amp, ph); P[..., 0] = Lk[..., 0]                      # zonal mean: linear
    Rk = torch.polar(amp, torch.angle(Lk)); Rk[..., 0] = Lk[..., 0]
    n = ff.shape[-1]; loss = 1 - (Lk[..., 1:].abs() ** 2).sum(-1) / (amp[..., 1:] ** 2).sum(-1).clamp_min(1e-20)
    return torch.fft.irfft(P, n=n, dim=-1), torch.fft.irfft(Rk, n=n, dim=-1), loss
pa_a, re_a, loss_a = blends(f, z)                      # phases of the anomalies
pa_f, re_f, loss_f = blends(f + C, z + C)              # phases of the full fields (stationary waves included)
V = {"core": f, "meas": z, "linear": lin, "position_anom": pa_a, "restore_anom": re_a, "position_full": pa_f - C, "restore_full": re_f - C}
Tm = {k: D.terms(v, T, w) for k, v in V.items()}
out["island_rows"] = {k: row(v) for k, v in Tm.items()}
wl = D.terms(D.wiener(lin, T, fold, a.pool), T, w); out["island_rows"]["wiener_linear"] = row(wl)
out["island_vs_linear"] = {}
for k in ("position_anom", "restore_anom", "position_full", "restore_full"):
    out["island_vs_linear"][k] = {"raw": diff(Tm[k], Tm["linear"]),
                                  "vs_match": diff(Tm[k], D.terms(D.match(lin, V[k], a.pool), T, w)),
                                  "wiener": diff(D.terms(D.wiener(V[k], T, fold, a.pool), T, w), wl)}
out["cancel_loss"] = {"anom": float(loss_a[:, :, sc].mean()), "full": float(loss_f[:, :, sc].mean()),
                      "note": "share of the eddy power (zonal wavenumber >= 1) that the linear blend loses against the blend of the amplitudes"}
g = lambda r: f"nRMSE {r['nrmse']:.3f} ACC {r['acc']:.3f} act {r['act']:.2f}"  # noqa: E731
print(f"== island day {a.island}: {a.core} + {a.meas}, {N} ICs, mean K {K[sc].mean():.2f}")
for k, r in out["island_rows"].items():
    print(f"  {k:15s} {g(r)}")
for k, d in out["island_vs_linear"].items():
    print(f"  {k:15s} vs linear: " + "; ".join(f"{n} {v['gain_pct']:+.2f} % ({v['gain_lo']:+.2f}..{v['gain_hi']:+.2f}), ACC {v['dacc']:+.4f} ({v['dacc_lo']:+.4f}..{v['dacc_hi']:+.4f})" for n, v in d.items()))
print(f"  eddy power lost by the linear blend: anomalies {100 * out['cancel_loss']['anom']:.1f} %, full fields {100 * out['cancel_loss']['full']:.1f} %")

# ---- 2. sensors at the island -> where is the later error?
wn = torch.as_tensor(coslat / coslat.mean(), dtype=torch.float64).view(1, 1, -1, 1)
def box(x, kh, kw):      # running mean over 2 kh + 1 rows and 2 kw + 1 columns (zonally periodic)
    x = D.pool(x, kh) if kh else x
    if kw:
        x = sum(torch.roll(x, s, -1) for s in range(-kw, kw + 1)) / (2 * kw + 1)
    return x
isl = (lin + C)[:, 0].double()                                  # island state, full field, z-scored (N, C, H, W)
dx = (torch.roll(isl, -1, -1) - torch.roll(isl, 1, -1)) / torch.as_tensor(np.maximum(coslat, 0.05)).view(1, 1, -1, 1)
dy = torch.zeros_like(isl); dy[:, :, 1:-1] = isl[:, :, 2:] - isl[:, :, :-2]
S = {"disagree": ((z - f)[:, 0].double()) ** 2, "gradient": dx ** 2 + dy ** 2}
S["disagree_all"] = (S["disagree"][:, sc] / S["disagree"][:, sc].mean((0, 2, 3), keepdim=True)).mean(1, keepdim=True).expand(-1, len(DYN), -1, -1)
norm = lambda x: x / x.mean(0, keepdim=True).clamp_min(1e-30)  # noqa: E731
def corr(x, y):          # per channel over ICs and points, cos-lat weights -> mean over the scored channels
    mx, my = (x * wn).mean((0, 2, 3), keepdim=True) / wn.mean(), (y * wn).mean((0, 2, 3), keepdim=True) / wn.mean()
    cxy = ((x - mx) * (y - my) * wn).mean((0, 2, 3)); vx = ((x - mx) ** 2 * wn).mean((0, 2, 3)); vy = ((y - my) ** 2 * wn).mean((0, 2, 3))
    return float((cxy / (vx * vy).sqrt().clamp_min(1e-30))[sc].mean())
SC = {"point": (1, 1), "region": (2, 4)}                        # 3 x 3 points; 5 rows x 9 columns (28 x 51 deg)
Sn = {k: {s: norm(box(v, *hw)) for s, hw in SC.items()} for k, v in S.items()}
Sg = {k: (v * wn).mean((2, 3)) / wn.mean() for k, v in S.items()}  # (N, C) global means
Ft = fc(a.target); out["sensor"] = {"leads": list(range(a.island, L + 1)), "corr": {}, "damping": {}, "quartile": {}}
quart = lambda s: torch.quantile(s.flatten()[::7], torch.tensor([0.25, 0.5, 0.75], dtype=s.dtype))  # noqa: E731
for k in S:
    out["sensor"]["corr"][k] = {"ic": [], "region": [], "point": []}
    out["sensor"]["damping"][k] = {s: {"gain_pct": [], "gain_lo": [], "gain_hi": [], "act_one": [], "act_sensor": []} for s in SC}
    out["sensor"]["quartile"][k] = {"err": [], "beta": [], "skill": []}
for l in range(li, L):
    P, Tl, _ = anom(Ft, l); P, Tl = P[:, 0].double(), Tl[:, 0].double(); e2 = (P - Tl) ** 2
    eg = (e2 * wn).mean((2, 3)) / wn.mean()
    for k in S:
        cg = [np.corrcoef(Sg[k][:, c].numpy(), eg[:, c].numpy())[0, 1] for c in sc]
        out["sensor"]["corr"][k]["ic"].append(float(np.mean(cg)))
        # by quartile of the regional sensor (all ICs): error variance / its mean, best damping factor, correlation
        qe, qb, qs, wf = [], [], [], wn[0].expand(N, -1, P.shape[-1])
        for c in sc:
            sv = Sn[k]["region"][:, c]; bi = torch.bucketize(sv, quart(sv)); pc, tc = P[:, c], Tl[:, c]
            sm = lambda x, j: (x * wf)[bi == j].sum()  # noqa: E731
            qe.append([float(sm(e2[:, c], j) / sm(torch.ones_like(pc), j) / (e2[:, c] * wf).mean()) for j in range(4)])
            qb.append([float(sm(pc * tc, j) / sm(pc ** 2, j).clamp_min(1e-30)) for j in range(4)])
            qs.append([float(sm(pc * tc, j) / (sm(pc ** 2, j) * sm(tc ** 2, j)).sqrt().clamp_min(1e-30)) for j in range(4)])
        for n_, v_ in (("err", qe), ("beta", qb), ("skill", qs)):
            out["sensor"]["quartile"][k][n_].append(np.mean(v_, 0).tolist())
        for s, hw in SC.items():
            out["sensor"]["corr"][k][s].append(corr(Sn[k][s], norm(box(e2, *hw))))
            # damping: one factor per lead and channel against one factor per sensor quartile, both from the other fold
            m1, m2 = torch.empty(N, len(DYN), dtype=torch.float64), torch.empty(N, len(DYN), dtype=torch.float64)
            p1, p2 = torch.empty_like(m1), torch.empty_like(m1); tt = (Tl ** 2 * wn).mean((2, 3))
            for q in (0, 1):
                fit, use = fold != q, fold == q
                b1 = ((P[fit] * Tl[fit] * wn).sum((0, 2, 3)) / (P[fit] ** 2 * wn).sum((0, 2, 3)).clamp_min(1e-30)).view(1, -1, 1, 1)
                x1 = b1 * P[use]; x2 = torch.empty_like(x1)
                for c in range(len(DYN)):
                    sv = Sn[k][s][:, c]; th = quart(sv[fit]); bi_f, bi_u = torch.bucketize(sv[fit], th), torch.bucketize(sv[use], th)
                    pc, tc, wc = P[fit][:, c], Tl[fit][:, c], wn[0].expand(int(fit.sum()), -1, sv.shape[-1])
                    b = torch.stack([((pc * tc * wc)[bi_f == j]).sum() / ((pc ** 2 * wc)[bi_f == j]).sum().clamp_min(1e-30) for j in range(4)])
                    x2[:, c] = b[bi_u] * P[use][:, c]
                m1[use] = ((x1 - Tl[use]) ** 2 * wn).mean((2, 3)); m2[use] = ((x2 - Tl[use]) ** 2 * wn).mean((2, 3))
                p1[use] = (x1 ** 2 * wn).mean((2, 3)); p2[use] = (x2 ** 2 * wn).mean((2, 3))
            nr = lambda m, ix=slice(None): float(m[ix].mean(0).sqrt()[sc].mean())  # noqa: E731
            bs = [100 * (1 - nr(m2, ix) / nr(m1, ix)) for ix in BOOT]; d = out["sensor"]["damping"][k][s]
            d["gain_pct"].append(100 * (1 - nr(m2) / nr(m1))); d["gain_lo"].append(float(np.percentile(bs, 2.5))); d["gain_hi"].append(float(np.percentile(bs, 97.5)))
            d["act_one"].append(float((p1.mean(0) / tt.mean(0)).sqrt()[sc].mean())); d["act_sensor"].append(float((p2.mean(0) / tt.mean(0)).sqrt()[sc].mean()))
    print(f"day {l + 1} done", flush=True)
h = lambda v, d=2: " ".join(f"{x:+.{d}f}" for x in v)  # noqa: E731
print(f"== sensors at day {a.island} -> squared error of {a.target}, day {a.island}..{L} (fixed table removed)")
for k in S:
    c = out["sensor"]["corr"][k]
    print(f"  {k:13s} corr: ICs {h(c['ic'])} | regions {h(c['region'])} | points {h(c['point'])}")
    for s in SC:
        d = out["sensor"]["damping"][k][s]
        print(f"  {k:13s} damping by quartile ({s}) vs one damping, nRMSE gain %: {h(d['gain_pct'])}  low {h(d['gain_lo'])}  high {h(d['gain_hi'])}")
    q = out["sensor"]["quartile"][k]
    for j, l in enumerate(out["sensor"]["leads"]):
        print(f"  {k:13s} day {l:2d} sensor quartiles 1..4 (regions): error variance / mean {h(q['err'][j])} | best damping {h(q['beta'][j])} | correlation with the truth {h(q['skill'][j])}")
json.dump(out, open(a.json, "w"), indent=1)
print("saved", a.json)
