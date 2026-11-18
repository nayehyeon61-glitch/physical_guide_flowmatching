"""The Kalman island built from MEMBERS (owner 2026-11-18 20:46 KST): does the SHAPE of the island's region add to the
scalar table gain? No training of a network; cached forecasts only.

The island now: x_a = x_f + K (z - x_f), K one number per channel and latitude (scripts/diag_island_kalman.py).
With members the gain is a matrix: the members here are the real cases of the library year (their day-n1 error of the
core and the innovation z - x_f that went with it, as in scripts/diag_island_noise.py), and the correction of a point
is the ridge regression, over the members, of what the scalar island still gets wrong on the innovations around it:

    cross   the innovations of ALL channels at the point                (cross-variable covariance)
    space   the innovations of the same channel on the 3 x 3 points     (spatial covariance)
    full    all channels on the 3 x 3 points, members pooled along the latitude circle (+- 1 row): a static,
            zonally homogeneous covariance (the B matrix of 3D-Var)
    ens     the same innovations, members of the POINT only (its own 119 cases, dual form = the ensemble-space
            update of an EnOI / local ensemble Kalman filter)

The island as a CONTINUOUS operator (owner 21:04 KST; Kovachki et al. 2022 "Neural Operator",
github.com/neuraloperator/neuraloperator): the gain is a linear integral operator on the innovation,
inc(x) = int K(x, y) d(y) dy, and ``full`` cuts its kernel to the 3 x 3 points. Without the cut and without the grid:

    sht       ONE linear spherical spectral layer in closed form: a cross-channel matrix per total wavenumber (an
              isotropic kernel of any range between the latitude-standardized fields), ridge over members x orders
    no_lin    the SFNO of the library with identity activations (the trained, low-rank twin of sht)
    no        the SFNO, input = the innovations (a nonlinear gain)
    no_state  the SFNO, input = the innovations and the core's forecast (a flow-dependent gain)

The SFNO variants start at a zero increment; the number of epochs comes from the same two folds. With ``--base full``
every variant is fitted on what the saved matrix island leaves (cross-fitted in the library year), so its scores say
what the continuous kernel ADDS to the 3 x 3 one.

The ridge strength of every variant comes from two folds of the library year; scores are of the test year at the
island: nRMSE, ACC, activity, and each variant against the scalar island raw, at equal smoothing and Wiener / Wiener.
``--save NAME`` writes the increment of one variant (island state minus the scalar island's, z-scored) for a run
through the second core, and the cross-fitted increment of the library year (each fold from the other fold,
isl_inc_<variant>_<library year>cv.npy) for scripts/diag_island_operator.py.

    python scripts/diag_island_ensemble.py --out runs/kaitend --core core_gndmz8 --meas kai_x8ro \
        --table runs/kaitend/kal_tab_core_gndmz8_5_2016_s5_kaix8ro.npz --json OUT.json
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
p.add_argument("--island", type=int, default=5); p.add_argument("--pool", type=int, default=2); p.add_argument("--fold", type=int, default=10)
p.add_argument("--lib-year", type=int, default=None); p.add_argument("--boot", type=int, default=300)
p.add_argument("--lams", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
p.add_argument("--variants", nargs="+", default=["cross", "space", "full", "ens"])
p.add_argument("--seasons", type=int, default=4, help="<variant>_season: members and cases of the same part of the year only")
p.add_argument("--save", default=None, help="variant whose increment is saved as isl_inc_<variant>_<year>.npy")
p.add_argument("--base", default=None, help="fit the variants on what the saved increment isl_inc_<base>_* leaves")
p.add_argument("--no-hidden", type=int, default=48); p.add_argument("--no-layers", type=int, default=2); p.add_argument("--no-epochs", type=int, default=60)
p.add_argument("--no-lr", type=float, default=2e-3); p.add_argument("--no-wd", type=float, default=1e-2); p.add_argument("--no-batch", type=int, default=16)
p.add_argument("--json", required=True)
a = p.parse_args()
torch.set_grad_enabled(False)
ly = a.lib_year or a.val_year
mean, std, inc, clim = R.stats(a, None if os.path.exists(os.path.join(a.out, "stats.npz")) else R.load_years(a, a.train_years))
cz = torch.from_numpy(R.zscore(clim, mean, std))[:, DYN]
L, li = a.leads, a.island - 1
sc = [i for i, c in enumerate(DYN) if c != TISR]; C = len(sc)
vi = [DYN.index(contract.CHANNEL_INDEX[v]) for v in VARS]
vstd = np.asarray(std[[contract.CHANNEL_INDEX[v] for v in VARS]], dtype=np.float64).reshape(-1)
tab = np.load(a.table); K = np.clip((tab["S"] - tab["c"]) / np.maximum(tab["S"] + tab["R"] - 2 * tab["c"], 1e-12), 0, 1)
Kt = torch.as_tensor(K, dtype=torch.float32).view(1, K.shape[0], K.shape[1], 1)
def year(y):             # anomalies at the island: core, measurement, truth (N, 66, H, W)
    te = R.load_years(a, [y])[y]; tez = torch.from_numpy(R.zscore(te, mean, std))
    ics = np.arange(0, len(tez) - L, a.ic_stride)
    Cl = cz[(torch.tensor(ics) + li + 1) % len(cz)]
    g = lambda s: torch.from_numpy(np.asarray(np.load(os.path.join(a.out, f"fc_{s}_{y}.npy"), mmap_mode="r")[:, li], np.float32)) - Cl  # noqa: E731
    f, z = g(a.core), g(a.meas); assert len(f) == len(ics), (len(f), len(ics))
    return f, z, tez[ics + li + 1][:, DYN] - Cl, te.shape[-2]
fL, zL, tL, H = year(ly); fT, zT, tT, _ = year(a.test_year)
lat = R.lat_of(os.path.join(a.data, f"{a.test_year}.nc"), H, a.era5); coslat = np.cos(np.deg2rad(lat))
w = torch.as_tensor(coslat, dtype=torch.float32); w = (w / w.mean()).view(1, -1, 1)
NL, NT, W = len(fL), len(fT), fL.shape[-1]
aL, aT = fL + Kt * (zL - fL), fT + Kt * (zT - fT)                      # the scalar island
dL, dT = (zL - fL)[:, sc].double(), (zT - fT)[:, sc].double()          # innovations
rL = (tL - aL)[:, sc].double()                                         # what the scalar island still gets wrong
bL = bT = None
if a.base:                                                             # the matrix island first; the variants get what it leaves
    ld = lambda y: torch.from_numpy(np.load(os.path.join(a.out, f"isl_inc_{a.base}_{y}.npy")).astype(np.float32))[:, sc].double()  # noqa: E731
    bL, bT = ld(f"{ly}cv"), ld(a.test_year); rL = rL - bL
sd = (dL ** 2).mean((0, 3), keepdim=True).sqrt().clamp_min(1e-8); sr = (rL ** 2).mean((0, 3), keepdim=True).sqrt().clamp_min(1e-8)
XL, XT, YL = dL / sd, dT / sd, rL / sr
foldL = torch.from_numpy((np.arange(NL) // a.fold) % 2)
def stencil(x):          # (N, C, H, W) -> (N, 9, C, H, W): the 3 x 3 points around every point
    o = []
    for dy in (-1, 0, 1):
        ix = (torch.arange(H) + dy).clamp(0, H - 1)
        o += [torch.roll(x[:, :, ix], -dx, -1) for dx in (-1, 0, 1)]
    return torch.stack(o, 1)
def solve(Xa, Ya, Xb, lams):      # ridge over the leading dims: Xa (.., n, p), Ya (.., n, q), Xb (.., k, p) -> [(.., k, q)] per lam
    n, pp = Xa.shape[-2:]
    if pp <= n:
        s, V = torch.linalg.eigh(Xa.transpose(-1, -2) @ Xa / n); b = V.transpose(-1, -2) @ (Xa.transpose(-1, -2) @ Ya / n); P = Xb @ V
    else:                         # dual (ensemble space): fewer members than innovations
        s, V = torch.linalg.eigh(Xa @ Xa.transpose(-1, -2) / n); b = V.transpose(-1, -2) @ Ya / n; P = Xb @ Xa.transpose(-1, -2) @ V
    return [P @ (b / (s.clamp_min(0) + lam).unsqueeze(-1)) for lam in lams]
def predict(kind, Xa, Ya, Xb, sel):   # members Xa / Ya (cases sel of the library) -> prediction for Xb, (lam, N, C, H, W)
    Sa, Sb = stencil(Xa[sel].float()), stencil(Xb.float()); na, nb = int(sel.sum()), len(Xb); out = torch.zeros(len(a.lams), nb, C, H, W, dtype=torch.float64)
    for j in range(H):
        rows = sorted({max(j - 1, 0), j, min(j + 1, H - 1)})
        if kind == "cross":       # (rows x W x members, C) -> (C)
            A = Xa[sel][:, :, rows].permute(0, 2, 3, 1).reshape(-1, C); Y = Ya[sel][:, :, rows].permute(0, 2, 3, 1).reshape(-1, C)
            B = Xb[:, :, j].permute(0, 2, 1).reshape(-1, C)
            for q, P in enumerate(solve(A, Y, B, a.lams)):
                out[q, :, :, j] = P.reshape(nb, W, C).permute(0, 2, 1)
        elif kind == "space":     # per channel: (C, rows x W x members, 9) -> (1)
            A = Sa[:, :, :, rows].permute(2, 0, 3, 4, 1).reshape(C, -1, 9).double(); Y = Ya[sel][:, :, rows].permute(1, 0, 2, 3).reshape(C, -1, 1)
            B = Sb[:, :, :, j].permute(2, 0, 3, 1).reshape(C, -1, 9).double()
            for q, P in enumerate(solve(A, Y, B, a.lams)):
                out[q, :, :, j] = P.reshape(C, nb, W).permute(1, 0, 2)
        elif kind == "full":      # (rows x W x members, 9 C) -> (C)
            A = Sa[:, :, :, rows].permute(0, 3, 4, 1, 2).reshape(-1, 9 * C).double(); Y = Ya[sel][:, :, rows].permute(0, 2, 3, 1).reshape(-1, C)
            B = Sb[:, :, :, j].permute(0, 3, 1, 2).reshape(-1, 9 * C).double()
            for q, P in enumerate(solve(A, Y, B, a.lams)):
                out[q, :, :, j] = P.reshape(nb, W, C).permute(0, 2, 1)
        elif kind == "ens":       # per point: (W, members, 9 C) -> (C), ensemble space
            A = Sa[:, :, :, j].permute(3, 0, 1, 2).reshape(W, na, 9 * C).double(); Y = Ya[sel][:, :, j].permute(2, 0, 1)
            B = Sb[:, :, :, j].permute(3, 0, 1, 2).reshape(W, nb, 9 * C).double()
            for q, P in enumerate(solve(A, Y, B, a.lams)):
                out[q, :, :, j] = P.permute(1, 2, 0)
    return out
wn = torch.as_tensor(coslat / coslat.mean(), dtype=torch.float64).view(1, 1, -1, 1)
base = (rL ** 2 * wn).mean((0, 2, 3))                                   # the MSE the variants start from, per channel (library year)
rng = np.random.default_rng(0); BOOT = [rng.integers(0, NT, NT) for _ in range(a.boot)]; foldT = torch.from_numpy((np.arange(NT) // a.fold) % 2)
SHT = ISHT = None
def sht_predict(Xa, Ya, Xb):          # one linear spherical spectral layer: a (C, C) matrix per total wavenumber l
    global SHT, ISHT
    if SHT is None:                   # the cell-centre grid is taken as the Gauss grid (at most 1.4 deg off), as the SFNO does
        import torch_harmonics as th
        SHT = th.RealSHT(H, W, lmax=H, mmax=H, grid="legendre-gauss", norm="ortho")
        ISHT = th.InverseRealSHT(H, W, lmax=H, mmax=H, grid="legendre-gauss", norm="ortho")
    ca, ya, cb = SHT(Xa.float()), SHT(Ya.float()), SHT(Xb.float()); nb = len(Xb); k = H / (4 * np.pi) ** 0.5     # a mode of a unit field ~ 1
    oc = torch.zeros(len(a.lams), nb, C, H, H, dtype=torch.complex64)
    for l in range(H):                # samples: members x (2 l + 1) real parts, m >= 1 counted twice as in the power
        re = lambda c: (k * torch.cat([c[..., l, :1].real, 2 ** 0.5 * c[..., l, 1:l + 1].real, 2 ** 0.5 * c[..., l, 1:l + 1].imag], -1)).permute(0, 2, 1).reshape(-1, C).double()  # noqa: E731
        for q, P in enumerate(solve(re(ca), re(ya), re(cb), a.lams)):
            P = (P / k).reshape(nb, 2 * l + 1, C).permute(0, 2, 1).float()
            oc[q, :, :, l, 0] = P[..., 0].to(torch.complex64)
            oc[q, :, :, l, 1:l + 1] = torch.complex(P[..., 1:l + 1], P[..., l + 1:]) / 2 ** 0.5
    return torch.stack([ISHT(o) for o in oc]).double()
sf = (fL[:, sc] ** 2).mean((0, 3), keepdim=True).sqrt().clamp_min(1e-8)
GEO = torch.as_tensor(np.stack([np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat))]), dtype=torch.float32).view(1, 2, H, 1)
def no_in(kind, X, F):                # inputs of the SFNO: innovations (+ the core's forecast) + the latitude
    return torch.cat([X.float()] + ([F[:, sc] / sf] if kind == "no_state" else []) + [GEO.expand(len(X), 2, H, W)], 1)
def no_train(kind, I, Y, epochs, Iv=None, Yv=None):
    from functools import partial
    from neuralop.layers.spherical_convolution import SphericalConv
    from neuralop.models import SFNO
    torch.manual_seed(0); wl = w.view(1, 1, -1, 1); curve = []
    m = SFNO(n_modes=(H, W), in_channels=I.shape[1], out_channels=C, hidden_channels=a.no_hidden, n_layers=a.no_layers, positional_embedding=None,
             conv_module=partial(SphericalConv, sht_grids="legendre-gauss"), **({"non_linearity": torch.nn.Identity()} if kind == "no_lin" else {}))
    torch.nn.init.zeros_(m.projection.fcs[-1].weight); torch.nn.init.zeros_(m.projection.fcs[-1].bias)   # starts at the island as it is
    opt = torch.optim.AdamW(m.parameters(), lr=a.no_lr, weight_decay=a.no_wd)
    ev = lambda J: torch.cat([m(J[i:i + 32]) for i in range(0, len(J), 32)])  # noqa: E731
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(I))
        with torch.enable_grad():
            for i in range(0, len(I), a.no_batch):
                ix = perm[i:i + a.no_batch]; loss = ((m(I[ix]) - Y[ix]) ** 2 * wl).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if Iv is not None:
            m.eval(); curve.append((((ev(Iv) - Yv) ** 2 * wl).mean((0, 2, 3)) / (Yv ** 2 * wl).mean((0, 2, 3))).mean().item())
    m.eval(); return (lambda J: ev(J).double()), np.asarray(curve)
def stat(X, ix=None):
    s = (lambda k: X[k].mean(0)) if ix is None else (lambda k: X[k][ix].mean(0))
    return (np.sqrt(s("mse"))[:, sc].mean(1), (s("pt") / np.sqrt(s("pp") * s("tt")))[:, sc].mean(1), np.sqrt(s("pp") / s("tt"))[:, sc].mean(1))
def row(X):
    r, c, s = stat(X); m = lambda k: X[k].mean(0)  # noqa: E731
    return {"nrmse": float(r[0]), "acc": float(c[0]), "act": float(s[0]),
            "vars": {v: {"rmse": float(np.sqrt(m("mse")[0, vi[q]]) * vstd[q]), "acc": float((m("pt") / np.sqrt(m("pp") * m("tt")))[0, vi[q]])} for q, v in enumerate(VARS)}}
def diff(A, B):
    (ra, ca, _), (rb, cb, _) = stat(A), stat(B); bs = []
    for ix in BOOT:
        (r1, c1, _), (r2, c2, _) = stat(A, ix), stat(B, ix); bs.append([100 * (1 - r1[0] / r2[0]), c1[0] - c2[0]])
    lo, hi = np.percentile(np.asarray(bs), [2.5, 97.5], axis=0)
    return {"gain_pct": float(100 * (1 - ra[0] / rb[0])), "gain_lo": float(lo[0]), "gain_hi": float(hi[0]),
            "dacc": float(ca[0] - cb[0]), "dacc_lo": float(lo[1]), "dacc_hi": float(hi[1])}
T5 = tT[:, None]
def ref(x):                           # a reference state: its terms, its Wiener-filtered terms, itself
    x = x[:, None]; return D.terms(x, T5, w), D.terms(D.wiener(x, T5, foldT, a.pool), T5, w), x
REF = {"scalar": ref(aT)}
if a.base:
    x = aT.clone(); x[:, sc] += bT.float(); REF["base"] = ref(x)
out = {"core": a.core, "meas": a.meas, "lib_year": ly, "year": a.test_year, "island": a.island, "members": int(NL), "lams": a.lams, "base": a.base,
       "rows": {"core": row(D.terms(fT[:, None], T5, w)), "meas": row(D.terms(zT[:, None], T5, w)), "scalar": row(REF["scalar"][0]), "wiener_scalar": row(REF["scalar"][1])}, "variants": {}}
if a.base:
    out["rows"]["base"] = row(REF["base"][0])
g = lambda r: f"nRMSE {r['nrmse']:.3f} ACC {r['acc']:.3f} act {r['act']:.2f}"  # noqa: E731
print(f"== island day {a.island}: members of {ly} ({NL}), scores of {a.test_year} ({NT} ICs)" + (f"; variants on top of the saved increment {a.base}" if a.base else ""))
for k, r in out["rows"].items():
    print(f"  {k:14s} {g(r)}")
for kind in a.variants:
    Pl = None
    if kind.startswith("no"):                                           # SFNO: epochs from the two folds, then all members
        IL, IT = no_in(kind, XL, fL), no_in(kind, XT, fT); YF = YL.float(); cv = 0
        for q in (0, 1):
            cv = cv + no_train(kind, IL[foldL != q], YF[foldL != q], a.no_epochs, IL[foldL == q], YF[foldL == q])[1] * float((foldL == q).float().mean())
        best = int(cv.argmin()); P = no_train(kind, IL, YF, best + 1)[0](IT) * sr; hp = {"epochs": best + 1, "cv_mse_ratio": cv.tolist()}
        note = f"{best + 1} epochs of {a.no_epochs}, library two-fold MSE / start {cv[best]:.4f} (after 1, 5, 10, 20, last: " + " ".join(f"{cv[min(e, len(cv)) - 1]:.3f}" for e in (1, 5, 10, 20, len(cv))) + ")"
    else:
        cv = torch.zeros(len(a.lams), dtype=torch.float64)              # two folds of the library year
        bk, groups = (kind[:-7], range(a.seasons)) if kind.endswith("_season") else (kind, [None])
        fit = (lambda Xb, sel: sht_predict(XL[sel], YL[sel], Xb)) if bk == "sht" else (lambda Xb, sel: predict(bk, XL, YL, Xb, sel))
        Pt = torch.zeros(len(a.lams), NT, C, H, W, dtype=torch.float64)
        Pl = torch.zeros(len(a.lams), NL, C, H, W, dtype=torch.float32) if a.save == kind else None
        for s_ in groups:                                               # _season: members of the same part of the year
            mL = torch.ones(NL, dtype=torch.bool) if s_ is None else torch.from_numpy(np.arange(NL) * a.seasons // NL == s_)
            mT = torch.ones(NT, dtype=torch.bool) if s_ is None else torch.from_numpy(np.arange(NT) * a.seasons // NT == s_)
            for q in (0, 1):
                use = mL & (foldL == q)
                P = fit(XL[use], mL & (foldL != q)) * sr
                cv += ((((P - rL[use]) ** 2 * wn).mean((1, 3, 4)) / base).mean(1)) * float(use.float().mean())
                if Pl is not None:
                    Pl[:, use] = P.float()
            Pt[:, mT] = fit(XT[mT], mL)
        best = int(cv.argmin()); P = Pt[best] * sr; del Pt; hp = {"lam": a.lams[best], "cv_mse_ratio": cv.tolist()}
        note = f"lam {a.lams[best]:g}, library two-fold MSE / start {cv[best]:.4f} (" + " ".join(f"{x:.3f}" for x in cv.tolist()) + ")"
    if a.base:
        P = P + bT
    arm = aT.clone(); arm[:, sc] += P.float(); arm = arm[:, None]; Ta = D.terms(arm, T5, w); Wa = D.terms(D.wiener(arm, T5, foldT, a.pool), T5, w)
    o = dict(hp, row=row(Ta), vars_gain_pct={v: float(100 * (1 - np.sqrt(Ta["mse"].mean(0)[0, vi[n]] / REF["scalar"][0]["mse"].mean(0)[0, vi[n]]))) for n, v in enumerate(VARS)})
    print(f"  {kind:14s} {g(o['row'])} | {note}")
    for rn, (Tr, Wr, xr) in REF.items():                                # against the scalar island, and against the base
        c = {"raw": diff(Ta, Tr), "vs_match": diff(Ta, D.terms(D.match(xr, arm, a.pool), T5, w)), "wiener": diff(Wa, Wr)}
        o.update(c if rn == "scalar" else {"vs_" + rn: c})
        print(f"      vs {rn}: " + "; ".join(f"{n} {c[n]['gain_pct']:+.2f} % ({c[n]['gain_lo']:+.2f}..{c[n]['gain_hi']:+.2f}), ACC {c[n]['dacc']:+.4f} ({c[n]['dacc_lo']:+.4f}..{c[n]['dacc_hi']:+.4f})" for n in ("raw", "vs_match", "wiener"))
              + (" | " + " ".join(f"{v} {x:+.2f}" for v, x in o["vars_gain_pct"].items()) if rn == "scalar" else ""), flush=True)
    out["variants"][kind] = o
    if a.save == kind:                                                  # the whole increment over the scalar island
        name = (a.base + "+" if a.base else "") + kind
        full = np.zeros((NT, len(DYN), H, W), np.float16); full[:, sc] = P.numpy().astype(np.float16)
        np.save(os.path.join(a.out, f"isl_inc_{name}_{a.test_year}.npy"), full)
        if Pl is not None:
            full = np.zeros((NL, len(DYN), H, W), np.float16); full[:, sc] = (Pl[best].double() + (bL if a.base else 0)).numpy().astype(np.float16)
            np.save(os.path.join(a.out, f"isl_inc_{name}_{ly}cv.npy"), full)
json.dump(out, open(a.json, "w"), indent=1)
print("saved", a.json)
