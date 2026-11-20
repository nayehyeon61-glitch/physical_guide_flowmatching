"""scripts/diag_stage_filter.py lead by lead: the same numbers from cached forecasts with about 1.5 GB instead of 4 GB
(for checks next to a running training). Arms and the control must already be cached (fc_<arm>_<year>.npy).

    python scripts/diag_stage_filter_light.py --out runs/kaitend --arms ARM ... --ctrl CTRL --json OUT.json
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
p.add_argument("--arms", nargs="+", required=True); p.add_argument("--ctrl", required=True)
p.add_argument("--pool", type=int, default=2); p.add_argument("--fold", type=int, default=10); p.add_argument("--json", required=True)
a = p.parse_args()
torch.manual_seed(a.seed)
tr = R.load_years(a, a.train_years); mean, std, inc, clim = R.stats(a, tr); del tr
te = R.load_years(a, [a.test_year])[a.test_year]
lat = R.lat_of(os.path.join(a.data, f"{a.test_year}.nc"), te.shape[-2], a.era5)
# a fine grid (1.5 deg) stays on disk and is normalized on access; the climatology of the days that are needed only
tez = R.ZYear(te, mean, std, as_torch=True) if isinstance(te, np.memmap) else torch.from_numpy(R.zscore(te, mean, std))
czd = lambda idx: torch.from_numpy(R.zscore(np.asarray(clim[idx]), mean, std))[:, DYN]  # noqa: E731
L = a.leads; ics = np.arange(0, len(tez) - L, a.ic_stride); N = len(ics)
w = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32); w = (w / w.mean()).view(1, -1, 1)
sc = [i for i, c in enumerate(DYN) if c != TISR]
vi = [DYN.index(contract.CHANNEL_INDEX[v]) for v in VARS]
vstd = np.asarray(std[[contract.CHANNEL_INDEX[v] for v in VARS]], dtype=np.float64).reshape(-1)
fold = torch.from_numpy((np.arange(N) // a.fold) % 2)
def fc(spec):
    y = np.load(os.path.join(a.out, f"fc_{spec.replace('>', '-').replace('@', '_')}_{a.test_year}.npy"), mmap_mode="r")
    assert y.shape[:2] == (N, L), (spec, y.shape); return y
Fc, Fa = fc(a.ctrl), {s: fc(s) for s in a.arms}
keys = ("ctrl", "wiener_ctrl"); acc = {k: [] for k in keys}
arm_acc = {s: {k: [] for k in ("arm", "match", "wiener_arm")} for s in a.arms}
for l in range(L):
    C = czd((ics + l + 1) % len(clim))[:, None]
    TA = tez[ics + l + 1][:, DYN][:, None] - C
    c = torch.from_numpy(np.asarray(Fc[:, l:l + 1], np.float32)) - C
    acc["ctrl"].append(D.terms(c, TA, w)); acc["wiener_ctrl"].append(D.terms(D.wiener(c, TA, fold, a.pool), TA, w))
    for s in a.arms:
        m = torch.from_numpy(np.asarray(Fa[s][:, l:l + 1], np.float32)) - C
        arm_acc[s]["arm"].append(D.terms(m, TA, w)); arm_acc[s]["match"].append(D.terms(D.match(c, m, a.pool), TA, w))
        arm_acc[s]["wiener_arm"].append(D.terms(D.wiener(m, TA, fold, a.pool), TA, w))
cat = lambda lst: {k: np.concatenate([d[k] for d in lst], 1) for k in lst[0]}
Tc, Twc = cat(acc["ctrl"]), cat(acc["wiener_ctrl"])
def stat(X, ix=None):
    s = (lambda k: X[k].mean(0)) if ix is None else (lambda k: X[k][ix].mean(0))
    return (np.sqrt(s("mse"))[:, sc].mean(1), (s("pt") / np.sqrt(s("pp") * s("tt")))[:, sc].mean(1), np.sqrt(s("pp") / s("tt"))[:, sc].mean(1))
def stat_var(X, ix=None):
    s = (lambda k: X[k].mean(0)[:, vi]) if ix is None else (lambda k: X[k][ix].mean(0)[:, vi])
    return np.sqrt(s("mse")) * vstd, s("pt") / np.sqrt(s("pp") * s("tt"))
def diff(A, B, nb=500):
    (ra, ca, _), (rb, cb, _) = stat(A), stat(B); (va, wa), (vb, wb) = stat_var(A), stat_var(B)
    rng, bs, bv = np.random.default_rng(0), [], []
    for _ in range(nb):
        ix = rng.integers(0, N, N)
        (r1, c1, _), (r2, c2, _) = stat(A, ix), stat(B, ix); bs.append(np.stack([100 * (1 - r1 / r2), c1 - c2]))
        (v1, w1), (v2, w2) = stat_var(A, ix), stat_var(B, ix); bv.append(np.stack([100 * (1 - v1 / v2), w1 - w2]))
    lo, hi = np.percentile(np.stack(bs), [2.5, 97.5], axis=0); vlo, vhi = np.percentile(np.stack(bv), [2.5, 97.5], axis=0)
    o = {"gain_pct": (100 * (1 - ra / rb)).tolist(), "gain_lo": lo[0].tolist(), "gain_hi": hi[0].tolist(),
         "dacc": (ca - cb).tolist(), "dacc_lo": lo[1].tolist(), "dacc_hi": hi[1].tolist(), "vars": {}}
    for q, v in enumerate(VARS):
        o["vars"][v] = {"gain_pct": (100 * (1 - va / vb))[:, q].tolist(), "gain_lo": vlo[0][:, q].tolist(), "gain_hi": vhi[0][:, q].tolist(),
                        "dacc": (wa - wb)[:, q].tolist(), "dacc_lo": vlo[1][:, q].tolist(), "dacc_hi": vhi[1][:, q].tolist()}
    return o
def split(A, B):
    ra, rb = np.sqrt(A["mse"].mean(0)), np.sqrt(B["mse"].mean(0))
    con = lambda k: 100 * ((B[k].mean(0) - A[k].mean(0))[:, sc] / (ra + rb)[:, sc]).mean(1) / rb[:, sc].mean(1)
    return {"total": (100 * (1 - ra[:, sc].mean(1) / rb[:, sc].mean(1))).tolist(), "info": con("info").tolist(), "amp": con("amp").tolist(), "bias": con("bias").tolist()}
def row(X):
    o = dict(zip(("nrmse", "acc", "act"), (v.tolist() for v in stat(X)))); m = lambda k: X[k].mean(0)
    o["acc_fc"] = (X["pt"] / np.sqrt(X["pp"] * X["tt"] + 1e-30)).mean(0)[:, sc].mean(1).tolist()
    o["act_fc"] = np.sqrt(X["vf"] / (X["vo"] + 1e-30)).mean(0)[:, sc].mean(1).tolist()
    o["vars"] = {v: {"rmse": (np.sqrt(m("mse")[:, vi[q]]) * vstd[q]).tolist(), "acc": (m("pt") / np.sqrt(m("pp") * m("tt")))[:, vi[q]].tolist(),
                     "act": np.sqrt(m("pp") / m("tt"))[:, vi[q]].tolist()} for q, v in enumerate(VARS)}
    return o
g = lambda v, d=3: " ".join(f"{x:.{d}f}" for x in v); f = lambda v, d=1: " ".join(f"{x:+.{d}f}" for x in v)
out = {"ctrl": a.ctrl, "ics": int(N), "pool": a.pool, "fold": a.fold, "arms": {}, "ctrl_rows": {"ctrl": row(Tc), "wiener_ctrl": row(Twc)}}
for s in a.arms:
    Ta, Tm, Twa = cat(arm_acc[s]["arm"]), cat(arm_acc[s]["match"]), cat(arm_acc[s]["wiener_arm"])
    rows = {"ctrl": Tc, "arm": Ta, "match": Tm, "wiener_ctrl": Twc, "wiener_arm": Twa}
    o = {"rows": {k: row(v) for k, v in rows.items()}, "arm_vs_ctrl": diff(Ta, Tc), "arm_vs_match": diff(Ta, Tm), "wiener_arm_vs_wiener_ctrl": diff(Twa, Twc),
         "split_arm_vs_ctrl": split(Ta, Tc), "split_arm_vs_match": split(Ta, Tm), "split_wiener": split(Twa, Twc)}
    out["arms"][s] = o
    print(f"== {s} vs {a.ctrl}, day 1..{L}")
    for k, v in o["rows"].items():
        print(f"  {k:12s} nRMSE {g(v['nrmse'])}\n  {'':12s} ACC   {g(v['acc'])}\n  {'':12s} act   {g(v['act'], 2)}")
    for k in ("arm_vs_ctrl", "arm_vs_match", "wiener_arm_vs_wiener_ctrl"):
        d = o[k]
        print(f"  {k}: nRMSE gain % {f(d['gain_pct'])}\n      95 % low {f(d['gain_lo'])}\n      95 % high {f(d['gain_hi'])}\n    ACC diff {f(d['dacc'], 3)}\n      95 % low {f(d['dacc_lo'], 3)}\n      95 % high {f(d['dacc_hi'], 3)}")
    for k in ("arm_vs_match", "wiener_arm_vs_wiener_ctrl"):
        for v in VARS:
            d = o[k]["vars"][v]
            print(f"  {k} {v}: RMSE gain % {f(d['gain_pct'])}\n      95 % low {f(d['gain_lo'])}  high {f(d['gain_hi'])}\n    ACC diff {f(d['dacc'], 3)}\n      95 % low {f(d['dacc_lo'], 3)}  high {f(d['dacc_hi'], 3)}")
    for k in ("split_arm_vs_ctrl", "split_arm_vs_match", "split_wiener"):
        q = o[k]; print(f"  {k}: total {f(q['total'])}\n    pattern {f(q['info'])}\n    amplitude {f(q['amp'])}\n    bias {f(q['bias'])}")
    sys.stdout.flush()
json.dump(out, open(a.json, "w"), indent=1)
