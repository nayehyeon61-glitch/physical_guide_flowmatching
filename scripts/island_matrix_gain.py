"""The matrix gain of the Kalman island from members, row by row (the ``full`` variant of
scripts/diag_island_ensemble.py, owner 2026-11-18 20:46 KST, written so that it runs on KAI's 1.5 deg grid).

The scalar island corrects the first core's island state toward KAI's forecast of that day with one gain per
channel and latitude, a = f + K (z - f) (scripts/diag_island_kalman.py). The members (the cases of the library year)
add the SHAPE of the region: what the scalar island still gets wrong, r = t - a, is regressed (ridge) on the
innovations d = z - f of all 65 channels on the 3 x 3 points around a point (the stencil, ``--step`` grid cells
apart), members pooled over the latitude band of +- ``--pool`` rows and the whole circle (a static, zonally
homogeneous covariance: the B matrix of 3D-Var). Both standardized per channel and latitude. The ridge strength
comes from two folds of the library year (blocks of ``--fold`` ICs). Nothing is trained by gradient.

On the 5.625 deg grid (step 1, pool 1) this is exactly the ``full`` variant (checked: the same increment). On a
finer grid ``--step auto`` keeps the stencil's physical size (round(5.625 / grid spacing): 4 at 1.5 deg, points
6 deg apart) and ``--pool auto`` the latitude band (+- step rows). Only one latitude row of features is in memory
at a time (at 1.5 deg about 0.6 GB for 119 members), the normal equations of every row are kept (585 x 585).

Writes the increment over the scalar island in the normalized units of the state (z-scores):
  isl_inc_<name>_<test year>.npy      (N, 66, H, W) float16, the test year, from all members
  isl_inc_<name>_<lib year>cv.npy     the library year itself, cross-fitted (each fold from the other fold): the
                                       input of the library year's island run (the later blend is fitted on it)
for scripts/diag_island_kalman.py --isl-inc, and a json with the ridge path and the island's nRMSE.

    python scripts/island_matrix_gain.py --out OUT --core-fc core_mem8d --meas kai_ro \
        --table OUT/kal_tab_core_mem8d_5_2016_s5_d1234o_kairo.npz --name full --device cuda
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_kai_tendency as R  # noqa: E402
from pgcfm import contract  # noqa: E402
from pgcfm.kai_tend import DYN, TISR  # noqa: E402

VARS = ["t2m", "t850", "z500", "u850", "msl"]


def main():
    p = R.parser()
    p.add_argument("--core-fc", required=True, help="cache spec of the first core's own forecast (its island day)")
    p.add_argument("--meas", required=True, help="cache spec of KAI's forecast (the measurement)")
    p.add_argument("--table", required=True, help="the scalar island's gain table (S, R, c per channel and latitude)")
    p.add_argument("--name", default="full", help="the increment files are isl_inc_<name>_<year>.npy")
    p.add_argument("--island", type=int, default=5)
    p.add_argument("--lib-year", type=int, default=None, help="the members' year (default: --val-year)")
    p.add_argument("--step", default="auto", help="grid cells between the stencil's points (auto: round(5.625 / spacing))")
    p.add_argument("--pool", default="auto", help="rows +- of the latitude band the members are pooled over (auto: step)")
    p.add_argument("--fold", type=int, default=10, help="ICs per block of the two folds")
    p.add_argument("--lams", type=float, nargs="+", default=[0.3, 1, 3, 10, 30])
    p.add_argument("--json", default=None)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    dev = torch.device(a.device)
    ly, ty, li = a.lib_year or a.val_year, a.test_year, a.island - 1
    tr = R.load_years(a, a.train_years)
    mean, std, inc, _ = R.stats(a, tr)
    del tr
    L = a.leads
    sc = [i for i, c in enumerate(DYN) if c != TISR]
    C = len(sc)
    tab = np.load(a.table)
    Kt = torch.as_tensor(np.clip((tab["S"] - tab["c"]) / np.maximum(tab["S"] + tab["R"] - 2 * tab["c"], 1e-12), 0, 1),
                         dtype=torch.float32)[None, :, :, None]

    def fc(spec, y):
        return np.load(os.path.join(a.out, f"fc_{spec.replace('>', '-').replace('@', '_')}_{y}.npy"), mmap_mode="r")

    def year(y):
        """At the island: the innovation d = z - f and the scalar island's residual r = t - a (65 channels)."""
        raw = R.load_years(a, [y])[y]
        Z = R.ZYear(raw, mean, std, as_torch=True) if isinstance(raw, np.memmap) else torch.from_numpy(R.zscore(raw, mean, std))
        ics = np.arange(0, len(Z) - L, a.ic_stride)
        F, Zm = fc(a.core_fc, y), fc(a.meas, y)
        assert F.shape[:2] == Zm.shape[:2] == (len(ics), L), (F.shape, Zm.shape, len(ics))
        d = torch.empty((len(ics), C) + F.shape[-2:])
        r = torch.empty_like(d)
        for i in range(0, len(ics), 16):           # 16 ICs at a time (1.5 deg: 0.9 GB a year per array in float32)
            f = torch.from_numpy(np.asarray(F[i:i + 16, li], np.float32))
            z = torch.from_numpy(np.asarray(Zm[i:i + 16, li], np.float32))
            t = Z[ics[i:i + 16] + li + 1][:, DYN]
            d[i:i + 16] = (z - f)[:, sc]
            r[i:i + 16] = (t - (f + Kt * (z - f)))[:, sc]
        return d, r, raw.shape[-2]

    dL, rL, H = year(ly)
    dT, rT, _ = year(ty)
    W = dL.shape[-1]
    lat = R.lat_of(os.path.join(a.data, f"{ty}.nc"), H, a.era5)
    spacing = abs(float(lat[1] - lat[0]))
    step = max(1, int(round(5.625 / spacing))) if a.step == "auto" else int(a.step)
    pool = step if a.pool == "auto" else int(a.pool)
    NL, NT = len(dL), len(dT)
    wl = torch.as_tensor(np.cos(np.deg2rad(lat)), dtype=torch.float64)
    wl = wl / wl.mean()
    sd = (dL.double() ** 2).mean((0, 3), keepdim=True).sqrt().clamp_min(1e-8).float()    # (1, C, H, 1)
    sr = (rL.double() ** 2).mean((0, 3), keepdim=True).sqrt().clamp_min(1e-8).float()
    XL, XT, YL = dL / sd, dT / sd, rL / sr
    foldL = torch.from_numpy((np.arange(NL) // a.fold) % 2)
    members = [torch.nonzero(foldL == q).flatten() for q in (0, 1)]
    assert all(len(m) for m in members), f"two folds need more than {a.fold} members ({NL})"
    base = (rL.double() ** 2 * wl.view(1, 1, -1, 1)).mean((0, 2, 3))                       # (C,)
    offs = [(dy, dx) for dy in (-step, 0, step) for dx in (-step, 0, step)]

    def feats(X, rows):
        """(n, C, H, W) -> (n x len(rows) x W, 9 C): the stencil's innovations of the rows (sample order n, row, lon)."""
        o = []
        for dy, dx in offs:
            rr = np.clip(np.asarray(rows) + dy, 0, H - 1)
            o.append(torch.roll(X[:, :, rr], -dx, -1))
        return torch.stack(o, 1).to(dev, torch.float64).permute(0, 3, 4, 1, 2).reshape(-1, 9 * C)

    def targ(Y, rows):
        return Y[:, :, rows].to(dev, torch.float64).permute(0, 2, 3, 1).reshape(-1, C)

    def fit(G, Q, n, lam):
        """Ridge in the eigenbasis of the normal equations (as solve() of diag_island_ensemble.py)."""
        s, V = torch.linalg.eigh(G / n)
        b = V.T @ (Q / n)
        return V @ (b / (s.clamp_min(0) + lam).unsqueeze(-1))                               # (9C, C)

    t0 = time.time()
    NE = []                                      # per row: (G, Q, n) of the two folds
    cv_err = torch.zeros(2, len(a.lams), C, dtype=torch.float64)
    for j in range(H):
        rows = sorted({min(max(j + k, 0), H - 1) for k in range(-pool, pool + 1)})
        ne = []
        for q in (0, 1):
            A, Y = feats(XL[members[q]], rows), targ(YL[members[q]], rows)
            ne.append((A.T @ A, A.T @ Y, A.shape[0]))
            del A, Y
        NE.append([(G.cpu(), Q.cpu(), n) for G, Q, n in ne])
        for q in (0, 1):                         # fold q predicted from the other fold, all ridge strengths
            G, Q, n = ne[1 - q]
            B = feats(XL[members[q]], [j])
            Yq = targ(YL[members[q]], [j]) * sr[0, :, j, 0].to(dev, torch.float64)
            for k, lam in enumerate(a.lams):
                P = (B @ fit(G, Q, n, lam)) * sr[0, :, j, 0].to(dev, torch.float64)
                cv_err[q, k] += (((P - Yq) ** 2).sum(0) * wl[j]).cpu()
        if j % max(1, H // 8) == 0:
            print(f"row {j + 1} / {H}: {time.time() - t0:.0f} s", flush=True)
    # mean over (members, rows, longitudes) per fold, over the channels relative to the scalar island's MSE
    cv = sum((cv_err[q] / (len(members[q]) * H * W) / base).mean(1) * len(members[q]) / NL for q in (0, 1))
    best = int(cv.argmin())
    lam = a.lams[best]
    print(f"ridge: lam {lam:g}, library two-fold MSE / scalar island " + " ".join(f"{x:.4f}" for x in cv.tolist()), flush=True)

    # the increments of the best strength: the test year from all members, the library year cross-fitted
    pt = os.path.join(a.out, f"isl_inc_{a.name}_{ty}.npy")
    pl = os.path.join(a.out, f"isl_inc_{a.name}_{ly}cv.npy")
    IT = np.lib.format.open_memmap(pt[:-4] + ".part.npy", mode="w+", dtype=np.float16, shape=(NT, len(DYN), H, W))
    IL = np.lib.format.open_memmap(pl[:-4] + ".part.npy", mode="w+", dtype=np.float16, shape=(NL, len(DYN), H, W))
    IT[:] = 0
    IL[:] = 0
    e_sc = torch.zeros(C, dtype=torch.float64)
    e_mx = torch.zeros(C, dtype=torch.float64)
    for j in range(H):
        (G0, Q0, n0), (G1, Q1, n1) = ((G.to(dev), Q.to(dev), n) for G, Q, n in NE[j])
        s_j = sr[0, :, j, 0].to(dev, torch.float64)
        P = (feats(XT, [j]) @ fit(G0 + G1, Q0 + Q1, n0 + n1, lam)) * s_j                       # (NT x W, C)
        P = P.reshape(NT, W, C).permute(0, 2, 1).float().cpu()
        IT[:, sc, j] = P.numpy().astype(np.float16)
        rt = rT[:, :, j].double()
        e_sc += (rt ** 2).sum((0, 2)) * wl[j]
        e_mx += ((rt - P.double()) ** 2).sum((0, 2)) * wl[j]
        for q, (G, Q, n) in ((0, (G1, Q1, n1)), (1, (G0, Q0, n0))):
            Pq = (feats(XL[members[q]], [j]) @ fit(G, Q, n, lam)) * s_j
            IL[members[q].numpy()[:, None], np.asarray(sc)[None, :], j] = (
                Pq.reshape(len(members[q]), W, C).permute(0, 2, 1).float().cpu().numpy().astype(np.float16))
    for mm, path in ((IT, pt), (IL, pl)):
        mm.flush()
        del mm
        os.replace(path[:-4] + ".part.npy", path)
    # the test year's island: the matrix gain against the scalar one (normalized RMSE, mean of the 65 channels)
    nr = lambda e: float(np.sqrt((e / (NT * H * W)).numpy()).mean())  # noqa: E731
    vi = [sc.index(DYN.index(contract.CHANNEL_INDEX[v])) for v in VARS]
    vs = [float(std[contract.CHANNEL_INDEX[v]]) for v in VARS]
    rms = lambda e, q: float(np.sqrt(float(e[q]) / (NT * H * W)))  # noqa: E731
    out = {"core_fc": a.core_fc, "meas": a.meas, "table": a.table, "lib_year": ly, "year": ty, "island": a.island,
           "members": NL, "ics": NT, "step": step, "pool": pool, "lams": a.lams, "cv_mse_ratio": cv.tolist(), "lam": lam,
           "nrmse_scalar": nr(e_sc), "nrmse_matrix": nr(e_mx), "gain_pct": 100 * (1 - nr(e_mx) / nr(e_sc)),
           "vars": {v: {"rmse_scalar": rms(e_sc, q) * s, "rmse_matrix": rms(e_mx, q) * s,
                        "gain_pct": 100 * (1 - rms(e_mx, q) / rms(e_sc, q))} for v, q, s in zip(VARS, vi, vs)},
           "files": [pt, pl], "sec": round(time.time() - t0)}
    print(f"== island day {a.island} of {ty} ({NT} ICs), members of {ly} ({NL}), stencil step {step}, pool +-{pool} rows: "
          f"nRMSE scalar {out['nrmse_scalar']:.4f}, matrix {out['nrmse_matrix']:.4f} ({out['gain_pct']:+.2f} %); "
          + " ".join(f"{v} {out['vars'][v]['gain_pct']:+.2f}" for v in VARS), flush=True)
    print("saved", pt, pl, flush=True)
    json.dump(out, open(a.json or os.path.join(a.out, f"island_matrix_{a.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
