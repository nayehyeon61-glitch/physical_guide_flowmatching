"""Step 11: PG-CFM against ClimODE (Verma et al., ICLR 2024) and its comparison models on real ERA5.

The owner asked (2026-10-24) to evaluate our best model against ClimODE and the models in its
paper with the ClimODE evaluation. Everything here is on ClimODE's data and protocol:

    data     ERA5 5.625 deg 32x64 (WeatherBench2 public zarr, scripts/wb2_to_kai.py), KAI channel layout,
             one 12 UTC snapshot per day (ClimODE's test ICs are 12 UTC), 24 h PG-CFM steps
    split    train 2006-2015, validation 2016 (every classical fit), test 2017 + 2018 (ClimODE's)
    ICs      ClimODE's: 6 h index 2 + 8 e of each test year = 12 UTC every second day
    scores   the ClimODE authors' functions (third_party/climode/utils.py): latitude-weighted RMSE of the
             ensemble mean, ACC against the time mean of each test year, Gaussian CRPS in ClimODE's
             min-max units; plus ensemble CRPS (properscoring), spread/skill, anomaly power ratio and
             the hydrostatic residual of the members
    vars     z500, t850, t2m directly. u10/v10 are not KAI channels: diagnosed from u1000/v1000 by a
             per-grid-cell linear fit on 2016 analyses (u10 = a + b u1000), labelled as such

Model = the step 9/10 best chain: frozen DCT latent encoder -> latent CFM -> per-member PINN Q -> IDCT,
Markov rollout, then MBM (a + b mu per lead/channel) + per-lead spread scale, change projected on the
hydrostatic tangent space (mos_ck_tan), all fitted on 2016. Arms:

    raw          the serial chain, no calibration
    noQ          PINN Q bypassed at inference (flow matching alone)
    mbm_tan      raw -> MBM + s_kc -> tangent projection (step 9 best)
    mbmx_tan     a + b mu + c x0 per lead/channel (step 10 best deterministic) + s_kc, tangent projected
    persistence, climatology   references (climatology = ClimODE's ACC climatology)

    python scripts/run_step11_climode.py --data data/era5_5p625 --out runs/step11 [--quick]
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from types import SimpleNamespace

import netCDF4 as nc
import numpy as np
import properscoring as ps
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "climode"))
from pgcfm import contract  # noqa: E402
from pgcfm.cfm import rollout  # noqa: E402
from pgcfm.coupling import estimate_scales  # noqa: E402
from pgcfm.data import PGCFMSequenceDataset, kai_data_config  # noqa: E402
from pgcfm.latent import LatentConfig, LatentPGCFM, floors, train_serial, train_stage1  # noqa: E402
from pgcfm.projection import freeze  # noqa: E402
from pgcfm.spread import EnsembleCalibrated  # noqa: E402
from pgcfm.structure import LinearMeanCalibrator, SpreadCalibrator, tangent_calibrated  # noqa: E402
from pgcfm.train import take  # noqa: E402
from scripts.run_step4_pgcfm import member_residuals  # noqa: E402
from scripts.run_step10_deterministic import PersistenceMBM  # noqa: E402
from scripts.train_pgcfm import DEFAULTS, build  # noqa: E402

VARS = ["z", "t", "t2m", "u10", "v10"]  # ClimODE order
FOLDERS = ["geopotential_500", "temperature_850", "2m_temperature", "10m_u_component_of_wind",
           "10m_v_component_of_wind"]
SRC = {"z": "z500", "t": "t850", "t2m": "t2m", "u10": "u1000", "v10": "v1000"}  # KAI channel per ClimODE var
DYN = {v: contract.DYNAMIC_CHANNELS.index(contract.CHANNEL_INDEX[c]) for v, c in SRC.items()}
ARMS = ("raw", "noQ", "mbm_tan", "mbmx_tan", "persistence", "climatology")


def link_years(src, dst, years):
    os.makedirs(dst, exist_ok=True)
    for y in years:
        p = os.path.join(dst, f"{y}.nc")
        if not os.path.exists(p):
            os.symlink(os.path.abspath(os.path.join(src, f"{y}.nc")), p)


def channel_stats(kai, years, out):
    """KAI global_{mean,std}.npy over the training years (per channel, all times and cells)."""
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "global_std.npy")):
        return
    s = ss = n = 0.0
    for y in years:
        with nc.Dataset(os.path.join(kai, f"{y}.nc")) as d:
            f = np.asarray(d["fields"][:], dtype=np.float64)
        s, ss, n = s + f.sum((0, 2, 3)), ss + (f ** 2).sum((0, 2, 3)), n + f.shape[0] * f.shape[2] * f.shape[3]
    m = s / n
    np.save(os.path.join(out, "global_mean.npy"), m.astype(np.float32))
    np.save(os.path.join(out, "global_std.npy"), np.sqrt(np.maximum(ss / n - m ** 2, 0)).astype(np.float32))


def climode_truth(era5_data, year):
    """(1460 or 1464, 5, 32, 64) 6-hourly truth of the ClimODE variables, physical units."""
    import xarray as xr
    out = []
    for v, f in zip(VARS, FOLDERS):
        with xr.open_dataset(os.path.join(era5_data, f, f"{f}_{year}_5.625deg.nc")) as d:
            out.append(d[v].values.astype(np.float32))
    return np.stack(out, 1)


def climode_minmax(era5_data):
    """ClimODE's normalization: min / max of each variable over 2006-2018 (utils.py)."""
    lo, hi = np.full(5, np.inf), np.full(5, -np.inf)
    for y in range(2006, 2019):
        t = climode_truth(era5_data, y)
        lo, hi = np.minimum(lo, t.min((0, 2, 3))), np.maximum(hi, t.max((0, 2, 3)))
    return hi, lo


def climode_scores(P, O, S, clim, lat, mx, mn):
    """Vectorized ClimODE scores (third_party/climode/utils.py), (..., 5, H, W) physical -> (..., 5):
    rmse  sqrt(mean_hw(alpha(h) (P - O)^2)), alpha = cos(lat) / mean cos(lat)       (evaluation_rmsd_mm)
    acc   P' = P - C - mean_hw(P - C), same for O; sum alpha P'O' / sqrt(sum alpha P'^2 sum alpha O'^2)
                                                                                     (evaluation_acc_mm)
    crps_gauss  mean_hw CRPS_Gaussian in ClimODE's min-max units (unweighted)        (evaluation_crps_mm)"""
    w = np.cos(np.deg2rad(lat))
    w = (w / w.mean())[:, None]
    rmse = np.sqrt(((P - O) ** 2 * w).mean((-1, -2)))
    pa, oa = P - clim, O - clim
    pa = pa - pa.mean((-1, -2), keepdims=True)
    oa = oa - oa.mean((-1, -2), keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = (w * pa * oa).sum((-1, -2)) / np.sqrt((w * pa ** 2).sum((-1, -2)) * (w * oa ** 2).sum((-1, -2)))
    sc_ = (mx - mn)[:, None, None]
    crps = ps.crps_gaussian((O - mn[:, None, None]) / sc_, mu=(P - mn[:, None, None]) / sc_, sig=S / sc_).mean((-1, -2))
    return {"rmse": rmse, "acc": acc, "crps_gauss": crps}


def check_climode(cu, P, O, S, clim, lat, lon, mx, mn):
    """Assert climode_scores == the vendored functions on one field (5, H, W)."""
    H, W = P.shape[-2:]
    one, zero = [1.0] * 5, [0.0] * 5
    T = lambda q: torch.from_numpy(np.ascontiguousarray(q))  # noqa: E731
    sc_ = (mx - mn)[:, None, None]
    ref = {"rmse": cu.evaluation_rmsd_mm(T(P), T(O), lat, lon, one, zero, H, W, VARS),
           "acc": cu.evaluation_acc_mm(T(P), T(O), lat, lon, one, zero, H, W, VARS, clim),
           "crps_gauss": [np.mean(v) for v in cu.evaluation_crps_mm(T((P - mn[:, None, None]) / sc_),
                                                                     T((O - mn[:, None, None]) / sc_), lat, lon,
                                                                     one, zero, H, W, VARS, T(S / sc_))]}
    mine = climode_scores(P, O, S, clim, lat, mx, mn)
    for k, v in ref.items():
        np.testing.assert_allclose(np.asarray(v, dtype=np.float64), mine[k], rtol=1e-4, atol=1e-7, equal_nan=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="output folder of scripts/wb2_to_kai.py")
    p.add_argument("--out", default="runs/step11")
    p.add_argument("--train-years", type=int, nargs="+", default=list(range(2006, 2016)))
    p.add_argument("--val-year", type=int, default=2016)
    p.add_argument("--test-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--horizon", type=int, default=6, help="days (ClimODE Table 7 goes to 144 h)")
    p.add_argument("--n-val", type=int, default=60, help="2016 ICs (every 6th day) for the classical fits")
    p.add_argument("--max-ics", type=int, default=None, help="test ICs per year (ClimODE: 179 at 144 h)")
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--k-lat", type=int, default=8)
    p.add_argument("--k-lon", type=int, default=16)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--enc-hidden", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--carry-highpass", action="store_true", help="W1 fix: modes above the DCT cut persist")
    p.add_argument("--carry-tendency", action="store_true",
                   help="x_{k+1} = x_k + D(z_{k+1}) - D(z_k): the autoencoder only carries the tendency")
    p.add_argument("--iters-stage1", type=int, default=1500)
    p.add_argument("--iters-serial", type=int, default=600)
    p.add_argument("--lambda-p", type=float, default=0.1)
    p.add_argument("--grad-ratio", type=float, default=0.5)
    p.add_argument("--tag", default="", help="suffix of the checkpoints / results (variants)")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        a.train_years, a.iters_stage1, a.iters_serial, a.n_val, a.max_ics = a.train_years[:1], 3, 2, 4, 2
        a.members, a.steps, a.horizon = 2, 2, 2
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag + ("_quick" if a.quick else "")  # a quick run never reuses / overwrites full checkpoints
    kai, era5_data = os.path.join(a.data, "kai"), os.path.join(a.data, "era5_data")
    root = os.path.join(a.out, "data")
    tr, va = os.path.join(root, "train"), os.path.join(root, "val")
    link_years(kai, tr, a.train_years)
    link_years(kai, va, [a.val_year])
    for y in a.test_years:
        link_years(kai, os.path.join(root, f"test{y}"), [y])
    stats = os.path.join(root, f"stats{'_quick' if a.quick else ''}")
    channel_stats(kai, a.train_years, stats)
    mean_p, std_p = os.path.join(stats, "global_mean.npy"), os.path.join(stats, "global_std.npy")
    with nc.Dataset(os.path.join(kai, f"{a.val_year}.nc")) as d:
        lat, lon = np.array(d["latitude"][:]), np.array(d["longitude"][:])
    contract.register_latitudes(lat)  # 32 rows without poles (ClimODE grid)
    mean, std = np.load(mean_p), np.load(std_p)
    dcfg = kai_data_config(mean_p, std_p)
    pair_ds = PGCFMSequenceDataset(dcfg, tr, horizon=1)
    H, W = pair_ds.img_shape_x, pair_ds.img_shape_y
    loader = torch.utils.data.DataLoader(pair_ds, batch_size=8, shuffle=True, drop_last=True, num_workers=2,
                                         persistent_workers=True)
    val_pairs = take(torch.utils.data.DataLoader(PGCFMSequenceDataset(dcfg, va, horizon=1, train=False),
                                                 batch_size=8, shuffle=True,
                                                 generator=torch.Generator().manual_seed(0)), 4)

    res_path = os.path.join(a.out, f"results{tag}.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    res["args"] = vars(a)

    def save():
        with open(res_path + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(res_path + ".tmp", res_path)

    # -- physics guide + chain (step 7 recipe) --------------------------------------------------
    torch.manual_seed(0)
    base = build(dict(DEFAULTS, model_channels=32, channel_mult=[1, 2, 2], num_res_blocks=1), H, W, mean, std)
    guide = base.guide
    guide.scales = estimate_scales(guide, take(loader, 10))
    freeze(guide)
    gm = SimpleNamespace(guide=guide)
    res["scales"] = vars(guide.scales)
    lcfg = LatentConfig(k_lat=a.k_lat, k_lon=a.k_lon, latent=a.latent, enc_hidden=tuple(a.enc_hidden),
                        dec_hidden=tuple(reversed(a.enc_hidden)), carry_highpass=a.carry_highpass,
                        carry_tendency=a.carry_tendency)

    s1_ck = os.path.join(a.out, f"stage1{tag}.pt")
    s1 = LatentPGCFM(lcfg, H, W)
    if os.path.exists(s1_ck):
        s1.load_state_dict(torch.load(s1_ck)["latent"])
    else:
        sub = torch.utils.data.default_collate([pair_ds[i] for i in
                                                np.random.default_rng(0).choice(len(pair_ds), min(512, len(pair_ds)), replace=False)])
        s1.fit_dct(sub["x0"], sub["static"])
        t = time.time()
        train_stage1(s1, guide, loader, a.iters_stage1, lambda_p=a.lambda_p, grad_ratio=a.grad_ratio,
                     log_every=max(1, a.iters_stage1 // 10))
        s1.fit_residual_scale(take(loader, 40))
        torch.save({"latent": s1.state_dict(), "secs": time.time() - t}, s1_ck)
        res["stage1_seconds"] = time.time() - t
    freeze(s1.pinn)
    res["floors_val"] = floors(s1, val_pairs)
    print("floors (2016)", json.dumps({k: round(v, 3) for k, v in res["floors_val"].items()}), flush=True)
    save()

    se_ck = os.path.join(a.out, f"serial{tag}.pt")
    serial = LatentPGCFM(dataclasses.replace(lcfg, coupling="serial"), H, W)
    if os.path.exists(se_ck):
        serial.load_state_dict(torch.load(se_ck)["latent"])
    else:
        serial.load_state_dict(s1.state_dict(), strict=False)
        serial.fit_residual_scale(take(loader, 40))
        t = time.time()
        train_serial(serial, guide, loader, a.iters_serial, grad_ratio=a.grad_ratio, lambda_p=a.lambda_p,
                     log_every=max(1, a.iters_serial // 10), ckpt=os.path.join(a.out, f"serial{tag}_train.pt"))
        torch.save({"latent": serial.state_dict(), "secs": time.time() - t}, se_ck)
        res["serial_seconds"] = time.time() - t
        if os.path.exists(os.path.join(a.out, f"serial{tag}_train.pt")):
            os.remove(os.path.join(a.out, f"serial{tag}_train.pt"))
    freeze(serial)
    serial.eval()
    save()

    trans = {"raw": EnsembleCalibrated(serial, "serial"), "noQ": EnsembleCalibrated(serial, "noQ")}

    def roll(name, batch, seed):
        tr_ = trans[name]
        tr_.layout(a.members, "interleave")
        with torch.no_grad():
            return rollout(tr_, batch["x0"], batch["static"], batch["doy"], a.members, steps=a.steps, seed=seed)

    def starts(path, idx):
        ds = PGCFMSequenceDataset(dcfg, path, horizon=a.horizon, train=False)
        return torch.utils.data.default_collate([ds[i] for i in idx if i < len(ds)])

    # -- classical fits on 2016 ---------------------------------------------------------------
    K = a.horizon
    val = starts(va, [6 * i for i in range(a.n_val)])
    vens = roll("raw", val, seed=3)
    mbm = LinearMeanCalibrator(vens.shape[3], K).fit_validation(vens, val["targets"])
    s_mbm = SpreadCalibrator(vens.shape[3], per_lead=True, leads=K)
    s_mbm.fit_validation(mbm(vens), val["targets"])
    mbmx = PersistenceMBM(vens.shape[3], K).fit(vens, val["targets"], val["x0"])
    s_mbmx = SpreadCalibrator(vens.shape[3], per_lead=True, leads=K)
    s_mbmx.fit_validation(mbmx(vens, val["x0"]), val["targets"])
    res["mbmx_coef_z500"] = mbmx.coef[:, DYN["z"]].tolist()
    del vens
    # u10/v10 diagnosed from u1000/v1000 (2016 12 UTC analyses, per grid cell)
    t16 = climode_truth(era5_data, a.val_year)[2::4]
    with nc.Dataset(os.path.join(kai, f"{a.val_year}.nc")) as d:
        f16 = np.asarray(d["fields"][:], dtype=np.float32)
    sfc = {}
    for v, c in (("u10", "u1000"), ("v10", "v1000")):
        x, y = f16[:, contract.CHANNEL_INDEX[c]], t16[:, VARS.index(v)]
        b = ((x - x.mean(0)) * (y - y.mean(0))).sum(0) / ((x - x.mean(0)) ** 2).sum(0).clip(1e-6)
        sfc[v] = (y.mean(0) - b * x.mean(0), b)
    res["sfc_fit_rmse_2016"] = {v: float(np.sqrt(((sfc[v][0] + sfc[v][1] * f16[:, contract.CHANNEL_INDEX[c]]
                                                   - t16[:, VARS.index(v)]) ** 2).mean()))
                                for v, c in (("u10", "u1000"), ("v10", "v1000"))}
    del f16, t16
    save()

    # -- test ------------------------------------------------------------------------------------
    import utils as cu  # ClimODE's evaluation functions (vendored)
    mx, mn = climode_minmax(era5_data)
    wlat = np.cos(np.deg2rad(lat))
    wlat = (wlat / wlat.mean())[:, None]
    sc = {arm: {} for arm in ARMS}  # arm -> metric -> (K, 5) lists over ICs
    resid = {arm: [] for arm in ("raw", "noQ", "mbm_tan", "mbmx_tan", "truth")}
    joint = {}  # ensemble means at ClimODE ICs, for the paired comparison
    unscale = lambda x: x * torch.as_tensor(std[contract.DYNAMIC_CHANNELS])[:, None, None] + \
        torch.as_tensor(mean[contract.DYNAMIC_CHANNELS])[:, None, None]  # noqa: E731

    def to_vars(ens):
        """(..., 66, H, W) normalized -> (..., 5, H, W) physical ClimODE variables."""
        e = unscale(ens)
        out = torch.stack([e[..., DYN[v], :, :] for v in VARS], -3).numpy()
        for v in ("u10", "v10"):
            i = VARS.index(v)
            out[..., i, :, :] = sfc[v][0] + sfc[v][1] * out[..., i, :, :]
        return out

    checked = []
    t0 = time.time()
    for y in a.test_years:
        truth = climode_truth(era5_data, y)
        clim = truth.mean(0)  # ClimODE: time mean of the test year
        n_ic = (len(truth) - 2 - 4 * K) // 8 + 1
        if a.max_ics:
            n_ic = min(n_ic, a.max_ics)
        path = os.path.join(root, f"test{y}")
        joint[y] = {arm: np.zeros((n_ic, K, 5, H, W), np.float32) for arm in ("raw", "mbm_tan", "mbmx_tan")}
        joint[y]["spread"] = np.zeros((n_ic, K, 5, H, W), np.float32)
        for c0 in range(0, n_ic, 8):
            ids = list(range(c0, min(n_ic, c0 + 8)))
            b = starts(path, [2 * e for e in ids])  # day 2e at 12 UTC = ClimODE 6 h index 2 + 8 e
            raw = roll("raw", b, seed=11 + c0)
            ens = {"raw": raw, "noQ": roll("noQ", b, seed=11 + c0)}
            ens["mbm_tan"] = tangent_calibrated(raw, s_mbm(mbm(raw)), guide, b["x0"], b["static"], b["doy"])
            ens["mbmx_tan"] = tangent_calibrated(raw, s_mbmx(mbmx(raw, b["x0"])), guide, b["x0"], b["static"],
                                                 b["doy"])
            for arm in resid:
                if arm == "truth":
                    resid[arm].append(member_residuals(gm, b["x0"], b["targets"][:, None], b["static"], b["doy"]))
                else:
                    resid[arm].append(member_residuals(gm, b["x0"], ens[arm], b["static"], b["doy"]))
            phys = {arm: to_vars(e) for arm, e in ens.items()}  # (B, M, K, 5, H, W)
            for j, e in enumerate(ids):
                obs = truth[[2 + 8 * e + 4 * (k + 1) for k in range(K)]]  # (K, 5, H, W)
                x0 = truth[2 + 8 * e]
                for arm in ARMS:
                    if arm == "persistence":
                        m_, sd_, mem = np.broadcast_to(x0, obs.shape), np.full(obs.shape, 1e-6), None
                    elif arm == "climatology":
                        m_, sd_, mem = np.broadcast_to(clim, obs.shape), np.full(obs.shape, 1e-6), None
                    else:
                        mem = phys[arm][j]
                        m_, sd_ = mem.mean(0), mem.std(0, ddof=1)
                        if arm in joint[y]:
                            joint[y][arm][e] = m_
                        if arm == "mbm_tan":
                            joint[y]["spread"][e] = sd_
                    if not checked:  # the vectorized scores reproduce the vendored ClimODE functions
                        check_climode(cu, m_[0], obs[0], sd_[0], clim, lat, lon, mx, mn)
                        checked.append(arm)
                    row = climode_scores(m_, obs, sd_, clim, lat, mx, mn)
                    if mem is not None:
                        nrm = lambda q: (q - mn[:, None, None]) / (mx - mn)[:, None, None]  # noqa: E731
                        row["crps_ens"] = ps.crps_ensemble(nrm(obs), np.moveaxis(nrm(mem), 0, -1)).mean((-1, -2))
                        row["spread"] = np.sqrt((sd_ ** 2 * wlat).mean((-1, -2)))
                        row["amp"] = np.sqrt(((m_ - clim) ** 2 * wlat).mean((-1, -2)) /
                                             ((obs - clim) ** 2 * wlat).mean((-1, -2)))
                    for key, v in row.items():
                        for k in range(K):
                            sc[arm].setdefault(key, [[] for _ in range(K)])[k].append(np.asarray(v[k]).tolist())
            print(f"{y} ic {ids[-1] + 1}/{n_ic} {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(os.path.join(a.out, f"pgcfm{tag}_pred_{y}.npz"), **joint[y], lat=lat, lon=lon)

    leads = [str(24 * (k + 1)) for k in range(K)]
    res["lead_hours"] = leads
    res["n_ic"] = len(sc["raw"]["rmse"][0])
    for arm in ARMS:
        res[arm] = {m: {h: dict(zip(VARS, np.nanmean(v[k], 0).tolist())) for k, h in enumerate(leads)}
                    for m, v in sc[arm].items()}
        res[arm]["rmse_std"] = {h: dict(zip(VARS, np.std(sc[arm]["rmse"][k], 0).tolist()))
                                for k, h in enumerate(leads)}
    res["residual"] = {arm: {eq: np.mean([r[eq] for r in v], 0).tolist() for eq in v[0]} for arm, v in resid.items()}
    res["eval_seconds"] = time.time() - t0
    save()
    for arm in ARMS:
        print(f"[{arm}] RMSE z/t/t2m/u10/v10:",
              " | ".join(f"{h}h " + "/".join(f"{res[arm]['rmse'][h][v]:.2f}" for v in VARS) for h in leads[::2]),
              flush=True)
    print("hyd", {k: [round(x, 2) for x in v["hyd"]] for k, v in res["residual"].items()}, flush=True)
    return res


if __name__ == "__main__":
    main()
