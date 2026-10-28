"""Step 23: NeuralGCM's public checkpoints on our test ICs, saved in the format of the ClimODE paired comparison.

NeuralGCM (Kochkov et al. 2024, https://arxiv.org/abs/2311.07222) publishes checkpoints under CC BY-SA 4.0 at
gs://neuralgcm/models/: v1/deterministic_{2_8,1_4,0_7}_deg.pkl and v1/stochastic_1_4_deg.pkl. The 2.8 and
1.4 deg deterministic models were trained on ERA5 1979-2017 (validated on 2018), the 0.7 deg and stochastic
ones on 1979-2019; the paper's held-out test year is 2020. So the comparison year here must be 2020 or later
(our ClimODE years 2017-2018 are NeuralGCM training data).

Per IC (our convention: 6 h index 2 + 8 e of the year, 12 UTC every second day):
  1. ERA5 0.25 deg, 37 pressure levels from the public ARCO zarr: the model's input variables (u, v, T, q,
     geopotential, cloud ice and liquid water) and forcings (SST, sea-ice cover), one time slice (about 1 GB);
  2. conservative regridding to the model's Gaussian grid (dinosaur), NaN fill (forcings over land);
  3. encode -> unroll 6 days with 24 h output steps, forcings held at the IC (persistence, as in the authors'
     demo) -> decode to pressure levels; stochastic models: --members M different encode keys;
  4. z500 and t850 (the only ClimODE variables NeuralGCM has: no 2 m temperature or 10 m wind) conservatively
     coarsened to the WB2 5.625 deg grid (64 x 32, no poles) of our truth, written like run_climode_eval.py:
     <out>/neuralgcm_pred.npz with mean, std (1e-6 for deterministic models, member std otherwise), truth,
     lead_hours, ics, years, max_lev, min_lev; t2m / u10 / v10 are NaN. Then
         python scripts/compare_step13_climode.py --era5 $E --climode <out>/neuralgcm_pred.npz --ref-name NeuralGCM ...
     scores every saved arm on exactly those ICs (arms must have <prefix>_ens_2020.npz).

Loading a checkpoint unpickles the authors' file (code execution); the ARCO read is anonymous over HTTPS.
JAX uses a GPU when one is visible; --cpu forces the CPU backend (slow: minutes per IC at 2.8 deg).

    python scripts/run_neuralgcm_eval.py --era5 data/era5_5p625/era5_data --out runs/step23/neuralgcm_2_8 \
        --model v1/deterministic_2_8_deg --year 2020 --ic-offset 10 --ic-stride 20
    python scripts/run_neuralgcm_eval.py ... --model v1/stochastic_1_4_deg --members 8 --out runs/step23/neuralgcm_ens
"""

import argparse
import json
import os
import pickle
import sys
import time
import urllib.request

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from scripts.run_step11_climode import VARS, climode_minmax, climode_truth  # noqa: E402
from scripts.run_step12_core import load_static  # noqa: E402

ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
CKPT = "https://storage.googleapis.com/neuralgcm/models/{model}.pkl"
PICK = {"z": ("geopotential", 500), "t": ("temperature", 850)}  # ClimODE variables NeuralGCM provides
LEADS = [24, 48, 72, 96, 120, 144]


def load_model(path):
    import neuralgcm  # noqa: E402  (JAX import happens here, after JAX_PLATFORMS is set)

    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)


def regridders(era5_latlon, model, lat_out, lon_out):
    """ERA5 0.25 deg -> model grid (conservative, NaN-aware) and model grid -> our 5.625 deg grid."""
    from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

    src = spherical_harmonic.Grid(
        latitude_nodes=era5_latlon.sizes["latitude"], longitude_nodes=era5_latlon.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(era5_latlon.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(era5_latlon.longitude))
    to_model = horizontal_interpolation.ConservativeRegridder(src, model.data_coords.horizontal, skipna=True)
    ours = spherical_harmonic.Grid(latitude_nodes=len(lat_out), longitude_nodes=len(lon_out),
                                   latitude_spacing=xarray_utils.infer_latitude_spacing(np.asarray(lat_out)),
                                   longitude_offset=xarray_utils.infer_longitude_offset(np.asarray(lon_out)))
    assert np.allclose(np.rad2deg(ours.latitudes), lat_out, atol=1e-3), "target grid is not our 5.625 deg grid"
    to_ours = horizontal_interpolation.ConservativeRegridder(model.data_coords.horizontal, ours)
    return to_model, to_ours


def forecast(model, ds_ic, to_model, t0, members, seed):
    """One IC -> dict var -> (M, 7, lon, lat) on the model grid at 0, 24, ..., 144 h.

    Mirrors the authors' forecasting demo: inputs and forcings regridded to the model grid, NaN (SST / sea ice
    over land) filled with the nearest value, encode with the IC forcings, unroll with those forcings held
    (the model takes the nearest forcing in time, so one time slice means persistence).
    """
    import jax
    from dinosaur import xarray_utils

    ds = xarray_utils.regrid_horizontal(ds_ic, to_model)
    ds = xarray_utils.fill_nan_with_nearest(ds)
    ds = ds.assign_coords(time=t0)
    inputs = model.inputs_from_xarray(ds)
    forcing0 = model.forcings_from_xarray(ds)
    forcings = model.forcings_from_xarray(ds[list(model.forcing_variables)].expand_dims("time"))
    times = np.array([0] + LEADS)  # hours after the IC, labels only
    out = {}
    for m in range(max(1, members)):
        state = model.encode(inputs, forcing0, rng_key=jax.random.key(seed + m))
        _, pred = model.unroll(state, forcings, steps=len(times), timedelta=np.timedelta64(24, "h"),
                               start_with_input=True)
        pds = model.data_to_xarray(pred, times=times)
        for v, (name, level) in PICK.items():
            out.setdefault(v, []).append(np.asarray(pds[name].sel(level=level).transpose("time", "longitude", "latitude")))
    return {v: np.stack(a) for v, a in out.items()}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--era5", required=True, help="our era5_data folder (truth, min / max)")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="v1/deterministic_2_8_deg")
    p.add_argument("--checkpoint", default=None, help="local .pkl (default: download <out>/<model>.pkl)")
    p.add_argument("--year", type=int, default=2020)
    p.add_argument("--max-ics", type=int, default=None)
    p.add_argument("--ic-stride", type=int, default=1)
    p.add_argument("--ic-offset", type=int, default=0)
    p.add_argument("--members", type=int, default=0, help="stochastic models: members per IC (0 = deterministic)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="force the JAX CPU backend")
    p.add_argument("--arco", default=ARCO)
    p.add_argument("--forcing-lag", type=int, default=24, help="hours before the IC the SST / sea-ice forcing is taken from")
    p.add_argument("--threads", type=int, default=0, help="XLA CPU threads (0 = default)")
    a = p.parse_args(argv)
    if a.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    if a.threads:
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false "
                                   f"intra_op_parallelism_threads={a.threads}").strip()
    os.makedirs(a.out, exist_ok=True)
    ckpt = a.checkpoint or os.path.join(a.out, a.model.replace("/", "_") + ".pkl")
    if not os.path.exists(ckpt):
        print(f"downloading {CKPT.format(model=a.model)}", flush=True)
        urllib.request.urlretrieve(CKPT.format(model=a.model), ckpt)
    t0 = time.time()
    model = load_model(ckpt)
    print(f"model {a.model}: load {time.time() - t0:.0f} s, timestep {model.timestep}, inputs {model.input_variables}, "
          f"forcings {model.forcing_variables}, grid {model.data_coords.horizontal.longitude_nodes}x"
          f"{model.data_coords.horizontal.latitude_nodes}", flush=True)

    lat, lon, _, _ = load_static(a.era5)
    truth_all = climode_truth(a.era5, a.year)  # (T, 5, 32, 64) physical
    n_all = (len(truth_all) - 2 - LEADS[-1] // 6) // 8 + 1
    es = list(range(a.ic_offset, n_all, a.ic_stride))
    if a.max_ics:
        es = es[:a.max_ics]
    ics = [2 + 8 * e for e in es]
    mx, mn = climode_minmax(a.era5)
    iz, it = VARS.index("z"), VARS.index("t")

    arco = xr.open_zarr(a.arco, chunks=None, storage_options={"client_kwargs": {"trust_env": True}})
    need = list(model.input_variables) + list(model.forcing_variables)
    to_model, to_ours = regridders(arco[need[:1]], model, lat, lon)
    H, W = len(lat), len(lon)
    M = max(1, a.members)
    mean = np.full((1, len(ics), len(LEADS), 5, H, W), np.nan, np.float32)
    std = np.full_like(mean, 1e-6)
    members = np.full((len(ics), M, len(LEADS), 2, H, W), np.nan, np.float32)
    truth = np.zeros((1, len(ics), len(LEADS) + 1, 5, H, W), np.float32)
    log = {"model": a.model, "year": a.year, "ics": ics, "members": a.members, "seconds": {}, "rmse_z500": {}, "rmse_t850": {},
           "persistence_z500": {}, "persistence_t850": {}}
    w = np.cos(np.deg2rad(lat))[:, None]
    w = w / w.mean()
    for e, ic in enumerate(ics):
        tic = time.time()
        t_ic = np.datetime64(f"{a.year}-01-01") + np.timedelta64(6 * ic, "h")
        # inputs at the IC; forcings (SST, sea ice) from 24 h earlier, as in the authors' demo (selective_temporal_shift
        # of the forcing variables by 24 h: the forcing is what was known the day before)
        ds_ic = xr.merge([arco[list(model.input_variables)].sel(time=t_ic).compute(),
                          arco[list(model.forcing_variables)].sel(time=t_ic - np.timedelta64(a.forcing_lag, "h"))
                          .compute().drop_vars("time")])
        t_read = time.time() - tic
        fc = forecast(model, ds_ic, to_model, t_ic, a.members, a.seed + 1000 * e)  # var -> (M, 7, lon, lat)
        for j, h in enumerate(LEADS):
            for k, (v, iv) in enumerate((("z", iz), ("t", it))):
                fields = np.stack([np.asarray(to_ours(fc[v][m, j + 1])).T for m in range(M)])  # (M, lat, lon)
                members[e, :, j, k] = fields
                mean[0, e, j, iv] = fields.mean(0)
                if M > 1:
                    std[0, e, j, iv] = fields.std(0, ddof=1)
        truth[0, e, 0] = truth_all[ic]
        for j, h in enumerate(LEADS):
            truth[0, e, j + 1] = truth_all[ic + h // 6]
        for key, iv in (("z500", iz), ("t850", it)):
            err = mean[0, e, :, iv] - truth[0, e, 1:, iv]
            pers = truth[0, e, :1, iv] - truth[0, e, 1:, iv]
            log["rmse_" + key][str(ic)] = np.sqrt((err ** 2 * w).mean((-1, -2))).tolist()
            log["persistence_" + key][str(ic)] = np.sqrt((pers ** 2 * w).mean((-1, -2))).tolist()
        log["seconds"][str(ic)] = {"read": round(t_read, 1), "total": round(time.time() - tic, 1)}
        print(f"ic {e + 1}/{len(ics)} ({t_ic}): read {t_read:.0f} s, total {time.time() - tic:.0f} s | z500 RMSE "
              + " ".join(f"{v:.0f}" for v in log["rmse_z500"][str(ic)]) + " (persistence "
              + " ".join(f"{v:.0f}" for v in log["persistence_z500"][str(ic)]) + ")", flush=True)
        np.savez_compressed(os.path.join(a.out, "neuralgcm_pred.npz"), mean=mean[:, :e + 1], std=std[:, :e + 1],
                            truth=truth[:, :e + 1], members=members[:e + 1], lead_hours=np.array(LEADS),
                            ics=np.array(ics[:e + 1]), years=np.array([a.year]), max_lev=mx, min_lev=mn)
        with open(os.path.join(a.out, "neuralgcm_eval.json"), "w") as f:
            json.dump(log, f, indent=1)
    n = len(ics)
    for key in ("z500", "t850"):
        r = np.sqrt(np.mean([np.square(log["rmse_" + key][str(ic)]) for ic in ics], 0))
        pr = np.sqrt(np.mean([np.square(log["persistence_" + key][str(ic)]) for ic in ics], 0))
        print(f"[{a.model}] {n} ICs of {a.year}, {key} RMSE 24..144 h: " + " ".join(f"{v:.2f}" for v in r)
              + " | persistence " + " ".join(f"{v:.2f}" for v in pr), flush=True)


if __name__ == "__main__":
    main()
