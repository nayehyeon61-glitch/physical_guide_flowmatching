"""Step 11a: run the published ClimODE checkpoint on the ClimODE test set with the authors' code.

Follows third_party/climode/evaluation_global.py (vendored unchanged) with three changes that the
upstream script leaves to the user: the test velocities are fitted here (upstream loads
'### Test velocity here'), the checkpoint is a full pickled module (torch.load weights_only=False),
and the lead can go past 42 h (``--lead-steps``; the ODE is integrated over the longer window,
as for the paper's Table 7). Initial conditions are upstream's: index 2 + 8 e of each test year
(12 UTC, every 2 days), 2017 and 2018 side by side.

Writes ``<out>/climode_scores.json`` (lead -> var -> mean RMSE / ACC / CRPS, upstream definitions)
and ``<out>/climode_pred.npz`` (de-normalized mean and std at every 24 h lead, for the joint
scoring in run_step11_climode.py), plus persistence / climatology scored with the same functions.

    python scripts/run_climode_eval.py --data data/era5_5p625/era5_data --out runs/step11 [--max-ics 8]
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CLIMODE = os.path.join(HERE, "..", "third_party", "climode")
CKPT_URL = "https://raw.githubusercontent.com/Aalto-QuML/ClimODE/e729d23e8799ce0e075699e76d60227d848d8d0c/checkpoints/ClimODE_global.pt"
LEVELS = ["z", "t", "t2m", "u10", "v10"]
FOLDERS = ["geopotential_500", "temperature_850", "2m_temperature", "10m_u_component_of_wind",
           "10m_v_component_of_wind"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="WB1-style era5_data folder (scripts/wb2_to_kai.py)")
    p.add_argument("--out", default="runs/step11")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--max-ics", type=int, default=None, help="per test year (upstream: all 182)")
    p.add_argument("--lead-steps", type=int, default=24, help="6 h steps (upstream eval: 7)")
    p.add_argument("--solver", default="euler")
    p.add_argument("--eval-mode", action="store_true",
                   help="model.eval(); upstream evaluation_global.py never calls it, so its batch norm uses the "
                        "statistics of the two test years in the batch and its dropout is on")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.abspath(a.out)
    sys.path.insert(0, CLIMODE)
    import model_function  # noqa: F401  (classes for the pickled checkpoint)
    from model_function import Optim_velocity
    import utils as cu

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = False  # 2x faster for these 128-channel 32x64 convs on CPU
    ckpt = a.ckpt or os.path.join(out, "ClimODE_global.pt")
    if not os.path.exists(ckpt):
        urllib.request.urlretrieve(CKPT_URL, ckpt)

    # -- data, exactly as upstream (min-max over 2006-2018, Feb 29 dropped) ---------------------
    test = []
    max_lev, min_lev = [], []
    for lev, folder in zip(LEVELS, FOLDERS):
        _, _, te, time_steps, lat, lon, mx, mn, _ = cu.get_train_test_data_without_scales_batched(
            os.path.join(a.data, folder, "*.nc"), slice("2006", "2016"), slice("2016", "2016"),
            slice("2017", "2018"), lev, 0)
        test.append(te)
        max_lev.append(mx)
        min_lev.append(mn)
    test = torch.cat(test, dim=2)  # (1460, 2 years, 5, 32, 64)
    const, lat_map, lon_map = cu.add_constant_info([os.path.join(a.data, "constants", "constants_5.625deg.nc")])
    H, W = test.shape[3], test.shape[4]
    clim = torch.mean(test, dim=0)  # upstream ACC climatology: time mean of each test year

    n_ic = (len(test) - 2 - a.lead_steps) // 8 + 1
    if a.max_ics:
        n_ic = min(n_ic, a.max_ics)
    ics = [2 + 8 * e for e in range(n_ic)]

    # -- test velocities with upstream fit_velocity (kernel cached in the out folder) -------------
    cwd = os.getcwd()
    os.chdir(out)
    try:
        if not os.path.exists("kernel.npy"):
            t0 = time.time()
            cu.get_gauss_kernel((H, W), lat, lon)
            print(f"kernel {time.time() - t0:.0f}s", flush=True)
        kernel = torch.from_numpy(np.load("kernel.npy"))
        vel_name = f"test_{n_ic}ic"
        if not os.path.exists(vel_name + "_vel.npy"):
            t0 = time.time()
            idx = torch.arange(1460).view(-1, 1)
            data_loader = torch.utils.data.DataLoader(test[2:2 + 8 * n_ic], batch_size=8, shuffle=False)
            time_loader = torch.utils.data.DataLoader(time_steps[2:2 + 8 * n_ic], batch_size=8, shuffle=False)
            idx_loader = torch.utils.data.DataLoader(idx[2:2 + 8 * n_ic], batch_size=8, shuffle=False)
            cu.fit_velocity(idx_loader, time_loader, test, data_loader, torch.device("cpu"), 2, LEVELS, 0, H, W,
                            types=vel_name, vel_model=Optim_velocity, kernel=kernel, lat=lat, lon=lon)
            print(f"velocity fit {time.time() - t0:.0f}s", flush=True)
        vel = torch.from_numpy(np.load(vel_name + "_vel.npy"))
    finally:
        os.chdir(cwd)

    model = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.method = a.solver
    if a.eval_mode:
        model.eval()

    L = a.lead_steps
    keys = ("rmse", "acc", "crps")
    arms = ("climode", "persistence", "climatology")
    sc = {arm: {k: [[[] for _ in LEVELS] for _ in range(L)] for k in keys} for arm in arms}
    day_leads = list(range(4, L + 1, 4))
    pred_mean = np.zeros((2, n_ic, len(day_leads), 5, H, W), np.float32)
    pred_std = np.zeros_like(pred_mean)
    truth = np.zeros((2, n_ic, len(day_leads) + 1, 5, H, W), np.float32)
    scale = np.array(max_lev) - np.array(min_lev)
    t0 = time.time()
    for e, ic in enumerate(ics):
        batch = test[ic:ic + L + 1]
        T = time_steps[ic:ic + L + 1].float().flatten()
        model.update_param([vel[e].view(2, 2 * 5, H, W), const, lat_map, lon_map])
        with torch.no_grad():
            mean, std, _ = model(T, batch[0].view(2, 1, 5, H, W))
        for yr in range(2):
            for k in range(1, L + 1):
                tr = batch[k, yr]
                for arm, m, s in (("climode", mean[k, yr], std[k, yr]),
                                  ("persistence", batch[0, yr], torch.full_like(tr, 1e-6)),
                                  ("climatology", clim[yr], torch.full_like(tr, 1e-6))):
                    r = cu.evaluation_rmsd_mm(m, tr, lat, lon, max_lev, min_lev, H, W, LEVELS)
                    c = cu.evaluation_acc_mm(m, tr, lat, lon, max_lev, min_lev, H, W, LEVELS, clim[yr].numpy())
                    cr = cu.evaluation_crps_mm(m, tr, lat, lon, max_lev, min_lev, H, W, LEVELS, s)
                    for i in range(5):
                        sc[arm]["rmse"][k - 1][i].append(r[i])
                        sc[arm]["acc"][k - 1][i].append(float(c[i]))
                        sc[arm]["crps"][k - 1][i].append(float(np.mean(cr[i])))
            truth[yr, e, 0] = batch[0, yr].numpy() * scale[:, None, None] + np.array(min_lev)[:, None, None]
            for j, k in enumerate(day_leads):
                pred_mean[yr, e, j] = mean[k, yr].numpy() * scale[:, None, None] + np.array(min_lev)[:, None, None]
                pred_std[yr, e, j] = std[k, yr].numpy() * scale[:, None, None]
                truth[yr, e, j + 1] = batch[k, yr].numpy() * scale[:, None, None] + np.array(min_lev)[:, None, None]
        if e % 10 == 0:
            print(f"ic {e + 1}/{n_ic} {time.time() - t0:.0f}s", flush=True)

    res = {"n_ic_per_year": n_ic, "ics": ics, "lead_hours": [6 * k for k in range(1, L + 1)], "levels": LEVELS}
    for arm in arms:
        res[arm] = {k: {str(6 * (j + 1)): {lev: float(np.mean(sc[arm][k][j][i])) for i, lev in enumerate(LEVELS)}
                        for j in range(L)} for k in keys}
        res[arm]["rmse_std"] = {str(6 * (j + 1)): {lev: float(np.std(sc[arm]["rmse"][j][i]))
                                                    for i, lev in enumerate(LEVELS)} for j in range(L)}
    with open(os.path.join(out, "climode_scores.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(os.path.join(out, "climode_pred.npz"), mean=pred_mean, std=pred_std, truth=truth,
                        lead_hours=np.array([6 * k for k in day_leads]), ics=np.array(ics),
                        max_lev=np.array(max_lev), min_lev=np.array(min_lev))
    for h in ("6", "24", "36", "72", "144"):
        if h in res["climode"]["rmse"]:
            print(h, "h RMSE", {k: round(v, 3) for k, v in res["climode"]["rmse"][h].items()},
                  "ACC", {k: round(v, 3) for k, v in res["climode"]["acc"][h].items()}, flush=True)
    return res


if __name__ == "__main__":
    main()
