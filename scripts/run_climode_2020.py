"""ClimODE public checkpoint on our 2020 ICs (2 + 8e on the full 1464-step calendar, Feb 29 kept) via
scripts/run_climode_eval.py unchanged except for the data loader: upstream batches two test years side by side
(batch norm in train mode sees both), so 2020 is batched with 2018 as the partner (same IC index) and only 2020 is
kept. Min-max normalization 2006-2018 as upstream. Output <out>/climode_pred.npz with years=[2020] for
compare_step13_climode.py."""
import os, sys, numpy as np, torch, xarray as xr
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "third_party", "climode"))
import utils as cu
from scripts import run_climode_eval as rce
from scripts.run_step11_climode import climode_minmax
E = os.path.join(R, "data/era5_5p625/era5_data")
YEARS = (2020, 2018)  # index 0 = 2020 (kept), 1 = batch partner
mx_all, mn_all = climode_minmax(E)

def loader(data_path, tr, va, te, lev, spectral):
    folder = os.path.basename(os.path.dirname(data_path))
    i = rce.LEVELS.index(lev)
    mx, mn = float(mx_all[i]), float(mn_all[i])
    arrs, lat, lon = [], None, None
    for y in YEARS:
        with xr.open_dataset(os.path.join(E, folder, f"{folder}_{y}_5.625deg.nc")) as d:
            v = d[lev].values.astype(np.float32)[:1460]
            lat, lon = d.lat.values, d.lon.values
        arrs.append((v - mn) / (mx - mn))
    test = torch.from_numpy(np.stack(arrs, 1)[:, :, None])  # (1460, 2, 1, H, W)
    return None, None, test, torch.arange(1460).view(-1, 1), lat, lon, mx, mn, None

cu.get_train_test_data_without_scales_batched = loader
out = sys.argv[1]
ckpt = R + "/runs/climode_34/jan/ClimODE_global.pt"
args = ["--data", E, "--out", out] + (["--ckpt", ckpt] if os.path.exists(ckpt) else []) + sys.argv[2:]
rce.main(args)
p = os.path.join(out, "climode_pred.npz")
z = dict(np.load(p))
for k in ("mean", "std", "truth"):
    z[k] = z[k][:1]
z["years"] = np.array([2020])
np.savez_compressed(os.path.join(out, "climode_pred_2020.npz"), **z)
print("wrote", os.path.join(out, "climode_pred_2020.npz"), z["mean"].shape)
ok = ~np.isnan(z["mean"][0]).any((1, 2, 3, 4))  # ClimODE diverges on some ICs; compare on the valid ones only
print("NaN ICs dropped:", z["ics"][~ok])
for k in ("mean", "std", "truth"):
    z[k] = z[k][:, ok]
z["ics"] = z["ics"][ok]
np.savez(os.path.join(out, "climode_pred_2020_valid.npz"), **z)
