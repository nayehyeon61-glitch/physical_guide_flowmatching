"""Primitive-equation residuals of observed daily-mean ERA5 pairs (KAI files).

Run on the cluster where the KAI yearly files live. For each within-year
pair (X_k, X_{k+1}) it evaluates the time-integrated residual (paper eq. 18,
linear bridge, omega diagnosed with omega_top = 0) on the 1.5 deg grid and
reports area-weighted, above-ground RMS per equation and per level. The
global RMS is the data-driven residual scale b_j of paper eq. (10), and
its size is the daily-mean closure + bridge error that C_j and h_psi must
explain.

    python -m scripts.step3_era5_residual_stats --data-dir /path/to/KAI/train \
        --years 2010 2011 --max-pairs 60 --out experiments/era5_residual_stats.json
"""

import argparse
import glob
import json
import os

import netCDF4 as nc
import numpy as np
import torch

from pgcfm import contract
from pgcfm.physics import grid as G, primitive


def year_files(data_dir, years):
    files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    if years:
        files = [f for f in files if any(str(y) in os.path.basename(f) for y in years)]
    return files


def run(data_dir, years=None, max_pairs=60, stride=6, n_sub=2):
    g = G.SphereGrid()
    w_lat = g.latitude_weight("area")  # [H-2, 1]
    acc = {}
    n_pairs = 0
    for path in year_files(data_dir, years):
        with nc.Dataset(path) as ds:
            n_t = ds.variables["fields"].shape[0]
            for t in range(0, n_t - 1, stride):
                if n_pairs >= max_pairs:
                    break
                X = torch.from_numpy(np.asarray(ds.variables["fields"][t:t + 2], dtype=np.float64))
                F0, F1 = G.fields_from_state(X[0]), G.fields_from_state(X[1])
                sp = torch.minimum(F0["sp"], F1["sp"])
                mask = G.above_ground_mask(sp, g)
                R = G.time_integrated_residuals(F0, F1, g, n_sub=n_sub)
                for k, r in R.items():
                    w = mask * w_lat
                    r2 = torch.where(w > 0, r, torch.zeros_like(r)) ** 2
                    s = acc.setdefault(k, {"sum": torch.zeros(12, dtype=torch.float64),
                                           "w": torch.zeros(12, dtype=torch.float64)})
                    s["sum"] += (r2 * w).sum(dim=(-1, -2))
                    s["w"] += w.sum(dim=(-1, -2))
                n_pairs += 1
    if n_pairs == 0:
        raise SystemExit(f"no pairs found in {data_dir}")
    out = {"n_pairs": n_pairs, "levels_hPa": contract.LEVELS_HPA, "units": {
        "u": "m s-2", "v": "m s-2", "T": "K s-1", "mass": "s-1", "hyd": "K", "q": "kg kg-1 s-1"},
        "default_b": {k: primitive.ResidualScales().of(k) for k in acc}}
    out["rms_by_level"] = {k: (s["sum"] / s["w"].clamp(min=1e-30)).sqrt().tolist() for k, s in acc.items()}
    out["rms_global(=suggested b_j)"] = {k: float((s["sum"].sum() / s["w"].sum()).sqrt()) for k, s in acc.items()}
    # area-weighted share of points above ground; 50 hPa is always above ground
    out["above_ground_area_fraction_by_level"] = (acc["u"]["w"] / acc["u"]["w"][0]).tolist()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--max-pairs", type=int, default=60)
    ap.add_argument("--stride", type=int, default=6, help="days between sampled pairs")
    ap.add_argument("--out", default="experiments/era5_residual_stats.json")
    a = ap.parse_args(argv)
    res = run(a.data_dir, a.years, a.max_pairs, a.stride)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["rms_global(=suggested b_j)"], indent=1))
    return res


if __name__ == "__main__":
    main()
