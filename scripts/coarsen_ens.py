"""Average forecasts of a finer grid (scripts/local_era5_to_kai.py --grid) to the 5.625 deg grid, exactly: each fine
cell splits one 64x32 cell, so the result is scored against the same truth as every 5.625 deg model.

    python scripts/coarsen_ens.py --era5 data/era5_5p625/era5_data --out runs/x/c5p625 \
        runs/x/seqfm_dct_pinn_blend_band_ens_2020.npz runs/x/dyn_gfeat_pred.npz

Ensemble files keep their keys (E_minus_x0 stays float16: it is averaged in float32 and rounded the same way); core
prediction files (keys = years) are averaged the same way. The saved truth O of an ensemble file must match the
5.625 deg truth of --era5 (checked, --no-check to skip), which catches any grid mismatch before a score is written.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract  # noqa: E402
from scripts.run_step11_climode import climode_truth  # noqa: E402

DEG0 = 5.625


def cell_weights(H_fine: int, factor: int) -> np.ndarray:
    """(H_coarse, factor) area shares of the fine rows inside each coarse row; S -> N grid without poles."""
    edges = np.deg2rad(np.linspace(-90.0, 90.0, H_fine + 1))
    a = np.diff(np.sin(edges)).reshape(-1, factor)
    return a / a.sum(1, keepdims=True)


def coarsen(x: np.ndarray, H: int = 32, W: int = 64, lon0: bool = False) -> np.ndarray:
    """(..., H_fine, W_fine) -> (..., H, W), area weighted. A field already on (H, W) is returned as it is.

    lon0=False: the fine cells split the coarse ones (local_era5_to_kai.py --grid), exact.
    lon0=True: a fine grid with a cell centred on 0 deg (WB2 128x64), so a coarse cell holds one whole fine cell and
    half of each neighbour: weights 1/4, 1/2, 1/4 along longitude. Only for factor 2. Not exact: against the 5.625 deg
    analysis it leaves z500 8 m2/s2, t2m 0.18 K (one day of 2016). Averaging lon pairs here would shift the fields by
    half a fine cell (z500 86 m2/s2)."""
    Hf, Wf = x.shape[-2:]
    if (Hf, Wf) == (H, W):
        return x
    f = Hf // H
    if Hf != f * H or Wf != f * W:
        raise ValueError(f"{Hf}x{Wf} is not a multiple of {H}x{W}")
    w = cell_weights(Hf, f).astype(np.float32).reshape(H, f, 1)
    if lon0:
        if f != 2:
            raise ValueError("lon0 grids: factor 2 only")
        y = x.reshape(*x.shape[:-2], H, f, Wf).astype(np.float32)
        y = (y * w).sum(-2)
        return (0.5 * y + 0.25 * np.roll(y, 1, -1) + 0.25 * np.roll(y, -1, -1))[..., ::2]
    y = x.reshape(*x.shape[:-2], H, f, W, f).astype(np.float32)
    return (y.mean(-1) * w).sum(-2)


def convert(path: str, out: str, era5: str, check: bool, lon0: bool = False) -> None:
    z = np.load(path, allow_pickle=False)
    res = {}
    for k in z.files:
        v = z[k]
        res[k] = coarsen(v, lon0=lon0).astype(v.dtype) if v.ndim >= 3 else v
    if check and "O" in res and "ics" in res:
        year = int(os.path.basename(os.path.realpath(path)).rsplit("_", 1)[1].split(".")[0])
        truth = climode_truth(era5, year)  # (T, 5, 32, 64)
        leads = res["leads"]
        steps = (leads // 6).astype(int)
        t = np.stack([truth[int(ic) + steps] for ic in res["ics"]])
        rel = np.sqrt(((res["O"] - t) ** 2).mean((0, 1, 3, 4))) / truth.std((0, 2, 3))
        print(f"{os.path.basename(path)}: truth check, RMS(O - O_5.625) / std per variable {np.round(rel, 4)}", flush=True)
        if rel.max() > 2e-2:  # lon pairs on a lon0 grid: z500 86 / 3358 = 0.026
            raise SystemExit(f"{path}: the averaged truth differs from {era5}; wrong grid or ICs")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, **res)
    os.replace(tmp, out)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--era5", required=True, help="the 5.625 deg era5_data folder (truth check)")
    p.add_argument("--out", required=True, help="output folder; names are kept")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--fine-era5", default=None, help="era5_data folder of the fine grid: picks the longitude weights")
    a = p.parse_args(argv)
    lon0 = False
    if a.fine_era5:
        import xarray as xr
        with xr.open_dataset(os.path.join(a.fine_era5, "constants", f"constants_{contract.era5_deg(a.fine_era5)}deg.nc")) as c:
            lon0 = bool(abs(float(c["lon"].values[0])) < 1e-6)
        print(f"fine grid lon[0] {'= 0: weights 1/4 1/2 1/4' if lon0 else '!= 0: nested cells'}", flush=True)
    for f in a.files:
        o = os.path.join(a.out, os.path.basename(f))
        if not os.path.exists(o):
            convert(f, o, a.era5, not a.no_check, lon0)


if __name__ == "__main__":
    main()
