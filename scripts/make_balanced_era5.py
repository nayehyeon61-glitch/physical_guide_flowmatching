"""Synthetic KAI-format ERA5 with hydrostatic and geostrophic balance.

``make_synthetic_era5.py`` gives smooth AR(1) fields that satisfy none of
the primitive equations, so a physics guide trained on it can only pull the
generator AWAY from the data. For the step 4 coupling test the truth has to
lie (approximately) on the constraint set A_delta of eq. (11). This script
writes the same files and then rebuilds, for every day,

  * z (geopotential) by integrating the hydrostatic equation upward from
    the 1000 hPa value with the file's own T and q (T_v = T(1 + 0.608 q)),
    then T back from z with the residual's own d/dln p,
  * u, v as the geostrophic wind of that z, with f replaced by
    sign(phi) * max(|f|, f(15 deg)) so the equatorial band stays finite.

So the true pairs have a small hydrostatic residual (discretization only)
and a momentum residual dominated by the AR(1) tendency, advection and the
equatorial f clamp. T, q and the surface fields keep the AR(1) dynamics.
It is still NOT a dynamical model: the momentum and T equations are not
integrated in time.

    python scripts/make_balanced_era5.py --out data/balanced --years 2011 --days 60 --n-lat 33 --n-lon 64
"""

import argparse
import math
import os
import sys

import netCDF4 as nc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract  # noqa: E402
from scripts.make_synthetic_era5 import main as make_synthetic  # noqa: E402

F_CLAMP_DEG = 15.0


def balance(fields: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """fields (T, 67, H, W) in SI -> same with balanced z, u, v."""
    c, k = contract.CONST, contract.level_slice
    out = fields.astype(np.float64).copy()
    T, q = out[:, k("t")], out[:, k("q")]
    tv = T * (1.0 + 0.608 * q)
    lnp = np.log(contract.LEVELS_PA)
    z = out[:, k("z")]
    # levels are ordered 50 -> 1000 hPa; keep z1000, integrate upward
    for j in range(contract.N_LEVELS - 2, -1, -1):
        z[:, j] = z[:, j + 1] + c.r_dry * 0.5 * (tv[:, j] + tv[:, j + 1]) * (lnp[j + 1] - lnp[j])
    out[:, k("z")] = z
    # Re-derive T from z with the same centred d/dlnp as physics.grid, so the
    # hydrostatic residual of the truth is zero up to round-off, not ~3 K of
    # trapezoid-vs-centred mismatch on the uneven KAI levels.
    dz_dlnp = np.gradient(z, lnp, axis=1, edge_order=2)
    out[:, k("t")] = -dz_dlnp / c.r_dry / (1.0 + 0.608 * q)

    phi = np.deg2rad(lat_deg)[:, None]
    f = 2.0 * c.omega * np.sin(phi)
    fmin = 2.0 * c.omega * math.sin(math.radians(F_CLAMP_DEG))
    f = np.where(phi >= 0, 1.0, -1.0) * np.maximum(np.abs(f), fmin)
    n_lon = fields.shape[-1]
    dlam = 2.0 * math.pi / n_lon
    cos = np.maximum(np.cos(phi), np.cos(np.deg2rad(89.0)))
    dz_dlam = (np.roll(z, -1, -1) - np.roll(z, 1, -1)) / (2 * dlam)
    dz_dphi = np.gradient(z, np.deg2rad(lat_deg), axis=-2, edge_order=2)
    out[:, k("u")] = -dz_dphi / (f * c.earth_radius)
    out[:, k("v")] = dz_dlam / (f * c.earth_radius * cos)
    return out.astype(np.float32)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--years", type=int, nargs="+", default=[2011])
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--n-lat", type=int, default=33)
    p.add_argument("--n-lon", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    make_synthetic(["--out", a.out, "--years", *map(str, a.years), "--days", str(a.days),
                    "--n-lat", str(a.n_lat), "--n-lon", str(a.n_lon), "--seed", str(a.seed)])
    for y in a.years:
        with nc.Dataset(os.path.join(a.out, f"{y}.nc"), "r+") as ds:
            lat = np.array(ds["latitude"][:])
            ds["fields"][:] = balance(np.array(ds["fields"][:]), lat)
            ds.title = "SYNTHETIC KAI-format data, hydrostatic + geostrophic z/u/v (not a dynamical model)"
    # Stats: balanced u, v are not centred on the KAI means; use this data's own.
    stats = os.path.join(a.out, "stats")
    arr = np.concatenate([np.array(nc.Dataset(os.path.join(a.out, f"{y}.nc"))["fields"][:]) for y in a.years])
    np.save(os.path.join(stats, "global_mean.npy"), arr.mean(axis=(0, 2, 3)).astype(np.float32))
    np.save(os.path.join(stats, "global_std.npy"), arr.std(axis=(0, 2, 3)).astype(np.float32))


if __name__ == "__main__":
    main()
