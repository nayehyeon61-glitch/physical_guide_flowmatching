"""Write small synthetic ERA5 yearly files in the exact KAI NetCDF layout.

For pipeline tests only: values are smooth random fields around the KAI
channel statistics, with orography-consistent surface pressure and
astronomical TOA insolation. They do NOT satisfy the primitive equations,
so physics diagnostics on this data are meaningless.

    python scripts/make_synthetic_era5.py --out data/synthetic --years 2011 2012 --days 20
"""

import argparse
import os
import sys

import netCDF4 as nc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pgcfm import contract  # noqa: E402

SOLAR_CONSTANT = 1361.0


def smooth_noise(rng, shape, n_lat, n_lon, n_modes=6):
    """Sum of low-wavenumber spherical-ish harmonics, unit variance."""
    lat = np.deg2rad(np.linspace(-90, 90, n_lat))[:, None]
    lon = np.deg2rad(np.linspace(0, 360, n_lon, endpoint=False))[None, :]
    out = np.zeros(shape + (n_lat, n_lon))
    for _ in range(n_modes):
        k = rng.integers(0, 5)
        m = rng.integers(1, 4)
        amp = rng.standard_normal(shape)[..., None, None]
        phase = rng.uniform(0, 2 * np.pi, shape)[..., None, None]
        out += amp * np.cos(k * lon + phase) * np.cos(lat) ** k * np.sin(m * lat + phase)
    return out / np.sqrt(n_modes / 4.0)


def daily_toa_insolation(doy, lat_deg, lon_deg):
    """KAI daily-mean tisr [J m-2], shape (lat, lon) (circular orbit approximation).

    ERA5 tisr is a 1-hour accumulation; KAI's daily value is the mean of the
    00/06/12/18 UTC fields (KAI stats: mean 1.07e6, i.e. ~300 W m-2 x 3600 s).
    So it is NOT the 24 h integral and it depends on longitude: four samples
    alias the diurnal cycle. Each hourly accumulation is approximated by the
    flux at the middle of its hour.
    """
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (doy + 284) / 365.0)
    phi = np.deg2rad(np.asarray(lat_deg))[:, None]
    lon = np.asarray(lon_deg)[None, :]
    acc = 0.0
    for utc in (0.0, 6.0, 12.0, 18.0):
        hour_angle = np.deg2rad(15.0 * (utc - 0.5 + lon / 15.0 - 12.0))
        cos_z = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
        acc = acc + SOLAR_CONSTANT * np.clip(cos_z, 0.0, None) * 3600.0
    return acc / 4.0


def make_year(path, year, n_days, n_lat, n_lon, seed):
    rng = np.random.default_rng(seed)
    mean, std = contract.REFERENCE_MEAN, contract.REFERENCE_STD
    lat = np.linspace(-90.0, 90.0, n_lat)
    lon = np.arange(n_lon) * (360.0 / n_lon)
    C = contract.N_CHANNELS
    idx = contract.CHANNEL_INDEX

    # Static orography: a few Gaussian mountains, geopotential in m2 s-2.
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    orog = np.zeros((n_lat, n_lon))
    for c_lat, c_lon, h in [(30, 90, 5000), (-20, 290, 4000), (45, 260, 2500), (-75, 0, 3000)]:
        orog += h * np.exp(-(((la - c_lat) / 12) ** 2 + ((lo - c_lon) / 15) ** 2))
    orog *= contract.CONST.gravity

    # AR(1) in time around the channel means.
    fields = np.empty((n_days, C, n_lat, n_lon), dtype=np.float32)
    anom = smooth_noise(rng, (C,), n_lat, n_lon)
    for d in range(n_days):
        anom = 0.8 * anom + 0.6 * smooth_noise(rng, (C,), n_lat, n_lon)
        f = mean[:, None, None] + 0.5 * std[:, None, None] * anom
        q = contract.level_slice("q")
        f[q] = np.abs(f[q])  # specific humidity >= 0
        sp = 101325.0 * np.exp(-orog / (contract.CONST.r_dry * 255.0))
        f[idx["sp"]] = sp + 0.1 * std[idx["sp"]] * anom[idx["sp"]]
        f[idx["orog"]] = orog
        f[idx["tisr"]] = daily_toa_insolation(d, lat, lon)
        fields[d] = f

    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", n_days)
        ds.createDimension("channel", C)
        ds.createDimension("latitude", n_lat)
        ds.createDimension("longitude", n_lon)
        v = ds.createVariable("fields", "f4", ("time", "channel", "latitude", "longitude"), fill_value=np.nan)
        v[:] = fields
        t = ds.createVariable("time", "i8", ("time",))
        t.units = f"days since {year}-01-01"
        t.calendar = "proleptic_gregorian"
        t[:] = np.arange(n_days)
        ch = ds.createVariable("channel", str, ("channel",))
        for i, name in enumerate(contract.CHANNEL_NAMES):
            ch[i] = name
        ds.createVariable("latitude", "f8", ("latitude",))[:] = lat
        ds.createVariable("longitude", "f8", ("longitude",))[:] = lon
        ds.title = "SYNTHETIC KAI-format data for pipeline tests (not physical)"


def write_stats(out_dir):
    stats = os.path.join(out_dir, "stats")
    os.makedirs(stats, exist_ok=True)
    np.save(os.path.join(stats, "global_mean.npy"), contract.REFERENCE_MEAN.astype(np.float32))
    np.save(os.path.join(stats, "global_std.npy"), contract.REFERENCE_STD.astype(np.float32))
    return stats


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--years", type=int, nargs="+", default=[2011, 2012])
    p.add_argument("--days", type=int, default=20)
    p.add_argument("--n-lat", type=int, default=contract.N_LAT)
    p.add_argument("--n-lon", type=int, default=contract.N_LON)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for i, y in enumerate(a.years):
        make_year(os.path.join(a.out, f"{y}.nc"), y, a.days, a.n_lat, a.n_lon, a.seed + i)
    write_stats(a.out)


if __name__ == "__main__":
    main()
