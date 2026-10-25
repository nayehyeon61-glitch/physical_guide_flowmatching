"""Create clearly labelled SYNTHETIC Step13 pipeline data, never observed ERA5.

The KAI layout, 6-hour cadence and ClimODE cell-centred grid match the model
contract. Smooth stochastic fields do NOT satisfy the primitive equations;
scores/physical residuals on these data cannot establish climate forecast skill.

    python scripts/make_step13_synthetic6h.py --out data/step13_synthetic6h
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pgcfm import contract  # noqa: E402


def coordinates():
    return -90.0 + (np.arange(32) + 0.5) * 5.625, np.arange(64) * 5.625


def static_fields(latitude, longitude):
    lat, lon = np.meshgrid(latitude, longitude, indexing="ij")
    orography_m = np.zeros_like(lat)
    for lat0, lon0, height in ((30, 90, 3500), (-20, 290, 2800), (45, 260, 1800), (-75, 0, 2200)):
        dlon = (lon - lon0 + 180) % 360 - 180
        orography_m += height * np.exp(-((lat - lat0) / 14) ** 2 - (dlon / 20) ** 2)
    lsm = (np.sin(np.deg2rad(lon * 1.5)) + 0.45 * np.cos(np.deg2rad(lat * 2)) > 0.3).astype(np.float32)
    return orography_m, lsm


def tisr_1h(times, latitude, longitude, n_sub=12):
    """Same hourly-accumulation astronomical approximation as wb2_to_kai.py.

    The synthetic field is known forcing; it is not measured ERA5 radiation.
    """
    lat = np.deg2rad(latitude)[:, None]
    out = np.empty((len(times), len(latitude), len(longitude)), dtype=np.float32)
    for index, valid in enumerate(times):
        valid = np.datetime64(valid, "s")
        doy = float((valid - valid.astype("datetime64[Y]")) / np.timedelta64(1, "s")) / 86400.0
        total = np.zeros((len(latitude), len(longitude)), dtype=np.float64)
        for sub in range(n_sub):
            day = doy - (sub + 0.5) / (n_sub * 24)
            declination = np.deg2rad(23.44) * np.sin(2 * np.pi * (day + 284 - 1) / 365)
            hour_angle = np.deg2rad(15 * ((day % 1) * 24 + longitude[None] / 15 - 12))
            cos_zenith = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
            total += 1361.0 * np.maximum(cos_zenith, 0) * 3600 / n_sub
        out[index] = total
    return out


def make_fields(year, steps, seed, latitude, longitude, orography_m):
    rng = np.random.default_rng(seed)
    channels = contract.N_CHANNELS
    lat = np.deg2rad(latitude)[:, None]
    lon = np.deg2rad(longitude)[None]
    modes = np.stack([
        np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
        np.cos(lat) ** 2 * np.cos(2 * lon), np.cos(lat) ** 2 * np.sin(2 * lon),
        np.sin(lat) * np.ones_like(lon), np.sin(2 * lat) * np.cos(lon),
    ])
    modes /= np.maximum(np.sqrt((modes ** 2).mean(axis=(1, 2), keepdims=True)), 1e-8)
    coefficients = rng.normal(size=(channels, len(modes))) / np.sqrt(len(modes))
    innovations = np.sqrt(1 - 0.98 ** 2)
    mean, std = contract.REFERENCE_MEAN[:, None, None], contract.REFERENCE_STD[:, None, None]
    output = np.empty((steps, channels, len(latitude), len(longitude)), dtype=np.float32)
    index = contract.CHANNEL_INDEX
    surface_pressure = 101325 * np.exp(-orography_m * contract.CONST.gravity / (contract.CONST.r_dry * 270))
    for step in range(steps):
        coefficients = 0.98 * coefficients + innovations * rng.normal(size=coefficients.shape) / np.sqrt(len(modes))
        anomaly = np.einsum("cm,mhw->chw", coefficients, modes, optimize=True)
        # Keep the fields moderate and differentiable in space; no abrupt noise.
        anomaly = 2 * np.tanh(anomaly / 2)
        field = mean + 0.35 * std * anomaly
        field[contract.level_slice("q")] = mean[contract.level_slice("q")] * np.exp(
            0.25 * anomaly[contract.level_slice("q")]
        )
        field[contract.level_slice("t")] = np.clip(field[contract.level_slice("t")], 180, 330)
        for wind in ("u", "v"):
            field[contract.level_slice(wind)] = np.clip(field[contract.level_slice(wind)], -50, 50)
        field[index["sp"]] = surface_pressure + 300 * anomaly[index["sp"]]
        field[index["msl"]] = 101325 + 500 * anomaly[index["msl"]]
        field[index["tcwv"]] = 18.0 * np.exp(0.2 * anomaly[index["tcwv"]])
        field[index["orog"]] = orography_m * contract.CONST.gravity
        field[index["skt"]] = field[index["t2m"]] + 0.5 * anomaly[index["skt"]]
        output[step] = field
    times = np.datetime64(f"{year}-01-01T00", "h") + np.arange(steps) * np.timedelta64(6, "h")
    output[:, index["tisr"]] = tisr_1h(times, latitude, longitude)
    if not np.isfinite(output).all():
        raise RuntimeError("nonfinite synthetic fields")
    if output[:, contract.level_slice("q")].min() < 0 or output[:, index["tcwv"]].min() < 0:
        raise RuntimeError("negative synthetic humidity/water")
    if min(output[:, contract.level_slice("t")].min(), output[:, index["t2m"]].min(),
           output[:, index["sp"]].min()) <= 0:
        raise RuntimeError("nonpositive synthetic temperature/pressure")
    return output


def write_year(path, year, fields, latitude, longitude):
    import netCDF4 as nc
    temporary = path.with_suffix(".nc.tmp")
    with nc.Dataset(temporary, "w", format="NETCDF4") as out:
        for dimension, size in (("time", len(fields)), ("channel", contract.N_CHANNELS), ("latitude", 32), ("longitude", 64)):
            out.createDimension(dimension, size)
        out.createVariable("fields", "f4", ("time", "channel", "latitude", "longitude"),
                           zlib=True, complevel=1, shuffle=True)[:] = fields
        time = out.createVariable("time", "i8", ("time",))
        time.units, time.calendar = f"hours since {year}-01-01 00:00:00", "proleptic_gregorian"
        time[:] = np.arange(len(fields)) * 6
        names = out.createVariable("channel", str, ("channel",))
        for index, name in enumerate(contract.CHANNEL_NAMES):
            names[index] = name
        out.createVariable("latitude", "f8", ("latitude",))[:] = latitude
        out.createVariable("longitude", "f8", ("longitude",))[:] = longitude
        out.title = "SYNTHETIC Step13 6-hourly pipeline data — NOT observed ERA5"
        out.data_kind = "synthetic_pipeline_only"
        out.physics_skill_valid = "false"
        out.disclaimer = "Smooth stochastic fields are not primitive-equation balanced. Scores do not establish climate forecast skill."
    temporary.replace(path)


def write_constants(path, latitude, longitude, orography_m, lsm):
    import netCDF4 as nc
    temporary = path.with_suffix(".nc.tmp")
    lon2d, lat2d = np.meshgrid(longitude, latitude)
    with nc.Dataset(temporary, "w", format="NETCDF4") as out:
        out.createDimension("lat", len(latitude))
        out.createDimension("lon", len(longitude))
        out.createVariable("lat", "f8", ("lat",))[:] = latitude
        out.createVariable("lon", "f8", ("lon",))[:] = longitude
        for name, values in (("orography", orography_m), ("lsm", lsm), ("lat2d", lat2d), ("lon2d", lon2d)):
            value = out.createVariable(name, "f4", ("lat", "lon"))
            value[:] = values
            if name == "orography":
                value.units = "m"
        out.title = "SYNTHETIC orography and land-sea mask for pipeline testing"
        out.data_kind = "synthetic_pipeline_only"
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/step13_synthetic6h"))
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--years", type=int, nargs="+", default=[2014, 2015, 2016])
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 64:
        parser.error("--steps must be >=64 for K=60 and two history frames")
    if len(set(args.years)) != len(args.years):
        parser.error("--years must be distinct")
    kai = args.out / "kai6h"
    constant_path = args.out / "era5_data/constants/constants_5.625deg.nc"
    manifest_path = args.out / "manifest.json"
    outputs = [kai / f"{year}.nc" for year in args.years] + [constant_path, manifest_path]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        parser.error("output already exists; use --overwrite explicitly: " + ", ".join(existing))
    kai.mkdir(parents=True, exist_ok=True)
    constant_path.parent.mkdir(parents=True, exist_ok=True)
    latitude, longitude = coordinates()
    orography_m, lsm = static_fields(latitude, longitude)
    manifest = {
        "kind": "synthetic_pipeline_only", "data_kind": "synthetic_pipeline_only",
        "physics_skill_valid": False, "observed_era5": False,
        "warning": "Not ERA5 observations. Not primitive-equation balanced. For training/execution pipeline tests only; no climate skill claims.",
        "generator": "scripts/make_step13_synthetic6h.py", "seed": args.seed,
        "years": args.years, "steps_per_year_file": args.steps, "cadence_hours": 6,
        "shape_per_year": [args.steps, contract.N_CHANNELS, 32, 64],
        "grid": "global cell-centred 5.625 degree, lat ascending, longitude periodic",
        "independent_seed_per_year": {str(year): args.seed + index for index, year in enumerate(args.years)},
        "channels": contract.CHANNEL_NAMES, "files": {},
    }
    for index, year in enumerate(args.years):
        fields = make_fields(year, args.steps, args.seed + index, latitude, longitude, orography_m)
        path = kai / f"{year}.nc"
        write_year(path, year, fields, latitude, longitude)
        manifest["files"][str(path.relative_to(args.out))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        print(f"{year}: {fields.shape}, {path.stat().st_size / 2**20:.1f} MiB; SYNTHETIC pipeline data", flush=True)
    write_constants(constant_path, latitude, longitude, orography_m, lsm)
    manifest["files"][str(constant_path.relative_to(args.out))] = {
        "bytes": constant_path.stat().st_size, "sha256": sha256(constant_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {manifest_path}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
