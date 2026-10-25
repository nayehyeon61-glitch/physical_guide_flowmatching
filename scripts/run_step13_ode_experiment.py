"""Reproducible CPU-budget comparison of original Step13 models and ODE correction.

Keeps PINNProjector/SeqFM/EnsembleToDeterministic source unchanged. Trains the
first two models from scratch on explicit training years, then compares paired
FM, post-PINN, ODE and zero-transport ODE ensembles on disjoint held-out years.
K=60 means 15 days at six-hourly resolution. CPU-small architecture and training
budgets are recorded; this runner does not reproduce the paper's mature models.

``--data-kind`` is required so synthetic pipeline checks cannot be mistaken for
observed ERA5 skill. All large checkpoints/arrays are written under runs/.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import netCDF4 as nc
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgcfm import contract
from pgcfm.cfm import TISR_DYN
from pgcfm.core import static_features
from pgcfm.coupling import PhysicsConfig, PhysicsGuide, estimate_scales
from pgcfm.seqdct import (
    PINNConfig, PINNProjector, SeqDCT, SeqFM, SeqFMConfig,
    hard_constraints, residual_rms, sequence_residual_loss,
)


DYN = contract.DYNAMIC_CHANNELS
OROG = contract.STATIC_CHANNELS
CHANNELS = [contract.CHANNEL_NAMES[index] for index in DYN]


def daily_leads(K: int) -> np.ndarray:
    """Daily leads generated from K, not the original six-element 144h tuple."""
    if K < 1:
        raise ValueError("K must be positive")
    return np.arange(4, K + 1, 4, dtype=np.int64) * 6


def cyclic_pairs(iteration: int, count: int, K: int) -> torch.Tensor:
    """Cover every sequence transition instead of starving unselected PINN heads."""
    if iteration < 0 or count < 1 or count > K:
        raise ValueError("need iteration >= 0 and 1 <= pair count <= K")
    return (iteration * count + torch.arange(count)) % K


def validate_time_axis(values, units, calendar="standard"):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("time must be a finite 1D coordinate with at least two values")
    if not units or "since" not in units:
        raise ValueError("time must declare CF units, such as hours since 2014-01-01")
    dates = nc.num2date(values, units=units, calendar=calendar, only_use_cftime_datetimes=True)
    seconds = np.asarray(nc.date2num(dates, units="seconds since 1970-01-01", calendar=calendar), dtype=np.float64)
    if not np.allclose(np.diff(seconds), 21600.0, rtol=0, atol=1e-3):
        raise ValueError("Step13 requires consecutive six-hourly times; K=60 must mean exactly 15 days")
    return {"cadence_seconds": 21600.0, "first_valid_time": str(dates[0]), "last_valid_time": str(dates[-1])}


def validate_provenance(data_kind, manifest):
    manifest = manifest or {}
    label = " ".join(str(manifest.get(key, "")) for key in ("kind", "data_kind")).lower()
    if data_kind == "era5" and ("synthetic" in label or manifest.get("observed_era5") is False):
        raise ValueError("synthetic provenance cannot be labeled as observed ERA5")
    if data_kind == "synthetic" and manifest.get("observed_era5") is True:
        raise ValueError("data-kind synthetic conflicts with observed ERA5 provenance")
    return manifest


def validate_static_coordinates(latitude, longitude, expected_latitude, expected_longitude):
    for name, actual, expected in (
        ("latitude", latitude, expected_latitude), ("longitude", longitude, expected_longitude),
    ):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-5):
            raise ValueError(f"constants {name} must match KAI ordering; reindex data and coordinates together")


def training_statistics(arrays: dict[int, np.ndarray], years: list[int]):
    """Float64 streaming moments, training split only; exact KAI denominator."""
    total, square, count = np.zeros(67), np.zeros(67), 0
    for year in years:
        values = arrays[year].astype(np.float64)
        total += values.sum(axis=(0, 2, 3))
        square += np.square(values).sum(axis=(0, 2, 3))
        count += values.shape[0] * values.shape[2] * values.shape[3]
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean**2, 0))
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_fields(fields: np.ndarray, mean, std) -> torch.Tensor:
    return torch.from_numpy(
        ((fields - mean[None, :, None, None]) /
         (std + contract.KAI_STD_EPS)[None, :, None, None]).astype(np.float32)
    )


def sample_batch(data, years, K, rng, count):
    histories, increments, statics, hours = [], [], [], []
    for year in rng.choice(years, count):
        values = data[int(year)]
        index = int(rng.integers(2, len(values) - K))
        hist = values[index - 2 : index + 1, DYN]
        histories.append(hist)
        increments.append(values[index + 1 : index + K + 1, DYN] - hist[-1])
        statics.append(values[index, OROG])
        hours.append(index * 6.0)
    return (torch.stack(histories), torch.stack(increments),
            torch.stack(statics), torch.tensor(hours))


def score_members(ensemble, truth, climatology, latitude, physical_scale):
    """Area-weighted scores [K,C], empirical CRPS without pairwise large tensor.

    Input is normalized state [B,M,K,C,H,W], target [B,K,C,H,W]. Physical
    RMSE/CRPS/spread use the exact std+EPS scaling. ACC uses training-only local
    climatology. Population ensemble spread is explicit, including M=1.
    """
    mean = ensemble.mean(dim=1)
    weights = torch.cos(torch.deg2rad(torch.as_tensor(latitude, dtype=ensemble.dtype)))
    weights = (weights / weights.sum()).reshape(1, 1, 1, -1, 1)
    reduce = lambda value: (value * weights).sum(-2).mean((0, -1))
    mse = reduce((mean - truth).square())
    spread = reduce(ensemble.var(dim=1, unbiased=False)).clamp_min(0).sqrt()
    crps_grid = (ensemble - truth[:, None]).abs().mean(1)
    pair = torch.zeros_like(crps_grid)
    for left in range(ensemble.shape[1]):
        for right in range(left):
            pair += (ensemble[:, left] - ensemble[:, right]).abs()
    crps = reduce(crps_grid - pair / ensemble.shape[1]**2)
    p, o = mean - climatology[None, None], truth - climatology[None, None]
    den = (reduce(p.square()) * reduce(o.square())).sqrt()
    acc = torch.where(den > 1e-12, reduce(p * o) / den.clamp_min(1e-12), torch.full_like(den, float("nan")))
    scale = torch.as_tensor(physical_scale, dtype=ensemble.dtype).reshape(1, -1)
    return {
        "rmse": (mse.sqrt() * scale).numpy(),
        "crps": (crps * scale).numpy(),
        "spread": (spread * scale).numpy(),
        "acc": acc.numpy(),
        "normalized_rmse": mse.sqrt().numpy(),
        "normalized_crps": crps.numpy(),
    }


def _dump(path: Path, value):
    # JSON null, rather than non-standard NaN, for undefined ACC or missing data.
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, np.integer):
            return int(item)
        return item
    with path.open("w") as handle:
        json.dump(clean(value), handle, indent=2, allow_nan=False)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kai", required=True)
    parser.add_argument("--era5", required=True, help="folder with constants/constants_5.625deg.nc")
    parser.add_argument("--data-kind", choices=("era5", "synthetic"), required=True)
    parser.add_argument("--provenance", help="dataset manifest JSON; copied into the report")
    parser.add_argument("--out", default="runs/step13/ode_experiment")
    parser.add_argument("--train-years", type=int, nargs="+", default=[2014])
    parser.add_argument("--val-year", type=int, default=2015)
    parser.add_argument("--test-year", type=int, default=2016)
    parser.add_argument("--K", type=int, default=60)
    parser.add_argument("--pinn-iters", type=int, default=20)
    parser.add_argument("--pinn-pairs", type=int, default=3, help="cyclic consecutive-pair losses per iteration; cover all K")
    parser.add_argument("--fm-iters", type=int, default=40)
    parser.add_argument("--pinn-hidden", type=int, default=8)
    parser.add_argument("--fm-channels", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--basis-batches", type=int, default=4)
    parser.add_argument("--lambda-p", type=float, default=0.05)
    parser.add_argument("--members", type=int, default=2)
    parser.add_argument("--n-eval", type=int, default=1)
    parser.add_argument("--heun", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ode-step-seconds", type=float, default=300.0)
    parser.add_argument("--nudging-timescale-seconds", type=float, default=3600.0)
    parser.add_argument("--wind-cap-m-s", type=float, default=80.0)
    parser.add_argument("--skip-ode", action="store_true")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--eval-tag", default="comparison")
    args = parser.parse_args(argv)
    if args.val_year in args.train_years or args.test_year in args.train_years or args.val_year == args.test_year:
        parser.error("training, validation and test years must be disjoint")
    if min(args.K, args.pinn_iters, args.pinn_pairs, args.fm_iters, args.members, args.n_eval, args.batch, args.basis_batches) < 1:
        parser.error("horizon, iteration counts, sampling and basis budgets must be positive")
    if args.pinn_pairs > args.K:
        parser.error("--pinn-pairs must not exceed K")
    if args.fm_channels % 32:
        parser.error("original U-Net GroupNorm requires --fm-channels divisible by 32")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    provenance = json.loads(Path(args.provenance).read_text()) if args.provenance else None
    validate_provenance(args.data_kind, provenance)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"{args.eval_tag}.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment report {report_path}; use another --eval-tag")
    log = lambda message: print(message, flush=True)
    run_started = time.perf_counter()
    arrays, sources, latitude, longitude = {}, {}, None, None
    for year in args.train_years + [args.val_year, args.test_year]:
        path = Path(args.kai) / f"{year}.nc"
        with nc.Dataset(path) as dataset:
            values = np.asarray(dataset["fields"][:], dtype=np.float32)
            lat = np.asarray(dataset["latitude"][:], dtype=np.float64)
            lon = np.asarray(dataset["longitude"][:], dtype=np.float64)
            attrs = {key: str(dataset.getncattr(key)) for key in dataset.ncattrs()}
            if "time" not in dataset.variables or len(dataset["time"]) != len(values):
                raise ValueError(f"{path}: time coordinate must match fields")
            time_metadata = validate_time_axis(
                dataset["time"][:], getattr(dataset["time"], "units", None),
                getattr(dataset["time"], "calendar", "standard"),
            )
        if values.shape[1:] != (67, 32, 64) or len(values) <= args.K + 2 or not np.isfinite(values).all():
            raise ValueError(f"{path}: need finite [T,67,32,64] and T > K+2")
        if latitude is not None and (not np.array_equal(lat, latitude) or not np.array_equal(lon, longitude)):
            raise ValueError("all splits must share identical coordinates")
        arrays[year], latitude, longitude = values, lat, lon
        sources[year] = {"path": str(path.resolve()), "sha256": _sha256(path), "snapshots": len(values), "attributes": attrs, "time_axis": time_metadata}
    from physical_guide_flowmatching import GlobalLatLonGrid
    GlobalLatLonGrid().validate_coordinates(latitude, longitude)
    contract.register_latitudes(latitude)
    mean, std = training_statistics(arrays, args.train_years)
    stats = out / "stats"
    stats.mkdir(exist_ok=True)
    for filename, value in (("global_mean.npy", mean), ("global_std.npy", std)):
        destination = stats / filename
        if destination.exists() and not np.array_equal(np.load(destination), value):
            raise ValueError("existing checkpoint normalization differs from current training data")
        np.save(destination, value)
    data = {year: normalize_fields(values, mean, std) for year, values in arrays.items()}
    del arrays
    with nc.Dataset(Path(args.era5) / "constants" / "constants_5.625deg.nc") as constants:
        if "lat" not in constants.variables or "lon" not in constants.variables:
            raise ValueError("constants must include explicit lat/lon coordinates")
        validate_static_coordinates(constants["lat"][:], constants["lon"][:], latitude, longitude)
        oro = np.asarray(constants["orography"][:], dtype=np.float32)
        lsm = np.asarray(constants["lsm"][:], dtype=np.float32)
    static = static_features(latitude, longitude, oro, lsm)
    if not torch.isfinite(static).all():
        raise ValueError("static features must be finite; constant orography cannot be standardized")
    weights = torch.cos(torch.deg2rad(torch.tensor(latitude, dtype=torch.float32)))
    weights = (weights / weights.mean()).reshape(1, 1, 1, 32, 1)
    t6 = torch.stack([
        (data[year][1:, DYN] - data[year][:-1, DYN]).square().mean((0, 2, 3))
        for year in args.train_years
    ]).mean(0).sqrt().clamp_min(1e-6).reshape(1, 1, -1, 1, 1)
    guide = PhysicsGuide(32, 64, mean, std, PhysicsConfig(dt_seconds=21600.0))
    hist, delta, orog, _ = sample_batch(data, args.train_years, 1, rng, 4)
    guide.scales = estimate_scales(guide, [{
        "x0": hist[:, -1], "targets": (hist[:, -1] + delta[:, 0])[:, None],
        "static": orog, "doy": torch.zeros(len(hist), 1),
    }])
    for parameter in guide.parameters():
        parameter.requires_grad_(False)
    perturb = SeqDCT(args.K, 66, 32, 64, spatial=True)
    perturb.fit(sample_batch(data, args.train_years, args.K, rng, 2)[1] for _ in range(args.basis_batches))
    p_cfg = PINNConfig(K=args.K, hidden=args.pinn_hidden, blocks=(1,))
    p_model = PINNProjector(p_cfg, static)
    fm_cfg = SeqFMConfig(K=args.K, model_channels=args.fm_channels, channel_mult=(1,))
    fm_model = SeqFM(fm_cfg, static)
    signature_fields = {
        "train_source_sha256": {year: sources[year]["sha256"] for year in args.train_years},
        "data_kind": args.data_kind, "train_years": args.train_years, "seed": args.seed,
        "K": args.K, "batch": args.batch, "basis_batches": args.basis_batches,
        "pinn_cfg": dataclasses.asdict(p_cfg), "fm_cfg": dataclasses.asdict(fm_cfg),
        "pinn_iters": args.pinn_iters, "fm_iters": args.fm_iters, "lambda_p": args.lambda_p,
        "pinn_pairs": args.pinn_pairs,
    }
    training_signature = hashlib.sha256(json.dumps(signature_fields, sort_keys=True).encode()).hexdigest()
    progress = {"args": vars(args), "data_kind": args.data_kind, "stages": {}}

    # Stage 1: original project objective, including original primitive residual.
    p_path = out / "pinn.pt"
    p_meta = out / "pinn.json"
    if args.reuse_checkpoints and p_path.exists():
        meta = json.loads(p_meta.read_text())
        if meta["cfg"] != json.loads(json.dumps(dataclasses.asdict(p_cfg))) or meta.get("training_signature") != training_signature:
            raise ValueError("PINN checkpoint data/normalization/training configuration differs from current run")
        p_model.load_state_dict(torch.load(p_path, weights_only=True))
        p_history, p_seconds = meta["loss"], meta["seconds"]
        covered_pairs = set(meta["covered_pairs"])
    else:
        if p_path.exists():
            raise FileExistsError(f"{p_path} exists; explicitly use --reuse-checkpoints")
        optimizer = torch.optim.Adam(p_model.parameters(), lr=5e-4)
        p_history, started, covered_pairs = [], time.perf_counter(), set()
        for iteration in range(args.pinn_iters):
            hist, truth, orog, hour = sample_batch(data, args.train_years, args.K, rng, args.batch)
            x0 = hist[:, -1]
            corrupted = truth + torch.rand(args.batch).reshape(-1, 1, 1, 1, 1) * perturb.noise(truth.shape) * p_model.keep
            predicted = p_model(x0, corrupted, hour)
            movement = (((predicted - corrupted) / t6).square() * weights).mean()
            pairs = cyclic_pairs(iteration, args.pinn_pairs, args.K)
            covered_pairs.update(pairs.tolist())
            physical, _ = sequence_residual_loss(guide, x0, predicted, truth, orog, pairs)
            loss = movement + args.lambda_p * physical
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p_model.parameters(), 1.0)
            optimizer.step()
            p_history.append({"loss": float(loss.detach()), "movement": float(movement.detach()), "physics": float(physical.detach())})
            log(f"PINN {iteration+1}/{args.pinn_iters} loss={p_history[-1]['loss']:.5g} elapsed={time.perf_counter()-started:.1f}s")
        p_seconds = time.perf_counter() - started
        torch.save(p_model.state_dict(), p_path)
        _dump(p_meta, {"cfg": dataclasses.asdict(p_cfg), "loss": p_history, "seconds": p_seconds, "training_signature": training_signature,
                       "pair_sampling": "deterministic_cyclic", "pairs_per_iteration": args.pinn_pairs,
                       "covered_pairs": sorted(covered_pairs)})
    p_model.eval()
    for parameter in p_model.parameters():
        parameter.requires_grad_(False)
    progress["stages"]["pinn"] = {"seconds": p_seconds, "iterations": len(p_history),
                                     "covered_pairs": sorted(covered_pairs), "coverage_fraction": len(covered_pairs)/args.K}
    _dump(out / "status.json", progress)

    # Stage 2: original independent CFM objective, never backpropagated via PINN.
    fm_path, fm_meta = out / "fm.pt", out / "fm.json"
    if args.reuse_checkpoints and fm_path.exists():
        meta = json.loads(fm_meta.read_text())
        if meta["cfg"] != json.loads(json.dumps(dataclasses.asdict(fm_cfg))) or meta.get("training_signature") != training_signature:
            raise ValueError("FM checkpoint data/normalization/training configuration differs from current run")
        fm_model.load_state_dict(torch.load(fm_path, weights_only=True))
        fm_history, fm_seconds = meta["loss"], meta["seconds"]
    else:
        if fm_path.exists():
            raise FileExistsError(f"{fm_path} exists; explicitly use --reuse-checkpoints")
        fm_model.fit_basis(sample_batch(data, args.train_years, args.K, rng, 2)[1] for _ in range(max(4, args.basis_batches)))
        optimizer = torch.optim.Adam(fm_model.unet.parameters(), lr=2e-4)
        fm_history, started = [], time.perf_counter()
        for iteration in range(args.fm_iters):
            hist, truth, _, hour = sample_batch(data, args.train_years, args.K, rng, args.batch)
            loss = fm_model.loss(hist, truth, hour)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm_model.unet.parameters(), 1.0)
            optimizer.step()
            fm_history.append(float(loss.detach()))
            log(f"FM {iteration+1}/{args.fm_iters} loss={fm_history[-1]:.5g} elapsed={time.perf_counter()-started:.1f}s")
        fm_seconds = time.perf_counter() - started
        torch.save(fm_model.state_dict(), fm_path)
        _dump(fm_meta, {"cfg": dataclasses.asdict(fm_cfg), "loss": fm_history, "seconds": fm_seconds, "training_signature": training_signature})
    fm_model.eval()
    progress["stages"]["fm"] = {"seconds": fm_seconds, "iterations": len(fm_history)}
    _dump(out / "status.json", progress)

    # Held-out evaluation. No hyperparameter selection is performed on test.
    test = data[args.test_year]
    candidates = np.arange(2, len(test) - args.K)
    indices = candidates[np.linspace(0, len(candidates)-1, min(args.n_eval, len(candidates))).astype(int)]
    histories = torch.stack([test[index-2:index+1, DYN] for index in indices])
    observed = histories[:, -1]
    truth = torch.stack([test[index+1:index+args.K+1, DYN] for index in indices])
    orog = torch.stack([test[index, OROG] for index in indices])
    hours = torch.tensor(indices * 6.0, dtype=torch.float32)
    B, M = len(indices), args.members
    expanded = histories.repeat_interleave(M, 0)
    expanded_hours = hours.repeat_interleave(M)
    expanded_x0 = expanded[:, -1]
    costs, ode_diagnostics = {}, {}
    with torch.no_grad():
        generator = torch.Generator().manual_seed(args.seed + 1001)
        noise = fm_model.noise(B*M, generator=generator)
        started = time.perf_counter()
        raw = fm_model.sample(expanded, expanded_hours, steps=args.heun, eps=noise)
        costs["raw"] = time.perf_counter() - started
        started = time.perf_counter()
        pinn = p_model(expanded_x0, raw, expanded_hours)
        costs["pinn"] = time.perf_counter() - started
        increments = {"raw": raw.reshape(B, M, args.K, 66, 32, 64),
                      "pinn": pinn.reshape(B, M, args.K, 66, 32, 64)}
        if not args.skip_ode:
            from pgcfm.ode_corrector import correct_sequence_ode
            for name, strength in (("ode", 1.0), ("ode_no_transport", 0.0)):
                log(f"starting {name}: {B} initial conditions, {M} members, {args.K*6}h horizon")
                started = time.perf_counter()
                corrected, diagnostic = correct_sequence_ode(
                    observed, increments["raw"], mean, std,
                    step_seconds=args.ode_step_seconds,
                    nudging_timescale_seconds=args.nudging_timescale_seconds,
                    transport_strength=strength, wind_cap_m_s=args.wind_cap_m_s,
                )
                if corrected.shape != increments["raw"].shape or not torch.isfinite(corrected).all():
                    raise RuntimeError("ODE correction must return finite full-66-channel trajectories")
                if not torch.equal(corrected[:, :, :, TISR_DYN], increments["raw"][:, :, :, TISR_DYN]):
                    raise RuntimeError("ODE must preserve original FM astronomical forcing channel")
                increments[name], costs[name], ode_diagnostics[name] = corrected, time.perf_counter()-started, diagnostic
                log(f"finished {name}: {costs[name]:.1f}s")
    violations = {}
    for name, inc in increments.items():
        fixed, violations[name] = hard_constraints(expanded_x0, inc.flatten(0, 1), mean, std)
        increments[name] = fixed.reshape_as(inc)
    states = {name: observed[:, None, None] + value for name, value in increments.items()}
    states["persistence"] = observed[:, None, None].expand(B, 1, args.K, 66, 32, 64)
    climatology = torch.cat([data[year][:, DYN] for year in args.train_years]).mean(0)
    scale = std[DYN] + contract.KAI_STD_EPS
    scores, physics = {}, {}
    pairs = torch.tensor(sorted(set([0, args.K // 2, args.K - 1])))
    physics["truth"] = residual_rms(guide, observed, truth-observed[:, None], truth-observed[:, None], orog, pairs)
    for name, ensemble in states.items():
        log(f"scoring {name}")
        scores[name] = score_members(ensemble, truth, climatology, latitude, scale)
        if name != "persistence":
            physics[name] = residual_rms(
                guide, expanded_x0, increments[name].flatten(0, 1),
                (truth-observed[:, None]).repeat_interleave(M, 0), orog.repeat_interleave(M, 0), pairs,
            )
    leads = daily_leads(args.K)
    rows = []
    for arm, metrics in scores.items():
        for lead in leads:
            index = int(lead // 6 - 1)
            for channel, name in enumerate(CHANNELS):
                rows.append({"arm": arm, "lead_hours": int(lead), "channel": name,
                             **{key: float(value[index, channel]) for key, value in metrics.items()}})
    if rows:
        with (out / f"{args.eval_tag}_scores.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "data_kind": args.data_kind,
        "physics_skill_valid": False,
        "observed_era5": args.data_kind == "era5",
        "interpretation": "synthetic pipeline/stress test only" if args.data_kind == "synthetic" else "held-out ERA5 exploratory experiment",
        "scope": "exploratory reduced-architecture CPU experiment; not a mature ERA5 benchmark",
        "args": vars(args), "data_sources": sources, "dataset_manifest": provenance,
        "stats_fit_years": args.train_years, "validation_used_for_selection": False,
        "training_signature": training_signature,
        "pinn_cfg": dataclasses.asdict(p_cfg), "fm_cfg": dataclasses.asdict(fm_cfg),
        "training": {"pinn": progress["stages"]["pinn"], "fm": progress["stages"]["fm"]},
        "test_initial_indices": indices, "daily_leads_hours": leads,
        "channels": CHANNELS, "members": M, "physics_scale_source": "training pairs only",
        "physics_scales": vars(guide.scales), "physics_residual_rms": physics,
        "negative_q_tcwv_fraction_before_common_clamp": violations,
        "cost_seconds": costs, "ode_diagnostics": ode_diagnostics,
        "cost_semantics": "raw is FM generation; PINN/ODE/null are incremental postprocessing, excluding shared generation",
        "normalized_rmse_all_channels_by_6h_lead": {name: np.sqrt(np.mean(value["normalized_rmse"]**2, axis=1)) for name, value in scores.items()},
        "normalized_crps_all_channels_by_6h_lead": {name: value["normalized_crps"].mean(1) for name, value in scores.items()},
        "spread_ratio_to_raw": {name: float((states[name].std(1, unbiased=False).square()*weights).mean().sqrt() / (states["raw"].std(1, unbiased=False).square()*weights).mean().sqrt().clamp_min(1e-12)) for name in increments},
        "wall_seconds": time.perf_counter()-run_started,
        "operator_modified": False,
    }
    _dump(report_path, result)
    # Full 66-channel normalized trajectories and exact statistics are retained,
    # unlike the original five-variable, daily-only NPZ export.
    np.savez(
        out / f"{args.eval_tag}_ensembles.npz", X0=observed.numpy(), truth=truth.numpy(),
        **{f"increments_{name}": value.numpy() for name, value in increments.items()},
        mean=mean, std=std, dynamic_indices=np.array(DYN), initial_indices=indices,
        leads_hours=np.arange(1, args.K+1)*6, latitude=latitude, longitude=longitude,
    )
    log(f"saved {report_path}; total {time.perf_counter()-run_started:.1f}s; DATA_KIND={args.data_kind}")
    return result


if __name__ == "__main__":
    main()
