"""
Copyright (c) 2025 MetaEarth-Lab
Licensed under the BSD 3-Clause License. Redistribution and use in source and binary forms, with or without modification, are permitted. See the LICENSE file in the project root for the full license text.
"""

import glob
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import netCDF4 as nc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from ruamel.yaml.scalarfloat import ScalarFloat


# --------------------------------------------------------------------------
# Metric functions
# --------------------------------------------------------------------------

def _lat_torch(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    """Convert latitude index j to degrees (90°N to 90°S, equidistant)."""
    return 90.0 - j * 180.0 / float(num_lat - 1)


def _latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    """Compute latitude-based area weighting factor for a given latitude index."""
    return num_lat * torch.cos(math.pi / 180.0 * _lat_torch(j, num_lat)) / s


def weighted_acc_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Latitude-weighted Anomaly Correlation Coefficient per channel.

    Inputs are expected to be *anomalies* (observation minus climatology).
    Input shape:  [N, C, H, W]
    Output shape: [N, C]

    """
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device, dtype=pred.dtype)
    s = torch.sum(torch.cos(math.pi / 180.0 * _lat_torch(lat_t, num_lat)))
    weight = _latitude_weighting_factor_torch(lat_t, num_lat, s).reshape(1, 1, -1, 1)

    xy = (pred * target * weight).reshape(pred.shape[0], pred.shape[1], -1).sum(dim=2)
    xx = (pred * pred * weight).reshape(pred.shape[0], pred.shape[1], -1).sum(dim=2)
    yy = (target * target * weight).reshape(target.shape[0], target.shape[1], -1).sum(dim=2)
    return xy / torch.sqrt(xx * yy + 1e-8)


# --------------------------------------------------------------------------
# Evaluation configuration dataclass
# --------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """
    Declarative, validated configuration for post-training evaluation.

    Hierarchy mirrors the YAML structure::

        eval:
          enabled: true
          data:
            test_dataset_path: ...
            climatology_path: ...
            output_dir: ...
          rollout:
            forecast_steps: 10
            start_timestep: 0
            n_initial_conditions: 5
            ic_stride: 1
          model:
            checkpoint_path: ...
    """

    enabled: bool
    test_dataset_path: str
    climatology_path: str
    output_dir: str
    # Rollout
    eval_forecast_steps: int          # number of *forecast* days (Day 1..N)
    eval_start_timestep: int
    eval_n_initial_conditions: int
    eval_ic_stride: int
    dt: int
    # Channels
    in_channels: List[int]
    out_channels: List[int]
    eval_variable_indices: List[int]
    # Normalization
    global_means_path: str
    global_stds_path: str
    # Optional
    eval_checkpoint_path: Optional[str] = None
    log_to_wandb: bool = False

    @classmethod
    def from_params(cls, config) -> "EvalConfig":
        """
        Build ``EvalConfig`` from a ``YParams`` namespace using flat-key
        attribute access.

        ``YParams`` stores YAML values as top-level attributes. Nested dicts
        (e.g. ``eval:``) are loaded as ``CommentedMap`` objects which do NOT
        support ``getattr``-style access, so we read everything from the
        flat namespace with safe defaults.
        """

        def _get(name: str, default=None):
            return getattr(config, name, default)

        raw_eval_variable_indices = _get("eval_variable_indices", [])
        if raw_eval_variable_indices is None:
            raw_eval_variable_indices = []

        return cls(
            enabled=bool(_get("enable_evaluation", False)),
            test_dataset_path=str(_get("test_dataset_path", "")),
            climatology_path=str(_get("climatology_path", "") or ""),
            output_dir=str(_get("experiment_dir", ".")),
            eval_forecast_steps=int(_get("eval_forecast_steps", 10)),
            eval_start_timestep=int(_get("eval_start_timestep", 0)),
            eval_n_initial_conditions=int(_get("n_initial_conditions", 1)),
            eval_ic_stride=int(_get("eval_ic_stride", 1)),
            dt=int(_get("dt", 1)),
            in_channels=list(_get("in_channels", [])),
            out_channels=list(_get("out_channels", [])),
            eval_variable_indices=[int(x) for x in raw_eval_variable_indices],
            global_means_path=str(_get("global_means_path", "")),
            global_stds_path=str(_get("global_stds_path", "")),
            eval_checkpoint_path=_get("eval_checkpoint_path", None),
            log_to_wandb=bool(_get("log_to_wandb", False)),
        )

    def validate(self) -> None:
        """Validate configuration at startup with clear error messages."""
        if not self.enabled:
            return
        if self.eval_forecast_steps <= 0:
            raise ValueError("Evaluation misconfiguration: forecast_steps must be > 0.")
        if self.eval_n_initial_conditions <= 0:
            raise ValueError("Evaluation misconfiguration: n_initial_conditions must be > 0.")
        if self.eval_ic_stride <= 0:
            raise ValueError("Evaluation misconfiguration: ic_stride must be > 0.")
        if not self.test_dataset_path:
            raise ValueError("Evaluation misconfiguration: test_dataset_path is required.")
        if not self.global_means_path or not self.global_stds_path:
            raise ValueError("Evaluation misconfiguration: global_means_path/global_stds_path are required.")
        if not os.path.exists(self.global_means_path) or not os.path.exists(self.global_stds_path):
            raise FileNotFoundError("Normalization statistics path does not exist.")


# --------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------

class PostTrainingEvaluator:
    """
    Autoregressive rollout evaluator
    """

    def __init__(self, model: nn.Module, cfg: EvalConfig, device: torch.device, logger):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.variable_names: Optional[List[str]] = None

        # Load normalization statistics and reduce to 1-D (C_all,)
        means = np.load(cfg.global_means_path).squeeze()  # e.g. (1,67,1,1) → (67,)
        stds = np.load(cfg.global_stds_path).squeeze()
        if means.ndim == 2:
            means = means[0]
            stds = stds[0]

        # Select output channels and move to device
        self.global_means = torch.from_numpy(means[cfg.out_channels]).float().to(device)
        self.global_stds = torch.from_numpy(stds[cfg.out_channels]).float().to(device)

        # Per-channel std for de-normalizing RMSE (1-D, shape [C])
        self.std_1d = self.global_stds.clone()

        # Load and normalize climatology
        self.climatology = self._load_and_normalize_climatology(cfg.climatology_path)

    def _load_variable_names(self, test_file: str) -> None:
        """Load channel names from the test NetCDF metadata if available."""
        self.variable_names = None
        try:
            xr_ds = xr.open_dataset(test_file)
            try:
                if "channel" in xr_ds.coords:
                    channel_names = [str(x) for x in xr_ds["channel"].values]
                    self.variable_names = []
                    for out_idx in self.cfg.out_channels:
                        if 0 <= out_idx < len(channel_names):
                            self.variable_names.append(channel_names[out_idx])
                        else:
                            self.variable_names.append(f"Var{out_idx}")

                    preview = ", ".join(self.variable_names[:10])
                    self.logger.info(
                        f"Loaded {len(self.variable_names)} variable names from NetCDF channel coordinate. "
                        f"First 10: {preview}"
                    )
                else:
                    self.logger.warning(
                        "Test NetCDF has no 'channel' coordinate. Falling back to Var{idx} names."
                    )
            finally:
                xr_ds.close()
        except Exception as e:
            self.logger.warning(f"Failed to load variable names from test NetCDF: {e}")

    def _variable_name(self, out_var_idx: int) -> str:
        """
        Return a readable variable name for an evaluator-local output index.

        Args:
            out_var_idx: Index in the evaluator-local output tensor, i.e. 0..len(out_channels)-1.
        """
        if self.variable_names is not None and 0 <= out_var_idx < len(self.variable_names):
            return str(self.variable_names[out_var_idx])
        original_channel_idx = self.cfg.out_channels[out_var_idx] if out_var_idx < len(self.cfg.out_channels) else out_var_idx
        return f"Var{original_channel_idx}"

    def _load_and_normalize_climatology(self, path: str) -> torch.Tensor:
        """
        Load daily climatology and normalize it with the same z-score
        transform applied to the test data
        """
        if not path:
            raise ValueError("Evaluation requires climatology_path for ACC computation.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Climatology file not found: {path}")

        ds = xr.open_dataset(path)
        try:
            var_name = "climatology" if "climatology" in ds.data_vars else list(ds.data_vars)[0]
            clim = ds[var_name].values  # [366, C_all, H, W]
        finally:
            ds.close()

        # Select the evaluated output channels, supporting both a climatology already
        # reduced to out_channels and a full climatology that carries every channel.
        n_clim_ch = clim.shape[1]
        n_out = len(self.cfg.out_channels)
        if n_clim_ch == n_out:
            # Already reduced to the evaluated channels at creation time.
            pass
        elif n_clim_ch > max(self.cfg.out_channels):
            clim = clim[:, self.cfg.out_channels, :, :]
        else:
            raise ValueError(
                f"Climatology has {n_clim_ch} channels, incompatible with "
                f"out_channels={self.cfg.out_channels}: expected either {n_out} "
                f"pre-subset channels or at least {max(self.cfg.out_channels) + 1}."
            )

        clim = torch.from_numpy(clim).float().to(self.device)

        # Normalize: (clim - mean) / std — no epsilon
        means = self.global_means.view(1, -1, 1, 1)
        stds = self.global_stds.view(1, -1, 1, 1)
        return (clim - means) / stds

    # ------------------------------------------------------------------
    # IC index construction
    # ------------------------------------------------------------------

    def _build_ic_indices(self, total_timesteps: int) -> List[int]:
        """
        Generate initial-condition (IC) start indices for the rollout.

        Produces up to ``eval_n_initial_conditions`` indices starting at
        ``eval_start_timestep`` and spaced ``eval_ic_stride`` apart, dropping any
        IC whose full rollout would run past the end of the file.

        Args:
            total_timesteps: Number of timesteps available in the test file.

        Returns:
            List[int]: IC start indices (length <= ``eval_n_initial_conditions``).

        Raises:
            ValueError: If no IC starting at ``eval_start_timestep`` leaves room
                for a full ``prediction_length`` rollout.
        """
        # prediction_length = forecast_steps + 1  (Day 0 .. Day N)
        prediction_length = self.cfg.eval_forecast_steps + 1
        # Largest IC start whose rollout still fits inside the file (inclusive).
        last_valid_ic = total_timesteps - prediction_length * self.cfg.dt
        start = self.cfg.eval_start_timestep
        stride = self.cfg.eval_ic_stride
        if last_valid_ic < start:
            raise ValueError(
                f"Not enough timesteps ({total_timesteps}) for a "
                f"{prediction_length}-step rollout with dt={self.cfg.dt} "
                f"starting at timestep {start}."
            )
        ics = [
            start + k * stride
            for k in range(self.cfg.eval_n_initial_conditions)
            if start + k * stride <= last_valid_ic
        ]
        if len(ics) < self.cfg.eval_n_initial_conditions:
            self.logger.warning(
                f"Requested {self.cfg.eval_n_initial_conditions} initial conditions but only "
                f"{len(ics)} fit within {total_timesteps} timesteps "
                f"(start={start}, stride={stride}, rollout={prediction_length}, dt={self.cfg.dt})."
            )
        return ics

    # ------------------------------------------------------------------
    # Single-IC rollout 
    # ------------------------------------------------------------------

    def _rollout_one_ic(
        self, data_slice: torch.Tensor, ic_val: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run autoregressive rollout for one initial condition.

        Args:
            data_slice: Normalized tensor [prediction_length, C, H, W].
            ic_val:     Index of the initial condition in the test file
                        (used only for DOY climatology lookup).

        Returns:
            Tuple of (pred_phys, true_phys, acc) where:
              pred_phys: np.ndarray [prediction_length, C, H, W] — denormalized
                         prediction fields in physical units.
              true_phys: np.ndarray [prediction_length, C, H, W] — denormalized
                         truth fields in physical units.
              acc:       np.ndarray [prediction_length, C].
        """
        pred_len = data_slice.shape[0]
        n_out = data_slice.shape[1]
        H, W = data_slice.shape[2], data_slice.shape[3]

        pred_phys_all = np.zeros((pred_len, n_out, H, W), dtype=np.float32)
        true_phys_all = np.zeros((pred_len, n_out, H, W), dtype=np.float32)
        acc = torch.zeros((pred_len, n_out), device=self.device, dtype=torch.float32)

        std_view = self.std_1d.view(1, -1, 1, 1)
        mean_view = self.global_means.view(1, -1, 1, 1)

        with torch.no_grad():
            for i in range(pred_len):
                if i == 0:
                    first = data_slice[0:1]            # [1, C, H, W]
                    future = data_slice[1:2]            # [1, C, H, W]
                    pred = first
                    tar = first
                    future_pred = self.model(first)     # first forecast step
                else:
                    if i < pred_len - 1:
                        future = data_slice[i + 1 : i + 2]
                    future_pred = self.model(future_pred)  # autoregressive

                # DOY index into the 366-entry daily climatology (index 0 = Jan 1).
                # Assumes the test file starts on Jan 1 with one timestep per day.
                doy_idx = (ic_val + i) % self.climatology.shape[0]
                daily_clim = self.climatology[doy_idx].unsqueeze(0)  # [1, C, H, W]

                # Denormalize to physical units: x_phys = x_norm * std + mean
                pred_phys_all[i] = (pred * std_view + mean_view)[0].cpu().numpy()
                true_phys_all[i] = (tar * std_view + mean_view)[0].cpu().numpy()

                # ACC: anomalies = field - climatology
                pred_anom = pred - daily_clim
                tar_anom = tar - daily_clim
                acc[i] = weighted_acc_torch_channels(pred_anom, tar_anom).mean(dim=0)

                # Advance pointers
                pred = future_pred
                tar = future

        return pred_phys_all, true_phys_all, acc.cpu().numpy()

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, np.ndarray]:
        """
        Run evaluation over all initial conditions and aggregate metrics.

        Returns:
            Dictionary with keys ``'rmse'`` and ``'acc'``, each an ndarray
            of shape ``[prediction_length, n_channels]`` averaged over ICs.
        """
        test_files = sorted(glob.glob(os.path.join(self.cfg.test_dataset_path, "*.nc")))
        if not test_files:
            raise FileNotFoundError(f"No .nc files found in {self.cfg.test_dataset_path}")

        test_file = test_files[0]
        self.logger.info(f"Running rollout evaluation on: {test_file}")
        self._load_variable_names(test_file)

        prediction_length = self.cfg.eval_forecast_steps + 1

        with nc.Dataset(test_file, "r") as ds:
            fields = ds["fields"]
            total_t = fields.shape[0]
            H = fields.shape[2]
            W = fields.shape[3]

            # Read latitude coordinates (degrees) — used by WB2 for cos(lat) weighting
            if "latitude" in ds.variables:
                lats = np.asarray(ds.variables["latitude"][:H], dtype=np.float64)
            elif "lat" in ds.variables:
                lats = np.asarray(ds.variables["lat"][:H], dtype=np.float64)
            else:
                lats = np.linspace(-90.0, 90.0, H)

            # Read longitude coordinates (degrees)
            if "longitude" in ds.variables:
                lons = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
            elif "lon" in ds.variables:
                lons = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            else:
                lons = np.linspace(0.0, 360.0, W, endpoint=False)

            ics = self._build_ic_indices(total_t)
            self.logger.info(f"Total timesteps: {total_t}; Evaluating {len(ics)} ICs.")
            self.logger.info(f"Detailed variable indices from config: {self.cfg.eval_variable_indices}")

            # Normalization stats as numpy for vectorized preprocessing
            means = self.global_means.cpu().numpy().reshape(1, -1, 1, 1)
            stds = self.global_stds.cpu().numpy().reshape(1, -1, 1, 1)

            n_out = len(self.cfg.out_channels)
            all_pred: List[np.ndarray] = []
            all_true: List[np.ndarray] = []
            all_acc: List[np.ndarray] = []
            for count, ic in enumerate(ics):
                # Extract data slice [prediction_length, C, H, W]
                end_idx = ic + prediction_length * self.cfg.dt
                raw = fields[
                    ic:end_idx:self.cfg.dt,
                    self.cfg.in_channels,
                    :,
                    :,
                ]

                # Normalize with a plain z-score (no epsilon), consistent with the
                # climatology normalization in _load_and_normalize_climatology.
                data_slice = (raw - means) / stds
                data_tensor = torch.as_tensor(
                    np.array(data_slice), device=self.device, dtype=torch.float32
                )

                pred_i, true_i, acc_i = self._rollout_one_ic(data_tensor, ic)
                all_pred.append(pred_i)
                all_true.append(true_i)
                all_acc.append(acc_i)

                if (count + 1) % 10 == 0 or (count + 1) == len(ics):
                    self.logger.info(f"Completed IC {count + 1}/{len(ics)} (ic={ic})")

        # Stack per-IC fields into (N_ic, T, C, H, W) for WB2
        fcst = np.stack(all_pred, axis=0)
        true = np.stack(all_true, axis=0)

        coords = {
            "init_time": np.array(ics, dtype="int64"),          # IC indices are fine; values are unused by RMSE
            "prediction_timedelta": np.arange(prediction_length),
            "latitude": lats,                                    # MUST be degrees
            "longitude": lons,
        }
        forecast_vars = {
            self._variable_name(c): (
                ("init_time", "prediction_timedelta", "latitude", "longitude"),
                fcst[:, :, c, :, :],
            )
            for c in range(n_out)
        }
        truth_vars = {
            self._variable_name(c): (
                ("init_time", "prediction_timedelta", "latitude", "longitude"),
                true[:, :, c, :, :],
            )
            for c in range(n_out)
        }
        forecast_ds = xr.Dataset(data_vars=forecast_vars, coords=coords)
        truth_ds = xr.Dataset(data_vars=truth_vars, coords=coords)

        # MSE computed by weatherbench2.metrics.MSE — area-weighted spatial mean
        # over (latitude, longitude) and arithmetic mean over init_time.
        # RMSE = sqrt(MSE), applied after both means, matching the WB2 paper / WMO definition.
        from weatherbench2.metrics import MSE as WB2_MSE
        mse_ds = WB2_MSE().compute(forecast_ds, truth_ds)

        # Equivalent reference computation (for verification):
        #   weights = np.cos(np.deg2rad(lats)); weights /= weights.mean()
        #   mse_per_ic = ((fcst - true) ** 2 * weights[None, :, None]).mean(axis=(-2, -1))
        #   mse        = mse_per_ic.mean(axis=0)        # mean over init_time
        #   rmse       = np.sqrt(mse)                   # sqrt LAST, after both means

        avg_rmse = np.stack(
            [np.sqrt(mse_ds[self._variable_name(c)].values) for c in range(n_out)],
            axis=-1,
        )  # shape (T, C)

        avg_acc = np.mean(np.asarray(all_acc), axis=0)
        metrics = {"rmse": avg_rmse, "acc": avg_acc}

        self._log_summary(metrics)
        self._save_csv(metrics)
        self._log_wandb(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _log_summary(self, metrics: Dict[str, np.ndarray]) -> None:
        self.logger.info("=" * 80)
        self.logger.info("EVALUATION SUMMARY (AVERAGED OVER ICS)")
        self.logger.info("=" * 80)
        for t in range(metrics["rmse"].shape[0]):
            self.logger.info(
                f"Lead {t:02d} | RMSE={metrics['rmse'][t].mean():.4f} "
                f"| ACC={metrics['acc'][t].mean():.4f}"
            )

        if self.cfg.eval_variable_indices:
            self.logger.info("\n--- Detailed Metrics for Selected Variables ---")
            n_vars = metrics["rmse"].shape[1]
            for requested_idx in self.cfg.eval_variable_indices:
                if requested_idx < 0 or requested_idx >= n_vars:
                    self.logger.warning(
                        f"Skipping eval_variable_indices entry {requested_idx}: valid range is 0..{n_vars - 1}."
                    )
                    continue

                var_name = self._variable_name(requested_idx)
                original_channel_idx = self.cfg.out_channels[requested_idx]
                self.logger.info(
                    f"\nVariable {requested_idx} (out_channel={original_channel_idx}, name={var_name}):"
                )
                for t in range(metrics["rmse"].shape[0]):
                    self.logger.info(
                        f"  Lead {t:02d} | RMSE={metrics['rmse'][t, requested_idx]:.4f} "
                        f"| ACC={metrics['acc'][t, requested_idx]:.4f}"
                    )

    def _save_csv(self, metrics: Dict[str, np.ndarray]) -> None:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        lead, n_vars = metrics["rmse"].shape
        for t in range(lead):
            for v in range(n_vars):
                rows.append(
                    {
                        "lead_time": t,
                        "variable_idx": v,
                        "original_channel_idx": int(self.cfg.out_channels[v]) if v < len(self.cfg.out_channels) else v,
                        "variable_name": self._variable_name(v),
                        "rmse": float(metrics["rmse"][t, v]),
                        "acc": float(metrics["acc"][t, v]),
                    }
                )
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "evaluation_metrics.csv", index=False)

        rmse_columns = [self._variable_name(v) for v in range(n_vars)]
        acc_columns = [self._variable_name(v) for v in range(n_vars)]

        rmse_df = pd.DataFrame(metrics["rmse"], columns=rmse_columns)
        rmse_df.insert(0, "lead_time", np.arange(lead))
        rmse_df.to_csv(out_dir / "rollout_rmse.csv", index=False)

        acc_df = pd.DataFrame(metrics["acc"], columns=acc_columns)
        acc_df.insert(0, "lead_time", np.arange(lead))
        acc_df.to_csv(out_dir / "rollout_acc.csv", index=False)

        self.logger.info(f"Saved detailed metrics CSV to: {out_dir / 'evaluation_metrics.csv'}")
        self.logger.info(f"Saved RMSE rollout CSV to: {out_dir / 'rollout_rmse.csv'}")
        self.logger.info(f"Saved ACC rollout CSV to: {out_dir / 'rollout_acc.csv'}")

    def _log_wandb(self, metrics: Dict[str, np.ndarray]) -> None:
        """Log evaluation figures and RMSE/ACC tables to Weights & Biases.

        No-op unless ``log_to_wandb`` is enabled. Any failure is caught and logged
        so that a plotting / W&B error never aborts evaluation.
        """
        if not self.cfg.log_to_wandb:
            return
        try:
            from .eval_logging import log_eval_to_wandb

            log_eval_to_wandb(
                metrics=metrics,
                variable_names=[self._variable_name(v) for v in range(metrics["rmse"].shape[1])],
                out_channels=self.cfg.out_channels,
                eval_variable_indices=self.cfg.eval_variable_indices,
                output_dir=self.cfg.output_dir,
                logger=self.logger,
            )
        except Exception as e:  # pragma: no cover - never let W&B logging abort eval
            self.logger.warning(f"Failed to log evaluation to W&B: {e}")


# --------------------------------------------------------------------------
# Entry point — callable from train.py after training completes
# --------------------------------------------------------------------------

def run_post_training_evaluation(model: nn.Module, config, logger, checkpoint_path: Optional[str] = None):
    """
    Convenience function to run evaluation from the training script.

    Handles checkpoint loading, device placement, and ``EvalConfig`` extraction
    from the ``YParams`` namespace.
    """
    cfg = EvalConfig.from_params(config)
    cfg.validate()
    if not cfg.enabled:
        logger.info("Evaluation disabled in config. Skipping.")
        return None

    if checkpoint_path:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        torch.serialization.add_safe_globals([ScalarFloat])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = checkpoint.get("model_state", checkpoint)
        try:
            model.load_state_dict(model_state)
        except RuntimeError:
            stripped = {k.replace("module.", ""): v for k, v in model_state.items()}
            model.load_state_dict(stripped)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    evaluator = PostTrainingEvaluator(model=model, cfg=cfg, device=device, logger=logger)
    results = evaluator.evaluate()
    logger.info("Evaluation completed successfully.")
    return results
