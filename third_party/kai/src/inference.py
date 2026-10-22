# src/inference.py
"""
Copyright (c) 2025 MetaEarth-Lab
Licensed under the BSD 3-Clause License. Redistribution and use in source and binary forms,
with or without modification, are permitted. See the LICENSE file in the project root for
the full license text.
"""
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import netCDF4 as nc
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from ruamel.yaml import YAML
from ruamel.yaml.scalarfloat import ScalarFloat

from .models import KAI

logger = logging.getLogger(__name__)


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class InferenceConfig:
    """
    Declarative configuration for autoregressive inference.

    YAML layout::

        inference:
          io:
            input_nc_path: /path/to/test/2018.nc
            output_dir: /path/to/output/
            field_variable: fields
          model:
            checkpoint_path: /path/to/best_ckpt.tar
            global_means_path: /path/to/global_mean.npy
            global_stds_path: /path/to/global_std.npy
          rollout:
            forecast_steps: 10
            n_initial_conditions: 5
            ic_stride: 1
          runtime:
            device: cuda
    """

    # --- I/O ---
    input_nc_path: Path            # single .nc file or directory of .nc files
    output_dir: Path               # directory for output .nc forecast files
    # --- Model ---
    checkpoint_path: Path
    global_means_path: Path
    global_stds_path: Path
    # --- Rollout ---
    forecast_steps: int = 10       # number of forecast days (Day 1..N)
    n_initial_conditions: int = 5  # how many ICs to produce forecasts for
    ic_stride: int = 1             # stride between ICs
    # --- Channels ---
    in_channels: Optional[List[int]] = None
    out_channels: Optional[List[int]] = None
    # --- Runtime ---
    device: str = "cuda"
    # --- Data ---
    field_variable: str = "fields"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "InferenceConfig":
        """Load from a YAML file with hierarchical ``inference:`` block."""
        yaml_loader = YAML()
        with open(yaml_path, "r") as f:
            raw = yaml_loader.load(f) or {}

        root = raw.get("inference", raw)
        io_cfg = root.get("io", root)
        model_cfg = root.get("model", root)
        rollout_cfg = root.get("rollout", root)
        runtime_cfg = root.get("runtime", root)

        def _req(d: dict, *keys: str) -> Any:
            for k in keys:
                if k in d and d[k] not in (None, ""):
                    return d[k]
            raise ValueError(f"Missing required config key: one of {keys}")

        config = cls(
            input_nc_path=Path(_req(io_cfg, "input_nc_path", "input_file")),
            output_dir=Path(io_cfg.get("output_dir", "./inference_output")),
            checkpoint_path=Path(_req(model_cfg, "checkpoint_path", "checkpoint")),
            global_means_path=Path(_req(model_cfg, "global_means_path")),
            global_stds_path=Path(_req(model_cfg, "global_stds_path")),
            forecast_steps=int(rollout_cfg.get("forecast_steps", 10)),
            n_initial_conditions=int(rollout_cfg.get("n_initial_conditions", 5)),
            ic_stride=int(rollout_cfg.get("ic_stride", 1)),
            in_channels=model_cfg.get("in_channels"),
            out_channels=model_cfg.get("out_channels"),
            device=str(runtime_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
            field_variable=str(io_cfg.get("field_variable", "fields")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate at startup with clear error messages."""
        if self.forecast_steps <= 0:
            raise ValueError("inference.rollout.forecast_steps must be > 0.")
        if self.n_initial_conditions <= 0:
            raise ValueError("inference.rollout.n_initial_conditions must be > 0.")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.input_nc_path.exists():
            raise FileNotFoundError(f"Input path not found: {self.input_nc_path}")
        if not self.global_means_path.exists() or not self.global_stds_path.exists():
            raise FileNotFoundError("Normalization stats paths are invalid.")


# =========================================================================
# Inference Engine
# =========================================================================

class AutoregressiveInference:
    """
    Autoregressive forecast generator.

    Produces de-normalized forecast fields as .nc files — one per initial
    condition. No metrics are computed here; that is the evaluator's job.

    Usage::

        config = InferenceConfig.from_yaml("configs/inference.yaml")
        engine = AutoregressiveInference(config)
        engine.run()   # saves forecasts to output_dir/
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)

        # --- Normalization stats ---
        means = np.load(config.global_means_path).squeeze()
        stds = np.load(config.global_stds_path).squeeze()
        if means.ndim == 2:
            means = means[0]
            stds = stds[0]

        # Resolve channel indices from data
        self._resolve_channels()

        # Subset to selected channels
        self.means_np = means[self.out_ch]           # numpy [C]
        self.stds_np = stds[self.out_ch]             # numpy [C]
        self.means_t = torch.from_numpy(self.means_np).float().to(self.device)
        self.stds_t = torch.from_numpy(self.stds_np).float().to(self.device)

        # --- Model ---
        self.model = self._load_model()
        logger.info("Inference engine initialized.")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _resolve_channels(self) -> None:
        """Determine channel indices from config or auto-detect from data."""
        test_files = self._find_nc_files()
        with nc.Dataset(test_files[0], "r") as ds:
            n_total = ds[self.config.field_variable].shape[1]

        self.in_ch = list(self.config.in_channels) if self.config.in_channels else list(range(n_total))
        self.out_ch = list(self.config.out_channels) if self.config.out_channels else list(range(n_total))
        logger.info(f"Channels: in={len(self.in_ch)}, out={len(self.out_ch)}")

    def _find_nc_files(self) -> List[str]:
        """Find .nc files — input can be a file or directory."""
        p = self.config.input_nc_path
        if p.is_file():
            return [str(p)]
        elif p.is_dir():
            files = sorted(str(f) for f in p.glob("*.nc"))
            if not files:
                raise FileNotFoundError(f"No .nc files found in {p}")
            return files
        else:
            raise FileNotFoundError(f"Input path not found: {p}")

    def _load_model(self) -> nn.Module:
        """Load model checkpoint with DDP-aware state dict handling."""
        model = KAI(in_chans=len(self.in_ch), out_chans=len(self.out_ch))

        logger.info(f"Loading checkpoint: {self.config.checkpoint_path}")
        torch.serialization.add_safe_globals([ScalarFloat])
        checkpoint = torch.load(
            self.config.checkpoint_path, map_location=self.device, weights_only=False,
        )

        model_state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
        cleaned = OrderedDict()
        for k, v in model_state.items():
            cleaned[k.replace("module.", "") if k.startswith("module.") else k] = v

        model.load_state_dict(cleaned)
        model = model.to(self.device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded. Parameters: {n_params:,}")
        return model

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout_one_ic(self, data_slice: torch.Tensor) -> np.ndarray:
        """
        Run autoregressive rollout for one initial condition.

        Args:
            data_slice: Normalized tensor [prediction_length, C, H, W].

        Returns:
            De-normalized forecast array [prediction_length, C, H, W].
        """
        pred_len = data_slice.shape[0]
        means = self.means_t.view(1, -1, 1, 1)
        stds = self.stds_t.view(1, -1, 1, 1)
        forecasts = []

        with torch.no_grad():
            for i in range(pred_len):
                if i == 0:
                    first = data_slice[0:1]               # [1, C, H, W]
                    pred = first                           # Day 0 = initial condition
                    future_pred = self.model(first)        # first forecast
                else:
                    pred = future_pred                     # Day i forecast
                    future_pred = self.model(future_pred)  # autoregressive

                # De-normalize and store
                denorm = pred * stds + means
                forecasts.append(denorm.squeeze(0).cpu().numpy())  # [C, H, W]

        return np.stack(forecasts, axis=0)  # [prediction_length, C, H, W]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """
        Execute end-to-end inference:

        1. Discover test file and build IC indices
        2. For each IC, run autoregressive rollout
        3. Save all de-normalized forecasts to a single .nc file

        Returns:
            Path to the output .nc file.
        """
        logger.info("=" * 80)
        logger.info("AUTOREGRESSIVE INFERENCE")
        logger.info(f"  Input:          {self.config.input_nc_path}")
        logger.info(f"  Output dir:     {self.config.output_dir}")
        logger.info(f"  Forecast steps: {self.config.forecast_steps}")
        logger.info(f"  ICs requested:  {self.config.n_initial_conditions} (stride={self.config.ic_stride})")
        logger.info("=" * 80)

        test_files = self._find_nc_files()
        test_file = test_files[0]
        prediction_length = self.config.forecast_steps + 1   # Day 0 .. Day N

        with nc.Dataset(test_file, "r") as ds:
            fields = ds[self.config.field_variable]
            total_t = fields.shape[0]
            max_ic = total_t - prediction_length
            if max_ic <= 0:
                raise ValueError(
                    f"Not enough timesteps ({total_t}) for "
                    f"{prediction_length}-step rollout."
                )

            # Build IC indices
            cap = min(max_ic, self.config.n_initial_conditions * self.config.ic_stride)
            ics = list(range(0, cap, self.config.ic_stride))[:self.config.n_initial_conditions]
            n_ics = len(ics)
            logger.info(f"Total timesteps: {total_t}; Running {n_ics} ICs.")

            all_forecasts = []   # will hold [n_ics, prediction_length, C, H, W]

            for count, ic in enumerate(ics):
                end_idx = ic + prediction_length
                raw = fields[ic:end_idx, self.in_ch, :, :]

                # Normalize
                m = self.means_np.reshape(1, -1, 1, 1)
                s = self.stds_np.reshape(1, -1, 1, 1)
                data_slice = (np.array(raw) - m) / s
                data_tensor = torch.as_tensor(data_slice, device=self.device, dtype=torch.float32)

                # Rollout
                forecast = self._rollout_one_ic(data_tensor)  # [pred_len, C, H, W]
                all_forecasts.append(forecast)

                if (count + 1) % 10 == 0 or (count + 1) == n_ics:
                    logger.info(f"  Completed IC {count + 1}/{n_ics} (ic={ic})")

        # Stack: [n_ics, prediction_length, C, H, W]
        all_forecasts = np.stack(all_forecasts, axis=0)

        # Save to NetCDF
        out_path = self._save_forecasts_nc(all_forecasts, ics, test_file)

        logger.info("=" * 80)
        logger.info(f"INFERENCE COMPLETE — saved to {out_path}")
        logger.info("=" * 80)
        return out_path

    # ------------------------------------------------------------------
    # NetCDF output
    # ------------------------------------------------------------------

    def _save_forecasts_nc(
        self,
        forecasts: np.ndarray,
        ic_indices: List[int],
        source_file: str,
    ) -> Path:
        """
        Save de-normalized forecasts to a NetCDF file.

        Output dimensions:
            (initial_condition, lead_time, channel, lat, lon)

        Args:
            forecasts:  Array of shape [n_ics, prediction_length, C, H, W].
            ic_indices: The IC indices used (for metadata).
            source_file: Path to the source test file (for metadata).

        Returns:
            Path to the saved .nc file.
        """
        out_dir = self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "forecast.nc"

        n_ics, pred_len, C, H, W = forecasts.shape

        with nc.Dataset(str(out_path), "w", format="NETCDF4") as dst:
            # --- Dimensions ---
            dst.createDimension("initial_condition", n_ics)
            dst.createDimension("lead_time", pred_len)
            dst.createDimension("channel", C)
            dst.createDimension("lat", H)
            dst.createDimension("lon", W)

            # --- Coordinate variables ---
            v_ic = dst.createVariable("initial_condition", "i4", ("initial_condition",))
            v_ic[:] = np.array(ic_indices)
            v_ic.description = "Index of initial condition in the source test file"

            v_lead = dst.createVariable("lead_time", "i4", ("lead_time",))
            v_lead[:] = np.arange(pred_len)
            v_lead.description = "Forecast lead time in days (0 = initial condition)"

            v_ch = dst.createVariable("channel", "i4", ("channel",))
            v_ch[:] = np.array(self.out_ch)
            v_ch.description = "Channel indices corresponding to ERA5 variables"

            # --- Forecast data ---
            v_forecast = dst.createVariable(
                "forecast", "f4",
                ("initial_condition", "lead_time", "channel", "lat", "lon"),
                zlib=True, complevel=4, shuffle=True,
            )
            v_forecast[:] = forecasts.astype(np.float32)
            v_forecast.description = "De-normalized autoregressive forecast fields"
            v_forecast.units = "original ERA5 units"

            # --- Global attributes ---
            dst.title = "KAI-Atmos Autoregressive Forecast"
            dst.source_file = os.path.basename(source_file)
            dst.checkpoint = str(self.config.checkpoint_path)
            dst.forecast_steps = self.config.forecast_steps
            dst.n_initial_conditions = n_ics
            dst.ic_stride = self.config.ic_stride

        size_mb = os.path.getsize(out_path) / (1024 ** 2)
        logger.info(f"Saved forecast: {out_path} ({size_mb:.1f} MB)")
        return out_path

StandaloneInference = AutoregressiveInference


# =========================================================================
# CLI entry point
# =========================================================================

def main():
    """CLI entry point for autoregressive inference."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Autoregressive inference for weather forecasting (KAI-Atmos)"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to inference YAML")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    config = InferenceConfig.from_yaml(args.config)
    engine = AutoregressiveInference(config)
    engine.run()


if __name__ == "__main__":
    main()
