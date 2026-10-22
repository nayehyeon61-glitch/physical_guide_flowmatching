"""PG-CFM data pipeline on top of the vendored KAI loader.

Keeps KAI's file discovery, z-score normalization and within-year pairing
(``third_party/kai/src/utils.py``) and adds what PG-CFM needs:
dynamic/static channel split, calendar condition and K-step sequences.
"""

import importlib.util
import logging
import os
from typing import Dict

import netCDF4 as nc
import numpy as np
import torch

from . import contract

_KAI_UTILS = os.path.join(os.path.dirname(__file__), "..", "third_party", "kai", "src", "utils.py")


def _load_kai_utils():
    spec = importlib.util.spec_from_file_location("kai_utils", os.path.abspath(_KAI_UTILS))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kai_utils = _load_kai_utils()


def kai_data_config(global_means_path: str, global_stds_path: str, **kwargs) -> "kai_utils.DataConfig":
    """DataConfig matching configs/KAI.yaml (dt=1, n_history=0, all 67 channels)."""
    params = dict(
        dt=1,
        n_history=0,
        in_channels=list(range(contract.N_CHANNELS)),
        out_channels=list(range(contract.N_CHANNELS)),
        normalize=True,
        normalization="zscore",
        global_means_path=global_means_path,
        global_stds_path=global_stds_path,
        add_grid=False,
    )
    params.update(kwargs)
    return kai_utils.DataConfig(**params)


class PGCFMSequenceDataset(kai_utils.ClimateNetCDFDataset):
    """K-step daily sequences from KAI yearly NetCDF files.

    Sample t of a file yields states t, t+1, ..., t+K, all inside that file
    (no 12/31 -> 1/1 pairs, as in KAI). ``horizon=1`` gives exactly KAI's
    (inp, tar) pairs.

    Returns a dict:
        x0      (66, H, W)    normalized dynamic state at t
        targets (K, 66, H, W) normalized dynamic states t+1..t+K
        static  (1, H, W)     normalized orography (copied, never generated)
        doy     (K+1,)        day-of-year index of each state (0-based)
    """

    def __init__(self, config, data_dir: str, horizon: int = 1, train: bool = True):
        if config.dt != 1 or config.n_history != 0:
            raise ValueError("PG-CFM uses the KAI contract dt=1, n_history=0.")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = horizon
        super().__init__(config, data_dir, train=train)
        self._check_channels()

    def _check_channels(self) -> None:
        if self.channel_names is not None:
            names = [str(n) for n in self.channel_names]
            if names != contract.CHANNEL_NAMES:
                raise ValueError(f"Channel order differs from the KAI contract: {names}")
        if (self.img_shape_x, self.img_shape_y) != (contract.N_LAT, contract.N_LON):
            # Allowed for reduced test grids, but physics operators derive the
            # grid spacing from the shape, so make it visible.
            logging.warning(
                "Grid %dx%d differs from the KAI 1.5 deg contract (%dx%d).",
                self.img_shape_x, self.img_shape_y, contract.N_LAT, contract.N_LON,
            )

    def _build_index(self) -> None:
        # Same as KAI, but the last valid start leaves room for `horizon` targets.
        self._valid_starts, self._valid_ends, self._num_valid = [], [], []
        for T in self._file_lengths:
            t_max = T - self.horizon - 1
            if t_max < 0:
                self._valid_starts.append(0)
                self._valid_ends.append(-1)
                self._num_valid.append(0)
            else:
                self._valid_starts.append(0)
                self._valid_ends.append(t_max)
                self._num_valid.append(t_max + 1)
        self._file_offsets = [0]
        for n in self._num_valid:
            self._file_offsets.append(self._file_offsets[-1] + n)
        self.n_samples_total = self._file_offsets[-1]
        if self.n_samples_total == 0:
            raise RuntimeError("No valid samples found in the provided NetCDF files.")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, t = self._locate_sample(idx)
        with nc.Dataset(self.files_paths[file_idx], "r") as ds:
            seq = np.array(ds["fields"][t:t + self.horizon + 1], dtype=np.float32)
            doy = np.array(ds["time"][t:t + self.horizon + 1], dtype=np.int64)

        seq = kai_utils._apply_normalization(
            seq,
            channels_idx=self.in_channels,
            means=self.means,
            stds=self.stds,
            normalization=self.config.normalization,
            normalize_flag=self.config.normalize,
        ).astype(np.float32)
        dyn = torch.from_numpy(seq[:, contract.DYNAMIC_CHANNELS])
        return {
            "x0": dyn[0],
            "targets": dyn[1:],
            "static": torch.from_numpy(seq[0, contract.STATIC_CHANNELS]),
            "doy": torch.from_numpy(doy),
        }


def assemble_full_state(dynamic: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
    """Re-insert the static channel(s) to get the 67-channel KAI layout.

    dynamic: (..., 66, H, W), static: (..., 1, H, W) -> (..., 67, H, W)
    """
    shape = dynamic.shape[:-3] + (contract.N_CHANNELS,) + dynamic.shape[-2:]
    full = dynamic.new_empty(shape)
    full[..., contract.DYNAMIC_CHANNELS, :, :] = dynamic
    full[..., contract.STATIC_CHANNELS, :, :] = static.expand(dynamic.shape[:-3] + static.shape[-3:])
    return full
