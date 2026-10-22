#src/utils.py
"""
Copyright (c) 2025 MetaEarth-Lab
Licensed under the BSD 3-Clause License. Redistribution and use in source and binary forms, with or without modification, are permitted. See the LICENSE file in the project root for the full license text.
"""


import glob
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple, Literal
import math
import netCDF4 as nc
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import sys
# ----------------------------------------------------------------------
# Config dataclass
# ----------------------------------------------------------------------

@dataclass
class DataConfig:
    # temporal settings
    dt: int                           # stride between input steps (in indices)
    n_history: int                    # number of past steps (not counting the current one)

    # channels (indices into the "fields" variable in the NetCDF files)
    in_channels: Sequence[int]
    out_channels: Sequence[int]

    # normalization
    normalize: bool = True
    normalization: str = "zscore"     # currently only "zscore" supported
    global_means_path: Optional[str] = None
    global_stds_path: Optional[str] = None

    # coordinate grid channels
    add_grid: bool = False
    gridtype: str = "linear"          # "linear" or "sinusoidal"
    N_grid_channels: int = 2          # 2 for linear, 4 for sinusoidal

    # loader settings
    batch_size: int = 1
    num_workers: int = 1


# ----------------------------------------------------------------------
# Normalization & grid helpers
# ----------------------------------------------------------------------

def _apply_normalization(
    img: np.ndarray,
    channels_idx: np.ndarray,
    means: Optional[np.ndarray],
    stds: Optional[np.ndarray],
    normalization: str,
    normalize_flag: bool,
) -> np.ndarray:
    """
    Normalize img using provided means/stds.

    img shape:
        - (T, C, H, W) or
        - (C, H, W)

    channels_idx:
        indices of channels that were selected.
    """
    if not normalize_flag:
        return img

    if means is None or stds is None:
        raise ValueError("Normalization requested but means/stds are None.")

    if normalization.lower() != "zscore":
        raise NotImplementedError("Only 'zscore' normalization is supported.")


    if means.ndim == 4:
        # e.g. (1, C, 1, 1) → (C,)
        m = means.squeeze()[channels_idx]
        s = stds.squeeze()[channels_idx]
    elif means.ndim == 2:
        m = means[0, channels_idx]
        s = stds[0, channels_idx]
    elif means.ndim == 1:
        m = means[channels_idx]
        s = stds[channels_idx]
    else:
        raise ValueError(f"Unexpected means/stds shape: {means.shape}")

    if img.ndim == 4:
        # (T, C, H, W)
        m_b = m.reshape(1, -1, 1, 1)
        s_b = s.reshape(1, -1, 1, 1)
    elif img.ndim == 3:
        # (C, H, W)
        m_b = m.reshape(-1, 1, 1)
        s_b = s.reshape(-1, 1, 1)
    else:
        raise ValueError(f"Unexpected img ndim for normalization: {img.ndim}")

    img = img - m_b
    img = img / (s_b + 1e-8)
    return img


def _add_grid_channels(
    img: np.ndarray,
    gridtype: str,
    N_grid_channels: int,
) -> np.ndarray:
    """
    Add coordinate grid channels to a 4D array (T, C, H, W) or 3D (C, H, W).
    Grid is identical for all timesteps if T is present.
    """
    add_time_dim = False
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)  # (1, C, H, W)
        add_time_dim = True

    _, _, H, W = img.shape

    if gridtype == "linear":
        assert N_grid_channels == 2, "N_grid_channels must be 2 for linear grid."
        xs = np.linspace(-1.0, 1.0, H, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, W, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(ys, xs)  # (H, W) each
        grid = np.stack([grid_x, grid_y], axis=0)  # (2, H, W)

    elif gridtype == "sinusoidal":
        assert N_grid_channels == 4, "N_grid_channels must be 4 for sinusoidal grid."
        xs = np.linspace(0.0, 2 * math.pi, H, dtype=np.float32)
        ys = np.linspace(0.0, 2 * math.pi, W, dtype=np.float32)
        sin_x = np.sin(xs)
        cos_x = np.cos(xs)
        sin_y = np.sin(ys)
        cos_y = np.cos(ys)
        grid_y1, grid_x1 = np.meshgrid(sin_y, sin_x)
        grid_y2, grid_x2 = np.meshgrid(cos_y, cos_x)
        grid = np.stack([grid_x1, grid_y1, grid_x2, grid_y2], axis=0)  # (4, H, W)

    else:
        raise ValueError(f"Unknown gridtype '{gridtype}'")

    grid = np.expand_dims(grid, axis=0)  # (1, G, H, W)
    img = np.concatenate([img, grid], axis=1)  # (T, C+G, H, W)

    if add_time_dim:
        return img[0]  # back to (C+G, H, W)
    return img


# ----------------------------------------------------------------------
# Reshaping utilities
# ----------------------------------------------------------------------

def reshape_fields(
    img: np.ndarray,
    inp_or_tar: Literal["inp", "tar"],
    config: DataConfig,
    means: Optional[np.ndarray] = None,
    stds: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Core reshape and preprocessing function.

    Args:
        img:
            - For inputs: np.ndarray of shape (T, C_in, H, W) with T = n_history + 1.
            - For targets: np.ndarray of shape (C_out, H, W) or (1, C_out, H, W).
        inp_or_tar: "inp" or "tar".
        config: DataConfig.
        means, stds: normalization statistics (see _apply_normalization).

    Returns:
        torch.Tensor of shape:
            - inputs: (C_eff * (n_history + 1), H, W)
            - targets: (C_eff, H, W)
    """
    # ensure a time dimension exists
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)  # (1, C, H, W)

    T, C, H, W = img.shape

    # figure out which channels we're using for statistics
    if inp_or_tar == "inp":
        channels_idx = np.array(config.in_channels, dtype=int)
    else:
        channels_idx = np.array(config.out_channels, dtype=int)

    # select channels
    img = img[:, channels_idx, :, :]  # (T, C_sel, H, W)
    C_sel = img.shape[1]

    # normalization
    img = _apply_normalization(
        img,
        channels_idx=channels_idx,
        means=means,
        stds=stds,
        normalization=config.normalization,
        normalize_flag=config.normalize,
    )

    # add grid channels (only for inputs)
    if config.add_grid and inp_or_tar == "inp":
        img = _add_grid_channels(img, config.gridtype, config.N_grid_channels)
        C_sel = img.shape[1]

    # reshape
    if inp_or_tar == "inp":
        # (T, C_eff, Hc, Wc) -> (T*C_eff, Hc, Wc)
        T_eff, C_eff, Hc, Wc = img.shape
        img = img.reshape(T_eff * C_eff, Hc, Wc)
    else:  # "tar"
        if T != 1:
            # if multiple times are present, flatten into channels
            img = img.reshape(T * C_sel, img.shape[2], img.shape[3])
        else:
            img = img[0]  # (C_eff, Hc, Wc)

    return torch.as_tensor(img)


# ----------------------------------------------------------------------
# Distributed helpers
# ----------------------------------------------------------------------

def is_distributed() -> bool:
    """Return True if torch.distributed is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank_safe() -> int:
    """Return the current process rank if in distributed mode, else 0."""
    if is_distributed():
        return dist.get_rank()
    return 0


# ----------------------------------------------------------------------
# NetCDF dataset
# ----------------------------------------------------------------------

class ClimateNetCDFDataset(Dataset):
    """
    Dataset for climate data stored as multiple NetCDF files, each containing
    a 'fields' variable of shape (time, channels, height, width).

    Features:
    - Uses per-file time axis (supports variable length per file).
    - Builds input sequences of length (n_history + 1) with stride dt.
    - Predicts target dt steps ahead.
    - Optional normalization and grid channels.
    """

    def __init__(self, config: DataConfig, data_dir: str, train: bool = True):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.train = train

        self.in_channels = np.array(config.in_channels, dtype=int)
        self.out_channels = np.array(config.out_channels, dtype=int)

        # normalization statistics
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        if config.normalize:
            if config.global_means_path is None or config.global_stds_path is None:
                raise ValueError(
                    "Normalization enabled but global_means_path or global_stds_path is None."
                )
            self.means = np.load(config.global_means_path)
            self.stds = np.load(config.global_stds_path)

        # discover files and gather per-file statistics
        self._discover_files()
        self.channel_names = None
        try:
            import xarray as xr

            # Load first file as xarray (lightweight)
            xr_ds = xr.open_dataset(self.files_paths[0])

            if "channel" in xr_ds.coords:
                self.channel_names = list(xr_ds["channel"].values)
                logging.info(f" Loaded {len(self.channel_names)} channel names from NetCDF file.")
            else:
                logging.warning(" NetCDF file has no 'channel' coordinate. Channel names unavailable.")

        except Exception as e:
            logging.warning(f" Failed to extract channel names from NetCDF: {e}")
        self._build_index()

    def _discover_files(self) -> None:
        """Find all .nc files in data_dir and read shape info per file."""
        pattern = os.path.join(self.data_dir, "*.nc")
        self.files_paths: List[str] = sorted(glob.glob(pattern))
        if len(self.files_paths) == 0:
            raise FileNotFoundError(f"No .nc files found in directory: {self.data_dir}")

        self.n_files = len(self.files_paths)

        self._file_lengths: List[int] = []
        self.img_shape_x: Optional[int] = None
        self.img_shape_y: Optional[int] = None

        for path in self.files_paths:
            with nc.Dataset(path, "r") as ds:
                fields = ds["fields"]
                T, C, H, W = fields.shape
                self._file_lengths.append(T)
                if self.img_shape_x is None:
                    self.img_shape_x = H
                    self.img_shape_y = W
                else:
                    if H != self.img_shape_x or W != self.img_shape_y:
                        raise ValueError(
                            f"Inconsistent spatial shapes across files. "
                            f"Got {(H, W)} in {path}, expected "
                            f"{(self.img_shape_x, self.img_shape_y)}."
                        )

        logging.info(
            f"Found {self.n_files} NetCDF files under {self.data_dir}. "
            f"Spatial dims: {self.img_shape_x} x {self.img_shape_y}."
        )

    def _build_index(self) -> None:
        """
        Pre-compute, for each file, the valid range of center indices t where:
            t - dt * n_history >= 0
            t + dt < T_file
        """
        dt = self.config.dt
        n_history = self.config.n_history

        self._valid_starts: List[int] = []
        self._valid_ends: List[int] = []  # inclusive
        self._num_valid: List[int] = []

        for T in self._file_lengths:
            t_min = dt * n_history
            t_max = T - dt - 1  # last center index with target inside this file

            if t_min > t_max:
                self._valid_starts.append(0)
                self._valid_ends.append(-1)
                self._num_valid.append(0)
            else:
                self._valid_starts.append(t_min)
                self._valid_ends.append(t_max)
                self._num_valid.append(t_max - t_min + 1)

        # prefix sums to map global index -> (file_idx, local_t)
        self._file_offsets: List[int] = [0]
        for n in self._num_valid:
            self._file_offsets.append(self._file_offsets[-1] + n)
        self.n_samples_total: int = self._file_offsets[-1]

        if self.n_samples_total == 0:
            raise RuntimeError("No valid samples found in the provided NetCDF files.")

        logging.info(f"Total valid samples: {self.n_samples_total}")

    def __len__(self) -> int:
        return self.n_samples_total

    def _locate_sample(self, idx: int) -> Tuple[int, int]:
        """
        Map a global index idx in [0, n_samples_total) to (file_idx, center_idx_in_file).
        """
        if idx < 0 or idx >= self.n_samples_total:
            raise IndexError(f"Index {idx} out of range [0, {self.n_samples_total}).")

        file_idx = 0
        while file_idx + 1 < len(self._file_offsets) and self._file_offsets[file_idx + 1] <= idx:
            file_idx += 1

        local_sample_idx = idx - self._file_offsets[file_idx]
        center_idx = self._valid_starts[file_idx] + local_sample_idx
        return file_idx, center_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, center_idx = self._locate_sample(idx)
        dt = self.config.dt
        n_history = self.config.n_history

        # read relevant data from file
        path = self.files_paths[file_idx]
        with nc.Dataset(path, "r") as ds:
            fields = ds["fields"]  # shape: (T, C, H, W)

            # input history: times [center_idx - dt*n_history, ..., center_idx]
            start_t = center_idx - dt * n_history
            end_t = center_idx + 1
            inp_seq = np.array(
                fields[start_t:end_t:dt, self.in_channels, :, :], dtype=np.float32
            )  # (n_history+1, C_in, H, W)

            # target at time center_idx + dt
            target_t = center_idx + dt
            tar_img = np.array(
                fields[target_t, self.out_channels, :, :], dtype=np.float32
            )  # (C_out, H, W)

        inp = reshape_fields(
            img=inp_seq,
            inp_or_tar="inp",
            config=self.config,
            means=self.means,
            stds=self.stds,
        )

        tar = reshape_fields(
            img=tar_img,
            inp_or_tar="tar",
            config=self.config,
            means=self.means,
            stds=self.stds,
        )

        return inp, tar


# ----------------------------------------------------------------------
# DataLoader factory
# ----------------------------------------------------------------------

def build_data_loader(
    config: DataConfig,
    data_dir: str,
    distributed: bool = False,
    train: bool = True,
) -> Tuple[DataLoader, ClimateNetCDFDataset, Optional[DistributedSampler]]:
    """
    Create a dataset + DataLoader (and optional DistributedSampler).

    Returns:
        dataloader, dataset, sampler (sampler will be None if not used or not train)
    """
    dataset = ClimateNetCDFDataset(config=config, data_dir=data_dir, train=train)

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=train)
        shuffle = False
    else:
        sampler = None
        shuffle = train

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader, dataset, sampler


# ----------------------------------------------------------------------
# Datetime → step-index helper
# ----------------------------------------------------------------------

def datetimes_to_step_indices(
    date_strings: Sequence[str],
    fmt: str = "%Y-%m-%d %H:%M:%S",
    step_hours: int = 6,
) -> List[int]:
    """
    Convert a list of datetime strings into step indices, assuming:
    - steps are spaced by `step_hours` hours
    - index 0 corresponds to Jan 1, 00:00 of the same year
    """
    indices: List[int] = []
    for s in date_strings:
        d = datetime.strptime(s, fmt)
        tt = d.timetuple()
        day_of_year = tt.tm_yday - 1  # 0-based
        hour_of_day = tt.tm_hour
        hours_since_jan_01 = day_of_year * 24 + hour_of_day
        idx = hours_since_jan_01 // step_hours
        indices.append(int(idx))
    return indices

# ----------------------------------------------------------------------
# Climatology Generation
# ----------------------------------------------------------------------

def compute_daily_climatology(
    config: 'DataConfig',
    data_dir: str,
    save_dir: str,
    filename: str = "daily_climatology.nc",
    channels_ids: Optional[Sequence[int]] = None,
    time_chunk_size: int = 1000,  
    force_recompute: bool = False
) -> str:
    """
    Computes the daily climatology (mean over years for each day-of-year)
    for the specified channels and saves it as a NetCDF file.
    
    Args:
        config: DataConfig object (used for spatial shapes).
        data_dir: Directory containing training .nc files.
        save_dir: Directory where the output .nc file will be saved.
        filename: Name of the output file.
        channels_ids: Indices of channels to compute. If None, uses config.out_channels.
        time_chunk_size: Number of timesteps to read from disk at once. 
                         Lower this if you hit MemoryErrors.
        force_recompute: Overwrite existing file.
    Returns:
        str: Full path to the saved climatology file.
    """
    save_path = os.path.join(save_dir, filename)
    
    # If the file already exists, we might want to skip or overwrite. 
    # Here we skip for efficiency if it exists.
    if os.path.exists(save_path):
        logging.info(f"Climatology file already exists at {save_path}. Skipping computation.")
        return save_path

    logging.info(f"Computing daily climatology from {data_dir}...")
    
    # 1. Discover files
    pattern = os.path.join(data_dir, "*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {data_dir}")

    # 2. Determine target channels
    if channels_ids is None:
        target_indices = np.array(config.out_channels, dtype=int)
    else:
        target_indices = np.array(channels_ids, dtype=int)

    with nc.Dataset(files[0], "r") as ds:
        if "fields" not in ds.variables or "time" not in ds.variables:
            raise ValueError(f"Missing 'fields' or 'time' in {files[0]}")
        
        _, C_total, H, W = ds["fields"].shape
        calendar = getattr(ds["time"], 'calendar', 'standard')
    
    logging.info(f"Data shape: C={C_total}, H={H}, W={W}")
    target_indices = np.array(
        config.out_channels if channels_ids is None else channels_ids, 
        dtype=int
    )
    
    if np.any(target_indices >= C_total):
        raise ValueError(f"Invalid channels: {target_indices.tolist()} (max: {C_total-1})")
    
    C_out = len(target_indices)
    logging.info(f"Computing climatology for {C_out} channels")

    # 4. ALLOCATE (with memory check)
    n_days = 366
    est_gb = (n_days * C_out * H * W * 8) / (1024**3)
    
    try:
        clim_sum = np.zeros((n_days, C_out, H, W), dtype=np.float64)
        clim_counts = np.zeros(n_days, dtype=np.int32)  # Efficient counter
    except MemoryError:
        logging.error(f"Not enough RAM for {est_gb:.2f} GB!")
        raise

    # 5. PROCESS FILES WITH CHUNKING
    total_timesteps = 0
    skipped = 0

    for f_path in tqdm(files, desc="Files"):
        try:
            steps = _process_file_chunked(
                f_path, target_indices, clim_sum, clim_counts, time_chunk_size
            )
            total_timesteps += steps
        except Exception as e:
            logging.warning(f"Skipping {os.path.basename(f_path)}: {e}")
            skipped += 1

    logging.info(f"Processed {total_timesteps} timesteps from {len(files)-skipped} files")
    if skipped > 0:
        logging.warning(f"Skipped {skipped} files due to errors")

    # 6. VALIDATE COVERAGE
    days_with_data = np.sum(clim_counts > 0)
    if days_with_data < 365:
        logging.warning(f"Only {days_with_data}/366 days have data!")
    
    valid_counts = clim_counts[clim_counts > 0]
    #logging.info(f"Samples per day: min={valid_counts.min()}, max={valid_counts.max()}, "
    #            f"mean={valid_counts.mean():.1f}")

    # 7. COMPUTE MEAN (vectorized)
    #logging.info("Computing mean...")
    valid_mask = clim_counts > 0
    climatology = np.zeros((n_days, C_out, H, W), dtype=np.float32)
    
    # Broadcast division
    np.divide(
        clim_sum, 
        clim_counts[:, None, None, None], 
        out=climatology, 
        where=valid_mask[:, None, None, None]
    )
    
    # Check for issues
    if np.any(~np.isfinite(climatology)):
        n_bad = np.sum(~np.isfinite(climatology))
        logging.warning(f"Found {n_bad} non-finite values in output!")

    # 8. ATOMIC SAVE WITH METADATA
    temp_path = save_path + ".tmp"
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        with nc.Dataset(temp_path, "w", format="NETCDF4") as dst:
            # Dimensions
            dst.createDimension("day_of_year", n_days)
            dst.createDimension("channel", C_out)
            dst.createDimension("height", H)
            dst.createDimension("width", W)

            # Variables
            v_doy = dst.createVariable("day_of_year", "i4", ("day_of_year",))
            v_doy[:] = np.arange(1, n_days + 1)
            v_doy.description = "1=Jan 1, 366=Dec 31 (leap year)"
            
            v_counts = dst.createVariable("sample_counts", "i4", ("day_of_year",))
            v_counts[:] = clim_counts
            v_counts.description = "Number of samples per day"
            
            v_channels = dst.createVariable("channel_indices", "i4", ("channel",))
            v_channels[:] = target_indices
            v_channels.description = "Original channel indices"
            
            v_clim = dst.createVariable(
                "climatology", "f4", 
                ("day_of_year", "channel", "height", "width"),
                zlib=True, complevel=4, shuffle=True
            )
            v_clim[:] = climatology
            v_clim.description = "Daily climatology mean"
            
            # Global metadata
            dst.title = "Daily Climatology"
            dst.history = f"Created {datetime.now().isoformat()}"
            dst.source_files = f"{len(files)-skipped} files"
            dst.total_timesteps = int(total_timesteps)
            dst.time_chunk_size = time_chunk_size
        
        # Atomic rename
        if os.path.exists(save_path):
            os.remove(save_path)
        os.rename(temp_path, save_path)
        
        size_mb = os.path.getsize(save_path) / (1024**2)
        logging.info(f"Saved to {save_path} ({size_mb:.1f} MB)")
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logging.error(f"Save failed: {e}")
        raise

    logging.info("="*80)
    logging.info("Climatology computation complete!")
    logging.info("="*80)

    return save_path


def _process_file_chunked(
    f_path: str,
    target_indices: np.ndarray,
    clim_sum: np.ndarray,
    clim_counts: np.ndarray,
    chunk_size: int
) -> int:
    """
    Process a single file in chunks to minimize memory usage.
    """
    with nc.Dataset(f_path, "r") as ds:
        if "fields" not in ds.variables or "time" not in ds.variables:
            raise ValueError("Missing required variables")

        # 1. Load time (small, always fits in RAM)
        time_var = ds["time"]
        times = nc.num2date(
            time_var[:], 
            units=time_var.units, 
            calendar=getattr(time_var, 'calendar', 'standard')
        )
        
        # Convert to day-of-year indices (0-365)
        doy_indices = np.array([t.timetuple().tm_yday - 1 for t in times], dtype=np.int32)
        doy_indices = np.clip(doy_indices, 0, 365)
        
        T = len(times)
        
        # 2. Process in chunks (THIS IS THE MEMORY-SAVING PART)
        for start_idx in range(0, T, chunk_size):
            end_idx = min(start_idx + chunk_size, T)
            
            # Read chunk from disk: (chunk_len, C_out, H, W)
            data_chunk = ds["fields"][start_idx:end_idx, target_indices, :, :]
            doy_chunk = doy_indices[start_idx:end_idx]
            

            

            for doy in range(366):
                 mask = (doy_chunk == doy)
                 n_samples = np.sum(mask)
                 if n_samples > 0:
                     clim_sum[doy] += data_chunk[mask].sum(axis=0, dtype=np.float64)
                     clim_counts[doy] += n_samples
    
    return T
# ----------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        """
        Args:
            d (int): Dimension of the geometric domain.
            p (int): Degree of the Lp-norm (e.g., 2 for L2).
            size_average (bool): If True, averages the loss across the batch size.
            reduction (bool): If True, applies batch-wise reduction (mean or sum).
        """
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        """
        Computes the absolute Lp-norm error between predictions and targets.
        Args:
            x: Predictions tensor, shape (N, ...).
            y: Targets tensor, shape (N, ...).
            
        Returns:
            Scalar absolute error or batch-wise absolute error tensor.
        """
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        """
        Computes the relative Lp-norm error between predictions and targets.
        Args:
            x: Predictions tensor, shape (N, ...).
            y: Targets tensor, shape (N, ...).
            
        Returns:
            Scalar relative error or batch-wise relative error tensor.
        """
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        """
        Args:
            x: Predictions tensor. 
            y: Targets tensor.
            
        Returns:
            Relative Lp-norm error based on initialization settings.
        """
        return self.rel(x, y)

# ----------------------------------------------------------------------
# Logging helpers
# ----------------------------------------------------------------------

import subprocess

_LOG_FORMAT = "%(asctime)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(rank: int = 0, log_file: Optional[str] = None) -> None:
    """
    Configures the root logger.
    Rank 0: Logs to Console + File.
    Rank > 0: Logs to File only (if provided), otherwise silent.
    """
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()
    
    # We keep the internal level at INFO, but we won't print "INFO" in the text
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # 1. File Handler (Everyone logs to file if provided)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a')
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # 2. Console Handler (Only Rank 0)
    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        root.addHandler(ch)

# ----------------------------------------------------------------------
# YAML config loader (YParams-like)
# ----------------------------------------------------------------------

from ruamel.yaml import YAML


class YParams:
    """
    Minimal YAML configuration loader.

    Usage:
        params = YParams("configs/KAI.yaml", "default")
        print(params.some_key)

    - Loads the subsection named `config_name` from the YAML file.
    - Stores all key/value pairs both in `params` dict and as attributes.
    """

    def __init__(self, yaml_filename: str, config_name: str, print_params: bool = False):
        self._yaml_filename = yaml_filename
        self._config_name = config_name
        self.params = {}

        if print_params:
            print("------------------ Configuration ------------------")

        with open(yaml_filename, "r") as f:
            root = YAML().load(f)
            yaml_content = root[config_name]

        for key, val in yaml_content.items():
            if print_params:
                print(key, val)
            if val == "None":
                val = None

            self.params[key] = val
            setattr(self, key, val)

        if print_params:
            print("---------------------------------------------------")

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        setattr(self, key, val)

    def __contains__(self, key):
        return key in self.params

    def update_params(self, config: dict) -> None:
        for key, val in config.items():
            self.params[key] = val
            setattr(self, key, val)

    def log(self) -> None:
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: %s", self._yaml_filename)
        logging.info("Configuration name: %s", self._config_name)
        for key, val in self.params.items():
            logging.info("%s %s", key, val)
        logging.info("---------------------------------------------------")
