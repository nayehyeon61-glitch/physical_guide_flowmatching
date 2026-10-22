"""KAI ERA5 data contract (KAI_ERA5_IO_FORMAT, 2026-10-22).

Single source of truth for channel order, units, normalization statistics,
grid and the physical constants used by the PG-CFM physics residuals.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Grid: 1.5 deg equiangular, poles included, lat ascending (S -> N)
# ----------------------------------------------------------------------
N_LAT, N_LON = 121, 240
LATITUDES = np.linspace(-90.0, 90.0, N_LAT)
LONGITUDES = np.arange(N_LON) * 1.5

# ----------------------------------------------------------------------
# Channels: z, q, t, u, v x 12 pressure levels, then 7 surface fields
# ----------------------------------------------------------------------
LEVELS_HPA = [50, 100, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
LEVELS_PA = np.array(LEVELS_HPA, dtype=np.float64) * 100.0
N_LEVELS = len(LEVELS_HPA)
UPPER_AIR_VARS = ["z", "q", "t", "u", "v"]
SURFACE_VARS = ["t2m", "sp", "tisr", "tcwv", "orog", "msl", "skt"]

CHANNEL_NAMES: List[str] = [
    f"{v}{p}" for v in UPPER_AIR_VARS for p in LEVELS_HPA
] + SURFACE_VARS
N_CHANNELS = len(CHANNEL_NAMES)  # 67

CHANNEL_INDEX: Dict[str, int] = {n: i for i, n in enumerate(CHANNEL_NAMES)}


def level_slice(var: str) -> slice:
    """Channel slice for one upper-air variable, ordered 50 -> 1000 hPa."""
    k = UPPER_AIR_VARS.index(var)
    return slice(k * N_LEVELS, (k + 1) * N_LEVELS)


UNITS: Dict[str, str] = {
    "z": "m2 s-2",  # geopotential, NOT height
    "q": "kg kg-1",
    "t": "K",
    "u": "m s-1",
    "v": "m s-1",
    "t2m": "K",
    "sp": "Pa",
    "tisr": "J m-2",  # 1 h accumulation, mean of 00/06/12/18 UTC (not a 24 h integral)
    "tcwv": "kg m-2",
    "orog": "m2 s-2",  # surface geopotential
    "msl": "Pa",
    "skt": "K",
}

# Static channel: copied from input to output, never generated (PG-CFM policy).
STATIC_CHANNELS = [CHANNEL_INDEX["orog"]]
DYNAMIC_CHANNELS = [i for i in range(N_CHANNELS) if i not in STATIC_CHANNELS]
N_DYNAMIC = len(DYNAMIC_CHANNELS)  # 66

# Known astronomical forcing. Kept as a generated channel in the baseline
# (KAI contract); the forcing-as-condition variant is an ablation.
FORCING_CHANNELS = [CHANNEL_INDEX["tisr"]]

# ----------------------------------------------------------------------
# Normalization statistics (KAI-New/stats, as printed in the format doc).
# The real .npy files take precedence; these are for synthetic data/tests.
# ----------------------------------------------------------------------
REFERENCE_MEAN = np.array([
    1.993e05, 1.576e05, 1.153e05, 1.012e05, 8.938e04, 6.995e04,
    5.409e04, 4.063e04, 2.891e04, 1.374e04, 7010.0, 735.7,
    2.678e-06, 2.636e-06, 1.939e-05, 5.764e-05, 0.0001271, 0.0003852,
    0.0008515, 0.001536, 0.002423, 0.004551, 0.005998, 0.006987,
    212.4, 208.4, 218.0, 222.7, 228.8, 242.0,
    252.8, 261.0, 267.2, 274.4, 277.2, 280.8,
    5.637, 10.21, 14.08, 13.22, 11.69, 8.736,
    6.502, 4.768, 3.311, 1.404, 0.6141, -0.03368,
    0.003459, 0.01258, -0.04655, -0.03181, -0.02469, -0.01919,
    -0.02439, -0.02637, 0.0213, 0.142, 0.2041, 0.1872,
    278.3, 9.657e04, 1.071e06, 18.23, 3791.0, 1.01e05, 278.6,
], dtype=np.float64)

REFERENCE_STD = np.array([
    5946.0, 5559.0, 5842.0, 5552.0, 5103.0, 4159.0,
    3359.0, 2699.0, 2135.0, 1462.0, 1216.0, 1055.0,
    3.621e-07, 5.566e-07, 2.157e-05, 6.997e-05, 0.0001585, 0.0004785,
    0.001023, 0.001688, 0.002446, 0.004006, 0.005017, 0.005877,
    10.32, 12.5, 7.11, 8.48, 10.71, 12.68,
    13.06, 13.43, 14.86, 15.64, 16.11, 17.16,
    15.16, 13.37, 17.28, 17.35, 16.38, 13.68,
    11.48, 9.918, 8.772, 7.726, 7.424, 5.744,
    6.751, 7.081, 11.04, 12.17, 11.98, 10.05,
    8.281, 7.083, 6.222, 5.566, 5.718, 4.761,
    21.33, 9598.0, 6.038e05, 16.2, 8342.0, 1308.0, 22.31,
], dtype=np.float64)

# KAI adds this to the std in the denominator (src/utils.py).
KAI_STD_EPS = 1e-8


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray, channel_axis: int = -3) -> np.ndarray:
    """KAI z-score, including its +1e-8 in the denominator."""
    shape = [1] * x.ndim
    shape[channel_axis] = -1
    return (x - mean.reshape(shape)) / (std.reshape(shape) + KAI_STD_EPS)


def denormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray, channel_axis: int = -3,
                exact_inverse: bool = True) -> np.ndarray:
    """Back to ERA5 units.

    KAI's inference uses ``x * std + mean`` (no eps), which is not the inverse
    of ``normalize``. ``exact_inverse=True`` uses ``std + eps`` instead.
    """
    shape = [1] * x.ndim
    shape[channel_axis] = -1
    s = std + KAI_STD_EPS if exact_inverse else std
    return x * s.reshape(shape) + mean.reshape(shape)


def below_ground_mask(sp: np.ndarray) -> np.ndarray:
    """True where a pressure level lies below the surface (p > sp).

    sp: surface pressure in Pa, shape (..., lat, lon).
    Returns shape (..., N_LEVELS, lat, lon). These points are ERA5
    extrapolations and are excluded from physics residuals.
    """
    p = LEVELS_PA.reshape((N_LEVELS,) + (1,) * 2)
    return p > sp[..., None, :, :]


# ----------------------------------------------------------------------
# Physical constants (SI)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Constants:
    earth_radius: float = 6.371e6  # a_E [m]
    omega: float = 7.292e-5  # [s-1]
    gravity: float = 9.80665  # [m s-2]
    r_dry: float = 287.05  # R_d [J kg-1 K-1]
    cp_dry: float = 1004.6  # c_p [J kg-1 K-1]
    dt_seconds: float = 86400.0  # one daily-mean transition


CONST = Constants()
