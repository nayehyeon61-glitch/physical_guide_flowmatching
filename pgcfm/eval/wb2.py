"""KAI channel tensors <-> WeatherBench2 xarray datasets.

WB2 expects one data variable per field with dims
``(init_time, prediction_timedelta, [realization], [level], latitude, longitude)``,
latitude ascending, physical units. KAI stores ``(..., channel, lat, lon)`` with
67 channels in ``pgcfm.contract`` order. Everything here is a reshape plus names;
no values change.
"""

import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np
import xarray as xr

from .. import contract

_WB2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "weatherbench2"))
if _WB2_ROOT not in sys.path:
    sys.path.insert(0, _WB2_ROOT)

from weatherbench2 import derived_variables, metrics, regions  # noqa: E402,F401

# KAI short name -> WB2/ERA5 long name.
WB2_NAMES: Dict[str, str] = {
    "z": "geopotential",
    "q": "specific_humidity",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t2m": "2m_temperature",
    "sp": "surface_pressure",
    "tisr": "toa_incident_solar_radiation",
    "tcwv": "total_column_water_vapour",
    "orog": "geopotential_at_surface",
    "msl": "mean_sea_level_pressure",
    "skt": "skin_temperature",
}
KAI_NAMES = {v: k for k, v in WB2_NAMES.items()}

# NeuralGCM Fig. 2 / WB2 headline set, written as (variable, level or None).
HEADLINE = [
    ("geopotential", 500),
    ("temperature", 850),
    ("specific_humidity", 700),
    ("u_component_of_wind", 850),
    ("v_component_of_wind", 850),
    ("2m_temperature", None),
    ("mean_sea_level_pressure", None),
]

REALIZATION = metrics.REALIZATION
LEAD = "prediction_timedelta"


def headline_label(var: str, level: Optional[int]) -> str:
    short = KAI_NAMES.get(var, var)
    return f"{short}{level}" if level is not None else short


def to_dataset(
    x: np.ndarray,
    init_times: Sequence[np.datetime64],
    lead_days: Sequence[int],
    member_axis: bool,
    variables: Optional[Sequence[str]] = None,
) -> xr.Dataset:
    """Wrap a KAI-ordered array as a WB2 dataset.

    x: (N_ic, L, [M], 67, H, W) in physical units.
    variables: KAI short names to keep (default: all).
    Adds a ``valid_time`` coordinate (init + lead) for WB2's ACC.
    """
    variables = list(variables) if variables is not None else contract.UPPER_AIR_VARS + contract.SURFACE_VARS
    init = np.asarray(init_times, dtype="datetime64[ns]")
    lead = np.asarray(lead_days, dtype=np.int64).astype("timedelta64[D]").astype("timedelta64[ns]")
    dims = ["init_time", LEAD] + ([REALIZATION] if member_axis else [])
    coords = {
        "init_time": init,
        LEAD: lead,
        "latitude": contract.LATITUDES[: x.shape[-2]] if x.shape[-2] == contract.N_LAT else np.linspace(-90, 90, x.shape[-2]),
        "longitude": np.arange(x.shape[-1]) * (360.0 / x.shape[-1]),
    }
    if member_axis:
        coords[REALIZATION] = np.arange(x.shape[2])
    c_axis = len(dims)
    data_vars = {}
    for v in variables:
        if v in contract.UPPER_AIR_VARS:
            sl = contract.level_slice(v)
            data_vars[WB2_NAMES[v]] = (dims + ["level", "latitude", "longitude"], np.take(x, range(sl.start, sl.stop), axis=c_axis))
        else:
            data_vars[WB2_NAMES[v]] = (dims + ["latitude", "longitude"], np.take(x, contract.CHANNEL_INDEX[v], axis=c_axis))
    if any(v in contract.UPPER_AIR_VARS for v in variables):
        coords["level"] = np.array(contract.LEVELS_HPA)
    ds = xr.Dataset(data_vars, coords=coords)
    return ds.assign_coords(valid_time=ds.init_time + ds[LEAD])


def select_headline(ds: xr.Dataset, headline=HEADLINE) -> xr.Dataset:
    """One flat 2-D variable per headline entry, named like ``z500``."""
    out = {}
    for var, level in headline:
        if var not in ds:
            continue
        da = ds[var] if level is None else ds[var].sel(level=level, drop=True)
        out[headline_label(var, level)] = da
    return xr.Dataset(out)


def from_kai_forecast_nc(path: str) -> Dict[str, np.ndarray]:
    """Read a KAI ``forecast.nc`` (src/inference.py) or a PG-CFM ensemble file.

    KAI layout: forecast(initial_condition, lead_time, channel, lat, lon), lead 0 = IC,
    de-normalized. PG-CFM ensembles add a ``member`` dim after ``lead_time``.
    Returns dict with ``forecast`` (N, L, [M], C, H, W), ``ic_index``, ``lead``, ``channel``.
    """
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        v = ds["forecast"]
        out = {
            "forecast": np.asarray(v[:], dtype=np.float32),
            "ic_index": np.asarray(ds["initial_condition"][:]),
            "lead": np.asarray(ds["lead_time"][:]),
            "channel": np.asarray(ds["channel"][:]),
            "has_members": "member" in v.dimensions,
        }
    return out
