"""ERA5/WeatherBench tensor conventions matching ClimODE's global variables.

No downloads, regridding, height-to-geopotential conversion or unit inference
are performed. Read NetCDF coordinates/units explicitly before calling this.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from .spherical import GlobalLatLonGrid


CLIMODE_VARIABLES = ("z500", "t850", "t2m", "u10", "v10")
ERA5_UNITS = {"z500": "m^2/s^2", "t850": "K", "t2m": "K", "u10": "m/s", "v10": "m/s"}


def prepare_era5_state(
    fields: Mapping[str, Tensor],
    latitude_degrees: Tensor,
    longitude_degrees: Tensor,
    *,
    grid: GlobalLatLonGrid | None = None,
    variables: Sequence[str] = CLIMODE_VARIABLES,
) -> Tensor:
    """Align coordinates AND data, then stack physical fields as [...,C,H,W].

    Accepts descending latitude or longitude in [-180,180), and explicitly
    reorders every supplied field to south-to-north / [0,360). It rejects a
    different grid, duplicate longitude endpoints and missing channels. This
    is coordinate reindexing only, never interpolation/regridding.

    Input keys are canonical aliases (z500/t850/t2m/u10/v10); select the pressure
    levels and rename NetCDF variables explicitly. z500 means geopotential,
    not geopotential height. Fit any normalization statistics on training only.
    """
    grid = grid or GlobalLatLonGrid()
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("variables must be a nonempty list of distinct channel names")
    latitude = torch.as_tensor(latitude_degrees, dtype=torch.float64)
    longitude = torch.as_tensor(longitude_degrees, dtype=torch.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("latitude and longitude must be one-dimensional coordinates")
    if not torch.isfinite(latitude).all() or not torch.isfinite(longitude).all():
        raise ValueError("coordinates must be finite")
    latitude, lat_order = latitude.sort()
    longitude, lon_order = longitude.remainder(360).sort()
    grid.validate_coordinates(latitude, longitude)
    aligned = []
    first = None
    for name in variables:
        if name not in fields:
            raise ValueError(f"missing ERA5 channel: {name}")
        field = fields[name]
        if not isinstance(field, Tensor) or field.ndim < 2:
            raise ValueError(f"{name} must be a tensor with shape [...,latitude,longitude]")
        if field.shape[-2:] != (grid.n_lat, grid.n_lon):
            raise ValueError(f"{name} does not match the configured grid")
        if field.dtype not in (torch.float32, torch.float64) or not torch.isfinite(field).all():
            raise ValueError(f"{name} must have finite float32 or float64 values")
        if first is not None and (
            field.shape != first.shape or field.dtype != first.dtype or field.device != first.device
        ):
            raise ValueError("all ERA5 channels must have matching shapes, dtypes and devices")
        first = field
        field = field.index_select(-2, lat_order.to(field.device))
        field = field.index_select(-1, lon_order.to(field.device))
        aligned.append(field)
    return torch.stack(aligned, dim=-3)
