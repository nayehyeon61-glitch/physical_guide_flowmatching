"""Reference forecasts: persistence and day-of-year climatology (WB2 conventions)."""

import glob
import os
from typing import Optional, Sequence

import numpy as np
import xarray as xr

from .. import contract
from .wb2 import LEAD, WB2_NAMES

_WB2_UTILS = None


def _wb2_utils():
    global _WB2_UTILS
    if _WB2_UTILS is None:
        from weatherbench2 import utils  # path set up by pgcfm.eval.wb2
        _WB2_UTILS = utils
    return _WB2_UTILS


def persistence(initial: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """x̂(lead) = x(0) for every lead."""
    out = initial.expand_dims({LEAD: truth[LEAD]}).transpose("init_time", LEAD, ...)
    return out.assign_coords(valid_time=truth.valid_time)


def climatology_forecast(climatology: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """x̂(lead) = climatology at the valid day of year."""
    doy = truth.valid_time.dt.dayofyear
    out = climatology[list(truth.data_vars)].sel(dayofyear=doy).drop_vars("dayofyear")
    return out.transpose("init_time", LEAD, ...)


def kai_years_to_dataset(paths: Sequence[str], variables: Optional[Sequence[str]] = None) -> xr.Dataset:
    """KAI yearly ``fields(time, channel, lat, lon)`` files -> WB2 dataset with a ``time`` dim."""
    variables = list(variables) if variables is not None else contract.UPPER_AIR_VARS + contract.SURFACE_VARS
    parts = []
    for p in sorted(paths):
        with xr.open_dataset(p) as ds:
            f = ds["fields"].load()
            t = ds["time"].values
        data = {}
        for v in variables:
            if v in contract.UPPER_AIR_VARS:
                sl = contract.level_slice(v)
                data[WB2_NAMES[v]] = (("time", "level", "latitude", "longitude"), f.values[:, sl])
            else:
                data[WB2_NAMES[v]] = (("time", "latitude", "longitude"), f.values[:, contract.CHANNEL_INDEX[v]])
        coords = {"time": t, "latitude": contract.LATITUDES, "longitude": contract.LONGITUDES}
        if any(v in contract.UPPER_AIR_VARS for v in variables):
            coords["level"] = np.array(contract.LEVELS_HPA)
        parts.append(xr.Dataset(data, coords=coords))
    return xr.concat(parts, dim="time")


def daily_climatology(obs: xr.Dataset, window_size: int = 61, years: Optional[slice] = None) -> xr.Dataset:
    """WB2 ``compute_daily_stat_fast``: day-of-year mean, smoothed with a rolling window.

    WB2's ERA5 climatology uses 1990–2019 and a 61-day window; for KAI use the
    training years (1980–2011) so the test year 2020 stays out of sample.
    """
    years = years if years is not None else slice(None)
    return _wb2_utils().compute_daily_stat_fast(obs, window_size=window_size, clim_years=years, stat_fn="mean")


def climatology_from_kai_dir(data_dir: str, years: Sequence[int], out_path: str, window_size: int = 61) -> str:
    paths = [os.path.join(data_dir, f"{y}.nc") for y in years]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(missing)
    clim = daily_climatology(kai_years_to_dataset(paths), window_size)
    clim.to_netcdf(out_path)
    return out_path


def find_years(data_dir: str):
    return sorted(int(os.path.basename(p)[:4]) for p in glob.glob(os.path.join(data_dir, "[0-9]" * 4 + ".nc")))
