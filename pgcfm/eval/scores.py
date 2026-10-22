"""NeuralGCM Fig. 2 scores via WeatherBench2: RMSE, RMSB, ACC, CRPS, spread-skill.

All functions take WB2-style datasets (see ``wb2.to_dataset`` / ``select_headline``):
forecast with optional ``realization`` dim, truth without it, both on
``(init_time, prediction_timedelta, latitude, longitude)``. Averaging follows
WB2: spatial (area-weighted) mean per forecast, then mean over init_time, then
sqrt for RMS-type scores.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from .wb2 import LEAD, REALIZATION, metrics, regions

REGIONS: Dict[str, Optional[regions.Region]] = {
    "global": None,
    "extratropics": regions.ExtraTropicalRegion(threshold_lat=20),
    "tropics": regions.SliceRegion(lat_slice=slice(-20, 20)),
}


def is_ensemble(ds: xr.Dataset) -> bool:
    return REALIZATION in ds.dims and ds.sizes[REALIZATION] > 1


def _squeeze_members(ds: xr.Dataset) -> xr.Dataset:
    if REALIZATION in ds.dims and ds.sizes[REALIZATION] == 1:
        return ds.isel({REALIZATION: 0}, drop=True)
    return ds


def _spatial_rms(ds: xr.Dataset, region) -> xr.Dataset:
    return metrics._spatial_average_l2_norm(ds, region=region, skipna=False)


def score_forecast(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    climatology: Optional[xr.Dataset] = None,
    region_names: Optional[List[str]] = None,
    ssr_finite_ensemble_correction: bool = True,
) -> xr.Dataset:
    """All scalar scores for one model, as a dataset with a ``metric`` and ``region`` dim.

    Deterministic forecasts get CRPS = MAE (CRPS of a point forecast), spread 0.
    Ensembles additionally get member-mean RMSE (the score of a single member).
    """
    forecast = _squeeze_members(forecast)
    ens = is_ensemble(forecast)
    M = forecast.sizes[REALIZATION] if ens else 1
    mean = forecast.mean(REALIZATION) if ens else forecast
    out = []
    for rname in region_names or list(REGIONS):
        reg = REGIONS[rname]
        s = {}
        mse = metrics.MSE().compute(mean, truth, region=reg)
        s["rmse"] = np.sqrt(mse)
        # RMSB: time-mean bias map, then spatial RMS (NeuralGCM Fig. 2c).
        bias_map = metrics.SpatialBias().compute(mean, truth)  # already init_time-averaged
        s["rmsb"] = _spatial_rms(bias_map, reg)
        s["bias"] = metrics.Bias().compute(mean, truth, region=reg)
        if climatology is not None:
            s["acc"] = metrics.ACC(climatology=climatology).compute(mean, truth, region=reg)
        if ens:
            s["crps"] = metrics.CRPS().compute(forecast, truth, region=reg)
            var = metrics.EnsembleVariance().compute(forecast, truth, region=reg)
            s["spread"] = np.sqrt(var)
            corr = np.sqrt((M + 1) / M) if ssr_finite_ensemble_correction else 1.0
            s["ssr"] = corr * s["spread"] / s["rmse"]
            member_mse = metrics.MSE().compute(forecast, truth, region=reg).mean(REALIZATION)
            s["member_rmse"] = np.sqrt(member_mse)
        else:
            s["crps"] = metrics.MAE().compute(mean, truth, region=reg)
            s["spread"] = xr.zeros_like(s["rmse"])
            s["member_rmse"] = s["rmse"]
        out.append(xr.concat(list(s.values()), dim=pd.Index(list(s), name="metric")).expand_dims(region=[rname]))
    return xr.concat(out, dim="region")


def to_frame(scores: Dict[str, xr.Dataset]) -> pd.DataFrame:
    """{model: score_forecast(...)} -> long table (model, region, metric, variable, lead_days, value)."""
    rows = []
    for model, ds in scores.items():
        df = ds.to_dataframe().reset_index()
        keep = ["region", "metric", LEAD] + list(ds.data_vars)
        df = df[keep].melt(id_vars=["region", "metric", LEAD], var_name="variable", value_name="value")
        df["model"] = model
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df["lead_days"] = (df.pop(LEAD) / np.timedelta64(1, "D")).astype(int)
    return df[["model", "region", "metric", "variable", "lead_days", "value"]]


def skill_vs(df: pd.DataFrame, reference: str, metric: str = "rmse") -> pd.DataFrame:
    """NeuralGCM-style relative score in %: (ref - model) / ref * 100 (positive = better)."""
    piv = df[df.metric == metric].pivot_table(index=["region", "variable", "lead_days"], columns="model", values="value")
    ref = piv[reference]
    return ((ref.values[:, None] - piv) / ref.values[:, None] * 100).reset_index().melt(
        id_vars=["region", "variable", "lead_days"], var_name="model", value_name=f"{metric}_skill_vs_{reference}_pct")


def spatial_maps(forecast: xr.Dataset, truth: xr.Dataset) -> Dict[str, xr.Dataset]:
    """NeuralGCM Fig. 2b,d,f,h: RMSE, bias, CRPS, spread-skill maps averaged over init_time."""
    forecast = _squeeze_members(forecast)
    ens = is_ensemble(forecast)
    mean = forecast.mean(REALIZATION) if ens else forecast
    maps = {
        "rmse": np.sqrt(metrics.SpatialMSE().compute(mean, truth)),
        "bias": metrics.SpatialBias().compute(mean, truth),
    }
    if ens:
        M = forecast.sizes[REALIZATION]
        maps["crps"] = metrics.SpatialCRPS().compute(forecast, truth)
        var = metrics.SpatialEnsembleVariance().compute(forecast, truth)
        maps["ssr"] = np.sqrt((M + 1) / M) * np.sqrt(var) / maps["rmse"]
    return maps


def rank_histogram(forecast: xr.Dataset, truth: xr.Dataset, seed: int = 0) -> xr.Dataset:
    """WB2 rank histogram, averaged over init_time and space (area-weighted)."""
    rh = metrics.RankHistogram(seed=seed).compute(forecast, truth)
    return metrics._spatial_average(rh, region=None, skipna=False)
