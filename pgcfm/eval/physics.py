"""Physical-consistency diagnostics (NeuralGCM sec. "Spectra", "Geostrophic wind
balance", "Water budget") plus the PG-CFM-specific tendency amplitude ratio.

Geostrophic winds and zonal spectra use WeatherBench2's ``derived_variables``.
Inputs are full WB2 datasets (``wb2.to_dataset``) with a ``level`` dim.
"""

import numpy as np
import xarray as xr

from .. import contract
from .wb2 import LEAD, REALIZATION, derived_variables, metrics, regions

C = contract.CONST


def _members_mean(da: xr.DataArray) -> xr.DataArray:
    return da.mean(REALIZATION) if REALIZATION in da.dims else da


def _wmean(ds, region=None):
    return metrics._spatial_average(ds, region=region, skipna=False)


# ----------------------------------------------------------------------
# Tendency amplitude ratio (lessons_climate_diffusion.md, PR #2)
# ----------------------------------------------------------------------
def tendency_amplitude_ratio(forecast: xr.Dataset, truth: xr.Dataset, initial: xr.Dataset) -> xr.Dataset:
    """sqrt(E||x̂_k - x̂_{k-1}||²) / sqrt(E||x_k - x_{k-1}||²) per lead, per member then averaged.

    1 = the model moves as much per day as the atmosphere; << 1 = quasi-static.
    ``initial`` is the analysis at lead 0 (init_time, ..., lat, lon).
    """
    def increments(ds):
        ic = initial.expand_dims({LEAD: [np.timedelta64(0, "ns")]})
        if REALIZATION in ds.dims:
            ic = ic.expand_dims({REALIZATION: ds[REALIZATION]})
        full = xr.concat([ic, ds.drop_vars("valid_time", errors="ignore")], dim=LEAD, coords="minimal", compat="override")
        return full.diff(LEAD, label="upper")

    def ms(d):
        out = _wmean(d ** 2).mean("init_time")
        out = out.mean(REALIZATION) if REALIZATION in out.dims else out
        return out.mean("level") if "level" in out.dims else out

    return np.sqrt(ms(increments(forecast)) / ms(increments(truth)))


# ----------------------------------------------------------------------
# Hydrostatic balance (primitive equation 4 in NeuralGCM / PG-CFM eq. 8)
# ----------------------------------------------------------------------
def hydrostatic_residual(ds: xr.Dataset) -> xr.DataArray:
    """Area-weighted RMS of the discrete hypsometric residual per layer [m2 s-2].

    r_i = (Φ_i - Φ_{i+1}) - R_d * (Tv_i + Tv_{i+1})/2 * ln(p_{i+1}/p_i),
    Tv = T (1 + 0.608 q). Layers whose lower level lies below the surface (p > sp)
    are masked. Returns dims (..., layer) where layer = upper level in hPa.
    """
    z, T, q = ds["geopotential"], ds["temperature"], ds["specific_humidity"]
    Tv = T * (1 + 0.608 * q)
    lev = ds.level.values
    up, lo = lev[:-1], lev[1:]
    sel = lambda da, l: da.sel(level=l).rename(level="layer").assign_coords(layer=up)  # noqa: E731
    dz = sel(z, up) - sel(z, lo)
    tv = 0.5 * (sel(Tv, up) + sel(Tv, lo))
    log_ratio = xr.DataArray(np.log(lo / up), dims="layer", coords={"layer": up})
    r = dz - C.r_dry * tv * log_ratio
    if "surface_pressure" in ds:
        p_lo = xr.DataArray(lo * 100.0, dims="layer", coords={"layer": up})
        r = r.where(p_lo <= ds["surface_pressure"], 0.0)
    rms = np.sqrt(_wmean(xr.Dataset({"r": r ** 2}))["r"].mean("init_time"))
    return rms.mean(REALIZATION) if REALIZATION in rms.dims else rms


# ----------------------------------------------------------------------
# Geostrophic / ageostrophic wind (NeuralGCM Extended Data Fig. 3)
# ----------------------------------------------------------------------
EXTRATROPICS = regions.SliceRegion(lat_slice=[slice(-70, -20), slice(20, 70)])


def geostrophic_balance(ds: xr.Dataset) -> xr.Dataset:
    """Extratropical (20–70°) mean |V_geo|, |V_ageo| and |V_ageo|/|V_geo| per level."""
    geo = derived_variables.GeostrophicWindSpeed().compute(ds)
    ageo = derived_variables.AgeostrophicWindSpeed().compute(ds)
    out = xr.Dataset({"geostrophic": geo, "ageostrophic": ageo})
    m = _wmean(out, EXTRATROPICS).mean("init_time")
    m = m.mean(REALIZATION) if REALIZATION in m.dims else m
    return m.assign(ratio=m.ageostrophic / m.geostrophic)


# ----------------------------------------------------------------------
# Global budgets: dry mass (sp), water (tcwv), heat (mass-weighted T)
# ----------------------------------------------------------------------
def global_budgets(ds: xr.Dataset) -> xr.Dataset:
    out = {}
    if "surface_pressure" in ds:
        out["mean_sp_Pa"] = ds["surface_pressure"]
    if "total_column_water_vapour" in ds:
        out["mean_tcwv_kgm2"] = ds["total_column_water_vapour"]
    if "temperature" in ds:
        dp = xr.DataArray(np.gradient(ds.level.values * 100.0), dims="level", coords={"level": ds.level})
        out["mean_T_K"] = (ds["temperature"] * dp).sum("level") / dp.sum()
    g = _wmean(xr.Dataset(out))
    return g.mean(REALIZATION) if REALIZATION in g.dims else g


def budget_drift(forecast: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """Forecast minus truth global means, averaged over init_time (per lead)."""
    return (global_budgets(forecast) - global_budgets(truth)).mean("init_time")


# ----------------------------------------------------------------------
# Zonal energy spectra (NeuralGCM Supplementary Figs. 17–18)
# ----------------------------------------------------------------------
def zonal_spectrum(da: xr.DataArray, lat_band=(30.0, 60.0)) -> xr.DataArray:
    """WB2 zonal energy spectrum, averaged over |lat| in band and init_time.

    Kept on the zonal-wavenumber axis (not interpolated to common wavelengths);
    ``wavelength_km`` is reported at 45°.
    """
    name = da.name or "x"
    spec = derived_variables.ZonalEnergySpectrum(name).compute(da.to_dataset(name=name))
    lat = spec.latitude
    band = (np.abs(lat) >= lat_band[0]) & (np.abs(lat) <= lat_band[1])
    s = spec.where(band, drop=True).mean(["latitude", "init_time"])
    s = s.drop_vars(["frequency", "wavelength"], errors="ignore")
    k = s.zonal_wavenumber.values
    with np.errstate(divide="ignore"):
        wl = np.where(k > 0, 2 * np.pi * C.earth_radius * np.cos(np.deg2rad(45)) / np.maximum(k, 1) / 1e3, np.inf)
    return s.assign_coords(wavelength_km=("zonal_wavenumber", wl))


def spectral_ratio(forecast: xr.DataArray, truth: xr.DataArray, band_km=(1000.0, 3000.0)) -> xr.DataArray:
    """Geometric-mean ratio of forecast to truth power for wavelengths in ``band_km``.

    For ensembles the per-member spectrum is used (an ensemble mean is blurry by
    construction). 1 = right small-scale energy, < 1 = blurry, > 1 = noisy.
    The band stays above the grid scale (1.5° ≈ 236 km Nyquist at 45°) so the
    ratio is not dominated by the round-off floor.
    """
    fs = zonal_spectrum(forecast)
    fs = fs.mean(REALIZATION) if REALIZATION in fs.dims else fs
    ts = zonal_spectrum(truth)
    inband = (ts.wavelength_km >= band_km[0]) & (ts.wavelength_km <= band_km[1])
    return np.exp(np.log(fs / ts).where(inband, drop=True).mean("zonal_wavenumber"))
