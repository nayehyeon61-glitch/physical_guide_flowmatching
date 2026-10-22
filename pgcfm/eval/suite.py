"""Run the whole NeuralGCM-style evaluation for a set of models and write tables + figures.

    from pgcfm.eval.suite import run_suite
    run_suite({"PG-CFM": ens, "PG-CFM eps=0": det}, truth, init_times, climatology, "out/")

``ens``: (N_ic, L, M, 67, H, W), ``det``: (N_ic, L, 67, H, W), physical units, leads 1..L.
``truth``: (N_ic, L+1, 67, H, W) with index 0 = analysis at init time.
Persistence and climatology baselines are added automatically.
"""

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

from . import baselines, physics, plots, scores
from .wb2 import LEAD, REALIZATION, select_headline, to_dataset

PHYS_VARS = ["z", "q", "t", "u", "v", "sp", "tcwv"]
TEND_VARS = ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind"]


def _wrap(x: np.ndarray, init_times, leads) -> xr.Dataset:
    member = x.ndim == 6
    return to_dataset(x, init_times, leads, member_axis=member)


def run_suite(
    forecasts: Dict[str, np.ndarray],
    truth: np.ndarray,
    init_times,
    climatology: Optional[xr.Dataset],
    out_dir: str,
    map_variable: str = "z500",
    map_lead: int = 3,
    map_model: Optional[str] = None,
    rank_variable: str = "z500",
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    leads = np.arange(1, truth.shape[1])
    truth_ds = to_dataset(truth[:, 1:], init_times, leads, member_axis=False)
    ic_full = to_dataset(truth[:, :1], init_times, [0], member_axis=False).isel({LEAD: 0}, drop=True).drop_vars("valid_time")
    truth_h = select_headline(truth_ds)
    clim_h = select_headline(climatology) if climatology is not None else None

    all_scores, rank, spec, tend, hydro, geo, drift, smallscale = {}, {}, {}, {}, {}, {}, {}, {}
    maps = None
    truth_phys = truth_ds
    hydro["truth"] = physics.hydrostatic_residual(truth_phys).mean(LEAD)
    geo["truth"] = physics.geostrophic_balance(truth_phys).mean(LEAD)
    spec["truth"] = physics.zonal_spectrum(truth_h["z500"].isel({LEAD: -1}))

    refs = {"persistence": baselines.persistence(select_headline(ic_full), truth_h)}
    if clim_h is not None:
        refs["climatology"] = baselines.climatology_forecast(clim_h, truth_h)
    for name, ds in refs.items():
        all_scores[name] = scores.score_forecast(ds, truth_h, clim_h)

    for name, arr in forecasts.items():
        fc = _wrap(arr, init_times, leads)
        fh = select_headline(fc)
        all_scores[name] = scores.score_forecast(fh, truth_h, clim_h)
        if scores.is_ensemble(fh):
            rank[name] = scores.rank_histogram(fh[[rank_variable]].isel({LEAD: [min(map_lead, len(leads)) - 1]}), truth_h[[rank_variable]].isel({LEAD: [min(map_lead, len(leads)) - 1]}))[rank_variable].squeeze(drop=True)
        if (map_model or name) == name and maps is None:
            sel = {LEAD: [min(map_lead, len(leads)) - 1]}
            maps = (name, {k: v[map_variable].isel({LEAD: 0}) for k, v in scores.spatial_maps(fh[[map_variable]].isel(sel), truth_h[[map_variable]].isel(sel)).items()})
        member = fh["z500"].isel({LEAD: -1})
        if REALIZATION in member.dims and member.sizes[REALIZATION] == 1:
            member = member.isel({REALIZATION: 0}, drop=True)
        spec[name] = physics.zonal_spectrum(member)
        spec[name] = spec[name].mean(REALIZATION) if REALIZATION in spec[name].dims else spec[name]
        smallscale[name] = physics.spectral_ratio(fh["z500"], truth_h["z500"])
        tr = physics.tendency_amplitude_ratio(fc[TEND_VARS], truth_ds[TEND_VARS], ic_full[TEND_VARS])
        tend[name] = tr.to_array().mean("variable")
        hydro[name] = physics.hydrostatic_residual(fc).mean(LEAD)
        geo[name] = physics.geostrophic_balance(fc).mean(LEAD)
        drift[name] = physics.budget_drift(fc, truth_ds)["mean_sp_Pa"]
        del fc, fh

    df = scores.to_frame(all_scores)
    df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    skill_c = scores.skill_vs(df, "climatology", "rmse") if clim_h is not None else None
    skill_p = scores.skill_vs(df, "persistence", "crps")

    phys_rows = []
    for m in forecasts:
        phys_rows.append(dict(
            model=m,
            tendency_ratio_lead1=float(tend[m].isel({LEAD: 0})),
            tendency_ratio_last=float(tend[m].isel({LEAD: -1})),
            hydrostatic_rms_mean=float(hydro[m].mean()),
            ageo_geo_ratio_500=float(geo[m].ratio.sel(level=500)),
            z500_smallscale_power_ratio=float(smallscale[m].mean()),
            sp_drift_last_Pa=float(drift[m].isel({LEAD: -1})),
        ))
    phys_rows.append(dict(model="truth", tendency_ratio_lead1=1.0, tendency_ratio_last=1.0,
                          hydrostatic_rms_mean=float(hydro["truth"].mean()),
                          ageo_geo_ratio_500=float(geo["truth"].ratio.sel(level=500)),
                          z500_smallscale_power_ratio=1.0, sp_drift_last_Pa=0.0))
    phys = pd.DataFrame(phys_rows)
    phys.to_csv(os.path.join(out_dir, "physics.csv"), index=False)

    model_order = ["persistence", "climatology"] + list(forecasts)
    variables = list(truth_h.data_vars)
    figs = [
        plots.score_lines(df, os.path.join(fig_dir, "fig1_scores_vs_lead.png"),
                          [v for v in ["z500", "t850", "q700", "u850", "t2m", "msl"] if v in variables],
                          metrics=("rmse", "crps", "ssr", "acc") if clim_h is not None else ("rmse", "crps", "ssr"),
                          models=[m for m in model_order if m in all_scores]),
        plots.scorecard(skill_p, "crps_skill_vs_persistence_pct", os.path.join(fig_dir, "fig2_scorecard_crps_vs_persistence.png"),
                        "CRPS skill vs persistence (%)", list(forecasts), variables),
    ]
    if skill_c is not None:
        figs.append(plots.scorecard(skill_c, "rmse_skill_vs_climatology_pct", os.path.join(fig_dir, "fig3_scorecard_rmse_vs_climatology.png"),
                                    "RMSE skill vs climatology (%)", list(forecasts), variables))
    if rank:
        figs.append(plots.rank_histograms(rank, os.path.join(fig_dir, "fig4_rank_histograms.png"),
                                          f"{rank_variable} rank histogram, lead {min(map_lead, len(leads))} d"))
    figs.append(plots.spectra(spec, os.path.join(fig_dir, "fig5_z500_spectra.png"),
                              f"z500 zonal spectrum, 30–60°, lead {len(leads)} d (per member)"))
    figs.append(plots.physics_panels(tend, hydro, geo, drift, os.path.join(fig_dir, "fig6_physical_consistency.png")))
    if maps:
        figs.append(plots.maps(maps[1], os.path.join(fig_dir, "fig7_maps.png"),
                               f"{maps[0]}: {map_variable} lead {min(map_lead, len(leads))} d"))
    summary = {"figures": [os.path.relpath(f, out_dir) for f in figs], "models": list(forecasts),
               "n_init": int(truth.shape[0]), "leads": leads.tolist()}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return dict(scores=df, physics=phys, skill_persistence=skill_p, skill_climatology=skill_c, summary=summary)
