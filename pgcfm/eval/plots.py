"""Figures for the evaluation report (NeuralGCM Fig. 2 / Ext. Data Fig. 3 layout).

Colors: categorical slots in fixed order per model; baselines (persistence,
climatology) are neutral gray dashes so they never take a model hue.
"""

import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BASELINE_STYLE = {
    "persistence": dict(color="#8a8984", ls="--"),
    "climatology": dict(color="#8a8984", ls=":"),
    "truth": dict(color=INK, ls="-"),
}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#1c5cab", "#6da7ec", "#f0efec", "#ec8a89", "#b8302f"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


def model_styles(models: Sequence[str]) -> Dict[str, dict]:
    styles, i = {}, 0
    for m in models:
        if m in BASELINE_STYLE:
            styles[m] = dict(BASELINE_STYLE[m], lw=1.6)
        else:
            styles[m] = dict(color=SERIES[i % len(SERIES)], ls="-", marker="o", ms=4)
            i += 1
    return styles


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


METRIC_LABEL = {
    "rmse": "RMSE (ens. mean)", "crps": "CRPS", "ssr": "spread-skill ratio", "acc": "ACC",
    "rmsb": "RMSB", "member_rmse": "RMSE (single member)", "spread": "spread",
}


def score_lines(df: pd.DataFrame, path: str, variables: List[str], metrics=("rmse", "crps", "ssr", "acc"),
                region: str = "global", models: Optional[List[str]] = None) -> str:
    """Grid of metric-vs-lead lines: rows = metrics, cols = variables (NeuralGCM Fig. 2a,c,e,g)."""
    d = df[df.region == region]
    models = models or list(dict.fromkeys(d.model))
    st = model_styles(models)
    fig, axes = plt.subplots(len(metrics), len(variables), figsize=(2.6 * len(variables), 2.1 * len(metrics)),
                             sharex=True, squeeze=False)
    for i, met in enumerate(metrics):
        for j, var in enumerate(variables):
            ax = axes[i, j]
            for m in models:
                s = d[(d.model == m) & (d.metric == met) & (d.variable == var)].sort_values("lead_days")
                if s.empty or (met == "ssr" and m in BASELINE_STYLE):
                    continue
                ax.plot(s.lead_days, s.value, label=m, **st[m])
            if met == "ssr":
                ax.axhline(1.0, color=INK2, lw=0.8)
            if i == 0:
                ax.set_title(var)
            if j == 0:
                ax.set_ylabel(METRIC_LABEL.get(met, met))
            if i == len(metrics) - 1:
                ax.set_xlabel("lead (days)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=min(len(lab), 8), bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    return _save(fig, path)


def scorecard(skill: pd.DataFrame, value_col: str, path: str, title: str, models: List[str], variables: List[str],
              region: str = "global") -> str:
    """Model x (variable, lead) heatmap of a relative skill in %, blue = better."""
    d = skill[(skill.region == region) & skill.model.isin(models) & skill.variable.isin(variables)]
    leads = sorted(d.lead_days.unique())
    cols = [(v, l) for v in variables for l in leads]
    grid = np.full((len(models), len(cols)), np.nan)
    for a, m in enumerate(models):
        for b, (v, l) in enumerate(cols):
            s = d[(d.model == m) & (d.variable == v) & (d.lead_days == l)][value_col]
            if len(s):
                grid[a, b] = s.iloc[0]
    lim = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
    fig, ax = plt.subplots(figsize=(0.32 * len(cols) + 2.2, 0.42 * len(models) + 1.4))
    ax.grid(False)
    im = ax.imshow(-grid, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for a in range(len(models)):
        for b in range(len(cols)):
            if np.isfinite(grid[a, b]):
                ax.text(b, a, f"{grid[a, b]:.0f}", ha="center", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(models)), models)
    ax.set_xticks(range(len(cols)), [str(l) for _, l in cols], fontsize=7)
    for k, v in enumerate(variables):
        ax.text(k * len(leads) + (len(leads) - 1) / 2, -0.6, v, ha="center", va="bottom", fontsize=8)
        if k:
            ax.axvline(k * len(leads) - 0.5, color=SURFACE, lw=2)
    ax.set_xlabel("lead (days)")
    ax.set_title(title, pad=30)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("blue = better than reference, red = worse")
    cb.set_ticks([])
    fig.tight_layout()
    return _save(fig, path)


def rank_histograms(hists: Dict[str, xr.DataArray], path: str, title: str) -> str:
    n = len(hists)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.0), sharey=True, squeeze=False)
    for ax, (name, h) in zip(axes[0], hists.items()):
        v = h.values / h.values.sum()
        ax.bar(np.arange(len(v)), v, color=SERIES[0], width=0.85)
        ax.axhline(1 / len(v), color=INK2, lw=0.8, ls="--")
        ax.set_title(name)
        ax.set_xlabel("rank of truth")
    axes[0, 0].set_ylabel("frequency")
    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    return _save(fig, path)


def spectra(specs: Dict[str, xr.DataArray], path: str, title: str) -> str:
    st = model_styles(list(specs))
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for name, s in specs.items():
        k = s.zonal_wavenumber.values
        keep = k > 0
        kw = dict(st[name])
        kw.pop("marker", None)
        ax.loglog(s.wavelength_km.values[keep], s.values[keep], label=name, **kw)
    ax.invert_xaxis()
    ax.set_xlabel("zonal wavelength at 45° (km)")
    ax.set_ylabel("energy (per zonal wavenumber)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def physics_panels(tend: Dict[str, xr.DataArray], hydro: Dict[str, xr.DataArray], geo: Dict[str, xr.Dataset],
                   drift: Dict[str, xr.DataArray], path: str) -> str:
    models = list(dict.fromkeys(list(tend) + list(hydro) + list(geo) + list(drift)))
    st = model_styles(models)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    ax = axes[0]
    for m, r in tend.items():
        ax.plot(r[_lead_dim(r)].values / np.timedelta64(1, "D"), r.values, label=m, **st[m])
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set(title="a. tendency amplitude ratio (z, t, u, v)", xlabel="lead (days)", ylabel="‖Δx̂‖ / ‖Δx‖")
    ax = axes[1]
    for m, r in hydro.items():
        kw = dict(st[m]); kw.pop("marker", None)
        ax.plot(r.values, r.layer.values, **kw)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set(title="b. hydrostatic residual RMS", xlabel="m² s⁻² (lead-mean, log)", ylabel="layer top (hPa)")
    ax = axes[2]
    for m, g in geo.items():
        kw = dict(st[m]); kw.pop("marker", None)
        ax.plot(g.ratio.values, g.level.values, **kw)
    ax.invert_yaxis()
    ax.set(title="c. |V_ageo| / |V_geo|, 20–70°", xlabel="ratio (lead-mean)", ylabel="level (hPa)")
    ax = axes[3]
    for m, d in drift.items():
        ax.plot(d[_lead_dim(d)].values / np.timedelta64(1, "D"), d.values, **st[m])
    ax.axhline(0.0, color=INK2, lw=0.8)
    ax.set(title="d. global-mean sp drift vs truth", xlabel="lead (days)", ylabel="Pa")
    h = [plt.Line2D([], [], **{k: v for k, v in st[m].items() if k != "marker"}) for m in models]
    fig.legend(h, models, loc="upper center", ncol=len(models), bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    return _save(fig, path)


def _lead_dim(da):
    return "prediction_timedelta"


def maps(m: Dict[str, xr.DataArray], path: str, title: str) -> str:
    """NeuralGCM Fig. 2b,d,f,h style maps (no coastlines; lat-lon grid)."""
    n = len(m)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 2.3), squeeze=False)
    for ax, (name, da) in zip(axes[0], m.items()):
        ax.grid(False)
        if name == "bias":
            lim = float(np.nanmax(np.abs(da.values))) or 1.0
            im = ax.pcolormesh(da.longitude, da.latitude, da.values, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
        elif name == "ssr":
            im = ax.pcolormesh(da.longitude, da.latitude, da.values, cmap=DIV, norm=TwoSlopeNorm(1.0, 0.0, 2.0), shading="auto")
        else:
            im = ax.pcolormesh(da.longitude, da.latitude, da.values, cmap=SEQ, shading="auto")
        ax.set_title(name)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([-60, 0, 60])
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    return _save(fig, path)
