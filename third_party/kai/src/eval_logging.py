"""
Copyright (c) 2025 MetaEarth-Lab
Licensed under the BSD 3-Clause License. Redistribution and use in source and binary forms, with or without modification, are permitted. See the LICENSE file in the project root for the full license text.

Weights & Biases logging helpers for the post-training WeatherBench evaluation.

This module is intentionally free of any dependency on the ``Trainer`` / ``PostTrainingEvaluator``
so it can be reused and unit-tested in isolation. It turns the ``{"rmse", "acc"}`` metric matrices
(shape ``[lead_time, n_variables]``) produced by the evaluator into:

  * ``wandb.Table`` objects for the raw RMSE and ACC results,
  * per-variable metric-grid figures (one small panel per output channel),
  * aggregate RMSE / ACC vs. lead-time line plots,

and pushes them to the active W&B run. Every figure is also written to
``<output_dir>/eval_plots/`` so the plots survive even when W&B is offline / disabled.
"""

import math
import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend — safe on compute nodes without a display
import matplotlib.pyplot as plt
import numpy as np


def _grid_shape(n: int) -> "tuple[int, int]":
    """Return a near-square ``(nrows, ncols)`` grid that fits ``n`` panels."""
    ncols = max(1, int(math.ceil(math.sqrt(n))))
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def _metric_grid_figure(
    metric: np.ndarray,
    variable_names: Sequence[str],
    ylabel: str,
    title: str,
):
    """Build a multi-panel figure with one small line plot per variable.

    Args:
        metric: Array of shape ``[lead_time, n_variables]``.
        variable_names: Human-readable name for each variable/column.
        ylabel: Y-axis label (e.g. ``"RMSE"`` or ``"ACC"``).
        title: Figure suptitle.

    Returns:
        The created ``matplotlib`` figure.
    """
    n_lead, n_vars = metric.shape
    leads = np.arange(n_lead)
    nrows, ncols = _grid_shape(n_vars)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.4 * ncols, 1.9 * nrows),
        squeeze=False,
    )
    for v in range(nrows * ncols):
        ax = axes[v // ncols][v % ncols]
        if v < n_vars:
            ax.plot(leads, metric[:, v], marker="o", markersize=2.5, linewidth=1.0)
            ax.set_title(str(variable_names[v]), fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis("off")

    fig.supxlabel("Lead time (days)", fontsize=9)
    fig.supylabel(ylabel, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.97))
    return fig


def _aggregate_line_figure(
    metric: np.ndarray,
    variable_names: Sequence[str],
    highlight_indices: Sequence[int],
    ylabel: str,
    title: str,
):
    """Build an aggregate line plot of a metric across all lead times.

    Plots the mean over all variables plus, overlaid, the individual curves for the
    variables in ``highlight_indices`` (typically ``eval_variable_indices``).
    """
    n_lead, n_vars = metric.shape
    leads = np.arange(n_lead)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        leads,
        metric.mean(axis=1),
        color="black",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="mean (all variables)",
    )
    for idx in highlight_indices:
        if 0 <= idx < n_vars:
            ax.plot(
                leads,
                metric[:, idx],
                linewidth=1.3,
                marker=".",
                markersize=6,
                alpha=0.9,
                label=str(variable_names[idx]),
            )

    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def _metric_table(metric: np.ndarray, variable_names: Sequence[str], wandb):
    """Build a ``wandb.Table`` with rows = lead time and columns = variable names."""
    n_lead, n_vars = metric.shape
    columns = ["lead_time"] + [str(variable_names[v]) for v in range(n_vars)]
    data = [
        [int(t)] + [float(metric[t, v]) for v in range(n_vars)]
        for t in range(n_lead)
    ]
    return wandb.Table(columns=columns, data=data)


def _safe_filename(name) -> str:
    """Turn a variable name into a filesystem-safe filename stem.

    Args:
        name: Arbitrary variable/channel name.

    Returns:
        The name with path-hostile characters replaced by underscores.
    """
    stem = str(name).strip()
    for ch in ("/", "\\", " ", ":", "*", "?", '"', "<", ">", "|"):
        stem = stem.replace(ch, "_")
    return stem or "var"


def _save_individual_variable_plots(
    metric: np.ndarray,
    variable_names: Sequence[str],
    ylabel: str,
    plots_dir: str,
) -> int:
    """Save one line plot per variable (metric vs lead time) into ``plots_dir``.

    Args:
        metric: Array of shape ``[lead_time, n_variables]``.
        variable_names: Human-readable name for each variable/column.
        ylabel: Y-axis label (e.g. ``"RMSE"`` or ``"ACC"``).
        plots_dir: Directory to write the per-variable PNGs into (created if needed).

    Returns:
        The number of PNG files written (one per variable).
    """
    os.makedirs(plots_dir, exist_ok=True)
    n_lead, n_vars = metric.shape
    leads = np.arange(n_lead)
    for v in range(n_vars):
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.plot(leads, metric[:, v], marker="o", markersize=3, linewidth=1.2)
        ax.set_title(str(variable_names[v]), fontsize=9)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            os.path.join(plots_dir, f"{_safe_filename(variable_names[v])}.png"),
            dpi=120,
            bbox_inches="tight",
        )
        plt.close(fig)
    return n_vars


def _log_native_line_charts(
    wandb,
    rmse: np.ndarray,
    acc: np.ndarray,
    variable_names: Sequence[str],
    highlight: Sequence[int],
    logger,
) -> None:
    """Log RMSE / ACC as native W&B line charts for cross-model comparison.

    Two complementary views are produced:
      * per-lead scalar series (``eval/rmse_mean``, ``eval/acc_mean`` and, for each
        highlighted variable, ``eval/rmse/<name>`` / ``eval/acc/<name>``) bound to a
        custom ``eval/lead_time`` step metric, which W&B overlays across runs;
      * ``wandb.plot.line_series`` panels drawing every variable's curve in one chart.

    Args:
        wandb: The imported ``wandb`` module (an active run is assumed).
        rmse: RMSE matrix of shape ``[lead_time, n_variables]``.
        acc: ACC matrix of shape ``[lead_time, n_variables]``.
        variable_names: Human-readable name for each variable/column.
        highlight: Variable indices to expose as their own comparable series.
        logger: Logger used for status messages.

    Returns:
        None: Series and charts are pushed to the active W&B run.
    """
    n_lead, n_vars = rmse.shape
    leads = list(range(n_lead))

    # Per-lead comparable series bound to a custom lead-time step metric.
    wandb.define_metric("eval/lead_time")
    for key in ("eval/rmse_mean", "eval/acc_mean"):
        wandb.define_metric(key, step_metric="eval/lead_time")
    for idx in highlight:
        name = str(variable_names[idx])
        wandb.define_metric(f"eval/rmse/{name}", step_metric="eval/lead_time")
        wandb.define_metric(f"eval/acc/{name}", step_metric="eval/lead_time")

    for t in range(n_lead):
        row = {
            "eval/lead_time": t,
            "eval/rmse_mean": float(rmse[t].mean()),
            "eval/acc_mean": float(acc[t].mean()),
        }
        for idx in highlight:
            name = str(variable_names[idx])
            row[f"eval/rmse/{name}"] = float(rmse[t, idx])
            row[f"eval/acc/{name}"] = float(acc[t, idx])
        wandb.log(row)

    # Multi-line panels: every variable's curve in a single interactive chart.
    keys = [str(v) for v in variable_names]
    wandb.log(
        {
            "eval/rmse_line_series": wandb.plot.line_series(
                xs=leads,
                ys=[rmse[:, v].tolist() for v in range(n_vars)],
                keys=keys,
                title="RMSE per variable vs lead time",
                xname="lead_time",
            ),
            "eval/acc_line_series": wandb.plot.line_series(
                xs=leads,
                ys=[acc[:, v].tolist() for v in range(n_vars)],
                keys=keys,
                title="ACC per variable vs lead time",
                xname="lead_time",
            ),
        }
    )
    logger.info("Logged native W&B line charts (per-lead series + multi-variable panels).")


def log_eval_to_wandb(
    metrics: Dict[str, np.ndarray],
    variable_names: Sequence[str],
    out_channels: Sequence[int],
    eval_variable_indices: Optional[Sequence[int]],
    output_dir: str,
    logger,
) -> Dict[str, float]:
    """Log the post-training evaluation to Weights & Biases.

    Logs three things (per the project requirements):
      * **RMSE / ACC results** as ``wandb.Table`` objects and run-summary scalars,
      * **figures of all variables** — per-variable RMSE and ACC metric grids,
      * **line plots of RMSE and ACC across all timesteps** — aggregate curves.

    ``metrics['rmse']`` / ``metrics['acc']`` are ``[lead_time, n_variables]`` with row 0 being
    the day-0 initial condition. Figures are always written to ``<output_dir>/eval_plots/``;
    W&B logging is skipped (with a warning) when no active run is present. Returns the headline
    scalar dict (also useful to callers / tests).
    """
    try:
        import wandb
    except Exception:  # pragma: no cover - wandb should be installed, but stay defensive
        wandb = None

    rmse = np.asarray(metrics["rmse"])
    acc = np.asarray(metrics["acc"])
    n_lead, n_vars = rmse.shape
    highlight = [int(i) for i in (eval_variable_indices or []) if 0 <= int(i) < n_vars]
    final_lead = n_lead - 1

    # ------------------------------------------------------------------
    # Figures — always rendered and saved to disk
    # ------------------------------------------------------------------
    plots_dir = os.path.join(output_dir, "eval_plots")
    os.makedirs(plots_dir, exist_ok=True)

    figures = {
        "eval/rmse_all_variables": (
            _metric_grid_figure(rmse, variable_names, "RMSE", "RMSE per variable vs lead time"),
            "rmse_all_variables.png",
        ),
        "eval/acc_all_variables": (
            _metric_grid_figure(acc, variable_names, "ACC", "ACC per variable vs lead time"),
            "acc_all_variables.png",
        ),
        "eval/rmse_vs_leadtime": (
            _aggregate_line_figure(rmse, variable_names, highlight, "RMSE", "RMSE vs lead time"),
            "rmse_vs_leadtime.png",
        ),
        "eval/acc_vs_leadtime": (
            _aggregate_line_figure(acc, variable_names, highlight, "ACC", "ACC vs lead time"),
            "acc_vs_leadtime.png",
        ),
    }
    for _, (fig, fname) in figures.items():
        fig.savefig(os.path.join(plots_dir, fname), dpi=120, bbox_inches="tight")
    logger.info(f"Saved evaluation figures to: {plots_dir}")

    # Individual per-variable plots — one PNG per output channel, grouped by metric.
    per_var_dir = os.path.join(plots_dir, "per_variable")
    n_rmse = _save_individual_variable_plots(rmse, variable_names, "RMSE", os.path.join(per_var_dir, "rmse"))
    n_acc = _save_individual_variable_plots(acc, variable_names, "ACC", os.path.join(per_var_dir, "acc"))
    logger.info(f"Saved {n_rmse} per-variable RMSE and {n_acc} per-variable ACC plots under: {per_var_dir}")

    # ------------------------------------------------------------------
    # Headline scalars (also returned for callers / tests)
    # ------------------------------------------------------------------
    scalars: Dict[str, float] = {
        "eval/final_lead_mean_rmse": float(rmse[final_lead].mean()),
        "eval/final_lead_mean_acc": float(acc[final_lead].mean()),
    }
    for idx in highlight:
        name = str(variable_names[idx])
        scalars[f"eval/final_lead_rmse_{name}"] = float(rmse[final_lead, idx])
        scalars[f"eval/final_lead_acc_{name}"] = float(acc[final_lead, idx])

    # ------------------------------------------------------------------
    # Push to W&B (only if a run is active)
    # ------------------------------------------------------------------
    if wandb is None or getattr(wandb, "run", None) is None:
        logger.warning("No active W&B run — evaluation figures saved to disk only.")
        for fig, _ in figures.values():
            plt.close(fig)
        return scalars

    payload: Dict[str, object] = dict(scalars)
    payload["eval/rmse_table"] = _metric_table(rmse, variable_names, wandb)
    payload["eval/acc_table"] = _metric_table(acc, variable_names, wandb)
    for key, (fig, _) in figures.items():
        payload[key] = wandb.Image(fig)

    wandb.log(payload)
    for key, value in scalars.items():
        wandb.run.summary[key] = value

    # Native line charts (comparable across runs / models).
    _log_native_line_charts(wandb, rmse, acc, variable_names, highlight, logger)

    for fig, _ in figures.values():
        plt.close(fig)

    logger.info("Logged evaluation figures, tables, scalars, and line charts to W&B.")
    return scalars


def log_loss_curves(
    train_hist: Sequence[float],
    valid_hist: Sequence[float],
    output_dir: str,
    logger,
    log_to_wandb: bool = True,
) -> str:
    """Save and log training / validation loss curves at the end of a run.

    Writes ``loss_curve.png`` and ``loss_history.csv`` under ``output_dir`` and,
    when a W&B run is active, logs the curve both as a native
    ``wandb.plot.line_series`` (train vs valid, comparable across models) and as a
    static image fallback.

    Args:
        train_hist: Per-epoch mean training loss values.
        valid_hist: Per-epoch validation loss values (may be empty).
        output_dir: Directory to write the PNG / CSV into (created if needed).
        logger: Logger used for status messages.
        log_to_wandb: Whether to attempt pushing the curve to the active W&B run.

    Returns:
        The path to the saved ``loss_curve.png``.
    """
    epochs = list(range(1, len(train_hist) + 1))

    # Figure
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(epochs, list(train_hist), marker="o", markersize=3, linewidth=1.6, label="train")
    if valid_hist:
        ax.plot(epochs, list(valid_hist), marker="o", markersize=3, linewidth=1.6, label="valid")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    logger.info(f"Saved loss curve to: {png_path}")

    # CSV (written without pandas to keep this module dependency-light)
    csv_path = os.path.join(output_dir, "loss_history.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,valid_loss\n")
        for i, ep in enumerate(epochs):
            tv = valid_hist[i] if i < len(valid_hist) else float("nan")
            f.write(f"{ep},{train_hist[i]},{tv}\n")
    logger.info(f"Saved loss history to: {csv_path}")

    # W&B — native line plot + image fallback
    if log_to_wandb:
        try:
            import wandb
        except Exception:  # pragma: no cover - stay defensive
            wandb = None
        if wandb is not None and getattr(wandb, "run", None) is not None:
            ys = [[float(x) for x in train_hist]]
            keys = ["train"]
            if valid_hist:
                ys.append([float(x) for x in valid_hist])
                keys.append("valid")
            wandb.log(
                {
                    "loss/curve": wandb.plot.line_series(
                        xs=epochs, ys=ys, keys=keys,
                        title="Training / validation loss", xname="epoch",
                    ),
                    "loss/curve_image": wandb.Image(fig),
                }
            )
            logger.info("Logged loss curves to W&B.")
        else:
            logger.warning("No active W&B run — loss curves saved to disk only.")

    plt.close(fig)
    return png_path
