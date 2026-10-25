"""Synthetic periodic tracer example; no trained climate/FМ model is assumed."""

import json
import math

import torch

from physical_guide_flowmatching import (
    ConservativeTransport2D,
    ODESolverConfig,
    PeriodicCartesianGrid,
    correct_forecast,
    integrate_ensemble,
)


def main():
    torch.set_num_threads(1)
    dtype = torch.float64
    height = width = 16
    x = torch.arange(width, dtype=dtype) * (2 * math.pi / width)
    y = torch.arange(height, dtype=dtype) * (2 * math.pi / height)
    pattern = torch.cos(x)[None, :] * torch.cos(y)[:, None]
    observed = (2 + 0.6 * pattern)[None, None]  # [B,C,H,W]
    # These deliberately biased synthetic endpoints stand in for FM predictions.
    # They are NOT output from a trained FM or observed climate data.
    amplitudes = torch.tensor([0.75, 0.9, 1.05], dtype=dtype).view(1, 3, 1, 1, 1)
    forecast = 2 + amplitudes * pattern
    duration_s = 3600.0
    times = torch.tensor([0.0, duration_s], dtype=dtype)
    grid = PeriodicCartesianGrid(dx_m=10000.0, dy_m=10000.0)
    diffusivity = 10000.0
    rhs = ConservativeTransport2D(grid, diffusivity_m2_s=diffusivity).double()
    config = ODESolverConfig(step_seconds=60.0)
    rate = 1 / (3 * duration_s)
    with torch.no_grad():
        corrected_path = correct_forecast(
            rhs, observed, forecast, times,
            nudging_rate_s=rate, config=config,
        )
        physical_path = integrate_ensemble(rhs, observed[:, None], times, config)

    # Exact solution of the SAME spatially discrete diffusion equation. This
    # isolates time integration from spatial truncation error.
    eigenvalue = diffusivity * (
        (2 * math.cos(2 * math.pi / width) - 2) / grid.dx_m**2
        + (2 * math.cos(2 * math.pi / height) - 2) / grid.dy_m**2
    )
    truth = 2 + 0.6 * math.exp(eigenvalue * duration_s) * pattern
    corrected = corrected_path[-1]
    mean_budget = observed[:, None].expand_as(forecast).mean(dim=(-2, -1))
    metrics = {
        "data": "synthetic periodic passive tracer; not a climate benchmark",
        "shape_B_E_C_H_W": list(corrected.shape),
        "fm_placeholder_rmse": float((forecast - truth).square().mean().sqrt()),
        "anchored_corrected_rmse": float((corrected - truth).square().mean().sqrt()),
        "physical_only_rmse": float((physical_path[-1] - truth).square().mean().sqrt()),
        "maximum_member_mean_drift": float(
            (corrected.mean(dim=(-2, -1)) - mean_budget).abs().max()
        ),
        "fm_placeholder_spread": float(forecast.std(dim=1, unbiased=False).mean()),
        "corrected_spread": float(corrected.std(dim=1, unbiased=False).mean()),
    }
    print(json.dumps(metrics, indent=2))
    if not metrics["anchored_corrected_rmse"] < metrics["fm_placeholder_rmse"]:
        raise RuntimeError("synthetic correction did not improve the deliberately biased endpoints")
    if metrics["maximum_member_mean_drift"] > 1e-10:
        raise RuntimeError("periodic tracer mean budget was not preserved")


if __name__ == "__main__":
    main()
