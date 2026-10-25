"""Validate 15-day integration on the ERA5 grid with manufactured scalar fields.

No ERA5 files or trained climate model are used. The five channels use ERA5-like
units, but a shared prescribed solid-body flow is a numerical test, not valid
multi-level atmospheric dynamics. Compare to the exact spatially discrete
solution to isolate time-integration error from upwind's numerical diffusion.
"""

import json
import math

import torch

from physical_guide_flowmatching import (
    CLIMODE_VARIABLES, GlobalLatLonGrid, SphericalTransport2D,
    ODESolverConfig, area_weighted_mean, forecast_times_seconds, integrate_windowed,
)


def main():
    torch.set_num_threads(1)
    grid = GlobalLatLonGrid()
    latitude = torch.deg2rad(grid.latitude_degrees)
    longitude = torch.deg2rad(grid.longitude_degrees)
    phase = torch.tensor([0.0, 0.2], dtype=torch.float64).reshape(1, 2, 1, 1, 1)
    offset = torch.tensor([55000, 275, 285, 0, 0], dtype=torch.float64).reshape(1, 1, 5, 1, 1)
    amplitude = torch.tensor([100, 5, 8, 4, 2], dtype=torch.float64).reshape(1, 1, 5, 1, 1)
    pattern = latitude.cos().reshape(1, 1, 1, 32, 1) * torch.cos(
        longitude.reshape(1, 1, 1, 1, 64) + phase
    )
    initial = offset + amplitude * pattern
    equatorial_speed_m_s = 20.0
    u = (equatorial_speed_m_s * latitude.cos()).reshape(1, 1, 1, 32, 1)
    rhs = SphericalTransport2D(grid, u_m_s=u, transport_form="advective").double()
    times = forecast_times_seconds(reference=initial)
    step = min(1800.0, rhs.explicit_step_limit_seconds())
    with torch.no_grad():
        path = integrate_windowed(rhs, initial, times, ODESolverConfig(step_seconds=step))
        fine = integrate_windowed(rhs, initial, times, ODESolverConfig(step_seconds=step / 2))
    # For this solid-body flow the zonal finite-volume rate is independent of
    # longitude. Use the exact row-specific geometric rate for the discrete mode.
    delta_lon = 2 * math.pi / grid.n_lon
    east_face_length = grid.radius_m * math.pi / grid.n_lat
    rate = equatorial_speed_m_s * latitude.cos() * east_face_length / grid.cell_area_m2[:, 0]
    duration = times[-1]
    damping = torch.exp(rate * (math.cos(delta_lon) - 1) * duration).reshape(1, 1, 1, 32, 1)
    rotation = (rate * math.sin(delta_lon) * duration).reshape(1, 1, 1, 32, 1)
    exact = offset + amplitude * latitude.cos().reshape(1, 1, 1, 32, 1) * damping * torch.cos(
        longitude.reshape(1, 1, 1, 1, 64) + phase - rotation
    )
    scale = amplitude.abs()
    relative_error = ((path[-1] - exact) / scale).abs().max()
    half_step_difference = ((path - fine) / scale[None]).abs().max()
    budget_drift = (area_weighted_mean(path, grid) - area_weighted_mean(initial, grid)[None]).abs().max()
    metrics = {
        "data": "manufactured scalar transport, no observed ERA5 or trained climate checkpoint",
        "channels": list(CLIMODE_VARIABLES),
        "trajectory_shape_T_B_E_C_H_W": list(path.shape),
        "horizon_days": float(times[-1] / 86400),
        "output_interval_hours": 6,
        "internal_step_seconds": step,
        "maximum_scaled_time_integration_error": float(relative_error),
        "maximum_scaled_half_step_difference": float(half_step_difference),
        "maximum_area_weighted_mean_drift": float(budget_drift),
        "finite": bool(torch.isfinite(path).all()),
    }
    print(json.dumps(metrics, indent=2))
    if not metrics["finite"] or relative_error > 1e-6 or half_step_difference > 1e-6:
        raise RuntimeError("15-day manufactured integration failed accuracy/convergence checks")


if __name__ == "__main__":
    main()
