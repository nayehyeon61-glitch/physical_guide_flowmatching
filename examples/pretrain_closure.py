"""Fit a closure through short ODE trajectories on manufactured diffusion data.

This verifies an optimization path, not climate generalization. No climate
checkpoint or real-data performance is claimed. All quantities use synthetic
metre/second scales chosen to keep this CPU example small.
"""

import argparse
import json
import math

import torch

from physical_guide_flowmatching import (
    ConservativeFluxClosure2D,
    ConservativeTransport2D,
    ODESolverConfig,
    PeriodicCartesianGrid,
    integrate_ensemble,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.manual_seed(7)
    torch.set_num_threads(1)
    dtype = torch.float64
    size = 8
    x = torch.arange(size, dtype=dtype) * (2 * math.pi / size)
    y = x[:, None]
    initial = torch.stack([
        2 + 0.5 * torch.cos(x)[None, :] * torch.cos(y),
        2 + 0.3 * torch.sin(2 * x)[None, :].expand(size, size),
        2 + 0.4 * torch.sin(x)[None, :] * torch.cos(y),
    ]).reshape(3, 1, 1, size, size)
    times = torch.tensor([0.0, 0.25, 0.5], dtype=dtype)
    config = ODESolverConfig(step_seconds=0.05)
    grid = PeriodicCartesianGrid(dx_m=1.0, dy_m=1.0)
    true_rhs = ConservativeTransport2D(grid, diffusivity_m2_s=0.3).double()
    base_rhs = ConservativeTransport2D(grid, diffusivity_m2_s=0.03).double()
    with torch.no_grad():
        target = integrate_ensemble(true_rhs, initial, times, config)
        base = integrate_ensemble(base_rhs, initial, times, config)
    closure = ConservativeFluxClosure2D(
        1, grid, hidden_channels=8, input_scale=2.0, flux_scale=0.3,
    ).double()
    hybrid_rhs = ConservativeTransport2D(
        grid, diffusivity_m2_s=0.03, closure=closure,
    ).double()
    optimizer = torch.optim.Adam(closure.parameters(), lr=0.01)
    before = float((base[1:] - target[1:]).square().mean())
    best_loss = before
    best_state = {key: value.detach().clone() for key, value in closure.state_dict().items()}
    for _ in range(args.steps):
        optimizer.zero_grad()
        predicted = integrate_ensemble(hybrid_rhs, initial, times, config)
        # Fit multiple lead times through the solver, rather than rewarding an
        # identity map or detaching the integrated prediction.
        loss = (predicted[1:] - target[1:]).square().mean()
        current = float(loss.detach())
        if current < best_loss:
            best_loss = current
            best_state = {
                key: value.detach().clone() for key, value in closure.state_dict().items()
            }
        loss.backward()
        torch.nn.utils.clip_grad_norm_(closure.parameters(), max_norm=1.0)
        optimizer.step()
    closure.load_state_dict(best_state)
    with torch.no_grad():
        final = integrate_ensemble(hybrid_rhs, initial, times, config)
        fine = integrate_ensemble(
            hybrid_rhs, initial, times, ODESolverConfig(step_seconds=0.025),
        )
    after = float((final[1:] - target[1:]).square().mean())
    print(json.dumps({
        "data": "manufactured diffusion training trajectories; no held-out climate evaluation",
        "optimizer_steps": args.steps,
        "baseline_trajectory_mse": before,
        "trained_trajectory_mse": after,
        "half_step_max_difference": float((final - fine).abs().max()),
        "maximum_mean_drift": float(
            (final.mean((-2, -1)) - initial.mean((-2, -1))[None]).abs().max()
        ),
    }, indent=2))
    if args.steps >= 20 and not after < before:
        raise RuntimeError("closure training did not improve the manufactured training trajectories")


if __name__ == "__main__":
    main()
