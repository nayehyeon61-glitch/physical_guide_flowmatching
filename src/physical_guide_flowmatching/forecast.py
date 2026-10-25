"""Forecast output schedules and memory-aware use of the existing ODE solver."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .ode import ODESolverConfig, _validate_state, _validate_times, integrate_ensemble


def forecast_times_seconds(
    horizon_days: float = 15.0,
    output_interval_hours: float = 6.0,
    *,
    reference: Tensor | None = None,
) -> Tensor:
    """15 days with six-hour output gives 61 times, including the initial state.

    These are output times, not internal solver steps. A fractional final
    interval is included when the horizon is not a multiple of the interval.
    """
    for name, value in (("horizon_days", horizon_days), ("output_interval_hours", output_interval_hours)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    horizon = horizon_days * 86400.0
    interval = output_interval_hours * 3600.0
    count = math.floor(horizon / interval)
    values = [index * interval for index in range(count + 1)]
    if math.isclose(values[-1], horizon, rel_tol=0, abs_tol=1e-8):
        values[-1] = horizon
    else:
        values.append(horizon)
    if reference is not None:
        if reference.dtype not in (torch.float32, torch.float64):
            raise ValueError("reference must have float32 or float64 dtype")
        return reference.new_tensor(values)
    return torch.tensor(values, dtype=torch.float64)


def integrate_windowed(
    rhs,
    initial: Tensor,
    times_seconds: Tensor,
    config: ODESolverConfig | None = None,
    *,
    checkpoint_segments: bool = False,
) -> Tensor:
    """Integrate between successive output times, keeping physical time intact.

    Calls the same torchdiffeq-based integrate_ensemble for each window. Each
    window starts from the previous prediction, never from future observations.
    No detach occurs: training remains differentiable across the whole horizon.

    checkpoint_segments recomputes window internals during backward, retaining
    direct-autograd gradients to RHS parameters and captured FM conditions.
    It reduces activation storage, not the number of forward time steps. RHS
    evaluation must be deterministic (no dropout or random sampling inside it).
    The returned output trajectory itself still consumes memory.

    Adaptive solvers restart their error controller at output boundaries; this
    may introduce ordinary tolerance-level differences from one long solve.
    Use torch.no_grad() around inference, and short-horizon/curriculum training
    before attempting a 15-day differentiable rollout.
    """
    _validate_state(initial, "initial")
    times = _validate_times(times_seconds, initial)
    config = config or ODESolverConfig()
    states = [initial]
    state = initial
    for index in range(times.numel() - 1):
        window_times = times[index : index + 2]

        # Bind the current window in the default argument. Backward checkpoint
        # recomputation must not observe the loop's final window for every step.
        def solve_window(value, local_times=window_times):
            return integrate_ensemble(rhs, value, local_times, config)[-1]

        if checkpoint_segments and torch.is_grad_enabled():
            state = checkpoint(solve_window, state, use_reentrant=False)
        else:
            state = solve_window(state)
        states.append(state)
    return torch.stack(states)
