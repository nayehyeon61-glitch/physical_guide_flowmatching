"""Time integration uses torchdiffeq; this module does not implement a solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import Tensor, nn
from torchdiffeq import odeint


@dataclass(frozen=True)
class ODESolverConfig:
    """Physical time is in seconds; step_seconds is also the adaptive max step.

    Fixed-step RK4 is the default so each member has the same prescribed time
    grid without an adaptive error norm shared across members. Neither method
    removes stiffness or guarantees positivity of the solution.
    """

    method: str = "rk4"
    step_seconds: float = 60.0
    rtol: float = 1e-5
    atol: float = 1e-6
    max_num_steps: int = 10000

    def __post_init__(self) -> None:
        if self.method not in {"rk4", "dopri5"}:
            raise ValueError("method must be 'rk4' or 'dopri5' for differentiable integration")
        for name in ("step_seconds", "rtol", "atol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.max_num_steps, int) or self.max_num_steps < 1:
            raise ValueError("max_num_steps must be a positive integer")


def _validate_state(state: Tensor, name: str) -> None:
    if not isinstance(state, Tensor) or state.ndim != 5:
        raise ValueError(f"{name} must have shape [batch, ensemble, channel, height, width]")
    if state.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64 for ODE integration")
    if any(size == 0 for size in state.shape) or not torch.isfinite(state).all():
        raise ValueError(f"{name} must be nonempty and finite")


def _validate_times(times_seconds: Tensor, initial: Tensor) -> Tensor:
    times = torch.as_tensor(times_seconds, dtype=initial.dtype, device=initial.device)
    if times.ndim != 1 or times.numel() < 2:
        raise ValueError("times_seconds must contain at least two 1D time points")
    if not torch.isfinite(times).all() or not (times[1:] > times[:-1]).all():
        raise ValueError("times_seconds must be finite and strictly increasing")
    return times


def _worst_member_rms(error: Tensor) -> Tensor:
    # Avoid diluting one member's error over the entire ensemble. Adaptive time
    # steps are still shared: use torchode or individual solves if independence
    # of the adaptive step schedule is a requirement.
    return error.flatten(0, 1).flatten(1).square().mean(1).sqrt().max()


def integrate_ensemble(
    rhs: Callable[[Tensor, Tensor], Tensor],
    initial: Tensor,
    times_seconds: Tensor,
    config: ODESolverConfig | None = None,
) -> Tensor:
    """Integrate a physical initial state, returning [time, B, E, C, H, W].

    No member reductions occur here except the adaptive error controller.
    The supplied RHS must itself avoid member mixing (including BatchNorm).
    Direct autograd preserves gradients to initial state and RHS parameters.
    """
    _validate_state(initial, "initial")
    times = _validate_times(times_seconds, initial)
    config = config or ODESolverConfig()
    if hasattr(rhs, "explicit_step_limit_seconds"):
        limit = rhs.explicit_step_limit_seconds()
        if config.step_seconds > limit * (1.0 + 1e-7):
            raise ValueError(
                f"step_seconds={config.step_seconds:g} exceeds the explicit transport/"
                f"nudging bound {limit:g}; reduce it. Learned-closure stability needs "
                "a separate step-convergence check."
            )
    options: dict = {}
    if config.method == "rk4":
        duration = float((times[-1] - times[0]).detach().cpu())
        if math.ceil(duration / config.step_seconds) > config.max_num_steps:
            raise ValueError("fixed-step integration would exceed max_num_steps")
        options["step_size"] = config.step_seconds
    else:
        options.update(
            max_step=config.step_seconds,
            max_num_steps=config.max_num_steps,
            norm=_worst_member_rms,
        )
    trajectory = odeint(
        rhs, initial, times, method=config.method,
        rtol=config.rtol, atol=config.atol, options=options,
    )
    if not torch.isfinite(trajectory).all():
        raise RuntimeError("ODE returned nonfinite states; inspect RHS, units and step size")
    return trajectory


class _ReferenceNudgedRHS(nn.Module):
    """A weak-constraint forecast, not a hard physical constraint projection."""

    def __init__(self, rhs, initial, forecast, times, rate, conserve_mean, spatial_weights=None):
        super().__init__()
        self.physical_rhs = rhs
        # Keep graph connections to the FM output; detach would silently stop
        # training the ensemble generator through this module.
        self.initial = initial
        self.forecast = forecast
        self.start = times[0]
        self.duration = times[-1] - times[0]
        self.rate = rate
        self.conserve_mean = conserve_mean
        self.spatial_weights = spatial_weights

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        fraction = (t - self.start) / self.duration
        reference = self.initial + fraction * (self.forecast - self.initial)
        tendency = self.rate * (reference - state)
        if self.conserve_mean:
            # Mean-budget projection is appropriate to a conserved tracer,
            # not generally to ERA5 temperature/geopotential/wind channels.
            if self.spatial_weights is None:
                mean = tendency.mean(dim=(-2, -1), keepdim=True)
            else:
                mean = (tendency * self.spatial_weights).sum(dim=(-2, -1), keepdim=True)
            tendency = tendency - mean
        return self.physical_rhs(t, state) + tendency

    def explicit_step_limit_seconds(self, safety: float = 0.5) -> float:
        # Includes the linear nudging rate. An arbitrary learned RHS may add
        # stiffness not represented by this advective/diffusive bound.
        if not hasattr(self.physical_rhs, "explicit_step_limit_seconds"):
            return safety / self.rate if self.rate > 0 else math.inf
        base_limit = self.physical_rhs.explicit_step_limit_seconds(safety=1.0)
        base_rate = 0.0 if math.isinf(base_limit) else 1.0 / base_limit
        rate = base_rate + self.rate
        return safety / rate if rate > 0 else math.inf


def correct_forecast(
    rhs: Callable[[Tensor, Tensor], Tensor],
    observed_initial: Tensor,
    forecast_ensemble: Tensor,
    times_seconds: Tensor,
    *,
    nudging_rate_s: float,
    conserve_spatial_mean: bool = True,
    spatial_weights: Tensor | None = None,
    config: ODESolverConfig | None = None,
) -> Tensor:
    """Reintegrate from t0 while referencing FM's prediction at t1.

    FM forecasts are *not* used as initial states and advanced a second time.
    Instead reference(t) linearly joins x0 and each FM endpoint, and
    dx/dt = rhs(t,x) + nudging_rate_s * (reference(t) - x).
    This auxiliary nudging tendency is not a physical law; tune/validate it.

    observed_initial: [B,C,H,W] or [B,E,C,H,W], in physical units.
    forecast_ensemble: [B,E,C,H,W], at times_seconds[-1], physical units.
    conserve_spatial_mean: remove the nudging term's spatial mean. This only
      preserves the original RHS's mean budget; it cannot repair that RHS.
    spatial_weights: optional nonnegative [H,W] areas/weights for that mean.
      A RHS exposing grid.cell_area_m2 uses those areas automatically. ERA5
      temperature/geopotential do not generally have conserved mean budgets.
    """
    _validate_state(forecast_ensemble, "forecast_ensemble")
    if not math.isfinite(nudging_rate_s) or nudging_rate_s < 0:
        raise ValueError("nudging_rate_s must be finite and nonnegative")
    if not isinstance(observed_initial, Tensor):
        raise ValueError("observed_initial must be a tensor")
    initial = observed_initial
    if initial.ndim == 4:
        initial = initial.unsqueeze(1)
    if (
        initial.ndim != 5
        or initial.shape[0] != forecast_ensemble.shape[0]
        or initial.shape[2:] != forecast_ensemble.shape[2:]
        or initial.shape[1] not in (1, forecast_ensemble.shape[1])
    ):
        raise ValueError("observed_initial must match forecast batch, channels and grid")
    if initial.device != forecast_ensemble.device or initial.dtype != forecast_ensemble.dtype:
        raise ValueError("initial and forecast must have the same dtype and device")
    initial = initial.expand_as(forecast_ensemble)
    _validate_state(initial, "observed_initial")
    times = _validate_times(times_seconds, initial)
    if spatial_weights is None and conserve_spatial_mean:
        spatial_weights = getattr(getattr(rhs, "grid", None), "cell_area_m2", None)
    if spatial_weights is not None:
        spatial_weights = torch.as_tensor(spatial_weights, dtype=initial.dtype, device=initial.device)
        if (
            spatial_weights.shape != initial.shape[-2:]
            or not torch.isfinite(spatial_weights).all()
            or (spatial_weights < 0).any()
            or not spatial_weights.sum() > 0
        ):
            raise ValueError("spatial_weights must be finite nonnegative [H,W] with positive sum")
        spatial_weights = spatial_weights / spatial_weights.sum()
    anchored_rhs = _ReferenceNudgedRHS(
        rhs, initial, forecast_ensemble, times, nudging_rate_s, conserve_spatial_mean, spatial_weights,
    )
    return integrate_ensemble(anchored_rhs, initial, times, config)
