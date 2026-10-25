"""Weak-constraint horizontal transport of Step13 ensemble members.

The existing SeqFM sampler already returns grid-space normalized increments;
this adapter must not apply another decoder or inverse DCT. It uses the
existing torchdiffeq-backed integrator, with no new numerical ODE solver.

Only pressure-level q and T are transported. Winds at the corresponding
pressure level are interpolated from the original FM trajectory, not observed
future data. This is a horizontal-transport experiment with reference nudging,
not a closed primitive-equation atmosphere: vertical transport, heating,
condensation, and momentum dynamics are not supplied here. Non-q/T channels,
including the known TISR field, are passed through exactly. Call the original
hard_constraints function on every experimental arm after correction.
"""

from __future__ import annotations

import math
import time

import torch
from torch import Tensor, nn

from physical_guide_flowmatching import (
    GlobalLatLonGrid, ODESolverConfig, SphericalTransport2D, integrate_ensemble,
)

from . import contract


_DYN = contract.DYNAMIC_CHANNELS
_GROUPS = {
    group: [_DYN.index(contract.CHANNEL_INDEX[f"{group}{level}"])
            for level in contract.LEVELS_HPA]
    for group in ("q", "t", "u", "v")
}
CORRECTED_DYNAMIC_INDICES = tuple(_GROUPS["q"] + _GROUPS["t"])


class _InterpolatedWind(nn.Module):
    def __init__(self, u0, u1, v0, v1, start_seconds, duration_seconds):
        super().__init__()
        self.u0, self.u1, self.v0, self.v1 = u0, u1, v0, v1
        self.start_seconds = start_seconds
        self.duration_seconds = duration_seconds

    def forward(self, t: Tensor, state: Tensor):
        del state
        fraction = ((t - self.start_seconds) / self.duration_seconds).clamp(0, 1)
        return (
            self.u0 + fraction * (self.u1 - self.u0),
            self.v0 + fraction * (self.v1 - self.v0),
        )


class _NormalizedWindowRHS(nn.Module):
    def __init__(self, transport, start_reference, end_reference, scale,
                 start_seconds, duration_seconds, rate, strength, limit_seconds):
        super().__init__()
        self.transport = transport
        self.start_reference = start_reference
        self.end_reference = end_reference
        self.scale = scale
        self.start_seconds = start_seconds
        self.duration_seconds = duration_seconds
        self.rate = rate
        self.strength = strength
        self.limit_seconds = limit_seconds
        self.nfe = 0

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        self.nfe += 1
        fraction = ((t - self.start_seconds) / self.duration_seconds).clamp(0, 1)
        reference = self.start_reference + fraction * (self.end_reference - self.start_reference)
        tendency = self.rate * (reference - state)
        if self.strength:
            # The advective-form operator annihilates a spatially constant
            # channel offset. Evaluating physical anomalies rather than
            # mean+anomaly is therefore mathematically equivalent and avoids
            # cancellation of large temperature means in float32. scale is
            # exactly std+KAI_STD_EPS, not std alone.
            physical_anomaly = state * self.scale
            tendency = tendency + self.strength * self.transport(t, physical_anomaly) / self.scale
        return tendency

    def explicit_step_limit_seconds(self, safety: float = 0.5) -> float:
        # limit_seconds was already computed with safety=.5. Match the API of
        # the transport modules in case this wrapper is inspected externally.
        if not math.isfinite(safety) or not 0 < safety <= 1:
            raise ValueError("safety must be in (0,1]")
        return self.limit_seconds * (safety / 0.5)


def _validate_inputs(x0, increments, mean, std):
    if not isinstance(x0, Tensor) or x0.ndim != 4 or x0.shape[1] != contract.N_DYNAMIC:
        raise ValueError("x0 must have shape [B,66,H,W]")
    if not isinstance(increments, Tensor) or increments.ndim != 6:
        raise ValueError("increments must have shape [B,M,K,66,H,W]")
    if increments.shape[0] != x0.shape[0] or increments.shape[3:] != x0.shape[1:]:
        raise ValueError("increments and x0 must have matching batch/channel/grid dimensions")
    if any(size < 1 for size in increments.shape) or min(x0.shape[-2:]) < 3:
        raise ValueError("all dimensions must be nonempty and the grid must be at least 3x3")
    if x0.dtype not in (torch.float32, torch.float64):
        raise ValueError("x0 must use float32 or float64")
    if x0.dtype != increments.dtype or x0.device != increments.device:
        raise ValueError("x0 and increments must have matching dtype/device")
    if not torch.isfinite(x0).all() or not torch.isfinite(increments).all():
        raise ValueError("x0 and increments must be finite")
    mean = torch.as_tensor(mean, dtype=x0.dtype, device=x0.device)
    std = torch.as_tensor(std, dtype=x0.dtype, device=x0.device)
    if mean.shape != (contract.N_CHANNELS,) or std.shape != mean.shape:
        raise ValueError("mean and std must each contain the 67 KAI channel values")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
        raise ValueError("normalization statistics must be finite with nonnegative std")
    return mean[_DYN], std[_DYN] + contract.KAI_STD_EPS


def correct_sequence_ode(
    x0: Tensor,
    increments: Tensor,
    mean,
    std,
    *,
    dt_seconds: float = 21600.0,
    step_seconds: float = 300.0,
    nudging_timescale_seconds: float = 3600.0,
    transport_strength: float = 1.0,
    wind_cap_m_s: float = 80.0,
) -> tuple[Tensor, dict]:
    """Return corrected normalized increments and transparent diagnostics.

    x0 is [B,66,H,W], increments is [B,M,K,66,H,W]. At window k the reference
    interpolates the *raw* FM states at k and k+1, but the solver starts from
    the previous corrected prediction. No future observations are accepted.

    transport_strength=0 is the matched nudging-only control; differences
    from raw FM can otherwise arise from statistical shrinkage alone.
    Temperature and q use advective rather than conserved-density transport.
    Their means are not projected to remain fixed. q positivity is deliberately
    left to the existing common hard-constraint step, outside this function.

    The supplied grid must already be the full global cell-centred lat/lon
    grid; the H/W dimensions set its uniform resolution. Six-hour output and
    15-day horizon never determine the smaller internal solver step.
    """
    started = time.perf_counter()
    means, scales = _validate_inputs(x0, increments, mean, std)
    for name, value in (
        ("dt_seconds", dt_seconds), ("step_seconds", step_seconds),
        ("nudging_timescale_seconds", nudging_timescale_seconds), ("wind_cap_m_s", wind_cap_m_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(transport_strength) or transport_strength < 0:
        raise ValueError("transport_strength must be finite and nonnegative")
    batch, members, windows, _, height, width = increments.shape
    grid = GlobalLatLonGrid(n_lat=height, n_lon=width)
    selected = list(CORRECTED_DYNAMIC_INDICES)
    scale = scales[selected].reshape(1, 1, -1, 1, 1)
    initial_reference = x0[:, None].expand(batch, members, contract.N_DYNAMIC, height, width)
    state = initial_reference[:, :, selected]
    initial_selected = state
    rate = 1.0 / nudging_timescale_seconds
    out, total_nfe, total_steps, minimum_step = [], 0, 0, math.inf
    clipped, wind_count, maximum_raw_wind = 0, 0, 0.0

    def physical_winds(reference):
        nonlocal clipped, wind_count, maximum_raw_wind
        wind = []
        for group in ("u", "v"):
            indices = _GROUPS[group]
            physical = reference[:, :, indices] * scales[indices].reshape(1, 1, -1, 1, 1)
            physical = physical + means[indices].reshape(1, 1, -1, 1, 1)
            clipped += int((physical.detach().abs() > wind_cap_m_s).sum())
            wind_count += physical.numel()
            maximum_raw_wind = max(maximum_raw_wind, float(physical.detach().abs().max()))
            capped = physical.clamp(-wind_cap_m_s, wind_cap_m_s)
            wind.append(torch.cat((capped, capped), dim=2))
        return wind[0], wind[1]

    raw_previous = initial_reference
    u0, v0 = physical_winds(raw_previous)
    for index in range(windows):
        raw_current = initial_reference + increments[:, :, index]
        u1, v1 = physical_winds(raw_current)
        start = index * dt_seconds
        wind = _InterpolatedWind(u0, u1, v0, v1, start, dt_seconds)
        transport = SphericalTransport2D(
            grid, transport_form="advective", velocity=wind, velocity_bound_m_s=wind_cap_m_s,
        ).to(device=x0.device, dtype=x0.dtype)

        # For linearly interpolated capped endpoint winds the in/out upwind
        # rate is convex in time, hence bounded by endpoint maxima. This is
        # tighter than assuming cap-speed flow in every cell for 15 days.
        endpoint_rates = []
        if transport_strength:
            for u, v in ((u0, v0), (u1, v1)):
                endpoint = SphericalTransport2D(
                    grid, u_m_s=u, v_m_s=v, transport_form="advective",
                ).to(device=x0.device, dtype=x0.dtype)
                limit = endpoint.explicit_step_limit_seconds(safety=1.0)
                endpoint_rates.append(0.0 if math.isinf(limit) else 1.0 / limit)
        transport_rate = max(endpoint_rates, default=0.0) * transport_strength
        bound = 0.5 / (rate + transport_rate)
        count = max(1, math.ceil(dt_seconds / min(step_seconds, bound)))
        # Fit an exact number of steps in the window, avoiding extrapolation
        # of the FM reference past its endpoint in the final RK step.
        actual_step = dt_seconds / count
        rhs = _NormalizedWindowRHS(
            transport, raw_previous[:, :, selected], raw_current[:, :, selected], scale,
            start, dt_seconds, rate, transport_strength, bound,
        )
        times = x0.new_tensor([start, start + dt_seconds])
        state = integrate_ensemble(
            rhs, state, times,
            ODESolverConfig(method="rk4", step_seconds=actual_step, max_num_steps=count + 2),
        )[-1]
        corrected = increments[:, :, index].clone()
        corrected[:, :, selected] = state - initial_selected
        out.append(corrected)
        total_nfe += rhs.nfe
        total_steps += count
        minimum_step = min(minimum_step, actual_step)
        raw_previous, u0, v0 = raw_current, u1, v1
        if windows >= 10 and ((index + 1) % 10 == 0 or index + 1 == windows):
            print(f"[step13 ODE strength={transport_strength:g}] window {index+1}/{windows} "
                  f"NFE={total_nfe} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    diagnostics = {
        "method": "torchdiffeq_rk4_reference_nudged_horizontal_q_T_transport",
        "interpretation": "weak_constraint_transport_experiment_not_full_atmospheric_core",
        "corrected_channels": [contract.CHANNEL_NAMES[_DYN[i]] for i in selected],
        "passthrough_channels": [contract.CHANNEL_NAMES[_DYN[i]] for i in range(contract.N_DYNAMIC) if i not in selected],
        "horizon_seconds": windows * dt_seconds,
        "output_interval_seconds": dt_seconds,
        "requested_internal_step_seconds": step_seconds,
        "minimum_internal_step_seconds": minimum_step,
        "internal_steps": total_steps,
        "nfe": total_nfe,
        "transport_strength": transport_strength,
        "nudging_timescale_seconds": nudging_timescale_seconds,
        "wind_cap_m_s": wind_cap_m_s,
        "wind_clipped_fraction": clipped / wind_count,
        "maximum_raw_wind_m_s": maximum_raw_wind,
        "seconds": time.perf_counter() - started,
        "hard_constraints_applied_here": False,
    }
    return torch.stack(out, dim=2), diagnostics
