"""Conservative transport on a periodic Cartesian grid.

This is a passive-tracer prototype, not a spherical atmospheric dynamical core.
States have shape ``[batch, ensemble, channel, height, width]`` and carry physical
units. Winds are prescribed at cell centres; both spatial directions are periodic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PeriodicCartesianGrid:
    """Uniform grid spacing in metres; x is width and y is height."""

    dx_m: float
    dy_m: float

    def __post_init__(self) -> None:
        for name in ("dx_m", "dy_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number in metres")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def _validate_state(state: Tensor) -> None:
    if not isinstance(state, Tensor):
        raise TypeError("state must be a torch.Tensor")
    if state.ndim != 5:
        raise ValueError("state must have shape [batch, ensemble, channel, height, width]")
    if not state.is_floating_point():
        raise TypeError("state must have a floating point dtype")
    if any(size < 1 for size in state.shape[:3]) or min(state.shape[-2:]) < 3:
        raise ValueError("batch, ensemble and channel must be nonempty; height and width must be >= 3")


def _fixed_coefficient(value: float | Tensor, name: str, *, scalar: bool = False) -> Tensor:
    # Python floats carry double precision. Preserve it until conversion to the
    # actual state's dtype, rather than rounding through the default float32.
    coefficient = (
        torch.as_tensor(value, dtype=torch.float64)
        if isinstance(value, Real) and not isinstance(value, bool)
        else torch.as_tensor(value)
    )
    if coefficient.is_complex() or coefficient.dtype == torch.bool:
        raise TypeError(f"{name} must be real valued")
    if not coefficient.is_floating_point():
        coefficient = coefficient.to(torch.get_default_dtype())
    if scalar and coefficient.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if not scalar and coefficient.ndim not in (0, 5):
        raise ValueError(
            f"{name} must be scalar or explicitly shaped [batch, ensemble, channel, height, width]; "
            "use singleton dimensions for broadcasting"
        )
    if coefficient.numel() == 0 or not bool(torch.isfinite(coefficient).all()):
        raise ValueError(f"{name} must contain only finite values")
    return coefficient.detach().clone()


def _broadcast_coefficient(coefficient: Tensor, state: Tensor, name: str) -> Tensor:
    try:
        shape = torch.broadcast_shapes(coefficient.shape, state.shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} cannot broadcast to state shape {tuple(state.shape)}") from exc
    if shape != state.shape:
        raise ValueError(f"{name} would expand the state rather than broadcast into it")
    return coefficient.to(device=state.device, dtype=state.dtype).expand_as(state)


def _flux_divergence(flux_x: Tensor, flux_y: Tensor, grid: PeriodicCartesianGrid) -> Tensor:
    """Divergence of fluxes stored at the right/upper face of each cell."""
    return (
        (flux_x - torch.roll(flux_x, 1, dims=-1)) / grid.dx_m
        + (flux_y - torch.roll(flux_y, 1, dims=-2)) / grid.dy_m
    )


class ConservativeTransport2D(nn.Module):
    """Method-of-lines RHS for conservative advection and scalar diffusion.

    ``d q / dt = -div(u q) + K laplacian(q) + closure(t, q)``.

    Winds are scalar or explicitly five-dimensional tensors broadcastable to the
    state, e.g. ``[1, ensemble, 1, height, width]``. Different members may have
    different winds without interacting. Coefficients are fixed buffers; a
    trainable parameterization belongs in ``closure``. Constant nonnegative K
    has units m²/s. The first-order face-upwind advection is conservative but
    introduces numerical diffusion. No positivity guarantee is made for an
    arbitrary external time integrator or learned closure.
    """

    def __init__(
        self,
        grid: PeriodicCartesianGrid,
        *,
        u_m_s: float | Tensor = 0.0,
        v_m_s: float | Tensor = 0.0,
        diffusivity_m2_s: float | Tensor = 0.0,
        closure: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(grid, PeriodicCartesianGrid):
            raise TypeError("grid must be a PeriodicCartesianGrid")
        self.grid = grid
        self.register_buffer("u_m_s", _fixed_coefficient(u_m_s, "u_m_s"))
        self.register_buffer("v_m_s", _fixed_coefficient(v_m_s, "v_m_s"))
        diffusivity = _fixed_coefficient(diffusivity_m2_s, "diffusivity_m2_s", scalar=True)
        if bool(diffusivity < 0):
            raise ValueError("diffusivity_m2_s must be nonnegative")
        self.register_buffer("diffusivity_m2_s", diffusivity)
        if closure is not None and not callable(closure):
            raise TypeError("closure must be callable or None")
        self.closure = closure

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        _validate_state(state)
        u = _broadcast_coefficient(self.u_m_s, state, "u_m_s")
        v = _broadcast_coefficient(self.v_m_s, state, "v_m_s")
        right = torch.roll(state, -1, dims=-1)
        upper = torch.roll(state, -1, dims=-2)
        u_face = 0.5 * (u + torch.roll(u, -1, dims=-1))
        v_face = 0.5 * (v + torch.roll(v, -1, dims=-2))
        flux_x = u_face * torch.where(u_face >= 0, state, right)
        flux_y = v_face * torch.where(v_face >= 0, state, upper)
        tendency = -_flux_divergence(flux_x, flux_y, self.grid)
        laplacian = (
            (right - 2 * state + torch.roll(state, 1, dims=-1)) / self.grid.dx_m**2
            + (upper - 2 * state + torch.roll(state, 1, dims=-2)) / self.grid.dy_m**2
        )
        tendency = tendency + self.diffusivity_m2_s.to(state) * laplacian
        if self.closure is not None:
            correction = self.closure(t, state)
            if not isinstance(correction, Tensor) or correction.shape != state.shape:
                raise ValueError("closure must return a tensor with exactly the state shape")
            if correction.device != state.device or correction.dtype != state.dtype:
                raise ValueError("closure output must have the state device and dtype")
            tendency = tendency + correction
        return tendency

    def explicit_step_limit_seconds(self, safety: float = 0.5) -> float:
        """Return a conservative advection/diffusion time-step estimate.

        This estimates the forward-Euler monotonicity bound times ``safety``
        using maximum outgoing face-upwind rates. It is useful when selecting an
        explicit RK step, but is not a stability/positivity proof for every RK
        method. It excludes any learned closure or external nudging. In
        particular, adaptive local-error control alone does not cure PDE
        stiffness. Spatially varying winds may compress a tracer locally.
        """
        if isinstance(safety, bool) or not isinstance(safety, Real):
            raise TypeError("safety must be a real number")
        if not math.isfinite(safety) or not 0 < safety <= 1:
            raise ValueError("safety must be in (0, 1]")
        def maximum_outgoing_rate(wind: Tensor, dimension: int, spacing: float) -> float:
            wind = wind.detach()
            if wind.ndim == 0:
                return wind.abs().item() / spacing
            face_wind = 0.5 * (wind + torch.roll(wind, -1, dims=dimension))
            outgoing = face_wind.clamp_min(0) + (
                -torch.roll(face_wind, 1, dims=dimension)
            ).clamp_min(0)
            return outgoing.max().item() / spacing

        rate = (
            maximum_outgoing_rate(self.u_m_s, -1, self.grid.dx_m)
            + maximum_outgoing_rate(self.v_m_s, -2, self.grid.dy_m)
            + 2 * self.diffusivity_m2_s.detach().item()
            * (self.grid.dx_m**-2 + self.grid.dy_m**-2)
        )
        return safety / rate if rate > 0 else math.inf


class ConservativeFluxClosure2D(nn.Module):
    """Learn a shared, member-independent correction through face fluxes.

    The CNN emits x/y face fluxes, and the returned tendency is their negative
    discrete divergence. Thus each member/channel has zero spatially summed
    closure tendency (up to roundoff) on the uniform periodic grid. This does
    not enforce momentum, energy, entropy or positivity constraints.

    ``input_scale`` has the state's units and normalizes the CNN input;
    ``flux_scale`` has state-units * m/s and converts network output to physical
    flux. Optional ``condition_channels`` allow each member's flow-matching
    latent to condition its fluxes even when all members share an observed
    initial state. Conditions are already-normalized neural features and are
    not divided by ``input_scale``. Use ``conditioned_on(condition)`` to adapt
    the three-argument conditioned closure to an ODE RHS.

    The final convolution starts at zero, so pretraining initially uses only
    the resolved physical RHS. Consequently gradients to upstream state or
    conditioning features through the closure initially vanish until that
    convolution learns nonzero weights. Move this module to the state
    dtype/device with ``.to(...)`` before integrating. No layer mixes members.
    """

    def __init__(
        self,
        channels: int,
        grid: PeriodicCartesianGrid,
        hidden_channels: int = 16,
        input_scale: float = 1.0,
        flux_scale: float = 1.0,
        *,
        condition_channels: int = 0,
    ) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer")
        if isinstance(hidden_channels, bool) or not isinstance(hidden_channels, int) or hidden_channels < 1:
            raise ValueError("hidden_channels must be a positive integer")
        if (
            isinstance(condition_channels, bool)
            or not isinstance(condition_channels, int)
            or condition_channels < 0
        ):
            raise ValueError("condition_channels must be a nonnegative integer")
        if not isinstance(grid, PeriodicCartesianGrid):
            raise TypeError("grid must be a PeriodicCartesianGrid")
        for name, value in (("input_scale", input_scale), ("flux_scale", flux_scale)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.channels = channels
        self.condition_channels = condition_channels
        self.grid = grid
        self.register_buffer("input_scale", torch.tensor(float(input_scale), dtype=torch.float64))
        self.register_buffer("flux_scale", torch.tensor(float(flux_scale), dtype=torch.float64))
        self.net = nn.Sequential(
            nn.Conv2d(channels + condition_channels, hidden_channels, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 2 * channels, 3, padding=1, padding_mode="circular"),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: Tensor, state: Tensor, condition: Tensor | None = None) -> Tensor:
        """Return flux-divergence tendency conditioned independently per member.

        Conditions have shape ``[B, E, D]`` or ``[B, E, D, H, W]``. Their B/E/D
        dimensions must match exactly; spatial singleton dimensions may
        broadcast to the grid. Condition dtype/device must match the state.
        The condition is never detached from the flow-matching graph.
        """
        del t  # autonomous closure; forcing/time dependence can be added explicitly
        _validate_state(state)
        if state.shape[2] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {state.shape[2]}")
        batch, members, channels, height, width = state.shape
        member_states = state.reshape(batch * members, channels, height, width)
        features = member_states / self.input_scale
        if self.condition_channels == 0:
            if condition is not None:
                raise ValueError("condition must be None when condition_channels is zero")
        else:
            if not isinstance(condition, Tensor):
                raise ValueError("condition is required when condition_channels is positive")
            if condition.dtype != state.dtype or condition.device != state.device:
                raise ValueError("condition must have the state device and dtype")
            if condition.ndim not in (3, 5):
                raise ValueError("condition must have shape [B,E,D] or [B,E,D,H,W]")
            if tuple(condition.shape[:3]) != (batch, members, self.condition_channels):
                raise ValueError("condition batch, ensemble and feature dimensions must match exactly")
            if condition.ndim == 3:
                condition = condition[..., None, None]
            if condition.shape[-2] not in (1, height) or condition.shape[-1] not in (1, width):
                raise ValueError("condition spatial dimensions must match the grid or be singleton")
            member_condition = condition.expand(
                batch, members, self.condition_channels, height, width
            ).reshape(batch * members, self.condition_channels, height, width)
            features = torch.cat((features, member_condition), dim=1)
        flux = self.net(features) * self.flux_scale
        flux_x, flux_y = flux.chunk(2, dim=1)
        tendency = -_flux_divergence(flux_x, flux_y, self.grid)
        return tendency.reshape_as(state)

    def conditioned_on(self, condition: Tensor) -> nn.Module:
        """Bind a batch's FM features without detaching or copying parameters.

        Create a fresh wrapper per forward pass. The wrapper holds the shared
        closure and the original condition tensor, preserving its upstream
        autograd graph. Conditions must already be on the integration dtype
        and device; moving the wrapper does not move this plain tensor.
        """
        if self.condition_channels == 0:
            raise ValueError("conditioned_on requires positive condition_channels")
        if not isinstance(condition, Tensor):
            raise TypeError("condition must be a torch.Tensor")
        return _ConditionedFluxClosure(self, condition)


class _ConditionedFluxClosure(nn.Module):
    """Adapt a shared conditioned closure to the ``(t, state)`` ODE API."""

    def __init__(self, closure: ConservativeFluxClosure2D, condition: Tensor) -> None:
        super().__init__()
        self.closure = closure
        # This is a graph-connected batch input, not persistent model state.
        object.__setattr__(self, "condition", condition)

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        return self.closure(t, state, self.condition)
