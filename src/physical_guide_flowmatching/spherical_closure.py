"""Member-conditioned learned fluxes and sources on a global lat/lon grid.

The metric divergence is spherical and area conservative. The local CNN uses
periodic longitude and replicated latitude padding; it is an approximation to
spherical covariance, not a rotation-equivariant spherical convolution or an
atmospheric momentum model.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .spherical import GlobalLatLonGrid, _metric_divergence


def _channel_scale(value, channels: int, name: str, *, strictly_positive: bool) -> Tensor:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar or one real value per channel")
    scale = (
        torch.as_tensor(value, dtype=torch.float64)
        if not isinstance(value, Tensor)
        else torch.as_tensor(value)
    )
    if scale.is_complex() or scale.dtype == torch.bool:
        raise TypeError(f"{name} must be real valued")
    if scale.ndim == 0:
        scale = scale.expand(channels)
    elif scale.ndim != 1 or scale.numel() != channels:
        raise ValueError(f"{name} must be scalar or have exactly {channels} channel values")
    scale = scale.detach().to(dtype=torch.float64).clone()
    if not bool(torch.isfinite(scale).all()):
        raise ValueError(f"{name} must contain only finite values")
    if bool((scale <= 0 if strictly_positive else scale < 0).any()):
        qualifier = "positive" if strictly_positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return scale.reshape(1, channels, 1, 1)


class _LongitudePeriodicLatitudeReplicatePad(nn.Module):
    """Pad local scalar features without connecting north and south rows."""

    def forward(self, features: Tensor) -> Tensor:
        return F.pad(F.pad(features, (1, 1, 0, 0), mode="circular"),
                     (0, 0, 1, 1), mode="replicate")


class SphericalFluxSourceClosure2D(nn.Module):
    """Learn independent member fluxes plus optional nonconservative sources.

    States use [B,E,C,H,W]. ``input_scale`` has each state's physical units;
    ``flux_scale`` has state-units*m/s; ``source_scale`` has state-units/s.
    Each scale is a scalar or C-element vector. The default source scale is
    zero, so only a conservative flux divergence is returned. A nonzero source
    scale allows temperature/geopotential budgets to change; it does not make
    those budgets physically correct without suitable supervision/forcing.

    Optional normalized conditions [B,E,D] or [B,E,D,H,W] carry FM latent or
    forecast features without detaching their graph. B/E/D must match exactly;
    spatial singleton dimensions may broadcast. Calendar, terrain and external
    forcing can be supplied explicitly as additional condition maps. Time is
    otherwise unused, so no particular season or daily forcing is assumed.

    The final convolution starts at zero. Initially upstream condition
    gradients through this closure vanish until that layer learns nonzero
    weights. Pretrain the shared closure before joint FM fine-tuning. Move the
    module to the state dtype/device with .to(...) before using it.
    """

    def __init__(
        self,
        channels: int,
        grid: GlobalLatLonGrid,
        hidden_channels: int = 16,
        input_scale=1.0,
        flux_scale=1.0,
        source_scale=0.0,
        *,
        condition_channels: int = 0,
    ) -> None:
        super().__init__()
        for name, value, minimum in (
            ("channels", channels, 1),
            ("hidden_channels", hidden_channels, 1),
            ("condition_channels", condition_channels, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(grid, GlobalLatLonGrid):
            raise TypeError("grid must be a GlobalLatLonGrid")
        self.channels = channels
        self.condition_channels = condition_channels
        self.grid = grid
        east_length, north_length, _, _ = grid._metrics()
        # Geometry is retained with the model and moves with .to(...), rather
        # than being rebuilt on CPU for every solver RHS evaluation.
        self.register_buffer("_area", grid.cell_area_m2.clone())
        self.register_buffer("_east_length", east_length)
        self.register_buffer("_north_length", north_length)
        self.register_buffer("input_scale", _channel_scale(
            input_scale, channels, "input_scale", strictly_positive=True,
        ))
        self.register_buffer("flux_scale", _channel_scale(
            flux_scale, channels, "flux_scale", strictly_positive=False,
        ))
        self.register_buffer("source_scale", _channel_scale(
            source_scale, channels, "source_scale", strictly_positive=False,
        ))
        self.net = nn.Sequential(
            _LongitudePeriodicLatitudeReplicatePad(),
            nn.Conv2d(channels + condition_channels, hidden_channels, 3),
            nn.SiLU(),
            _LongitudePeriodicLatitudeReplicatePad(),
            nn.Conv2d(hidden_channels, 3 * channels, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: Tensor, state: Tensor, condition: Tensor | None = None) -> Tensor:
        """Return -div_sphere(learned flux) + learned source in physical units/s."""
        del t
        if not isinstance(state, Tensor) or state.ndim != 5 or not state.is_floating_point():
            raise ValueError("state must be floating [B,E,C,H,W]")
        batch, members, channels, height, width = state.shape
        if batch < 1 or members < 1 or channels != self.channels:
            raise ValueError("state batch/ensemble must be nonempty and channel count must match")
        if (height, width) != (self.grid.n_lat, self.grid.n_lon):
            raise ValueError("state spatial shape must match the global grid")
        if state.dtype != self.net[1].weight.dtype or state.device != self.net[1].weight.device:
            raise ValueError("move the closure to the state dtype/device with .to(...) before integration")
        features = state.reshape(batch * members, channels, height, width) / self.input_scale.to(state)
        if self.condition_channels == 0:
            if condition is not None:
                raise ValueError("condition must be None when condition_channels is zero")
        else:
            if not isinstance(condition, Tensor) or condition.ndim not in (3, 5):
                raise ValueError("condition must have shape [B,E,D] or [B,E,D,H,W]")
            if condition.device != state.device or condition.dtype != state.dtype:
                raise ValueError("condition must have the state dtype and device")
            if tuple(condition.shape[:3]) != (batch, members, self.condition_channels):
                raise ValueError("condition batch, ensemble and feature dimensions must match exactly")
            if condition.ndim == 3:
                condition = condition[..., None, None]
            if condition.shape[-2] not in (1, height) or condition.shape[-1] not in (1, width):
                raise ValueError("condition spatial dimensions must match the grid or be singleton")
            condition_features = condition.expand(
                batch, members, self.condition_channels, height, width,
            ).reshape(batch * members, self.condition_channels, height, width)
            features = torch.cat((features, condition_features), dim=1)
        east_centres, north_centres, source = self.net(features).chunk(3, dim=1)
        east_centres = east_centres * self.flux_scale.to(state)
        north_centres = north_centres * self.flux_scale.to(state)
        east_faces = 0.5 * (east_centres + torch.roll(east_centres, -1, dims=-1))
        boundary = torch.zeros_like(north_centres[..., :1, :])
        north_faces = torch.cat((
            boundary,
            0.5 * (north_centres[..., :-1, :] + north_centres[..., 1:, :]),
            boundary,
        ), dim=-2)
        tendency = -_metric_divergence(
            east_faces, north_faces, self._area.to(east_faces),
            self._east_length.to(east_faces), self._north_length.to(east_faces),
        )
        tendency = tendency + source * self.source_scale.to(state)
        return tendency.reshape_as(state)

    def conditioned_on(self, condition: Tensor) -> nn.Module:
        """Bind this batch's FM features while sharing weights and preserving autograd.

        Construct a fresh wrapper per model forward pass. The bound tensor is
        not a parameter or persistent buffer and must already match the state
        dtype/device; moving the wrapper does not move that batch tensor.
        """
        if self.condition_channels == 0:
            raise ValueError("conditioned_on requires positive condition_channels")
        if not isinstance(condition, Tensor):
            raise TypeError("condition must be a tensor")
        return _ConditionedSphericalClosure(self, condition)


class _ConditionedSphericalClosure(nn.Module):
    def __init__(self, closure: SphericalFluxSourceClosure2D, condition: Tensor) -> None:
        super().__init__()
        self.closure = closure
        object.__setattr__(self, "condition", condition)

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        return self.closure(t, state, self.condition)
