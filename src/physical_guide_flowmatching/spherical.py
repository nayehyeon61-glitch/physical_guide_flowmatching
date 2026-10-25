"""Finite-volume scalar transport on a global cell-centred latitude/longitude grid.

This supplies spherical geometry for an existing ODE integrator. It is not a
primitive-equation atmospheric dynamical core. Longitude is periodic; latitude
is not. The polar faces have exactly zero length and zero normal flux.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable

import torch
from torch import Tensor, nn

from .physics import _broadcast_coefficient, _fixed_coefficient, _validate_state


@dataclass(frozen=True)
class GlobalLatLonGrid:
    """Uniform full-sphere cells, ascending latitude and longitude starting at 0.

    The default is WeatherBench/ClimODE's 5.625-degree 32-by-64 global grid:
    latitude -87.1875..87.1875, longitude 0..354.375. Coordinates are cell
    centres, not poles or duplicate longitude endpoints. Regional grids need
    different boundary conditions and are intentionally not accepted here.
    """

    n_lat: int = 32
    n_lon: int = 64
    radius_m: float = 6_371_000.0

    def __post_init__(self) -> None:
        for name in ("n_lat", "n_lon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 3:
                raise ValueError(f"{name} must be an integer >= 3")
        if isinstance(self.radius_m, bool) or not isinstance(self.radius_m, Real):
            raise TypeError("radius_m must be a real number")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("radius_m must be finite and positive")

    @property
    def latitude_degrees(self) -> Tensor:
        return -90.0 + (torch.arange(self.n_lat, dtype=torch.float64) + 0.5) * (180.0 / self.n_lat)

    @property
    def longitude_degrees(self) -> Tensor:
        return torch.arange(self.n_lon, dtype=torch.float64) * (360.0 / self.n_lon)

    @property
    def cell_area_m2(self) -> Tensor:
        edges = torch.linspace(-math.pi / 2, math.pi / 2, self.n_lat + 1, dtype=torch.float64)
        rows = self.radius_m**2 * (2 * math.pi / self.n_lon) * (edges[1:].sin() - edges[:-1].sin())
        return rows[:, None].expand(self.n_lat, self.n_lon)

    def validate_coordinates(self, latitude_degrees, longitude_degrees) -> None:
        """Reject incompatible/oriented coordinates instead of silently reordering data."""
        for name, actual, expected in (
            ("latitude", latitude_degrees, self.latitude_degrees),
            ("longitude", longitude_degrees, self.longitude_degrees),
        ):
            values = torch.as_tensor(actual, dtype=torch.float64).detach().cpu()
            if values.shape != expected.shape or not torch.isfinite(values).all():
                raise ValueError(f"{name} must be a finite 1D coordinate of length {expected.numel()}")
            if not torch.allclose(values, expected, rtol=0.0, atol=1e-5):
                raise ValueError(
                    f"{name} does not match this full-sphere cell-centred grid; "
                    "reorder/regrid data and coordinates together explicitly"
                )

    def _metrics(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dphi = math.pi / self.n_lat
        dlambda = 2 * math.pi / self.n_lon
        phi = torch.deg2rad(self.latitude_degrees)
        edges = torch.linspace(-math.pi / 2, math.pi / 2, self.n_lat + 1, dtype=torch.float64)
        east_face_length = torch.tensor(self.radius_m * dphi, dtype=torch.float64)
        north_face_length = (self.radius_m * dlambda * edges.cos())[:, None]
        # cos(pi/2) is nonzero at roundoff; set polar face lengths exactly zero.
        north_face_length[0] = 0.0
        north_face_length[-1] = 0.0
        zonal_distance = (self.radius_m * dlambda * phi.cos())[:, None]
        meridional_distance = torch.tensor(self.radius_m * dphi, dtype=torch.float64)
        return east_face_length, north_face_length, zonal_distance, meridional_distance

    def flux_divergence(self, east_flux: Tensor, north_flux: Tensor) -> Tensor:
        """Metric divergence from east-face [..,H,W] and north-face [..,H+1,W] fluxes.

        Flux units are quantity*m/s, and the result has quantity/s. The result
        telescopes under exact cell-area weighting. No latitude wrap is used.
        """
        _validate_fluxes(east_flux, north_flux, self)
        east_length, north_length, _, _ = self._metrics()
        return _metric_divergence(
            east_flux, north_flux, self.cell_area_m2.to(east_flux),
            east_length.to(east_flux), north_length.to(east_flux),
        )


def _validate_fluxes(east: Tensor, north: Tensor, grid: GlobalLatLonGrid) -> None:
    if not isinstance(east, Tensor) or not isinstance(north, Tensor) or east.ndim < 2:
        raise ValueError("east_flux and north_flux must be floating tensors with spatial axes")
    if not east.is_floating_point() or north.dtype != east.dtype or north.device != east.device:
        raise ValueError("fluxes must share a floating dtype and device")
    if east.shape[-2:] != (grid.n_lat, grid.n_lon):
        raise ValueError("east_flux spatial shape must match the global grid")
    if north.shape != (*east.shape[:-2], grid.n_lat + 1, grid.n_lon):
        raise ValueError("north_flux must have one more latitude face than cells")


def _metric_divergence(east, north, area, east_length, north_length):
    east_net = (east - torch.roll(east, 1, dims=-1)) * east_length
    north_net = north[..., 1:, :] * north_length[1:] - north[..., :-1, :] * north_length[:-1]
    return (east_net + north_net) / area


def area_weighted_mean(state: Tensor, grid: GlobalLatLonGrid) -> Tensor:
    """Reduce only H/W using exact spherical cell areas, preserving autograd."""
    if not isinstance(state, Tensor) or state.ndim < 2 or not state.is_floating_point():
        raise ValueError("state must be a floating tensor with latitude/longitude axes")
    if state.shape[-2:] != (grid.n_lat, grid.n_lon):
        raise ValueError("state spatial shape does not match grid")
    areas = grid.cell_area_m2
    weights = (areas / areas.sum()).to(state)
    return (state * weights).sum(dim=(-2, -1))


def _north_faces(centre_values: Tensor) -> Tensor:
    boundary = torch.zeros_like(centre_values[..., :1, :])
    interior = 0.5 * (centre_values[..., :-1, :] + centre_values[..., 1:, :])
    return torch.cat((boundary, interior, boundary), dim=-2)


class SphericalTransport2D(nn.Module):
    """Upwind scalar transport and diffusion in physical metre/second units.

    ``transport_form='conservative'`` implements -div(u*q)+K*laplacian(q),
    appropriate to a conserved surface density/tracer. ``'advective'`` adds
    q*div(u), so a constant material scalar stays constant in divergent flow;
    its area mean need not be conserved. Neither treats five ERA5 fields as a
    closed atmospheric system or supplies momentum/thermal/vertical dynamics.

    Fixed u/v are scalars or explicit five-dimensional broadcastable fields.
    A differentiable ``velocity(t,state)->(u,v)`` may instead supply evolving
    member/channel-specific transport velocities. Its components must be
    bounded by ``velocity_bound_m_s``; values are checked at each RHS call.
    Fixed u/v must remain zero when velocity is supplied, avoiding ambiguity.
    A velocity magnitude bound constrains instantaneous transport rates, not
    the Jacobian/stiffness of a learned velocity network.

    closure(t,state) is an optional physical/learned source tendency with the
    same shape and units as dq/dt. It need not preserve a mean budget. Its
    additional stiffness is not included in explicit_step_limit_seconds.
    """

    def __init__(
        self,
        grid: GlobalLatLonGrid,
        *,
        u_m_s: float | Tensor = 0.0,
        v_m_s: float | Tensor = 0.0,
        diffusivity_m2_s: float | Tensor = 0.0,
        transport_form: str = "conservative",
        closure: Callable[[Tensor, Tensor], Tensor] | None = None,
        velocity: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
        velocity_bound_m_s: float | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(grid, GlobalLatLonGrid):
            raise TypeError("grid must be a GlobalLatLonGrid")
        if transport_form not in {"conservative", "advective"}:
            raise ValueError("transport_form must be 'conservative' or 'advective'")
        if closure is not None and not callable(closure):
            raise TypeError("closure must be callable")
        if velocity is not None and not callable(velocity):
            raise TypeError("velocity must be callable")
        self.grid = grid
        self.transport_form = transport_form
        self.register_buffer("u_m_s", _fixed_coefficient(u_m_s, "u_m_s"))
        self.register_buffer("v_m_s", _fixed_coefficient(v_m_s, "v_m_s"))
        for name in ("u_m_s", "v_m_s"):
            wind = getattr(self, name)
            if wind.ndim == 5 and (
                wind.shape[-2] not in (1, grid.n_lat) or wind.shape[-1] not in (1, grid.n_lon)
            ):
                raise ValueError(f"{name} spatial shape cannot broadcast to this grid")
        diffusivity = _fixed_coefficient(diffusivity_m2_s, "diffusivity_m2_s", scalar=True)
        if bool(diffusivity < 0):
            raise ValueError("diffusivity_m2_s must be nonnegative")
        self.register_buffer("diffusivity_m2_s", diffusivity)
        if velocity is not None:
            if bool(self.u_m_s.ne(0).any()) or bool(self.v_m_s.ne(0).any()):
                raise ValueError("fixed winds must be zero when a velocity callable is supplied")
            if (
                isinstance(velocity_bound_m_s, bool)
                or not isinstance(velocity_bound_m_s, Real)
                or not math.isfinite(velocity_bound_m_s)
                or velocity_bound_m_s <= 0
            ):
                raise ValueError("velocity callable requires a finite positive velocity_bound_m_s")
        elif velocity_bound_m_s is not None:
            raise ValueError("velocity_bound_m_s is only used with a velocity callable")
        self.velocity = velocity
        self.velocity_bound_m_s = velocity_bound_m_s
        self.closure = closure
        east_length, north_length, zonal_distance, meridional_distance = grid._metrics()
        self.register_buffer("_area", grid.cell_area_m2.clone())
        self.register_buffer("_east_length", east_length)
        self.register_buffer("_north_length", north_length)
        self.register_buffer("_zonal_distance", zonal_distance)
        self.register_buffer("_meridional_distance", meridional_distance)

    def _divergence(self, east: Tensor, north: Tensor) -> Tensor:
        return _metric_divergence(
            east, north, self._area.to(east), self._east_length.to(east), self._north_length.to(east),
        )

    def _winds(self, t: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        if self.velocity is None:
            return (
                _broadcast_coefficient(self.u_m_s, state, "u_m_s"),
                _broadcast_coefficient(self.v_m_s, state, "v_m_s"),
            )
        result = self.velocity(t, state)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError("velocity must return (u_m_s, v_m_s)")
        winds = []
        for name, value in zip(("u", "v"), result):
            if not isinstance(value, Tensor) or value.ndim not in (0, 5):
                raise ValueError(f"velocity {name} must be a scalar or explicit 5D tensor")
            if value.device != state.device or value.dtype != state.dtype:
                raise ValueError("velocity outputs must match the state dtype and device")
            if not bool(torch.isfinite(value).all()) or bool((value.abs() > self.velocity_bound_m_s * (1 + 1e-7)).any()):
                raise ValueError("velocity output is nonfinite or exceeds velocity_bound_m_s")
            winds.append(_broadcast_coefficient(value, state, f"velocity {name}"))
        return winds[0], winds[1]

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        _validate_state(state)
        if state.shape[-2:] != (self.grid.n_lat, self.grid.n_lon):
            raise ValueError("state spatial shape must match the global grid")
        u, v = self._winds(t, state)
        right = torch.roll(state, -1, dims=-1)
        u_face = 0.5 * (u + torch.roll(u, -1, dims=-1))
        v_face = _north_faces(v)
        east_flux = u_face * torch.where(u_face >= 0, state, right)
        v_interior = v_face[..., 1:-1, :]
        north_internal = v_interior * torch.where(
            v_interior >= 0, state[..., :-1, :], state[..., 1:, :],
        )
        boundary = torch.zeros_like(state[..., :1, :])
        north_flux = torch.cat((boundary, north_internal, boundary), dim=-2)
        tendency = -self._divergence(east_flux, north_flux)
        if self.transport_form == "advective":
            tendency = tendency + state * self._divergence(u_face, v_face)
        k = self.diffusivity_m2_s.to(state)
        east_diffusion = -k * (right - state) / self._zonal_distance.to(state)
        north_diffusion = torch.cat((
            boundary,
            -k * (state[..., 1:, :] - state[..., :-1, :]) / self._meridional_distance.to(state),
            boundary,
        ), dim=-2)
        tendency = tendency - self._divergence(east_diffusion, north_diffusion)
        if self.closure is not None:
            correction = self.closure(t, state)
            if not isinstance(correction, Tensor) or correction.shape != state.shape:
                raise ValueError("closure must return exactly the state shape")
            if correction.device != state.device or correction.dtype != state.dtype:
                raise ValueError("closure output must match the state dtype and device")
            tendency = tendency + correction
        return tendency

    def explicit_step_limit_seconds(self, safety: float = 0.5) -> float:
        """Bound resolved upwind/diffusion rates; exclude learned-source stiffness.

        This is a conservative explicit-step estimate, not a proof of RK
        positivity or long-term stability. Both inward/outward advective rates
        are bounded to support either scalar transport form. The callable-wind
        bound is deliberately pessimistic at high latitudes.
        """
        if isinstance(safety, bool) or not isinstance(safety, Real) or not math.isfinite(safety) or not 0 < safety <= 1:
            raise ValueError("safety must be finite and in (0,1]")
        area = self._area.detach()
        east_length = self._east_length.detach()
        north_length = self._north_length.detach()
        if self.velocity is not None:
            speed = self.velocity_bound_m_s
            advection_rate = (
                2 * speed * east_length + speed * (north_length[1:] + north_length[:-1])
            ) / area
            advection_max = advection_rate.max().item()
        else:
            def grid_wind(wind):
                shape = (1, 1, 1) if wind.ndim == 0 else wind.shape[:3]
                return wind.detach().to(area).expand(*shape, self.grid.n_lat, self.grid.n_lon)

            u = grid_wind(self.u_m_s)
            v = grid_wind(self.v_m_s)
            u_face = 0.5 * (u + torch.roll(u, -1, dims=-1))
            v_face = _north_faces(v)
            # Separate maxima are a safe upper bound if u/v carry different
            # singleton member/channel axes; no state allocation is required.
            east_out = u_face.clamp_min(0) + (-torch.roll(u_face, 1, dims=-1)).clamp_min(0)
            east_in = (-u_face).clamp_min(0) + torch.roll(u_face, 1, dims=-1).clamp_min(0)
            north_out = v_face[..., 1:, :].clamp_min(0) * north_length[1:] + (-v_face[..., :-1, :]).clamp_min(0) * north_length[:-1]
            north_in = (-v_face[..., 1:, :]).clamp_min(0) * north_length[1:] + v_face[..., :-1, :].clamp_min(0) * north_length[:-1]
            advection_max = (
                (torch.maximum(east_out, east_in) * east_length / area).max().item()
                + (torch.maximum(north_out, north_in) / area).max().item()
            )
        k = self.diffusivity_m2_s.detach().to(area)
        diffusion_rate = k * (
            2 * east_length / self._zonal_distance.detach()
            + (north_length[1:] + north_length[:-1]) / self._meridional_distance.detach()
        ) / area
        rate = advection_max + diffusion_rate.max().item()
        return safety / rate if rate > 0 else math.inf
