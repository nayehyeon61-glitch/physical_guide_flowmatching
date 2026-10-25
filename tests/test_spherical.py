"""Geometry and transport checks for the ClimODE/WeatherBench global grid."""

import math

import pytest
import torch
from torch import nn

from physical_guide_flowmatching import ODESolverConfig, integrate_ensemble
from physical_guide_flowmatching.spherical import (
    GlobalLatLonGrid,
    SphericalTransport2D,
    area_weighted_mean,
)


DTYPE = torch.float64


def test_weatherbench_global_grid_coordinates_and_total_surface_area():
    grid = GlobalLatLonGrid()
    assert grid.cell_area_m2.shape == (32, 64)
    torch.testing.assert_close(
        grid.latitude_degrees,
        torch.arange(32, dtype=DTYPE) * 5.625 - 87.1875,
    )
    torch.testing.assert_close(
        grid.longitude_degrees, torch.arange(64, dtype=DTYPE) * 5.625,
    )
    expected_area = 4 * math.pi * grid.radius_m**2
    assert math.isclose(float(grid.cell_area_m2.sum()), expected_area, rel_tol=2e-14)
    assert (grid.cell_area_m2 > 0).all()
    assert grid.cell_area_m2[16, 0] > grid.cell_area_m2[0, 0]
    grid.validate_coordinates(grid.latitude_degrees, grid.longitude_degrees)


def test_solid_body_zonal_rotation_preserves_a_constant_concentration():
    grid = GlobalLatLonGrid(n_lat=6, n_lon=12)
    u = (20 * torch.cos(torch.deg2rad(grid.latitude_degrees))).reshape(1, 1, 1, 6, 1)
    rhs = SphericalTransport2D(grid, u_m_s=u, transport_form="conservative").double()
    state = torch.full((2, 3, 1, 6, 12), 4.0, dtype=DTYPE)
    tendency = rhs(state.new_tensor(0.0), state)
    torch.testing.assert_close(tendency, torch.zeros_like(state), atol=1e-16, rtol=0)


def test_transport_and_diffusion_preserve_each_member_area_weighted_mean():
    generator = torch.Generator().manual_seed(17)
    grid = GlobalLatLonGrid(n_lat=6, n_lon=12)
    state = torch.rand((2, 3, 2, 6, 12), generator=generator, dtype=DTYPE)
    u = 20 * torch.randn((2, 3, 1, 6, 12), generator=generator, dtype=DTYPE)
    v = 10 * torch.randn((2, 3, 1, 6, 12), generator=generator, dtype=DTYPE)
    rhs = SphericalTransport2D(
        grid, u_m_s=u, v_m_s=v, diffusivity_m2_s=2e4,
        transport_form="conservative",
    ).double()
    tendency = rhs(state.new_tensor(0.0), state)
    torch.testing.assert_close(
        area_weighted_mean(tendency, grid), torch.zeros(2, 3, 2, dtype=DTYPE),
        atol=1e-18, rtol=0,
    )
    config = ODESolverConfig(step_seconds=30.0)
    trajectory = integrate_ensemble(rhs, state, [0.0, 300.0], config)
    torch.testing.assert_close(
        area_weighted_mean(trajectory[-1], grid), area_weighted_mean(state, grid),
        atol=1e-12, rtol=1e-12,
    )
    assert not torch.equal(trajectory[-1], state)


def test_advective_form_keeps_constant_scalar_under_divergent_winds():
    generator = torch.Generator().manual_seed(18)
    grid = GlobalLatLonGrid(n_lat=6, n_lon=12)
    state = torch.full((1, 2, 2, 6, 12), 285.0, dtype=DTYPE)
    u = 20 * torch.randn((1, 2, 1, 6, 12), generator=generator, dtype=DTYPE)
    v = 10 * torch.randn((1, 2, 1, 6, 12), generator=generator, dtype=DTYPE)
    advective = SphericalTransport2D(
        grid, u_m_s=u, v_m_s=v, diffusivity_m2_s=2e4,
        transport_form="advective",
    ).double()
    conservative = SphericalTransport2D(
        grid, u_m_s=u, v_m_s=v, transport_form="conservative",
    ).double()
    actual = advective(state.new_tensor(0.0), state)
    torch.testing.assert_close(actual, torch.zeros_like(state), atol=1e-16, rtol=0)
    assert conservative(state.new_tensor(0.0), state).abs().max() > 1e-6


def test_latitude_does_not_wrap_but_longitude_does():
    grid = GlobalLatLonGrid(n_lat=6, n_lon=12)
    south_only = torch.zeros((1, 1, 1, 6, 12), dtype=DTYPE)
    south_only[..., 0, :] = 1
    northward = SphericalTransport2D(grid, v_m_s=10.0).double()
    tendency = northward(south_only.new_tensor(0.0), south_only)
    assert tendency[..., 1, :].abs().sum() > 0
    torch.testing.assert_close(
        tendency[..., -1, :], torch.zeros_like(tendency[..., -1, :]), atol=0, rtol=0,
    )
    seam_only = torch.zeros_like(south_only)
    seam_only[..., -1] = 1
    eastward = SphericalTransport2D(grid, u_m_s=10.0).double()
    seam_tendency = eastward(seam_only.new_tensor(0.0), seam_only)
    assert (seam_tendency[..., 0] > 0).all()


def test_dynamic_velocity_preserves_gradients_to_state_and_velocity_parameters():
    class StateVelocity(nn.Module):
        def __init__(self):
            super().__init__()
            self.speed = nn.Parameter(torch.tensor(5.0, dtype=DTYPE))

        def forward(self, t, state):
            return self.speed * state.tanh(), torch.zeros_like(state)

    torch.manual_seed(19)
    grid = GlobalLatLonGrid(n_lat=6, n_lon=12)
    velocity = StateVelocity()
    rhs = SphericalTransport2D(
        grid, velocity=velocity, velocity_bound_m_s=10.0,
        transport_form="advective",
    ).double()
    state = torch.rand((1, 2, 1, 6, 12), dtype=DTYPE, requires_grad=True)
    trajectory = integrate_ensemble(
        rhs, state, [0.0, 300.0], ODESolverConfig(step_seconds=60.0),
    )
    trajectory[-1].square().mean().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert state.grad.abs().sum() > 1e-8
    assert velocity.speed.grad is not None and torch.isfinite(velocity.speed.grad)
    assert velocity.speed.grad.abs() > 1e-12


def test_polar_cell_width_limits_explicit_step_with_strong_fixed_winds():
    rhs = SphericalTransport2D(GlobalLatLonGrid(), u_m_s=100.0)
    limit = rhs.explicit_step_limit_seconds()
    assert 10.0 < limit < 300.0


def test_coordinate_order_and_duplicate_endpoint_are_not_silently_accepted():
    grid = GlobalLatLonGrid()
    with pytest.raises(ValueError):
        grid.validate_coordinates(grid.latitude_degrees.flip(0), grid.longitude_degrees)
    with pytest.raises(ValueError):
        grid.validate_coordinates(
            grid.latitude_degrees,
            torch.cat((grid.longitude_degrees, torch.tensor([360.0], dtype=DTYPE))),
        )


def test_state_shape_must_match_grid():
    rhs = SphericalTransport2D(GlobalLatLonGrid(n_lat=6, n_lon=12))
    with pytest.raises(ValueError):
        rhs(torch.tensor(0.0), torch.ones(1, 2, 1, 5, 12))


def test_unknown_transport_form_is_rejected():
    with pytest.raises(ValueError):
        SphericalTransport2D(GlobalLatLonGrid(), transport_form="temperature")
