import pytest
import torch

from physical_guide_flowmatching import (
    GlobalLatLonGrid, SphericalTransport2D, SphericalFluxSourceClosure2D,
    ODESolverConfig, integrate_ensemble, area_weighted_mean, correct_forecast,
)


def test_spherical_closure_conserves_weighted_budget_and_condition_gradients():
    torch.manual_seed(37)
    grid = GlobalLatLonGrid(6, 12, radius_m=10.0)
    closure = SphericalFluxSourceClosure2D(
        1, grid, hidden_channels=4, condition_channels=2,
    ).double()
    torch.nn.init.normal_(closure.net[-1].weight, std=0.1)
    context = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
    initial = torch.rand(1, 1, 1, 6, 12, dtype=torch.float64).expand(1, 3, 1, 6, 12)
    rhs = SphericalTransport2D(grid, closure=closure.conditioned_on(context))
    path = integrate_ensemble(rhs, initial, [0.0, 0.2], ODESolverConfig(step_seconds=0.05))
    torch.testing.assert_close(
        area_weighted_mean(path[-1], grid), area_weighted_mean(initial, grid),
        rtol=0, atol=1e-12,
    )
    assert not torch.allclose(path[-1, :, 0], path[-1, :, 1], atol=1e-10, rtol=0)
    separate_rhs = SphericalTransport2D(grid, closure=closure.conditioned_on(context[:, 1:2]))
    separate = integrate_ensemble(separate_rhs, initial[:, 1:2], [0.0, 0.2], ODESolverConfig(step_seconds=0.05))
    torch.testing.assert_close(path[:, :, 1:2], separate, rtol=0, atol=1e-12)
    path[-1].square().mean().backward()
    assert context.grad is not None and (context.grad.abs().sum(-1) > 1e-12).all()


def test_learned_source_can_change_temperature_budget_independently_of_flux():
    grid = GlobalLatLonGrid(6, 12)
    closure = SphericalFluxSourceClosure2D(
        2, grid, source_scale=[0.0, 0.01],
    ).double()
    with torch.no_grad():
        # Last third of channels are source, separate from x/y flux outputs.
        closure.net[-1].bias[4:] = 1.0
    state = torch.ones(1, 2, 2, 6, 12, dtype=torch.float64)
    result = closure(state.new_tensor(0.0), state)
    expected = torch.tensor([0.0, 0.01], dtype=state.dtype).expand(1, 2, 2)
    torch.testing.assert_close(area_weighted_mean(result, grid), expected, rtol=1e-7, atol=1e-12)


def test_nudging_uses_spherical_area_budget_when_requested():
    grid = GlobalLatLonGrid(6, 12)
    initial = torch.zeros(1, 1, 6, 12, dtype=torch.float64)
    forecast = torch.zeros(1, 2, 1, 6, 12, dtype=torch.float64)
    forecast[..., 0, :] = 1.0
    rhs = SphericalTransport2D(grid)
    result = correct_forecast(
        rhs, initial, forecast, [0.0, 1.0], nudging_rate_s=1.0,
        conserve_spatial_mean=True, config=ODESolverConfig(step_seconds=0.1),
    )
    torch.testing.assert_close(area_weighted_mean(result[-1], grid), torch.zeros(1, 2, 1, dtype=initial.dtype), rtol=0, atol=1e-14)
    # An unweighted projection would incorrectly make this arithmetic mean 0.
    assert result[-1].mean().abs() > 1e-3


def test_spherical_closure_padding_does_not_wrap_north_to_south():
    grid = GlobalLatLonGrid(8, 16)
    closure = SphericalFluxSourceClosure2D(1, grid, hidden_channels=2, source_scale=1.0).double()
    with torch.no_grad():
        for parameter in closure.parameters():
            parameter.fill_(0.1)
    first = torch.ones(1, 1, 1, 8, 16, dtype=torch.float64)
    second = first.clone()
    second[..., 0, :] += 2
    t = first.new_tensor(0.0)
    a, b = closure(t, first), closure(t, second)
    torch.testing.assert_close(a[..., -1, :], b[..., -1, :], rtol=0, atol=0)


@pytest.mark.parametrize("weights", [torch.ones(2, 2), torch.full((6, 12), -1.0)])
def test_invalid_nudging_area_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="spatial_weights"):
        correct_forecast(
            lambda t, x: torch.zeros_like(x), torch.zeros(1, 1, 6, 12),
            torch.ones(1, 2, 1, 6, 12), [0.0, 1.0],
            nudging_rate_s=1, spatial_weights=weights,
        )
