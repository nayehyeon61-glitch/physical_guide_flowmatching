"""Functional checks for physical-time ensemble integration and anchoring."""

import math

import pytest
import torch

from physical_guide_flowmatching import (
    ConservativeFluxClosure2D,
    ConservativeTransport2D,
    ODESolverConfig,
    PeriodicCartesianGrid,
    correct_forecast,
    integrate_ensemble,
)


DTYPE = torch.float64


def zero_rhs(t, state):
    return torch.zeros_like(state)


def test_periodic_diffusion_matches_exact_discrete_mode():
    height, width = 8, 10
    y = torch.arange(height, dtype=DTYPE)
    x = torch.arange(width, dtype=DTYPE)
    mode = torch.cos(2 * math.pi * y / height)[:, None] * torch.cos(
        2 * math.pi * x / width
    )[None, :]
    amplitudes = torch.tensor([0.2, 0.4, 0.7], dtype=DTYPE).reshape(1, 3, 1, 1, 1)
    initial = 2.0 + amplitudes * mode
    grid = PeriodicCartesianGrid(dx_m=1.0, dy_m=1.0)
    diffusivity = 0.1
    rhs = ConservativeTransport2D(grid, diffusivity_m2_s=diffusivity).double()
    times = torch.tensor([0.0, 0.4, 0.8], dtype=DTYPE)

    actual = integrate_ensemble(
        rhs, initial, times, ODESolverConfig(step_seconds=0.05)
    )

    eigenvalue = diffusivity * (
        2 * math.cos(2 * math.pi / width) - 2
        + 2 * math.cos(2 * math.pi / height) - 2
    )
    expected = 2.0 + torch.exp(eigenvalue * times).reshape(-1, 1, 1, 1, 1, 1) * (
        initial - 2.0
    )
    assert actual.shape == (3, 1, 3, 1, height, width)
    assert torch.isfinite(actual).all()
    assert not torch.equal(actual[-1], initial)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)


def test_variable_wind_transport_preserves_each_member_mean():
    generator = torch.Generator().manual_seed(12)
    initial = torch.rand((2, 3, 2, 6, 7), generator=generator, dtype=DTYPE)
    wind_u = torch.randn((2, 3, 1, 6, 7), generator=generator, dtype=DTYPE) * 0.2
    wind_v = torch.randn((2, 3, 1, 6, 7), generator=generator, dtype=DTYPE) * 0.2
    rhs = ConservativeTransport2D(
        PeriodicCartesianGrid(dx_m=1.0, dy_m=1.0),
        u_m_s=wind_u,
        v_m_s=wind_v,
        diffusivity_m2_s=0.05,
    ).double()

    trajectory = integrate_ensemble(
        rhs, initial, [0.0, 0.2], ODESolverConfig(step_seconds=0.01)
    )

    torch.testing.assert_close(
        trajectory[-1].mean(dim=(-2, -1)),
        initial.mean(dim=(-2, -1)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_fixed_step_members_match_separate_solves():
    generator = torch.Generator().manual_seed(23)
    initial = torch.rand((2, 3, 1, 5, 6), generator=generator, dtype=DTYPE)
    rhs = ConservativeTransport2D(
        PeriodicCartesianGrid(dx_m=1.0, dy_m=1.0),
        u_m_s=0.3,
        v_m_s=-0.1,
        diffusivity_m2_s=0.02,
    ).double()
    config = ODESolverConfig(step_seconds=0.025)
    together = integrate_ensemble(rhs, initial, [0.0, 0.25], config)

    separate = torch.cat(
        [
            integrate_ensemble(rhs, initial[:, member : member + 1], [0.0, 0.25], config)
            for member in range(initial.shape[1])
        ],
        dim=2,
    )
    torch.testing.assert_close(together, separate, rtol=1e-13, atol=1e-13)


def test_nudging_integrates_from_observation_without_double_advancing():
    observed = torch.full((2, 1, 4, 5), 3.0, dtype=DTYPE)
    forecast = torch.full((2, 3, 1, 4, 5), 8.0, dtype=DTYPE)
    result = correct_forecast(
        zero_rhs,
        observed,
        forecast,
        [0.0, 2.0],
        nudging_rate_s=0.0,
        config=ODESolverConfig(step_seconds=0.1),
    )

    expected = observed[:, None].expand_as(forecast)
    torch.testing.assert_close(result[0], expected)
    torch.testing.assert_close(result[-1], expected)
    assert not torch.equal(result[-1], forecast)


def test_linear_reference_nudging_matches_analytic_solution():
    initial = torch.full((1, 1, 4, 4), 2.0, dtype=DTYPE)
    target = torch.tensor([3.0, 5.0], dtype=DTYPE).reshape(1, 2, 1, 1, 1).expand(
        1, 2, 1, 4, 4
    )
    duration = 1.2
    rate = 0.7
    trajectory = correct_forecast(
        zero_rhs,
        initial,
        target,
        [0.0, duration],
        nudging_rate_s=rate,
        conserve_spatial_mean=False,
        config=ODESolverConfig(step_seconds=0.02),
    )

    fraction = 1 - (1 - math.exp(-rate * duration)) / (rate * duration)
    expected = initial[:, None] + fraction * (target - initial[:, None])
    torch.testing.assert_close(trajectory[-1], expected, rtol=2e-9, atol=2e-9)


def test_default_nudging_preserves_mean_and_endpoint_gradients():
    generator = torch.Generator().manual_seed(32)
    initial = torch.rand((1, 1, 5, 6), generator=generator, dtype=DTYPE, requires_grad=True)
    forecast = torch.rand(
        (1, 3, 1, 5, 6), generator=generator, dtype=DTYPE, requires_grad=True
    )
    trajectory = correct_forecast(
        zero_rhs,
        initial,
        forecast,
        [0.0, 0.5],
        nudging_rate_s=2.0,
        config=ODESolverConfig(step_seconds=0.025),
    )
    initial_mean = initial[:, None].expand_as(forecast).mean(dim=(-2, -1))
    torch.testing.assert_close(
        trajectory[-1].mean(dim=(-2, -1)), initial_mean, rtol=1e-12, atol=1e-12
    )
    assert not torch.equal(trajectory[-1, :, 0], trajectory[-1, :, 1])

    trajectory[-1].square().mean().backward()
    for tensor in (initial, forecast):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 1e-10


def test_gradients_reach_conservative_closure_and_learned_flux_conserves_mean():
    torch.manual_seed(4)
    grid = PeriodicCartesianGrid(dx_m=1.0, dy_m=1.0)
    closure = ConservativeFluxClosure2D(channels=1, grid=grid, hidden_channels=4).double()
    rhs = ConservativeTransport2D(grid, diffusivity_m2_s=0.03, closure=closure).double()
    initial = torch.rand((1, 2, 1, 5, 6), dtype=DTYPE, requires_grad=True)
    target_pattern = torch.randn_like(initial)
    trajectory = integrate_ensemble(
        rhs, initial, [0.0, 0.2], ODESolverConfig(step_seconds=0.025)
    )
    loss = (trajectory[-1] * target_pattern).mean()
    loss.backward()

    assert initial.grad is not None and initial.grad.abs().sum() > 0
    gradients = [parameter.grad for parameter in closure.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 1e-10 for gradient in gradients)

    torch.optim.SGD(closure.parameters(), lr=0.1).step()
    learned_tendency = closure(torch.tensor(0.0, dtype=DTYPE), initial.detach())
    assert learned_tendency.abs().max() > 1e-10
    torch.testing.assert_close(
        learned_tendency.sum(dim=(-2, -1)),
        torch.zeros_like(learned_tendency.sum(dim=(-2, -1))),
        rtol=0,
        atol=1e-12,
    )


def test_adaptive_solver_matches_known_decay():
    generator = torch.Generator().manual_seed(8)
    initial = torch.rand((1, 3, 1, 4, 4), generator=generator, dtype=DTYPE)
    result = integrate_ensemble(
        lambda t, state: -state,
        initial,
        [0.0, 0.7],
        ODESolverConfig(method="dopri5", step_seconds=0.1, rtol=1e-9, atol=1e-11),
    )
    torch.testing.assert_close(result[-1], initial * math.exp(-0.7), rtol=2e-8, atol=2e-10)


def test_fixed_step_refinement_reduces_integration_error():
    initial = torch.ones((1, 2, 1, 4, 4), dtype=DTYPE)
    exact = initial * math.exp(-1.3 * 2.0)
    errors = []
    for step in (0.4, 0.2):
        trajectory = integrate_ensemble(
            lambda t, state: -1.3 * state,
            initial,
            [0.0, 2.0],
            ODESolverConfig(step_seconds=step),
        )
        errors.append((trajectory[-1] - exact).abs().max())
    assert errors[0] > 0
    assert errors[1] < errors[0] / 10


def test_transport_stability_bound_rejects_an_oversized_step():
    rhs = ConservativeTransport2D(
        PeriodicCartesianGrid(dx_m=1.0, dy_m=1.0), u_m_s=2.0
    ).double()
    with pytest.raises(ValueError):
        integrate_ensemble(
            rhs,
            torch.ones((1, 2, 1, 4, 4), dtype=DTYPE),
            [0.0, 1.0],
            ODESolverConfig(step_seconds=1.0),
        )


@pytest.mark.parametrize("times", [[0.0, 0.0], [1.0, 0.0], [0.0, float("nan")]])
def test_invalid_physical_times_are_rejected(times):
    with pytest.raises(ValueError):
        integrate_ensemble(zero_rhs, torch.ones((1, 2, 1, 4, 4), dtype=DTYPE), times)


def test_zero_step_size_is_rejected():
    with pytest.raises(ValueError):
        config = ODESolverConfig(step_seconds=0.0)
        integrate_ensemble(
            zero_rhs, torch.ones((1, 2, 1, 4, 4), dtype=DTYPE), [0.0, 1.0], config
        )


def test_member_axis_is_required():
    with pytest.raises(ValueError):
        integrate_ensemble(zero_rhs, torch.ones((2, 1, 4, 4), dtype=DTYPE), [0.0, 1.0])
