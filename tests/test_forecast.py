import math

import pytest
import torch

from physical_guide_flowmatching.forecast import forecast_times_seconds, integrate_windowed
from physical_guide_flowmatching.ode import ODESolverConfig, integrate_ensemble


def test_fifteen_day_schedule_has_sixty_one_six_hour_outputs():
    times = forecast_times_seconds()
    assert len(times) == 61
    assert times[0] == 0 and times[-1] == 1296000
    torch.testing.assert_close(times.diff(), torch.full((60,), 21600.0, dtype=times.dtype))
    assert forecast_times_seconds(1, 7).tolist() == [0, 25200, 50400, 75600, 86400]


def test_windowed_integration_keeps_time_and_matches_direct_solve():
    initial = torch.ones(1, 2, 1, 3, 4, dtype=torch.float64)
    times = torch.tensor([0, 0.2, 0.4, 0.6], dtype=initial.dtype)
    config = ODESolverConfig(step_seconds=0.01)
    rhs = lambda t, state: t * torch.ones_like(state)
    expected = initial[None] + 0.5 * times.square().reshape(-1, 1, 1, 1, 1, 1)
    actual = integrate_windowed(rhs, initial, times, config)
    direct = integrate_ensemble(rhs, initial, times, config)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual, direct, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_windowed_gradients_reach_captured_condition_and_rhs_parameters(use_checkpoint):
    class ConditionedRHS(torch.nn.Module):
        def __init__(self, context):
            super().__init__()
            self.rate = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
            self.context = context

        def forward(self, t, state):
            return -self.rate * state + self.context * t

    initial = torch.ones(1, 2, 1, 3, 4, dtype=torch.float64, requires_grad=True)
    context = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    rhs = ConditionedRHS(context)
    times = torch.tensor([0, 0.2, 0.4, 0.6], dtype=initial.dtype)
    actual = integrate_windowed(
        rhs, initial, times, ODESolverConfig(step_seconds=0.025),
        checkpoint_segments=use_checkpoint,
    )
    expected_final = math.exp(-0.12) + 0.3 * (0.6 / 0.2 - (1 - math.exp(-0.12)) / 0.2**2)
    torch.testing.assert_close(actual[-1], torch.full_like(initial, expected_final), rtol=1e-10, atol=1e-10)
    actual[-1].sum().backward()
    for value in (initial, context, rhs.rate):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 1e-8


def test_windowing_does_not_detach_future_from_initial():
    initial = torch.ones(1, 1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
    times = torch.tensor([0, 1, 2, 3], dtype=initial.dtype)
    # Each one-second window fits max_num_steps, whereas a full three-second
    # solve at .1 would exceed it. This limits per-solve work, not total cost.
    config = ODESolverConfig(step_seconds=0.1, max_num_steps=11)
    result = integrate_windowed(lambda t, x: -0.1 * x, initial, times, config)
    result[-1].sum().backward()
    torch.testing.assert_close(initial.grad, torch.full_like(initial, math.exp(-0.3)), atol=1e-9, rtol=0)
