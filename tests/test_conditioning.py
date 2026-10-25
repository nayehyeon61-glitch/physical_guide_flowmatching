"""The recommended path uses FM output as closure context, not a new initial time."""

import pytest
import torch

from physical_guide_flowmatching import (
    ConservativeFluxClosure2D,
    ConservativeTransport2D,
    ODESolverConfig,
    PeriodicCartesianGrid,
    integrate_ensemble,
)


def test_conditioned_members_evolve_independently_and_backpropagate_to_fm_context():
    torch.manual_seed(29)
    grid = PeriodicCartesianGrid(1.0, 1.0)
    closure = ConservativeFluxClosure2D(
        1, grid, hidden_channels=4, condition_channels=2,
    ).double()
    # Emulate a pretrained nonzero closure. With the documented zero init the
    # first training step updates the last layer before upstream context.
    torch.nn.init.normal_(closure.net[-1].weight, std=0.03)
    context = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
    initial = torch.rand(1, 1, 1, 5, 6, dtype=torch.float64).expand(1, 3, 1, 5, 6)
    rhs = ConservativeTransport2D(grid, closure=closure.conditioned_on(context))
    config = ODESolverConfig(step_seconds=0.05)
    trajectory = integrate_ensemble(rhs, initial, [0.0, 0.2], config)

    assert not torch.allclose(trajectory[-1, :, 0], trajectory[-1, :, 1], atol=1e-10, rtol=0)
    torch.testing.assert_close(
        trajectory[-1].mean((-2, -1)), initial.mean((-2, -1)), atol=1e-12, rtol=0,
    )
    member_rhs = ConservativeTransport2D(
        grid, closure=closure.conditioned_on(context[:, 1:2]),
    )
    member = integrate_ensemble(member_rhs, initial[:, 1:2], [0.0, 0.2], config)
    torch.testing.assert_close(member, trajectory[:, :, 1:2], atol=1e-12, rtol=0)
    trajectory[-1].square().mean().backward()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert (context.grad.abs().sum(-1) > 1e-12).all()


def test_spatial_context_map_is_differentiable():
    torch.manual_seed(3)
    closure = ConservativeFluxClosure2D(
        1, PeriodicCartesianGrid(1.0, 1.0), hidden_channels=4, condition_channels=1,
    ).double()
    torch.nn.init.normal_(closure.net[-1].weight, std=0.1)
    context = torch.randn(1, 2, 1, 4, 5, dtype=torch.float64, requires_grad=True)
    state = torch.ones(1, 2, 1, 4, 5, dtype=torch.float64)
    tendency = closure.conditioned_on(context)(state.new_tensor(0.0), state)
    tendency.square().mean().backward()
    assert context.grad is not None and context.grad.abs().sum() > 1e-12
    torch.testing.assert_close(tendency.sum((-2, -1)), torch.zeros(1, 2, 1, dtype=state.dtype), atol=1e-12, rtol=0)


def test_condition_shape_cannot_broadcast_across_members():
    closure = ConservativeFluxClosure2D(
        1, PeriodicCartesianGrid(1.0, 1.0), condition_channels=2,
    ).double()
    state = torch.ones(1, 3, 1, 4, 5, dtype=torch.float64)
    with pytest.raises(ValueError):
        closure.conditioned_on(torch.ones(1, 1, 2, dtype=state.dtype))(state.new_tensor(0.0), state)
