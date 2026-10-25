import pytest
import torch
import torch_dct

from physical_guide_flowmatching import decode_ensemble_forecast


def test_inverse_dct_and_physical_units_preserve_ensemble_gradient():
    torch.manual_seed(19)
    state = torch.randn(2, 3, 2, 5, 6, dtype=torch.float64)
    mean = torch.tensor([280.0, 5.0], dtype=torch.float64)
    scale = torch.tensor([10.0, 2.0], dtype=torch.float64)
    coeff = torch_dct.dct_2d(state.flatten(0, 1), norm="ortho").reshape_as(state)
    coeff.requires_grad_()
    actual = decode_ensemble_forecast(
        coeff, torch.nn.Identity(),
        lambda x: torch_dct.idct_2d(x, norm="ortho"),
        channel_mean=mean, channel_scale=scale,
    )
    expected = state * scale.reshape(1, 1, 2, 1, 1) + mean.reshape(1, 1, 2, 1, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual.square().mean().backward()
    assert coeff.grad is not None
    assert torch.isfinite(coeff.grad).all() and coeff.grad.abs().sum() > 0


def test_normalization_scale_must_be_positive():
    with pytest.raises(ValueError):
        decode_ensemble_forecast(
            torch.ones(1, 2, 1, 4, 4), torch.nn.Identity(), torch.nn.Identity(),
            channel_mean=torch.tensor([0.0]), channel_scale=torch.tensor([0.0]),
        )
