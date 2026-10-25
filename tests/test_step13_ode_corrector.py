import math

import numpy as np
import pytest
import torch

from pgcfm import contract
from pgcfm.ode_corrector import CORRECTED_DYNAMIC_INDICES, correct_sequence_ode


def inputs(members=2, windows=1):
    x0 = torch.zeros(1, contract.N_DYNAMIC, 6, 12, dtype=torch.float64)
    d = torch.zeros(1, members, windows, contract.N_DYNAMIC, 6, 12, dtype=x0.dtype)
    mean = np.zeros(contract.N_CHANNELS)
    std = np.ones(contract.N_CHANNELS)
    return x0, d, mean, std


def fast_correct(x0, d, mean, std, **kwargs):
    return correct_sequence_ode(
        x0, d, mean, std, dt_seconds=1.0, step_seconds=0.05,
        nudging_timescale_seconds=1.0, **kwargs,
    )


def test_constant_reference_is_fixed_and_unselected_channels_pass_through_exactly():
    x0, d, mean, std = inputs(windows=2)
    torch.manual_seed(3)
    x0[:] = torch.rand_like(x0)
    passthrough = [index for index in range(contract.N_DYNAMIC) if index not in CORRECTED_DYNAMIC_INDICES]
    d[:, :, :, passthrough] = torch.randn_like(d[:, :, :, passthrough])
    out, diag = fast_correct(x0, d, mean, std, transport_strength=0)
    torch.testing.assert_close(out[:, :, :, list(CORRECTED_DYNAMIC_INDICES)], torch.zeros_like(out[:, :, :, list(CORRECTED_DYNAMIC_INDICES)]), rtol=0, atol=0)
    assert torch.equal(out[:, :, :, passthrough], d[:, :, :, passthrough])
    assert "tisr" in diag["passthrough_channels"]
    assert diag["hard_constraints_applied_here"] is False


def test_nudging_control_matches_analytic_and_preserves_member_pairing():
    x0, d, mean, std = inputs()
    q500 = contract.DYNAMIC_CHANNELS.index(contract.CHANNEL_INDEX["q500"])
    d[0, 0, 0, q500] = 0.2
    d[0, 1, 0, q500] = -0.4
    out, _ = fast_correct(x0, d, mean, std, transport_strength=0)
    factor = math.exp(-1.0)  # 1-(1-exp(-1))/1
    torch.testing.assert_close(out[:, :, :, q500], d[:, :, :, q500] * factor, rtol=1e-7, atol=1e-9)
    alone, _ = fast_correct(x0, d[:, 1:2], mean, std, transport_strength=0)
    torch.testing.assert_close(alone, out[:, 1:2], rtol=0, atol=0)


def test_pressure_level_winds_and_exact_std_eps_drive_nontrivial_transport():
    x0, d, mean, std = inputs()
    q = contract.CHANNEL_INDEX["q500"]
    u = contract.CHANNEL_INDEX["u500"]
    std[q] = 1e-8
    mean[q] = 2e-6
    mean[u] = 40.0
    x0[:, q] = torch.sin(torch.arange(12, dtype=x0.dtype) * 2 * math.pi / 12)
    physical, diag = fast_correct(x0, d, mean, std, transport_strength=1)
    control, _ = fast_correct(x0, d, mean, std, transport_strength=0)
    assert (physical[:, :, :, q] - control[:, :, :, q]).abs().max() > 1e-7
    assert diag["maximum_raw_wind_m_s"] == 40.0
    assert diag["wind_clipped_fraction"] == 0.0
    assert torch.isfinite(physical).all()


def test_gradients_reach_initial_and_fm_forecast_without_second_decoding():
    x0, d, mean, std = inputs()
    x0.requires_grad_()
    d[..., list(CORRECTED_DYNAMIC_INDICES), :, :] = 0.2
    d.requires_grad_()
    out, _ = fast_correct(x0, d, mean, std, transport_strength=0)
    final = x0[:, None, None] + out
    final[..., list(CORRECTED_DYNAMIC_INDICES), :, :].square().mean().backward()
    assert x0.grad is not None and x0.grad.abs().sum() > 0
    assert d.grad is not None and d.grad.abs().sum() > 0
    assert torch.isfinite(x0.grad).all() and torch.isfinite(d.grad).all()


def test_excessive_winds_are_reported_and_hard_constraints_are_left_to_caller():
    x0, d, mean, std = inputs()
    mean[contract.CHANNEL_INDEX["u500"]] = 1000.0
    d[..., contract.CHANNEL_INDEX["q500"], :, :] = -0.2
    out, diag = fast_correct(x0, d, mean, std, transport_strength=0, wind_cap_m_s=80)
    assert diag["maximum_raw_wind_m_s"] == 1000.0
    assert diag["wind_clipped_fraction"] > 0
    assert out[..., contract.CHANNEL_INDEX["q500"], :, :].max() < 0


def test_invalid_shape_and_time_configuration_are_rejected():
    x0, d, mean, std = inputs()
    with pytest.raises(ValueError, match="increments"):
        fast_correct(x0, d[:, 0], mean, std)
    with pytest.raises(ValueError, match="dt_seconds"):
        correct_sequence_ode(x0, d, mean, std, dt_seconds=0)
