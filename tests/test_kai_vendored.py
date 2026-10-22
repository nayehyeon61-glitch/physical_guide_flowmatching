import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "kai"))
from src.inference import AutoregressiveInference  # noqa: E402


def test_rollout_lead_i_is_day_i_forecast():
    # Model adds 1, so lead i must equal IC + i.
    stub = types.SimpleNamespace(
        model=lambda x: x + 1.0,
        means_t=torch.zeros(2),
        stds_t=torch.ones(2),
    )
    data = torch.zeros(4, 2, 3, 3)
    out = AutoregressiveInference._rollout_one_ic(stub, data)
    assert out.shape == (4, 2, 3, 3)
    for i in range(4):
        assert (out[i] == i).all()
