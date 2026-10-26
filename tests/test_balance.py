import numpy as np
import torch

from pgcfm import contract
from pgcfm.balance import BalanceProjector, PositiveMass

LAT = np.linspace(-87.1875, 87.1875, 32)


def _stats():
    rng = np.random.default_rng(0)
    return rng.uniform(1, 2, contract.N_DYNAMIC), rng.uniform(0.5, 1, contract.N_DYNAMIC)


def test_positive_mass_removes_negatives_and_keeps_the_mass_of_every_level():
    mean, std = _stats()
    pm = PositiveMass(mean, std, LAT)
    x = torch.randn(3, contract.N_DYNAMIC, 32, 64) * 2  # moisture values well below zero in places
    assert pm.negative_fraction(x) > 0.05
    y = pm(x)
    assert pm.negative_fraction(y) == 0.0
    assert torch.allclose(pm.mass(y), pm.mass(x), rtol=1e-5)
    other = [i for i in range(contract.N_DYNAMIC) if i not in pm.idx]
    assert torch.equal(y[:, other], x[:, other])


def test_balance_projector_starts_as_the_identity_on_physical_states():
    mean, std = _stats()
    P = BalanceProjector(torch.zeros(8, 32, 64), mean, std, LAT, width=8, modes=(4, 4), n_layers=2, bound=0.8)
    x0 = torch.zeros(2, contract.N_DYNAMIC, 32, 64)  # every moisture value = its mean > 0
    d = 0.01 * torch.randn(2, 3, contract.N_DYNAMIC, 32, 64)
    assert torch.allclose(P(x0, d), d, atol=1e-6)
    assert P.lipschitz_bound() == 1.0  # the head starts at zero


def test_balance_projector_keeps_the_moisture_mass_of_its_input_frames():
    mean, std = _stats()
    P = BalanceProjector(torch.zeros(8, 32, 64), mean, std, LAT, width=8, modes=(4, 4), n_layers=2, bound=0.8)
    torch.nn.init.normal_(P.net.head.weight, std=0.5)  # a correction that moves the global moisture
    x0 = torch.zeros(2, contract.N_DYNAMIC, 32, 64)
    d = 0.3 * torch.randn(2, 3, contract.N_DYNAMIC, 32, 64)
    out = P(x0, d)
    f_in, f_out = (x0[:, None] + d).flatten(0, 1), (x0[:, None] + out).flatten(0, 1)
    assert not torch.allclose(out, d, atol=1e-3)
    assert P.pos.negative_fraction(f_out) == 0.0
    assert torch.allclose(P.pos.mass(f_out), P.pos.mass(f_in), rtol=1e-5)
