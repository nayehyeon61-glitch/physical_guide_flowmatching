import numpy as np
import torch

from pgcfm import contract
from pgcfm.cfm import TISR_DYN
from pgcfm.coupling import PhysicsConfig, PhysicsGuide
from pgcfm.dyncore import (DynCore, DynCoreConfig, StrongPhysics, estimate_strong_scales, kinetic_energy)
from pgcfm.physics import grid as pgrid

LAT = np.linspace(-87.1875, 87.1875, 32)
N = contract.N_DYNAMIC


def _stats():
    """Atmosphere-like channel means and stds (SI), so the physics guide's pressure mask is not empty."""
    phys = {"z": (5e4, 1e3), "q": (2e-3, 1e-3), "t": (250.0, 10.0), "u": (0.0, 10.0), "v": (0.0, 5.0),
            "t2m": (280.0, 10.0), "sp": (1e5, 1e3), "tisr": (1e6, 5e5), "tcwv": (20.0, 10.0), "orog": (0.0, 1.0),
            "msl": (1e5, 1e3), "skt": (280.0, 10.0)}
    mean, std = np.zeros(contract.N_CHANNELS), np.ones(contract.N_CHANNELS)
    for name, i in contract.CHANNEL_INDEX.items():
        key = name.rstrip("0123456789") if name not in phys else name
        mean[i], std[i] = phys[key]
    return mean, std


def _setup(positive=True, solver="heun"):
    contract.register_latitudes(LAT)
    mean, std = _stats()
    core = DynCore(DynCoreConfig(hidden=8, blocks=(1, 1), solver=solver, positive=positive),
                   torch.zeros(8, 32, 64), torch.full((N,), 0.1), mean[contract.DYNAMIC_CHANNELS],
                   std[contract.DYNAMIC_CHANNELS], LAT)
    guide = PhysicsGuide(32, 64, mean, std, PhysicsConfig(dt_seconds=6 * 3600.0))
    return core, StrongPhysics(guide, torch.full((N,), 0.1)), mean, std


def test_rollout_starts_as_persistence_with_the_tisr_forcing_from_the_data():
    core, _, _, _ = _setup()
    x0 = torch.zeros(2, N, 32, 64)  # every moisture value = its mean > 0: PositiveMass is the identity
    tisr = torch.rand(2, 3, 32, 64)
    y = core.rollout(x0, torch.tensor([0.0, 600.0]), 3, tisr)
    assert y.shape == (2, 3, N, 32, 64)
    other = [i for i in range(N) if i != TISR_DYN]
    assert torch.equal(y[:, :, other], x0[:, None, other].expand(-1, 3, -1, -1, -1))
    assert torch.equal(y[:, :, TISR_DYN], tisr)
    assert torch.equal(core.tendency(x0, torch.zeros(2)), torch.zeros(2, N, 32, 64))


def test_rollout_is_deterministic_and_a_trained_head_moves_the_state():
    core, _, _, _ = _setup(positive=False, solver="euler")
    core.eval()  # the trunk has BatchNorm and Dropout (ClimODE's ResidualBlock)
    torch.nn.init.normal_(core.net.head.weight, std=0.1)
    x0 = torch.randn(1, N, 32, 64)
    y1 = core.rollout(x0, torch.zeros(1), 2)
    y2 = core.rollout(x0, torch.zeros(1), 2)
    assert torch.equal(y1, y2)
    assert not torch.allclose(y1[:, 0], x0)
    assert torch.equal(y1[:, :, TISR_DYN], x0[:, None, TISR_DYN].expand(-1, 2, -1, -1))  # never predicted


def test_zero_tendency_residual_is_the_tendency_free_operator():
    core, phys, _, _ = _setup()
    x = 0.1 * torch.randn(1, N, 32, 64)
    orog = torch.zeros(1, 1, 32, 64)
    R, mask = phys.residuals(x, orog, torch.zeros_like(x))
    F = phys.guide.to_si(x, orog)
    A = pgrid.strong_residuals(F, phys.guide.grid(x.device))
    for k in A:
        assert torch.allclose(R[k], A[k])
    assert mask.dtype == torch.bool or mask.dtype == torch.float64 or mask.dtype == torch.float32
    assert set(R) >= {"u", "v", "T", "mass", "hyd"}


def test_tendency_enters_the_prognostic_residuals_only():
    core, phys, _, _ = _setup()
    x = 0.1 * torch.randn(1, N, 32, 64)
    orog = torch.zeros(1, 1, 32, 64)
    f = 0.1 * torch.randn(1, N, 32, 64)
    R0, _ = phys.residuals(x, orog, torch.zeros_like(x))
    R1, _ = phys.residuals(x, orog, f)
    for k in ("u", "v", "T"):
        assert not torch.allclose(R0[k], R1[k])
    for k in ("mass", "hyd"):
        assert torch.allclose(R0[k], R1[k])


def test_strong_scales_are_the_rms_of_era5_like_triples_and_floored():
    core, phys, _, _ = _setup()
    g = torch.Generator().manual_seed(1)
    x = 0.1 * torch.randn(3, N, 32, 64, generator=g)
    orog = torch.zeros(1, 1, 32, 64)
    triples = [(x[i:i + 1] - 0.01, x[i:i + 1], x[i:i + 1] + 0.01, orog) for i in range(3)]
    s = estimate_strong_scales(phys, triples, floor_frac=0.1)
    assert s.mom > 0 and s.T > 0 and s.mass > 0 and s.hyd > 0
    phys.scales = s
    ms = {k: v for k, v in phys.per_equation(x, orog.expand(3, -1, -1, -1), torch.full_like(x, 0.01)).items()}
    assert abs(ms["mass"] - 1.0) < 1e-6  # mass does not depend on the tendency: by construction RMS 1
    loss = phys.loss({k: torch.tensor(v ** 2) for k, v in ms.items()}, {"u": 1.0, "mass": 1.0}, target=1.0)
    assert float(loss) >= 0.0


def test_kinetic_energy_is_area_weighted_mean_of_the_wind_energy():
    _, _, mean, std = _setup()
    x = torch.zeros(2, N, 32, 64)
    w = torch.ones(32, 1)
    m, s = torch.from_numpy(mean[contract.DYNAMIC_CHANNELS]), torch.from_numpy(std[contract.DYNAMIC_CHANNELS])
    ke = kinetic_energy(x, s.float(), m.float(), w)
    from pgcfm.coupling import LEVEL_DYN
    mu, mv = m[LEVEL_DYN["u"]], m[LEVEL_DYN["v"]]
    assert torch.allclose(ke, torch.full((2,), float((0.5 * (mu ** 2 + mv ** 2)).mean())), rtol=1e-4)


def test_balance_head_starts_at_zero_and_corrects_only_z_and_t():
    contract.register_latitudes(LAT)
    mean, std = _stats()
    from pgcfm.dyncore import BAL_DYN
    core = DynCore(DynCoreConfig(hidden=8, blocks=(1, 1), balance=True, positive=False),
                   torch.zeros(8, 32, 64), torch.full((N,), 0.1), mean[contract.DYNAMIC_CHANNELS],
                   std[contract.DYNAMIC_CHANNELS], LAT)
    core.eval()
    x0 = torch.randn(2, N, 32, 64)
    y, xb = core.rollout(x0, torch.zeros(2), 2, balanced=True)
    assert torch.equal(xb[:, 0], x0) and torch.equal(y[:, 0], x0)  # zero heads: persistence, no correction
    torch.nn.init.normal_(core.bal_head.weight, std=0.1)
    _, xb = core.rollout(x0, torch.zeros(2), 1, balanced=True)
    other = [i for i in range(N) if i not in BAL_DYN]
    assert torch.equal(xb[:, 0, other], x0[:, other])
    assert not torch.allclose(xb[:, 0, BAL_DYN], x0[:, BAL_DYN])
