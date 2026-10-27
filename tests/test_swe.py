"""Step 17 shallow-water test bed (pgcfm/swe.py)."""
import math

import torch

from pgcfm.swe import (CHANNELS, Normalizer, SWEPinn, TendencyNet, World, rollout_heun, rollout_node,
                       sample_sphere_time, swe_residuals)

torch.set_num_threads(1)


def _world():
    return World(32, 64, 11, 600.0)


def test_steady_zonal_flow_has_zero_residual():
    """Williamson case 2: u = u0 cos(phi), Phi = Phi0 - (a Omega u0 + u0^2 / 2) sin^2(phi) is steady."""
    a, om, u0, phi0 = 6.37122e6, 7.292e-5, 20.0, 9.8e4
    phi = torch.linspace(-1.4, 1.4, 57, dtype=torch.float64)
    cos, sin = torch.cos(phi), torch.sin(phi)
    c = a * om * u0 + 0.5 * u0 ** 2
    z = torch.zeros_like(phi)
    F = {"u": u0 * cos, "v": z, "Phi": phi0 - c * sin ** 2,
         "dlam_u": z, "dlam_v": z, "dlam_Phi": z, "ds_u": z, "ds_v": z, "ds_Phi": z,
         "dphi_u": -u0 * sin, "dphi_v": z, "dphi_Phi": -2 * c * sin * cos}
    R = swe_residuals(F, phi, a, om)
    scale = {"u": om * u0, "v": om * u0, "Phi": phi0 * u0 / a}
    for k, r in R.items():
        assert float(r.abs().max()) < 1e-9 * scale[k], k


def test_world_tendency_matches_first_solver_step():
    w = _world()
    s0 = w.galewsky()
    x0 = w.grid(s0)
    F = w.tendency(x0[None], closure=True)[0]
    s1 = w.solver.timestep(s0, 1)  # the first step is forward Euler, then hyperdiffusion on vrt / div
    fd = (w.grid(s1) - x0) / w.dt
    for i, k in enumerate(CHANNELS):
        rel = float((F[i] - fd[i]).norm() / fd[i].norm())
        assert rel < 0.1, (k, rel)
    inviscid = w.tendency(x0[None])[0]  # the hyperdiffusion is a large part of the jet's first-step tendency
    assert float((inviscid[1] - fd[1]).norm() / fd[1].norm()) > 0.5


def test_grid_spec_roundtrip_and_truncation():
    w = _world()
    x = w.grid(w.galewsky())
    back = w.grid(w.spec(x))
    assert float((back - x).norm() / x.norm()) < 1e-6
    coarse = World(16, 32, 6, 600.0)
    xc = coarse.grid(w.truncate(w.galewsky(), 6))
    assert xc.shape == (3, 16, 32)


def test_rollouts_start_as_persistence_and_agree_in_shape():
    w = _world()
    frames = torch.stack([w.grid(s).float() for s in [w.galewsky()] + w.step_hours(w.galewsky(), 2)])
    nz = Normalizer(frames)
    net = TendencyNet(w, nz.t1, hidden=8, blocks=(1, 1)).eval()
    z0 = nz.norm(frames[:1])
    yh = rollout_heun(net, z0, 3, substeps=2)
    yn = rollout_node(net, z0, 3, method="rk4", step=0.5)
    assert yh.shape == yn.shape == (1, 3, 3, 32, 64)
    assert torch.allclose(yh, z0[:, None].expand_as(yh))  # zero-initialized head
    assert torch.allclose(yn, z0[:, None].expand_as(yn), atol=1e-6)


def test_swe_pinn_residual_shapes_and_grid_prediction():
    w = _world()
    T = 6 * 3600.0
    P = SWEPinn([4, 16, 16, 4], T, {"u": (0.0, 20.0), "v": (0.0, 20.0), "Phi": (9.8e4, 1e3)},
                {"u": 1e-3, "v": 1e-3, "Phi": 1.0}, 6.37122e6, 7.292e-5)
    lam, phi, s = sample_sphere_time(64, T, generator=torch.Generator().manual_seed(0))
    R = P.net_f(lam, phi, s)
    assert set(R) == {"u", "v", "Phi"} and all(r.shape == (64,) for r in R.values())
    L = P.loss((lam, phi, s), {"u": torch.zeros(64), "Phi": torch.full((64,), 9.8e4)}, (lam, phi, s))
    assert math.isfinite(float(L["loss"]))
    g = P.predict_grid(w, [0.0, 1.0])
    assert g.shape == (2, 3, 32, 64)
