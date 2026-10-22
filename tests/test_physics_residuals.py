import math

import numpy as np
import pytest
import torch

from pgcfm import contract
from pgcfm.physics import analytic, grid as G, pinn, primitive

SC = primitive.ResidualScales()
DAY = contract.CONST.dt_seconds


def rel(R):
    """max |R_j| / b_j per equation."""
    return {k: float(v.detach().abs().max() / SC.of(k)) for k, v in R.items()}


@pytest.fixture(scope="module")
def grid():
    return G.SphereGrid()


@pytest.fixture(scope="module")
def colloc():
    return pinn.sample_collocation(1000, generator=torch.Generator().manual_seed(0))


# --- autograd (PINN) path: exact derivatives -----------------------------------------

@pytest.mark.parametrize("alpha", [0.0, math.pi / 4, math.pi / 2])
def test_autograd_residuals_vanish_on_solid_body(alpha, colloc):
    R = pinn.autograd_residuals(lambda *c: analytic.solid_body(*c, alpha=alpha), *colloc, create_graph=False)
    assert set(R) == {"u", "v", "T", "mass", "hyd", "q"}
    r = rel(R)
    assert max(v for k, v in r.items() if k != "hyd") < 1e-10
    assert r["hyd"] < 1e-9  # K; Phi ~ 1e5 m2 s-2 limits float64 cancellation


def test_negative_control_breaks_balance(colloc):
    R = pinn.autograd_residuals(lambda *c: analytic.solid_body(*c, phi_scale=0.9), *colloc, create_graph=False)
    r = rel(R)
    assert r["u"] > 0.1 and r["v"] > 0.1 and r["hyd"] > 1.0


def test_rest_state_is_a_zero_residual_solution(colloc):
    # Documents the trivial solution the physics loss cannot distinguish (docs, weakness W6).
    R = pinn.autograd_residuals(lambda *c: analytic.solid_body(*c, U0=0.0, U1=0.0), *colloc, create_graph=False)
    assert max(rel(R).values()) < 1e-9


def test_tangent_projection_recovers_solid_body_wind():
    lam, phi, p, s = pinn.sample_collocation(500, pole_cap_deg=90.0, generator=torch.Generator().manual_seed(1))
    alpha, U = math.pi / 3, 10.0
    n = torch.tensor([-math.sin(alpha), 0.0, math.cos(alpha)], dtype=torch.float64)
    r = torch.stack([torch.cos(phi) * torch.cos(lam), torch.cos(phi) * torch.sin(lam), torch.sin(phi)], -1)
    V = U * torch.cross(n.expand_as(r), r, dim=-1)  # rigid rotation, smooth through the poles
    u, v = pinn.tangent_to_uv(V[:, 0], V[:, 1], V[:, 2], lam, phi)
    F = analytic.solid_body(lam, phi, torch.full_like(lam, 1e5), s, alpha=alpha, U0=U, U1=0.0)
    assert torch.allclose(u, F["u"], atol=1e-12) and torch.allclose(v, F["v"], atol=1e-12)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 5e-3), (torch.float64, 1e-10)])
def test_pole_conditioning_near_89_9(dtype, tol):
    lam = torch.linspace(0, 2 * math.pi, 33, dtype=dtype)[:-1]
    phi = torch.full_like(lam, math.radians(89.9))
    R = pinn.autograd_residuals(lambda *c: analytic.solid_body(*c, alpha=math.pi / 2), lam, phi,
                                torch.full_like(lam, 5e4), torch.zeros_like(lam), create_graph=False)
    assert max(rel(R)[k] for k in ("u", "v", "mass")) < tol


# --- grid (finite-difference) path ---------------------------------------------------

def test_grid_residuals_skip_pole_rows_and_are_finite(grid):
    F, T = analytic.on_grid(grid, alpha=math.pi / 2, with_tendencies=True)
    R = G.strong_residuals(F, grid, tendencies=T)
    for v in R.values():
        assert v.shape == (12, 119, 240)
        assert torch.isfinite(v).all()
    lat = grid.lat[grid.lat_rows] * 180 / math.pi
    assert float(lat.min()) == pytest.approx(-88.5) and float(lat.max()) == pytest.approx(88.5)


@pytest.mark.parametrize("alpha", [0.0, math.pi / 2])
def test_grid_discretization_error_on_solid_body(alpha, grid):
    F, T = analytic.on_grid(grid, alpha=alpha, with_tendencies=True)
    r = rel(G.strong_residuals(F, grid, tendencies=T))
    assert max(r[k] for k in ("u", "v", "T", "mass", "q")) < 2e-3
    assert r["hyd"] < 1.2  # K, dominated by the one-sided 50 hPa level (docs W3)


def test_vertical_derivative_in_p_is_much_worse_than_log_p():
    g = G.SphereGrid(vertical="p")
    F, T = analytic.on_grid(g, alpha=0.0, with_tendencies=True)
    assert rel(G.strong_residuals(F, g, tendencies=T))["hyd"] > 5.0


def test_polar_filter_removes_noise_amplification():
    def ratio(g):
        F, T = analytic.on_grid(g, alpha=0.0, with_tendencies=True)
        gen = torch.Generator().manual_seed(0)
        # geopotential noise enters R_u through d_lam Phi / (a cos phi)
        F = dict(F, Phi=F["Phi"] + 30.0 * torch.randn(F["Phi"].shape, generator=gen, dtype=torch.float64))
        rms = G.residual_rms_by_lat(G.strong_residuals(F, g, tendencies=T)["u"], g)
        return float(rms[-1] / rms[len(rms) // 2])  # 88.5 N vs equator

    assert ratio(G.SphereGrid(lon_method="centered", polar_filter_deg=None)) > 10.0
    assert ratio(G.SphereGrid()) < 2.0


def test_diagnosed_omega_of_nondivergent_flow_is_small(grid):
    F = {k: v for k, v in analytic.on_grid(grid, alpha=math.pi / 2).items() if k != "omega"}
    om = G.diagnose_omega(F, grid).index_select(-2, grid.lat_rows)
    assert float(om[0].abs().max()) == 0.0  # omega_top
    assert float(om.abs().max()) < 5e-3  # Pa s-1, discretization only


def test_time_integrated_residual_steady_flow(grid):
    F0 = analytic.on_grid(grid, s=0.0, alpha=math.pi / 2)
    F1 = analytic.on_grid(grid, s=DAY, alpha=math.pi / 2)
    r = rel(G.time_integrated_residuals(F0, F1, grid, n_sub=4))
    assert max(r[k] for k in ("u", "v", "T")) < 2e-3


def test_linear_bridge_fails_for_synoptic_transport(grid):
    # Known limitation (docs W1): h_psi = 0 cannot represent a day of advection at m = 8.
    kw = dict(alpha=0.0, q_wavenumber=8)
    F0 = analytic.on_grid(grid, s=0.0, **kw)
    F1, T1 = analytic.on_grid(grid, s=DAY, with_tendencies=True, **kw)
    R = G.time_integrated_residuals(F0, F1, grid, n_sub=8)
    assert float(R["q"].abs().max() / T1["q"].abs().max()) > 0.3


def test_physics_gradient_reaches_endpoint(grid):
    F0 = {k: v for k, v in analytic.on_grid(grid, s=0.0, alpha=0.0).items() if k != "omega"}
    F1 = {k: v.clone().requires_grad_(k in ("u", "v", "T", "Phi"))
          for k, v in analytic.on_grid(grid, s=DAY, alpha=0.0).items() if k != "omega"}
    R = G.time_integrated_residuals(F0, F1, grid, n_sub=2)
    primitive.physics_loss(R, SC, area_weight=grid.latitude_weight()).backward()
    for k in ("u", "v", "T", "Phi"):
        assert torch.isfinite(F1[k].grad).all() and F1[k].grad.abs().max() > 0


def test_physics_loss_mask_does_not_leak_nan():
    R = {"u": torch.tensor([1e-3, float("nan")])}
    L = primitive.physics_loss(R, SC, mask=torch.tensor([1.0, 0.0]))
    assert float(L) == pytest.approx(1.0)


def test_above_ground_mask_matches_contract(grid):
    sp = torch.full((121, 240), 9.0e4, dtype=torch.float64)
    m = G.above_ground_mask(sp, grid)
    ref = ~contract.below_ground_mask(sp.numpy())
    assert m.shape == (12, 119, 240)
    assert np.array_equal(m.numpy().astype(bool), ref[:, 1:-1, :])


def test_fields_from_state_channel_mapping():
    X = torch.arange(67, dtype=torch.float64).view(67, 1, 1).expand(67, 121, 240)
    F = G.fields_from_state(X)
    assert float(F["Phi"][6, 0, 0]) == 6  # z500
    assert float(F["T"][9, 0, 0]) == 33  # t850
    assert float(F["u"][0, 0, 0]) == 36 and float(F["v"][-1, 0, 0]) == 59
    assert float(F["sp"][0, 0]) == 61


# --- Raissi PINN port ----------------------------------------------------------------

def test_pinn_port_trains_and_residuals_are_finite():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    lb, ub = [math.log(5e3), 0.0], [math.log(1e5), DAY]
    net = pinn.PhysicsInformedNN([5, 16, 16, 7], lb, ub, dtype=torch.float64)
    fn = lambda *c: analytic.solid_body(*c, alpha=math.pi / 2)
    Xu = pinn.sample_collocation(64, generator=gen)
    tgt = {k: v.detach() for k, v in fn(*Xu).items() if k in ("u", "v", "T", "Phi")}
    Xf = pinn.sample_collocation(64, generator=gen)
    w = {k: 1e-2 for k in primitive.EQUATIONS}
    hist = net.train_model(Xu, tgt, Xf, n_adam=30, lr=1e-2, weights=w)
    assert hist[-1]["loss"] < hist[0]["loss"]
    F, R = net.predict(*Xf)
    assert set(F) == {"u", "v", "T", "Phi", "omega", "q"}
    assert all(torch.isfinite(v).all() for v in R.values())


def test_era5_residual_stats_script_runs_on_kai_layout(tmp_path):
    from scripts.make_synthetic_era5 import main as make_synthetic
    from scripts.step3_era5_residual_stats import main as stats

    make_synthetic(["--out", str(tmp_path), "--years", "2011", "--days", "3"])
    res = stats(["--data-dir", str(tmp_path), "--max-pairs", "2", "--stride", "1",
                 "--out", str(tmp_path / "stats.json")])
    assert res["n_pairs"] == 2
    assert set(res["rms_global(=suggested b_j)"]) == {"u", "v", "T", "mass", "hyd", "q"}
    assert all(np.isfinite(v) for v in res["rms_global(=suggested b_j)"].values())
    frac = res["above_ground_area_fraction_by_level"]
    assert frac[0] == 1.0 and frac[-1] < 1.0  # synthetic orography puts 1000 hPa underground
