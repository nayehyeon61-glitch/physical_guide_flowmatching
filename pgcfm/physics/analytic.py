"""Analytic primitive-equation solutions for residual tests.

``solid_body`` is a 3-D pressure-level extension of Williamson et al.
(1992) shallow-water test case 2: steady solid-body rotation about an axis
tilted by ``alpha`` from the pole, in gradient-wind balance, with a
vertically sheared speed U(p) (thermal wind) and a hydrostatic temperature.
With alpha = pi/2 the flow crosses both poles, which stresses the
1/cos(phi) terms. As in Williamson TC2, the flow is steady only if the
planetary rotation axis is tilted with it, so ``f`` (returned as a field)
is 2 Omega (r.n); for alpha = 0 it is the ordinary 2 Omega sin(phi).
A passive moisture tracer q is carried by the same
rotation, so d_s q != 0 while D_s q = 0.

All four constraint groups (plus moisture) vanish exactly:
  momentum: steady gradient-wind balance on each level (omega = 0),
  thermodynamic: T_v depends only on S = r.n and p, and u.grad S = 0,
  mass: solid-body rotation is non-divergent, omega = 0,
  hydrostatic: T_v = -(p/R_d) d_p Phi by construction.
"""

import math
from typing import Dict

import torch

from ..contract import CONST, Constants
from .primitive import VIRTUAL_EPS

P0 = 1.0e5


def _rotate(v, axis, angle):
    """Rodrigues rotation of vectors v [..., 3] about unit ``axis`` by ``angle`` [...]."""
    c, s = torch.cos(angle)[..., None], torch.sin(angle)[..., None]
    k = axis.expand_as(v)
    return v * c + torch.cross(k, v, dim=-1) * s + k * (k * v).sum(-1, keepdim=True) * (1 - c)


def solid_body(lam, phi, p, s, alpha: float = math.pi / 4, U0: float = 10.0, U1: float = 20.0,
               Ts: float = 288.0, lapse_exp: float = 0.19, q_amp: float = 5e-3,
               phi_scale: float = 1.0, q_wavenumber: int = 1,
               const: Constants = CONST) -> Dict[str, torch.Tensor]:
    """Fields u, v, omega, T, Phi, q and Coriolis f at broadcastable coordinates (rad, rad, Pa, s).

    ``phi_scale`` multiplies the balancing geopotential term; any value other
    than 1 breaks gradient-wind balance (negative control). U0 = U1 = 0 gives
    the resting atmosphere. ``q_wavenumber`` m sets the tracer pattern
    Re((x0 + i y0)^m) = cos^m(phi0) cos(m lam0) in the co-rotating frame
    (smooth at the poles); m = 1 is a planetary wave, m ~ 8 synoptic.
    """
    lam, phi, p, s = torch.broadcast_tensors(lam, phi, p, s)
    a, Om, rd = const.earth_radius, const.omega, const.r_dry
    sa, ca = math.sin(alpha), math.cos(alpha)
    U = U0 + U1 * (1.0 - p / P0)
    dU = -U1 / P0
    # S = r_hat . n with rotation axis n = (-sin a, 0, cos a)
    S = torch.sin(phi) * ca - torch.cos(lam) * torch.cos(phi) * sa
    u = U * (torch.cos(phi) * ca + torch.sin(phi) * torch.cos(lam) * sa)
    v = -U * torch.sin(lam) * sa
    k = lapse_exp
    phi0 = -rd * Ts / k * ((p / P0) ** k - 1.0)
    Phi = phi0 - phi_scale * (a * Om * U + 0.5 * U ** 2) * S ** 2
    Tv = Ts * (p / P0) ** k + (p / rd) * (a * Om + U) * dU * S ** 2

    # passive tracer: q(r, s) = g(R(-U s / a) r_hat), rigid rotation about n
    r_hat = torch.stack([torch.cos(phi) * torch.cos(lam), torch.cos(phi) * torch.sin(lam),
                         torch.sin(phi)], dim=-1)
    n = torch.tensor([-sa, 0.0, ca], dtype=r_hat.dtype, device=r_hat.device)
    r0 = _rotate(r_hat, n, -U * s / a)
    if q_wavenumber == 1:
        c0 = torch.tensor([0.6, 0.0, 0.8], dtype=r_hat.dtype, device=r_hat.device)
        pattern = (r0 * c0).sum(-1)
    else:
        pattern = torch.real(torch.complex(r0[..., 0], r0[..., 1]) ** q_wavenumber)
    q = q_amp * (1.2 + pattern)
    T = Tv / (1.0 + VIRTUAL_EPS * q)
    f = 2.0 * Om * S
    return {"u": u, "v": v, "omega": torch.zeros_like(u), "T": T, "Phi": Phi, "q": q, "f": f}


def on_grid(grid, s: float = 0.0, with_tendencies: bool = False, **kw):
    """``solid_body`` on a ``SphereGrid`` -> fields [L, H, W] (and d_s of u, v, T, q)."""
    shape = (len(grid.p), len(grid.lat), len(grid.lon))
    lam = grid.lon.view(1, 1, -1).expand(shape)
    phi = grid.lat.view(1, -1, 1).expand(shape)
    p = grid.p.view(-1, 1, 1).expand(shape)
    st = torch.full(shape, float(s), dtype=grid.lat.dtype, requires_grad=with_tendencies)
    F = solid_body(lam, phi, p, st, **kw)
    if not with_tendencies:
        return F
    tend = {}
    for n in ("u", "v", "T", "q"):  # u, v do not depend on s (steady flow)
        tend[n] = (torch.autograd.grad(F[n].sum(), st, retain_graph=True)[0] if F[n].requires_grad
                   else torch.zeros_like(F[n]))
    return {k: v.detach() for k, v in F.items()}, tend
