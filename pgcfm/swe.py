"""Step 17: a spherical shallow-water test bed for "learning the physics directly" (owner 2026-10-27 12:35).

Truth comes from torch-harmonics' spectral ``ShallowWaterSolver`` (examples/shallow_water_equations.py, used
unchanged: Galewsky jet and random initial conditions, AB3 time stepping, implicit hyperdiffusion on vorticity
and divergence). The learners work on grid fields (Phi = g h, u, v):

  * ``TendencyNet`` f_theta: ClimODE's ResNet trunk (pgcfm.core.Net) with a zero-initialized head, integrated
    by Heun substeps (as pgcfm/dyncore.py) or by ``torchdiffeq.odeint`` (the neural ODE of rtqichen/torchdiffeq,
    as ClimODE uses it).
  * ``World.tendency``: the solver's own spectral right-hand side at the learners' resolution, so a strong-form
    residual R = f_theta(x) - F(x) is exact when the truth lives at that resolution and truncated when the truth
    is finer (the closure problem of real data, docs/step16 sec. 4).
  * ``SWEPinn``: the Raissi coordinate PINN (pgcfm/physics/pinn.py port of maziarraissi/PINNs) with the
    shallow-water residual by automatic differentiation in (lambda, phi, t): the forward problem of the
    upstream Burgers example (data at t = 0, physics inside the space-time domain).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch_harmonics.examples.shallow_water_equations import ShallowWaterSolver

from .core import Net
from .physics.pinn import PhysicsInformedNN, tangent_to_uv

CHANNELS = ("Phi", "u", "v")
HOUR = 3600.0


@dataclass
class _NetCfg:  # the fields of core.CoreConfig that core.Net reads
    hidden: int
    blocks: tuple


class World(nn.Module):
    """The solver at one resolution plus grid <-> spectral conversions of (Phi, u, v)."""

    def __init__(self, nlat: int, nlon: int, lmax: int, dt: float):
        super().__init__()
        self.solver = ShallowWaterSolver(nlat, nlon, dt, lmax=lmax, mmax=lmax).double()
        self.nlat, self.nlon, self.lmax, self.dt = nlat, nlon, lmax, dt
        lats = self.solver.lats.clone()
        self.register_buffer("lat", lats)  # (nlat,), radians, north first
        self.register_buffer("lon", self.solver.lons.clone())
        self.register_buffer("w", (self.solver.quad_weights.view(-1) / self.solver.quad_weights.sum()).clone())
        self.steps_per_hour = int(round(HOUR / dt))
        if abs(self.steps_per_hour * dt - HOUR) > 1e-6:
            raise ValueError("dt must divide one hour")

    @property
    def g(self):
        return float(self.solver.gravity)

    def grid(self, uspec: torch.Tensor) -> torch.Tensor:
        """(..., 3, lmax, mmax) spectral (Phi, vrt, div) -> (..., 3, nlat, nlon) grid (Phi, u, v)."""
        if uspec.dim() == 4:  # the solver's gethuv is written for one state
            return torch.stack([self.solver.gethuv(u) for u in uspec])
        return self.solver.gethuv(uspec)

    def spec(self, x: torch.Tensor) -> torch.Tensor:
        """Grid (..., 3, nlat, nlon) float64 -> spectral (Phi, vrt, div)."""
        phi = self.solver.grid2spec(x[..., :1, :, :])
        vd = self.solver.vrtdivspec(x[..., 1:, :, :])
        return torch.cat([phi, vd], -3)

    def truncate(self, uspec: torch.Tensor, lmax: int) -> torch.Tensor:
        return uspec[..., :lmax, :lmax]

    def tendency_spec(self, us: torch.Tensor, closure: bool = False) -> torch.Tensor:
        """Spectral right-hand side of one state (3, lmax, mmax): the inviscid equations (the physics the
        learners are given), plus the solver's implicit hyperdiffusion of vorticity and divergence as an
        equivalent tendency (hyperdiff - 1) / dt x state when ``closure`` (what the data actually do)."""
        d = self.solver.dudtspec(us)
        if closure:
            d = d.clone()
            d[1:] = d[1:] + (self.solver.hyperdiff - 1.0) / self.dt * us[1:]
        return d

    def tendency(self, x: torch.Tensor, closure: bool = False) -> torch.Tensor:
        """Right-hand side F(x) (per second) for grid fields (B, 3, nlat, nlon); see ``tendency_spec``."""
        us = self.spec(x.double())
        dudt = torch.stack([self.tendency_spec(u, closure) for u in us]) if us.dim() == 4 \
            else self.tendency_spec(us, closure)
        return self.grid(dudt)

    def step_hours(self, uspec: torch.Tensor, hours: int) -> List[torch.Tensor]:
        """Spectral states after 1, 2, ..., hours hours."""
        out = []
        for _ in range(hours):
            uspec = self.solver.timestep(uspec, self.steps_per_hour)
            out.append(uspec)
        return out

    def galewsky(self, umax: float = 80.0, lon0: float = math.pi, bump: float = 1.0) -> torch.Tensor:
        """The solver's Galewsky initial condition with a movable, scalable bump (upstream: umax 80, lon0 pi,
        bump 1); the balance solve is the solver's."""
        s = self.solver
        phi0 = math.pi / 7.0
        phi1 = 0.5 * math.pi - phi0
        phi2 = 0.25 * math.pi
        en = math.exp(-4.0 / (phi1 - phi0) ** 2)
        alpha, beta = 1.0 / 3.0, 1.0 / 15.0
        lats, lons = torch.meshgrid(s.lats, s.lons, indexing="ij")
        mask = torch.logical_and(lats > phi0, lats < phi1)
        product = torch.where(mask, (lats - phi0) * (lats - phi1), -torch.ones_like(lats))
        u1 = (umax / en) * torch.exp(1.0 / product)
        ugrid = torch.where(mask, u1, torch.zeros_like(lats))
        vgrid = torch.zeros_like(lats)
        dlon = torch.remainder(lons - lon0 + math.pi, 2 * math.pi) - math.pi
        hbump = bump * s.hamp * torch.cos(lats) * torch.exp(-((dlon / alpha) ** 2)) * torch.exp(-((phi2 - lats) ** 2) / beta)
        ugrid = torch.stack((ugrid, vgrid))
        vrtdivspec = s.vrtdivspec(ugrid)
        vrtdivgrid = s.spec2grid(vrtdivspec)
        tmp = ugrid * (vrtdivgrid + s.coriolis)
        tmpspec = s.vrtdivspec(tmp)
        tmpspec[1] = s.grid2spec(0.5 * torch.sum(ugrid ** 2, dim=0))
        phispec = s.invlap * tmpspec[0] - tmpspec[1] + s.grid2spec(s.gravity * (s.havg + hbump))
        uspec = torch.zeros(3, s.lmax, s.mmax, dtype=vrtdivspec.dtype)
        uspec[0] = phispec
        uspec[1:] = vrtdivspec
        return torch.tril(uspec)

    def random_ic(self, mach: float = 0.1) -> torch.Tensor:
        return self.solver.random_initial_condition(mach=mach)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Area-weighted total energy (Phi (u^2 + v^2) / 2 + Phi^2 / 2) / g of grid fields (..., 3, H, W)."""
        e = (x[..., 0, :, :] * (x[..., 1, :, :] ** 2 + x[..., 2, :, :] ** 2) / 2 + x[..., 0, :, :] ** 2 / 2) / self.g
        return (e * self.w.view(-1, 1).to(e.dtype)).sum(-2).mean(-1)

    def wrmse(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Area-weighted RMSE per channel (..., 3, H, W) -> (..., 3)."""
        return (((a - b) ** 2) * self.w.view(-1, 1).to(a.dtype)).sum(-2).mean(-1).sqrt()


# -- grid learners -------------------------------------------------------------------------------


class TendencyNet(nn.Module):
    """f_theta(x): normalized tendency per hour of (Phi, u, v) on the grid; static input sin / cos lat."""

    def __init__(self, world: World, t1: torch.Tensor, hidden: int = 32, blocks: Tuple[int, ...] = (2, 2)):
        super().__init__()
        lat = world.lat.float()
        static = torch.stack([torch.sin(lat), torch.cos(lat)]).view(1, 2, -1, 1).expand(1, 2, world.nlat, world.nlon)
        self.register_buffer("static", static.contiguous())
        self.register_buffer("t1", t1.view(1, 3, 1, 1).clone())
        self.net = Net(3 + 2, 3, _NetCfg(hidden, tuple(blocks)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.static.expand(x.shape[0], -1, -1, -1)], 1)) * self.t1


def rollout_heun(net: nn.Module, x0: torch.Tensor, hours: int, substeps: int = 3) -> torch.Tensor:
    """(B, hours, 3, H, W): Heun substeps of f_theta, as pgcfm/dyncore.py."""
    h, x, out = 1.0 / substeps, x0, []
    for _ in range(hours):
        for _ in range(substeps):
            k1 = net(x)
            k2 = net(x + h * k1)
            x = x + 0.5 * h * (k1 + k2)
        out.append(x)
    return torch.stack(out, 1)


def rollout_node(net: nn.Module, x0: torch.Tensor, hours: int, method: str = "rk4", step: float = 1.0 / 3.0,
                 adjoint: bool = False) -> torch.Tensor:
    """(B, hours, 3, H, W): torchdiffeq.odeint of f_theta over the hours (the neural-ODE route)."""
    from torchdiffeq import odeint, odeint_adjoint
    t = torch.arange(0, hours + 1, dtype=x0.dtype, device=x0.device)
    f = lambda s, x: net(x)  # noqa: E731
    opts = {"step_size": step} if method in ("rk4", "euler", "midpoint") else None
    y = (odeint_adjoint if adjoint else odeint)(f, x0, t, method=method, options=opts)
    return y[1:].transpose(0, 1)


class Normalizer:
    """z-scores per channel; tendency std per hour; physical <-> normalized."""

    def __init__(self, frames: torch.Tensor):
        """frames (N, 3, H, W) physical float32 (training states; consecutive frames 1 h apart)."""
        self.mean = frames.mean((0, 2, 3)).view(1, 3, 1, 1)
        self.std = frames.std((0, 2, 3)).view(1, 3, 1, 1)
        d = (frames[1:] - frames[:-1]) / self.std
        self.t1 = d.pow(2).mean((0, 2, 3)).sqrt()

    def norm(self, x):
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

    def phys(self, z):
        return z * self.std.to(z.dtype) + self.mean.to(z.dtype)


def strong_residual(world: World, nz: Normalizer, net: nn.Module, z: torch.Tensor, b: torch.Tensor,
                    F_true: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-channel mean of ((f_theta(x) - F(x)) / b)^2 for normalized states z (N, 3, H, W); F is the solver's
    right-hand side at this world's resolution unless the oracle tendency ``F_true`` (per second) is given."""
    f = net(z) * nz.std.to(z.dtype) / HOUR  # per second, physical
    F = world.tendency(nz.phys(z).double()).to(f.dtype) if F_true is None else F_true.to(f.dtype)
    return ((f - F) / b.view(1, 3, 1, 1).to(f.dtype)).pow(2).mean((0, 2, 3))


# -- the Raissi coordinate PINN for the shallow-water equations ------------------------------------


def swe_residuals(F: Dict[str, torch.Tensor], phi: torch.Tensor, radius: float, omega: float) -> Dict[str, torch.Tensor]:
    """Advective-form shallow-water residuals on the sphere from a field dict with the derivatives
    d{lam,phi,s}_{u,v,Phi} (SI; s in seconds)."""
    a, cos, tan = radius, torch.cos(phi), torch.tan(phi)
    f = 2.0 * omega * torch.sin(phi)
    u, v, P = F["u"], F["v"], F["Phi"]
    adv = lambda n: (u / (a * cos)) * F[f"dlam_{n}"] + (v / a) * F[f"dphi_{n}"]  # noqa: E731
    R = {"u": F["ds_u"] + adv("u") - u * v * tan / a - f * v + F["dlam_Phi"] / (a * cos),
         "v": F["ds_v"] + adv("v") + u * u * tan / a + f * u + F["dphi_Phi"] / a,
         "Phi": F["ds_Phi"] + adv("Phi") + P * (F["dlam_u"] + F["dphi_v"] * cos - v * torch.sin(phi)) / (a * cos)}
    return R


class SWEPinn(PhysicsInformedNN):
    """maziarraissi/PINNs continuous-time inference for the shallow-water equations on the sphere: input
    (x, y, z, s), output a Cartesian tangent wind (-> u, v) and Phi, residuals by autograd (net_f)."""

    OUTPUTS = ("Vx", "Vy", "Vz", "Phi")

    def __init__(self, layers: Sequence[int], T: float, field_scales: Dict[str, tuple], b: Dict[str, float],
                 radius: float, omega: float, dtype=torch.float32):
        nn.Module.__init__(self)
        assert layers[0] == 4 and layers[-1] == 4
        self.register_buffer("lb", torch.tensor([0.0], dtype=dtype))
        self.register_buffer("ub", torch.tensor([T], dtype=dtype))
        self.field_scales, self.b, self.radius, self.omega = field_scales, b, radius, omega
        self.layers = self.initialize_NN(layers).to(dtype)
        self.dtype = dtype

    def net_u(self, lam, phi, s):
        cl, sl, cp, sp = torch.cos(lam), torch.sin(lam), torch.cos(phi), torch.sin(phi)
        X = torch.stack([cp * cl, cp * sl, sp, s], 1).to(self.dtype)
        Y = self.neural_net(X)
        u, v = tangent_to_uv(Y[:, 0], Y[:, 1], Y[:, 2], lam, phi)
        F = {"u": u, "v": v, "Phi": Y[:, 3]}
        return {k: self.field_scales[k][0] + self.field_scales[k][1] * x for k, x in F.items()}

    def net_f(self, lam, phi, s, create_graph: bool = True):
        coords = [c.detach().clone().requires_grad_(True) for c in (lam, phi, s)]
        F = dict(self.net_u(*coords))
        for name in ("u", "v", "Phi"):
            g = torch.autograd.grad(F[name].sum(), coords, create_graph=create_graph, retain_graph=True)
            for c, gc in zip(("lam", "phi", "s"), g):
                F[f"d{c}_{name}"] = gc
        return swe_residuals(F, coords[1], self.radius, self.omega)

    def loss(self, X_u, targets, X_f, weights=None):
        pred = self.net_u(*X_u)
        data = sum(torch.mean(((pred[k] - targets[k].to(pred[k].dtype)) / self.field_scales[k][1]) ** 2)
                   for k in targets)
        R = self.net_f(*X_f)
        phys = sum((weights or {}).get(k, 1.0) * torch.mean((r / self.b[k]) ** 2) for k, r in R.items())
        return {"loss": data + phys, "data": data, "phys": phys}

    def predict_grid(self, world: World, hours: Sequence[float]) -> torch.Tensor:
        """(len(hours), 3, nlat, nlon) physical fields on the world's grid at the given hours."""
        lat, lon = torch.meshgrid(world.lat.float(), world.lon.float(), indexing="ij")
        out = []
        with torch.no_grad():
            for h in hours:
                s = torch.full_like(lat.reshape(-1), h * HOUR)
                F = self.net_u(lon.reshape(-1), lat.reshape(-1), s)
                out.append(torch.stack([F[k].view(world.nlat, world.nlon) for k in CHANNELS]))
        return torch.stack(out)


def sample_sphere_time(n: int, T: float, pole_cap_deg: float = 85.0, generator=None):
    """Area-uniform points with |lat| <= cap and s uniform in [0, T] (seconds)."""
    zmax = math.sin(math.radians(pole_cap_deg))
    z = (2 * torch.rand(n, generator=generator) - 1) * zmax
    lam = 2 * math.pi * torch.rand(n, generator=generator)
    s = T * torch.rand(n, generator=generator)
    return lam, torch.asin(z), s
