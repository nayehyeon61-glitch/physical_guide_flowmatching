"""Step 14, the owner's H-bal (2026-10-26): a balance projection Pi_theta as the replacement of the stage-1 PINN.

The stage-1 PINN (pgcfm/seqdct.py PINNProjector) may change only the low 8 x 16 DCT modes of the sequence and fixed
only the hydrostatic residual (docs/step13_sequence.md sec. 3). Pi_theta acts on every frame of the sequence with a
full-mode FNO (pgcfm/fno_core.py, the latitude axis doubled over the poles, all 64 x 33 Fourier modes of the doubled
grid), and its operator norm is projected after every step:

    Pi(x) = pos(x + N(x, static)),   Lip(N) <= sqrt(2) (1.13 B)^n,   so Lip(Pi) <= 1 + Lip(N)

(the residual form keeps the identity exact; a nonexpansive map without the skip cannot learn it). ``pos`` is the
mass-conserving projection onto q >= 0 and tcwv >= 0: per frame and level, max(q, 0) rescaled so that the
area-weighted global mass of that level is unchanged (step 13 clamped instead, which adds mass).

Geostrophic balance, the owner's R_geo: the ageostrophic wind |u - u_g| with u_g = -(1/(f a)) dPhi/dphi,
v_g = (1/(f a cos phi)) dPhi/dlam on 200-850 hPa, |lat| >= 20 deg, above ground (pgcfm/physics/grid.py operators).
"""

import math

import numpy as np
import torch
import torch.nn as nn

from . import contract
from .fno_core import FNONet

GEO_LEVELS = [p for p in contract.LEVELS_HPA if 200 <= p <= 850]


def ageostrophic(guide, dyn: torch.Tensor, orog: torch.Tensor, lat_min: float = 20.0):
    """(B, 66, H, W) normalized states -> (B,) RMS ageostrophic wind [m/s] and (B,) RMS geostrophic wind."""
    g, c = guide.grid(dyn.device), contract.CONST
    F = guide.to_si(dyn, orog)
    lev = [contract.LEVELS_HPA.index(p) for p in GEO_LEVELS]
    Phi, u, v = (F[k][:, lev] for k in ("Phi", "u", "v"))
    lat = g.lat.view(1, 1, -1, 1)
    f = 2 * c.omega * torch.sin(lat)
    f = torch.where(f.abs() > 1e-12, f, torch.full_like(f, 1e-12))
    ug = -g.d_phi(Phi) / (f * c.earth_radius)
    vg = g.d_lam(Phi) / (f * c.earth_radius * torch.cos(lat).clamp_min(1e-6))
    keep = (lat.abs() >= math.radians(lat_min)).to(Phi.dtype)
    above = (g.p3()[lev] <= F["sp"].unsqueeze(1)).to(Phi.dtype)
    wt = keep * above * torch.cos(lat)
    den = wt.sum((1, 2, 3)).clamp_min(1e-12)
    ageo = (((u - ug) ** 2 + (v - vg) ** 2) * wt).sum((1, 2, 3)) / den
    geo = ((ug ** 2 + vg ** 2) * wt).sum((1, 2, 3)) / den
    return ageo.sqrt(), geo.sqrt()


def moisture_index():
    """Positions of q (all levels) and tcwv among the 66 dynamic channels."""
    names = [f"q{p}" for p in contract.LEVELS_HPA] + ["tcwv"]
    return [contract.DYNAMIC_CHANNELS.index(contract.CHANNEL_INDEX[c]) for c in names]


class PositiveMass(nn.Module):
    """q >= 0, tcwv >= 0 with the area-weighted global mass of every level and frame kept (normalized KAI units)."""

    def __init__(self, mean, std, lat_deg):
        super().__init__()
        idx = moisture_index()
        full = [contract.DYNAMIC_CHANNELS[i] for i in idx]
        self.idx = idx
        self.register_buffer("m", torch.as_tensor(mean[full], dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("s", torch.as_tensor(std[full] + contract.KAI_STD_EPS, dtype=torch.float32).view(1, -1, 1, 1))
        w = torch.cos(torch.deg2rad(torch.tensor(np.asarray(lat_deg, dtype=np.float32))))
        self.register_buffer("w", (w / w.mean()).view(1, 1, -1, 1))

    def mass(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 66, H, W) -> (N, 13) area-weighted mean of q per level and of tcwv, physical units."""
        return ((x[:, self.idx] * self.s + self.m) * self.w).mean((-2, -1))

    def negative_fraction(self, x: torch.Tensor) -> float:
        return float((x[:, self.idx] * self.s + self.m < -1e-5 * self.s).float().mean())  # below rounding

    def forward(self, x: torch.Tensor, ref: torch.Tensor = None) -> torch.Tensor:
        """Clamp q at 0 and rescale every level to the global mass of ``ref`` (default: of x itself)."""
        q = x[:, self.idx] * self.s + self.m
        qp = q.clamp_min(0.0)
        q_ref = q if ref is None else ref[:, self.idx] * self.s + self.m
        M, Mp = (q_ref * self.w).mean((-2, -1), keepdim=True), (qp * self.w).mean((-2, -1), keepdim=True)
        qp = qp * torch.where(Mp > 0, M.clamp_min(0.0) / Mp.clamp_min(1e-30), torch.zeros_like(Mp))
        out = x.clone()
        out[:, self.idx] = (qp - self.m) / self.s
        return out


class BalanceProjector(nn.Module):
    """Same interface as PINNProjector: (x0, d~, hour0) -> d with x0 + d_k = Pi(x0 + d~_k) for every frame k.

    The IC x0 is not changed (it is the analysis); tisr is a known forcing and is never changed."""

    def __init__(self, static: torch.Tensor, mean, std, lat_deg, width: int = 32, modes=(32, 33),
                 n_layers: int = 4, bound: float = 0.8, positive: bool = True, chunk: int = 32):
        super().__init__()
        from .cfm import TISR_DYN
        N_DYN = contract.N_DYNAMIC
        self.register_buffer("static", static)
        keep = torch.ones(N_DYN)
        keep[TISR_DYN] = 0
        self.register_buffer("keep", keep.view(1, 1, N_DYN, 1, 1))
        self.net = FNONet(N_DYN + static.shape[0], N_DYN, width=width, modes=tuple(modes), n_layers=n_layers,
                          bound=bound)
        self.pos = PositiveMass(mean, std, lat_deg) if positive else None
        self.chunk = chunk

    def project_frames(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 66, H, W) -> Pi(x), ``chunk`` frames at a time."""
        out = []
        for i in range(0, len(x), self.chunk):
            f = x[i:i + self.chunk]
            y = f + self.net(torch.cat([f, self.static.expand(len(f), -1, -1, -1)], 1)) * self.keep[0]
            out.append(self.pos(y, ref=f) if self.pos is not None else y)  # Pi keeps the input frame's mass
        return torch.cat(out)

    def forward(self, x0, d, hour0=None):
        B, K, C, H, W = d.shape
        f = (x0[:, None] + d).reshape(B * K, C, H, W)
        return self.project_frames(f).view(B, K, C, H, W) - x0[:, None]

    def correction(self, x0, d, hour0=None):
        return self.forward(x0, d, hour0) - d

    def project_(self):
        self.net.project_()

    def lipschitz_bound(self) -> float:
        """A priori Lip(Pi) <= 1 + Lip(N) (the positivity layer is a clamp and a rescaling close to 1)."""
        return 1.0 + self.net.lipschitz_bound()
