"""Step 16: a tendency network integrated by an ODE solver, trained on rollouts and on the STRONG-form
primitive-equation residual with its own tendency (docs/step16_dynpinn.md).

The stage-1 PINN of step 13 (pgcfm/seqdct.py PINNProjector) corrects a given increment sequence and
learns its physics from the time-integrated residual of 6 h pairs joined by a linear bridge; it only
lowered the hydrostatic residual. Here the network is the time derivative itself:

    dx/dt = f_theta(x, static, time)          x: the 66 KAI dynamic channels, normalized
    x_{k+1} = solver(f_theta, x_k, 6 h)        Euler or Heun substeps of ``dt_hours``
    R_j(x, f_theta(x)) = 0                     eq. (16), (17), (6), (7), (8) with d_s X := f_theta(x)

so the physics loss needs no bridge between frames (pgcfm/physics/grid.py ``strong_residuals`` with
``tendencies``), and the solver path replaces the linear interpolation. Top-of-atmosphere radiation
(tisr) is a known forcing: the rollout takes it from the data. ``PositiveMass`` (step 14) keeps
q, tcwv >= 0 after every step.

Reused: ClimODE's Climate_ResNet_2D through pgcfm.core.Net (zero head = persistence at the start),
the step 3 residual operators, the step 13 physics guide for units and masks.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from . import contract
from .balance import PositiveMass
from .cfm import TISR_DYN
from .core import Net, time_features
from .coupling import LEVEL_DYN, PhysicsGuide
from .physics import grid as pgrid
from .physics import primitive
from .seqdct import _NetCfg

N_DYN = contract.N_DYNAMIC
HOURS_PER_STEP = 6.0
SI_NAME = {"u": "u", "v": "v", "t": "T", "q": "q"}  # prognostic level variables with a tendency


@dataclass
class DynCoreConfig:
    hidden: int = 48
    blocks: tuple = (2, 2)
    dt_hours: float = 6.0     # solver substep
    solver: str = "heun"      # euler | heun
    positive: bool = True     # PositiveMass after every 6 h step
    balance: bool = False     # head 1: a correction c(x) of z and t (hydrostatic structure) before every step
    bal_detach: bool = False  # the solver starts from x + c.detach(): c learns from the hydrostatic loss only
    gfeat: bool = False       # step 18: the discrete operator's tendency G(x) = -R0(x) as 48 extra input channels


BAL_DYN = LEVEL_DYN["z"] + LEVEL_DYN["t"]  # the channels head 1 may correct (the hydrostatic pair)
G_EQS = ("u", "v", "T", "q")  # prognostic equations whose operator tendency becomes a feature
N_GFEAT = len(G_EQS) * len(contract.LEVELS_HPA)  # 48


class DynCore(nn.Module):
    """x0 (B, 66, H, W) normalized, hour0 (B,) hours since Jan 1 -> states every 6 h.

    ``tendency`` is in normalized units per 6 h step (scaled by the training tendency std ``t6`` so the
    zero-initialized head starts as persistence); tisr is never predicted.

    With ``cfg.balance`` (the owner's two-output PINN, 2026-10-27): one trunk pass on the data x gives
    (1) a correction c(x) of the z and t levels, x~ = x + c(x) (the current state on the hydrostatic
    manifold), and (2) dx/dt = f(x); the solver starts from x~ with k1 = f(x) and evaluates f again at its
    intermediate states: x_{t+6h} = x~ + (k1 + f(x~ + k1)) / 2 (Heun).
    """

    def __init__(self, cfg: DynCoreConfig, static: torch.Tensor, t6: torch.Tensor, mean, std, lat_deg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("static", static)
        self.register_buffer("t6", t6.view(1, N_DYN, 1, 1).clone())
        keep = torch.ones(N_DYN)
        keep[TISR_DYN] = 0
        self.register_buffer("keep", keep.view(1, N_DYN, 1, 1))
        self.net = Net(N_DYN + static.shape[0] + 4 + (N_GFEAT if cfg.gfeat else 0), N_DYN,
                       _NetCfg(cfg.hidden, cfg.blocks))
        self.pm = PositiveMass(mean, std, lat_deg) if cfg.positive else None
        self.features = None  # cfg.gfeat: an OperatorFeatures callable x -> G(x) (attached after construction)
        if cfg.gfeat:
            self.register_buffer("gscale", torch.ones(1, N_GFEAT, 1, 1))
        if cfg.balance:
            self.bal_head = nn.Conv2d(cfg.hidden, len(BAL_DYN), 1)
            nn.init.zeros_(self.bal_head.weight)
            nn.init.zeros_(self.bal_head.bias)
            self.register_buffer("bal_idx", torch.tensor(BAL_DYN))
        self.last_balanced = None  # x~ of the last step (for the loss and the gates)

    def _trunk(self, x: torch.Tensor, hours: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _, H, W = x.shape
        parts = [x, self.static.expand(B, -1, -1, -1), time_features(hours, H, W)]
        if self.cfg.gfeat:
            if self.features is None:
                raise RuntimeError("cfg.gfeat needs model.features = OperatorFeatures(...)")
            parts.append(self.features(x) if g is None else g)
        return self.net.trunk(torch.cat(parts, 1))

    def tendency(self, x: torch.Tensor, hours: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        """f_theta(x): normalized increment per 6 h (B, 66, H, W); ``g`` = precomputed G(x) (cfg.gfeat)."""
        return self.net.head(self._trunk(x, hours, g)) * self.t6 * self.keep

    def heads(self, x: torch.Tensor, hours: torch.Tensor, g: Optional[torch.Tensor] = None
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both outputs of one pass: (f(x), x~ = x + c(x)), c in units of the 6 h tendency std."""
        h = self._trunk(x, hours, g)
        f = self.net.head(h) * self.t6 * self.keep
        c = self.bal_head(h) * self.t6[:, self.bal_idx]
        return f, torch.index_add(x, 1, self.bal_idx, c)

    def step(self, x: torch.Tensor, hours: torch.Tensor, tisr_next: Optional[torch.Tensor]) -> torch.Tensor:
        """One 6 h step of the solver; tisr_next (B, H, W) is the forcing at the end of the step."""
        n_sub = max(1, int(round(HOURS_PER_STEP / self.cfg.dt_hours)))
        h = 1.0 / n_sub  # in units of the 6 h step
        for i in range(n_sub):
            t = hours + i * HOURS_PER_STEP * h
            g = self.features(x) if self.cfg.gfeat else None  # one operator evaluation per substep (k2 reuses it)
            if self.cfg.balance and i == 0:
                k1, xb = self.heads(x, t, g)
                self.last_balanced = xb
                x = x + (xb - x).detach() if self.cfg.bal_detach else xb
            else:
                k1 = self.tendency(x, t, g)
            if self.cfg.solver == "euler":
                x = x + h * k1
            elif self.cfg.solver == "heun":
                k2 = self.tendency(x + h * k1, t + HOURS_PER_STEP * h, g)
                x = x + 0.5 * h * (k1 + k2)
            else:
                raise ValueError(self.cfg.solver)
        if tisr_next is not None:
            x = torch.cat([x[:, :TISR_DYN], tisr_next[:, None], x[:, TISR_DYN + 1:]], 1)
        if self.pm is not None:
            x = self.pm(x)
        return x

    def rollout(self, x0: torch.Tensor, hour0: torch.Tensor, steps: int,
                tisr: Optional[torch.Tensor] = None, balanced: bool = False):
        """(B, steps, 66, H, W): the states 6 h, 12 h, ... after x0; tisr (B, steps, H, W) from the data.
        ``balanced`` (with cfg.balance) also returns the corrected states x~ at 0, 6, ... (steps - 1) x 6 h."""
        x, hours, out, bal = x0, hour0.float(), [], []
        for k in range(steps):
            x = self.step(x, hours, None if tisr is None else tisr[:, k])
            hours = hours + HOURS_PER_STEP
            out.append(x)
            if balanced:
                bal.append(self.last_balanced)
        y = torch.stack(out, 1)
        return (y, torch.stack(bal, 1)) if balanced else y


class StrongPhysics:
    """Strong-form residuals of a normalized state x with the network's tendency f (per 6 h), through the
    step 13 guide for the units, the grid and the masks. ``scales`` are the RMS of the same residuals on
    ERA5 with centred-difference tendencies (``estimate_strong_scales``), so 1 = ERA5's own level."""

    def __init__(self, guide: PhysicsGuide, t6: torch.Tensor, scales: Optional[primitive.ResidualScales] = None,
                 dt_seconds: float = HOURS_PER_STEP * 3600.0):
        self.guide, self.scales, self.dt = guide, scales or primitive.ResidualScales(), dt_seconds
        self.t6 = t6.view(N_DYN, 1, 1).double()
        self.std = guide.std.view(-1)[contract.DYNAMIC_CHANNELS].view(N_DYN, 1, 1)

    def residuals(self, x: torch.Tensor, orog: torch.Tensor, f: torch.Tensor
                  ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """x, f (N, 66, H, W) normalized (f per 6 h) -> residuals on the lat rows, collocation mask."""
        F = self.guide.to_si(x, orog)
        si = f.double() * self.std / self.dt  # SI units per second (t6 is already inside f)
        tend = {SI_NAME[v]: si[:, LEVEL_DYN[v]] for v in SI_NAME}
        R = pgrid.strong_residuals(F, self.guide.grid(x.device), tendencies=tend)
        return R, self.guide.mask(x, x, orog)

    def mean_squares(self, R: Dict[str, torch.Tensor], mask: torch.Tensor,
                     scales: Optional[primitive.ResidualScales] = None) -> Dict[str, torch.Tensor]:
        """Weighted mean of (R_j / b_j)^2 per equation (1 = ERA5's level), kept in the graph."""
        scales = scales or self.scales
        g = self.guide.grid(mask.device)
        lw = g.latitude_weight(self.guide.cfg.lat_weight, pole_cap_deg=self.guide.cfg.pole_cap_deg)
        wt = lw * mask
        out = {}
        for k, r in R.items():
            sq = (r / scales.of(k)) ** 2
            sq = torch.where(wt > 0, sq, torch.zeros_like(sq))
            out[k] = (sq * wt).sum() / wt.sum()
        return out

    def loss(self, ms: Dict[str, torch.Tensor], gamma: Dict[str, float], target: float = 1.0) -> torch.Tensor:
        """sum_j gamma_j max(m_j - target, 0): ``target`` = 1 asks for ERA5's residual level, not zero
        (a resting atmosphere has zero residual, step 3's amplitude damping); ``target`` = 0 is eq. (10)."""
        total = None
        for k, m in ms.items():
            g = gamma.get(k, 0.0)
            if g <= 0:
                continue
            term = g * torch.clamp(m - target, min=0.0) if target > 0 else g * m
            total = term if total is None else total + term
        return total if total is not None else torch.zeros((), dtype=torch.float64, device=mask_device(ms))

    def per_equation(self, x: torch.Tensor, orog: torch.Tensor, f: torch.Tensor,
                     x_obs: Optional[torch.Tensor] = None, chunk: int = 8) -> Dict[str, float]:
        """RMS of R_j / b_j per equation over the states x (N, 66, H, W), ``chunk`` states per call; the
        collocation mask from ``x_obs`` (default x itself)."""
        acc, n = {}, x.shape[0]
        with torch.no_grad():
            for i in range(0, n, chunk):
                j = slice(i, min(n, i + chunk))
                R, m = self.residuals(x[j], orog[j], f[j])
                if x_obs is not None:
                    m = self.guide.mask(x_obs[j], x_obs[j], orog[j])
                for k, v in self.mean_squares(R, m).items():
                    acc[k] = acc.get(k, 0.0) + float(v) * (j.stop - j.start) / n
        return {k: math.sqrt(v) for k, v in acc.items()}


class OperatorFeatures:
    """Step 18: G(x) = -R0(x), the discrete primitive-equation operator's own tendency of (u, v, T, q) on the
    12 levels (48 channels), as an input feature of the core. The operator predicts ERA5's 6 h tendency with
    correlation 0.5 (u, v) / 0.35 (T) and twice its magnitude (docs/step18), so it is offered to the network
    as a hint, not integrated: computed without gradient, divided by its ERA5 RMS per channel (``scale``,
    a buffer of the core), zero below ground, clamped at ``clamp``. The KAI grid has no pole rows, so the
    residual rows are all H rows."""

    def __init__(self, phys: StrongPhysics, orog: torch.Tensor, scale: Optional[torch.Tensor] = None,
                 clamp: float = 5.0):
        self.phys, self.orog, self.scale, self.clamp = phys, orog[:1], scale, clamp
        if len(phys.guide.grid(orog.device).lat_rows) != orog.shape[-2]:
            raise ValueError("operator features need a grid without pole rows (residual rows = all rows)")

    def raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(-R0 (B, 48, H, W) float64, valid-point mask (B, 48, H, W))."""
        R0, m = self.phys.residuals(x, self.orog.expand(x.shape[0], -1, -1, -1), torch.zeros_like(x))
        return -torch.cat([R0[q] for q in G_EQS], 1), m.repeat(1, len(G_EQS), 1, 1)

    def estimate_scale(self, states: torch.Tensor, chunk: int = 8) -> torch.Tensor:
        """RMS of G over valid points of ERA5 ``states`` (N, 66, H, W) per channel -> (1, 48, 1, 1)."""
        sq, n = 0.0, 0.0
        with torch.no_grad():
            for i in range(0, states.shape[0], chunk):
                g, m = self.raw(states[i:i + chunk])
                sq = sq + ((g * m) ** 2).sum((0, 2, 3))
                n = n + m.sum((0, 2, 3))
        return torch.sqrt(sq / n.clamp_min(1.0)).clamp_min(1e-30).view(1, -1, 1, 1).float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            g, m = self.raw(x.detach())
            g = (g / self.scale.to(g.dtype)).clamp(-self.clamp, self.clamp) * m
        return g.to(x.dtype)


def mask_device(ms):
    return next(iter(ms.values())).device if ms else "cpu"


def finite_difference_tendency(x_prev: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
    """ERA5's own tendency per 6 h at the middle frame (centred difference over 12 h)."""
    return 0.5 * (x_next - x_prev)


def estimate_strong_scales(phys: StrongPhysics, triples, floor_frac: float = 0.1) -> primitive.ResidualScales:
    """b_j = RMS of the strong residual on ERA5 states with centred-difference tendencies, floored at
    ``floor_frac`` x the synoptic default (as coupling.estimate_scales)."""
    sums, n = {}, 0
    with torch.no_grad():
        for x_prev, x, x_next, orog in triples:
            R, m = phys.residuals(x, orog, finite_difference_tendency(x_prev, x_next))
            for k, v in phys.mean_squares(R, m, _unit_scales()).items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
    rms = {k: math.sqrt(v / n) for k, v in sums.items()}
    d = primitive.ResidualScales()
    raw = {"mom": math.sqrt(0.5 * (rms["u"] ** 2 + rms["v"] ** 2)), "T": rms["T"], "mass": rms["mass"],
           "hyd": rms["hyd"], "q": rms.get("q", d.q)}
    return primitive.ResidualScales(**{k: max(v, floor_frac * getattr(d, k)) for k, v in raw.items()})


def _unit_scales() -> primitive.ResidualScales:
    return primitive.ResidualScales(mom=1.0, T=1.0, mass=1.0, hyd=1.0, q=1.0)


def kinetic_energy(x: torch.Tensor, std: torch.Tensor, mean: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Area-weighted mean of (u^2 + v^2) / 2 over the 12 levels, physical units; x (..., 66, H, W)."""
    u = x[..., LEVEL_DYN["u"], :, :] * std[LEVEL_DYN["u"]].view(-1, 1, 1) + mean[LEVEL_DYN["u"]].view(-1, 1, 1)
    v = x[..., LEVEL_DYN["v"], :, :] * std[LEVEL_DYN["v"]].view(-1, 1, 1) + mean[LEVEL_DYN["v"]].view(-1, 1, 1)
    return (0.5 * (u ** 2 + v ** 2) * w).mean((-1, -2, -3))
