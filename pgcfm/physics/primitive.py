"""Pressure-coordinate primitive-equation residuals (paper eqs. 5-8, 14-17).

The four constraint groups of PG-CFM, adapted from the NeuralGCM dynamical
core (Kochkov et al. 2024) to pressure levels:

    R_u    = D_s u - (f + u tan(phi)/a) v + d_lam Phi / (a cos phi) - C_u        (16)
    R_v    = D_s v + (f + u tan(phi)/a) u + d_phi Phi / a           - C_v        (17)
    R_T    = D_s T - R_d T_v omega / (c_p p)                         - C_T        (6)
    R_mass = [d_lam u + d_phi (v cos phi)] / (a cos phi) + d_p omega               (7, 15)
    R_hyd  = d_p Phi + R_d T_v / p                                                  (8)
    R_q    = D_s q - C_q                                       (optional moisture)

    D_s X  = d_s X + u d_lam X / (a cos phi) + v d_phi X / a + omega d_p X         (14)

Everything is in SI units on physical coordinates: lam, phi [rad], p [Pa],
s [s]. The functions take values and partial derivatives, so the same code
serves the autograd PINN (``pinn.py``) and the finite-difference grid
diagnostics (``grid.py``).

A field dict ``F`` maps a name to its value, and ``F["d<coord>_<name>"]`` to
its partial derivative, e.g. ``F["dlam_u"]``, ``F["dp_Phi"]``, ``F["ds_T"]``.
A missing ``ds_*`` term is treated as zero, which gives the tendency-free
operator A_j used by the time-integrated residual (paper eq. 18). An
optional ``F["f"]`` overrides f = 2 Omega sin(phi) (only the tilted-axis
analytic tests use it).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from ..contract import CONST, Constants

Fields = Dict[str, torch.Tensor]

PROGNOSTIC = ("u", "v", "T", "q")
EQUATIONS = ("u", "v", "T", "mass", "hyd", "q")
# Equation group of paper eq. (10): momentum has two components.
GROUP = {"u": "mom", "v": "mom", "T": "T", "mass": "mass", "hyd": "hyd", "q": "q"}

VIRTUAL_EPS = 0.608  # T_v = T (1 + 0.608 q), condensate neglected


@dataclass(frozen=True)
class ResidualScales:
    """Physical scales b_j of paper eq. (10), SI units.

    Defaults are synoptic order-of-magnitude values; ``estimate_scales``
    replaces them with residual RMS on observed pairs, which is what the
    paper prescribes.
    """

    mom: float = 1e-3  # f U ~ 1e-4 s-1 * 10 m s-1  [m s-2]
    T: float = 1e-4  # ~8.6 K/day  [K s-1]
    mass: float = 1e-5  # U / L ~ 10 m s-1 / 1000 km  [s-1]
    hyd: float = 1.0  # R_hyd is reported as a temperature error [K], see hydrostatic_in_kelvin
    q: float = 1e-8  # ~1 g/kg/day  [kg kg-1 s-1]

    def of(self, eq: str) -> float:
        return getattr(self, GROUP[eq])


@dataclass
class Closures:
    """Effective closures C_u, C_v, C_T, C_q (paper section 5.2). Zero by default."""

    terms: Dict[str, torch.Tensor] = field(default_factory=dict)

    def get(self, eq: str, like: torch.Tensor) -> torch.Tensor:
        c = self.terms.get(eq)
        return torch.zeros_like(like) if c is None else c


def coriolis(phi: torch.Tensor, const: Constants = CONST) -> torch.Tensor:
    return 2.0 * const.omega * torch.sin(phi)


def virtual_temperature(F: Fields, virtual: bool = True) -> torch.Tensor:
    if virtual and "q" in F:
        return F["T"] * (1.0 + VIRTUAL_EPS * F["q"])
    return F["T"]


def material_derivative(F: Fields, name: str, phi: torch.Tensor, const: Constants = CONST) -> torch.Tensor:
    """D_s X of eq. (14). Uses d_s X if present, otherwise only the advective part."""
    a = const.earth_radius
    out = (F["u"] / (a * torch.cos(phi))) * F[f"dlam_{name}"] + (F["v"] / a) * F[f"dphi_{name}"]
    if "omega" in F:
        out = out + F["omega"] * F[f"dp_{name}"]
    ds = F.get(f"ds_{name}")
    return out if ds is None else out + ds


def residuals(F: Fields, phi: torch.Tensor, p: torch.Tensor, const: Constants = CONST,
              closures: Optional[Closures] = None, virtual: bool = True,
              hydrostatic_in_kelvin: bool = True) -> Dict[str, torch.Tensor]:
    """Strong-form residuals R_u, R_v, R_T, R_mass, R_hyd (and R_q if q is given).

    phi, p broadcast against the fields. ``hydrostatic_in_kelvin`` reports
    R_hyd * p / R_d, i.e. the virtual-temperature error implied by the
    geopotential thickness [K], instead of [m2 s-2 Pa-1].
    """
    closures = closures or Closures()
    a, rd, cp = const.earth_radius, const.r_dry, const.cp_dry
    cos, tan = torch.cos(phi), torch.tan(phi)
    f = F["f"] if "f" in F else coriolis(phi, const)
    f_eff = f + F["u"] * tan / a
    tv = virtual_temperature(F, virtual)

    R = {}
    R["u"] = (material_derivative(F, "u", phi, const) - f_eff * F["v"]
              + F["dlam_Phi"] / (a * cos) - closures.get("u", F["u"]))
    R["v"] = (material_derivative(F, "v", phi, const) + f_eff * F["u"]
              + F["dphi_Phi"] / a - closures.get("v", F["v"]))
    omega = F.get("omega", torch.zeros_like(F["T"]))
    R["T"] = (material_derivative(F, "T", phi, const) - rd * tv * omega / (cp * p)
              - closures.get("T", F["T"]))
    # d_phi(v cos phi) / cos phi = d_phi v - v tan phi
    R["mass"] = F["dlam_u"] / (a * cos) + (F["dphi_v"] - F["v"] * tan) / a
    if "dp_omega" in F:
        R["mass"] = R["mass"] + F["dp_omega"]
    R["hyd"] = F["dp_Phi"] + rd * tv / p
    if hydrostatic_in_kelvin:
        R["hyd"] = R["hyd"] * p / rd
    if "q" in F and "dlam_q" in F:
        R["q"] = material_derivative(F, "q", phi, const) - closures.get("q", F["q"])
    return R


def physics_loss(R: Dict[str, torch.Tensor], scales: ResidualScales = ResidualScales(),
                 weights: Optional[Dict[str, float]] = None,
                 area_weight: Optional[torch.Tensor] = None,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """sum_j gamma_j E ||R_j / b_j||^2 (paper eq. 10, without the boundary term).

    ``area_weight`` (e.g. cos phi) and ``mask`` (valid collocation points)
    broadcast against the residuals; the mean is taken over valid points with
    weights normalized to sum to 1.
    """
    weights = weights or {}
    total = 0.0
    for eq, r in R.items():
        g = weights.get(eq, 1.0)
        if g == 0.0:
            continue
        sq = (r / scales.of(eq)) ** 2
        w = torch.ones_like(sq)
        if area_weight is not None:
            w = w * area_weight
        if mask is not None:
            w = w * mask
        # masked points (e.g. the 1/cos(phi) = inf pole rows) must not leak NaN
        sq = torch.where(w > 0, sq, torch.zeros_like(sq))
        total = total + g * (sq * w).sum() / w.sum()
    return total
