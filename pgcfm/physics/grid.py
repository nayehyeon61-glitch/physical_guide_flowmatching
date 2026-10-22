"""Finite-difference residuals on the KAI grid (independent physics diagnostic).

Evaluates the primitive-equation residuals of ``primitive.py`` directly on
gridded daily-mean states [..., level, lat, lon] (1.5 deg, 121 x 240, 12
pressure levels, lat ascending). The paper uses this as the independent
check of the PINN (section 6.3) and for the residual of generated endpoints;
all operations are torch ops, so the residual is differentiable with respect
to the input fields.

Pole handling (decided in docs/step3_pinn_residual.md, experiment E5):
  * The two pole rows (lat = +-90) are never collocation points. The terms
    1/(a cos phi) and tan(phi) are infinite there and u, v depend on the
    meridian. Residuals are returned on rows 1..H-2 only (``lat_rows``).
  * Pole rows still enter as neighbours in the centred d/dphi of row 1 and
    H-2, which is consistent for smooth fields.
  * d/dlam is spectral with a polar Fourier filter poleward of 60 deg, so
    grid-scale zonal noise is not amplified by 1/cos(phi).
  * Residual losses use area weights cos(phi) (``latitude_weight``).
"""

import math
from typing import Dict, Optional, Sequence

import torch

from .. import contract
from ..contract import CONST, Constants
from . import primitive

Fields = Dict[str, torch.Tensor]


class SphereGrid:
    """Coordinates and derivative operators for the KAI pressure-level grid."""

    def __init__(self, lat_deg: Sequence[float] = contract.LATITUDES,
                 lon_deg: Sequence[float] = contract.LONGITUDES,
                 levels_pa: Sequence[float] = contract.LEVELS_PA,
                 lon_method: str = "spectral", vertical: str = "logp",
                 polar_filter_deg: Optional[float] = 60.0,
                 dtype=torch.float64, device=None):
        kw = dict(dtype=dtype, device=device)
        self.lat = torch.as_tensor(lat_deg, **kw) * math.pi / 180.0
        self.lon = torch.as_tensor(lon_deg, **kw) * math.pi / 180.0
        self.p = torch.as_tensor(levels_pa, **kw)
        self.dlon = float(self.lon[1] - self.lon[0])
        if lon_method not in ("centered", "spectral"):
            raise ValueError(lon_method)
        if vertical not in ("p", "logp"):
            raise ValueError(vertical)
        self.lon_method, self.vertical = lon_method, vertical
        self.polar_filter = None
        if polar_filter_deg is not None:
            # Arakawa-Lamb style polar Fourier filter: keep zonal wavenumbers
            # k <= (N/2) cos(phi) / cos(phi_c) poleward of phi_c, so the
            # resolved zonal length scale never drops below its value at phi_c.
            n = len(self.lon)
            k = torch.arange(n // 2 + 1, **kw)
            kmax = (n / 2) * torch.cos(self.lat) / math.cos(math.radians(polar_filter_deg))
            self.polar_filter = (k.view(1, -1) <= kmax.clamp(min=1.0).view(-1, 1)).to(dtype)  # [H, K]
        pole = torch.isclose(self.lat.abs(), torch.tensor(math.pi / 2, **kw))
        self.lat_rows = torch.nonzero(~pole).squeeze(-1)  # rows where residuals are defined

    # broadcast helpers for [..., L, H, W]
    def phi3(self, rows: bool = False) -> torch.Tensor:
        lat = self.lat[self.lat_rows] if rows else self.lat
        return lat.view(1, -1, 1)

    def p3(self) -> torch.Tensor:
        return self.p.view(-1, 1, 1)

    def d_lam(self, x: torch.Tensor) -> torch.Tensor:
        """d/dlam, periodic. The polar filter (if set) is applied to the result."""
        n = x.shape[-1]
        if self.lon_method == "centered":
            d = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1)) / (2.0 * self.dlon)
            if self.polar_filter is None:
                return d
            return torch.fft.irfft(torch.fft.rfft(d, dim=-1) * self.polar_filter, n=n, dim=-1)
        k = torch.fft.rfftfreq(n, d=1.0 / n).to(device=x.device, dtype=x.dtype)  # integer wavenumbers
        if n % 2 == 0:
            k[-1] = 0.0  # drop the Nyquist mode
        xh = torch.fft.rfft(x, dim=-1) * (1j * k)
        if self.polar_filter is not None:
            xh = xh * self.polar_filter
        return torch.fft.irfft(xh, n=n, dim=-1)

    def d_phi(self, x: torch.Tensor) -> torch.Tensor:
        return torch.gradient(x, spacing=(self.lat,), dim=-2, edge_order=2)[0]

    def d_p(self, x: torch.Tensor) -> torch.Tensor:
        if self.vertical == "p":
            return torch.gradient(x, spacing=(self.p,), dim=-3, edge_order=2)[0]
        dx = torch.gradient(x, spacing=(torch.log(self.p),), dim=-3, edge_order=2)[0]
        return dx / self.p3()

    def latitude_weight(self, mode: str = "area", pole_cap_deg: Optional[float] = None) -> torch.Tensor:
        """Loss weight on residual rows [H-2, 1], zero poleward of the cap.

        "area":   cos(phi), the grid-cell area.
        "metric": cos(phi)^2, i.e. area weight on the residual of the
                  equations multiplied by cos(phi) (flux form). This cancels
                  the 1/cos(phi) amplification of grid-scale noise in the
                  d_lam terms (docs/step3, E5) at the cost of down-weighting
                  genuine polar residuals by another cos(phi).
        "none":   1.
        """
        lat = self.lat[self.lat_rows]
        w = {"area": torch.cos(lat), "metric": torch.cos(lat) ** 2, "none": torch.ones_like(lat)}[mode]
        if pole_cap_deg is not None:
            w = torch.where(lat.abs() <= math.radians(pole_cap_deg) + 1e-12, w, torch.zeros_like(w))
        return w.view(-1, 1)


def fields_from_state(X: torch.Tensor) -> Fields:
    """67-channel state in SI units [..., 67, H, W] -> pressure-level fields [..., 12, H, W].

    Phi is geopotential z (m2 s-2), not height. omega is not in the data.
    """
    sl = contract.level_slice
    return {"u": X[..., sl("u"), :, :], "v": X[..., sl("v"), :, :], "T": X[..., sl("t"), :, :],
            "Phi": X[..., sl("z"), :, :], "q": X[..., sl("q"), :, :],
            "sp": X[..., contract.CHANNEL_INDEX["sp"], :, :]}


def divergence(F: Fields, grid: SphereGrid, const: Constants = CONST) -> torch.Tensor:
    """Horizontal divergence of eq. (15) on all rows (inf/NaN on pole rows)."""
    cos = torch.cos(grid.phi3())
    return (grid.d_lam(F["u"]) + grid.d_phi(F["v"] * cos)) / (const.earth_radius * cos)


def diagnose_omega(F: Fields, grid: SphereGrid, omega_top: float = 0.0,
                   const: Constants = CONST) -> torch.Tensor:
    """omega(p) = omega_top - int_{p_top}^{p} div(u) dp' (trapezoid over levels).

    The data contain no omega. Diagnosing it from continuity makes R_mass
    zero up to the mismatch between trapezoid integration and the centred
    d/dp, so R_mass is not an independent check on reanalysis pairs.
    ``omega_top`` at 50 hPa is an assumption (paper appendix B).
    """
    div = divergence(F, grid, const)
    dp = torch.diff(grid.p).view(-1, 1, 1)
    inc = 0.5 * (div[..., 1:, :, :] + div[..., :-1, :, :]) * dp
    top = torch.full_like(div[..., :1, :, :], omega_top)
    return torch.cat([top, top - torch.cumsum(inc, dim=-3)], dim=-3)


def _derivatives(F: Fields, grid: SphereGrid, names: Sequence[str]) -> Fields:
    out = dict(F)
    for n in names:
        out[f"dlam_{n}"] = grid.d_lam(F[n])
        out[f"dphi_{n}"] = grid.d_phi(F[n])
        out[f"dp_{n}"] = grid.d_p(F[n])
    return out


def _rows(F: Fields, grid: SphereGrid) -> Fields:
    return {k: v.index_select(-2, grid.lat_rows) for k, v in F.items() if v.dim() >= 3}


def strong_residuals(F: Fields, grid: SphereGrid, const: Constants = CONST,
                     closures: Optional[primitive.Closures] = None, virtual: bool = True,
                     tendencies: Optional[Fields] = None) -> Dict[str, torch.Tensor]:
    """Residuals on rows ``grid.lat_rows`` from gridded fields.

    F needs u, v, T, Phi and optionally q, omega (diagnosed from continuity
    if absent). ``tendencies`` maps a prognostic name to d_s X; without it
    the result is the tendency-free operator A_j (eq. 18).
    """
    F = dict(F)
    F.pop("sp", None)
    if "omega" not in F:
        F["omega"] = diagnose_omega(F, grid, const=const)
    names = [n for n in ("u", "v", "T", "Phi", "q", "omega") if n in F]
    D = _rows(_derivatives(F, grid, names), grid)
    for n, t in (tendencies or {}).items():
        D[f"ds_{n}"] = t.index_select(-2, grid.lat_rows)
    return primitive.residuals(D, grid.phi3(rows=True), grid.p3(), const, closures, virtual)


def time_integrated_residuals(F0: Fields, F1: Fields, grid: SphereGrid, dt: float = CONST.dt_seconds,
                              n_sub: int = 4, const: Constants = CONST,
                              closures: Optional[primitive.Closures] = None,
                              virtual: bool = True) -> Dict[str, torch.Tensor]:
    """Paper eq. (18) with the linear bridge B(s) = (1-a) X0 + a X1 (h_psi = 0).

    R_j = (X1 - X0)/dt + (1/dt) int_0^dt A_j(B(s)) ds for the prognostic
    equations (Simpson rule over n_sub intervals, n_sub even); the
    diagnostic constraints (mass, hydrostatic) are averaged over the same
    nodes. The linear bridge is exact only for fields linear in s; for a
    moving pattern it is not (see docs, weakness "linear bridge").
    """
    if n_sub % 2:
        raise ValueError("Simpson rule needs an even n_sub")
    w = torch.ones(n_sub + 1, dtype=torch.float64)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    w = w / w.sum()
    keys = [k for k in ("u", "v", "T", "Phi", "q", "omega", "f") if k in F0 and k in F1]
    acc = None
    for i in range(n_sub + 1):
        a = i / n_sub
        Fs = {k: (1.0 - a) * F0[k] + a * F1[k] for k in keys}
        A = strong_residuals(Fs, grid, const, closures, virtual)
        acc = {k: w[i].item() * v for k, v in A.items()} if acc is None else \
            {k: acc[k] + w[i].item() * v for k, v in A.items()}
    for n in primitive.PROGNOSTIC:
        if n in acc:
            acc[n] = acc[n] + ((F1[n] - F0[n]) / dt).index_select(-2, grid.lat_rows)
    return acc


def residual_rms_by_lat(R: torch.Tensor, grid: SphereGrid) -> torch.Tensor:
    """RMS over (level, lon) for each residual row -> [H-2]."""
    dims = tuple(d for d in range(R.dim()) if d != R.dim() - 2)
    return torch.sqrt((R ** 2).mean(dim=dims))


def above_ground_mask(sp: torch.Tensor, grid: SphereGrid) -> torch.Tensor:
    """1 where the pressure level is above the surface (p <= sp), on residual rows.

    Same rule as ``contract.below_ground_mask``; below-ground levels are
    ERA5 extrapolations and are excluded from the physics loss.
    """
    m = (grid.p3() <= sp.unsqueeze(-3)).to(sp.dtype)
    return m.index_select(-2, grid.lat_rows)
