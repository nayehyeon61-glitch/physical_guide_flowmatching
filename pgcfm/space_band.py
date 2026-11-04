"""The owner's band x cell resolution (2026-10-26, second revision): T_c = sum_b P_b M_b(c) P_b.

    resolution  a band-limited field's local energy cannot vary faster than its band's wavelength, so band b
                gets a coarse cell grid with about n_min coefficients per cell (low bands one cell = the band
                scalar of pgcfm/adaptive_band.py, high bands tens of cells = "where is the error large")
    target      Ebar_{b,j} = mean over cell j of |P_b r|^2, r = o - mu*(c)
    net         log s^2_{b,j}(c) = clim_{b,j}[lead] + a_b(c) + m_{b,j}(c), m_b centered over the cells (a_b is the
                band amplitude, m_b the spatial redistribution); zero heads, so training starts at climatology
    operator    y_k = mu* + sum_b P_b [M_b(c, x) P_b (x_k - xbar)], M_b = min(sqrt(s^2_b / realized cell spread),
                Gamma_max) interpolated to the grid: ||T_c|| <= max_b sup_x M_b <= Gamma_max (orthogonal ranges;
                the interpolation is a convex combination, so it does not raise the sup)

Follows the owner's text. Changes: the inverse DCT of pgcfm/adaptive_band.py (fixed there), upsample_sphere
rewritten as separable linear interpolation between cell centers (longitude periodic, latitude clamped;
the text's version cropped half a cell off and returned 62 columns when 64 is not a multiple of the cell
count), latitude replicate padding in the convolutions (the text's padding_mode="circular" also wraps the
poles), and the climatological start is per lead.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_band import _pad_sphere


def band_cell_grids(n_b, H: int, W: int, n_min: int = 16):
    """Cells of band b: about n_b / n_min, aspect H : W -> list of (h_b, w_b)."""
    grids = []
    for nb in n_b.tolist():
        m = max(1, int(nb // n_min))
        h = max(1, int(round(math.sqrt(m * H / W))))
        w = max(1, m // h)
        grids.append((min(h, H), min(w, W)))
    return grids


def pool_local_energy(f: torch.Tensor, grid) -> torch.Tensor:
    """f (B, C, H, W) band-passed field -> cell mean energy (B, C, h_b, w_b)."""
    return F.adaptive_avg_pool2d(f ** 2, grid)


def _interp(n_coarse: int, n_fine: int, periodic: bool) -> torch.Tensor:
    """(n_fine, n_coarse) linear interpolation between cell centers; rows are convex weights."""
    A = torch.zeros(n_fine, n_coarse)
    if n_coarse == 1:
        return A + 1.0
    u = (torch.arange(n_fine, dtype=torch.float64) + 0.5) * n_coarse / n_fine - 0.5
    if periodic:
        i0 = torch.floor(u)
    else:
        u = u.clamp(0, n_coarse - 1)
        i0 = torch.floor(u).clamp(max=n_coarse - 2)
    t = (u - i0).float()
    i0 = i0.long()
    i1 = (i0 + 1) % n_coarse if periodic else i0 + 1
    rows = torch.arange(n_fine)
    A.index_put_((rows, i0 % n_coarse), 1 - t, accumulate=True)
    A.index_put_((rows, i1), t, accumulate=True)
    return A


def upsample_sphere(m: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(..., h_b, w_b) -> (..., H, W): longitude periodic, latitude clamped, convex weights."""
    return torch.einsum("hi,...ij,wj->...hw", _interp(m.shape[-2], H, False).to(m), m,
                        _interp(m.shape[-1], W, True).to(m))


class BandProjector(nn.Module):
    def __init__(self, dct, pool):
        super().__init__()
        self.dct = dct
        self.register_buffer("masks", F.one_hot(pool.band_idx, int(pool.n_b.numel())).permute(2, 0, 1).float())

    def project(self, u: torch.Tensor, b: int) -> torch.Tensor:
        return self.dct.inverse(self.dct(u) * self.masks[b])

    def project_all(self, u: torch.Tensor):
        c = self.dct(u)
        return [self.dct.inverse(c * mk) for mk in self.masks]


def local_error_energy(proj: BandProjector, r: torch.Tensor, grids):
    return [pool_local_energy(rb, g) for rb, g in zip(proj.project_all(r), grids)]


class CondSpaceBandVariance(nn.Module):
    """log s^2_b(x; c) = clim_b[lead] + a_b(c) + m_b(x; c): list of (B, C, h_b, w_b)."""

    def __init__(self, n_feats: int, n_scalars: int, C: int, grids, log_clim, width: int = 64):
        super().__init__()
        self.C, self.grids = C, grids
        self.convs = nn.ModuleList([nn.Conv2d(n_feats if i == 0 else width, width, 3) for i in range(3)])
        self.scal = nn.Linear(n_scalars, width)
        self.amp_head = nn.Linear(2 * width, C * len(grids))
        self.sp_heads = nn.ModuleList([nn.Conv2d(width, C, 1) for _ in grids])
        for h in [self.amp_head, *self.sp_heads]:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        for b, lc in enumerate(log_clim):  # (L, C, h_b, w_b)
            self.register_buffer(f"log_clim_{b}", lc)

    def forward(self, feats, scalars, lead_idx):
        h = feats
        for cv in self.convs:
            h = F.gelu(cv(_pad_sphere(h)))
        h = h + self.scal(scalars)[:, :, None, None]
        a = self.amp_head(torch.cat([h.mean((-2, -1)), h.amax((-2, -1))], -1)).view(-1, self.C, len(self.grids))
        out = []
        for b, g in enumerate(self.grids):
            m = self.sp_heads[b](F.adaptive_avg_pool2d(h, g))
            m = m - m.mean((-2, -1), keepdim=True)
            out.append(getattr(self, f"log_clim_{b}")[lead_idx] + a[:, :, b, None, None] + m)
        return out


def space_band_nll(log_s2_list, Ebar_list, n_eff):
    tot, wsum = 0.0, 0.0
    for b, (ls, E) in enumerate(zip(log_s2_list, Ebar_list)):
        tot = tot + n_eff[b] * (ls + E * torch.exp(-ls)).mean()
        wsum = wsum + n_eff[b]
    return tot / wsum


@torch.no_grad()
def estimate_cell_n_eff(Ebar_list, clim_list, lead_idx, n_b, grids):
    """Per band: 2 / Var(xi), xi = Ebar_{b,j} / clim_{b,j}[lead] normalized to mean 1 per (C, cell); capped at
    n_b / cells. Counts the case-to-case signal as noise, so a lower bound."""
    out = []
    for E, cl, nb, g in zip(Ebar_list, clim_list, n_b.tolist(), grids):
        xi = E / cl[lead_idx]
        xi = xi / xi.mean(0, keepdim=True)
        out.append(min(float(2.0 / xi.var().clamp_min(1e-6)), nb / (g[0] * g[1])))
    return torch.tensor(out)


class SpaceBandOperator(nn.Module):
    def __init__(self, proj: BandProjector, grids, gamma_max: float = 1.0):
        super().__init__()
        self.proj, self.grids, self.gamma_max = proj, grids, gamma_max

    def forward(self, x, mu_star, log_s2_list):
        """x (B, N, C, H, W), mu_star (B, C, H, W): y_k = mu* + sum_b P_b[M_b(x) P_b(x_k - xbar)], realized spread."""
        B, N, C, H, W = x.shape
        d = (x - x.mean(1, keepdim=True)).reshape(B * N, C, H, W)
        y = mu_star[:, None].expand_as(x).clone()
        sup, clip = torch.zeros(B), 0.0
        for b, (db, g) in enumerate(zip(self.proj.project_all(d), self.grids)):
            ref = pool_local_energy(db, g).reshape(B, N, C, *g).sum(1) / (N - 1)
            gam = torch.sqrt(torch.exp(log_s2_list[b]) / ref.clamp_min(1e-12)).clamp(max=self.gamma_max)
            clip += float((gam >= self.gamma_max - 1e-6).float().mean()) / len(self.grids)
            Gfull = upsample_sphere(gam, H, W)  # (B, C, H, W)
            sup = torch.maximum(sup, Gfull.amax((-1, -2, -3)))
            y = y + self.proj.project((Gfull[:, None] * db.reshape(B, N, C, H, W)).reshape(B * N, C, H, W),
                                      b).reshape(B, N, C, H, W)
        return y, dict(op_norm_bound=sup, clip_frac=clip)


@torch.no_grad()
def cell_corr_and_ceiling(log_s2_list, Ebar_list, n_eff):
    """Within one lead: per band, corr over cases of s^2 and Ebar per (C, cell) and the ceiling, averaged."""
    out = []
    for b, (ls, E) in enumerate(zip(log_s2_list, Ebar_list)):
        s2 = torch.exp(ls)
        a, e = s2 - s2.mean(0), E - E.mean(0)
        corr = (a * e).mean(0) / (a.std(0, unbiased=False) * e.std(0, unbiased=False) + 1e-12)
        kappa = s2.std(0) / s2.mean(0)
        ceil = kappa / torch.sqrt(kappa ** 2 + (2 / n_eff[b]) * (1 + kappa ** 2))
        out.append((corr.mean(-1).mean(-1), ceil.mean(-1).mean(-1)))  # (C,), (C,)
    return out
