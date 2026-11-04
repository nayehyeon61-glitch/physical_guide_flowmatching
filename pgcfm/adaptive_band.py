"""The owner's revised hypothesis (2026-10-26): case-adaptive band Gamma, T_c = sum_b Gamma_b(c) P_b.

    target    r = o - mu*(c): per-coefficient mean error energy of each orthonormal DCT band,
              Ebar_b = s_b^2(c) xi_b with xi_b ~ chi^2_{n_b} / n_b (noise CV sqrt(2 / n_b) instead of sqrt(2)
              for one grid cell)
    net       g_phi(c) -> log s_b^2 per variable and band: three 3 x 3 convolutions (longitude circular,
              latitude replicate padding), mean and max pooling, MLP with the scalars; zero last layer,
              so training starts at the climatological band energy
    loss      chi^2 NLL sum_b n_eff,b (log s_b^2 + Ebar_b / s_b^2), n_eff from the residual ratios
    operator  y_k = mu* + T_c (x_k - xbar), Gamma_b(c) = min(sqrt(s_b^2(c) / spread_b), Gamma_max):
              ||T_c||_{L2 -> L2} = max_b Gamma_b(c) exactly (Parseval), Gamma_max = 1 is non-expansive;
              spread_b = this case's realized member band spread ("realized") or the lead's
              climatological one ("marginal")

The code follows the owner's text. Changes: the climatological start log_clim is per lead (the error
energy grows about 2.5x from 24 to 144 h, so one start for all leads is far from the marginal variance),
the inverse DCT is D^T c D (the text's einsum applied D c D^T, the forward transform again, so its own
round-trip assert would fail), estimate_n_eff normalizes by a reference per row (the lead mean or the fitted s^2) instead of the mean
over all rows, which would mix the leads, and the band split can be the one regm_band uses
(pgcfm/ens_guide.py band_index: 5 radial bands, n_b = 10, 21, 80, 313, 1624 on 32 x 64).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthoDCT2(nn.Module):
    def __init__(self, H: int, W: int):
        super().__init__()
        self.register_buffer("DH", self._mat(H))
        self.register_buffer("DW", self._mat(W))

    @staticmethod
    def _mat(N: int) -> torch.Tensor:
        n = torch.arange(N, dtype=torch.float64)
        k = n[:, None]
        D = torch.cos(math.pi * (2 * n[None] + 1) * k / (2 * N)) * math.sqrt(2 / N)
        D[0] /= math.sqrt(2)
        return D.float()  # D D^T = I

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return torch.einsum("hi,...ij,wj->...hw", self.DH, u, self.DW)

    def inverse(self, c: torch.Tensor) -> torch.Tensor:  # D^T c D (the text's "ih,...hw,jw" applied D again)
        return torch.einsum("hi,...hw,wj->...ij", self.DH, c, self.DW)


def radial_bands(H: int, W: int, n_bands: int) -> torch.Tensor:
    """The text's split: coefficient (i, j) -> band, uniform in the normalized radius."""
    i = torch.arange(H)[:, None] / max(H - 1, 1)
    j = torch.arange(W)[None] / max(W - 1, 1)
    rho = torch.sqrt(i ** 2 + j ** 2) / math.sqrt(2)
    return (rho * n_bands).floor().clamp(max=n_bands - 1).long()


class BandPool(nn.Module):
    """Coefficients (..., H, W) -> per-coefficient mean energy per band (..., nb)."""

    def __init__(self, band_idx: torch.Tensor, n_bands: int):
        super().__init__()
        M = F.one_hot(band_idx.reshape(-1), n_bands).float()
        self.register_buffer("M", M)
        self.register_buffer("n_b", M.sum(0))
        self.register_buffer("band_idx", band_idx)

    def mean_energy(self, coef: torch.Tensor) -> torch.Tensor:
        return ((coef ** 2).reshape(*coef.shape[:-2], -1) @ self.M) / self.n_b

    def expand(self, per_band: torch.Tensor) -> torch.Tensor:
        return per_band[..., self.band_idx]


def _pad_sphere(x: torch.Tensor) -> torch.Tensor:  # longitude circular, latitude replicate
    x = F.pad(x, (1, 1, 0, 0), mode="circular")
    return F.pad(x, (0, 0, 1, 1), mode="replicate")


class CondBandVariance(nn.Module):
    """feats (B, F, H, W), scalars (B, S), lead_idx (B,) -> log s^2 (B, C, nb), starting at log_clim[lead]."""

    def __init__(self, n_feats: int, n_scalars: int, C: int, n_bands: int, log_clim: torch.Tensor, width: int = 64):
        super().__init__()
        self.C, self.nb = C, n_bands
        self.convs = nn.ModuleList([nn.Conv2d(n_feats if i == 0 else width, width, 3) for i in range(3)])
        self.head = nn.Sequential(nn.Linear(2 * width + n_scalars, 256), nn.GELU(), nn.Linear(256, C * n_bands))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("log_clim", log_clim)  # (L, C, nb)

    def forward(self, feats, scalars, lead_idx):
        h = feats
        for cv in self.convs:
            h = F.gelu(cv(_pad_sphere(h)))
        pooled = torch.cat([h.mean((-2, -1)), h.amax((-2, -1)), scalars], dim=-1)
        return self.log_clim[lead_idx] + self.head(pooled).view(-1, self.C, self.nb)


def chi2_band_nll(log_s2, Ebar, n_eff):
    """Ebar (B, C, nb), Ebar / s^2 ~ chi^2_{n_eff} / n_eff: pooled Gaussian NLL of the band."""
    nll = n_eff * (log_s2 + Ebar * torch.exp(-log_s2))
    return nll.sum(-1).mean() / n_eff.sum()


@torch.no_grad()
def estimate_n_eff(Ebar, ref, n_b):
    """Var(xi) = 2 / n_eff for xi = Ebar / ref (normalized to mean 1 per variable and band): (B, C, nb) -> (nb,).
    With ref = the lead's mean the case-to-case signal is counted as noise (a lower bound); with ref = the
    fitted s^2 it is the residual ratio of the text."""
    xi = Ebar / ref
    xi = xi / xi.mean(0, keepdim=True)
    return torch.minimum(2.0 / xi.var(dim=(0, 1)).clamp_min(1e-6), n_b)


class AdaptiveBandOperator(nn.Module):
    def __init__(self, dct, pool, gamma_max: float = 1.0, normalize: str = "realized"):
        super().__init__()
        self.dct, self.pool, self.gamma_max, self.normalize = dct, pool, gamma_max, normalize

    def ensemble_band_spread(self, dhat):  # (B, N, C, H, W) deviation coefficients -> (B, C, nb) unbiased
        return self.pool.mean_energy(dhat).sum(1) / (dhat.shape[1] - 1)

    def gamma(self, log_s2, dhat, sigma_marg=None):
        ref = self.ensemble_band_spread(dhat) if self.normalize == "realized" else sigma_marg
        return torch.sqrt(torch.exp(log_s2) / ref.clamp_min(1e-12)).clamp(max=self.gamma_max)

    def forward(self, x, mu_star, log_s2, sigma_marg=None):
        """x (B, N, C, H, W) members, mu_star (B, C, H, W): y_k = mu* + T_c (x_k - xbar)."""
        dhat = self.dct(x - x.mean(1, keepdim=True))
        g = self.gamma(log_s2, dhat, sigma_marg)
        y = mu_star[:, None] + self.dct.inverse(self.pool.expand(g)[:, None] * dhat)
        info = dict(op_norm=g.amax(dim=(-1, -2)), clip_frac=(g >= self.gamma_max - 1e-6).float().mean())
        return y, g, info


@torch.no_grad()
def band_corr_with_ceiling(log_s2, Ebar, n_eff):
    """corr over cases of s^2_b and Ebar_b, and the ceiling kappa / sqrt(kappa^2 + (2 / n_eff)(1 + kappa^2))
    of a perfect s^2 with this CV: (B, C, nb) -> (C, nb), (C, nb). Call it within one lead."""
    s2 = torch.exp(log_s2)

    def corr(a, b):
        a, b = a - a.mean(0), b - b.mean(0)
        return (a * b).mean(0) / (a.std(0) * b.std(0) + 1e-12)

    kappa = s2.std(0) / s2.mean(0)
    return corr(s2, Ebar), kappa / torch.sqrt(kappa ** 2 + (2 / n_eff) * (1 + kappa ** 2))


@torch.no_grad()
def case_ssr(y, obs, mu_star):
    """Per-case spread / skill over all bands: y (B, N, C, H, W)."""
    N = y.shape[1]
    spread2 = ((y - y.mean(1, keepdim=True)) ** 2).sum((2, 3, 4)).sum(1) / (N - 1)
    return torch.sqrt(spread2 / ((obs - mu_star) ** 2).sum((1, 2, 3)))
