"""Step 13, stage 3 variant: the calibration guide (owner, 2026-10-25).

The owner's proposal: split the frozen FM + PINN ensemble into two paths. One goes straight to the
PINN (the members keep their physics-corrected structure); the other goes to a neural operator that
removes the excess (over-dispersed) energy and finds the main stream. Combined per member m:

    y_m = mu*(ensemble, x0) + IDCT( Gamma(lead, var, band) * DCT(x_m - mean_m x_m) )

    mu*     the owner's EnsembleToDeterministic (pgcfm/ens_to_det.py, official neuraloperator FNO;
            zero head = the ensemble mean at the start)
    Gamma   one spread factor per spatial DCT band of the member deviations: < 1 takes energy out
            of an over-dispersed band, > 1 adds it. ``BandGuide`` predicts log Gamma from the
            ensemble's own band energies and the lead (a diagonal spectral multiplier, like the
            FNO's spectral weights but set by the ensemble state), starting at the closed form.

Controls (steps 9-10: every learned corrector lost to classical calibration):

    band MBM   the same formula with mu* = a + b mu + c x0 (step 10) and Gamma from spread-skill
               matching per (lead, var, band): Gamma^2 = E|DCT_b(o - mu*)|^2 / ((M + 1) / M E s_b^2),
               s_b^2 the unbiased member variance of the band coefficients
    global MBM one Gamma per (lead, var)

The DCT is the orthonormal 2-D DCT-II of pgcfm/seqdct.py on the 32 x 64 grid (not area weighted).
"""

import math
from typing import Sequence

import torch
import torch.nn as nn

from .seqdct import dct_2d, idct_2d

BAND_EDGES = (1 / 16, 1 / 8, 1 / 4, 1 / 2)  # normalized radius sqrt((ky / H)^2 + (kx / W)^2)


def band_index(H: int, W: int, edges: Sequence[float] = BAND_EDGES) -> torch.Tensor:
    """(H, W) band of every DCT coefficient, 0 = largest scales, len(edges) bands + 1."""
    ky, kx = torch.arange(H)[:, None] / H, torch.arange(W)[None] / W
    return torch.bucketize((ky ** 2 + kx ** 2).sqrt(), torch.tensor(edges, dtype=torch.float32))


def band_mean(c2: torch.Tensor, bidx: torch.Tensor, nb: int) -> torch.Tensor:
    """Mean of c2 (..., H, W) over the coefficients of each band -> (..., nb)."""
    return torch.stack([c2[..., bidx == b].mean(-1) for b in range(nb)], -1)


def band_spread2(members: torch.Tensor, bidx: torch.Tensor, nb: int) -> torch.Tensor:
    """Unbiased member variance of the band coefficients: members (N, M, C, H, W) -> (N, C, nb)."""
    M = members.shape[1]
    c = dct_2d(members - members.mean(1, keepdim=True))
    return band_mean((c ** 2).sum(1) / (M - 1), bidx, nb)


def band_error2(mu: torch.Tensor, obs: torch.Tensor, bidx: torch.Tensor, nb: int) -> torch.Tensor:
    """(N, C, H, W) -> (N, C, nb)."""
    return band_mean(dct_2d(obs - mu) ** 2, bidx, nb)


def apply_gamma(mu: torch.Tensor, members: torch.Tensor, gamma: torch.Tensor, bidx: torch.Tensor) -> torch.Tensor:
    """y_m = mu + IDCT(Gamma * DCT(x_m - mean)): mu (N, C, H, W), members (N, M, C, H, W), gamma (N, C, nb)."""
    dev = dct_2d(members - members.mean(1, keepdim=True))
    return mu[:, None] + idct_2d(dev * gamma[..., bidx][:, None])


def spread_skill_gamma(err2: torch.Tensor, spread2: torch.Tensor, M: int) -> torch.Tensor:
    """Closed-form Gamma from the mean over cases (dim 0) of err2 and spread2."""
    return (err2.mean(0) / (spread2.mean(0) * (M + 1) / M).clamp_min(1e-30)).sqrt()


def fair_crps(ens: torch.Tensor, obs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Area-weighted fair (finite-ensemble unbiased) CRPS: ens (N, M, C, H, W), obs (N, C, H, W)."""
    M = ens.shape[1]
    skill = (ens - obs[:, None]).abs().mean(1)
    spread = (ens[:, :, None] - ens[:, None]).abs().sum((1, 2)) / (2 * M * (M - 1))
    return ((skill - spread) * w).mean()


class BandGuide(nn.Module):
    """Gamma (N, C, nb) = exp(base[lead] + MLP(standardized log band spread, lead)); zero last layer,
    so it starts at exp(base) (base = log of the closed-form spread-skill Gamma per lead)."""

    def __init__(self, C: int, nb: int, L: int, hidden: int = 64, gamma_max: float = 4.0):
        super().__init__()
        self.C, self.nb, self.log_max = C, nb, math.log(gamma_max)
        self.register_buffer("base", torch.zeros(L, C, nb))
        self.register_buffer("s_mean", torch.zeros(C, nb))
        self.register_buffer("s_std", torch.ones(C, nb))
        self.net = nn.Sequential(nn.Linear(C * nb + 1, hidden), nn.GELU(), nn.Linear(hidden, C * nb))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @torch.no_grad()
    def fit_inputs(self, spread2: torch.Tensor) -> None:
        ls = spread2.clamp_min(1e-30).log()
        self.s_mean.copy_(ls.mean(0))
        self.s_std.copy_(ls.std(0).clamp_min(1e-3))

    def forward(self, spread2: torch.Tensor, lead_idx: torch.Tensor, lead_frac: torch.Tensor) -> torch.Tensor:
        x = ((spread2.clamp_min(1e-30).log() - self.s_mean) / self.s_std).flatten(1)
        g = self.net(torch.cat([x, lead_frac[:, None]], 1)).view(-1, self.C, self.nb)
        return torch.exp((self.base[lead_idx] + g).clamp_max(self.log_max))


# -- energy control along the lead (owner, 2026-10-25: Doob-type inequality control) ---------------
#
# If the member deviation energy |e_k|^2 = |x_m,k - mu_k|^2 is a nonnegative submartingale in the
# lead k, Doob's L^2 maximal inequality gives E[max_k |e_k|^2] <= 4 E|e_K|^2: the energy at the last
# lead bounds the whole trajectory. The FM samples the sequence in one shot and its time DCT mixes
# the leads, so the property is measured (``doob_stats``), not assumed; ``monotone_gamma`` and the
# guide's penalty make the expected energy non-decreasing in the lead.


def member_energy(ens: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """ens (n, M, L, C, H, W) -> area-weighted deviation energy per member and lead (n, M, L, C)."""
    return (((ens - ens.mean(1, keepdim=True)) ** 2) * w).mean((-1, -2))


def doob_stats(e: torch.Tensor) -> dict:
    """e (n, M, L, C) energies (members, or (n, 1, L, C) for an error process). Per channel:
    ratio E[max_k e_k] / E[e_K] (<= 4 under the submartingale property), the mean energy by lead and
    the fraction of lead steps where it falls."""
    mean_k = e.mean((0, 1))  # (L, C)
    return {"doob_ratio": (e.max(2).values.mean((0, 1)) / e[:, :, -1].mean((0, 1)).clamp_min(1e-30)).tolist(),
            "energy_by_lead": mean_k.tolist(),
            "falling_steps": (mean_k[1:] < mean_k[:-1]).float().mean(0).tolist()}


def band_counts(bidx: torch.Tensor, nb: int) -> torch.Tensor:
    return torch.bincount(bidx.flatten(), minlength=nb).float()


def calibrated_energy(gamma: torch.Tensor, spread2: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Member variance summed over the coefficients after the guide (Parseval): (..., C, nb) -> (..., C)."""
    return (gamma ** 2 * spread2 * counts).sum(-1)


def monotone_gamma(gamma: torch.Tensor, spread2: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """gamma (L, C, nb), spread2 (L, C, nb) the mean band variance per lead: raise Gamma at a lead
    (all bands alike) where the calibrated energy would fall below the previous lead's."""
    S = calibrated_energy(gamma, spread2, counts)  # (L, C)
    target = torch.cummax(S, 0).values
    return gamma * (target / S.clamp_min(1e-30)).sqrt()[..., None]
