"""Area-weighted ensemble scores (paper sec. 6.3). All inputs are normalized fields.

ens: (M, ..., H, W) members, obs: (..., H, W). Reductions keep the leading
dims of ``obs`` except the grid, so per-channel / per-lead scores come out
of a single call.
"""

import torch

from .cfm import area_weights


def _wmean(x: torch.Tensor) -> torch.Tensor:
    w = area_weights(x.shape[-2], device=x.device)
    return (x * w).mean(dim=(-2, -1))


def ens_mean_rmse(ens: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return _wmean((ens.mean(0) - obs) ** 2).sqrt()


def ens_spread(ens: torch.Tensor) -> torch.Tensor:
    """sqrt of the area mean of the unbiased member variance."""
    return _wmean(ens.var(0, unbiased=True)).sqrt()


def spread_skill_ratio(ens: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """sqrt((M+1)/M) * spread / RMSE; 1 for a calibrated ensemble (Fortin et al. 2014)."""
    M = ens.shape[0]
    return ((M + 1) / M) ** 0.5 * ens_spread(ens) / ens_mean_rmse(ens, obs)


def crps(ens: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Fair ensemble CRPS, area-weighted."""
    M = ens.shape[0]
    skill = (ens - obs).abs().mean(0)
    # sum_{i,j} |x_i - x_j| = 2 sum_i (2i - M + 1) x_(i) over sorted members (no M x M tensor)
    w = (2 * torch.arange(M, device=ens.device, dtype=ens.dtype) - M + 1).view((M,) + (1,) * (ens.dim() - 1))
    pair = (w * ens.sort(dim=0).values).sum(0) / (M * (M - 1))
    return _wmean(skill - pair)
