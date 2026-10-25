"""Adapter for an existing decoder and differentiable inverse DCT."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def decode_ensemble_forecast(
    latent_ensemble: Tensor,
    decoder: Callable[[Tensor], Tensor],
    inverse_dct: Callable[[Tensor], Tensor],
    *,
    channel_mean: Tensor,
    channel_scale: Tensor,
) -> Tensor:
    """Decode [B,E,...] to physical [B,E,C,H,W] without detaching gradients.

    Contract: channel-wise affine normalization was done in physical space
    before the original DCT. Therefore decode -> IDCT -> undo normalization.
    For z-score, channel_mean=mean and channel_scale=standard deviation; for
    min-max (ClimODE reference), channel_mean=min and channel_scale=max-min.
    If the original pipeline normalizes DCT coefficients instead, reverse that
    normalization *before* IDCT in your own adapter.

    decoder and inverse_dct process [B*E,...]; they must not mix examples.
    In particular set an existing BatchNorm decoder to eval or freeze its
    pretrained statistics when independent member behavior is required.
    """
    if latent_ensemble.ndim < 3 or any(s == 0 for s in latent_ensemble.shape):
        raise ValueError("latent_ensemble must have nonempty shape [B,E,...]")
    batch, members = latent_ensemble.shape[:2]
    normalized = inverse_dct(decoder(latent_ensemble.flatten(0, 1)))
    if normalized.ndim != 4 or normalized.shape[0] != batch * members:
        raise ValueError("decoder + inverse_dct must return [B*E,C,H,W]")
    if not normalized.is_floating_point() or not torch.isfinite(normalized).all():
        raise ValueError("decoded fields must be finite floating point tensors")
    channels = normalized.shape[1]
    mean = torch.as_tensor(channel_mean, dtype=normalized.dtype, device=normalized.device)
    scale = torch.as_tensor(channel_scale, dtype=normalized.dtype, device=normalized.device)
    if mean.numel() != channels or scale.numel() != channels:
        raise ValueError("channel_mean and channel_scale require one value per channel")
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("normalization values must be finite with positive scales")
    physical = normalized * scale.reshape(1, channels, 1, 1) + mean.reshape(1, channels, 1, 1)
    return physical.reshape(batch, members, *physical.shape[1:])
