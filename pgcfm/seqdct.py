"""Step 13: sequence PG-CFM in a lossless DCT basis (docs/step13_sequence.md).

The owner's design (2026-10-25), three stages:

    1. PINN pretraining: DCT(field) -> PINN -> IDCT -> physical field; the PINN is then frozen.
    2. time sequence -> DCT -> flow matching (a sequence ensemble) -> the frozen PINN on each
       member -> IDCT.
    3. FM and PINN frozen; a neural operator on the members at each t -> deterministic forecast.

The owner's reason for the DCT: without it, the FM ensemble just fits itself to the PINN (step 8
measured this: a physics-trained Q removed 15-24 % of the member variance per step).

What the step 13 measurements fixed (docs/step13_sequence.md, section 1):

  * no truncation. At 6 h, cutting the fields to 16 x 32 modes loses 8.8x the 6 h change of sp
    (orography) and 1.3x of q; on increments x(t + tau) - x(t) it still loses 0.5-0.7x of q, u and
    tcwv. The DCT is a lossless orthogonal basis here, standardized per coefficient, and the
    sequence is the increment d_k = x(t + 6k h) - x(t), k = 1..K.
  * the PINN corrects only the low spatial modes (k_lat x k_lon) and passes every other mode through
    (identity skip, zero-initialized head), so it cannot absorb the FM's diversity there.
  * ERA5's own 6 h primitive-equation residual (linear bridge, no closure) is large: an error of
    one 6 h tendency raises it only 1.3-3.6x. The PINN is therefore trained as a projection
    (stay close to its input while lowering the residual), with denoising (towards the truth) as
    the alternative objective.

The PINN is ClimODE's Climate_ResNet_2D (vendored) on the grid, like pgcfm/core.py: its input is
the whole increment sequence, its output is restricted to the low DCT modes before the IDCT.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch_dct

from . import contract
from .cfm import TISR_DYN
from .core import Net, time_features

N_DYN = contract.N_DYNAMIC
HOURS_PER_STEP = 6


def dct_t(d: torch.Tensor) -> torch.Tensor:
    """Orthonormal DCT-II over the sequence axis (dim 1) of (B, K, ...)."""
    return torch_dct.dct(d.movedim(1, -1), norm="ortho").movedim(-1, 1)


def idct_t(c: torch.Tensor) -> torch.Tensor:
    return torch_dct.idct(c.movedim(1, -1), norm="ortho").movedim(-1, 1)


def lowpass(d: torch.Tensor, k_lat: int, k_lon: int) -> torch.Tensor:
    """Projection of (..., H, W) onto the k_lat x k_lon lowest spatial DCT modes."""
    c = torch_dct.dct_2d(d, norm="ortho")
    m = torch.zeros(c.shape[-2:], dtype=c.dtype, device=c.device)
    m[:k_lat, :k_lon] = 1
    return torch_dct.idct_2d(c * m, norm="ortho")


class SeqDCT(nn.Module):
    """Lossless 3-D DCT (sequence, lat, lon) of an increment sequence d (B, K, C, H, W), standardized
    per coefficient: a = (DCT(d) - mean) / std, both (K, C, H, W). ``spatial`` = the 2-D DCT only
    (per-lead coefficients, used for the stage-1 perturbations)."""

    def __init__(self, K: int, C: int, H: int, W: int, spatial: bool = False):
        super().__init__()
        self.spatial = spatial
        self.register_buffer("mean", torch.zeros(K, C, H, W))
        self.register_buffer("std", torch.ones(K, C, H, W))

    def coeffs(self, d: torch.Tensor) -> torch.Tensor:
        c = torch_dct.dct_2d(d, norm="ortho")
        return c if self.spatial else dct_t(c)

    def inverse_coeffs(self, c: torch.Tensor) -> torch.Tensor:
        return torch_dct.idct_2d(c if self.spatial else idct_t(c), norm="ortho")

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return (self.coeffs(d) - self.mean) / self.std

    def inverse(self, a: torch.Tensor) -> torch.Tensor:
        return self.inverse_coeffs(a * self.std + self.mean)

    @torch.no_grad()
    def fit(self, batches) -> None:
        n, s1, s2 = 0, 0.0, 0.0
        for d in batches:
            c = self.coeffs(d).double()
            n, s1, s2 = n + c.shape[0], s1 + c.sum(0), s2 + (c ** 2).sum(0)
        mean = s1 / n
        self.mean.copy_(mean.float())
        self.std.copy_(((s2 - n * mean ** 2) / max(n - 1, 1)).clamp_min(0).sqrt().float().clamp_min(1e-6))

    def noise(self, shape, generator=None) -> torch.Tensor:
        """Zero-mean Gaussian increments with the fitted per-coefficient std (B, K, C, H, W)."""
        eps = torch.randn(shape, generator=generator, device=self.std.device)
        return self.inverse_coeffs(eps * self.std)


@dataclass
class PINNConfig:
    K: int = 24  # 6 h increments in the sequence
    k_lat: int = 8  # low spatial modes the PINN may change
    k_lon: int = 16
    hidden: int = 64
    blocks: tuple = (2, 2)


class PINNProjector(nn.Module):
    """P: (x(t), d~) -> d~ + lowpass(N(x(t), d~, static, t)), the stage-1 PINN.

    x0 (B, 66, H, W) normalized state at the IC, d (B, K, 66, H, W) increments (generated or
    perturbed), hour0 (B,) hours since Jan 1. tisr is a known forcing and is never changed."""

    def __init__(self, cfg: PINNConfig, static: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("static", static)
        keep = torch.ones(N_DYN)
        keep[TISR_DYN] = 0
        self.register_buffer("keep", keep.view(1, 1, N_DYN, 1, 1))
        self.net = Net((cfg.K + 1) * N_DYN + static.shape[0] + 4, cfg.K * N_DYN,
                       _NetCfg(cfg.hidden, cfg.blocks))

    def correction(self, x0, d, hour0):
        B, K, C, H, W = d.shape
        h = torch.cat([x0, d.reshape(B, K * C, H, W), self.static.expand(B, -1, -1, -1),
                       time_features(hour0.float(), H, W)], 1)
        c = self.net(h).view(B, K, C, H, W) * self.keep
        return lowpass(c, self.cfg.k_lat, self.cfg.k_lon)

    def forward(self, x0, d, hour0):
        return d + self.correction(x0, d, hour0)


@dataclass
class _NetCfg:  # the fields of core.CoreConfig that core.Net reads
    hidden: int
    blocks: tuple


def frames(x0: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """(B, K + 1, 66, H, W): x(t), x(t) + d_1, ..., x(t) + d_K."""
    return torch.cat([x0[:, None], x0[:, None] + d], 1)


def sequence_residual_loss(guide, x0, d, d_obs, orog, pairs: torch.Tensor):
    """Mean primitive-equation loss (guide.loss, scales b_j) of the consecutive pairs ``pairs`` (n,)
    of the sequence x0 + d; collocation masks from the observed sequence x0 + d_obs (sec. 5.3)."""
    f, fo = frames(x0, d), frames(x0, d_obs)
    B = x0.shape[0]
    a, b = f[:, pairs].flatten(0, 1), f[:, pairs + 1].flatten(0, 1)
    ao, bo = fo[:, pairs].flatten(0, 1), fo[:, pairs + 1].flatten(0, 1)
    s = orog.repeat_interleave(len(pairs), 0)
    R, _ = guide.residuals(a, b, s, torch.zeros(B * len(pairs)), use_bridge=False, use_closure=False)
    return guide.loss(R, guide.mask(ao, bo, s)), R


@torch.no_grad()
def residual_rms(guide, x0, d, d_obs, orog, pairs, chunk: int = 32) -> Dict[str, float]:
    """Normalized RMS per equation (guide.per_equation) over the pairs ``pairs`` of every sample,
    ``chunk`` pairs per call (the float64 SI fields of 66 channels are large)."""
    f, fo = frames(x0, d), frames(x0, d_obs)
    a, b = f[:, pairs].flatten(0, 1), f[:, pairs + 1].flatten(0, 1)
    ao, bo = fo[:, pairs].flatten(0, 1), fo[:, pairs + 1].flatten(0, 1)
    s = orog.repeat_interleave(len(pairs), 0)
    acc, n = {}, a.shape[0]
    for i in range(0, n, chunk):
        j = slice(i, min(n, i + chunk))
        R, _ = guide.residuals(a[j], b[j], s[j], torch.zeros(a[j].shape[0]), use_bridge=False, use_closure=False)
        for k, v in guide.per_equation(R, guide.mask(ao[j], bo[j], s[j])).items():
            acc[k] = acc.get(k, 0.0) + v ** 2 * (j.stop - j.start) / n
    return {k: math.sqrt(v) for k, v in acc.items()}


def band_spread(e: torch.Tensor, k_lat: int, k_lon: int) -> Dict[str, float]:
    """Member spread of e (M, B, K, C, H, W) in the low (k_lat x k_lon) and the other DCT modes."""
    c = torch_dct.dct_2d(e - e.mean(0, keepdim=True), norm="ortho")
    m = torch.zeros(c.shape[-2:], dtype=torch.bool)
    m[:k_lat, :k_lon] = True
    low, high = (c ** 2)[..., m], (c ** 2)[..., ~m]
    return {"low": math.sqrt(float(low.mean())), "high": math.sqrt(float(high.mean()))}


# -- stage 2: the sequence flow matching ---------------------------------------------------------


@dataclass
class SeqFMConfig:
    K: int = 24
    basis: str = "dct"  # "dct": flow on standardized 3-D DCT coefficients; "grid": standardized per grid value
    model_channels: int = 64
    channel_mult: tuple = (1, 2, 2)
    num_res_blocks: int = 1


class SeqFM(nn.Module):
    """p(d_1..K | x(t - 12 h), x(t - 6 h), x(t)) as a conditional flow on a = (B(d) - mean) / std.

    basis "dct": B = 3-D DCT (sequence, lat, lon), a per-coefficient Gaussian source = noise with the
    climatological spectrum of each lead; "grid": B = identity, the step 2 CFM's source (the control
    for the owner's DCT claim). The path and target velocity are torchcfm's (sigma = 0, as
    pgcfm/cfm.py); the velocity network is torchcfm's U-Net on the grid (lon periodic): its input is
    the current a mapped back to increments, its output is mapped into the coefficient space, so
    the loss weighs every coefficient alike."""

    def __init__(self, cfg: SeqFMConfig, static: torch.Tensor, H: int = 32, W: int = 64):
        super().__init__()
        from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
        from torchcfm.models.unet.unet import UNetModel
        from .cfm import make_lon_periodic
        self.cfg = cfg
        self.register_buffer("static", static)
        self.basis = SeqDCT(cfg.K, N_DYN, H, W, spatial=False)
        self.grid_basis = cfg.basis == "grid"
        self.fm = ConditionalFlowMatcher(sigma=0.0)
        n_out = cfg.K * N_DYN
        self.unet = make_lon_periodic(UNetModel(
            image_size=max(H, W), in_channels=n_out + 3 * N_DYN + static.shape[0] + 4,
            model_channels=cfg.model_channels, out_channels=n_out, num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=(), dropout=0.0, channel_mult=cfg.channel_mult))

    # basis maps (standardized coefficients <-> increments)
    def to_coef(self, d):
        if self.grid_basis:
            return (d - self.basis.mean) / self.basis.std
        return self.basis(d)

    def to_incr(self, a):
        if self.grid_basis:
            return a * self.basis.std + self.basis.mean
        return self.basis.inverse(a)

    @torch.no_grad()
    def fit_basis(self, batches) -> None:
        if not self.grid_basis:
            return self.basis.fit(batches)
        n, s1, s2 = 0, 0.0, 0.0
        for d in batches:
            d = d.double()
            n, s1, s2 = n + d.shape[0], s1 + d.sum(0), s2 + (d ** 2).sum(0)
        m = s1 / n
        self.basis.mean.copy_(m.float())
        self.basis.std.copy_(((s2 - n * m ** 2) / max(n - 1, 1)).clamp_min(0).sqrt().float().clamp_min(1e-6))

    def velocity(self, tau, a, hist, hour0):
        B, K, C, H, W = a.shape
        h = torch.cat([self.to_incr(a).reshape(B, K * C, H, W), hist.reshape(B, 3 * C, H, W),
                       self.static.expand(B, -1, -1, -1), time_features(hour0.float(), H, W)], 1)
        # The output is the coefficient-space velocity laid out on the grid (orthonormal map, no
        # standardization): mapping a grid-unit dd/dtau through 1 / std instead amplifies the output
        # of the weak coefficients (high modes, q50) by up to 1e3 (loss 2 -> 4e4 after one step).
        o = self.unet(tau, h).view(B, K, C, H, W)
        return o if self.grid_basis else self.basis.coeffs(o)

    def loss(self, hist, d, hour0):
        """hist (B, 3, 66, H, W) = x(t - 12 h), x(t - 6 h), x(t); d (B, K, 66, H, W)."""
        a1 = self.to_coef(d)
        tau, z, u = self.fm.sample_location_and_conditional_flow(torch.randn_like(a1), a1)
        return ((self.velocity(tau, z, hist, hour0) - u) ** 2).mean()

    @torch.no_grad()
    def sample(self, hist, hour0, steps: int = 8, generator=None):
        """One member per row of hist: increments (B, K, 66, H, W) by Heun on the flow."""
        from .cfm import heun
        B = hist.shape[0]
        a = torch.randn((B,) + tuple(self.basis.std.shape), generator=generator, device=hist.device)
        a = heun(lambda t, s: self.velocity(t, s, hist, hour0), a, steps)
        return self.to_incr(a)
