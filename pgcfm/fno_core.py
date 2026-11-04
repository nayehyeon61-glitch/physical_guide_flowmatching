"""Step 14 (the owner's H-dyn, 2026-10-26): a norm-bounded FNO as the tendency network D of the step 12 core.

    sphere   the latitude axis is doubled before the FFT (the field, then the field flipped over the poles
             and shifted by 180 degrees in longitude), so the spectral layers see a field that is periodic
             across the poles instead of the planar wrap-around from pole to pole; the output keeps the
             first H rows
    D        lift (1 x 1) -> n blocks v -> GELU(K v + W v) -> head (1 x 1, zero init: the core starts as
             persistence like step 12's Net)
    bound    ``project_(B)`` after every optimizer step: lift and head <= 1, and per block
             max_k ||R(k)||_2 <= B / 2, ||W||_2 <= B / 2 (pgcfm/ens_relation.py FNOBlock), so with the
             domain-normalized L2 norm
                 Lip(D) <= sqrt(2) (1.13 B)^n          (1.13 = max GELU', sqrt(2) = cropping the doubled field)
             ``lipschitz_bound()`` returns this number for the current weights (it uses the actual norms,
             so it also holds for the unbounded network)
"""

import math

import torch
import torch.nn as nn

from .ens_relation import FNOBlock

GELU_LIP = 1.1290  # max_x d/dx GELU(x)


def double_sphere(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C, 2H, W): the field, then its continuation over the poles."""
    over = torch.flip(torch.roll(x, x.shape[-1] // 2, dims=-1), dims=[-2])
    return torch.cat([x, over], dim=-2)


def _spec_norm(w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(w[:, :, 0, 0], ord=2)


class FNONet(nn.Module):
    """Drop-in for pgcfm/core.py Net: (B, c_in, H, W) -> (B, c_out, H, W)."""

    def __init__(self, c_in: int, c_out: int, width: int = 32, modes=(12, 12), n_layers: int = 4,
                 bound: float = None):
        super().__init__()
        self.lift = nn.Conv2d(c_in, width, 1)
        self.blocks = nn.ModuleList(FNOBlock(width, *modes) for _ in range(n_layers))
        self.head = nn.Conv2d(width, c_out, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.bound = bound
        self.clip_log = []  # per projection: fraction of the norms that the projection actually reduced
        if bound is not None:
            self.project_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = x.shape[-2]
        v = self.lift(double_sphere(x))
        for blk in self.blocks:
            v = blk(v)
        return self.head(v)[..., :H, :]

    @torch.no_grad()
    def project_(self):
        B, hits, total = self.bound, 0, 0
        for conv, b in ((self.lift, 1.0), (self.head, 1.0)):
            s = _spec_norm(conv.weight)
            hits, total = hits + int(s > b * (1 + 1e-4)), total + 1
            conv.weight.mul_(torch.clamp(b / (s + 1e-12), max=1.0))
        for blk in self.blocks:
            for w in (blk.spec.w_pos, blk.spec.w_neg):
                s = torch.linalg.matrix_norm(w.permute(2, 3, 0, 1), ord=2)
                hits, total = hits + int((s > (B / 2) * (1 + 1e-4)).sum()), total + s.numel()
            s = _spec_norm(blk.w.weight)
            hits, total = hits + int(s > (B / 2) * (1 + 1e-4)), total + 1
            blk.project_(B)
        self.clip_log.append(hits / total)

    @torch.no_grad()
    def lipschitz_bound(self) -> float:
        """sqrt(2) ||lift|| prod_blocks GELU' (max_k ||R(k)|| + ||W||) ||head|| (the last block's GELU included)."""
        L = math.sqrt(2) * float(_spec_norm(self.lift.weight)) * float(_spec_norm(self.head.weight))
        for blk in self.blocks:
            L *= GELU_LIP * float(blk.l2_operator_norm())
        return L
