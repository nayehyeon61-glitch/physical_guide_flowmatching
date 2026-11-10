"""The step 16 tendency core (pgcfm/dyncore.py DynCore) put on the KAI-alpha backbone (third_party/kai).

KAI (MetaEarth-Lab, third_party/kai/src/models.py) maps the z-scored daily state x_d (67 channels) to the
full next-day state, trained one step at a time with the relative L2 loss (LpLoss). The tendency core of
step 16 instead learns dx/dt = f(x) and integrates it, with these parts (all optional here, so each one
can be switched on separately and located in the comparison):

    residual   x_{d+1} = x_d + s * net(x_d)       s = std of the daily increment, zero head = persistence
    heun       x_{d+1} = x_d + (k1 + k2) / 2,  k1 = f(x_d, t), k2 = f(x_d + k1, t + 1 day)
    time       season phase (sin, cos of day of year) as 2 extra input channels
    forcing    tisr of day d+1 from the data (a known forcing, never predicted); orog copied
    positive   q >= 0 and tcwv >= 0 with the global mass of every level kept (pgcfm.balance.PositiveMass)
    rollout    training on K-day rollouts with a curriculum K = 1 -> K_max (scripts/run_kai_tendency.py)

The backbone is KAI's own InceptionNeXt, unchanged; only its input channel count changes with ``time``
and the last 1x1 conv of its head is zero-initialized with ``residual``.
"""

import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from . import contract

_KAI_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "kai")
if _KAI_ROOT not in sys.path:
    sys.path.insert(0, _KAI_ROOT)
from src.models import KAI  # noqa: E402  (vendored KAI-alpha backbone)
from src.utils import LpLoss  # noqa: E402

TISR = contract.CHANNEL_INDEX["tisr"]
OROG = contract.CHANNEL_INDEX["orog"]


@dataclass
class KAITendConfig:
    depths: tuple = (3, 3, 15, 3)        # KAI.yaml / models.py default
    dims: tuple = (48, 96, 192, 288)
    residual: bool = False
    heun: bool = False
    time: bool = False
    forcing: bool = False
    positive: bool = False

    @classmethod
    def arm(cls, name: str, depths, dims) -> "KAITendConfig":
        """The comparison ladder: kai (original) -> res -> tend1 (single-step) -> tend (rollout; same net)."""
        on = {"kai": {}, "res": dict(residual=True),
              "tend1": dict(residual=True, heun=True, time=True, forcing=True, positive=True),
              "tend": dict(residual=True, heun=True, time=True, forcing=True, positive=True)}[name]
        return cls(depths=tuple(depths), dims=tuple(dims), **on)


def season_features(doy: torch.Tensor, H: int, W: int) -> torch.Tensor:
    s = 2 * math.pi * doy.float() / 365.25
    return torch.stack([torch.sin(s), torch.cos(s)], 1)[:, :, None, None].expand(-1, -1, H, W)


class KAITend(nn.Module):
    """x (B, 67, H, W) z-scored, doy (B,) -> next-day state (B, 67, H, W)."""

    def __init__(self, cfg: KAITendConfig, inc_std: np.ndarray, mean: np.ndarray, std: np.ndarray, lat_deg):
        super().__init__()
        self.cfg = cfg
        n_in = contract.N_CHANNELS + (2 if cfg.time else 0)
        self.kai = KAI(in_chans=n_in, out_chans=contract.N_CHANNELS, depths=list(cfg.depths), dims=list(cfg.dims))
        if cfg.residual:
            last = self.kai.core_model.head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.register_buffer("s", torch.as_tensor(inc_std, dtype=torch.float32).view(1, -1, 1, 1))
        keep = torch.ones(contract.N_CHANNELS)
        keep[OROG] = 0
        if cfg.forcing:
            keep[TISR] = 0
        self.register_buffer("keep", keep.view(1, -1, 1, 1))
        self.pm = None
        if cfg.positive:
            from .balance import PositiveMass
            self.pm = PositiveMass(mean, std, lat_deg)
            self.register_buffer("dyn", torch.tensor(contract.DYNAMIC_CHANNELS))

    def f(self, x: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, season_features(doy, *x.shape[-2:])], 1) if self.cfg.time else x
        return self.kai(inp)

    def forward(self, x: torch.Tensor, doy: torch.Tensor, tisr_next: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.cfg.residual:  # original KAI: the full next state
            y = self.f(x, doy)
        else:
            k1 = self.f(x, doy) * self.s * self.keep
            if self.cfg.heun:
                k2 = self.f(x + k1, doy + 1) * self.s * self.keep
                y = x + 0.5 * (k1 + k2)
            else:
                y = x + k1
        if self.cfg.forcing and tisr_next is not None:
            y = torch.cat([y[:, :TISR], tisr_next[:, None], y[:, TISR + 1:]], 1)
        if self.pm is not None:
            yd = self.pm(y[:, self.dyn])
            y = y.index_copy(1, self.dyn, yd)
        return y

    def rollout(self, x0, doy0, steps, tisr=None):
        """(B, steps, 67, H, W); tisr (B, steps, H, W) = the forcing of day 1..steps (used with cfg.forcing)."""
        x, out = x0, []
        for k in range(steps):
            x = self(x, doy0 + k, None if tisr is None else tisr[:, k])
            out.append(x)
        return torch.stack(out, 1)


def config_dict(cfg: KAITendConfig) -> dict:
    return asdict(cfg)


__all__ = ["KAITend", "KAITendConfig", "LpLoss", "season_features", "config_dict"]
