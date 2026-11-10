"""Where to put the tendency core in KAI-alpha (docs/kai_tendency.md).

Two models, trained separately:

* the tendency core: pgcfm.dyncore.DynCore of step 16, unchanged (ClimODE's ResNet as dx/dt = f(x), one Heun
  step, tisr forcing, PositiveMass, rollout training), here with one step = one day on KAI's daily means;
* KAI-alpha: third_party/kai (InceptionNeXt, full next-day state, single-step training, relative L2 loss).

The core is trained first and frozen. Its one-day forecast c(x) and tendency tau = (c(x) - x) / s
(s = std of the daily increment per channel) then enter KAI at ONE position:

    inject  in        tau as 67 extra input channels of the stem
            s1 .. s4  tau added to the features after stage 1 .. 4 through a zero-initialized 1x1 conv
                      (s4 = just before KAI's head)
    form    full      KAI's own output: the full next state                       (original KAI)
            res       x + s * net(.)        zero head = persistence               (control, no core)
            core      c(x) + s * net(.)     zero head = the core's forecast       (the core at the output)

A fourth position needs no training: a lead-wise blend of the two finished forecasts
(scripts/run_kai_tendency.py, arm ``blend``).
"""

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
DYN = contract.DYNAMIC_CHANNELS
TISR_DYN = DYN.index(TISR)

# arm -> (form, inject)
ARMS = {"kai": ("full", "none"), "in": ("full", "in"), "s1": ("full", "s1"), "s2": ("full", "s2"),
        "s3": ("full", "s3"), "s4": ("full", "s4"), "res": ("res", "none"), "inres": ("res", "in"),
        "out": ("core", "none"), "inout": ("core", "in")}


def needs_core(arm: str) -> bool:
    form, inject = ARMS[arm]
    return form == "core" or inject != "none"


# ------------------------------------------------------------------------------------- the tendency core
def make_core(static: torch.Tensor, inc_std: np.ndarray, mean: np.ndarray, std: np.ndarray, lat_deg,
              hidden: int = 48, blocks=(2, 2)):
    """Step 16's DynCore on daily data: ``t6`` is the std of the DAILY increment, one solver step = one day."""
    from .dyncore import DynCore, DynCoreConfig
    cfg = DynCoreConfig(hidden=hidden, blocks=tuple(blocks), dt_hours=6.0, solver="heun", positive=True)
    return DynCore(cfg, static, torch.as_tensor(inc_std[DYN], dtype=torch.float32), mean[DYN], std[DYN], lat_deg)


def core_step(core, xd: torch.Tensor, doy: torch.Tensor, tisr_next: Optional[torch.Tensor]) -> torch.Tensor:
    """One day of the core on the 66 dynamic channels (season phase from the day of year)."""
    return core.step(xd, doy.float() * 24.0, tisr_next)


def core_rollout(core, xd, doy, steps, tisr=None):
    x, out = xd, []
    for k in range(steps):
        x = core_step(core, x, doy + k, None if tisr is None else tisr[:, k])
        out.append(x)
    return torch.stack(out, 1)


# ------------------------------------------------------------------------------- KAI with the core inside
@dataclass
class KAIPosConfig:
    depths: tuple = (3, 3, 15, 3)        # KAI.yaml / models.py default
    dims: tuple = (48, 96, 192, 288)
    form: str = "full"
    inject: str = "none"

    @classmethod
    def arm(cls, name: str, depths, dims) -> "KAIPosConfig":
        form, inject = ARMS[name]
        return cls(depths=tuple(depths), dims=tuple(dims), form=form, inject=inject)


class KAIPos(nn.Module):
    """x (B, 67, H, W) z-scored, doy (B,), tisr_next (B, H, W) -> next-day state (B, 67, H, W).

    ``core`` is the frozen tendency core (not a submodule: it is neither trained nor saved with KAI)."""

    def __init__(self, cfg: KAIPosConfig, inc_std: np.ndarray, core=None):
        super().__init__()
        self.cfg = cfg
        n = contract.N_CHANNELS
        self.kai = KAI(in_chans=n * (2 if cfg.inject == "in" else 1), out_chans=n,
                       depths=list(cfg.depths), dims=list(cfg.dims))
        if cfg.form != "full":
            last = self.kai.core_model.head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        if cfg.inject.startswith("s"):
            self.stage = int(cfg.inject[1]) - 1
            self.proj = nn.Conv2d(n, cfg.dims[self.stage], 1)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.register_buffer("s", torch.as_tensor(inc_std, dtype=torch.float32).view(1, -1, 1, 1))
        keep = torch.ones(n)
        keep[OROG] = 0
        self.register_buffer("keep", keep.view(1, -1, 1, 1))
        self.register_buffer("dyn", torch.tensor(DYN))
        self._core = [core]
        self.ablate = False  # diagnostic: zero the injected tau (the output anchor of form "core" is kept)
        if (cfg.form == "core" or cfg.inject != "none") and core is None:
            raise ValueError("this arm needs the frozen tendency core")

    def core_next(self, x, doy, tisr_next):
        with torch.no_grad():
            yd = core_step(self._core[0], x[:, self.dyn], doy, tisr_next)
        return x.index_copy(1, self.dyn, yd)  # orog copied

    def forward(self, x, doy, tisr_next=None):
        cfg, c, tau = self.cfg, None, None
        if cfg.form == "core" or cfg.inject != "none":
            c = self.core_next(x, doy, tisr_next)
            tau = (c - x) / self.s
            if self.ablate:
                tau = torch.zeros_like(tau)
        m = self.kai.core_model
        h = torch.cat([x, tau], 1) if cfg.inject == "in" else x
        for i in range(4):
            h = m.stages[i](m.downsample_layers[i](h))
            if cfg.inject.startswith("s") and i == self.stage:
                h = h + self.proj(tau)
        o = m.head(h)
        if cfg.form == "full":
            return o
        return (x if cfg.form == "res" else c) + o * self.s * self.keep

    def rollout(self, x0, doy0, steps, tisr=None):
        """(B, steps, 67, H, W); tisr (B, steps, H, W) = the core's forcing on day 1..steps."""
        x, out = x0, []
        for k in range(steps):
            x = self(x, doy0 + k, None if tisr is None else tisr[:, k])
            out.append(x)
        return torch.stack(out, 1)


def config_dict(cfg) -> dict:
    return asdict(cfg)


__all__ = ["ARMS", "KAIPos", "KAIPosConfig", "LpLoss", "make_core", "core_step", "core_rollout", "needs_core",
           "config_dict", "TISR", "OROG", "DYN", "TISR_DYN"]
