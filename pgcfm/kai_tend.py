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
            gate      k(x) + g * (c(x) - k(x))   k = KAI's full next state, g = one weight per channel,
                      zero g = the original KAI: KAI keeps its own form and its trained weights (--init kai),
                      the core is mixed in at the output every step (the trained version of the blend)

A further position needs no training: a lead-wise blend of the two finished forecasts
(scripts/run_kai_tendency.py, ``blend``; ``loop`` = the same weights applied every step and fed back).
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
        "out": ("core", "none"), "inout": ("core", "in"), "gate": ("gate", "none"), "ingate": ("gate", "in"),
        # control of ``out``: the frozen first stage is not the tendency core but a trained KAI (arm res)
        "stack": ("core", "none")}


def needs_core(arm: str) -> bool:
    form, inject = ARMS[arm]
    return form in ("core", "gate") or inject != "none"


# ------------------------------------------------------------------------------------- the tendency core
def stateless_norm(module: nn.Module) -> nn.Module:
    """Replace every BatchNorm2d by GroupNorm(1, C) and switch Dropout off, in place.

    ClimODE's ResidualBlock (the core's trunk) has BatchNorm + Dropout(0.1). Trained with batch statistics
    and run with running statistics, the core's tendency gets a systematic offset that a rollout integrates:
    on 2017 the global mean of z500 drifts by -0.14 sigma in 10 days (z1000 -0.47, q700 +0.29) with running
    statistics and by +0.004 with batch statistics, same weights (docs/kai_tendency.md). GroupNorm has no
    batch statistics, so the trained and the deployed function are the same."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(1, child.num_features))
        elif isinstance(child, nn.Dropout):
            setattr(module, name, nn.Identity())
        else:
            stateless_norm(child)
    return module


def make_core(static: torch.Tensor, inc_std: np.ndarray, mean: np.ndarray, std: np.ndarray, lat_deg,
              hidden: int = 48, blocks=(2, 2), norm: str = "bn", substeps: int = 1):
    """Step 16's DynCore on daily data: ``t6`` is the std of the DAILY increment, one solver step = one day.

    ``norm`` "bn" = ClimODE's block as it is; "gn" = stateless_norm (the drift fix). ``substeps`` Heun steps
    per day (DynCore's dt_hours, in units of its nominal 6 h step)."""
    from .dyncore import DynCore, DynCoreConfig
    cfg = DynCoreConfig(hidden=hidden, blocks=tuple(blocks), dt_hours=6.0 / substeps, solver="heun", positive=True)
    core = DynCore(cfg, static, torch.as_tensor(inc_std[DYN], dtype=torch.float32), mean[DYN], std[DYN], lat_deg)
    if norm == "gn":
        stateless_norm(core.net)
    elif norm != "bn":
        raise ValueError(norm)
    return core


def core_step(core, xd: torch.Tensor, doy: torch.Tensor, tisr_next: Optional[torch.Tensor]) -> torch.Tensor:
    """One day of the core on the 66 dynamic channels (season phase from the day of year)."""
    return core.step(xd, doy.float() * 24.0, tisr_next)


def core_rollout(core, xd, doy, steps, tisr=None):
    x, out = xd, []
    for k in range(steps):
        x = core_step(core, x, doy + k, None if tisr is None else tisr[:, k])
        out.append(x)
    return torch.stack(out, 1)


class CoreBase:
    """The frozen tendency core as a map of the full 67-channel state: (x, doy, tisr_next) -> next-day state."""

    def __init__(self, core):
        self.core, self.dyn = core, torch.tensor(DYN)

    def __call__(self, x, doy, tisr_next=None):
        return x.index_copy(1, self.dyn, core_step(self.core, x[:, self.dyn], doy, tisr_next))  # orog copied


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

    ``core`` is the frozen first stage, a callable (x, doy, tisr_next) -> next state: CoreBase(tendency core),
    or a trained KAI for the control arm ``stack`` (not a submodule: neither trained nor saved with KAI)."""

    def __init__(self, cfg: KAIPosConfig, inc_std: np.ndarray, core=None):
        super().__init__()
        self.cfg = cfg
        n = contract.N_CHANNELS
        self.kai = KAI(in_chans=n * (2 if cfg.inject == "in" else 1), out_chans=n,
                       depths=list(cfg.depths), dims=list(cfg.dims))
        if cfg.form in ("res", "core"):
            last = self.kai.core_model.head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        if cfg.form == "gate":
            self.gate = nn.Parameter(torch.zeros(1, n, 1, 1))
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
        self.last_core = None
        self.ablate = False  # diagnostic: zero the injected tau (the output anchor of form "core" is kept)
        # rollout training: the gradient of a later day's loss has to pass through the frozen core's input
        # (c(x) = x + ...: without it the main path of forms "core" / "gate" is cut). Off = single-step training.
        self.core_grad = False
        self.uses_core = cfg.form in ("core", "gate") or cfg.inject != "none"
        if self.uses_core and core is None:
            raise ValueError("this arm needs the frozen tendency core")

    def core_next(self, x, doy, tisr_next):
        with torch.set_grad_enabled(self.core_grad and torch.is_grad_enabled()):
            return self._core[0](x, doy, tisr_next)

    def forward(self, x, doy, tisr_next=None, core_state=None):
        """``core_state``: evaluate the core on this state instead of x (the open-loop diagnostic of ``rollout``)."""
        cfg, c, tau = self.cfg, None, None
        if self.uses_core:
            xc = x if core_state is None else core_state
            c = self.core_next(xc, doy, tisr_next)
            tau = (c - xc) / self.s
            if self.ablate:
                tau = torch.zeros_like(tau)
        m = self.kai.core_model
        h = torch.cat([x, tau], 1) if cfg.inject == "in" else x
        for i in range(4):
            h = m.stages[i](m.downsample_layers[i](h))
            if cfg.inject.startswith("s") and i == self.stage:
                h = h + self.proj(tau)
        o = m.head(h)
        self.last_core = c
        if cfg.form == "full":
            return o
        if cfg.form == "gate":
            return o + self.gate * self.keep * (c - o)
        return (x if cfg.form == "res" else c) + o * self.s * self.keep

    def rollout(self, x0, doy0, steps, tisr=None, open_loop=False):
        """(B, steps, 67, H, W); tisr (B, steps, H, W) = the core's forcing on day 1..steps.

        ``open_loop`` (diagnostic, inject arms): the core runs on its OWN trajectory from x0 and only its tau
        enters KAI, so KAI's output never feeds the core (closed loop = the trained coupling, the default)."""
        x, xc, out = x0, x0, []
        for k in range(steps):
            x = self(x, doy0 + k, None if tisr is None else tisr[:, k], xc if open_loop else None)
            if open_loop:
                xc = self.last_core
            out.append(x)
        return torch.stack(out, 1)


def config_dict(cfg) -> dict:
    return asdict(cfg)


__all__ = ["ARMS", "CoreBase", "KAIPos", "KAIPosConfig", "LpLoss", "make_core", "stateless_norm", "core_step", "core_rollout", "needs_core",
           "config_dict", "TISR", "OROG", "DYN", "TISR_DYN"]
