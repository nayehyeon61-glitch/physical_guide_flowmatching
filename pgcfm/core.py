"""Step 12: a deterministic grid core in the ClimODE setting (docs/step12_core.md).

The step 11 comparison with ClimODE (docs/step11_climode.md) traced the PG-CFM gap to where the
dynamics live, not to the generator: a latent floor above the daily tendency (L1), one snapshot as
input (L5) and one 24 h jump trained on K = 1 (L4). The revised hypothesis puts the mean dynamics in a
deterministic core and leaves flow matching the ensemble around it. This module is that core,
on ClimODE's five variables (z500, t850, t2m, u10, v10) at 6 h resolution, in four modes that add
one change each:

    jump24   x(t + 24 h) = x(t) + D(x(t)): one frame, one 24 h jump, trained on the 24 h target only
             (today's PG-CFM transition, without the latent)
    jump6    x(t + 6 h) = x(t) + D(x(t), x(t) - x(t - 6 h), x(t - 6 h) - x(t - 12 h)): three frames
             (IC - 12 h, IC - 6 h, IC, as ClimODE), chained, trained on a 24 h rollout of 6 h outputs
    ode      dx/dt = f(x, the same two differences, t), integrated in 1 h Euler steps (ClimODE's solver)
             between the 6 h outputs; each output refreshes the differences
    transport  ClimODE's own structure (third_party/climode/model_function.py ``pde``): a velocity per
             variable, du/dt = -(v . grad u + u div v) and dv/dt = g(u, grad u, v, t); v(0) from the same
             three frames by a learned map instead of ClimODE's per-IC optimization (Optim_velocity)

The networks are ClimODE's ``Climate_ResNet_2D`` (vendored, lat reflect / lon circular padding) at a
CPU size, with a zero-initialized 1x1 head so every mode starts as persistence. States are z-scored per
variable; the output of each mode is scaled by the training tendency std over its step. Step 14 swaps the
network for a norm-bounded FNO (``CoreConfig(net="fno", bound=B)``, pgcfm/fno_core.py).
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "climode"))
from model_function import Climate_ResNet_2D  # noqa: E402

N_VAR = 5
HOURS_PER_STEP = 6  # the data's step


@dataclass
class CoreConfig:
    mode: str = "ode"  # jump24 | jump6 | ode | transport
    hidden: int = 48
    blocks: tuple = (2, 2)
    dt_hours: float = 1.0  # Euler step of ode / transport
    net: str = "resnet"  # resnet (step 12) | fno (step 14, pgcfm/fno_core.py FNONet)
    fno_width: int = 32
    fno_modes: int = 12
    fno_layers: int = 4
    bound: float = None  # FNO operator-norm bound B per block (None: unbounded)


def static_features(lat_deg: np.ndarray, lon_deg: np.ndarray, oro: np.ndarray, lsm: np.ndarray) -> torch.Tensor:
    """(8, H, W): ClimODE's position features (cos/sin lat, lon, sin lat x cos/sin lon) + orography + land-sea."""
    la, lo = np.meshgrid(np.deg2rad(lat_deg), np.deg2rad(lon_deg), indexing="ij")
    f = [np.cos(la), np.sin(la), np.cos(lo), np.sin(lo), np.sin(la) * np.cos(lo), np.sin(la) * np.sin(lo),
         (oro - oro.mean()) / oro.std(), lsm]
    return torch.from_numpy(np.stack(f).astype(np.float32))


def time_features(hours: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(B, 4, H, W) day and season phase of hours since Jan 1 00 UTC."""
    d, s = 2 * math.pi * hours / 24.0, 2 * math.pi * hours / (24.0 * 365.25)
    f = torch.stack([torch.sin(d), torch.cos(d), torch.sin(s), torch.cos(s)], 1)
    return f[:, :, None, None].expand(-1, -1, H, W)


def make_net(c_in: int, c_out: int, cfg: CoreConfig) -> nn.Module:
    if cfg.net == "fno":
        from .fno_core import FNONet
        return FNONet(c_in, c_out, cfg.fno_width, (cfg.fno_modes, cfg.fno_modes), cfg.fno_layers, cfg.bound)
    return Net(c_in, c_out, cfg)


class Net(nn.Module):
    """Climate_ResNet_2D trunk + zero-initialized linear head (output 0 = persistence at the start)."""

    def __init__(self, c_in: int, c_out: int, cfg: CoreConfig):
        super().__init__()
        self.trunk = Climate_ResNet_2D(c_in, list(cfg.blocks), [cfg.hidden] * len(cfg.blocks))
        self.head = nn.Conv2d(cfg.hidden, c_out, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.trunk(x))


def _grad(u):
    """Centred differences on the grid (lat reflect is not needed for a tendency; lon is periodic)."""
    gx = (torch.roll(u, -1, -1) - torch.roll(u, 1, -1)) / 2
    gy = torch.cat([u[..., 1:2, :] - u[..., :1, :], (u[..., 2:, :] - u[..., :-2, :]) / 2,
                    u[..., -1:, :] - u[..., -2:-1, :]], -2)
    return gx, gy


class Core(nn.Module):
    """Deterministic core. ``rollout(frames, hour0, steps)`` returns the 6 h states after the IC.

    frames: (B, 3, 5, H, W) normalized states at IC - 12 h, IC - 6 h, IC (jump24 uses the last one).
    hour0:  (B,) hours since Jan 1 00 UTC of the IC.
    """

    def __init__(self, cfg: CoreConfig, static: torch.Tensor, tend_std: Dict[int, torch.Tensor]):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("static", static)
        # per-variable tendency std over 6 h and 24 h, normalized units, shape (5,)
        self.register_buffer("s6", tend_std[6].view(1, N_VAR, 1, 1))
        self.register_buffer("s24", tend_std[24].view(1, N_VAR, 1, 1))
        n_static, n_time = static.shape[0], 4
        if cfg.mode == "jump24":
            self.net = make_net(N_VAR + n_static + n_time, N_VAR, cfg)
        elif cfg.mode in ("jump6", "ode"):
            self.net = make_net(3 * N_VAR + n_static + n_time, N_VAR, cfg)
        elif cfg.mode == "transport":
            # dv/dt from (u, grad u, v, static, time), as ClimODE's vel_f; v(0) from the three frames
            self.net = Net(N_VAR * 5 + n_static + n_time, 2 * N_VAR, cfg)
            self.v0 = Net(3 * N_VAR + n_static, 2 * N_VAR, cfg)
            # the zero head would freeze transport at v = 0 forever: start v(0) from small noise
            nn.init.normal_(self.v0.head.weight, std=1e-3)
        else:
            raise ValueError(cfg.mode)

    def _inputs(self, parts: List[torch.Tensor], hours: torch.Tensor) -> torch.Tensor:
        B, _, H, W = parts[0].shape
        return torch.cat(parts + [self.static.expand(B, -1, -1, -1), time_features(hours, H, W)], 1)

    def rollout(self, frames: torch.Tensor, hour0: torch.Tensor, steps: int) -> torch.Tensor:
        """(B, steps, 5, H, W): states every 6 h after the IC (jump24: every 24 h, steps counted in days)."""
        m = self.cfg.mode
        hist = [frames[:, 0], frames[:, 1], frames[:, 2]]
        out, hours = [], hour0.float()
        if m == "jump24":
            x = hist[-1]
            for _ in range(steps):
                x = x + self.s24 * self.net(self._inputs([x], hours))
                hours = hours + 24
                out.append(x)
            return torch.stack(out, 1)
        if m == "transport":
            d1, d2 = hist[2] - hist[1], hist[1] - hist[0]
            v = self.v0(torch.cat([hist[2], d1, d2, self.static.expand(frames.shape[0], -1, -1, -1)], 1))
        n_sub = int(round(HOURS_PER_STEP / self.cfg.dt_hours))
        dt = HOURS_PER_STEP / n_sub
        for _ in range(steps):
            x = hist[-1]
            if m == "jump6":
                x = x + self.s6 * self.net(self._inputs([x, (hist[2] - hist[1]) / self.s6,
                                                         (hist[1] - hist[0]) / self.s6], hours))
                hours = hours + HOURS_PER_STEP
            elif m == "ode":
                c1, c2 = (hist[2] - hist[1]) / self.s6, (hist[1] - hist[0]) / self.s6
                for _ in range(n_sub):  # explicit Euler, 1 h steps
                    x = x + dt / HOURS_PER_STEP * self.s6 * self.net(self._inputs([x, c1, c2], hours))
                    hours = hours + dt
            else:  # transport
                for _ in range(n_sub):
                    vx, vy = v[:, :N_VAR], v[:, N_VAR:]
                    gx, gy = _grad(x)
                    vgx, _ = _grad(vx)
                    _, vgy = _grad(vy)
                    dx = -(vx * gx + vy * gy + x * (vgx + vgy))
                    dv = self.net(self._inputs([x / 3, gx, gy, vx, vy], hours))
                    x = x + dt * dx
                    v = v + dt * 1e-2 * dv
                    hours = hours + dt
            out.append(x)
            hist = [hist[1], hist[2], x]
        return torch.stack(out, 1)


def area_weights(lat_deg: np.ndarray) -> torch.Tensor:
    w = np.cos(np.deg2rad(lat_deg))
    return torch.from_numpy((w / w.mean()).astype(np.float32)).view(1, 1, 1, -1, 1)


def rollout_loss(core: Core, frames, hour0, targets, w) -> torch.Tensor:
    """Latitude-weighted MSE over every output of the rollout (targets (B, K, 5, H, W))."""
    pred = core.rollout(frames, hour0, targets.shape[1])
    return ((pred - targets) ** 2 * w).mean()
