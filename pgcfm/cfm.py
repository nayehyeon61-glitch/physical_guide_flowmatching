"""Step 2: conditional flow matching alone (no PINN) for p(x_{k+1} | x_k, c_k).

Built on torchcfm (Tong et al., https://github.com/atong01/conditional-flow-matching):
  * probability path and target velocity: ``ConditionalFlowMatcher(sigma=0)``,
    which is exactly the paper's eq. (3) z_tau = (1 - tau) eps + tau y, u = y - eps
    when torchcfm's source x0 is the noise eps and its target x1 is y = x_{k+1};
  * backbone: torchcfm's guided-diffusion ``UNetModel`` (multi-scale CNN with a
    sinusoidal time embedding, paper sec. 4.3).

What is added here is only what the paper asks for on top of that: the KAI
conditioning (x_k, orography, calendar), the area-weighted loss W of eq. (4),
the Heun sampler of sec. 6.2 and the member-wise Markov rollout.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModel

from . import contract

# Position of tisr inside the 66 dynamic channels.
TISR_DYN = contract.DYNAMIC_CHANNELS.index(contract.CHANNEL_INDEX["tisr"])


@dataclass
class CFMConfig:
    tisr_mode: str = "target"  # "target": generated (KAI baseline); "forcing": known condition
    model_channels: int = 32
    channel_mult: tuple = (1, 2, 2)
    num_res_blocks: int = 1
    attention_resolutions: tuple = ()
    dropout: float = 0.0


def calendar_features(doy: torch.Tensor) -> torch.Tensor:
    """(B,) 0-based day-of-year -> (B, 2) annual sin/cos."""
    a = 2 * math.pi * doy.float() / 365.25
    return torch.stack([torch.sin(a), torch.cos(a)], dim=-1)


def area_weights(n_lat: int, device=None) -> torch.Tensor:
    """cos(lat) weights (equiangular with poles unless registered in contract), mean 1. Shape (H, 1)."""
    lat = torch.as_tensor(contract.grid_latitudes(n_lat), dtype=torch.float32, device=device)
    w = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
    return (w / w.mean())[:, None]


def _sphere_pad_hook(module: nn.Conv2d, args):
    """Pad longitude periodically and latitude by replication before the conv."""
    (x,) = args
    ph, pw = module._sphere_pad
    if pw:
        x = F.pad(x, (pw, pw, 0, 0), mode="circular")
    if ph:
        x = F.pad(x, (0, 0, ph, ph), mode="replicate")
    return (x,)


def make_lon_periodic(net: nn.Module) -> nn.Module:
    """Replace the zero padding of every Conv2d by circular-in-longitude padding.

    torchcfm's U-Net zero-pads, which would make 0 and 358.5 deg image edges
    (a dateline seam). Only the padding changes; weights and shapes do not.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d) and any(p > 0 for p in m.padding):
            m._sphere_pad = tuple(m.padding)
            m.padding = (0, 0)
            m.register_forward_pre_hook(_sphere_pad_hook)
    return net


class VelocityNet(nn.Module):
    """v_theta(z_tau, tau | x_k, c_k) on the lat-lon grid.

    Input channels: z_tau (G) + x_k (66) + orography (1) + calendar (2)
    [+ tisr at k+1 when tisr_mode == "forcing"]. Output: G velocity channels,
    G = 66 (tisr generated) or 65 (tisr given).
    """

    def __init__(self, cfg: CFMConfig, n_lat: int, n_lon: int):
        super().__init__()
        self.cfg = cfg
        self.n_gen = contract.N_DYNAMIC - (cfg.tisr_mode == "forcing")
        n_cond = contract.N_DYNAMIC + 1 + 2 + (cfg.tisr_mode == "forcing")
        # The U-Net halves the grid len(channel_mult)-1 times.
        m = 2 ** (len(cfg.channel_mult) - 1)
        self.pad_lat = (-n_lat) % m
        self.pad_lon = (-n_lon) % m
        self.unet = UNetModel(
            image_size=max(n_lat + self.pad_lat, n_lon + self.pad_lon),
            in_channels=self.n_gen + n_cond,
            model_channels=cfg.model_channels,
            out_channels=self.n_gen,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=cfg.attention_resolutions,
            dropout=cfg.dropout,
            channel_mult=cfg.channel_mult,
        )
        make_lon_periodic(self.unet)

    def forward(self, tau: torch.Tensor, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        H, W = z.shape[-2:]
        h = torch.cat([z, cond], dim=1)
        if self.pad_lon:
            h = F.pad(h, (0, self.pad_lon), mode="circular")
        if self.pad_lat:
            h = F.pad(h, (0, 0, 0, self.pad_lat), mode="replicate")
        return self.unet(tau, h)[..., :H, :W]


class CFMTransition(nn.Module):
    """One 24 h transition x_k -> x_{k+1} as a conditional flow."""

    def __init__(self, cfg: CFMConfig, n_lat: int, n_lon: int):
        super().__init__()
        if cfg.tisr_mode not in ("target", "forcing"):
            raise ValueError(cfg.tisr_mode)
        self.cfg = cfg
        self.net = VelocityNet(cfg, n_lat, n_lon)
        self.fm = ConditionalFlowMatcher(sigma=0.0)
        self.register_buffer("w_area", area_weights(n_lat))
        gen = [i for i in range(contract.N_DYNAMIC) if not (cfg.tisr_mode == "forcing" and i == TISR_DYN)]
        self.register_buffer("gen_idx", torch.tensor(gen))

    # -- conditioning -------------------------------------------------------
    def condition(self, x: torch.Tensor, static: torch.Tensor, doy_next: torch.Tensor,
                  tisr_next: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x (B,66,H,W), static (B,1,H,W), doy_next (B,), tisr_next (B,1,H,W)."""
        B, _, H, W = x.shape
        cal = calendar_features(doy_next).to(x)[:, :, None, None].expand(B, 2, H, W)
        parts = [x, static, cal]
        if self.cfg.tisr_mode == "forcing":
            if tisr_next is None:
                raise ValueError("tisr_mode='forcing' needs tisr at k+1")
            parts.append(tisr_next)
        return torch.cat(parts, dim=1)

    # -- training -----------------------------------------------------------
    def loss(self, x: torch.Tensor, y: torch.Tensor, static: torch.Tensor,
             doy_next: torch.Tensor) -> torch.Tensor:
        """Eq. (4): area-weighted velocity MSE, fresh eps and tau per sample."""
        tisr_next = y[:, TISR_DYN:TISR_DYN + 1]
        cond = self.condition(x, static, doy_next, tisr_next)
        y_gen = y[:, self.gen_idx]
        eps = torch.randn_like(y_gen)
        tau, z, u = self.fm.sample_location_and_conditional_flow(eps, y_gen)
        v = self.net(tau, z, cond)
        return ((v - u) ** 2 * self.w_area).mean()

    # -- sampling -----------------------------------------------------------
    def sample(self, x: torch.Tensor, static: torch.Tensor, doy_next: torch.Tensor,
               tisr_next: Optional[torch.Tensor] = None, steps: int = 8,
               generator: Optional[torch.Generator] = None, grad: bool = False,
               eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Heun integration of eq. (2) from tau=0 to 1. Returns x_{k+1} (B,66,H,W).

        NFE = 2 * steps. ``grad=True`` keeps the graph through every Heun step
        so a loss on the endpoint (PG-CFM's L_phys) reaches theta; ``eps``
        overrides the initial noise (B, G, H, W).
        """
        with torch.set_grad_enabled(grad):
            cond = self.condition(x, static, doy_next, tisr_next)
            B, _, H, W = x.shape
            if eps is None:
                eps = torch.randn((B, len(self.gen_idx), H, W), generator=generator,
                                  device=x.device, dtype=x.dtype)
            z = heun(lambda t, s: self.net(t, s, cond), eps, steps)
            return self._assemble(x, z, tisr_next)

    def _assemble(self, x: torch.Tensor, z: torch.Tensor, tisr_next: Optional[torch.Tensor]) -> torch.Tensor:
        out = z.new_empty(x.shape)
        out[:, self.gen_idx] = z
        if self.cfg.tisr_mode == "forcing":
            out[:, TISR_DYN:TISR_DYN + 1] = tisr_next
        return out


def heun(f, z: torch.Tensor, steps: int) -> torch.Tensor:
    """Explicit trapezoidal (Heun) rule for dz/dtau = f(tau, z) on [0, 1]."""
    h = 1.0 / steps
    B = z.shape[0]
    for i in range(steps):
        t0 = torch.full((B,), i * h, device=z.device, dtype=z.dtype)
        k1 = f(t0, z)
        k2 = f(t0 + h, z + h * k1)
        z = z + 0.5 * h * (k1 + k2)
    return z


@torch.no_grad()
def rollout(model: CFMTransition, x0: torch.Tensor, static: torch.Tensor, doy: torch.Tensor,
            n_members: int, tisr: Optional[torch.Tensor] = None, steps: int = 8,
            seed: int = 0) -> torch.Tensor:
    """Member-wise Markov rollout (paper sec. 6.2).

    x0 (B,66,H,W), static (B,1,H,W), doy (B,K+1), tisr (B,K,1,H,W) for leads 1..K
    (only used when tisr_mode == "forcing"). Each member feeds its own endpoint
    into the next transition and draws fresh noise at every transition; the
    ensemble mean is never fed back.

    Returns (B, M, K, 66, H, W).
    """
    B, C, H, W = x0.shape
    K = doy.shape[1] - 1
    g = torch.Generator(device=x0.device).manual_seed(seed)
    x = x0[:, None].expand(B, n_members, C, H, W).reshape(B * n_members, C, H, W)
    s = static[:, None].expand(B, n_members, 1, H, W).reshape(B * n_members, 1, H, W)
    out = []
    for k in range(K):
        d = doy[:, k + 1].repeat_interleave(n_members)
        f = None
        if tisr is not None:
            f = tisr[:, k][:, None].expand(B, n_members, 1, H, W).reshape(B * n_members, 1, H, W)
        x = model.sample(x, s, d, f, steps=steps, generator=g)
        out.append(x.view(B, n_members, C, H, W))
    return torch.stack(out, dim=2)
