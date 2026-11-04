"""Step 19: super-resolution of the step 13 output (5.625 deg -> 2.8125 deg), a reduced form of
Chiang et al. 2024 (arXiv 2403.17847, deterministic CNN downscaling of precipitation with topography):

  cascaded conv ("bias correction")  ->  residual attention blocks (CBAM, channel + spatial attention)
  with skip connections fused by a 1x1 conv  ->  pixel shuffle (ESPCN, one-step x2)  ->  fine-topography
  head (fine orography / land-sea / latitude concatenated after the upscaling)  ->  + band-limited
  interpolation of the input (global residual: the net learns what interpolation misses).

Padding: latitude reflect, longitude circular (ClimODE's ``boundarypad``). Inputs and outputs are per-variable
z-scores (the 5 ClimODE variables); the model is applied to each ensemble member separately.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def bpad(x: torch.Tensor, p: int) -> torch.Tensor:
    """Reflect in latitude, circular in longitude, ``p`` cells each side."""
    return F.pad(F.pad(x, (0, 0, p, p), "reflect"), (p, p, 0, 0), "circular")


class Conv(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3):
        super().__init__()
        self.conv, self.p = nn.Conv2d(c_in, c_out, k, padding=0), k // 2

    def forward(self, x):
        return self.conv(bpad(x, self.p) if self.p else x)


class CBAM(nn.Module):
    """Convolutional block attention: channel attention (avg + max pooled MLP), then spatial attention
    (7x7 conv over the channel avg and max)."""

    def __init__(self, c: int, r: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, max(1, c // r)), nn.ReLU(), nn.Linear(max(1, c // r), c))
        self.spatial = Conv(2, 1, 7)

    def forward(self, x):
        a = torch.sigmoid(self.mlp(x.mean((-1, -2))) + self.mlp(x.amax((-1, -2))))
        x = x * a[:, :, None, None]
        s = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * s


class RAB(nn.Module):
    """Residual attention block: conv-ReLU-conv, CBAM, residual."""

    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(Conv(c, c), nn.ReLU(), Conv(c, c))
        self.att = CBAM(c)

    def forward(self, x):
        return x + self.att(self.body(x))


class SRNet(nn.Module):
    def __init__(self, c_in: int = 5, c_out: int = 5, c_static_lr: int = 3, c_static_hr: int = 3,
                 hidden: int = 32, n_blocks: int = 4, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.bias = nn.Sequential(Conv(c_in + c_static_lr, hidden), nn.ReLU(), Conv(hidden, hidden), nn.ReLU())
        self.blocks = nn.ModuleList([RAB(hidden) for _ in range(n_blocks)])
        self.fuse = nn.Conv2d(hidden * n_blocks, hidden, 1)
        self.up = nn.Sequential(Conv(hidden, hidden * scale * scale), nn.PixelShuffle(scale))
        self.head = nn.Sequential(Conv(hidden + c_static_hr, hidden), nn.ReLU(), Conv(hidden, c_out))
        nn.init.zeros_(self.head[-1].conv.weight)
        nn.init.zeros_(self.head[-1].conv.bias)

    def forward(self, x_lr: torch.Tensor, static_lr: torch.Tensor, static_hr: torch.Tensor,
                x_interp: torch.Tensor) -> torch.Tensor:
        """x_lr (B, 5, H, W), static_lr (1 or B, 3, H, W), static_hr (1 or B, 3, 2H, 2W),
        x_interp (B, 5, 2H, 2W) -> (B, 5, 2H, 2W)."""
        B = x_lr.shape[0]
        h = self.bias(torch.cat([x_lr, static_lr.expand(B, -1, -1, -1)], 1))
        outs = []
        for b in self.blocks:
            h = b(h)
            outs.append(h)
        h = self.up(self.fuse(torch.cat(outs, 1)))
        return x_interp + self.head(torch.cat([h, static_hr.expand(B, -1, -1, -1)], 1))


def interpolate(x: torch.Tensor, lat_c, lat_f, n_lon_f: int) -> torch.Tensor:
    """Band-limited (FFT zero padding) in longitude, cubic spline in latitude (the step 18 control);
    x (N, C, Hc, Wc) float -> (N, C, Hf, Wf) same dtype. Linear, so it commutes with z-scores."""
    import numpy as np
    from scipy.interpolate import CubicSpline
    n, C, Hc, Wc = x.shape
    X = torch.fft.rfft(x.double(), dim=-1) * (n_lon_f / Wc)
    X[..., Wc // 2] *= 0.5
    Xf = torch.zeros(n, C, Hc, n_lon_f // 2 + 1, dtype=X.dtype, device=x.device)
    Xf[..., : Wc // 2 + 1] = X
    xf = torch.fft.irfft(Xf, n=n_lon_f, dim=-1).cpu().numpy()
    yf = CubicSpline(np.asarray(lat_c), xf, axis=-2, extrapolate=True)(np.asarray(lat_f))
    return torch.from_numpy(np.ascontiguousarray(yf)).to(x.dtype).to(x.device)


class Interpolator:
    """The interpolation as a fixed linear map (matrix in latitude x FFT in longitude), built once; much
    faster than the spline per call and exact for the same grids."""

    def __init__(self, lat_c, lat_f, n_lon_c: int, n_lon_f: int, device=None):
        import numpy as np
        from scipy.interpolate import CubicSpline
        A = CubicSpline(np.asarray(lat_c), np.eye(len(lat_c)), axis=0, extrapolate=True)(np.asarray(lat_f))
        self.A = torch.from_numpy(np.ascontiguousarray(A)).to(device)  # (Hf, Hc): y = A x in latitude
        self.n_lon_c, self.n_lon_f = n_lon_c, n_lon_f

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        Wc, Wf = self.n_lon_c, self.n_lon_f
        X = torch.fft.rfft(x.to(torch.float64), dim=-1) * (Wf / Wc)
        X[..., Wc // 2] *= 0.5
        Xf = torch.zeros(*x.shape[:-1], Wf // 2 + 1, dtype=X.dtype, device=x.device)
        Xf[..., : Wc // 2 + 1] = X
        xf = torch.fft.irfft(Xf, n=Wf, dim=-1)
        return torch.einsum("fh,...hw->...fw", self.A.to(xf.dtype), xf).to(x.dtype)
