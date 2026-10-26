"""Step 13, calibration with a flow-dependent spread field (owner's choice 2026-10-26, docs/step13_sequence.md 4.3).

The FM spread does not vary from case to case beyond sampling noise (scripts/diag_step13_flowdep.py), and
the band Gamma of pgcfm/ens_guide.py is a constant per (lead, var, band). Here a network predicts the size
of the main-stream error for each case and grid point, and that factor modulates the calibrated members:

    sigma(x)  = sigma_clim(lead, var, cell) * exp(g(features))
    y_m       = mu* + f ⊙ (gamma_m - mu*),   f = sigma(x) / rho,   rho^2 = E_cal[sigma(x)^2] per (lead, var, cell)

    gamma_m   the closed-form band-Gamma member (apply_gamma), so the climatological spread per band stays
              as fitted and f only moves it between cases (E_cal f^2 = 1 on the calibration year)
    g         ClimODE's uncertainty network (third_party/climode/model_function.py: Climate_ResNet_2D, the
              ``noise_net`` of Climate_encoder_free_uncertain, with its periodic-lon / reflect-lat padding)
              plus a zero-initialized 1x1 head, so training starts at sigma_clim
    features  x(t), mu*, mu* - x(t) (5 each, normalized), lead / 144 h, time of year and day of the
              valid time (pgcfm.core.time_features), ClimODE's position + orography + land-sea features
    loss      ClimODE's Gaussian NLL (third_party/climode/utils.py: nll) of the truth around mu*
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

from pgcfm.core import time_features

CLIMODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "climode")
if CLIMODE not in sys.path:
    sys.path.insert(0, CLIMODE)
from model_function import Climate_ResNet_2D  # noqa: E402
from utils import nll  # noqa: E402


def flow_features(x0: torch.Tensor, mu: torch.Tensor, lead_h: torch.Tensor, ic_hour: torch.Tensor,
                  static: torch.Tensor) -> torch.Tensor:
    """x0, mu (N, 5, H, W) normalized; lead_h, ic_hour (N,) hours; static (S, H, W) -> (N, 20 + S, H, W)."""
    N, _, H, W = mu.shape
    lead = (lead_h.float() / 144.0).view(N, 1, 1, 1).expand(N, 1, H, W)
    return torch.cat([x0, mu, mu - x0, lead, time_features(ic_hour.float() + lead_h.float(), H, W),
                      static.expand(N, -1, -1, -1)], 1)


class FlowSigma(nn.Module):
    def __init__(self, in_channels: int, sigma_clim: torch.Tensor, layers=(3, 2, 2), hidden=(64, 32, 32),
                 g_max: float = 3.0):
        super().__init__()
        self.net = Climate_ResNet_2D(in_channels, list(layers), list(hidden))
        self.head = nn.Conv2d(hidden[-1], sigma_clim.shape[1], 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("sigma_clim", sigma_clim)  # (L, C, H, W)
        self.g_max = g_max

    def forward(self, feats: torch.Tensor, lead_idx: torch.Tensor) -> torch.Tensor:
        g = self.head(self.net(feats)).clamp(-self.g_max, self.g_max)
        return self.sigma_clim[lead_idx] * g.exp()


def fit_flow_sigma(model: FlowSigma, feats, mu, tgt, lead_idx, train, val, epochs=30, lr=1e-3, batch=16,
                   weight_decay=1e-4, log=print) -> dict:
    """ClimODE's NLL of tgt around mu with std = model(feats); Adam, checkpoint on the ``val`` rows."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def val_nll():
        model.eval()
        with torch.no_grad():
            return float(np.mean([float(nll(mu[j], model(feats[j], lead_idx[j]), tgt[j], None, 0.0))
                                  for j in val.split(64)]))

    best = start = val_nll()
    best_state, hist = {k: v.clone() for k, v in model.state_dict().items()}, []
    for ep in range(epochs):
        model.train()
        for j in train[torch.randperm(len(train))].split(batch):
            loss = nll(mu[j], model(feats[j], lead_idx[j]), tgt[j], None, 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = val_nll()
        hist.append(v)
        if v < best:
            best, best_state = v, {k: q.clone() for k, q in model.state_dict().items()}
        log(f"  [flow sigma] epoch {ep + 1:2d} val NLL {v:.4f} (start {start:.4f})")
    model.load_state_dict(best_state)
    model.eval()
    return {"val_nll_start": start, "val_nll": hist, "best": best}


def predict_sigma(model: FlowSigma, feats, lead_idx, chunk=64) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(feats[j], lead_idx[j]) for j in torch.arange(len(feats)).split(chunk)])
