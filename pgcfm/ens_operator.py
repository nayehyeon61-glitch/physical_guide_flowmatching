"""Step 7, last block: ensemble neural operator on top of the serial chain (owner's
``ensembl_operator.py`` sketch, 2026-10-23).

    DCT encoder -> latent CFM members A, B, C, ... -> PINN Q_phi per member -> decoder/IDCT
                -> EnsembleOperator (this module) -> corrected members

``EnsembleOperator`` is the sketch's design with neuraloperator's FNO
(https://github.com/neuraloperator/neuraloperator): a member-shared FNO encoder,
the ensemble mean of its features as a summary, and an FNO corrector fed
[member features, summary, condition], output added to the member. It learns
in function space (spectral convolutions), so the same weights apply at any
resolution.

Changes from the sketch:
  * one Markov step (T = 1, C = 66 dynamic channels); the condition per member
    is [x_k, orography, calendar] (K = 69), all known at forecast time;
  * the correction passes through a gate initialized at 0, so training starts
    from the serial chain's members;
  * the energy score is the fair estimator of ``coupling.energy_score``
    (pairs m != n). The sketch's ``cdist(p, p).mean()`` includes the zero
    diagonal, which rewards extra spread for small M;
  * the physics term is the step 3/4 primitive-equation residual of each member
    transition (x_k, y^m), linear bridge, no closure, capped like every other
    physics term here (``latent._capped_backward``).

The FNO pads nothing in latitude, so its FFT treats the poles as neighbours
(lon is periodic, lat is not); see docs/step7_latent.md.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from neuralop.models import FNO

from . import contract
from .cfm import TISR_DYN, calendar_features
from .coupling import energy_score
from .latent import _capped_backward, _log_avg, _set_trainable

N_COND = contract.N_DYNAMIC + 1 + 2


class EnsembleOperator(nn.Module):
    def __init__(self, C: int = contract.N_DYNAMIC, K: int = N_COND, D: int = 32, n_modes=(12, 12),
                 hidden: int = 64):
        super().__init__()
        self.encoder = FNO(n_modes=n_modes, in_channels=C + K, out_channels=D, hidden_channels=hidden)
        self.corrector = FNO(n_modes=n_modes, in_channels=2 * D + K, out_channels=C, hidden_channels=hidden)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, U: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """U (B, M, C, H, W) members, cond (B, M, K, H, W) -> corrected (B, M, C, H, W)."""
        B, M, C, H, W = U.shape
        h = self.encoder(torch.cat([U, cond], dim=2).flatten(0, 1)).reshape(B, M, -1, H, W)
        summary = h.mean(dim=1, keepdim=True).expand_as(h)
        corr = self.corrector(torch.cat([h, summary, cond], dim=2).flatten(0, 1)).reshape(B, M, C, H, W)
        return U + self.gate * corr


def member_condition(x: torch.Tensor, static: torch.Tensor, doy_next: torch.Tensor) -> torch.Tensor:
    B, _, H, W = x.shape
    cal = calendar_features(doy_next).to(x)[:, :, None, None].expand(B, 2, H, W)
    return torch.cat([x, static, cal], dim=1)


class OperatorCorrected(nn.Module):
    """base transition (e.g. the serial ``LatentPGCFM``) + ``EnsembleOperator``, with the
    ``CFMTransition.sample`` interface. The operator pools over members, so it has to know
    how the flat batch is laid out: ``order="repeat"`` (member-major, ``t.repeat(M)``, as in
    ``validate_one_step``) or ``"interleave"`` (sample-major, as in ``cfm.rollout``); ``members=1``
    treats every row as its own one-member ensemble (eps = 0 runs, single draws).

    ``band``: project the corrected members onto the kept DCT modes, so the FNO cannot put
    back the grid-scale noise the DCT truncation removed (hard constraint)."""

    def __init__(self, base, op: EnsembleOperator, members: int = 1, order: str = "repeat", band=None):
        super().__init__()
        self.base, self.op, self.band = base, op, band  # band: a latent.DCTTruncation (hard band limit)
        self.cfg, self.gen_idx, self.w_area = base.cfg, base.gen_idx, base.w_area
        self.layout(members, order)

    def layout(self, members: int, order: str = "repeat"):
        if order not in ("repeat", "interleave"):
            raise ValueError(order)
        self.members, self.order = members, order
        return self

    def _group(self, t: torch.Tensor) -> torch.Tensor:  # (N, ...) -> (B, M, ...)
        M = self.members
        if self.order == "repeat":
            return t.view(M, t.shape[0] // M, *t.shape[1:]).transpose(0, 1)
        return t.view(t.shape[0] // M, M, *t.shape[1:])

    def _ungroup(self, t: torch.Tensor) -> torch.Tensor:
        return (t.transpose(0, 1) if self.order == "repeat" else t).reshape(-1, *t.shape[2:])

    def correct(self, x, y, static, doy_next):
        if y.shape[0] % self.members:
            raise ValueError(f"batch {y.shape[0]} is not a multiple of members={self.members}")
        cond = member_condition(x, static, doy_next)
        out = self._ungroup(self.op(self._group(y), self._group(cond)))
        return out if self.band is None else self.band.filter(out)

    def sample(self, x, static, doy_next, tisr_next=None, steps=8, generator=None, grad=False, eps=None):
        y = self.base.sample(x, static, doy_next, tisr_next, steps=steps, generator=generator, eps=eps)
        with torch.set_grad_enabled(grad):
            return self.correct(x, y, static, doy_next)


def train_operator(wrapped: OperatorCorrected, guide, loader, iters: int, members: int = 4, lr: float = 1e-3,
                   lambda_p: float = 0.1, grad_ratio: float = 0.5, steps: int = 8, log_every: int = 50,
                   log=print, device="cpu") -> List[Dict[str, float]]:
    """Operator only (the base chain is frozen; its members are drawn fresh every batch, the
    sketch's 'PINN outputs generated and fixed'): fair energy score + capped lambda_P x physics."""
    from .train import _cycle, _to
    _set_trainable([wrapped.base], False)
    _set_trainable([wrapped.op], True)
    wrapped.layout(members, "repeat")
    params = list(wrapped.op.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, iters))
    keep = [i for i in range(contract.N_DYNAMIC) if i != TISR_DYN]
    use_phys = guide is not None and lambda_p > 0
    hist, batches = [], _cycle(loader)
    for it in range(iters):
        b = _to(next(batches), device)
        x, y, st, d = b["x0"], b["targets"][:, 0], b["static"], b["doy"]
        rep = lambda t: t.repeat(members, *([1] * (t.dim() - 1)))  # noqa: E731
        xs, sts, dn = rep(x), rep(st), rep(d[:, 1])
        with torch.no_grad():
            u = wrapped.base.sample(xs, sts, dn, steps=steps)
        out = wrapped.correct(xs, u, sts, dn)
        es = energy_score(out.view(members, *x.shape)[:, :, keep], y[:, keep], wrapped.w_area)
        terms, logs = {"data": es}, {"es": es.item(), "gate": wrapped.op.gate.item()}
        if use_phys:
            R, _ = guide.residuals(xs, out, sts, rep(d[:, 0]), use_bridge=False, use_closure=False)
            terms["phys"] = guide.loss(R, rep(guide.mask(x, y, st)))
            logs["phys"] = terms["phys"].item()
        opt.zero_grad(set_to_none=True)
        _capped_backward(terms, params, lambda_p, grad_ratio, logs)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        hist.append(logs)
        if (it + 1) % log_every == 0 or it + 1 == iters:
            _log_avg(hist, logs, it + 1, log_every, "operator", log)
    _set_trainable([wrapped.op], False)
    return hist
