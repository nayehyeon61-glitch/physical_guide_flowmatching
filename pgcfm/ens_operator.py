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

Second version (``op2`` arms, after the owner's ``possibility_operator`` package and design
note, 2026-10-23): the first version diverged in rollout (docs/step7_latent.md W6), so
``train_operator`` can now unroll several days with a horizon curriculum (the package's
``rollout_steps`` / step 6), add the correction-size penalty lambda_corr ||X_hat - X_tilde||^2,
pad latitude (``lat_padding``) so the FFT does not wrap pole to pole, and be fit on a
calibration period the base chain never saw.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from neuralop.models import FNO

from . import contract
from .cfm import TISR_DYN, calendar_features
from .coupling import energy_score
from .latent import _capped_backward, _checkpoint, _log_avg, _resume, _set_trainable

N_COND = contract.N_DYNAMIC + 1 + 2


class EnsembleOperator(nn.Module):
    def __init__(self, C: int = contract.N_DYNAMIC, K: int = N_COND, D: int = 32, n_modes=(12, 12),
                 hidden: int = 64, lat_padding: float = 0.0):
        super().__init__()
        # lat_padding > 0: neuraloperator domain padding in latitude only, so the FFT does not
        # treat the two poles as neighbours (longitude stays periodic, no padding)
        pad = [lat_padding, 0.0] if lat_padding > 0 else None
        self.encoder = FNO(n_modes=n_modes, in_channels=C + K, out_channels=D, hidden_channels=hidden,
                           domain_padding=pad)
        self.corrector = FNO(n_modes=n_modes, in_channels=2 * D + K, out_channels=C, hidden_channels=hidden,
                             domain_padding=pad)
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


class MemberGrouped(nn.Module):
    """A transition that pools over ensemble members, with the ``CFMTransition.sample`` interface.
    It has to know how the flat batch is laid out: ``order="repeat"`` (member-major,
    ``t.repeat(M)``, as in ``validate_one_step``) or ``"interleave"`` (sample-major, as in
    ``cfm.rollout``); ``members=1`` treats every row as its own one-member ensemble (eps = 0 runs,
    single draws)."""

    def __init__(self, base, members: int = 1, order: str = "repeat"):
        super().__init__()
        self.base = base
        self.cfg, self.gen_idx, self.w_area = base.cfg, base.gen_idx, base.w_area
        self.layout(members, order)

    def layout(self, members: int, order: str = "repeat"):
        if order not in ("repeat", "interleave"):
            raise ValueError(order)
        self.members, self.order = members, order
        if isinstance(self.base, MemberGrouped):  # stacked ensemble maps share the layout
            self.base.layout(members, order)
        return self

    def _group(self, t: torch.Tensor) -> torch.Tensor:  # (N, ...) -> (B, M, ...)
        M = self.members
        if self.order == "repeat":
            return t.view(M, t.shape[0] // M, *t.shape[1:]).transpose(0, 1)
        return t.view(t.shape[0] // M, M, *t.shape[1:])

    def _ungroup(self, t: torch.Tensor) -> torch.Tensor:
        return (t.transpose(0, 1) if self.order == "repeat" else t).reshape(-1, *t.shape[2:])


class OperatorCorrected(MemberGrouped):
    """base transition (e.g. the serial ``LatentPGCFM``) + ``EnsembleOperator`` (or any module
    ``op(U (B,M,C,H,W), cond (B,M,K,H,W)) -> (B,M,C,H,W)`` with a ``gate``).

    ``band``: project the corrected members onto the kept DCT modes, so the FNO cannot put
    back the grid-scale noise the DCT truncation removed (hard constraint)."""

    def __init__(self, base, op: nn.Module, members: int = 1, order: str = "repeat", band=None):
        super().__init__(base, members, order)
        self.op, self.band = op, band  # band: a latent.DCTTruncation (hard band limit)

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
                   lambda_p: float = 0.1, grad_ratio: float = 0.5, steps: int = 8, curriculum=(1,),
                   lambda_corr: float = 0.0, bptt: bool = True, log_every: int = 50, log=print,
                   device="cpu", ckpt: Optional[str] = None, ckpt_every: int = 20,
                   lambda_grad: float = 0.0) -> List[Dict[str, float]]:
    """Operator only; the base chain (encoder, CFM, PINNs, decoder) is frozen.

    Loss per lead k of an unrolled member-wise rollout (horizon from ``curriculum``, as in
    step 6 / the owner's possibility_operator ``rollout_steps``):
        fair energy score(corrected members, truth)
      + lambda_corr x mean ||X_hat^m - X_tilde^m||^2 (correction size, owner's note)
      + lambda_grad x fair energy score of the spatial differences (step 9: spatial structure)
      + lambda_P x primitive-equation residual of each corrected member transition (capped).
    Corrected members are fed back as the next initial state; with ``bptt`` the gradient
    crosses the whole rollout (through the frozen base, w.r.t. its inputs). ``curriculum=(1,)``
    is the one-step training of the first version. The loader must give >= max(curriculum)
    targets; members are drawn fresh every batch. ``ckpt``: resumable checkpoint, as in
    ``latent.train_serial``.
    """
    from .projection import curriculum_horizon, ProjectionConfig
    from .structure import spatial_diff
    from .train import _cycle, _to
    _set_trainable([wrapped.base], False)
    _set_trainable([wrapped.op], True)
    wrapped.layout(members, "repeat")
    params = list(wrapped.op.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, iters))
    keep = [i for i in range(contract.N_DYNAMIC) if i != TISR_DYN]
    use_phys = guide is not None and lambda_p > 0
    pc = ProjectionConfig(curriculum=tuple(curriculum))
    batches = _cycle(loader)
    hist, start = _resume(ckpt, {"op": wrapped.op}, opt, sched, device)
    if start:
        log(f"  [operator] resumed at it {start}")
    for it in range(start, iters):
        if ckpt is not None and it > start and it % ckpt_every == 0:
            _checkpoint(ckpt, {"op": wrapped.op}, opt, sched, hist, it)
        b = _to(next(batches), device)
        x0, tg, st, d = b["x0"], b["targets"], b["static"], b["doy"]
        K = min(curriculum_horizon(pc, it, iters), tg.shape[1])
        rep = lambda t: t.repeat(members, *([1] * (t.dim() - 1)))  # noqa: E731
        xs, sts = rep(x0), rep(st)
        es_l, corr_l, phys_l, gr_l = [], [], [], []
        for k in range(K):
            y, dn = tg[:, k], rep(d[:, k + 1])
            with torch.set_grad_enabled(bptt and k > 0):
                u = wrapped.base.sample(xs, sts, dn, steps=steps, grad=bptt and k > 0)
            out = wrapped.correct(xs, u, sts, dn)
            es_l.append(energy_score(out.view(members, *x0.shape)[:, :, keep], y[:, keep], wrapped.w_area))
            corr_l.append(((out - u) ** 2 * wrapped.w_area).mean())
            if lambda_grad > 0:
                gr_l.append(energy_score(spatial_diff(out.view(members, *x0.shape)[:, :, keep]),
                                         spatial_diff(y[:, keep]), wrapped.w_area))
            if use_phys:
                R, _ = guide.residuals(xs, out, sts, rep(d[:, k]), use_bridge=False, use_closure=False)
                prev = x0 if k == 0 else tg[:, k - 1]
                phys_l.append(guide.loss(R, rep(guide.mask(prev, y, st))))
            xs = out if bptt else out.detach()
        es, corr = torch.stack(es_l).mean(), torch.stack(corr_l).mean()
        terms = {"data": es + lambda_corr * corr}
        logs = {"es": es.item(), "corr": corr.item(), "gate": wrapped.op.gate.item(), "horizon": K}
        if gr_l:
            gr = torch.stack(gr_l).mean()
            terms["data"] = terms["data"] + lambda_grad * gr
            logs["es_grad"] = gr.item()
        if use_phys:
            terms["phys"] = torch.stack(phys_l).mean()
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
