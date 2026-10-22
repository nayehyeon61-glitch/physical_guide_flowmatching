"""Step 6: staged PG-CFM in the NeuralGCM style.

The joint objective of step 4 (eq. 12) failed for one structural reason: the
physics gradient on theta is ~3000x the CFM gradient, so it either takes the
update over or, when capped, does nothing (docs/step4_pgcfm.md W2). This
module separates the three jobs instead of mixing them in one loss:

  1. PINN first, then frozen. The bridge h_psi and closure C_chi (step 4's
     ``PhysicsGuide``) are trained on OBSERVED pairs until the held-out
     physics loss stops improving, then every parameter is frozen. From then
     on the guide is a fixed differentiable operator, like the dynamical core
     of NeuralGCM (Kochkov et al. 2024).
  2. The CFM transition (step 2) stays as it is, over-dispersed, and frozen.
     Its endpoint y_raw is a proposal cloud.
  3. A projection ODE in pseudo-time s in [0, 1],
         dy/ds = g_phi(y, s | x_k, c_k),     y(0) = y_raw,     y(1) = x_{k+1},
     integrated with torchdiffeq (https://github.com/rtqichen/torchdiffeq),
     compresses the cloud toward the data manifold. g_phi is the same torchcfm
     U-Net as the CFM velocity, whose last layer is zero-initialized, so the
     projection starts as the identity (= CFM alone).
     It is trained the way NeuralGCM trains its stochastic model: unrolled
     member-wise Markov rollouts x_k -> x_{k+1} -> ... with a fair CRPS over
     2 members at every lead, a horizon curriculum, gradients through the
     whole rollout, plus optionally the frozen PINN residual of every
     transition (lambda_P) or the non-physical smoothing control of step 4.

``GradientFlowProjection`` is the untrained alternative: the projection ODE
is the normalized gradient flow of the frozen PINN loss, dy/ds = -eta g/|g|.
It shows what "project onto the physics manifold" does with no data term.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from . import metrics
from .cfm import TISR_DYN, CFMConfig, CFMTransition, VelocityNet
from .coupling import PhysicsGuide, laplacian


@dataclass
class ProjectionConfig:
    model_channels: int = 32
    channel_mult: tuple = (1, 2, 2)
    num_res_blocks: int = 1
    ode_method: str = "rk4"  # any fixed-grid torchdiffeq solver (euler, midpoint, rk4)
    ode_steps: int = 2  # NFE = 4 x ode_steps for rk4
    curriculum: tuple = (1, 2, 4)  # rollout horizons, equal share of the iterations
    members: int = 2  # NeuralGCM stochastic models: CRPS with 2 members
    lambda_p: float = 0.0  # weight of the frozen PINN residual per transition
    control: str = "physics"  # "physics" | "smooth" (non-physical Laplacian control)
    grad_ratio: float = 0.0  # > 0: cap ||grad phys|| at grad_ratio x ||grad crps||
    cfm_steps: int = 8  # Heun steps of the frozen CFM (same in training and inference)
    bptt: bool = True  # gradient flows through the frozen CFM sampler between steps


def freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module.eval()


class ODEProjection(nn.Module):
    """Frozen CFM proposal + learned projection ODE. Same ``sample`` interface as
    ``CFMTransition`` so ``cfm.rollout``, ``coupling.deterministic_rollout`` and
    ``train.validate_one_step`` run on it unchanged."""

    def __init__(self, cfm: CFMTransition, pcfg: ProjectionConfig, n_lat: int, n_lon: int):
        super().__init__()
        self.cfm, self.pcfg = freeze(cfm), pcfg
        self.cfg, self.gen_idx, self.w_area = cfm.cfg, cfm.gen_idx, cfm.w_area
        self.field = VelocityNet(CFMConfig(tisr_mode=cfm.cfg.tisr_mode, model_channels=pcfg.model_channels,
                                           channel_mult=tuple(pcfg.channel_mult),
                                           num_res_blocks=pcfg.num_res_blocks), n_lat, n_lon)

    def train(self, mode: bool = True):
        super().train(mode)
        self.cfm.eval()  # the CFM stays frozen
        return self

    def project(self, x: torch.Tensor, y: torch.Tensor, static: torch.Tensor, doy_next: torch.Tensor,
                tisr_next: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond = self.cfm.condition(x, static, doy_next, tisr_next)
        B = x.shape[0]
        ts = torch.linspace(0.0, 1.0, self.pcfg.ode_steps + 1, device=x.device, dtype=x.dtype)
        z = odeint(lambda s, z: self.field(s.expand(B), z, cond), y[:, self.gen_idx], ts,
                   method=self.pcfg.ode_method)[-1]
        return self.cfm._assemble(x, z, tisr_next)

    def sample(self, x, static, doy_next, tisr_next=None, steps=8, generator=None, grad=False, eps=None):
        y = self.cfm.sample(x, static, doy_next, tisr_next, steps=steps, generator=generator, grad=grad, eps=eps)
        with torch.set_grad_enabled(grad):
            return self.project(x, y, static, doy_next, tisr_next)


class GradientFlowProjection(nn.Module):
    """Untrained projection: dy/ds = -eta * grad_y L_phys / rms(grad_y L_phys), frozen PINN.

    Each Euler step moves the generated channels by ``eta / steps`` (normalized
    units, RMS), in the direction that lowers the residual the frozen bridge and
    closure see (``control="smooth"``: the same flow on the grid Laplacian
    energy, no physics). ``doy`` of x_k is taken as doy_next - 1 (the residual needs
    the calendar of x_k; only the closure uses it).
    """

    def __init__(self, cfm: CFMTransition, guide: PhysicsGuide, eta: float = 0.5, steps: int = 8,
                 control: str = "physics"):
        super().__init__()
        self.cfm, self.guide = freeze(cfm), freeze(guide)
        self.cfg, self.gen_idx, self.w_area = cfm.cfg, cfm.gen_idx, cfm.w_area
        self.eta, self.steps, self.control = eta, steps, control

    def project(self, x, y, static, doy_next, tisr_next=None):
        doy0 = (doy_next - 1) % 366
        z = y[:, self.gen_idx].detach()
        h = self.eta / self.steps
        m = self.guide.mask(x, y.detach(), static)  # surface fixed by the proposal
        for _ in range(self.steps):
            with torch.enable_grad():
                z = z.detach().requires_grad_(True)
                yy = self.cfm._assemble(x, z, tisr_next)
                if self.control == "smooth":  # non-physical control: same flow on ||lap y||^2
                    L = (laplacian(yy) ** 2).mean()
                else:
                    R, _ = self.guide.residuals(x, yy, static, doy0)
                    L = self.guide.loss(R, m)
                g, = torch.autograd.grad(L, z)
            g = g / g.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-20)
            z = z - h * g
        return self.cfm._assemble(x, z.detach(), tisr_next)

    def sample(self, x, static, doy_next, tisr_next=None, steps=8, generator=None, grad=False, eps=None):
        y = self.cfm.sample(x, static, doy_next, tisr_next, steps=steps, generator=generator, eps=eps)
        return self.project(x, y, static, doy_next, tisr_next)


# -- stage 1: PINN to convergence, then frozen --------------------------------

def train_pinn(model, loader, val_batches: Sequence[Dict[str, torch.Tensor]], max_iters: int = 5000,
               lr: float = 1e-3, window: int = 100, tol: float = 0.01, patience: int = 2,
               device="cpu", log=print) -> List[Dict[str, float]]:
    """Bridge h_psi + closure C_chi on observed pairs (``PGCFM.pretrain_loss``) until the
    held-out physics loss improves by less than ``tol`` (relative) for ``patience``
    consecutive windows; then the guide is frozen. ``model`` is a ``coupling.PGCFM``.
    Returns the per-window history."""
    from .train import _allreduce_grads, _cycle, _to
    g = model.guide
    params = list(g.bridge.parameters()) + list(g.closure.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)

    @torch.no_grad()
    def held_out():
        v = []
        for b in val_batches:
            b = _to(b, device)
            R, _ = g.residuals(b["x0"], b["targets"][:, 0], b["static"], b["doy"][:, 0])
            v.append(g.loss(R, g.mask(b["x0"], b["targets"][:, 0], b["static"])).item())
        return float(np.mean(v))

    hist, best, bad, it = [], held_out(), 0, 0
    hist.append({"it": 0, "val_phys": best})
    log(f"  [pinn] it      0 val_phys {best:.4g}")
    batches = _cycle(loader)
    while it < max_iters:
        tr = []
        for _ in range(window):
            b = _to(next(batches), device)
            loss, logs = model.pretrain_loss(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            _allreduce_grads(params)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr.append(logs["phys"])
            it += 1
        v = held_out()
        rel = (best - v) / max(abs(best), 1e-12)
        hist.append({"it": it, "train_phys": float(np.mean(tr)), "val_phys": v, "rel_improvement": rel})
        log(f"  [pinn] it {it:6d} train_phys {np.mean(tr):.4g} val_phys {v:.4g} rel {rel:+.3f}")
        bad = bad + 1 if rel < tol else 0
        best = min(best, v)
        if bad >= patience:
            break
    freeze(g)
    return hist


# -- stage 3: sequence (rollout) loss -----------------------------------------

def _keep(t: torch.Tensor) -> torch.Tensor:
    keep = [i for i in range(t.shape[-3]) if i != TISR_DYN]
    return t[..., keep, :, :]


def sequence_terms(proj: ODEProjection, guide: PhysicsGuide, batch, horizon: int
                   ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Unrolled member-wise rollout of ``horizon`` days (NeuralGCM training).

    Every member feeds its own projected state into the next transition, fresh
    CFM noise per transition. Returns {"crps": mean over leads of the fair
    area-weighted CRPS (2 members), "phys": mean over leads of the frozen PINN
    loss of each member transition (or the smoothing control)}.
    """
    pc = proj.pcfg
    M = pc.members
    x0, tg, st, doy = batch["x0"], batch["targets"], batch["static"], batch["doy"]
    B = x0.shape[0]
    rep = lambda t: t.repeat(M, *([1] * (t.dim() - 1)))  # noqa: E731
    x, sts = rep(x0), rep(st)
    crps, phys, logs = [], [], {}
    prev_obs = x0
    for k in range(horizon):
        y_obs = tg[:, k]
        tn = rep(y_obs[:, TISR_DYN:TISR_DYN + 1]) if proj.cfg.tisr_mode == "forcing" else None
        d1 = rep(doy[:, k + 1])
        y_raw = proj.cfm.sample(x, sts, d1, tn, steps=pc.cfm_steps, grad=pc.bptt)
        if not pc.bptt:
            x = x.detach()
        y = proj.project(x, y_raw, sts, d1, tn)
        ens = y.view(M, B, *y.shape[1:])
        crps.append(metrics.crps(_keep(ens), _keep(y_obs)).mean())
        if pc.lambda_p > 0 and pc.control == "smooth":
            phys.append((laplacian(y) ** 2).mean() / (laplacian(y_obs) ** 2).mean().clamp_min(1e-12))
        elif pc.lambda_p > 0:
            R, _ = guide.residuals(x, y, sts, rep(doy[:, k]))
            phys.append(guide.loss(R, rep(guide.mask(prev_obs, y_obs, st))))
        x, prev_obs = y, y_obs
    terms = {"crps": torch.stack(crps).mean()}
    logs["crps"] = terms["crps"].item()
    logs.update({f"crps_d{k + 1}": c.item() for k, c in enumerate(crps)})
    if phys:
        p = torch.stack(phys).mean()
        terms["phys"] = pc.lambda_p * p
        logs["phys"] = p.item()
    return terms, logs


def curriculum_horizon(pcfg: ProjectionConfig, it: int, iters: int) -> int:
    stages = list(pcfg.curriculum)
    return stages[min(len(stages) - 1, it * len(stages) // max(1, iters))]


def train_projection(proj: ODEProjection, guide: PhysicsGuide, loader, iters: int, lr: float = 2e-4,
                     device="cpu", log_every: int = 50, clip: float = 1.0, log=print,
                     on_log=None) -> List[Dict[str, float]]:
    """Stage 3. Only g_phi is trained; CFM and PINN are frozen. With ``grad_ratio`` the
    physics gradient is capped at grad_ratio x the CRPS gradient (the step 4 W2 fix)."""
    from .train import _allreduce_grads, _cycle, _to, is_main
    params = [p for p in proj.field.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, iters))
    proj.train()
    hist, batches = [], _cycle(loader)
    norm = lambda gs: torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None))  # noqa: E731
    for it in range(iters):
        b = _to(next(batches), device)
        K = min(curriculum_horizon(proj.pcfg, it, iters), b["targets"].shape[1])
        terms, logs = sequence_terms(proj, guide, b, K)
        opt.zero_grad(set_to_none=True)
        if "phys" in terms:
            g_c = torch.autograd.grad(terms["crps"], params, retain_graph=True, allow_unused=True)
            g_p = torch.autograd.grad(terms["phys"], params, allow_unused=True)
            n_c, n_p = norm(g_c), norm(g_p)
            c = 1.0
            if proj.pcfg.grad_ratio > 0:
                c = float(torch.clamp(proj.pcfg.grad_ratio * n_c / (n_p + 1e-12), max=1.0))
            for p, a, q in zip(params, g_c, g_p):
                p.grad = None if a is None and q is None else \
                    (a if q is None else (c * q if a is None else a + c * q))
            logs.update(grad_crps=n_c.item(), grad_phys=n_p.item(), phys_scale=c)
        else:
            terms["crps"].backward()
        _allreduce_grads(params)
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sched.step()
        logs["horizon"] = K
        hist.append(logs)
        if (it + 1) % log_every == 0 or it + 1 == iters:
            avg = {k: float(np.mean([h[k] for h in hist[-log_every:] if k in h])) for k in logs}
            if is_main():
                log(f"  [proj] it {it + 1:6d} " + " ".join(f"{k} {v:.4g}" for k, v in avg.items()))
            if on_log is not None and is_main():
                on_log(it + 1, avg)
    proj.eval()
    return hist
