"""PG-CFM training procedure of paper appendix C, shared by the ERA5 script
(``scripts/train_pgcfm.py``) and the step 4 experiment.

    phase "cfm":   (x_k, x_{k+1}) pairs, L_CFM only                (C.2)
    phase "guide": bridge h_psi + closure C_chi on OBSERVED pairs  (C.3)
                   theta untouched; afterwards C_chi is frozen
    phase "joint": eq. (12) with lambda_P ramped up, theta and psi (C.4)
"""

import itertools
import time
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch

from .coupling import PGCFM


def _cycle(loader) -> Iterable:
    epoch = 0
    while True:
        if hasattr(getattr(loader, "sampler", None), "set_epoch"):
            loader.sampler.set_epoch(epoch)  # DistributedSampler reshuffle
        for b in loader:
            yield b
        epoch += 1


def _ddp() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def is_main() -> bool:
    return not _ddp() or torch.distributed.get_rank() == 0


def broadcast_module(module: torch.nn.Module) -> None:
    """Rank 0's parameters and buffers to every rank (multi-GPU start / resume)."""
    if _ddp():
        for t in list(module.parameters()) + list(module.buffers()):
            torch.distributed.broadcast(t.data, 0)


def _allreduce_grads(params) -> None:
    """Average gradients over ranks. Used instead of DDP's wrapper because the
    phases call different methods (cfm.loss, pretrain_loss, joint_loss), not
    forward(); same arithmetic as DDP, one all-reduce per parameter."""
    if not _ddp():
        return
    ws = torch.distributed.get_world_size()
    for p in params:
        if p.grad is not None:
            torch.distributed.all_reduce(p.grad)
            p.grad.div_(ws)


def _to(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_phase(model: PGCFM, phase: str, loader, iters: int, lr: float, device="cpu",
              log_every: int = 100, start_iter: int = 0, clip: float = 1.0,
              on_log: Optional[Callable[[int, Dict[str, float]], None]] = None) -> List[Dict[str, float]]:
    """One phase of appendix C. Returns the per-iteration logs."""
    if phase == "cfm":
        params = list(model.cfm.parameters())
    elif phase == "guide":
        params = list(model.guide.bridge.parameters()) + list(model.guide.closure.parameters())
    elif phase == "joint":
        for p in model.guide.closure.parameters():  # frozen closure (sec. 5.2)
            p.requires_grad_(False)
        params = list(model.cfm.parameters()) + list(model.guide.bridge.parameters())
    else:
        raise ValueError(phase)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, iters))
    model.train()
    hist, t0 = [], time.time()
    for it, b in zip(range(iters), _cycle(loader)):
        b = _to(b, device)
        if phase == "cfm":
            loss = model.cfm.loss(b["x0"], b["targets"][:, 0], b["static"], b["doy"][:, 1])
            logs = {"cfm": loss.item()}
        elif phase == "guide":
            loss, logs = model.pretrain_loss(b)
        else:
            loss, logs = None, None
        opt.zero_grad(set_to_none=True)
        if phase == "joint" and model.guide.cfg.grad_ratio > 0:
            loss, logs = balanced_backward(model, b, start_iter + it, params, model.guide.cfg.grad_ratio)
        elif phase == "joint":
            loss, logs = model.joint_loss(b, start_iter + it)
            loss.backward()
        else:
            loss.backward()
        _allreduce_grads(params)
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sched.step()
        logs["loss"] = loss.item()
        hist.append(logs)
        if (it + 1) % log_every == 0 or it + 1 == iters:
            avg = {k: float(np.mean([h[k] for h in hist[-log_every:] if k in h])) for k in logs}
            avg["sec"] = time.time() - t0
            if is_main():
                print(f"  [{phase}] it {it + 1:6d} " + " ".join(f"{k} {v:.4g}" for k, v in avg.items()), flush=True)
            if on_log is not None and is_main():
                on_log(it + 1, avg)
    model.eval()
    return hist


def balanced_backward(model: PGCFM, batch, it: int, params, ratio: float):
    """Backward of eq. (12) with the physics gradient on theta capped at ``ratio``
    times the CFM gradient norm (step 4 weakness: with global clipping, once
    L_phys >> L_CFM the update direction is L_phys alone and lambda_P stops
    mattering). The bridge psi only receives the physics gradient, uncapped.
    """
    terms, logs = model.joint_terms(batch, it)
    theta = [p for p in model.cfm.parameters() if p.requires_grad]
    tid = {id(p) for p in theta}
    g_cfm = torch.autograd.grad(terms["cfm"], theta, retain_graph=True, allow_unused=True)
    rest = sum(v for k, v in terms.items() if k != "cfm")
    if not torch.is_tensor(rest):
        for p, g in zip(theta, g_cfm):
            p.grad = g
        return terms["cfm"], logs
    g_rest = torch.autograd.grad(rest, params, allow_unused=True)
    norm = lambda gs: torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None))  # noqa: E731
    n_cfm = norm(g_cfm)
    n_phys = norm([g for p, g in zip(params, g_rest) if id(p) in tid])
    c = torch.clamp(ratio * n_cfm / (n_phys + 1e-12), max=1.0)
    logs.update(grad_cfm=n_cfm.item(), grad_phys=n_phys.item(), phys_scale=c.item())
    gc = {id(p): g for p, g in zip(theta, g_cfm)}
    for p, g in zip(params, g_rest):
        if id(p) in tid:
            a = gc.get(id(p))
            g = (c * g if g is not None else None)
            g = a if g is None else (g if a is None else a + g)
        p.grad = g
    return sum(terms.values()), logs


def take(loader, n: int) -> List[Dict[str, torch.Tensor]]:
    return list(itertools.islice(_cycle(loader), n))


@torch.no_grad()
def validate_one_step(model: PGCFM, batches, n_members: int = 4, steps: int = 8,
                      device="cpu") -> Dict[str, float]:
    """1-day ensemble scores and the independent physics residual (sec. 6.3).

    The residual of generated members uses the LINEAR bridge and no closure,
    so neither h_psi nor C_chi can make it small; it is reported next to the
    same quantity for the observed pair.
    """
    from . import metrics
    from .cfm import TISR_DYN

    acc: Dict[str, List[float]] = {}

    def add(k, v):
        acc.setdefault(k, []).append(float(v))

    g = model.guide
    for b in batches:
        b = _to(b, device)
        x, y, st = b["x0"], b["targets"][:, 0], b["static"]
        B = x.shape[0]
        tn = y[:, TISR_DYN:TISR_DYN + 1] if model.cfm.cfg.tisr_mode == "forcing" else None
        rep = lambda t: t.repeat(n_members, *([1] * (t.dim() - 1)))  # noqa: E731
        ens = model.cfm.sample(rep(x), rep(st), rep(b["doy"][:, 1]), None if tn is None else rep(tn), steps=steps)
        ens = ens.view(n_members, B, *x.shape[1:])
        keep = [i for i in range(x.shape[1]) if i != TISR_DYN]
        e, o = ens[:, :, keep], y[:, keep]
        add("crps", metrics.crps(e, o).mean())
        add("rmse", metrics.ens_mean_rmse(e, o).pow(2).mean().sqrt())
        add("spread", metrics.ens_spread(e).pow(2).mean().sqrt())
        tend_hat = metrics._wmean((e - x[None, :, keep]) ** 2).mean().sqrt()
        tend_obs = metrics._wmean((o - x[:, keep]) ** 2).mean().sqrt()
        add("tendency_ratio", tend_hat / tend_obs)
        m = g.mask(x, y, st)
        R, _ = g.residuals(rep(x), ens.reshape(-1, *x.shape[1:]), rep(st), rep(b["doy"][:, 0]),
                           use_bridge=False, use_closure=False)
        for k, v in g.per_equation(R, rep(m)).items():
            add(f"res_gen_{k}", v)
        R, _ = g.residuals(x, y, st, b["doy"][:, 0], use_bridge=False, use_closure=False)
        for k, v in g.per_equation(R, m).items():
            add(f"res_obs_{k}", v)
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["ssr"] = ((n_members + 1) / n_members) ** 0.5 * out["spread"] / out["rmse"]
    return out
