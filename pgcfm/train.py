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
    while True:
        for b in loader:
            yield b


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
            loss, logs = model.joint_loss(b, start_iter + it)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sched.step()
        logs["loss"] = loss.item()
        hist.append(logs)
        if (it + 1) % log_every == 0 or it + 1 == iters:
            avg = {k: float(np.mean([h[k] for h in hist[-log_every:] if k in h])) for k in logs}
            avg["sec"] = time.time() - t0
            print(f"  [{phase}] it {it + 1:6d} " + " ".join(f"{k} {v:.4g}" for k, v in avg.items()), flush=True)
            if on_log is not None:
                on_log(it + 1, avg)
    model.eval()
    return hist


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
