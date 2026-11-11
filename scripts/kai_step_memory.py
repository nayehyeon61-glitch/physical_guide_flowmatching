"""Peak memory, time and gradient of ONE KAI training step, with or without gradient checkpointing per block
(docs/kai_tendency_plus.md, section 4.4). Random input: the numbers are about memory, not skill.

KAI keeps the full grid in all four stages, so the stored activations grow with the number of grid points:
measure on a coarse grid on a small machine and scale with H x W (the part that does not scale is the
difference between two grids).

    python scripts/kai_step_memory.py --grid 61 120 --ckpt 0
    python scripts/kai_step_memory.py --grid 61 120 --ckpt 1 --days 3
    python scripts/kai_step_memory.py --grid 121 240 --batch 8 --ckpt 1 --device cuda   # the real size, on a GPU

Run every case in its own process (the peak is the process's peak). ``--save g.pt`` writes the gradient: two
runs with --ckpt 0 / 1 must give the same one.
"""
import argparse
import os
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pgcfm.kai_tend import KAIPos, KAIPosConfig  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, nargs=2, default=[61, 120], help="H W")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--days", type=int, default=1, help="rollout days of the loss (the fine-tuning: 2 to 3)")
    p.add_argument("--ckpt", type=int, default=0, help="1 = gradient checkpointing per block (--kai-ckpt)")
    p.add_argument("--depths", type=int, nargs=4, default=[3, 3, 15, 3])
    p.add_argument("--dims", type=int, nargs=4, default=[48, 96, 192, 288])
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--save", default="", help="write the gradient vector here")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    dev = torch.device(a.device)
    H, W = a.grid
    m = KAIPos(KAIPosConfig.arm("kai", a.depths, a.dims), np.ones(67, np.float32), None).to(dev)
    m.grad_ckpt = bool(a.ckpt)
    m.train()
    x = torch.randn(a.batch, 67, H, W, device=dev)
    tg = torch.randn(a.batch, a.days, 67, H, W, device=dev)
    doy = torch.zeros(a.batch, dtype=torch.long, device=dev)
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    t0 = time.time()
    y = m.rollout(x, doy, a.days, None) if a.days > 1 else m(x, doy)[:, None]
    loss = ((y - tg) ** 2).mean()
    loss.backward()
    g = torch.cat([q.grad.flatten() for q in m.parameters() if q.grad is not None])
    sec = time.time() - t0
    out = dict(grid=f"{H}x{W}", batch=a.batch, days=a.days, ckpt=a.ckpt, params=sum(q.numel() for q in m.parameters()),
               loss=round(float(loss.detach()), 8), grad_norm=round(float(g.norm()), 8), sec=round(sec, 1))
    if dev.type == "cuda":
        out["gpu_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
    else:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        out.update(base_gb=round(base, 2), peak_gb=round(peak, 2), step_gb=round(peak - base, 2))
    print(out)
    if a.save:
        torch.save(g.cpu(), a.save)


if __name__ == "__main__":
    main()
