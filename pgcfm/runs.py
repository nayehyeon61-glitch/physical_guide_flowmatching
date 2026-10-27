"""Shared pieces of the stage trainers for long runs on large data (docs/step15_training_system.md).

    device      ``device_from("auto")`` = cuda when there is one; ``autocast(device, amp)`` = bf16 on cuda
    data        ``load_years(..., cache=dir)`` keeps every year as a memory-mapped normalized .npy
                (``YearArray``, indexed like the in-RAM tensor) instead of 0.8 GB of RAM per year;
                ``Prefetch`` gathers the next batches in a thread while the step runs
    training    ``fit``: one resumable loop for every trainer: atomic ``<ckpt>.train`` state (model,
                optimizer, schedule, every RNG), validation every ``val_every`` steps on fixed windows,
                ``<ckpt>.best`` and selection of the best state, early stopping, a wall-clock budget,
                ``<ckpt>_metrics.csv`` (step, seconds, lr, train and validation losses)
    record      ``run_record``: commit, torch version, device, host and start time, for the result json
"""

import contextlib
import csv
import dataclasses
import datetime
import json
import os
import queue
import socket
import subprocess
import threading
import time
from typing import Callable, Dict, Optional

import netCDF4 as nc
import numpy as np
import torch

from . import contract

EXIT_BUDGET = 75  # EX_TEMPFAIL: the wall-clock budget ended the run; the same command resumes it


def device_from(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def autocast(device: torch.device, amp: bool):
    """bf16 autocast on cuda (no loss scaling needed); a no-op elsewhere."""
    if amp and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def run_record(device) -> Dict:
    """What a result json needs to say where its numbers came from."""
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=here, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:  # not a checkout, no git
            return ""

    dev = torch.device(device)
    return {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "torch": torch.__version__, "device": str(dev),
            "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "host": socket.gethostname(), "threads": torch.get_num_threads(),
            "start": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


# -- data ---------------------------------------------------------------------------------------------

class YearArray:
    """One year of normalized KAI fields (T, 67, H, W) as a memory-mapped float32 .npy. Indexing returns a
    torch tensor, so it stands in for the in-RAM tensor of ``load_years`` (slices, channel lists, ``len``)."""

    def __init__(self, path: str):
        self.a = np.load(path, mmap_mode="r")
        self.shape = self.a.shape

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.a[idx], dtype=np.float32))  # a writable copy of the window

    def numpy(self):
        return np.asarray(self.a)


def normalize_year(kai: str, year: int, mean, std) -> np.ndarray:
    """(T, 67, H, W) float32, KAI's z-score (std + eps), the arithmetic of the original loader."""
    with nc.Dataset(os.path.join(kai, f"{year}.nc")) as d:
        f = np.asarray(d["fields"][:], dtype=np.float32)
    f -= mean[:, None, None]
    f /= (std + contract.KAI_STD_EPS)[:, None, None]
    return f


def load_years(kai: str, years, mean, std, cache: Optional[str] = None) -> Dict:
    """{year: (T, 67, H, W) normalized}: tensors in RAM, or with ``cache`` memory-mapped ``YearArray``s
    built once per year under ``cache`` (normalized with these statistics; a different mean/std
    refuses the cache instead of mixing normalizations)."""
    out = {}
    if cache is None:
        for y in years:
            out[y] = torch.from_numpy(normalize_year(kai, y, mean, std))
        return out
    os.makedirs(cache, exist_ok=True)
    fp = os.path.join(cache, "stats.npz")
    if os.path.exists(fp):
        with np.load(fp) as z:
            if not (np.array_equal(z["mean"], mean) and np.array_equal(z["std"], std)):
                raise ValueError(f"{cache} was normalized with other statistics; use another --cache")
    else:
        np.savez(fp, mean=np.asarray(mean), std=np.asarray(std))
    for y in years:
        path = os.path.join(cache, f"{y}.npy")
        if not os.path.exists(path):
            np.save(path + ".tmp.npy", normalize_year(kai, y, mean, std))
            os.replace(path + ".tmp.npy", path)
        out[y] = YearArray(path)
    return out


def tendency_std(data, years, dyn, chunk: int = 256) -> torch.Tensor:
    """Per-channel RMS of the 6 h tendency over ``years``, ``chunk`` steps at a time (a whole year in RAM
    is not needed)."""
    s, n = 0.0, 0
    for y in years:
        x = data[y]
        for i in range(0, len(x) - 1, chunk):
            a = x[i:min(len(x), i + chunk + 1)][:, dyn]
            d = a[1:] - a[:-1]
            s, n = s + (d.double() ** 2).sum((0, 2, 3)), n + d.shape[0] * d.shape[2] * d.shape[3]
    return (s / n).sqrt().float()


class Prefetch:
    """Batches of the next ``depth`` steps gathered in a thread. ``picks(rng)`` runs in the caller's
    thread, so the random stream is drawn in the same order as without prefetching (every numpy draw of
    a step belongs in ``picks``); ``gather(picks)`` (the slow window copies) runs in the worker. Each
    item carries the RNG state from before its picks, which ``fit`` stores in the checkpoint of that
    step, so a resumed run draws the same windows; ``close`` rewinds the rng past the unconsumed items."""

    def __init__(self, picks: Callable, gather: Callable, rng, depth: int = 2):
        self.picks, self.gather, self.rng, self.depth = picks, gather, rng, max(0, depth)
        self.pending, self.thread = [], None

    def _work(self):
        while True:
            item = self.q_in.get()
            if item is None:
                return
            state, p = item
            try:
                self.q_out.put((state, self.gather(p)))
            except BaseException as e:  # surfaces in the caller
                self.q_out.put((state, e))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.depth:
            state = self.rng.bit_generator.state
            return state, self.gather(self.picks(self.rng))
        if self.thread is None:
            self.q_in, self.q_out = queue.Queue(), queue.Queue()
            self.thread = threading.Thread(target=self._work, daemon=True)
            self.thread.start()
        while len(self.pending) < self.depth + 1:
            item = (self.rng.bit_generator.state, self.picks(self.rng))
            self.pending.append(item)
            self.q_in.put(item)
        self.pending.pop(0)
        state, batch = self.q_out.get()
        if isinstance(batch, BaseException):
            raise batch
        return state, batch

    def close(self):
        """Stops the worker and rewinds the rng to the first batch that was not consumed, so prefetching is
        invisible to whatever draws from the rng next (error fits, scale estimates, a later stage)."""
        if self.thread is not None:
            self.q_in.put(None)
            self.thread.join(timeout=5)
            self.thread = None
        if self.pending:
            self.rng.bit_generator.state = self.pending[0][0]
            self.pending = []


# -- training -----------------------------------------------------------------------------------------

@dataclasses.dataclass
class FitConfig:
    iters: int
    ckpt_every: int = 100
    log_every: int = 50
    val_every: int = 0          # 0 = no validation during training
    patience: int = 0           # validations without improvement before stopping; 0 = never
    select: str = "best"        # final weights: the best validation state, or "last"
    max_hours: float = 0.0      # wall clock of this invocation; 0 = none
    max_steps: int = 0          # steps of this invocation (job arrays with a time limit); 0 = all
    clip: float = 1.0           # gradient norm


def _rng_states(generators: Dict[str, torch.Generator]) -> Dict:
    s = {"torch": torch.get_rng_state(), "gen": {k: g.get_state() for k, g in generators.items()}}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_states(s: Dict, generators: Dict[str, torch.Generator]) -> None:
    torch.set_rng_state(s["torch"])
    for k, g in generators.items():
        if k in s["gen"]:
            g.set_state(s["gen"][k])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def _mean(hist, n):
    return float(np.mean([h["loss"] for h in hist[-n:]])) if hist else float("nan")


def fit(model: torch.nn.Module, params, opt, sched, batches: Prefetch, loss_fn: Callable, cfg: FitConfig,
        ckpt: str, *, val_fn: Optional[Callable] = None, generators: Optional[Dict[str, torch.Generator]] = None,
        after_step: Optional[Callable] = None, amp=contextlib.nullcontext(), log: Callable = print) -> Dict:
    """Train ``model`` for ``cfg.iters`` steps of ``loss_fn(batch) -> loss | (loss, {part: float})``,
    resuming from ``<ckpt>.train``.

    ``val_fn() -> float | {"loss": float, ...}`` runs every ``cfg.val_every`` steps and at the end on fixed
    windows (it must not touch the training random streams: use its own generator or fork_rng). The
    lowest validation loss is kept in ``<ckpt>.best`` and loaded back at the end when ``cfg.select`` is
    "best". ``generators`` are the torch generators the loss draws from; they are saved and restored with
    the checkpoint, with the global torch RNG and the numpy state of the batch picks.

    Returns {"hist": [{"loss", ...parts}], "val": [(it, {...})], "best_it", "best_val", "stopped",
    "seconds", "start_it"}; ``stopped`` is "iters", "patience" or "budget" (wall clock or ``max_steps`` of
    this invocation: state saved, nothing selected, rerun the same command to resume)."""
    generators = generators or {}
    params = list(params)  # a generator would be empty after the first clipping
    part, metrics = ckpt + ".train", ckpt + "_metrics.csv"
    hist, vals, start, t0 = [], [], 0, time.time()
    best_val, best_it, bad = float("inf"), -1, 0
    if os.path.exists(part):
        s = torch.load(part, weights_only=False)
        model.load_state_dict(s["model"]), opt.load_state_dict(s["opt"]), sched.load_state_dict(s["sched"])
        hist, vals, start, t0 = s["hist"], s["val"], s["it"], time.time() - s["seconds"]
        best_val, best_it, bad = s["best_val"], s["best_it"], s["bad"]
        batches.rng.bit_generator.state = s["rng"]
        _set_rng_states(s["rng_torch"], generators)
        log(f"  resumed at it {start} ({s['seconds']:.0f}s so far)")
    elif os.path.exists(metrics):
        os.remove(metrics)
    budget = time.time() + 3600.0 * cfg.max_hours if cfg.max_hours > 0 else None
    tick = time.time()

    def row(it, val=None):
        new = not os.path.exists(metrics)
        with open(metrics, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["it", "seconds", "lr", "train_loss", "val_loss", "train_parts", "val_parts"])
            v = val if isinstance(val, dict) else ({"loss": val} if val is not None else {})
            parts = {k: float(np.mean([h[k] for h in hist[-cfg.log_every:]])) for k in hist[-1] if k != "loss"} \
                if hist else {}
            w.writerow([it, f"{time.time() - t0:.0f}", f"{opt.param_groups[0]['lr']:.3e}",
                        f"{_mean(hist, cfg.log_every):.6f}" if hist else "", f"{v['loss']:.6f}" if v else "",
                        json.dumps(parts), json.dumps({k: q for k, q in v.items() if k != "loss"}) if v else ""])

    def validate(it):
        nonlocal best_val, best_it, bad
        model.eval()
        with torch.no_grad():
            v = val_fn()
        model.train()
        v = v if isinstance(v, dict) else {"loss": float(v)}
        vals.append((it, v))
        better = v["loss"] < best_val
        if better:
            best_val, best_it, bad = v["loss"], it, 0
            torch.save(model.state_dict(), ckpt + ".best.tmp")
            os.replace(ckpt + ".best.tmp", ckpt + ".best")
        else:
            bad += 1
        row(it, v)
        log(f"  val it {it:5d} " + " ".join(f"{k} {q:.4f}" for k, q in v.items())
            + (" *" if better else f" (best {best_val:.4f} at {best_it})"))
        return better

    def save(it, rng_state):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "hist": hist, "val": vals, "it": it, "seconds": time.time() - t0, "rng": rng_state,
                    "rng_torch": _rng_states(generators), "best_val": best_val, "best_it": best_it, "bad": bad},
                   part + ".tmp")
        os.replace(part + ".tmp", part)

    model.train()
    stopped = "iters"
    for it in range(start, cfg.iters):
        rng_state, batch = next(batches)
        if it > start and it % cfg.ckpt_every == 0:
            save(it, rng_state)
        if (budget is not None and time.time() > budget) or (cfg.max_steps and it - start >= cfg.max_steps):
            save(it, rng_state)
            stopped = "budget"
            log(f"  budget of this invocation reached at it {it}: state saved, rerun the same command to resume")
            break
        with amp:
            out = loss_fn(batch)
        loss, parts = out if isinstance(out, tuple) else (out, {})
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.clip > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.clip)
        opt.step()
        sched.step()
        if after_step is not None:
            after_step()
        hist.append({"loss": loss.item(), **parts})
        logged = False
        if (it + 1) % cfg.log_every == 0:
            parts_s = " ".join(f"{k} {float(np.mean([h[k] for h in hist[-cfg.log_every:]])):.4f}" for k in parts)
            log(f"  it {it + 1:5d} loss {_mean(hist, cfg.log_every):.4f} {parts_s} {time.time() - t0:.0f}s "
                f"({(time.time() - tick) / cfg.log_every:.2f} s/it)")
            tick = time.time()
            logged = True
        if val_fn is not None and cfg.val_every and (it + 1) % cfg.val_every == 0 and it + 1 < cfg.iters:
            validate(it + 1)
            if cfg.patience and bad >= cfg.patience:
                stopped = "patience"
                log(f"  no improvement in {cfg.patience} validations: stopped at it {it + 1}")
                break
        elif logged and not (val_fn is not None and it + 1 == cfg.iters):  # the final validation writes it
            row(it + 1)
    batches.close()
    out = {"hist": hist, "val": vals, "best_it": best_it, "best_val": best_val if best_it >= 0 else None,
           "stopped": stopped, "seconds": time.time() - t0, "start_it": start}
    if stopped == "budget":
        return out
    if val_fn is not None:
        validate(len(hist))
        if cfg.select == "best" and os.path.exists(ckpt + ".best"):
            model.load_state_dict(torch.load(ckpt + ".best"))
            log(f"  selected the best validation state (it {best_it}, {best_val:.4f})")
    model.eval()
    if os.path.exists(part):
        os.remove(part)
    return out
