"""pgcfm/runs.py: the shared training loop resumes exactly, selects the best validation state, stops on
patience and on the wall-clock budget; prefetching does not change the batches; the memory-mapped year
cache equals the in-RAM loader."""

import os

import netCDF4 as nc
import numpy as np
import torch

from pgcfm import contract
from pgcfm.runs import FitConfig, Prefetch, YearArray, fit, load_years, tendency_std


def _problem(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40)
    rng = np.random.default_rng(seed)
    w_true = torch.tensor([1.0, -2.0, 0.5, 0.0])

    def picks(r):  # every random draw of a step happens here (the caller's thread, the checkpointed rng)
        return r.standard_normal((8, 4)).astype(np.float32), r.standard_normal(8).astype(np.float32)

    def gather(p):
        x, noise = (torch.from_numpy(q) for q in p)
        return x, x @ w_true + 0.1 * noise
    xv = torch.from_numpy(np.random.default_rng(5).standard_normal((64, 4)).astype(np.float32))
    yv = xv @ w_true

    def loss_fn(b):
        x, y = b
        return ((model(x).squeeze(-1) - y) ** 2).mean(), {"mse": float(((model(x).squeeze(-1) - y) ** 2).mean())}

    def val_fn():
        return ((model(xv).squeeze(-1) - yv) ** 2).mean().item()

    return model, opt, sched, rng, picks, gather, loss_fn, val_fn


def _run(tmp_path, iters, name, **kw):
    model, opt, sched, rng, picks, gather, loss_fn, val_fn = _problem()
    cfg = FitConfig(iters=iters, ckpt_every=5, log_every=5, val_every=kw.pop("val_every", 10), **kw)
    r = fit(model, model.parameters(), opt, sched, Prefetch(picks, gather, rng, 2), loss_fn, cfg,
            str(tmp_path / name), val_fn=val_fn, log=lambda s: None)
    return model, r


def test_fit_resumes_to_the_same_result(tmp_path):
    _, full = _run(tmp_path, 40, "full")
    model, opt, sched, rng, picks, gather, loss_fn, val_fn = _problem()
    cfg = FitConfig(iters=40, ckpt_every=5, log_every=5, val_every=10, max_steps=17)
    ck = str(tmp_path / "part")
    r1 = fit(model, model.parameters(), opt, sched, Prefetch(picks, gather, rng, 2), loss_fn, cfg, ck,
             val_fn=val_fn, log=lambda s: None)
    assert r1["stopped"] == "budget" and len(r1["hist"]) == 17 and os.path.exists(ck + ".train")
    model, opt, sched, rng, picks, gather, loss_fn, val_fn = _problem(seed=3)  # other seed: the state must win
    cfg = FitConfig(iters=40, ckpt_every=5, log_every=5, val_every=10)
    r2 = fit(model, model.parameters(), opt, sched, Prefetch(picks, gather, rng, 2), loss_fn, cfg, ck,
             val_fn=val_fn, log=lambda s: None)
    assert r2["start_it"] > 0 and r2["stopped"] == "iters"
    assert [h["loss"] for h in r2["hist"]] == [h["loss"] for h in full["hist"]]
    assert r2["val"] == full["val"] and not os.path.exists(ck + ".train")


def test_prefetch_depth_does_not_change_the_batches(tmp_path):
    hs, nxt = [], []
    for depth in (0, 3):
        model, opt, sched, rng, picks, gather, loss_fn, val_fn = _problem()
        r = fit(model, model.parameters(), opt, sched, Prefetch(picks, gather, rng, depth), loss_fn,
                FitConfig(iters=20, ckpt_every=100, log_every=5), str(tmp_path / f"p{depth}"), log=lambda s: None)
        hs.append([h["loss"] for h in r["hist"]])
        nxt.append(rng.standard_normal(3).tolist())  # the stream after training does not see the prefetched picks
    assert hs[0] == hs[1] and nxt[0] == nxt[1]


def test_best_state_is_selected_and_patience_stops(tmp_path):
    model, r = _run(tmp_path, 40, "best", patience=0)
    assert r["best_it"] >= 10 and os.path.exists(str(tmp_path / "best.best"))
    best = torch.load(str(tmp_path / "best.best"))
    assert all(torch.equal(best[k], v) for k, v in model.state_dict().items())
    assert os.path.exists(str(tmp_path / "best_metrics.csv"))
    model, r = _run(tmp_path, 10000, "stop", patience=2, val_every=5)
    assert r["stopped"] == "patience" and len(r["hist"]) < 10000


def _tiny_kai(path, T=12, C=contract.N_CHANNELS, H=4, W=6, seed=0):
    rng = np.random.default_rng(seed)
    with nc.Dataset(path, "w") as d:
        d.createDimension("time", T), d.createDimension("channel", C)
        d.createDimension("latitude", H), d.createDimension("longitude", W)
        v = d.createVariable("fields", "f4", ("time", "channel", "latitude", "longitude"))
        v[:] = rng.standard_normal((T, C, H, W)).astype(np.float32) * 3 + 1


def test_year_cache_equals_the_in_ram_loader(tmp_path):
    kai = tmp_path / "kai"
    kai.mkdir()
    _tiny_kai(str(kai / "2001.nc")), _tiny_kai(str(kai / "2002.nc"), seed=1)
    mean, std = np.full(contract.N_CHANNELS, 1.0, np.float32), np.full(contract.N_CHANNELS, 3.0, np.float32)
    ram = load_years(str(kai), [2001, 2002], mean, std)
    mm = load_years(str(kai), [2001, 2002], mean, std, cache=str(tmp_path / "cache"))
    assert isinstance(mm[2001], YearArray) and len(mm[2001]) == 12
    dyn = contract.DYNAMIC_CHANNELS
    assert torch.equal(ram[2001][3:7], mm[2001][3:7])
    assert torch.equal(ram[2002][2:5, dyn], mm[2002][2:5, dyn])
    assert torch.equal(ram[2001][4, contract.STATIC_CHANNELS], mm[2001][4, contract.STATIC_CHANNELS])
    assert torch.allclose(tendency_std(ram, [2001, 2002], dyn), tendency_std(mm, [2001, 2002], dyn, chunk=5))
    try:
        load_years(str(kai), [2001], mean, std * 2, cache=str(tmp_path / "cache"))
    except ValueError:
        pass
    else:
        raise AssertionError("a cache with other statistics must be refused")
