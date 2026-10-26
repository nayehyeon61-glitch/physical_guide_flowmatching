"""Step 10: comparison models from the owner's references, adopted from their code (search-first).

    U-Cast        Rühling Cachay et al. (ICML 2026), github.com/Rose-STL-Lab/u-cast, vendored unchanged
                  in ``third_party/ucast``: DhariwalUNet + MC dropout, stage 1 weighted MAE, stage 2
                  fair CRPS with M = 2 dropout members, stage 3 deep ensemble of stage 2 runs,
                  Muon + AdamW (``HybridOptim``), linear warmup + cosine schedule.
    Scher+Messori "Ensemble methods for neural network-based weather forecasts" (JAMES 2021),
                  github.com/sipposip/ensemble-neural-network-weather-forecasts (GPL-3, so the
                  algorithms are re-written here, not copied): random initial perturbations
                  re-standardized over members, singular-vector perturbations of the network
                  Jacobian (truncated Gaussian coefficients, +/- pairs), dropout ensembles with a
                  tuned rate, and independently trained networks ("multitrain", ``netens``).
    dduq          Bülte et al. (AIES 2025), github.com/cbuelt/dduq: random field perturbations
                  (Magnusson et al. 2009: difference of two random same-season states scaled by
                  alpha / total energy, ``tke`` and constants as upstream) and the post-hoc
                  distributional regression network (DRN: forecast features + grid point
                  embedding -> Gaussian (mu, sigma), Gaussian CRPS loss).

What is ours is only the glue to the KAI contract: inputs x_k (66 dynamic channels) + orography +
calendar (U-Cast's static/forcing fields), one input state (U-Cast uses two), the output is the
next state through U-Cast's residual normalization, and the lat-lon field is transposed to
U-Cast's (lon, lat) layout so its circular padding acts on longitude.
"""

import contextlib
import math
import os
import sys
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import contract
from .cfm import area_weights, calendar_features

_UCAST = os.path.join(os.path.dirname(__file__), "..", "third_party", "ucast")
if _UCAST not in sys.path:
    sys.path.insert(0, os.path.abspath(_UCAST))
from src.models.modules.padding import pad_before_conv2d_circular_height_only  # noqa: E402
from src.models.networks_edm import DhariwalUNet  # noqa: E402
from src.optimization.hybrid_optim import HybridOptim  # noqa: E402
from src.optimization.losses import get_loss  # noqa: E402
from src.optimization.lr_scheduler import LinearWarmupCosineAnnealingLR  # noqa: E402
from src.optimization.muon import Muon, get_muon_momentum  # noqa: E402

N_COND = 1 + 2  # orography + calendar sin/cos


@contextlib.contextmanager
def circular_lon_padding():
    """U-Cast's ``set_global_padding_mode("circular_height_only")``, scoped to one forward."""
    orig = F.conv2d
    F.conv2d = lambda input, weight, padding=0, **kw: pad_before_conv2d_circular_height_only(
        input, weight, padding, torch_func=orig, **kw)
    try:
        yield
    finally:
        F.conv2d = orig


class UCastStep(nn.Module):
    """x_{k+1} = x_k + r * DhariwalUNet([x_k, orography, calendar]) on the (lon, lat) layout.
    ``r``: per-channel std of the 24 h residual (U-Cast's residual normalization)."""

    def __init__(self, n_lat: int, n_lon: int, residual_std: torch.Tensor, model_channels: int = 32,
                 channel_mult=(1, 2, 2), num_blocks: int = 2, dropout: float = 0.1, loss_function: str = "wmae",
                 members: int = 2):
        super().__init__()
        C = contract.N_DYNAMIC
        self.net = DhariwalUNet(model_channels=model_channels, channel_mult=list(channel_mult),
                                num_blocks=num_blocks, dropout=dropout, num_input_channels=C,
                                num_output_channels=C, num_conditional_channels=N_COND,
                                spatial_shape=(n_lon, n_lat), loss_function=loss_function,
                                num_training_ensemble_members=members)
        self.register_buffer("rstd", residual_std.view(1, C, 1, 1).float())
        self.register_buffer("w_area", area_weights(n_lat).T[None, None])  # (1, 1, 1, lat) on (lon, lat)

    def cond(self, x, static, doy_next):
        B, _, H, W = x.shape
        cal = calendar_features(doy_next).to(x)[:, :, None, None].expand(B, 2, H, W)
        return torch.cat([static, cal], 1).transpose(-1, -2)

    def forward(self, x, static, doy_next):
        with circular_lon_padding():
            r = self.net(x.transpose(-1, -2), dynamical_condition=self.cond(x, static, doy_next))
        return x + self.rstd * r.transpose(-1, -2)

    def loss(self, x, y, static, doy_next, criterion: str):
        """U-Cast's ``BaseModel.get_loss`` on normalized residual targets ("wmae" or "wcrps")."""
        crit = get_loss(criterion)
        crit.weights = self.w_area
        target = ((y - x) / self.rstd).transpose(-1, -2)
        with circular_lon_padding():
            if criterion == "wcrps":
                n = self.net.hparams.num_training_ensemble_members
                pred = self.net.ensemble_forward(x.transpose(-1, -2), n_ens_mems=n,
                                                 dynamical_condition=self.cond(x, static, doy_next))
            else:
                pred = self.net(x.transpose(-1, -2), dynamical_condition=self.cond(x, static, doy_next))
        return crit(pred, target)

    def set_dropout(self, on: bool, p: Optional[float] = None):
        (self.net.enable_inference_dropout if on else self.net.disable_inference_dropout)()
        if p is not None:
            for m in self.net.modules():
                if isinstance(m, nn.Dropout):
                    m.p = p


def residual_std(loader, n_batches: int = 50) -> torch.Tensor:
    acc, n = 0.0, 0
    for i, b in enumerate(loader):
        if i == n_batches:
            break
        d = b["targets"][:, 0] - b["x0"]
        acc = acc + d.pow(2).mean((0, 2, 3))
        n += 1
    return (acc / n).sqrt().clamp_min(1e-3)


def ucast_optimizer(net: nn.Module, lr: float, weight_decay: float, muon_lr: float, muon_wd: float,
                    momentum: float = 0.95):
    """``UCastExperiment._get_optim``: Muon for the hidden weight matrices (ndim >= 2, not the
    output conv), AdamW for the rest; no weight decay on biases / norms."""
    no_decay = {k for k, _ in net.named_parameters() if "weight" not in k or "norm" in k}
    adam, muon = {}, []
    for name, p in net.named_parameters():
        if p.ndim >= 2 and "out_conv" not in name:
            muon.append(p)
            continue
        wd = 0.0 if name in no_decay else weight_decay
        adam.setdefault(wd, []).append(p)
    opt = torch.optim.AdamW([{"params": v, "weight_decay": k} for k, v in adam.items()], lr=lr,
                            betas=(0.9, 0.95), eps=1e-8)
    return HybridOptim([opt, Muon(muon, lr=muon_lr, weight_decay=muon_wd, momentum=momentum)])


def train_ucast(step: UCastStep, batches, iters: int, criterion: str, lr: float, weight_decay: float,
                muon_lr: float, muon_wd: float, warmup_frac: float = 0.1, log=print, log_every: int = 100,
                ckpt: Optional[str] = None):
    """U-Cast stage 1 (``criterion="wmae"``, 100 epochs upstream) or stage 2 (``"wcrps"``, 8 epochs).
    Warmup + cosine per step, Muon momentum schedule, gradient clip 1, dropout on (training)."""
    opt = ucast_optimizer(step.net, lr, weight_decay, muon_lr, muon_wd)
    warm = max(2, int(warmup_frac * iters))
    sched = LinearWarmupCosineAnnealingLR(opt, warmup_steps=warm, max_steps=iters, warmup_start_lr=1e-8, eta_min=1e-8)
    start, hist = 0, []
    if ckpt and os.path.exists(ckpt):
        s = torch.load(ckpt)
        step.load_state_dict(s["step"])
        for o, sd in zip(opt.optimizers, s["opt"]):
            o.load_state_dict(sd)
        sched.load_state_dict(s["sched"])
        start, hist = s["it"], s["hist"]
        log(f"  [ucast {criterion}] resumed at it {start}")
    step.train()
    for it in range(start, iters):
        if ckpt and it > start and it % 200 == 0:
            torch.save({"step": step.state_dict(), "opt": [o.state_dict() for o in opt.optimizers],
                        "sched": sched.state_dict(), "it": it, "hist": hist}, ckpt)
        b = next(batches)
        loss = step.loss(b["x0"], b["targets"][:, 0], b["static"], b["doy"][:, 1], criterion)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(step.parameters(), 1.0)
        mom = get_muon_momentum(it, iters, muon_warmup_steps=warm, muon_cooldown_steps=max(1, iters // 20))
        for g in opt.optimizers[-1].param_groups:
            g["momentum"] = mom
        opt.step()
        sched.step()
        hist.append(loss.item())
        if (it + 1) % log_every == 0 or it + 1 == iters:
            log(f"  [ucast {criterion}] it {it + 1} loss {np.mean(hist[-log_every:]):.4f}", flush=True)
    step.eval()
    if ckpt and os.path.exists(ckpt):
        os.remove(ckpt)
    return hist


# -- ensemble generation --------------------------------------------------------------------

@torch.no_grad()
def member_rollout(steps: Sequence[Callable], x0m: torch.Tensor, static: torch.Tensor, doy: torch.Tensor,
                   dropout_p: Optional[float] = None) -> torch.Tensor:
    """x0m (B, M, C, H, W) initial members; member m is advanced by ``steps[m % len(steps)]`` (one net
    for IC / dropout ensembles, K nets for multitrain / deep ensembles). Returns (B, M, K, C, H, W)."""
    B, M = x0m.shape[:2]
    K = doy.shape[1] - 1
    x = x0m.clone()
    out = []
    for k in range(K):
        nxt = torch.empty_like(x)
        for j, f in enumerate(steps):
            idx = list(range(j, M, len(steps)))
            if dropout_p is not None:
                f.set_dropout(dropout_p > 0, dropout_p)
            xs = x[:, idx].flatten(0, 1)
            st = static[:, None].expand(B, len(idx), *static.shape[1:]).flatten(0, 1)
            nxt[:, idx] = f(xs, st, doy[:, k + 1].repeat_interleave(len(idx))).view(B, len(idx), *x.shape[2:])
        x = nxt
        out.append(x)
    return torch.stack(out, 2)


def scher_rand(x0: torch.Tensor, M: int, scale: float, gen: torch.Generator) -> torch.Tensor:
    """Scher & Messori ``create_pertubed_states_rand``: Gaussian perturbations re-standardized over
    the members at every grid point and channel (exact zero mean, std ``scale``). (B,C,H,W) -> (B,M,C,H,W)."""
    p = torch.randn((x0.shape[0], M, *x0.shape[1:]), generator=gen)
    p = (p - p.mean(1, keepdim=True)) / p.std(1, unbiased=False, keepdim=True)
    return x0[:, None] + scale * p


@torch.enable_grad()
def jacobian_svecs(step: UCastStep, x: torch.Tensor, static: torch.Tensor, doy_next: torch.Tensor, k: int,
                   iters: int = 6, seed: int = 0) -> torch.Tensor:
    """Leading right singular vectors of the one-day Jacobian of the deterministic net at x (C,H,W)
    (Scher & Messori ``compute_svs``: they take ``scipy.sparse.linalg.svds`` of the full Jacobian;
    here subspace iteration on J^T J with Jacobian-vector products: the same leading vectors).
    Returns (k, C*H*W)."""
    step.set_dropout(False)
    n = x.numel()
    xs = x[None].expand(k, *x.shape).contiguous()
    st, d = static[None].expand(k, *static.shape), doy_next.expand(k)
    f = lambda z: step(z, st, d)  # noqa: E731
    g = torch.Generator().manual_seed(seed)
    V = torch.linalg.qr(torch.randn(n, k, generator=g))[0]
    for _ in range(iters):
        # torch.autograd.functional (double backward): U-Cast's AttentionOp has no functorch rule
        JV = torch.autograd.functional.jvp(f, xs, V.T.reshape(xs.shape))[1]
        JtJV = torch.autograd.functional.vjp(f, xs, JV)[1].reshape(k, n).T
        V = torch.linalg.qr(JtJV.detach())[0]
    return V.T


def scher_svd(x0: torch.Tensor, svecs: torch.Tensor, M: int, scale: float, gen: torch.Generator) -> torch.Tensor:
    """Scher & Messori ``create_pertubed_states_svd``: coefficients ~ N(0, 1) truncated at +/-3, times
    ``scale``, on the leading singular vectors; +/- pairs. x0 (B,C,H,W), svecs (B, k, n)."""
    B = x0.shape[0]
    p = torch.randn((B, M // 2, svecs.shape[1]), generator=gen).clamp(-3, 3) * scale
    pert = torch.einsum("bmk,bkn->bmn", p, svecs).view(B, M // 2, *x0.shape[1:])
    return torch.stack([x0[:, None] + pert, x0[:, None] - pert], 2).flatten(1, 2)


# dduq Pangu-Weather/constants.py
CC = {"T_r": 300.0, "P_r": 800.0, "g": 9.80665, "Rd": 8.314462618 / 28.96546e-3}
CC["Cp_d"] = 1.4 * CC["Rd"] / 0.4


def dduq_tke(dphys: np.ndarray) -> float:
    """dduq ``tke`` of a de-normalized state difference (67 KAI channels, lat, lon): area-weighted mean
    of (u^2 + v^2 + Cp/T_r T^2) / (2 g) integrated over pressure (Pa), + Rd T_r P_r / (2 g) (ln sp)^2.
    As upstream, ``ln`` is applied to the surface-pressure *difference* (``np.log(ds_sfc.SP)``), so
    its sign is dropped here with abs."""
    ci, lev = contract.CHANNEL_INDEX, np.array(contract.LEVELS_HPA, dtype=float) * 100.0
    H = dphys.shape[-2]
    w = np.cos(np.deg2rad(np.linspace(-90, 90, H)))[:, None]
    wm = lambda f: float((f * w).sum() / (w.sum() * f.shape[-1]))  # noqa: E731
    e1 = []
    for p in contract.LEVELS_HPA:
        u, v, t = dphys[ci[f"u{p}"]], dphys[ci[f"v{p}"]], dphys[ci[f"t{p}"]]
        e1.append(wm((u ** 2 + v ** 2 + CC["Cp_d"] / CC["T_r"] * t ** 2) / CC["g"] * 0.5))
    order = np.argsort(lev)
    e1 = np.trapezoid(np.array(e1)[order], lev[order])
    sp = np.abs(dphys[ci["sp"]]) + 1e-6
    e2 = wm(0.5 * CC["Rd"] * CC["T_r"] * CC["P_r"] / CC["g"] * np.log(sp) ** 2)
    return float(e1 + e2)


class RandomFieldPerturber:
    """dduq ``random_field_perturbation``: per member, the difference of two random states of the
    same season from two different years (upstream: same month), times alpha / tke(difference).
    ``pool``: normalized states (N, 66, H, W) with their year and day of year; ``mean``/``std``
    de-normalize for the energy."""

    def __init__(self, pool: torch.Tensor, years: np.ndarray, doys: np.ndarray, static: torch.Tensor,
                 mean: np.ndarray, std: np.ndarray, window: int = 15):
        self.pool, self.years, self.doys, self.window = pool, years, doys, window
        self.static, self.mean, self.std = static, mean, std
        self.dyn = np.array(contract.DYNAMIC_CHANNELS)

    def unit(self, doy: int, rng: np.random.Generator) -> torch.Tensor:
        """One perturbation with alpha = 1, in normalized units."""
        dist = np.abs((self.doys - doy + 183) % 366 - 183)
        near = dist <= max(self.window, np.sort(dist)[min(len(dist) - 1, 8 * self.window)])  # pool may miss a season
        i = rng.choice(np.flatnonzero(near))
        j = rng.choice(np.flatnonzero(near & (self.years != self.years[i])))
        d = (self.pool[i] - self.pool[j]).numpy()
        full = np.zeros((contract.N_CHANNELS, *d.shape[1:]))
        full[self.dyn] = d * self.std[self.dyn, None, None]
        return torch.from_numpy(d / dduq_tke(full)).float()

    def __call__(self, x0: torch.Tensor, doy0: torch.Tensor, M: int, alpha: float, seed: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        p = torch.stack([torch.stack([self.unit(int(doy0[b]), rng) for _ in range(M)]) for b in range(len(x0))])
        return x0[:, None] + alpha * p


# -- dduq DRN (post-hoc) ------------------------------------------------------------------

def crps_normal(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """dduq ``crps_cost_function`` (sigma enters through its square, so its sign is free)."""
    s = sigma.abs() + 1e-6
    z = (y - mu) / s
    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return s * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))


class DRN(nn.Module):
    """dduq ``build_emb_model`` for one lead: [forecast features, grid point embedding] -> Dense(hidden,
    relu) -> Dense(2) per target. Upstream fits one model per variable and lead; here one model per
    lead predicts (mu, sigma) of every channel from the forecast of every channel at the point."""

    def __init__(self, C: int, n_points: int, emb: int = 5, hidden: int = 512):
        super().__init__()
        self.emb = nn.Embedding(n_points, emb)
        self.mlp = nn.Sequential(nn.Linear(C + emb, hidden), nn.ReLU(), nn.Linear(hidden, 2 * C))
        self.register_buffer("lo", torch.zeros(C))
        self.register_buffer("hi", torch.ones(C))

    def forward(self, f: torch.Tensor, pid: torch.Tensor):
        """f (N, C) forecast at a grid point, pid (N,) -> mu, sigma (N, C)."""
        z = (f - self.lo) / (self.hi - self.lo)  # upstream: min-max scaling of the features
        out = self.mlp(torch.cat([z, self.emb(pid)], -1))
        mu, sigma = out.chunk(2, -1)
        return f + mu, sigma  # the network learns the correction of the forecast


def _points(f: torch.Tensor):
    """(B, C, H, W) -> (B*H*W, C) features and grid point ids."""
    B, C, H, W = f.shape
    return f.permute(0, 2, 3, 1).reshape(-1, C), torch.arange(H * W).repeat(B)


def fit_drn(fc: torch.Tensor, obs: torch.Tensor, n_valid: int, epochs: int = 10, lr: float = 0.01,
            batch: int = 1024, patience: int = 3, seed: int = 0) -> Dict[int, DRN]:
    """fc, obs (B, K, C, H, W): deterministic forecasts and truth of the fitting period; the last
    ``n_valid`` initial states are the early-stopping split (upstream: validation year, patience 3)."""
    torch.manual_seed(seed)
    B, K, C, H, W = fc.shape
    models = {}
    for k in range(K):
        Xt, Pt = _points(fc[:B - n_valid, k])
        Yt = _points(obs[:B - n_valid, k])[0]
        Xv, Pv = _points(fc[B - n_valid:, k])
        Yv = _points(obs[B - n_valid:, k])[0]
        m = DRN(C, H * W)
        m.lo.copy_(Xt.min(0).values)
        m.hi.copy_(torch.maximum(Xt.max(0).values, m.lo + 1e-3))
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        best, best_sd, bad = float("inf"), None, 0
        for ep in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), batch):
                j = perm[i:i + batch]
                mu, s = m(Xt[j], Pt[j])
                loss = crps_normal(mu, s, Yt[j]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                v = crps_normal(*m(Xv, Pv), Yv).mean().item()
            if v < best:
                best, best_sd, bad = v, {kk: t.clone() for kk, t in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        m.load_state_dict(best_sd)
        models[k] = m.eval()
    return models


@torch.no_grad()
def drn_members(models: Dict[int, DRN], fc: torch.Tensor, M: int) -> torch.Tensor:
    """Gaussian of every point -> M equidistant quantiles (levels (i - 0.5) / M) as members.
    fc (B, K, C, H, W) -> (B, M, K, C, H, W). The members only carry the marginals."""
    B, K, C, H, W = fc.shape
    q = torch.distributions.Normal(0, 1).icdf((torch.arange(M) + 0.5) / M)
    out = torch.empty(B, M, K, C, H, W)
    for k, m in models.items():
        X, P = _points(fc[:, k])
        mu, s = m(X, P)
        mu = mu.view(B, H, W, C).permute(0, 3, 1, 2)
        s = (s.abs() + 1e-6).view(B, H, W, C).permute(0, 3, 1, 2)
        out[:, :, k] = mu[:, None] + q.view(1, M, 1, 1, 1) * s[:, None]
    return out
