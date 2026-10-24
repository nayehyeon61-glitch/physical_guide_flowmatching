"""Step 7: DCT latent PG-CFM, two stages.

Built from the owner's ``dct_encoder.py`` sketch (DCT -> MLP encoder with a
condition -> latent; MLP decoder -> IDCT), fixed and wired into the existing
pieces:

  * DCT: orthonormal 2-D DCT-II per channel over (lat, lon) from torch-dct
    (https://github.com/zh217/torch-dct), keeping the ``k_lat x k_lon``
    lowest modes. Truncation is the low-pass filter; everything above it is
    gone for good (there is no learned skip connection).
  * MLPs are built like Raissi's PINNs (``physics.pinn.PhysicsInformedNN``:
    Xavier init, tanh). The sketch used ``F.sinh``, which does not exist and
    would grow exponentially as ``torch.sinh``; tanh is Raissi's activation.
    The last encoder layer is linear, so the latent is not squashed to
    [-1, 1] while the CFM noise is N(0, I).

Stage 1 (``train_stage1``): encoder -> latent PINN -> decoder, jointly:
    reconstruction of x_k and x_{k+1}, decoded prediction D(P(E(x_k))) vs
    x_{k+1}, latent consistency P(E(x_k)) vs E(x_{k+1}), and lambda_P x the
    time-integrated primitive-equation residual (``coupling.PhysicsGuide``,
    linear bridge, no closure) of the decoded pair (D(z_k), D(P(z_k))). The
    physics gradient is capped at ``grad_ratio`` x the data gradient (the
    step 4 W2 fix). Then the latent PINN P is frozen.
Serial variant (``coupling="serial"``, ``train_serial``, owner 2026-10-23):
    DCT encoder -> latent CFM makes members A, B, C, ... (z_k + s * delta^m) -> a
    trainable PINN Q_phi corrects each member -> decoder/IDCT. CFM and Q_phi are
    trained in the same step on L_CFM + decoded member CRPS + capped lambda_P x
    physics, gradient through the Heun sampler into the CFM.
Stage 2 (``train_stage2``): latent CFM (torchcfm path + torchcfm MLP)
    generates the deviation of z_{k+1} from the frozen PINN step,
        z_{k+1} = P(z_k) + s * delta,   delta ~ CFM(eps | z_k, P(z_k), c),
    s = per-dimension std of E(x_{k+1}) - P(E(x_k)) on training pairs. The
    ensemble comes from fresh eps ~ N(0, I) in the latent at every transition;
    a member is decoded to the grid, which cannot hold anything above the DCT
    cut. Encoder/decoder are either frozen or retrained (``mode``);
    ``finetune_decoder`` is the low-LR decoder ablation on decoded CRPS.

``LatentPGCFM.sample`` has the ``CFMTransition.sample`` interface, so
``cfm.rollout`` (member-wise Markov chain x_k -> x_{k+1} -> ... in grid space,
re-encoded every day), ``coupling.deterministic_rollout`` and the step 4/6
evaluation run on it unchanged.
"""

import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.models import MLP

from . import contract, metrics
from .cfm import TISR_DYN, area_weights, calendar_features, heun
from .physics.pinn import PhysicsInformedNN

ACTS = {"tanh": torch.tanh, "silu": F.silu, "sin": torch.sin}


def dct(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2-D DCT-II over the last two dims (lat, lon)."""
    return torch_dct.dct_2d(x, norm="ortho")


def idct(c: torch.Tensor) -> torch.Tensor:
    """Inverse of ``dct`` (orthonormal DCT-III)."""
    return torch_dct.idct_2d(c, norm="ortho")


def mlp(layers: Sequence[int]) -> nn.ModuleList:
    """Raissi's ``initialize_NN``: Xavier (truncated normal) weights, zero biases."""
    mods = nn.ModuleList()
    for i in range(len(layers) - 1):
        lin = nn.Linear(layers[i], layers[i + 1])
        PhysicsInformedNN.xavier_init(lin.weight)
        nn.init.zeros_(lin.bias)
        mods.append(lin)
    return mods


def run_mlp(layers: nn.ModuleList, h: torch.Tensor, act) -> torch.Tensor:
    for lin in layers[:-1]:
        h = act(lin(h))
    return layers[-1](h)


class DCTTruncation(nn.Module):
    """(B, C, H, W) <-> standardized low-mode DCT coefficients (B, C * k_lat * k_lon)."""

    def __init__(self, channels: int, n_lat: int, n_lon: int, k_lat: int, k_lon: int):
        super().__init__()
        if not (0 < k_lat <= n_lat and 0 < k_lon <= n_lon):
            raise ValueError("need 0 < k <= grid size")
        self.C, self.n_lat, self.n_lon, self.k_lat, self.k_lon = channels, n_lat, n_lon, k_lat, k_lon
        self.register_buffer("mean", torch.zeros(self.dim))
        self.register_buffer("std", torch.ones(self.dim))

    @property
    def dim(self) -> int:
        return self.C * self.k_lat * self.k_lon

    def coeffs(self, x: torch.Tensor) -> torch.Tensor:
        return dct(x)[..., :self.k_lat, :self.k_lon].flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.coeffs(x) - self.mean) / self.std

    def inverse(self, a: torch.Tensor) -> torch.Tensor:
        c = (a * self.std + self.mean).view(a.shape[0], self.C, self.k_lat, self.k_lon)
        c = F.pad(c, (0, self.n_lon - self.k_lon, 0, self.n_lat - self.k_lat))
        return idct(c)

    def filter(self, x: torch.Tensor) -> torch.Tensor:
        """Projection onto the kept modes (the truncation floor, no learning)."""
        return self.inverse(self.forward(x))

    @torch.no_grad()
    def fit(self, xs) -> None:
        """Per-coefficient mean/std from a tensor (N, C, H, W) or an iterable of such batches."""
        n, s1, s2 = 0, 0.0, 0.0
        for x in ([xs] if torch.is_tensor(xs) else xs):
            c = self.coeffs(x.to(self.mean.device)).double()
            n, s1, s2 = n + c.shape[0], s1 + c.sum(0), s2 + (c ** 2).sum(0)
        mean = s1 / n
        var = (s2 - n * mean ** 2) / max(n - 1, 1)
        self.mean.copy_(mean.float())
        self.std.copy_(var.clamp_min(0).sqrt().float().clamp_min(1e-6))


class Encoder(nn.Module):
    """The sketch's encoder: [DCT(x), DCT(c), calendar] -> 256 -> 128 -> 64 -> latent."""

    def __init__(self, dct_x: DCTTruncation, dct_c: DCTTruncation, latent: int,
                 hidden: Sequence[int] = (256, 128, 64), act: str = "tanh"):
        super().__init__()
        self.dct_x, self.dct_c, self.act = dct_x, dct_c, ACTS[act]
        self.layers = mlp([dct_x.dim + dct_c.dim + 2, *hidden, latent])

    def forward(self, x: torch.Tensor, c: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.dct_x(x), self.dct_c(c), calendar_features(doy).to(x)], dim=-1)
        return run_mlp(self.layers, h, self.act)


class Decoder(nn.Module):
    """The sketch's decoder: latent -> 64 -> 128 -> 256 -> DCT coefficients -> IDCT."""

    def __init__(self, dct_x: DCTTruncation, latent: int, hidden: Sequence[int] = (64, 128, 256),
                 act: str = "tanh"):
        super().__init__()
        self.dct_x, self.act = dct_x, ACTS[act]
        self.layers = mlp([latent, *hidden, dct_x.dim])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.dct_x.inverse(run_mlp(self.layers, z, self.act))


class LatentPINN(nn.Module):
    """P_psi: z_{k+1} = z_k + MLP(z_k, calendar(k+1)), Raissi tanh MLP.

    The last layer starts at zero, so training starts from persistence.
    """

    def __init__(self, latent: int, hidden: Sequence[int] = (256, 256), act: str = "tanh"):
        super().__init__()
        self.act = ACTS[act]
        self.layers = mlp([latent + 2, *hidden, latent])
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, z: torch.Tensor, doy_next: torch.Tensor) -> torch.Tensor:
        return z + run_mlp(self.layers, torch.cat([z, calendar_features(doy_next).to(z)], -1), self.act)


class SerialPINN(nn.Module):
    """Q_phi for the serial coupling CFM -> PINN: z_{k+1} = z_cfm + MLP(z_cfm, z_k, calendar).

    Trained (not frozen) on decoded member CRPS + lambda_P x physics residual; the
    last layer starts at zero, so training starts from the CFM sample itself.
    """

    def __init__(self, latent: int, hidden: Sequence[int] = (256, 256), act: str = "tanh"):
        super().__init__()
        self.act = ACTS[act]
        self.layers = mlp([2 * latent + 2, *hidden, latent])
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, z: torch.Tensor, z0: torch.Tensor, doy_next: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, z0, calendar_features(doy_next).to(z)], -1)
        return z + run_mlp(self.layers, h, self.act)


@dataclass
class LatentConfig:
    k_lat: int = 8
    k_lon: int = 16
    latent: int = 64
    enc_hidden: tuple = (256, 128, 64)  # the sketch's widths
    dec_hidden: tuple = (64, 128, 256)
    pinn_hidden: tuple = (256, 256)
    act: str = "tanh"
    cfm_width: int = 256
    tisr_mode: str = "target"
    carry_highpass: bool = False  # copy the modes above the DCT cut from x_k (W1 fix)
    carry_tendency: bool = False  # x_{k+1} = x_k + D(z_{k+1}) - D(z_k): only the tendency passes the AE (step 11)
    coupling: str = "anchor"  # "anchor": z = P(z_k) + s * CFM; "serial": z = Q(z_k + s * CFM, z_k), Q trained


def _wmse(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return ((a - b) ** 2 * w).mean()


class LatentPGCFM(nn.Module):
    """Encoder -> frozen latent PINN + latent CFM -> decoder, as one 24 h transition."""

    def __init__(self, cfg: LatentConfig, n_lat: int, n_lon: int):
        super().__init__()
        if cfg.tisr_mode != "target":
            raise ValueError("latent model generates tisr (KAI baseline); 'forcing' not implemented")
        self.cfg, L = cfg, cfg.latent
        self.dct_x = DCTTruncation(contract.N_DYNAMIC, n_lat, n_lon, cfg.k_lat, cfg.k_lon)
        self.dct_c = DCTTruncation(1, n_lat, n_lon, cfg.k_lat, cfg.k_lon)
        self.encoder = Encoder(self.dct_x, self.dct_c, L, cfg.enc_hidden, cfg.act)
        self.decoder = Decoder(self.dct_x, L, cfg.dec_hidden, cfg.act)
        self.pinn = LatentPINN(L, cfg.pinn_hidden, cfg.act)
        if cfg.coupling not in ("anchor", "serial"):
            raise ValueError(cfg.coupling)
        self.serial = SerialPINN(L, cfg.pinn_hidden, cfg.act) if cfg.coupling == "serial" else None
        # torchcfm MLP; input [z_tau, z_k, P(z_k), calendar] + tau
        self.net = MLP(dim=3 * L + 2, out_dim=L, w=cfg.cfm_width, time_varying=True)
        self.fm = ConditionalFlowMatcher(sigma=0.0)
        self.register_buffer("res_std", torch.ones(L))
        self.register_buffer("w_area", area_weights(n_lat))
        self.register_buffer("gen_idx", torch.arange(contract.N_DYNAMIC))

    # -- pieces -------------------------------------------------------------
    def encode(self, x, static, doy):
        return self.encoder(x, static, doy)

    def decode(self, z, carry=None, z_carry=None):
        """D(z). With ``cfg.carry_highpass`` and ``carry`` = the state the latent came from
        (or advances), the modes above the DCT cut are copied from it: they persist instead of
        being lost (docs/step7_latent.md, weakness W1). With ``cfg.carry_tendency`` the output is
        carry + D(z) - D(z_carry) (z_carry = E(carry)): the autoencoder error cancels at z = z_carry,
        so a latent step that does nothing returns persistence (docs/step11_climode.md)."""
        out = self.decoder(z)
        if self.cfg.carry_tendency and carry is not None:
            if z_carry is None:
                raise ValueError("carry_tendency needs z_carry = E(carry)")
            return carry + out - self.decoder(z_carry)
        if self.cfg.carry_highpass and carry is not None:
            out = out + carry - self.dct_x.filter(carry)
        return out

    def anchor(self, z0, doy_next):
        """Where the CFM deviation starts: the frozen PINN step (anchor) or z_k (serial)."""
        return self.pinn(z0, doy_next) if self.serial is None else z0

    def condition(self, z0, zp, doy_next):
        return torch.cat([z0, zp, calendar_features(doy_next).to(z0)], dim=-1)

    def velocity(self, tau, z, cond):
        return self.net(torch.cat([z, cond, tau[:, None].to(z)], dim=-1))

    @torch.no_grad()
    def fit_dct(self, xs, statics) -> None:
        """Coefficient statistics; tensors or iterables of batches (large grids)."""
        self.dct_x.fit(xs)
        self.dct_c.fit(statics)

    @torch.no_grad()
    def fit_residual_scale(self, batches) -> None:
        r = []
        for b in batches:
            b = {k: v.to(self.res_std.device) for k, v in b.items()}
            x, y, st, d = b["x0"], b["targets"][:, 0], b["static"], b["doy"]
            z0 = self.encode(x, st, d[:, 0])
            r.append(self.encode(y, st, d[:, 1]) - self.anchor(z0, d[:, 1]))
        self.res_std.copy_(torch.cat(r).std(0).clamp_min(1e-4))

    # -- stage 1 ------------------------------------------------------------
    def stage1_terms(self, batch, guide=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        x, y, st, d = batch["x0"], batch["targets"][:, 0], batch["static"], batch["doy"]
        w = self.w_area
        z0, z1 = self.encode(x, st, d[:, 0]), self.encode(y, st, d[:, 1])
        x0r, x1r = self.decode(z0, x, z0), self.decode(z1, y, z1)
        zp = self.pinn(z0, d[:, 1])
        xp = self.decode(zp, x, z0)
        if self.cfg.carry_tendency:  # x0r = x, x1r = y exactly: reconstruct the tendency instead
            recon = _wmse(self.decoder(z1) - self.decoder(z0), y - x, w)
        else:
            recon = 0.5 * (_wmse(x0r, x, w) + _wmse(x1r, y, w))
        pred = _wmse(xp, y, w)
        lat = F.mse_loss(zp, z1.detach())
        terms = {"data": recon + pred + lat}
        logs = {"recon": recon.item(), "pred": pred.item(), "latent": lat.item()}
        if guide is not None:
            R, _ = guide.residuals(x0r, xp, st, d[:, 0], use_bridge=False, use_closure=False)
            terms["phys"] = guide.loss(R, guide.mask(x, y, st))
            logs["phys"] = terms["phys"].item()
        return terms, logs

    # -- stage 2 ------------------------------------------------------------
    def cfm_loss(self, batch, train_ae: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Latent CFM loss on delta = (E(x_{k+1}) - P(E(x_k))) / s; with ``train_ae`` the
        encoder/decoder get this loss + the stage 1 reconstruction (P stays frozen)."""
        x, y, st, d = batch["x0"], batch["targets"][:, 0], batch["static"], batch["doy"]
        with torch.set_grad_enabled(train_ae):
            z0, z1 = self.encode(x, st, d[:, 0]), self.encode(y, st, d[:, 1])
            zp = self.anchor(z0, d[:, 1])
            target = (z1 - zp) / self.res_std
        cond = self.condition(z0, zp, d[:, 1])
        eps = torch.randn_like(target)
        tau, zt, u = self.fm.sample_location_and_conditional_flow(eps, target)
        loss = ((self.velocity(tau, zt, cond) - u) ** 2).mean()
        logs = {"cfm": loss.item()}
        if train_ae:
            recon = 0.5 * (_wmse(self.decode(z0, x, z0), x, self.w_area) + _wmse(self.decode(z1, y, z1), y, self.w_area))
            loss = loss + recon
            logs["recon"] = recon.item()
        return loss, logs

    def sample_latent(self, x, static, doy_next, steps=8, generator=None, eps=None, refine=True):
        z0 = self.encode(x, static, (doy_next - 1) % 366)
        zp = self.anchor(z0, doy_next)
        cond = self.condition(z0, zp, doy_next)
        B, L = z0.shape[0], self.cfg.latent
        if eps is None:
            eps = torch.randn((B, L), generator=generator, device=x.device, dtype=x.dtype)
        elif eps.shape != (B, L):
            if eps.abs().max() > 0:  # grid-shaped noise only makes sense as the eps = 0 baseline
                raise ValueError(f"latent eps must be {(B, L)}")
            eps = x.new_zeros((B, L))
        delta = heun(lambda t, s: self.velocity(t, s, cond), eps, steps)
        z = zp + self.res_std * delta
        if self.serial is not None and refine:
            z = self.serial(z, z0, doy_next)
        return z

    def sample(self, x, static, doy_next, tisr_next=None, steps=8, generator=None, grad=False, eps=None):
        with torch.set_grad_enabled(grad):
            z0 = self.encode(x, static, (doy_next - 1) % 366) if self.cfg.carry_tendency else None
            return self.decode(self.sample_latent(x, static, doy_next, steps, generator, eps), x, z0)

    def decoder_crps_loss(self, batch, members: int = 2, steps: int = 8):
        """Decoded fair CRPS of ``members`` latent samples, gradient only into the decoder."""
        x, y, st, d = batch["x0"], batch["targets"][:, 0], batch["static"], batch["doy"]
        rep = lambda t: t.repeat(members, *([1] * (t.dim() - 1)))  # noqa: E731
        with torch.no_grad():
            z = self.sample_latent(rep(x), rep(st), rep(d[:, 1]), steps)
            z0 = self.encode(rep(x), rep(st), rep(d[:, 0])) if self.cfg.carry_tendency else None
        ens = self.decode(z, rep(x), z0).view(members, *x.shape)
        keep = [i for i in range(x.shape[1]) if i != TISR_DYN]
        loss = metrics.crps(ens[:, :, keep], y[:, keep]).mean()
        return loss, {"crps": loss.item()}


def serial_terms(model: "LatentPGCFM", batch, guide=None, members: int = 2, steps: int = 8,
                 joint: bool = True):
    """Serial coupling DCT encoder -> CFM members A, B, ... -> PINN Q_phi on each member -> decoder.

    Decoded fair CRPS over the ``members`` + the primitive-equation residual of every member
    transition (D(z_k), D(Q(z_cfm^m, z_k))). With ``joint`` the gradient of both terms also
    reaches the CFM through the differentiable Heun sampler (generation and PINN learn together);
    otherwise the CFM members are detached and only Q_phi learns from them.
    """
    x, y, st, d = batch["x0"], batch["targets"][:, 0], batch["static"], batch["doy"]
    rep = lambda t: t.repeat(members, *([1] * (t.dim() - 1)))  # noqa: E731
    xs, sts, dn = rep(x), rep(st), rep(d[:, 1])
    with torch.no_grad():
        z0 = model.encode(xs, sts, rep(d[:, 0]))
        x0r = model.decode(z0, xs, z0)
    with torch.set_grad_enabled(joint):
        zc = model.sample_latent(xs, sts, dn, steps, refine=False)
    xh = model.decode(model.serial(zc, z0, dn), xs, z0)
    keep = [i for i in range(x.shape[1]) if i != TISR_DYN]
    crps = metrics.crps(xh.view(members, *x.shape)[:, :, keep], y[:, keep]).mean()
    terms, logs = {"crps": crps}, {"crps": crps.item()}
    if guide is not None:
        R, _ = guide.residuals(x0r, xh, sts, rep(d[:, 0]), use_bridge=False, use_closure=False)
        terms["phys"] = guide.loss(R, rep(guide.mask(x, y, st)))
        logs["phys"] = terms["phys"].item()
    return terms, logs


class DeterministicPINN(nn.Module):
    """Stage 1 model as a transition: D(P(E(x_k))), no noise (members identical)."""

    def __init__(self, model: LatentPGCFM):
        super().__init__()
        self.m, self.cfg, self.gen_idx, self.w_area = model, model.cfg, model.gen_idx, model.w_area

    def sample(self, x, static, doy_next, tisr_next=None, steps=8, generator=None, grad=False, eps=None):
        with torch.set_grad_enabled(grad):
            m = self.m
            z0 = m.encode(x, static, (doy_next - 1) % 366)
            return m.decode(m.pinn(z0, doy_next), x, z0)


class DCTFiltered(nn.Module):
    """Non-learned control: any transition's output projected onto the kept DCT modes."""

    def __init__(self, cfm, dct_x: DCTTruncation):
        super().__init__()
        self.cfm, self.dct_x = cfm, dct_x
        self.cfg, self.gen_idx, self.w_area = cfm.cfg, cfm.gen_idx, cfm.w_area

    def sample(self, x, static, doy_next, tisr_next=None, steps=8, generator=None, grad=False, eps=None):
        y = self.cfm.sample(x, static, doy_next, tisr_next, steps=steps, generator=generator, grad=grad, eps=eps)
        with torch.set_grad_enabled(grad):
            return self.dct_x.filter(y)


# -- training loops -------------------------------------------------------------

def _set_trainable(mods, flag: bool):
    for m in mods:
        for p in m.parameters():
            p.requires_grad_(flag)


def _resume(ckpt: Optional[str], mods: Dict[str, nn.Module], opt, sched, device) -> Tuple[list, int]:
    """Load a ``_checkpoint`` file into ``mods``/``opt``/``sched`` -> (history, next iteration)."""
    if ckpt is None or not os.path.exists(ckpt):
        return [], 0
    ck = torch.load(ckpt, map_location=device, weights_only=False)  # own file; history holds numpy scalars
    for k, m in mods.items():
        # neuraloperator's FNO adds a "_metadata" entry that a wrapping module does not expect
        m.load_state_dict({n: v for n, v in ck[k].items() if n != "_metadata"})
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    return ck["hist"], ck["it"]


def _checkpoint(ckpt: str, mods: Dict[str, nn.Module], opt, sched, hist, it: int) -> None:
    torch.save({**{k: m.state_dict() for k, m in mods.items()}, "opt": opt.state_dict(),
                "sched": sched.state_dict(), "hist": hist, "it": it}, ckpt + ".tmp")
    os.replace(ckpt + ".tmp", ckpt)  # never leave a half-written file behind a restart


def _capped_backward(terms, params, lambda_p: float, grad_ratio: float, logs) -> None:
    """data + lambda_P x phys, with ||g_phys|| <= grad_ratio x ||g_data|| (step 4 W2 fix)."""
    if "phys" not in terms:
        terms["data"].backward()
        return
    # abs(): FNO spectral weights are complex
    norm = lambda gs: torch.sqrt(sum(g.abs().pow(2).sum() for g in gs if g is not None))  # noqa: E731
    g_d = torch.autograd.grad(terms["data"], params, retain_graph=True, allow_unused=True)
    g_p = torch.autograd.grad(lambda_p * terms["phys"], params, allow_unused=True)
    n_d, n_p = norm(g_d), norm(g_p)
    c = float(torch.clamp(grad_ratio * n_d / (n_p + 1e-12), max=1.0)) if grad_ratio > 0 else 1.0
    for p, a, q in zip(params, g_d, g_p):
        p.grad = None if a is None and q is None else \
            (a if q is None else (c * q if a is None else a + c * q))
    logs.update(grad_data=n_d.item(), grad_phys=n_p.item(), phys_scale=c)


def _log_avg(hist, logs, it, every, tag, log):
    avg = {k: float(np.mean([h[k] for h in hist[-every:] if k in h])) for k in logs}
    log(f"  [{tag}] it {it:6d} " + " ".join(f"{k} {v:.4g}" for k, v in avg.items()))


def train_stage1(model: LatentPGCFM, guide, loader, iters: int, lr: float = 1e-3, lambda_p: float = 0.1,
                 grad_ratio: float = 0.5, log_every: int = 100, log=print, device="cpu") -> List[Dict[str, float]]:
    """Encoder + latent PINN + decoder jointly; then P is frozen. ``guide=None`` or
    ``lambda_p=0``: no physics term. ``grad_ratio > 0`` caps ||g_phys|| <= ratio x ||g_data||."""
    from .projection import freeze
    from .train import _cycle, _to
    mods = [model.encoder.layers, model.decoder.layers, model.pinn]
    _set_trainable(mods, True)
    params = [p for m in mods for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, iters))
    use_phys = guide is not None and lambda_p > 0
    hist, batches = [], _cycle(loader)
    for it in range(iters):
        terms, logs = model.stage1_terms(_to(next(batches), device), guide if use_phys else None)
        opt.zero_grad(set_to_none=True)
        _capped_backward(terms, params, lambda_p, grad_ratio, logs)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        hist.append(logs)
        if (it + 1) % log_every == 0 or it + 1 == iters:
            _log_avg(hist, logs, it + 1, log_every, "stage1", log)
    freeze(model.pinn)
    return hist


def train_serial(model: LatentPGCFM, guide, loader, iters: int, lr: float = 1e-3, lambda_p: float = 0.1,
                 grad_ratio: float = 0.5, members: int = 2, steps: int = 4, joint: bool = True,
                 log_every: int = 100, log=print, device="cpu", ckpt: Optional[str] = None,
                 ckpt_every: int = 50) -> List[Dict[str, float]]:
    """Serial coupling CFM -> PINN, both trained in the same step (encoder/decoder from stage 1 frozen).

    Loss = L_CFM (flow matching on z_{k+1} - z_k) + decoded member CRPS + lambda_P x physics of
    every member after Q_phi, physics gradient capped at ``grad_ratio`` x the data gradient.
    ``steps`` Heun steps are back-propagated through when ``joint``.
    ``ckpt``: file written every ``ckpt_every`` iterations (CFM, Q_phi, optimizer, schedule) and
    resumed from if it exists; the data order and noise after a resume differ from an unbroken run.
    """
    from .train import _cycle, _to
    if model.serial is None:
        raise ValueError("model built with coupling='anchor'")
    _set_trainable([model.encoder.layers, model.decoder.layers, model.pinn], False)
    _set_trainable([model.net, model.serial], True)
    params = list(model.net.parameters()) + list(model.serial.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, iters))
    use_phys = guide is not None and lambda_p > 0
    mods = {"net": model.net, "serial": model.serial}
    batches = _cycle(loader)
    hist, start = _resume(ckpt, mods, opt, sched, device)
    if start:
        log(f"  [serial] resumed at it {start}")
    for it in range(start, iters):
        if ckpt is not None and it > start and it % ckpt_every == 0:
            _checkpoint(ckpt, mods, opt, sched, hist, it)
        b = _to(next(batches), device)
        l_fm, logs = model.cfm_loss(b)
        terms, lq = serial_terms(model, b, guide if use_phys else None, members, steps, joint)
        logs.update(lq)
        terms = {"data": l_fm + terms["crps"], **({"phys": terms["phys"]} if "phys" in terms else {})}
        opt.zero_grad(set_to_none=True)
        _capped_backward(terms, params, lambda_p, grad_ratio, logs)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        hist.append(logs)
        if (it + 1) % log_every == 0 or it + 1 == iters:
            _log_avg(hist, logs, it + 1, log_every, "serial", log)
    _set_trainable([model.net, model.serial], False)
    return hist


def train_stage2(model: LatentPGCFM, loader, iters: int, mode: str = "frozen", lr: float = 1e-3,
                 ae_lr: float = 1e-3, log_every: int = 100, log=print, device="cpu") -> List[Dict[str, float]]:
    """Latent CFM with the frozen PINN. ``mode``: "frozen" (encoder/decoder fixed) or
    "retrain" (encoder/decoder trained on CFM + reconstruction at ``ae_lr``)."""
    from .train import _cycle, _to
    if mode not in ("frozen", "retrain"):
        raise ValueError(mode)
    _set_trainable([model.pinn], False)
    ae = [model.encoder.layers, model.decoder.layers]
    _set_trainable(ae, mode == "retrain")
    groups = [{"params": list(model.net.parameters()), "lr": lr}]
    if mode == "retrain":
        groups.append({"params": [p for m in ae for p in m.parameters()], "lr": ae_lr})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, iters))
    hist, batches = [], _cycle(loader)
    for it in range(iters):
        loss, logs = model.cfm_loss(_to(next(batches), device), train_ae=mode == "retrain")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        hist.append(logs)
        if (it + 1) % log_every == 0 or it + 1 == iters:
            avg = {k: float(np.mean([h[k] for h in hist[-log_every:]])) for k in logs}
            log(f"  [stage2 {mode}] it {it + 1:6d} " + " ".join(f"{k} {v:.4g}" for k, v in avg.items()))
    _set_trainable(ae, False)
    return hist


def finetune_decoder(model: LatentPGCFM, loader, iters: int, lr: float = 1e-5, members: int = 2,
                     steps: int = 8, log_every: int = 50, log=print, device="cpu") -> List[Dict[str, float]]:
    """Ablation: encoder, PINN and CFM fixed; decoder on decoded member CRPS at low LR."""
    from .train import _cycle, _to
    _set_trainable([model.encoder.layers, model.pinn, model.net], False)
    _set_trainable([model.decoder.layers], True)
    params = list(model.decoder.layers.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    hist, batches = [], _cycle(loader)
    for it in range(iters):
        loss, logs = model.decoder_crps_loss(_to(next(batches), device), members, steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(logs)
        if (it + 1) % log_every == 0 or it + 1 == iters:
            log(f"  [dec-ft] it {it + 1:6d} crps {np.mean([h['crps'] for h in hist[-log_every:]]):.4g}")
    _set_trainable([model.decoder.layers], False)
    return hist


# -- diagnostics ------------------------------------------------------------------

def _keep(t: torch.Tensor) -> torch.Tensor:
    return t[..., [i for i in range(t.shape[-3]) if i != TISR_DYN], :, :]


def _rms(t: torch.Tensor) -> float:
    return float(metrics._wmean(_keep(t) ** 2).mean().sqrt())


@torch.no_grad()
def floors(model: LatentPGCFM, batches) -> Dict[str, float]:
    """The climate_diffusion check: reconstruction floors vs the 24 h tendency.

    rmse_dct: truncation only; rmse_ae: D(E(x)); rmse_ae_tendency: D(E(x_{k+1})) - D(E(x_k)) vs the
    tendency (the floor that matters with carry_tendency); rmse_pinn: D(P(E(x_k))) vs x_{k+1};
    tendency_rms: |x_{k+1} - x_k| (= persistence RMSE); pinn_tendency_ratio: |D(P(E(x_k))) - x_k| / |x_{k+1} - x_k|.
    """
    acc = {}
    for b in batches:
        x, y, st, d = b["x0"], b["targets"][:, 0], b["static"], b["doy"]
        z0 = model.encode(x, st, d[:, 0])
        xp = model.decode(model.pinn(z0, d[:, 1]), x, z0)
        tend = _rms(y - x)
        z1 = model.encode(y, st, d[:, 1])
        for k, v in {"rmse_dct": _rms(model.dct_x.filter(x) - x), "rmse_ae": _rms(model.decode(z0, x, z0) - x),
                     "rmse_ae_tendency": _rms(model.decoder(z1) - model.decoder(z0) - (y - x)),
                     "rmse_pinn": _rms(xp - y), "tendency_rms": tend,
                     "pinn_tendency_ratio": _rms(xp - x) / tend,
                     "tendency_rms_after_dct": _rms(model.dct_x.filter(y - x))}.items():
            acc.setdefault(k, []).append(v)
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["ae_floor_over_tendency"] = out["rmse_ae"] / out["tendency_rms"]
    out["ae_tendency_floor_over_tendency"] = out["rmse_ae_tendency"] / out["tendency_rms"]
    out["dct_floor_over_tendency"] = out["rmse_dct"] / out["tendency_rms"]
    out["pinn_over_persistence"] = out["rmse_pinn"] / out["tendency_rms"]
    return out


@torch.no_grad()
def member_structure(transition, batches, members: int = 8, steps: int = 8, seed: int = 3) -> Dict[str, object]:
    """Spatial structure of 1-day member deviations from the ensemble mean.

    lon_lag1: neighbour correlation along longitude (white noise ~0, truth tendency ~1);
    spectrum: zonal power of member deviations per wavenumber band, divided by the
    same band of the truth tendency (high band >> 1 = grid-scale noise).
    """
    g = torch.Generator().manual_seed(seed)
    lag, bands, amp = [], {}, []
    for b in batches:
        x, y, st, d = b["x0"], b["targets"][:, 0], b["static"], b["doy"][:, 1]
        rep = lambda t: t.repeat(members, *([1] * (t.dim() - 1)))  # noqa: E731
        e = _keep(transition.sample(rep(x), rep(st), rep(d), steps=steps, generator=g).view(members, *x.shape))
        dev = e - e.mean(0, keepdim=True)
        lag.append(float((dev * torch.roll(dev, 1, -1)).mean() / (dev * dev).mean().clamp_min(1e-20)))
        amp.append(_rms(e))
        W = x.shape[-1]
        p_dev = torch.fft.rfft(dev, dim=-1).abs().pow(2).mean(dim=(0, 1, 2, 3))
        p_tru = torch.fft.rfft(_keep(y - x), dim=-1).abs().pow(2).mean(dim=(0, 1, 2))
        edges = [(1, 4), (5, W // 6), (W // 6 + 1, W // 2)]
        for lo, hi in edges:
            bands.setdefault(f"k{lo}-{hi}", []).append(float(p_dev[lo:hi + 1].sum() / p_tru[lo:hi + 1].sum()))
    return {"lon_lag1": float(np.mean(lag)), "member_amp": float(np.mean(amp)),
            "spread_over_tendency_power": {k: float(np.mean(v)) for k, v in bands.items()}}


@torch.no_grad()
def range_coverage(transition, batches, members: int = 8, steps: int = 8, seed: int = 4) -> Dict[str, float]:
    """Fraction of truth points inside the 1-day member range [min, max]. For an exchangeable,
    calibrated M-member ensemble the expected value is (M - 1) / (M + 1)."""
    g = torch.Generator().manual_seed(seed)
    acc = []
    for b in batches:
        x, y, st, d = b["x0"], b["targets"][:, 0], b["static"], b["doy"][:, 1]
        rep = lambda t: t.repeat(members, *([1] * (t.dim() - 1)))  # noqa: E731
        e = _keep(transition.sample(rep(x), rep(st), rep(d), steps=steps, generator=g).view(members, *x.shape))
        o = _keep(y)
        inside = ((o >= e.min(0).values) & (o <= e.max(0).values)).float()
        acc.append(float(metrics._wmean(inside).mean()))
    return {"coverage": float(np.mean(acc)), "nominal": (members - 1) / (members + 1)}


@torch.no_grad()
def encoder_drift(new: LatentPGCFM, old: LatentPGCFM, batches) -> Dict[str, float]:
    """How far a retrained encoder moved, and what that does to the frozen PINN step."""
    dz, rp = [], []
    for b in batches:
        x, y, st, d = b["x0"], b["targets"][:, 0], b["static"], b["doy"]
        z_new, z_old = new.encode(x, st, d[:, 0]), old.encode(x, st, d[:, 0])
        dz.append(float((z_new - z_old).pow(2).mean().sqrt() / z_old.pow(2).mean().sqrt()))
        rp.append(_rms(new.decode(new.pinn(z_new, d[:, 1]), x, z_new) - y))
    return {"latent_rel_shift": float(np.mean(dz)), "rmse_frozen_pinn_after": float(np.mean(rp))}


def snapshot(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model).eval()
