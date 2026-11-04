"""The owner's ensemble-relation proposal (2026-10-26), on the real step 13 ensembles.

    G           leave-one-out operator G(x_{-k}) -> x_k: shared pointwise member embedding, LOO mean and
                std pooling (DeepSets, permutation invariant), grid coordinates, FNO blocks, MLP head
    norm        hard L2 operator-norm projection: with an orthonormal FFT, ||K||_{L2->L2} = max_k ||R(k)||_2,
                so rescaling every mode matrix R(k) bounds the spectral part exactly; per block the
                linear part K + W is bounded by bound / 2 + bound / 2 (the nonlinearities, lifting and head
                are not bounded, so the network's Lipschitz constant is only bounded layer by layer)
    energy      E_dual = mean_k |G(x_{-k}) - xbar|^2 <= alpha E(x), augmented Lagrangian (lambda >= 0)
    generation  relation penalty mean_k |y_k - G(y_{-k})|^2 and over-dispersion relu(E(y) - alpha E_ref)

The code follows the owner's text (planar FFT with norm="ortho" on the 32 x 64 latitude-longitude grid,
as written; the text itself suggests SFNO for a global grid). Changes: the area weights are (H, W)
cos(lat), the operator norms are logged without gradients, relation_refine applies the generation
penalty to finished members (normalized gradient steps instead of guidance inside the FM sampler), and
project_fno_ / fno_norms apply the same projection to the neuraloperator FNOs of pgcfm/ens_to_det.py.
"""

import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_sq(u: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
    """u (..., C, H, W) -> (...): ||u||^2_{L2(D)} with the domain measure normalized to 1, summed over C."""
    if w is None:
        return (u ** 2).mean(dim=(-2, -1)).sum(dim=-1)
    w = w / w.sum()
    return ((u ** 2) * w).sum(dim=(-2, -1)).sum(dim=-1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch, self.out_ch, self.m1, self.m2 = in_ch, out_ch, modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w_pos = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w_neg = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def mode_operator_norms(self) -> torch.Tensor:
        """||R(k)||_2 of every retained mode -> (2 m1 m2,)."""
        W = torch.cat([self.w_pos, self.w_neg], dim=-1)
        return torch.linalg.matrix_norm(W.permute(2, 3, 0, 1).reshape(-1, self.in_ch, self.out_ch), ord=2)

    def l2_operator_norm(self) -> torch.Tensor:
        return self.mode_operator_norms().max()

    @torch.no_grad()
    def project_(self, bound: float):
        for w in (self.w_pos, self.w_neg):
            s = torch.linalg.matrix_norm(w.permute(2, 3, 0, 1), ord=2)  # (m1, m2)
            w.mul_(torch.clamp(bound / (s + 1e-12), max=1.0)[None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1, m2 = self.m1, self.m2
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.w_pos)
        out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.w_neg)
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    """v -> act(K v + W v); the linear part has L2 norm <= max_k ||R(k)|| + ||W||_2."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        self.act = nn.GELU()

    def l2_operator_norm(self) -> torch.Tensor:
        return self.spec.l2_operator_norm() + torch.linalg.matrix_norm(self.w.weight[:, :, 0, 0], ord=2)

    @torch.no_grad()
    def project_(self, bound: float):
        self.spec.project_(bound / 2)
        s = torch.linalg.matrix_norm(self.w.weight[:, :, 0, 0], ord=2)
        self.w.weight.mul_(torch.clamp((bound / 2) / (s + 1e-12), max=1.0))

    def forward(self, x: torch.Tensor, last: bool = False) -> torch.Tensor:
        y = self.spec(x) + self.w(x)
        return y if last else self.act(y)


class EnsembleRelationFNO(nn.Module):
    """x (B, N, C, H, W) -> x_hat with x_hat[:, k] = G(x_{-k})."""

    def __init__(self, in_ch: int, width: int = 32, modes=(12, 12), n_layers: int = 4, use_std: bool = True):
        super().__init__()
        self.use_std = use_std
        self.member_lift = nn.Conv2d(in_ch, width, 1)
        self.lift = nn.Conv2d(width * (2 if use_std else 1) + 2, width, 1)
        self.blocks = nn.ModuleList(FNOBlock(width, *modes) for _ in range(n_layers))
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, in_ch, 1))

    @staticmethod
    def _grid(H: int, W: int, device) -> torch.Tensor:
        gy, gx = torch.meshgrid(torch.linspace(0, 1, H, device=device), torch.linspace(0, 1, W, device=device),
                                indexing="ij")
        return torch.stack([gy, gx])[None, None]

    def loo_context(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        phi = self.member_lift(x.reshape(B * N, C, H, W)).reshape(B, N, -1, H, W)
        S1 = phi.sum(1, keepdim=True)
        mean_loo = (S1 - phi) / (N - 1)
        feats = [mean_loo]
        if self.use_std:
            var_loo = ((phi ** 2).sum(1, keepdim=True) - phi ** 2) / (N - 1) - mean_loo ** 2
            feats.append(torch.sqrt(var_loo.clamp_min(1e-8)))
        feats.append(self._grid(H, W, x.device).expand(B, N, -1, -1, -1))
        return torch.cat(feats, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        h = self.lift(self.loo_context(x).reshape(B * N, -1, H, W))
        for i, blk in enumerate(self.blocks):
            h = blk(h, last=(i == len(self.blocks) - 1))
        return self.proj(h).reshape(B, N, C, H, W)

    def l2_operator_norms(self) -> torch.Tensor:
        return torch.stack([b.l2_operator_norm() for b in self.blocks])

    @torch.no_grad()
    def project_(self, bound: float):
        for b in self.blocks:
            b.project_(bound)


def ensemble_terms(x, x_hat, w=None):
    xbar = x.mean(1, keepdim=True)
    recon = l2_sq(x_hat - x, w).mean()
    E_true = l2_sq(x - xbar, w).mean(1)
    E_dual = l2_sq(x_hat - xbar, w).mean(1)
    mean_cons = l2_sq(x_hat.mean(1) - xbar.squeeze(1), w).mean()
    return recon, E_true, E_dual, mean_cons


class EnergyConstraint:
    """g = E_dual - alpha E_true <= 0: L = recon + beta mean_cons + lambda g + rho / 2 relu(g)^2,
    lambda <- max(0, lambda + rho g)."""

    def __init__(self, alpha: float = 1.0, rho: float = 10.0, beta: float = 1.0, lam_init: float = 0.0):
        self.alpha, self.rho, self.beta, self.lam = alpha, rho, beta, lam_init

    def loss(self, x, x_hat, w=None):
        recon, E_true, E_dual, mean_cons = ensemble_terms(x, x_hat, w)
        g = (E_dual - self.alpha * E_true).mean()
        loss = recon + self.beta * mean_cons + self.lam * g + 0.5 * self.rho * F.relu(g) ** 2
        return loss, dict(recon=recon.item(), E_true=E_true.mean().item(), E_dual=E_dual.mean().item(),
                          g=g.item(), lam=self.lam)

    def dual_step(self, g_value: float):
        self.lam = max(0.0, self.lam + self.rho * g_value)


def train_step(model, opt, constraint, x, w=None, operator_norm_bound=None):
    model.train()
    x_hat = model(x)
    loss, log = constraint.loss(x, x_hat, w)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    constraint.dual_step(log["g"])
    if operator_norm_bound is not None:
        model.project_(operator_norm_bound)
    with torch.no_grad():
        log["op_norm_max"] = model.l2_operator_norms().max().item()
    return log


def fit_relation(model, train_x, val_x, w, epochs: int = 20, batch: int = 4, lr: float = 1e-3,
                 constraint: Optional[EnergyConstraint] = None, bound: Optional[float] = None,
                 log: Callable = print):
    """train_x / val_x (S, N, C, H, W) normalized members. Without ``constraint`` the loss is the plain
    LOO reconstruction (alpha = rho = beta = 0 keeps lambda at 0). Checkpoint = best validation recon."""
    constraint = constraint or EnergyConstraint(alpha=0.0, rho=0.0, beta=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    if bound is not None:
        model.project_(bound)
    best, best_state, hist = float("inf"), None, []
    for ep in range(epochs):
        logs = [train_step(model, opt, constraint, train_x[i], w, bound)
                for i in torch.randperm(len(train_x)).split(batch)]
        model.eval()
        with torch.no_grad():
            q = [ensemble_terms(val_x[i], model(val_x[i]), w) for i in torch.arange(len(val_x)).split(16)]
        val = float(torch.stack([r for r, *_ in q]).mean())
        shrink = float(torch.cat([d for *_, d, _ in q]).mean() / torch.cat([e for _, e, *_ in q]).mean())
        h = {k: float(sum(l_[k] for l_ in logs) / len(logs)) for k in ("recon", "g", "lam", "op_norm_max")}
        hist.append({**h, "val_recon": val, "val_E_dual_over_E": shrink})
        log(f"[relation] epoch {ep} train recon {h['recon']:.4f} g {h['g']:.4f} lambda {h['lam']:.3f} "
            f"op norm {h['op_norm_max']:.2f} | val recon {val:.4f} E_dual/E {shrink:.4f}")
        if val < best:
            best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return {"best_val_recon": best, "history": hist}


def ensemble_relation_penalty(G, y, E_ref, alpha: float = 1.0, w=None):
    """y (B, N, C, H, W): consistency mean_k |y_k - G(y_{-k})|^2 and over-dispersion relu(E(y) - alpha E_ref)."""
    consistency = l2_sq(y - G(y), w).mean()
    E_y = l2_sq(y - y.mean(1, keepdim=True), w).mean(1)
    return consistency, F.relu(E_y - alpha * E_ref).mean()


def relation_refine(G, y, E_ref, w, tau: float = 0.5, steps: int = 2, gamma_c: float = 1.0,
                    gamma_e: float = 1.0, alpha: float = 1.0):
    """The generation penalty applied to finished members: y <- y - tau * grad, with the gradient of
    gamma_c consistency + gamma_e overdisp rescaled by B N / (2 w_cell) (w normalized to sum 1), so the
    direct consistency term is y_k - G(y_{-k}) and the over-dispersion term is y_k - ybar while
    E(y) > alpha E_ref. y (B, N, C, H, W), E_ref (B,), w (H, W)."""
    B, N = y.shape[:2]
    wn = w / w.sum()
    scale = B * N / (2 * wn)
    for _ in range(steps):
        with torch.enable_grad():
            y_ = y.detach().requires_grad_(True)
            c, e = ensemble_relation_penalty(G, y_, E_ref, alpha, w)
            grad = torch.autograd.grad(gamma_c * c + gamma_e * e, y_)[0]
        y = (y_ - tau * grad * scale).detach()
    return y


# -- the same projection on the neuraloperator FNOs of pgcfm/ens_to_det.py -----------------------------
def _fno_blocks(module: nn.Module):
    from neuralop.layers.fno_block import FNOBlocks
    return [m for m in module.modules() if isinstance(m, FNOBlocks)]


@torch.no_grad()
def project_fno_(module: nn.Module, bound: float):
    """Every FNO block of ``module``: max_k ||R(k)||_2 <= bound / 2 and ||W_skip||_2 <= bound / 2, so the
    block's linear part K + W has L2 operator norm <= bound (neuraloperator's forward / backward FFT
    pair is a scaled unitary pair, so the mode bound is exact for K)."""
    for fb in _fno_blocks(module):
        for conv, skip in zip(fb.convs, fb.fno_skips):
            W = conv.weight.tensor  # (in, out, m1, m2) complex
            s = torch.linalg.matrix_norm(W.permute(2, 3, 0, 1), ord=2)
            W.mul_(torch.clamp((bound / 2) / (s + 1e-12), max=1.0)[None, None])
            Ws = skip.conv.weight  # (out, in, 1)
            Ws.mul_(torch.clamp((bound / 2) / (torch.linalg.matrix_norm(Ws[..., 0], ord=2) + 1e-12), max=1.0))


@torch.no_grad()
def fno_norms(module: nn.Module) -> list:
    """Per FNO block: (max_k ||R(k)||_2, ||W_skip||_2)."""
    out = []
    for fb in _fno_blocks(module):
        for conv, skip in zip(fb.convs, fb.fno_skips):
            s = torch.linalg.matrix_norm(conv.weight.tensor.permute(2, 3, 0, 1), ord=2)
            out.append((float(s.max()), float(torch.linalg.matrix_norm(skip.conv.weight[..., 0], ord=2))))
    return out
