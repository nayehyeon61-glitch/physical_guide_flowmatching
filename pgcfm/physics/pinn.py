"""PyTorch port of maziarraissi/PINNs (continuous-time inference) for the
pressure-level primitive equations.

Structure follows ``third_party/pinns/Burgers.py`` (see UPSTREAM.md for the
line-by-line mapping): a coordinate MLP ``net_u`` and an autograd residual
``net_f``, trained on ``mean((u - u_pred)^2) + mean(f_pred^2)``.

Changes needed for the sphere and the PG-CFM paper:
  * Input is (x, y, z, ln p, s) with (x, y, z) the unit-sphere point, so
    scalar outputs are smooth across the poles and periodic in longitude.
  * Wind is predicted as a Cartesian tangent vector V and projected to
    u = V.e_lam, v = V.e_phi (the "Cartesian tangent-vector representation"
    of paper appendix B), so u, v are well defined up to the pole.
  * Residuals are differentiated with respect to the physical coordinates
    (lam, phi, p, s), so the chain-rule factors 1/a, 1/dt and the pressure
    scale are exact; outputs are rescaled to SI units before the residual.
  * Each equation is divided by its scale b_j (eq. 10).

This module is the coordinate PINN only. The conditional bridge B_psi of
paper eq. (9) that ties it to CFM endpoints is the next step.
"""

import math
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn

from ..contract import CONST, Constants
from . import primitive

Fields = Dict[str, torch.Tensor]
FieldFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Fields]

COORDS = ("lam", "phi", "p", "s")

# (shift, scale) per output, SI units: field = shift + scale * net_output
DEFAULT_FIELD_SCALES = {
    "u": (0.0, 20.0), "v": (0.0, 20.0), "T": (250.0, 30.0), "Phi": (8.0e4, 6.0e4),
    "omega": (0.0, 0.5), "q": (3.0e-3, 3.0e-3),
}


def autograd_residuals(field_fn: FieldFn, lam, phi, p, s, const: Constants = CONST,
                       closures: Optional[primitive.Closures] = None, virtual: bool = True,
                       create_graph: bool = True) -> Dict[str, torch.Tensor]:
    """net_f: strong-form residuals of ``field_fn`` by automatic differentiation.

    ``field_fn(lam, phi, p, s)`` returns u, v, T, Phi, omega and optionally q
    for 1-D coordinate tensors of equal length (one collocation point each).
    """
    coords = [c.detach().clone().requires_grad_(True) for c in (lam, phi, p, s)]
    F = dict(field_fn(*coords))
    for name in [n for n in F if n != "f"]:
        if not F[name].requires_grad:  # e.g. an analytic omega = 0
            grads = [None] * len(coords)
        else:
            grads = torch.autograd.grad(F[name].sum(), coords, create_graph=create_graph,
                                        retain_graph=True, allow_unused=True)
        for c, g in zip(COORDS, grads):
            F[f"d{c}_{name}"] = torch.zeros_like(F[name]) if g is None else g
    F.pop("ds_Phi"), F.pop("ds_omega")  # not used by any equation
    return primitive.residuals(F, coords[1], coords[2], const, closures, virtual)


def sample_collocation(n: int, p_range=(5.0e3, 1.0e5), s_range=(0.0, CONST.dt_seconds),
                       pole_cap_deg: float = 89.0, generator: Optional[torch.Generator] = None,
                       dtype=torch.float64):
    """Area-uniform points on the sphere (sin phi uniform), |phi| <= cap; ln p and s uniform."""
    g = generator
    zmax = math.sin(math.radians(pole_cap_deg))
    z = (2 * torch.rand(n, generator=g, dtype=dtype) - 1) * zmax
    phi = torch.asin(z)
    lam = 2 * math.pi * torch.rand(n, generator=g, dtype=dtype)
    lnp = math.log(p_range[0]) + (math.log(p_range[1]) - math.log(p_range[0])) * torch.rand(n, generator=g, dtype=dtype)
    s = s_range[0] + (s_range[1] - s_range[0]) * torch.rand(n, generator=g, dtype=dtype)
    return lam, phi, torch.exp(lnp), s


class PhysicsInformedNN(nn.Module):
    """Raissi's PhysicsInformedNN, ported to PyTorch for the primitive equations."""

    OUTPUTS = ("Vx", "Vy", "Vz", "T", "Phi", "omega", "q")

    def __init__(self, layers: Sequence[int], lb: Sequence[float], ub: Sequence[float],
                 const: Constants = CONST, scales: primitive.ResidualScales = primitive.ResidualScales(),
                 field_scales: Optional[Dict[str, tuple]] = None, virtual: bool = True,
                 dtype=torch.float32):
        """layers: [5, ..., 7]; lb/ub: lower/upper bounds of (ln p, s) for the input scaling."""
        super().__init__()
        assert layers[0] == 5 and layers[-1] == len(self.OUTPUTS)
        self.register_buffer("lb", torch.as_tensor(lb, dtype=dtype))
        self.register_buffer("ub", torch.as_tensor(ub, dtype=dtype))
        self.const, self.scales, self.virtual = const, scales, virtual
        self.field_scales = dict(DEFAULT_FIELD_SCALES, **(field_scales or {}))
        self.layers = self.initialize_NN(layers).to(dtype)
        self.dtype = dtype

    def initialize_NN(self, layers: Sequence[int]) -> nn.ModuleList:
        mods = nn.ModuleList()
        for i in range(len(layers) - 1):
            lin = nn.Linear(layers[i], layers[i + 1])
            self.xavier_init(lin.weight)
            nn.init.zeros_(lin.bias)
            mods.append(lin)
        return mods

    @staticmethod
    def xavier_init(w: torch.Tensor) -> None:
        out_dim, in_dim = w.shape
        std = np.sqrt(2.0 / (in_dim + out_dim))
        nn.init.trunc_normal_(w, std=std, a=-2 * std, b=2 * std)

    def neural_net(self, X: torch.Tensor) -> torch.Tensor:
        """X = [x, y, z, ln p, s]; (ln p, s) scaled to [-1, 1] as in upstream."""
        H = torch.cat([X[:, :3], 2.0 * (X[:, 3:] - self.lb) / (self.ub - self.lb) - 1.0], dim=1)
        for lin in self.layers[:-1]:
            H = torch.tanh(lin(H))
        return self.layers[-1](H)

    def net_u(self, lam, phi, p, s) -> Fields:
        cl, sl, cp, sp = torch.cos(lam), torch.sin(lam), torch.cos(phi), torch.sin(phi)
        X = torch.stack([cp * cl, cp * sl, sp, torch.log(p), s], dim=1).to(self.dtype)
        Y = self.neural_net(X)
        out = {k: Y[:, i] for i, k in enumerate(self.OUTPUTS)}
        # project the Cartesian vector onto the local east / north unit vectors
        u = -sl * out["Vx"] + cl * out["Vy"]
        v = -sp * cl * out["Vx"] - sp * sl * out["Vy"] + cp * out["Vz"]
        F = {"u": u, "v": v}
        F.update({k: out[k] for k in ("T", "Phi", "omega", "q")})
        return {k: self.field_scales[k][0] + self.field_scales[k][1] * x for k, x in F.items()}

    def net_f(self, lam, phi, p, s, closures=None, create_graph: bool = True) -> Dict[str, torch.Tensor]:
        return autograd_residuals(self.net_u, lam, phi, p, s, self.const, closures, self.virtual,
                                  create_graph=create_graph)

    def loss(self, X_u, targets: Fields, X_f, weights: Optional[Dict[str, float]] = None,
             closures=None) -> Dict[str, torch.Tensor]:
        """Data misfit (normalized by field scales) + sum_j gamma_j mean((R_j / b_j)^2)."""
        pred = self.net_u(*X_u)
        data = sum(torch.mean(((pred[k] - targets[k].to(pred[k].dtype)) / self.field_scales[k][1]) ** 2)
                   for k in targets)
        R = self.net_f(*X_f, closures=closures)
        phys = primitive.physics_loss(R, self.scales, weights)
        return {"loss": data + phys, "data": data, "phys": phys}

    def train_model(self, X_u, targets: Fields, X_f, n_adam: int = 1000, lr: float = 1e-3,
                    n_lbfgs: int = 0, weights=None, callback=None):
        """Adam, then optional L-BFGS (upstream uses L-BFGS-B only)."""
        history = []
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for it in range(n_adam):
            opt.zero_grad()
            L = self.loss(X_u, targets, X_f, weights)
            L["loss"].backward()
            opt.step()
            history.append({k: float(torch.as_tensor(v).detach()) for k, v in L.items()})
            if callback is not None:
                callback(it, L)
        if n_lbfgs:
            opt = torch.optim.LBFGS(self.parameters(), max_iter=n_lbfgs, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                L = self.loss(X_u, targets, X_f, weights)
                L["loss"].backward()
                return L["loss"]
            opt.step(closure)
            L = self.loss(X_u, targets, X_f, weights)
            history.append({k: float(torch.as_tensor(v).detach()) for k, v in L.items()})
        return history

    def predict(self, lam, phi, p, s):
        R = self.net_f(lam, phi, p, s, create_graph=False)
        with torch.no_grad():
            F = self.net_u(lam, phi, p, s)
        return F, {k: v.detach() for k, v in R.items()}
