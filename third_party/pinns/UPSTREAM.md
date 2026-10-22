# Vendored reference: maziarraissi/PINNs

- Source: https://github.com/maziarraissi/PINNs, commit `932f50a2d8ef4e80d1456bbae6887a73ff5166ef`
- File: `appendix/continuous_time_inference (Burgers)/Burgers.py` (unmodified, TensorFlow 1)
- License: MIT, Copyright (c) 2018 maziarraissi (see `LICENSE`)

The TF1 code (`tf.Session`, `tf.placeholder`, `tf.contrib.opt.ScipyOptimizerInterface`) does not
run on current TensorFlow and cannot share a graph with the PyTorch CFM generator, so it is kept
here as the reference and is not imported. `pgcfm/physics/pinn.py` is the PyTorch port.

## Mapping to the port

| Upstream (`Burgers.py`) | `pgcfm/physics/pinn.py` | Change |
| --- | --- | --- |
| `PhysicsInformedNN.__init__(X_u, u, X_f, layers, lb, ub, nu)` | `PhysicsInformedNN.__init__(layers, lb, ub, ...)` | data passed to `loss`/`train` instead of placeholders |
| `initialize_NN`, `xavier_init` (truncated normal, std `sqrt(2/(in+out))`) | same names | `nn.Linear` + `trunc_normal_` |
| `neural_net`: `H = 2(X-lb)/(ub-lb) - 1`, tanh hidden layers, linear output | same | input is `[x, y, z, ln p, s]`; the sphere point `(x, y, z)` is already in [-1, 1] |
| `net_u(x, t)` | `net_u(lam, phi, p, s)` | outputs u, v (from a Cartesian tangent vector), T, Phi, omega, q in SI units |
| `net_f(x, t)`: `tf.gradients` + Burgers residual | `net_f(lam, phi, p, s)` | `torch.autograd.grad` + primitive-equation residuals (`primitive.py`) |
| `loss = mean((u - u_pred)^2) + mean(f_pred^2)` | `loss(...)` | per-equation scales `b_j` and weights `gamma_j` (paper eq. 10) |
| `train` (L-BFGS-B via ScipyOptimizerInterface) | `train` | Adam, then optional `torch.optim.LBFGS` |
| `predict(X_star)` | `predict(...)` | returns fields and residuals |
