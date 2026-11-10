"""Where to put the tendency core in KAI-alpha (docs/kai_tendency.md).

Two models, trained separately:

* the tendency core: pgcfm.dyncore.DynCore of step 16, unchanged (ClimODE's ResNet as dx/dt = f(x), one Heun
  step, tisr forcing, PositiveMass, rollout training), here with one step = one day on KAI's daily means;
* KAI-alpha: third_party/kai (InceptionNeXt, full next-day state, single-step training, relative L2 loss).

The core is trained first and frozen. Its one-day forecast c(x) and tendency tau = (c(x) - x) / s
(s = std of the daily increment per channel) then enter KAI at ONE position:

    inject  in        tau as 67 extra input channels of the stem
            s1 .. s4  tau added to the features after stage 1 .. 4 through a zero-initialized 1x1 conv
                      (s4 = just before KAI's head)
    form    full      KAI's own output: the full next state                       (original KAI)
            res       x + s * net(.)        zero head = persistence               (control, no core)
            core      c(x) + s * net(.)     zero head = the core's forecast       (the core at the output)
            gate      k(x) + g * (c(x) - k(x))   k = KAI's full next state, g = one weight per channel,
                      zero g = the original KAI: KAI keeps its own form and its trained weights (--init kai),
                      the core is mixed in at the output every step (the trained version of the blend)

A further position needs no training: a lead-wise blend of the two finished forecasts
(scripts/run_kai_tendency.py, ``blend``; ``loop`` = the same weights applied every step and fed back).

Memory (the owner's Mori-Zwanzig reading, 2026-11-10): the daily 5.625 deg state is a projection of the
atmosphere, so its exact equation is  dx/dt = Markov term f(x) + memory integral over the past + noise.
DynCore is the Markov term alone. ``make_core(hist=True)`` adds one lag (the observed trend),
``make_core(mem=W)`` a ConvGRU state m that integrates the past (W days before the IC, then the rollout's own
states) and enters the tendency as input before every step: dx/dt = f(x, m).
"""

import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import contract

_KAI_ROOT = os.path.join(os.path.dirname(__file__), "..", "third_party", "kai")
if _KAI_ROOT not in sys.path:
    sys.path.insert(0, _KAI_ROOT)
from src.models import KAI  # noqa: E402  (vendored KAI-alpha backbone)
from src.utils import LpLoss  # noqa: E402

TISR = contract.CHANNEL_INDEX["tisr"]
OROG = contract.CHANNEL_INDEX["orog"]
DYN = contract.DYNAMIC_CHANNELS
TISR_DYN = DYN.index(TISR)

# arm -> (form, inject)
ARMS = {"kai": ("full", "none"), "in": ("full", "in"), "s1": ("full", "s1"), "s2": ("full", "s2"),
        "s3": ("full", "s3"), "s4": ("full", "s4"), "res": ("res", "none"), "inres": ("res", "in"),
        "out": ("core", "none"), "inout": ("core", "in"), "gate": ("gate", "none"), "ingate": ("gate", "in"),
        # control of a history core: KAI itself gets the observed trend (x - x_prev) / s as input, no core
        "kaih": ("full", "hist"),
        # control of ``out``: the frozen first stage is not the tendency core but a trained KAI (arm res)
        "stack": ("core", "none")}


def needs_core(arm: str) -> bool:
    form, inject = ARMS[arm]
    return form in ("core", "gate") or inject not in ("none", "hist")


# ------------------------------------------------------------------------------------- the tendency core
def stateless_norm(module: nn.Module) -> nn.Module:
    """Replace every BatchNorm2d by GroupNorm(1, C) and switch Dropout off, in place.

    ClimODE's ResidualBlock (the core's trunk) has BatchNorm + Dropout(0.1). Trained with batch statistics
    and run with running statistics, the core's tendency gets a systematic offset that a rollout integrates:
    on 2017 the global mean of z500 drifts by -0.14 sigma in 10 days (z1000 -0.47, q700 +0.29) with running
    statistics and by +0.004 with batch statistics, same weights (docs/kai_tendency.md). GroupNorm has no
    batch statistics, so the trained and the deployed function are the same."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(1, child.num_features))
        elif isinstance(child, nn.Dropout):
            setattr(module, name, nn.Identity())
        else:
            stateless_norm(child)
    return module


def _hist_core_class():
    """DynCore with the observed trend as input: f(x, x - x_prev). ``prev`` = the state one step before the
    step's start (data at the IC, the model's own previous state afterwards); None = no trend (zeros)."""
    from .dyncore import DynCore, Net, _NetCfg, N_DYN

    class HistCore(DynCore):
        hist = True

        def __init__(self, cfg, static, t6, mean, std, lat_deg):
            super().__init__(cfg, static, t6, mean, std, lat_deg)
            self.net = Net(2 * N_DYN + static.shape[0] + 4, N_DYN, _NetCfg(cfg.hidden, cfg.blocks))
            self.prev, self.trend = None, None

        def _trunk(self, x, hours, g=None):
            from .dyncore import time_features
            B, _, H, W = x.shape
            return self.net.trunk(torch.cat([x, self.trend, self.static.expand(B, -1, -1, -1),
                                             time_features(hours, H, W)], 1))

        def step(self, x, hours, tisr_next):
            # the trend of the step's start, in units of the increment std; the same for both Heun evaluations
            self.trend = torch.zeros_like(x) if self.prev is None else (x - self.prev) / self.t6 * self.keep
            y = super().step(x, hours, tisr_next)
            self.prev = x
            return y
    return HistCore


class ConvGRUCell(nn.Module):
    """The standard convolutional GRU cell (Ballas et al. 2016): gates and candidate are 3x3 convolutions of
    [input, state]; padding as in the core's trunk (longitude circular, latitude reflect). The state is a leaky
    time integral of its inputs with a learned, state-dependent kernel: m <- (1 - z) m + z n."""

    def __init__(self, cin: int, hid: int):
        super().__init__()
        self.hid = hid
        self.gates = nn.Conv2d(cin + hid, 2 * hid, 3)
        self.cand = nn.Conv2d(cin + hid, hid, 3)

    @staticmethod
    def pad(x):
        return F.pad(F.pad(x, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="reflect")

    def forward(self, u, m=None):
        if m is None:
            m = u.new_zeros(u.shape[0], self.hid, *u.shape[-2:])
        z, r = torch.sigmoid(self.gates(self.pad(torch.cat([u, m], 1)))).chunk(2, 1)
        n = torch.tanh(self.cand(self.pad(torch.cat([u, r * m], 1))))
        return (1 - z) * m + z * n


class LongTerm(nn.Module):
    """The long-memory term of the accumulated memory: two 3x3 convolutions of the accumulators z (padding as in
    the trunk), zero-initialized output, so a new core starts without it."""

    def __init__(self, cin: int, hid: int, cout: int):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(cin, hid, 3), nn.Conv2d(hid, cout, 3)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, z):
        return self.c2(ConvGRUCell.pad(F.silu(self.c1(ConvGRUCell.pad(z)))))


def _mem_core_class():
    """DynCore with a memory term in the Mori-Zwanzig sense: dx/dt = f(x, m), m = ConvGRU state.

    The memory is updated BEFORE every time step with the state the step starts from and its observed trend,
    m_t = GRU(m_{t-1}, [x_t, (x_t - x_{t-1}) / s]), and enters both Heun evaluations of the step as input.
    ``reset(history)`` runs the GRU over the W days before the IC (data); in the rollout it goes on with the
    core's own states. The Markov core is the case m = const, the history core (one lag) the case W = 1
    without recurrence (cf. the LSTM memory closures of Ma, Wang & E 2018 and Wang, Ripamonti & Hesthaven 2020).

    ``mz`` = the Mori-Zwanzig form on a sequence (owner, 2026-11-10: the loss has to be taken over a sequence in the
    form of the equation, not on single terms). The W days before the IC are not only read: the memory starts
    EMPTY at the first day of the sequence (the memory integral runs from 0 to t), every day of the sequence is
    forecast from the day before with the memory of the days up to it (the first forecast is the Markov term
    alone), and the error of each forecast, the innovation e_t = (x_t - forecast of x_t) / s, is the third input
    of the memory: m_t = GRU(m_{t-1}, [x_t, trend_t, e_t]) (the memory term is what the past forecasts missed).
    ``reset`` keeps these W forecasts in ``seq`` for the sequence loss (scripts/run_kai_tendency.py); in the free
    rollout the state is the core's own forecast, so e = 0.

    With ``cfg.gfeat`` (owner, 2026-11-10: the physical terms need memory terms next to them) the trunk also
    gets the operator's tendency G(x) (48 physical terms, step 18) and the memory integrates the physical terms:
    m_t = GRU(m_{t-1}, [x_t, trend_t, (e_t,) G(x_t), G(x_{t-1})]), dx/dt = f(x, G(x), m).

    ``acc`` = time constants in days, e.g. (3, 5) (owner, 2026-11-11: the memory has to hold the past of 3 to 5
    days for certain, as an accumulation structure with its own term and loss). The trained GRU uses one day of
    past, so next to it the same inputs are accumulated with FIXED exponential kernels,
    z_k(t) = exp(-1/tau_k) z_k(t-1) + (1 - exp(-1/tau_k)) tanh(enc_k([x_t, trend_t, e_t, G_t, G_{t-1}])),
    the Markovian embedding of a memory kernel that is a sum of exponentials: z_k is the time integral of the past
    with an e-folding time of tau_k days whatever the training does. The accumulators enter as their own term,
    dx/dt = f(x, G(x), m) + g(z_3, z_5), the long-memory term (LongTerm; ``acc_off`` switches it off, a diagnostic).
    The accumulators start empty with the sequence and are filled over the W days before the IC (W >= 8 for
    tau = 5 d), in the rollout they go on with the states the core is stepped along."""
    from .dyncore import DynCore, Net, _NetCfg, N_DYN, N_GFEAT
    from .core import time_features

    class MemCore(DynCore):
        hist = True  # the caller has to hand over the past

        def __init__(self, cfg, static, t6, mean, std, lat_deg, mem=8, mem_ch=24, mz=False, nomem=False,
                     acc=(), acc_ch=12):
            super().__init__(cfg, static, t6, mean, std, lat_deg)
            self.mem, self.mem_ch, self.mz = int(mem), int(mem_ch), bool(mz)
            self.nomem = bool(nomem)  # control: the same sequence training, the memory state stays zero
            ng = N_GFEAT if cfg.gfeat else 0
            self.net = Net(N_DYN + mem_ch + static.shape[0] + 4 + ng, N_DYN, _NetCfg(cfg.hidden, cfg.blocks))
            self.enc = nn.Conv2d((3 if mz else 2) * N_DYN + 2 * ng, mem_ch, 1)
            self.gru = ConvGRUCell(mem_ch, mem_ch)
            self.acc, self.z, self.acc_off = tuple(float(t) for t in acc), None, False
            if self.acc:  # the accumulated memory: fixed exponential kernels of tau days, their own term
                if nomem:
                    raise ValueError("the no-memory control has no accumulated memory")
                self.enc_acc = nn.Conv2d(self.enc.in_channels, int(acc_ch) * len(self.acc), 1)
                lam = torch.exp(-1.0 / torch.tensor(self.acc)).repeat_interleave(int(acc_ch))
                self.register_buffer("lam", lam.view(1, -1, 1, 1))
                self.long = LongTerm(int(acc_ch) * len(self.acc), cfg.hidden, N_DYN)
            self.m, self.prev, self.gprev, self.started = None, None, None, False
            self.fc, self.seq = None, None  # mz: the forecast of the next state; the forecasts of the sequence
            self.freeze_mem = False  # diagnostic: no update of the memory after the IC (the rollout's own states)

        def reset(self, history=None, hours0=None, x0=None):
            """history (B, W, 66, H, W): the states before the IC, oldest first; None = an empty memory.
            mz: ``hours0`` (B,) = the IC's time, ``x0`` = the IC (its tisr is the forcing of the last forecast)."""
            self.m, self.prev, self.gprev, self.started, self.fc, self.seq = None, None, None, False, None, None
            self.z = None
            if history is None:
                return
            W = history.shape[1]
            if not self.mz:
                for k in range(W):
                    self.remember(history[:, k])
                return
            out = []
            for k in range(W):  # teacher forcing over the sequence: day k -> forecast of day k + 1
                nxt = history[:, k + 1] if k + 1 < W else x0
                xk = history[:, k]  # one tensor for the memory and the step (the operator's G is cached per tensor)
                self.remember(xk)
                self.fc = super().step(xk, hours0 - (W - k) * 24.0, None if nxt is None else nxt[:, TISR_DYN])
                out.append(self.fc)
            self.seq = torch.stack(out, 1)

        def remember(self, x):
            if self.nomem:
                self.m, self.prev = x.new_zeros(x.shape[0], self.mem_ch, *x.shape[-2:]), x
                return
            trend = torch.zeros_like(x) if self.prev is None else (x - self.prev) / self.t6 * self.keep
            parts = [x, trend]
            if self.mz:  # the innovation: zero at the start of a sequence and on the core's own states
                parts.append(torch.zeros_like(x) if self.fc is None
                             else (x.detach() - self.fc.detach()) / self.t6 * self.keep)
            if self.cfg.gfeat:
                # the memory terms of the physical terms: the memory integrates the operator's tendency G of the
                # state now and of the day before (next to the observed trend, so what the physical terms missed
                # over the last day, trend - G_prev, is a linear combination of its inputs)
                g = self.features(x)
                parts += [g, torch.zeros_like(g) if self.gprev is None else self.gprev]
                self.gprev = g
            v = torch.cat(parts, 1)
            self.m = self.gru(self.enc(v), self.m)
            if self.acc:  # the time integral of the past with the fixed kernels (an empty accumulator = no past)
                u = (1 - self.lam) * torch.tanh(self.enc_acc(v))
                self.z = u if self.z is None else self.lam * self.z + u
            self.prev = x

        def _trunk(self, x, hours, g=None):
            B, _, H, W = x.shape
            parts = [x, self.m, self.static.expand(B, -1, -1, -1), time_features(hours, H, W)]
            if self.cfg.gfeat:
                parts.append(self.features(x) if g is None else g)
            return self.net.trunk(torch.cat(parts, 1))

        def tendency(self, x, hours, g=None):
            f = super().tendency(x, hours, g)
            if self.acc and self.z is not None and not self.acc_off:
                f = f + self.long(self.z) * self.t6 * self.keep  # the long-memory term, the same in both stages
            return f

        def step(self, x, hours, tisr_next):
            if not (self.freeze_mem and self.started):
                self.remember(x)  # the time integral of the past first, then the prediction in time
            self.started = True
            self.fc = super().step(x, hours, tisr_next)
            return self.fc
    return MemCore


def make_core(static: torch.Tensor, inc_std: np.ndarray, mean: np.ndarray, std: np.ndarray, lat_deg,
              hidden: int = 48, blocks=(2, 2), norm: str = "bn", substeps: int = 1, stride: int = 1,
              hist: bool = False, mem: int = 0, mem_ch: int = 24, mz: bool = False, gfeat: bool = False,
              nomem: bool = False, acc=(), acc_ch: int = 12):
    """Step 16's DynCore on daily data: ``t6`` is the std of the DAILY increment, one solver step = one day.

    ``norm`` "bn" = ClimODE's block as it is; "gn" = stateless_norm (the drift fix). ``substeps`` Heun steps
    per day (DynCore's dt_hours, in units of its nominal 6 h step). ``stride`` > 1 = the long-scale core: one
    solver step = ``stride`` days (``inc_std`` is then the std of the stride-day increment). ``hist`` = the
    field extended to its history: the previous state enters as the observed trend (_hist_core_class).
    ``mem`` = W > 0: the memory core (_mem_core_class), a ConvGRU state of ``mem_ch`` channels warmed up on the
    W days before the IC; ``mz`` = its Mori-Zwanzig sequence form (innovation input, forecasts of the sequence).
    ``nomem`` (with mem and mz) = the control of the memory term: the same sequence training with a zero memory.
    ``acc`` (with mem) = the accumulated memory: time constants in days of fixed exponential accumulators of the
    memory's inputs (``acc_ch`` channels each) and their own long-memory term in the tendency.
    ``gfeat`` = the physical terms: the operator's tendency G(x) as 48 input channels (step 18); the caller attaches
    ``core.features`` (operator_features) and fills ``core.gscale``. With ``mem`` the memory integrates G as well."""
    from .dyncore import DynCore, DynCoreConfig
    if gfeat and hist and not mem:
        raise ValueError("the history core has no G(x) input; use mem for a memory of the physical terms")
    if len(acc) and not mem:
        raise ValueError("the accumulated memory belongs to the memory core (mem > 0)")
    cfg = DynCoreConfig(hidden=hidden, blocks=tuple(blocks), dt_hours=6.0 / substeps, solver="heun", positive=True,
                        gfeat=gfeat)
    args = (cfg, static, torch.as_tensor(inc_std[DYN], dtype=torch.float32), mean[DYN], std[DYN], lat_deg)
    if mem:
        core = _mem_core_class()(*args, mem=mem, mem_ch=mem_ch, mz=mz, nomem=nomem, acc=tuple(acc), acc_ch=acc_ch)
    else:
        core = (_hist_core_class() if hist else DynCore)(*args)
    if norm == "gn":
        stateless_norm(core.net)
    elif norm != "bn":
        raise ValueError(norm)
    core.stride = int(stride)
    return core


class CachedFeatures:
    """One-entry cache of the operator features: the memory update and the solver step of the same day ask for
    G of the same state tensor."""

    def __init__(self, feats):
        self.feats, self.key, self.val = feats, None, None

    def __call__(self, x):
        if self.key is not x:
            self.key, self.val = x, self.feats(x)
        return self.val


def operator_features(mean, std, x_full: torch.Tensor, states: Optional[torch.Tensor] = None, scale=None):
    """Step 18's operator features on KAI's daily contract: G(x) = -R0(x), the discrete primitive-equation
    operator's own tendency of (u, v, T, q) on the 12 levels (48 physical terms) at a normalized state of the 66
    dynamic channels. ``x_full`` (>= 1, 67, H, W) normalized: its orography channel is the lower boundary.
    ``states`` (N, 66, H, W): estimate the per-channel scale (RMS of G over the valid points) from them."""
    from .coupling import PhysicsConfig, PhysicsGuide
    from .dyncore import OperatorFeatures, StrongPhysics
    H, W = x_full.shape[-2:]
    guide = PhysicsGuide(H, W, mean, std, PhysicsConfig(dt_seconds=24 * 3600.0))
    for q in guide.parameters():
        q.requires_grad_(False)
    feats = OperatorFeatures(StrongPhysics(guide, torch.ones(contract.N_DYNAMIC)), x_full[:1, [OROG]].float(), scale)
    if states is not None:
        feats.scale = feats.estimate_scale(states)
    return feats


def core_step(core, xd: torch.Tensor, doy: torch.Tensor, tisr_next: Optional[torch.Tensor]) -> torch.Tensor:
    """One day of the core on the 66 dynamic channels (season phase from the day of year)."""
    return core.step(xd, doy.float() * 24.0, tisr_next)


def core_rollout(core, xd, doy, steps, tisr=None, prev=None):
    """``steps`` solver steps of core.stride days each; tisr (B, steps, H, W) = the forcing at the end of each.
    ``prev`` (history cores): the dynamic channels one step before xd; memory cores: the W steps before xd
    (B, W, 66, H, W), oldest first (None = an empty memory)."""
    x, out, st = xd, [], getattr(core, "stride", 1)
    if getattr(core, "mem", 0):
        core.reset(prev, doy.float() * 24.0, xd)
    elif getattr(core, "hist", False):
        core.prev = prev
    for k in range(steps):
        x = core_step(core, x, doy + k * st, None if tisr is None else tisr[:, k])
        out.append(x)
    return torch.stack(out, 1)


class CoreBase:
    """The frozen tendency core as a map of the full 67-channel state: (x, doy, tisr_next) -> next-day state.
    ``prev`` (history cores): the full state one day before x; None = no trend.

    A memory core keeps its state across the steps, so it can only be stepped along ONE trajectory at a time
    (``stateful=True``, a coupled rollout): ``start(history, doy, x0)`` warms the memory up on the days before the
    IC, then every call advances the memory with the state it is given. In a coupled rollout that is the mixed
    state, so the memory integrates the trajectory actually followed, and a Mori-Zwanzig core sees the difference
    between the mixed state and its own last forecast as the innovation."""

    def __init__(self, core, stateful: bool = False):
        self.mem = getattr(core, "mem", 0)
        if self.mem and not stateful:  # e.g. inside KAI's training, where every batch is a new trajectory
            raise NotImplementedError("a memory core is stepped along one trajectory only (CoreBase(core, stateful=True))")
        self.core, self.dyn, self.hist = core, torch.tensor(DYN), getattr(core, "hist", False) and not self.mem

    def start(self, history, doy, x0):
        """history (B, W, 67, H, W): the full states of the W days before the IC x0 (B, 67, H, W), oldest first."""
        self.core.reset(None if history is None else history.index_select(-3, self.dyn), doy.float() * 24.0,
                        x0[:, self.dyn])

    def __call__(self, x, doy, tisr_next=None, prev=None):
        if self.hist:
            self.core.prev = None if prev is None else prev[:, self.dyn]
        return x.index_copy(1, self.dyn, core_step(self.core, x[:, self.dyn], doy, tisr_next))  # orog copied


# ------------------------------------------------------------------------------- KAI with the core inside
@dataclass
class KAIPosConfig:
    depths: tuple = (3, 3, 15, 3)        # KAI.yaml / models.py default
    dims: tuple = (48, 96, 192, 288)
    form: str = "full"
    inject: str = "none"

    @classmethod
    def arm(cls, name: str, depths, dims) -> "KAIPosConfig":
        form, inject = ARMS[name]
        return cls(depths=tuple(depths), dims=tuple(dims), form=form, inject=inject)


class KAIPos(nn.Module):
    """x (B, 67, H, W) z-scored, doy (B,), tisr_next (B, H, W) -> next-day state (B, 67, H, W).

    ``core`` is the frozen first stage, a callable (x, doy, tisr_next) -> next state: CoreBase(tendency core),
    or a trained KAI for the control arm ``stack`` (not a submodule: neither trained nor saved with KAI)."""

    def __init__(self, cfg: KAIPosConfig, inc_std: np.ndarray, core=None):
        super().__init__()
        self.cfg = cfg
        n = contract.N_CHANNELS
        self.kai = KAI(in_chans=n * (2 if cfg.inject in ("in", "hist") else 1), out_chans=n,
                       depths=list(cfg.depths), dims=list(cfg.dims))
        if cfg.form in ("res", "core"):
            last = self.kai.core_model.head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        if cfg.form == "gate":
            self.gate = nn.Parameter(torch.zeros(1, n, 1, 1))
        if cfg.inject.startswith("s"):
            self.stage = int(cfg.inject[1]) - 1
            self.proj = nn.Conv2d(n, cfg.dims[self.stage], 1)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.register_buffer("s", torch.as_tensor(inc_std, dtype=torch.float32).view(1, -1, 1, 1))
        keep = torch.ones(n)
        keep[OROG] = 0
        self.register_buffer("keep", keep.view(1, -1, 1, 1))
        self.register_buffer("dyn", torch.tensor(DYN))
        self._core = [core]
        self.last_core = None
        self.ablate = False  # diagnostic: zero the injected tau (the output anchor of form "core" is kept)
        # rollout training: the gradient of a later day's loss has to pass through the frozen core's input
        # (c(x) = x + ...: without it the main path of forms "core" / "gate" is cut). Off = single-step training.
        self.core_grad = False
        self.uses_core = cfg.form in ("core", "gate") or cfg.inject not in ("none", "hist")
        if self.uses_core and core is None:
            raise ValueError("this arm needs the frozen tendency core")

    def core_next(self, x, doy, tisr_next, prev=None):
        with torch.set_grad_enabled(self.core_grad and torch.is_grad_enabled()):
            return self._core[0](x, doy, tisr_next, prev=prev)

    def forward(self, x, doy, tisr_next=None, core_state=None, prev=None):
        """``core_state``: evaluate the core on this state instead of x (the open-loop diagnostic of ``rollout``).
        ``prev``: the state one day before (a history core's trend, or arm kaih's own input); None = no trend."""
        cfg, c, tau = self.cfg, None, None
        if self.uses_core:
            xc = x if core_state is None else core_state
            c = self.core_next(xc, doy, tisr_next, prev)
            tau = (c - xc) / self.s
        elif cfg.inject == "hist":
            tau = torch.zeros_like(x) if prev is None else (x - prev) / self.s * self.keep
        if self.ablate and tau is not None:
            tau = torch.zeros_like(tau)
        m = self.kai.core_model
        h = torch.cat([x, tau], 1) if cfg.inject in ("in", "hist") else x
        for i in range(4):
            h = m.stages[i](m.downsample_layers[i](h))
            if cfg.inject.startswith("s") and i == self.stage:
                h = h + self.proj(tau)
        o = m.head(h)
        self.last_core = c
        if cfg.form == "full":
            return o
        if cfg.form == "gate":
            return o + self.gate * self.keep * (c - o)
        return (x if cfg.form == "res" else c) + o * self.s * self.keep

    def rollout(self, x0, doy0, steps, tisr=None, open_loop=False, prev=None):
        """(B, steps, 67, H, W); tisr (B, steps, H, W) = the core's forcing on day 1..steps.

        ``open_loop`` (diagnostic, inject arms): the core runs on its OWN trajectory from x0 and only its tau
        enters KAI, so KAI's output never feeds the core (closed loop = the trained coupling, the default).
        ``prev``: the state one day before x0; later steps use the trajectory's own previous state."""
        x, xc, p, out = x0, x0, prev, []
        for k in range(steps):
            y = self(x, doy0 + k, None if tisr is None else tisr[:, k], xc if open_loop else None, prev=p)
            p = xc if open_loop else x  # the state the next step's trend starts from
            if open_loop:
                xc = self.last_core
            x = y
            out.append(x)
        return torch.stack(out, 1)


def config_dict(cfg) -> dict:
    return asdict(cfg)


__all__ = ["ARMS", "ConvGRUCell", "CoreBase", "KAIPos", "KAIPosConfig", "LpLoss", "make_core", "stateless_norm", "core_step", "core_rollout", "needs_core",
           "config_dict", "TISR", "OROG", "DYN", "TISR_DYN", "operator_features", "CachedFeatures"]
