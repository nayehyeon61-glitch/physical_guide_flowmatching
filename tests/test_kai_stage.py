"""The tendency core inside every KAI stage (pgcfm.kai_tend arms stg, stgg, outg; docs/kai_stage_core.md)."""
import numpy as np
import pytest
import torch

from pgcfm import contract
from pgcfm.kai_tend import ARMS, N_STAGES, CoreBase, KAIPos, KAIPosConfig, make_core, needs_core

N, H, W = contract.N_CHANNELS, 8, 16
DEPTHS, DIMS = (1, 1, 1, 1), (8, 16, 16, 24)
INC = np.full(N, 0.5, dtype=np.float32)


class FakeCore:
    """A frozen 'core' as KAIPos sees it: (x, doy, tisr_next, prev) -> next state; with ``mem`` it has to be
    started on the days before the IC and counts its steps."""

    def __init__(self, mem=0):
        self.mem, self.started, self.calls = mem, None, 0

    def start(self, history, doy, x0):
        self.started, self.calls = history, 0

    def __call__(self, x, doy, tisr_next=None, prev=None):
        if self.mem:
            assert self.started is not None, "a memory core is stepped before it was started"
        self.calls += 1
        return x + 0.3 * torch.roll(x, 1, -1)


def model(arm, core=None, **kw):
    torch.manual_seed(0)
    return KAIPos(KAIPosConfig.arm(arm, DEPTHS, DIMS), INC, core, **kw)


def batch():
    torch.manual_seed(1)
    return torch.randn(2, N, H, W), torch.tensor([3, 40])


def test_the_new_arms_need_the_core():
    assert all(needs_core(a) for a in ("stg", "stgg", "outg"))
    assert ARMS["stg"] == ("full", "all") and ARMS["outg"][1] == "none"
    with pytest.raises(ValueError):
        model("stg")


def test_stg_starts_as_the_kai_it_was_initialized_from():
    x, doy = batch()
    kai, stg = model("kai"), model("stg", FakeCore())
    miss = stg.load_state_dict(kai.state_dict(), strict=False)
    assert all(k.startswith("projs.") for k in miss.missing_keys) and not miss.unexpected_keys
    assert len(stg.projs) == N_STAGES and [q.out_channels for q in stg.projs] == list(DIMS)
    assert torch.equal(stg(x, doy), kai(x, doy))


def test_stgg_starts_as_the_loop_with_the_given_weights():
    x, doy = batch()
    core = FakeCore()
    g0 = np.linspace(0.0, 1.0, N).astype(np.float32)
    kai, m = model("kai"), model("stgg", core, gate_init=g0)
    m.load_state_dict(kai.state_dict(), strict=False)
    g = torch.from_numpy(g0).view(1, N, 1, 1)
    want = (1 - g) * kai(x, doy) + g * core(x, doy)  # the loop's daily mix (run_kai_tendency.py loop_fn)
    assert torch.allclose(m(x, doy), want, atol=1e-6)
    out = model("outg", core, gate_init=0.25)
    assert not hasattr(out, "projs") and torch.allclose(out.gate, torch.full((1, N, 1, 1), 0.25))


def test_every_stage_and_the_gate_get_a_gradient_and_the_gate_stays_a_mix():
    x, doy = batch()
    m = model("stgg", FakeCore(), gate_init=0.0)  # at the lower bound: the gradient must still pass the clamp
    m.core_grad = True
    y = m.rollout(x, doy, 2)
    y.square().mean().backward()
    # the first step from zero: only the output layer of each new path has a gradient (the 1x1 convs are zero)
    assert all(float(q.weight.grad.abs().sum()) > 0 for q in m.projs)
    assert float(m.gate_head.weight.grad.abs().sum()) > 0 and float(m.gate.grad.abs().sum()) > 0
    with torch.no_grad():
        m.gate_head.bias.fill_(5.0)
    m.gate_log = []
    m(x, doy)
    assert float(m.gate_log[0].min()) == 1.0  # the weight never leaves [0, 1]: the output stays a mix


def test_switching_one_stage_off_changes_the_forecast_only_through_that_stage():
    x, doy = batch()
    m = model("stg", FakeCore())
    with torch.no_grad():
        for q in m.projs:
            q.weight.normal_(std=0.1)
    m.stage_log = [[] for _ in range(N_STAGES)]
    full = m(x, doy)
    assert all(len(v) == 1 and v[0] > 0 for v in m.stage_log)
    m.stage_log = None
    outs = []
    for i in range(N_STAGES):
        m.stage_off = (i,)
        outs.append(m(x, doy))
        assert not torch.allclose(outs[-1], full)
    m.stage_off = tuple(range(N_STAGES))
    m2 = model("kai")
    m2.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("projs.")})
    assert torch.allclose(m(x, doy), m2(x, doy), atol=1e-6)  # all stages off = no core inside
    m.stage_off, m.ablate = (), True
    assert torch.allclose(m(x, doy), m2(x, doy), atol=1e-6)  # tau = 0: only the conv biases (zero) remain


def test_a_memory_core_inside_kai_is_started_on_the_days_before_the_ic():
    x, doy = batch()
    core = FakeCore(mem=3)
    m = model("stg", core)
    assert m.mem == 3
    hist = torch.randn(2, 3, N, H, W)
    y = m.rollout(x, doy, 4, prev=hist)
    assert y.shape == (2, 4, N, H, W) and core.started is hist and core.calls == 4
    assert model("kai").mem == 0


def test_a_real_memory_core_runs_inside_the_stages():
    lat = np.linspace(-78.75, 78.75, H)
    contract.register_latitudes(lat)
    mean, std = np.zeros(N), np.ones(N)
    core = make_core(torch.zeros(8, H, W), INC, mean, std, lat, hidden=8, blocks=(1, 1), norm="gn", mem=2, mem_ch=4,
                     mz=True)
    for q in core.parameters():
        q.requires_grad_(False)
    m = model("stgg", CoreBase(core.eval(), stateful=True), gate_init=0.5)
    m.core_grad = True
    x, doy = batch()
    y = m.rollout(x, doy, 2, tisr=torch.zeros(2, 2, H, W), prev=torch.randn(2, 2, N, H, W))
    y.square().mean().backward()
    assert torch.isfinite(y).all() and all(q.weight.grad is not None for q in m.projs)
