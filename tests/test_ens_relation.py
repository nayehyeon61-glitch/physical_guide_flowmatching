import torch

from pgcfm.ens_relation import (EnergyConstraint, EnsembleRelationFNO, fit_relation, fno_norms, project_fno_,
                                relation_refine)
from pgcfm.ens_to_det import EnsembleToDeterministic


class LOOMean(torch.nn.Module):
    def forward(self, x):
        return (x.sum(1, keepdim=True) - x) / (x.shape[1] - 1)


def test_relation_fno_is_permutation_equivariant_and_projection_bounds_every_block():
    torch.manual_seed(0)
    G = EnsembleRelationFNO(3, width=8, modes=(3, 3), n_layers=2)
    x = torch.randn(2, 5, 3, 16, 32)
    perm = torch.tensor([3, 0, 4, 1, 2])
    with torch.no_grad():
        torch.testing.assert_close(G(x[:, perm]), G(x)[:, perm], atol=1e-5, rtol=1e-4)
    G.project_(1.0)
    for b in G.blocks:
        assert float(b.spec.l2_operator_norm()) <= 0.5 + 1e-5
        assert float(b.l2_operator_norm()) <= 1.0 + 1e-4


def test_fit_relation_runs_with_the_energy_constraint_and_bound():
    torch.manual_seed(0)
    G = EnsembleRelationFNO(2, width=8, modes=(2, 2), n_layers=2)
    x = torch.randn(6, 4, 2, 8, 16)
    info = fit_relation(G, x[:4], x[4:], torch.ones(8, 16), epochs=2, constraint=EnergyConstraint(alpha=0.8),
                        bound=2.0, log=lambda q: None)
    assert len(info["history"]) == 2 and info["history"][-1]["op_norm_max"] <= 2.0 + 1e-4


def test_relation_refine_with_the_loo_mean_is_a_uniform_contraction():
    """G = LOO mean: y_k - G(y_{-k}) = a d_k with a = N / (N - 1), and the full gradient of the consistency
    term (direct and through the other members) is a^2 d_k, so one step scales the deviations by 1 - tau a^2
    and keeps the mean; the over-dispersion term alone (E_ref = 0) scales them by 1 - tau."""
    torch.manual_seed(0)
    y = torch.randn(3, 4, 2, 8, 16)
    w = torch.rand(8, 16) + 0.5
    d = y - y.mean(1, keepdim=True)
    a = 4 / 3
    r = relation_refine(LOOMean(), y, torch.zeros(3), w, tau=0.1, steps=1, gamma_c=1.0, gamma_e=0.0)
    torch.testing.assert_close(r.mean(1), y.mean(1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(r - r.mean(1, keepdim=True), (1 - 0.1 * a ** 2) * d, atol=1e-5, rtol=1e-4)
    r = relation_refine(LOOMean(), y, torch.zeros(3), w, tau=0.1, steps=1, gamma_c=0.0, gamma_e=1.0)
    torch.testing.assert_close(r - r.mean(1, keepdim=True), 0.9 * d, atol=1e-5, rtol=1e-4)


def test_project_fno_bounds_the_neuraloperator_blocks_and_keeps_the_mean_start():
    torch.manual_seed(0)
    op = EnsembleToDeterministic(2, width=8, modes=3)
    project_fno_(op, 1.0)
    for r, w in fno_norms(op):
        assert r <= 0.5 + 1e-5 and w <= 0.5 + 1e-5
    x = torch.randn(2, 4, 2, 8, 8)
    torch.testing.assert_close(op(x), x.mean(1))
