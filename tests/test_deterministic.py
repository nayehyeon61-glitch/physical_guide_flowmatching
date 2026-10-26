import torch

from pgcfm.ens_to_det import EnsembleToDeterministic, mean_only, self_test
from scripts.run_step10_deterministic import LocalCNN


def test_owner_operator_self_test():
    self_test()


def test_local_cnn_starts_at_mean():
    torch.manual_seed(0)
    x, ctx = torch.randn(2, 4, 3, 10, 12), torch.randn(2, 2, 10, 12)
    m = LocalCNN(3, 2, width=8, layers=2)
    torch.testing.assert_close(m(x, ctx), x.mean(1))


def test_mean_only_operator_ignores_spread():
    torch.manual_seed(0)
    m = EnsembleToDeterministic(2, context_channels=1, width=8, modes=2)
    torch.nn.init.normal_(m.head.weight, std=0.1)
    x, ctx = torch.randn(2, 4, 2, 8, 8), torch.randn(2, 1, 8, 8)
    y = x.mean(1, keepdim=True) + 3 * (x - x.mean(1, keepdim=True))  # same mean, larger spread
    torch.testing.assert_close(m(mean_only(x), ctx), m(mean_only(y), ctx))
    assert not torch.allclose(m(x, ctx), m(y, ctx))
