import numpy as np

from scripts.coarsen_ens import cell_weights, coarsen


def test_constant_and_area_mean_preserved():
    x = np.full((3, 2, 64, 128), 7.0, np.float32)
    assert np.allclose(coarsen(x), 7.0)
    r = np.random.default_rng(0).normal(size=(64, 128)).astype(np.float32)
    area = np.diff(np.sin(np.deg2rad(np.linspace(-90, 90, 65))))[:, None]
    area32 = np.diff(np.sin(np.deg2rad(np.linspace(-90, 90, 33))))[:, None]
    assert abs((r * area).sum() / area.sum() / 128 - (coarsen(r) * area32).sum() / area32.sum() / 64) < 1e-5


def test_identity_on_target_grid_and_weights():
    x = np.ones((32, 64), np.float32)
    assert coarsen(x) is x
    assert np.allclose(cell_weights(64, 2).sum(1), 1)


def test_lon0_weights_keep_a_smooth_wave_in_place():
    lon_f = 2.8125 * np.arange(128)
    x = np.tile(np.cos(np.deg2rad(lon_f)).astype(np.float32), (64, 1))
    lon_c = 5.625 * np.arange(64)
    y = coarsen(x, lon0=True)
    assert np.abs(y - np.cos(np.deg2rad(lon_c))).max() < 1e-3
    assert np.abs(coarsen(x) - np.cos(np.deg2rad(lon_c))).max() > 1e-2  # lon pairs are off by half a fine cell
