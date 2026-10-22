import numpy as np

from pgcfm import contract


def test_channel_layout_matches_format_doc():
    assert contract.N_CHANNELS == 67
    assert contract.N_DYNAMIC == 66
    expected = {"z50": 0, "z500": 6, "z1000": 11, "q50": 12, "t850": 33, "u50": 36,
                "v1000": 59, "t2m": 60, "sp": 61, "tisr": 62, "tcwv": 63, "orog": 64,
                "msl": 65, "skt": 66}
    for name, i in expected.items():
        assert contract.CHANNEL_INDEX[name] == i
    assert contract.STATIC_CHANNELS == [64]
    assert contract.level_slice("t") == slice(24, 36)
    assert contract.REFERENCE_MEAN.shape == contract.REFERENCE_STD.shape == (67,)


def test_grid_includes_poles_ascending():
    assert contract.LATITUDES[0] == -90.0 and contract.LATITUDES[-1] == 90.0
    assert np.allclose(np.diff(contract.LATITUDES), 1.5)
    assert contract.LONGITUDES[-1] == 358.5


def test_normalize_roundtrip_is_exact_with_eps():
    rng = np.random.default_rng(0)
    m, s = contract.REFERENCE_MEAN, contract.REFERENCE_STD
    x = m[:, None, None] + s[:, None, None] * rng.standard_normal((67, 4, 5))
    back = contract.denormalize(contract.normalize(x, m, s), m, s)
    assert np.allclose(back, x, rtol=1e-12, atol=0)


def test_kai_style_denormalize_bias_on_small_std_channels():
    """KAI inference denormalizes with `std` while training normalizes with
    `std + 1e-8`. For q50/q100 that is a few-percent scale error on the
    anomaly; for all other channels it is negligible."""
    m, s = contract.REFERENCE_MEAN, contract.REFERENCE_STD
    x = m[:, None, None] + 3.0 * s[:, None, None] * np.ones((67, 1, 1))
    kai = contract.denormalize(contract.normalize(x, m, s), m, s, exact_inverse=False)
    rel_anom_err = np.abs((kai - x)[:, 0, 0]) / (3.0 * s)
    q50, q100 = contract.CHANNEL_INDEX["q50"], contract.CHANNEL_INDEX["q100"]
    assert rel_anom_err[q50] > 0.02 and rel_anom_err[q100] > 0.01
    others = [i for i in range(67) if i not in (q50, q100)]
    assert rel_anom_err[others].max() < 1e-3


def test_below_ground_mask():
    sp = np.array([[101000.0, 85000.0, 60000.0]])  # (lat=1, lon=3)
    mask = contract.below_ground_mask(sp)  # (12, 1, 3)
    lv = {p: i for i, p in enumerate(contract.LEVELS_HPA)}
    assert not mask[:, 0, 0].any()
    assert mask[lv[925], 0, 1] and mask[lv[1000], 0, 1] and not mask[lv[850], 0, 1]
    assert mask[lv[700], 0, 2] and not mask[lv[600], 0, 2]
