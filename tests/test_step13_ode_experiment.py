import numpy as np
import pytest
import torch

from scripts.run_step13_ode_experiment import (
    cyclic_pairs, daily_leads, normalize_fields, score_members, training_statistics,
    validate_provenance, validate_static_coordinates, validate_time_axis,
)


def test_fifteen_day_daily_leads_are_not_truncated_at_six_days():
    np.testing.assert_array_equal(daily_leads(60), np.arange(1, 16)*24)
    np.testing.assert_array_equal(daily_leads(24), np.arange(1, 7)*24)


def test_training_statistics_exclude_validation_and_test():
    arrays = {2014: np.ones((2, 67, 3, 4), dtype=np.float32),
              2015: np.full((2, 67, 3, 4), 1000.0, dtype=np.float32)}
    mean, std = training_statistics(arrays, [2014])
    np.testing.assert_array_equal(mean, np.ones(67))
    np.testing.assert_array_equal(std, np.zeros(67))


def test_normalization_inverse_retains_q_epsilon():
    from pgcfm.contract import KAI_STD_EPS
    mean = np.zeros(67, dtype=np.float32)
    std = np.full(67, 1e-8, dtype=np.float32)
    fields = np.full((1, 67, 2, 3), 2e-8, dtype=np.float32)
    normalized = normalize_fields(fields, mean, std)
    restored = normalized.numpy() * (std+KAI_STD_EPS)[None, :, None, None]
    np.testing.assert_allclose(restored, fields, rtol=1e-7)


def test_empirical_crps_scores_paired_members_without_collapsing_spread():
    ensemble = torch.tensor([-1., 1.]).reshape(1, 2, 1, 1, 1, 1).expand(1, 2, 3, 1, 2, 4)
    truth = torch.zeros(1, 3, 1, 2, 4)
    result = score_members(ensemble, truth, torch.zeros(1, 2, 4), [-45., 45.], [2.0])
    np.testing.assert_allclose(result["rmse"], 0)
    np.testing.assert_allclose(result["crps"], 1.0)
    np.testing.assert_allclose(result["spread"], 2.0)
    assert np.isnan(result["acc"]).all()


def test_twenty_iterations_of_three_cyclic_pairs_cover_all_sixty_transitions():
    pairs = torch.cat([cyclic_pairs(iteration, 3, 60) for iteration in range(20)])
    torch.testing.assert_close(pairs, torch.arange(60))
    assert len(set(pairs.tolist())) == 60


def test_time_axis_accepts_cf_hours_and_rejects_daily_or_missing_steps():
    metadata = validate_time_axis(np.arange(64)*6, "hours since 2014-01-01")
    assert metadata["cadence_seconds"] == 21600
    for values in (np.arange(64)*24, np.array([0, 6, 18])):
        with pytest.raises(ValueError, match="six-hourly"):
            validate_time_axis(values, "hours since 2014-01-01")


def test_provenance_cannot_relabel_synthetic_data_as_observed_era5():
    manifest = {"kind": "synthetic_pipeline_only", "observed_era5": False}
    assert validate_provenance("synthetic", manifest) is manifest
    with pytest.raises(ValueError, match="synthetic provenance"):
        validate_provenance("era5", manifest)


def test_static_features_cannot_silently_reverse_latitude():
    lat, lon = np.linspace(-87.1875, 87.1875, 32), np.arange(64)*5.625
    validate_static_coordinates(lat, lon, lat, lon)
    with pytest.raises(ValueError, match="constants latitude"):
        validate_static_coordinates(lat[::-1], lon, lat, lon)
