import pytest
import torch

from physical_guide_flowmatching import CLIMODE_VARIABLES, GlobalLatLonGrid, prepare_era5_state


def test_reorders_latitude_and_wrapped_longitude_together_with_all_fields():
    grid = GlobalLatLonGrid(6, 12)
    physical = torch.arange(72, dtype=torch.float64).reshape(1, 6, 12)
    lat = grid.latitude_degrees.flip(0)
    # Simulate a NetCDF file with [-180,180) longitude in increasing order.
    lon, order = ((grid.longitude_degrees + 180).remainder(360) - 180).sort()
    raw = physical.flip(-2).index_select(-1, order)
    fields = {name: raw + index for index, name in enumerate(CLIMODE_VARIABLES)}
    actual = prepare_era5_state(fields, lat, lon, grid=grid)
    expected = torch.stack([physical + index for index in range(5)], dim=-3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (1, 5, 6, 12)


def test_data_preparation_preserves_gradient():
    grid = GlobalLatLonGrid(6, 12)
    field = torch.rand(1, 6, 12, dtype=torch.float64, requires_grad=True)
    result = prepare_era5_state(
        {"t2m": field}, grid.latitude_degrees, grid.longitude_degrees,
        grid=grid, variables=("t2m",),
    )
    result.sum().backward()
    torch.testing.assert_close(field.grad, torch.ones_like(field))


def test_different_resolution_cannot_silently_pass_as_weatherbench_grid():
    grid = GlobalLatLonGrid(6, 12)
    with pytest.raises(ValueError):
        prepare_era5_state(
            {"t2m": torch.ones(1, 6, 12)},
            grid.latitude_degrees + 1.0, grid.longitude_degrees,
            grid=grid, variables=("t2m",),
        )


def test_missing_climode_channel_is_explicit_error():
    grid = GlobalLatLonGrid(6, 12)
    with pytest.raises(ValueError, match="missing ERA5 channel"):
        prepare_era5_state({}, grid.latitude_degrees, grid.longitude_degrees, grid=grid)
