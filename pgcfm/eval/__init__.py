"""NeuralGCM / WeatherBench2-style evaluation for PG-CFM (step 5).

Metrics come from the vendored WeatherBench2 library (third_party/weatherbench2);
this package only adapts KAI tensors to its xarray conventions and adds the
PG-CFM-specific diagnostics (tendency amplitude, hydrostatic residual, budgets).
"""
