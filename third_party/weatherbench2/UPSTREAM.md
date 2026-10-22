# Vendored: WeatherBench 2

- Source: https://github.com/google-research/weatherbench2, commit `1f1ff39b954fdb79c96e11ad9750212eb7dd14eb`
- License: Apache 2.0, Copyright 2023 Google LLC (see `LICENSE`)
- Files: `weatherbench2/{__init__,metrics,utils,thresholds,regions,derived_variables,schema}.py`, unmodified.
  The beam/zarr pipeline (`evaluation.py`, `config.py`) is left out; PG-CFM calls the
  metric classes directly on in-memory xarray datasets (`pgcfm/eval/`).
- NeuralGCM (Kochkov et al. 2024, Nature 632) reports its scores with this library, so
  using it keeps our numbers comparable to the paper's definitions.
- Runtime deps: numpy, scipy, xarray, pandas, fsspec (imported by `utils.py`).
