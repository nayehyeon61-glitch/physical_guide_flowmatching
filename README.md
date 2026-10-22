# physical_guide_flowmatching

PG-CFM: physics-guided conditional flow matching for autoregressive daily weather ensembles,
built on the KAI-α ERA5 data contract (1.5°, 121×240, 67 channels).

| Path | Purpose |
| --- | --- |
| `pgcfm/contract.py` | Channel order, units, normalization stats, grid, below-ground mask, constants |
| `pgcfm/data.py` | K-step sequence dataset on top of the KAI loader (dynamic/static split, day-of-year) |
| `pgcfm/physics/` | Primitive-equation residuals (4 constraint groups), grid diagnostics, PyTorch port of Raissi's PINN |
| `third_party/kai/` | Vendored KAI-Atmos code (BSD-3) |
| `third_party/pinns/` | Reference copy of maziarraissi/PINNs `Burgers.py` (MIT, TF1, not imported) |
| `scripts/make_synthetic_era5.py` | Small synthetic files in the KAI NetCDF layout, for tests only |
| `scripts/step3_residual_experiment.py` | Residual experiments on analytic fields (docs/step3) |
| `scripts/step3_era5_residual_stats.py` | Residual RMS of observed ERA5 pairs, for the scales b_j (run on the data cluster) |
| `docs/` | Per-step implementation notes and weakness analysis |

```bash
pip install -r requirements.txt
python -m pytest
```
