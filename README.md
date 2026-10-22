# physical_guide_flowmatching

PG-CFM: physics-guided conditional flow matching for autoregressive daily weather ensembles,
built on the KAI-α ERA5 data contract (1.5°, 121×240, 67 channels).

| Path | Purpose |
| --- | --- |
| `pgcfm/contract.py` | Channel order, units, normalization stats, grid, below-ground mask, constants |
| `pgcfm/data.py` | K-step sequence dataset on top of the KAI loader (dynamic/static split, day-of-year) |
| `third_party/kai/` | Vendored KAI-Atmos code (BSD-3) |
| `scripts/make_synthetic_era5.py` | Small synthetic files in the KAI NetCDF layout, for tests only |
| `docs/` | Per-step implementation notes and weakness analysis |

```bash
pip install -r requirements.txt
python -m pytest
```
