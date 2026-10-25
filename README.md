# physical_guide_flowmatching

PG-CFM: physics-guided conditional flow matching for autoregressive daily weather ensembles,
built on the KAI-α ERA5 data contract (1.5°, 121×240, 67 channels).

| Path | Purpose |
| --- | --- |
| `pgcfm/contract.py` | Channel order, units, normalization stats, grid, below-ground mask, constants |
| `pgcfm/data.py` | K-step sequence dataset on top of the KAI loader (dynamic/static split, day-of-year) |
| `pgcfm/cfm.py` | Step 2: CFM transition on torchcfm (path, U-Net, Heun, member-wise rollout) |
| `pgcfm/physics/` | Step 3: primitive-equation residuals (4 constraint groups), grid diagnostics, PyTorch port of Raissi's PINN |
| `pgcfm/metrics.py` | Area-weighted CRPS, RMSE, spread, spread–skill |
| `pgcfm/coupling.py` | Step 4: PG-CFM coupling (bridge B_psi eq. 9, closure, joint loss eq. 12, L_phys -> y_hat -> theta) |
| `pgcfm/train.py` | Step 4: appendix C training phases (cfm -> guide -> joint) and 1-day validation |
| `pgcfm/eval/` | Step 5: NeuralGCM / WeatherBench2 evaluation: scores, physical consistency, baselines, figures |
| `third_party/kai/` | Vendored KAI-Atmos code (BSD-3) |
| `third_party/pinns/` | Reference copy of maziarraissi/PINNs `Burgers.py` (MIT, TF1, not imported) |
| `third_party/weatherbench2/` | Vendored WeatherBench2 metrics (Apache-2.0) |
| `configs/pgcfm_era5.yaml` | PG-CFM training config for the real KAI ERA5 data |
| `scripts/train_pgcfm.py` | Train CFM-only or PG-CFM on KAI ERA5 (single GPU or `torchrun`), checkpoint/resume |
| `scripts/infer_pgcfm.py` | Ensemble + eps=0 rollout into KAI `forecast.nc` |
| `scripts/evaluate.py` | Score KAI / PG-CFM `forecast.nc` files against ERA5 (+ climatology builder) |
| `scripts/step3_era5_residual_stats.py` | Residual RMS of observed ERA5 pairs, for the scales b_j (run on the data cluster) |
| `scripts/make_synthetic_era5.py`, `scripts/make_balanced_era5.py` | Synthetic files in the KAI NetCDF layout (tests / toy experiments) |
| `scripts/run_step2_cfm.py`, `scripts/step3_residual_experiment.py`, `scripts/run_step4_pgcfm.py`, `scripts/eval_synthetic_demo.py` | Per-step small experiments (docs/) |
| `docs/RUN_ON_GPU.md` | How to train and evaluate on the GPU server |
| `docs/` | Per-step implementation notes and weakness analysis (`docs/step4_pgcfm.md`: coupling results) |

```bash
pip install -r requirements.txt
python -m pytest
```

## Step13 ODE comparison branch

`experiment/step13-spherical-ode` extends commit `276a61f` with the original
66-channel `SeqFM`/`PINNProjector` and an existing `torchdiffeq` physical-time
integrator. `K=60` covers 15 days at six-hour output. The owner operator
`pgcfm/ens_to_det.py` is unchanged. See
[experiment protocol and results](docs/step13_ode_experiment.md).

The CPU pipeline experiment uses explicitly synthetic 32×64 KAI-format data;
its errors do **not** establish ERA5 forecasting skill. Public observed ERA5
requires network access to `storage.googleapis.com`.
