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
| `pgcfm/seqdct.py`, `pgcfm/ens_guide.py` | Step 13: sequence 3-D DCT, PINN projector, sequence FM / regression, band-energy calibration guide |
| `pgcfm/flow_sigma.py` | Step 13: flow-dependent spread field on the calibrated members (ClimODE's uncertainty net, or the pretrained operator fine-tuned into sigma) |
| `pgcfm/adaptive_band.py`, `pgcfm/space_band.py`, `pgcfm/ens_relation.py` | Step 13 owner's proposals: case-adaptive band Γ, band × cell spread multipliers, LOO ensemble-relation FNO (`scripts/run_step13_aband.py`, `scripts/run_step13_relation.py`) |
| `scripts/run_step13_best.sh` | Step 13 reported ensembles (regm_band, regm_flow, regm_flowop, regm_sband_clim_flow) end to end: ERA5 download, PINN, FM, regression, calibration, spread field, scores |
| `docs/step13_reproduce.md` | How to run step 13 yourself, with timings and the expected scores |
| `docs/` | Per-step implementation notes and weakness analysis (`docs/step4_pgcfm.md`: coupling results) |

```bash
pip install -r requirements.txt
python -m pytest
```
