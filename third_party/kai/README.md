# KAI-α

KAI-α (KAIST AI model for Atmosphere) is an ultralight CNN-based model for global daily weather forecasting. This repository contains the implementation of a modernized convolutional architecture built on InceptionNeXt with a scale-invariant structure and geophysically-aware design, trained to deliver competitive medium-range forecast accuracy with significantly reduced computational requirements

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Layout](#repository-layout)
3. [Installation](#installation)
4. [Data & Configuration](#data--configuration)
5. [Model Architecture](#model-architecture)
6. [Training Workflow](#training-workflow)
7. [Evaluation & Inference](#evaluation--inference)
8. [Monitoring & Logging](#monitoring--logging)
9. [Troubleshooting](#troubleshooting)
10. [Citation](#citation)

---

## Project Overview

- **Goal:** Deliver accurate medium-range global weather forecasts with a lightweight CNN that significantly reduces computational requirements compared to Transformer-based architectures
- **Key ideas:**  InceptionNeXt-based blocks with parallel depthwise convolutions (square, horizontal, and vertical branches), geocyclic padding for geospatial continuity, scale-invariant architecture without upsampling paths, and inverted residual blocks with pointwise convolutions for efficient channel mixing.
- **Scope:** Supports autoregressive rollouts up to 10+ days, trained on ERA5 daily dataset with 67 atmospheric variables at 2.5° resolution, with LP loss function and cosine annealing learning rate schedule.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/models.py` | KAI-α backbone with InceptionNeXt blocks, geocyclic padding, and inverted residual structure. |
| `src/trainer.py` | End-to-end training loop (DDP-ready, AMP, checkpointing) using `LpLoss`. |
| `src/evaluator.py` | Post-training evaluation with autoregressive rollout, latitude-weighted RMSE/ACC metrics, and CSV export. |
| `src/inference.py` | Autoregressive forecast generator — produces de-normalized `.nc` forecast files. |
| `src/utils.py` | DataConfig, NetCDF dataset, normalization, `LpLoss`, climatology generation, logging helpers, YParams loader. |
| `src/eval_logging.py` | Weights & Biases logging helpers: RMSE/ACC tables, per-variable figures, and end-of-run loss curves (also written to disk). |
| `scripts/train.py` | CLI entry-point for launching training experiments from YAML configs. |
| `scripts/run_inference.py` | CLI entry-point for autoregressive inference. |
| `configs/KAI.yaml` | Training and evaluation configuration. |
| `configs/inference.yaml` | Inference configuration (rollout settings, I/O paths). |
| `experiments/` | Default location for checkpoints and logs (auto-created). |


## Installation

1. **Clone & create environment**
   ```bash
   git clone <repo_url> && cd KAI-atmos
   python -m venv .venv && source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # pick the wheel you need
   pip install numpy netCDF4 xarray wandb matplotlib ruamel.yaml tqdm h5py scipy
   ```
3. **Install WeatherBench2** (required for evaluation — the evaluator computes RMSE with
   `weatherbench2.metrics.MSE`)
   ```bash
   pip install weatherbench2==0.2.1
   ```
   >  WeatherBench2 pulls in
   > `xarray`; if you install a newer WeatherBench2 release, keep `xarray`/`numpy` compatible
   > with it. On a shared conda environment you may need
   > `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` so `netCDF4` picks up the
   > right `libstdc++`.

## Data & Configuration

Data ingestion is controlled through YAML configs in `configs/`. Copy and edit the baseline file:

```bash
cp configs/KAI.yaml configs/my_run.yaml
```

Important fields in `configs/KAI.yaml`:

- `train_data_path`, `valid_data_path`: folders of NetCDF files containing ERA5 daily mean atmospheric variables `fields[num_time, num_vars, lat, lon]`.
- `in_channels`, `out_channels`: 67 input/output channels (indices into the NetCDF `fields` variable).
- `global_means_path`, `global_stds_path`: `.npy` stats used for z-score normalization.
- `add_grid`: append coordinate-grid channels to inputs (`gridtype`: linear or sinusoidal).
- `n_history`: must be `0`. History stacking (`n_history > 0`) is rejected at startup because the autoregressive rollout requires `in_channels == out_channels`.
- `n_initial_conditions`, `eval_ic_stride`, `eval_start_timestep`: how many initial conditions the post-training evaluation rolls out and how they are spaced. (Iterative multi-step forecasting is not implemented; the trainer does single-step, autoregressive-at-eval stepping.)
- `exp_dir`: parent folder where `scripts/train.py` writes `run_name/ckpt.tar` and `best_ckpt.tar`.

**Evaluation settings:**

- `enable_evaluation`: set to `true` to run post-training evaluation.
- `test_dataset_path`: path to test dataset directory with NetCDF files.
- `eval_start_timestep`: starting timestep index for initial conditions (default: 0).
- `eval_forecast_steps`: number of forecast days.
- `eval_ic_stride`: stride (in timesteps) between successive initial conditions (default: 1).
- `eval_variable_indices`: 0-based positions into `out_channels` to report detailed per-variable metrics for and to highlight in the aggregate W&B line plots (empty = summary only).
- `eval_checkpoint_path`: custom checkpoint path (uses `best_ckpt.tar` if null).
- `climatology_path`: path to daily climatology `.nc` file (required for ACC computation).


`YParams` loads the YAML block named by `--config` and exposes attributes with dot access while keeping a mutable dict for logging.

## Model Architecture

The KAI-α architecture lives in `src/models.py` and is built on four design principles:

1. **Geodesic-aware depthwise convolutions.** `GeoCyclicPadding` mirrors the poles and wraps longitude to remove edge artifacts before applying filters.
2. **Directional Inception branches.** `InceptionDWConv2d` splits channels into square, horizontal, and vertical depthwise paths with configurable kernel sizes to capture anisotropic transport, then recombines them via concatenation.
3. **Grouped channel mixers.** `PointwiseGroupConv` replaces dense 1x1 projections with grouped 1x1s, which drastically cuts FLOPs while still mixing channels.
4. **Residual InceptionNeXt stages.** Four stages of `Block` modules stack the directional depthwise unit, layernorm (channel-first), GELU, and residual connections with optional layer scaling and drop-path.

The assembled model looks like:

```python
# src/models.py — InceptionNeXt assembly (stem, stages, head)
self.downsample_layers.append(stem)
...
self.stages.append(stage)
...
self.head = nn.Sequential(
    nn.Conv2d(dims[-1], dims[-1], kernel_size=3, stride=1, padding=1),
    nn.GELU(),
    nn.Conv2d(dims[-1], num_classes, kernel_size=1)
)
```

- Spatial resolution is preserved (stride 1) to keep geophysical grids aligned.
- Default `dims=[48, 96, 192, 288]` and `depths=[3, 3, 15, 3]`, but feel free to edit the constructor call in `Trainer`.
- The outer `KAI` module is a thin wrapper that exposes `in_chans` / `out_chans` inferred from the dataset.

### Input pipeline

`ClimateNetCDFDataset` prepares tensors by:

```python
# src/utils.py — reshape_fields preprocessing
img = _apply_normalization(..., normalization=config.normalization, normalize_flag=config.normalize)
...
if config.add_grid and inp_or_tar == "inp":
    img = _add_grid_channels(img, config.gridtype, config.N_grid_channels)
```
- Z-score normalization using pre-computed global means and stds
- Supports optional coordinate grids (linear or sinusoidal).
- Automatic concatenation of history steps by reshaping `(T, C, H, W)` into `(T*C, H, W)` for the backbone.

### Loss & metrics

`Trainer` uses `LpLoss` (relative L2-norm error) as the training objective. This computes `||pred - target||₂ / ||target||₂` per sample and averages across the batch, providing a resolution-invariant loss signal.

Post-training evaluation uses **WeatherBench2** for the RMSE metric and a **latitude-weighted ACC**
(Anomaly Correlation Coefficient):

- **RMSE** is computed with [`weatherbench2.metrics.MSE`](https://github.com/google-research/weatherbench2):
  the squared error is averaged over the grid with cosine-of-latitude area weights **and** over all
  initial conditions, and the square root is taken **last** (`RMSE = sqrt(mean MSE)`).
- **ACC** is the latitude-weighted anomaly correlation of forecast vs. truth anomalies (field minus
  daily climatology), averaged over initial conditions.

Both metrics are reported per variable and per lead day (Day 0 is the initial condition compared to
itself, so RMSE≈0 and ACC≈1).

## Training Workflow

Launch training from the project root:

```bash
python scripts/train.py \
  --yaml_config /abs/path/to/configs/my_run.yaml \
  --config default \
  --run_num 01 \
  --enable_amp                 # optional mixed precision
```

What happens under the hood:

1. `scripts/train.py`
   - loads YAML via `YParams`, injects runtime overrides, and sets up distributed training if `WORLD_SIZE>1`.
   - creates experiment directories `<exp_dir>/<config>_<run_num>/` and logs versions.  
2. `Trainer`
   - draws a sample batch to detect `in_chans/out_chans` and instantiates `KAI`.
   - builds `DataLoader`s with `DistributedSampler` when needed, applies `enable_nhwc`, AMP scaler, and cos-annealing or ReduceLROnPlateau scheduler.
   - runs the epoch loop with gradient clipping, checkpointing (`ckpt.tar`, `best_ckpt.tar`)

The core training loop:

```python
# src/trainer.py — Trainer.train() epoch loop
for epoch in range(self.start_epoch, self.params.max_epochs):
    tr_time, data_time, train_logs = self.train_one_epoch()
    valid_time, valid_logs = self.validate_one_epoch()
    ...
    if self.world_rank == 0 and self.params.save_checkpoint:
        self.save_checkpoint(...)
```

**Distributed mode:** Export `WORLD_SIZE`, `RANK`, `LOCAL_RANK` (e.g., via `torchrun`). The script automatically splits the batch per GPU and wraps the model in `DDP`.

## Evaluation & Inference

The pipeline cleanly separates **evaluation** (metric computation) from **inference** (forecast generation):

| Module | Responsibility | Output |
| --- | --- | --- |
| `src/evaluator.py` | Compute RMSE/ACC from autoregressive rollouts | CSV metrics |
| `src/inference.py` | Generate de-normalized forecast fields | NetCDF forecasts |

### Validation During Training
- `Trainer.validate_one_epoch()` runs after each training epoch with `torch.no_grad()` to compute validation loss on the validation dataset.
- **Printed outputs:** Displays epoch number, validation loss, and elapsed time. Loss values are logged to console and optionally to WandB if enabled.

### Post-Training Evaluation
After training completes, comprehensive evaluation is performed automatically:
- Set `enable_evaluation: true` in your YAML config to trigger post-training evaluation.
- The evaluator runs autoregressive rollouts from configurable initial conditions and computes:
  - **WeatherBench2 RMSE** (`weatherbench2.metrics.MSE`, de-normalized, per channel, per lead day)
  - **Latitude-weighted ACC** (anomaly correlation against daily climatology, per channel, per lead day)
- Day 0 always yields RMSE≈0 and ACC≈1.0 (initial condition compared to itself).
- **Output:** Per-timestep and per-variable metrics exported to `evaluation_metrics.csv`, `rollout_rmse.csv`, and `rollout_acc.csv` in the experiment directory.
- When `log_to_wandb: true`, the evaluation also logs RMSE/ACC tables, per-variable metric-grid
  figures, and aggregate RMSE/ACC-vs-lead-time line plots to the active W&B run (see
  [Monitoring & Logging](#monitoring--logging)). The figures are additionally saved as PNGs under
  `<experiment_dir>/eval_plots/`.

### Standalone Inference
For generating forecast fields without computing metrics:
```bash
python scripts/run_inference.py --config configs/inference.yaml
```
**Key configuration parameters in `configs/inference.yaml`:**
- `checkpoint_path`: Path to trained model checkpoint (`.tar` file)
- `input_nc_path`: NetCDF file or directory containing test data
- `forecast_steps`: Number of forecast days to produce
- `n_initial_conditions`: How many initial conditions to generate forecasts for
- `ic_stride`: Stride between initial conditions in the test file
- `output_dir`: Directory where `forecast.nc` will be saved
- `global_means_path`, `global_stds_path`: Normalization statistics (same as training)

**Output:** A single `forecast.nc` file with dimensions `(initial_condition, lead_time, channel, lat, lon)` containing de-normalized forecast fields. Channel indices are stored as a coordinate variable for traceability.

**Note:** The inference module is independent of the training pipeline and can be integrated into operational forecast systems.


## Monitoring & Logging

- **Console/file logs:** `setup_logging()` sets a consistent format and writes to `<exp_dir>/out.log`.
- **CSV metrics:** the evaluator writes `evaluation_metrics.csv`, `rollout_rmse.csv`, and
  `rollout_acc.csv` to the experiment directory (independent of W&B).
- **Checkpoints:** each save stores the epoch, iteration counter, model weights, and optimizer state
  (`ckpt.tar`, and `best_ckpt.tar` for the best validation loss).

### Weights & Biases

W&B logging is **optional** and controlled entirely from the YAML config — it is off by default, so
no account is needed to train or evaluate. Enable it by setting `log_to_wandb: true`.

**Authentication.** Log in once per machine with `wandb login` (or place your key in `~/.netrc`, or
export `WANDB_API_KEY`). To run without contacting the W&B servers, set `wandb_mode: 'offline'`
(runs are written to `./wandb/` and can be synced later with `wandb sync`) or `wandb_mode: 'disabled'`.

**Configuration keys** (in `configs/KAI.yaml`):

| Key | Purpose |
| --- | --- |
| `log_to_wandb` | Master switch for W&B logging (training **and** evaluation). |
| `wandb_project` | W&B project name (default `KAI`). |
| `wandb_entity` | W&B user/team; `null` uses your default account. |
| `wandb_group` | Run group; `null` auto-uses `KAI_<config>`. |
| `wandb_tags` | Optional list of run tags. |
| `wandb_mode` | `online` \| `offline` \| `disabled`. |
| `early_stopping_patience` | Epochs of no val-loss improvement before stopping (`0` disables). |

Only rank 0 logs to W&B in distributed runs.

**Logged during training** (per epoch):

- `train_loss` — training loss.
- `valid_loss`, `valid/l1` — validation loss and L1.
- `lr` — learning rate.
- `epoch_time_sec`, `train_time_sec`, `valid_time_sec`, `data_time_sec` — training time breakdown.
- `peak_memory_gb` — peak GPU memory (`torch.cuda.max_memory_allocated`) — GPU usage.
- `early_stopping/best_valid_loss`, `early_stopping/epochs_without_improvement`,
  `early_stopping/patience`, and `early_stopping/stopped` (logged once when early stopping fires).

**Logged during evaluation** (post-training, same run):

- `eval/rmse_table`, `eval/acc_table` — RMSE and ACC per variable × lead day.
- `eval/rmse_all_variables`, `eval/acc_all_variables` — per-variable metric-grid figures.
- `eval/rmse_vs_leadtime`, `eval/acc_vs_leadtime` — aggregate line plots across all timesteps.
- `eval/rmse_mean`, `eval/acc_mean`, and per-highlighted-variable `eval/rmse/<var>` /
  `eval/acc/<var>` — **native W&B line series** plotted against the custom `eval/lead_time`
  step metric, so curves overlay across runs for **model-to-model comparison**.
- `eval/rmse_line_series`, `eval/acc_line_series` — interactive multi-variable line panels
  (every output channel on one chart).
- `eval/final_lead_mean_rmse`, `eval/final_lead_mean_acc`, and per-highlighted-variable scalars
  (from `eval_variable_indices`) in the run summary.

**Loss curves** (logged at the end of training):

- `loss/curve` — native train-vs-valid loss line plot (comparable across models);
  `loss/curve_image` — static image fallback.

**Files written to disk** (independent of W&B):

- Combined figures → `<experiment_dir>/eval_plots/*.png`.
- One plot per output channel → `<experiment_dir>/eval_plots/per_variable/rmse/<var>.png` and
  `.../per_variable/acc/<var>.png`.
- Loss curve and history → `<experiment_dir>/loss_curve.png` and `<experiment_dir>/loss_history.csv`.

## Troubleshooting

- **CUDA OOM**
  - lower `batch_size`
  - set `enable_amp: true`
  - toggle `enable_nhwc` for better memory layout on Ampere+ GPUs
- **Divergence / NaNs**
  - reduce `lr` or switch to `ReduceLROnPlateau`
  - verify normalization stats (wrong `.npy` paths will silently degrade performance)
- **Throughput bottlenecks**
  - increase `num_data_workers`
  - pre-cache NetCDF files on local SSD
  - profile `torch.backends.cudnn.benchmark`

Key entry points for debugging:
- Model definition: `src/models.py`
- Data loader & normalization: `src/utils.py`
- Training loop: `src/trainer.py`
- CLI harness: `scripts/train.py`

## Citation

```bibtex
@misc{cheon2025modernizingcnnbasedweatherforecast,
  title={Modernizing CNN-based Weather Forecast Model towards Higher Computational Efficiency},
  author={Minjong Cheon and Eunhan Goo and Su-Hyeon Shin and Muhammad Ahmed and Hyungjun Kim},
  year={2025},
  eprint={2507.10893},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2507.10893}
}
```

---
## Contributers
### Minjong Cheon
### Eunhan Goo 
### Su-Hyeon Shin 
### Muhammad Ahmed
### Hyungjun Kim
For questions or issues, please open a GitHub issue.
