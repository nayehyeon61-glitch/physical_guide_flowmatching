# Vendored: U-Cast

- Source: https://github.com/Rose-STL-Lab/u-cast, commit `f6ca2ae408f9c36174dadc7902a08e64daff123b`
- Paper: Rühling Cachay, Watson-Parris, Yu, "U-Cast: A Surprisingly Simple and Efficient Frontier
  Probabilistic AI Weather Forecaster", ICML 2026 (arXiv 2604.09041).
- License: Apache-2.0 (see `LICENSE`). `src/models/networks_edm.py` keeps NVIDIA's header
  (CC BY-NC-SA 4.0, from the EDM code base); research use only.
- Only the files the step 10 baselines need are copied, unchanged: the `DhariwalUNet` backbone
  and its `BaseModel` (ensemble forward, inference dropout), the weighted MAE / fair CRPS losses,
  the Muon optimizer, `HybridOptim`, `LinearWarmupCosineAnnealingLR`, the circular padding function and `utilities/utils.py`.
- `pgcfm/baselines.py` puts this folder on `sys.path` (package name `src`) and applies the
  upstream `pad_before_conv2d_circular_height_only` only inside the U-Net forward, instead of the
  upstream global monkey patch of `torch.nn.functional.conv2d` (which would break every
  `nn.Conv2d` of the PG-CFM models in the same process).

## Local changes
None.
