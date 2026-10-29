# Vendored: ClimODE (Verma, Heinonen, Garg, ICLR 2024)

Source: https://github.com/Aalto-QuML/ClimODE @ e729d23e8799ce0e075699e76d60227d848d8d0c (MIT, see LICENSE).
Files copied unchanged except one line: model_function.py, model_utils.py, utils.py (incl. the evaluation functions
`evaluation_rmsd_mm`, `evaluation_acc_mm`, `evaluation_crps_mm`), evaluation_global.py, README.md.
The pretrained checkpoint `checkpoints/ClimODE_global.pt` (11 MB) is not committed;
`scripts/run_climode_eval.py` downloads it from the upstream repository.

The owner pointed at https://github.com/dataymeric/ClimODE (a PyTorch-Ignite re-implementation) for
evaluation. Its `src/utils/evaluation.py` only has a batch RMSE and a zarr writer, and its published
baseline numbers (`visualization/data/other_model_*.csv`) are the ClimODE paper's ClimaX/IFS values,
so the scoring here uses the original authors' functions, which that repository also follows.

Local change (pandas >= 2.2 compatibility only): `utils.py` `resample(time="6H").nearest(tolerance="1H")` -> `"6h"` / `"1h"` (3 occurrences).

`train_global.py` (added in step 12) is the upstream training script, copied unchanged, so the original-size
model can be retrained on our files on a GPU (docs/RUN_ON_GPU.md section 11).
