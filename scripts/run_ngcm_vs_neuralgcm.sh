#!/usr/bin/env bash
# The owner's NeuralGCM comparison (2026-11-02): our blend_band trained with a NeuralGCM-like budget
# (run_step13_best.sh PROFILE=ngcm: train 1979-2015 = NeuralGCM-2.8 deg's 1979-2017 without our calibration year
# 2016 and test year 2017, FM / regression 120000 x 16, G(x) core on 5-day rollouts) against NeuralGCM 2.8 deg's
# official checkpoint, both scored on our 5.625 deg grid in 2020 (outside both training sets). NeuralGCM runs
# twice: from its own ERA5 0.25 deg / 37-level IC and from our 5.625 deg / 12-level IC (--coarse-ic, the
# equal-information test of docs/step23_neuralgcm.md 4.4). Every step is skipped when its .done marker exists.
#
#   bash scripts/run_ngcm_vs_neuralgcm.sh                       # train (resumes) -> 2020 -> NeuralGCM x 2 -> scores
#   NGCM_PY=.venv-ngcm/bin/python bash scripts/run_ngcm_vs_neuralgcm.sh   # python with neuralgcm + JAX for NeuralGCM
#   NGCM_ARGS="--cpu --threads 8 --ic-offset 5 --ic-stride 10"  # NeuralGCM on CPU, 18 of the 180 ICs
#   SKIP_TRAIN=1                                                # weights already in $OUT
#   SKIP_NGCM=1                                                 # stop after our $YEAR members (no JAX needed)
#   DRYRUN=1                                                    # print the commands only
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data/era5_5p625}
OUT=${OUT:-runs/blend_ngcm}
DEVICE=${DEVICE:-cuda}
TRAIN_YEARS=${TRAIN_YEARS:-"$(seq -s ' ' 1979 2015)"}
YEAR=${YEAR:-2020}
NGCM_PY=${NGCM_PY:-python}
NGCM_ARGS=${NGCM_ARGS:-}
K=$DATA/kai6h
E=$DATA/era5_data
N=$OUT/ngcm$YEAR
step() {  # step <name> <command...>: skip when $N/<name>.done exists, log to $N/<name>.log
    local name=$1; shift
    if [ -e "$N/$name.done" ]; then echo "[ngcm] $name: done, skipped"; return; fi
    if [ "${DRYRUN:-0}" = 1 ]; then echo "[dry] $name: $(printf '%q ' "$@")"; return; fi
    echo "[ngcm] $name: start $(date +%H:%M), log $N/$name.log"
    if "$@" > "$N/$name.log" 2>&1; then
        touch "$N/$name.done"; echo "[ngcm] $name: done $(date +%H:%M)"
    else
        echo "[ngcm] $name failed, see $N/$name.log"; tail -n 20 "$N/$name.log"; exit 1
    fi
}

# 1. our model with the NeuralGCM-like budget (also scores ClimODE's 2017-2018 test ICs)
if [ "${SKIP_TRAIN:-0}" != 1 ]; then
    env BLEND=1 SIGMA=0 PROFILE=ngcm DEVICE="$DEVICE" DATA="$DATA" OUT="$OUT" TRAIN_YEARS="$TRAIN_YEARS" \
        bash scripts/run_step13_best.sh
fi
mkdir -p "$N"
[ -e "$K/$YEAR.nc" ] || step data python scripts/wb2_to_kai.py --out "$DATA" --all-hours --years "$YEAR"

# 2. our members of $YEAR: frozen weights and the 2016 calibration constants, no training
for f in pinn_project.pt pinn_project.json seqfm_dct.pt seqregress.pt dyn_gfeat.pt dyn_gfeat.json stats \
         seqfm_dct_pinn_ens_2016.npz seqregress_pinn_ens_2016.npz; do
    [ -e "$N/$f" ] || ln -sfn "../$f" "$N/$f"
done
C="--kai $K --era5 $E --out $N --pinn $N/pinn_project.pt --save-ens --resume-ens --train-years $TRAIN_YEARS"
C="$C --val-year 2016 --test-years $YEAR --device $DEVICE --val-every 0 --patience 0 --max-hours 0 --val-members 0"
[ -d "$OUT/cache" ] && C="$C --cache $OUT/cache"
step seqfm python scripts/run_step13_seqfm.py $C --basis dct
step seqregress python scripts/run_step13_seqfm.py $C --model regress
step core python scripts/run_step16_predict.py --kai "$K" --era5 "$E" --run "$N" --tag dyn_gfeat --val-year 2016 \
    --test-years "$YEAR"
step guide python scripts/run_step13_guide.py --ens "$N/seqfm_dct_pinn" --era5 "$E" --out "$N" --core "$N/dyn_gfeat" \
    --reg "$N/seqregress_pinn" --reg-mbm --blend --val-year 2016 --test-years "$YEAR" --save-arms regm_band blend_band
ARMS="$N/seqregress_pinn $N/seqfm_dct_pinn_regm_band $N/seqfm_dct_pinn_blend_band"
step score python scripts/score_step13.py --era5 "$E" --out "$N/score_$YEAR.json" --ref "$N/seqregress_pinn" \
    --test-years "$YEAR" --arms $ARMS

# 3. NeuralGCM 2.8 deg deterministic, official checkpoint, regridded to our 5.625 deg grid
if [ "${SKIP_NGCM:-0}" = 1 ]; then echo "[ngcm] SKIP_NGCM=1: $YEAR members and $N/score_$YEAR.json only"; exit 0; fi
step ngcm_full $NGCM_PY scripts/run_neuralgcm_eval.py --era5 "$E" --out "$N/ngcm_full" \
    --model v1/deterministic_2_8_deg --year "$YEAR" $NGCM_ARGS
step ngcm_coarse $NGCM_PY scripts/run_neuralgcm_eval.py --era5 "$E" --out "$N/ngcm_coarse" \
    --model v1/deterministic_2_8_deg --year "$YEAR" --coarse-ic $NGCM_ARGS

# 4. paired scores on the same ICs (z500, t850, our members' CRPS), share of the gap that is IC information, figure
for R in ngcm_full ngcm_coarse; do
    step "paired_$R" python scripts/compare_step13_climode.py --era5 "$E" --climode "$N/$R/neuralgcm_pred.npz" \
        --ref-name "NeuralGCM 2.8° ($R)" --arms $ARMS --core "$N/dyn_gfeat" --out "$N/${R}_paired.json"
done
step coarse_ic python scripts/step23_coarse_ic.py --era5 "$E" --full "$N/ngcm_full/neuralgcm_pred.npz" \
    --coarse "$N/ngcm_coarse/neuralgcm_pred.npz" --out "$N/coarse_ic.json"
step figure python scripts/plot_step23_coarse.py --era5 "$E" --coarse-json "$N/coarse_ic.json" \
    --paired "$N/ngcm_coarse_paired.json" --full-paired "$N/ngcm_full_paired.json" --out "$N/fig_ngcm_5p625.png"
echo "[ngcm] finished: $N/ngcm_full_paired.json, $N/ngcm_coarse_paired.json, $N/coarse_ic.json, $N/fig_ngcm_5p625.png"
