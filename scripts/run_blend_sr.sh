#!/usr/bin/env bash
# blend_sr S1 (docs/step28_blend_sr.md): the blend_band+ structure on the 2.8125 deg grid, GPU. The chain:
#   1. base      run_step13_best.sh PROFILE=sr28: PINN, FM, regression, small core, 2016-18 members, blend_band
#   2. hold-out  run_ngcm_vs_neuralgcm.sh SKIP_NGCM=1: frozen members of 2020
#   3. plus      run_blend_plus.sh on that base: wide cores, blend, native-grid scores
#   4. control   the same plus blend on the members without the PINN (separates resolution from physics)
#   5. S1 scores everything averaged to 5.625 deg, against the 5.625 deg truth: score_s1_2020.json, reference
#      $REF (the 5.625 deg model), and paired with ClimODE when its 2020 forecasts exist. s1_gate.json is the verdict.
# Every step resumes. Start it through scripts/gpu_free.sh, which refuses an occupied GPU:
#   bash scripts/gpu_free.sh bash scripts/run_blend_sr.sh
#   SMOKE=1 DEVICE=cpu OUT=runs/sr28_smoke bash scripts/run_blend_sr.sh     # the chain on a few iterations
#   DRYRUN=1 bash scripts/run_blend_sr.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data/era5_2p8125}
DATA0=${DATA0:-data/era5_5p625}
OUT=${OUT:-runs/sr28}
DEVICE=${DEVICE:-cuda}
HOLDOUT=${HOLDOUT:-2020}
THREADS=${THREADS:-8}
SMOKE=${SMOKE:-0}
BASE=$OUT/base
PLUS=$OUT/plus
C=$OUT/c5p625
PLUS_HIDDEN=${PLUS_HIDDEN:-128}; PLUS_BLOCKS=${PLUS_BLOCKS:-"3 3 3 3"}; BATCH=${BATCH:-8}
REF=${REF:-runs/blend_plus/y$HOLDOUT/plus_blend_band}
[ -e "${REF}_ens_$HOLDOUT.npz" ] || REF=runs/blend_ngcm/ngcm$HOLDOUT/seqfm_dct_pinn_blend_band
FM=seqfm_dct_pinn
E0=$DATA0/era5_data
mkdir -p "$OUT" "$C"
echo "[sr] out $OUT data $DATA device $DEVICE reference $REF"
common=(DATA="$DATA" DEVICE="$DEVICE" DRYRUN="${DRYRUN:-0}")
gpu=(PREDICT_DEVICE="$DEVICE" CHANNEL_MULT="${CHANNEL_MULT:-1 2 2 2}")
if [ "$SMOKE" = 1 ]; then
    # the real chain on tiny settings; only the years of $TRAIN_YEARS must exist
    TRAIN_YEARS=${TRAIN_YEARS:-"2015"}
    env "${common[@]}" "${gpu[@]}" OUT="$BASE" BLEND=1 SIGMA=0 PROFILE=smoke TRAIN_YEARS="$TRAIN_YEARS" PINN_K_LAT=16 PINN_K_LON=32 \
        CORE_BLOCKS="1 1" CORE_CKPT=1 THREADS="$THREADS" bash scripts/run_step13_best.sh
else
    TRAIN_YEARS=${TRAIN_YEARS:-"$(seq -s ' ' 1979 2015)"}
    env "${common[@]}" "${gpu[@]}" OUT="$BASE" BLEND=1 SIGMA=0 PROFILE=sr28 TRAIN_YEARS="$TRAIN_YEARS" THREADS="$THREADS" \
        bash scripts/run_step13_best.sh
fi
env "${common[@]}" "${gpu[@]}" OUT="$BASE" SKIP_TRAIN=1 SKIP_NGCM=1 YEAR="$HOLDOUT" TRAIN_YEARS="$TRAIN_YEARS" \
    bash scripts/run_ngcm_vs_neuralgcm.sh
env "${common[@]}" BASE="$BASE" OUT="$PLUS" TRAIN_YEARS="$TRAIN_YEARS" HOLDOUT="$HOLDOUT" SMOKE="$SMOKE" NGCM_DIR=/nonexistent \
    PLUS_HIDDEN="$PLUS_HIDDEN" PLUS_BLOCKS="$PLUS_BLOCKS" BATCH="$BATCH" THREADS="$THREADS" AMP="${AMP:-0}" bash scripts/run_blend_plus.sh
[ "${DRYRUN:-0}" = 1 ] && { echo "[dry] control, coarsen, S1 scores"; exit 0; }

# control: the plus blend on the members without the PINN
N=$PLUS/y$HOLDOUT; H=$BASE/ngcm$HOLDOUT; NP=$OUT/nopinn
mkdir -p "$NP"
for y in 2016 "$HOLDOUT"; do for f in seqfm_dct_raw seqregress_raw; do ln -sfn "$(realpath -m "$H/${f}_ens_$y.npz")" "$NP/${f}_ens_$y.npz"; done; done
[ -e "$H/seqfm_dct_raw_ens_2016.npz" ] || for f in seqfm_dct_raw seqregress_raw; do ln -sfn "$(realpath -m "$BASE/${f}_ens_2016.npz")" "$NP/${f}_ens_2016.npz"; done
ln -sfn "$(realpath -m "$N/stats")" "$NP/stats"
EXTRA="$N/plus_short $N/dyn_gfeat"
if [ ! -e "$NP/seqfm_dct_raw_blend_band_ens_$HOLDOUT.npz" ]; then
    python scripts/run_step13_guide.py --era5 "$DATA/era5_data" --reg-mbm --blend --val-year 2016 --save-arms blend_band --epochs 1 \
        --guide-epochs 1 --threads "$THREADS" $([ "$SMOKE" = 1 ] && echo --n-valid 1) --ens "$NP/seqfm_dct_raw" --reg "$NP/seqregress_raw" \
        --test-years "$HOLDOUT" --out "$NP" --core "$N/plus_long" --core-extra $EXTRA > "$OUT/control.log" 2>&1 \
        || { tail -n 20 "$OUT/control.log"; exit 1; }
fi

# S1 scores on the 5.625 deg truth
A=$C/sr_plus_blend_band_ens_$HOLDOUT.npz; B=$C/sr_base_blend_band_ens_$HOLDOUT.npz; D=$C/sr_nopinn_blend_band_ens_$HOLDOUT.npz
coarsen1() { [ -e "$2" ] || { python scripts/coarsen_ens.py --era5 "$E0" --fine-era5 "$DATA/era5_data" --out "$C/tmp" "$1"; mv "$C/tmp/$(basename "$1")" "$2"; }; }
coarsen1 "$N/${FM}_blend_band_ens_$HOLDOUT.npz" "$A"
coarsen1 "$H/${FM}_blend_band_ens_$HOLDOUT.npz" "$B"
coarsen1 "$NP/seqfm_dct_raw_blend_band_ens_$HOLDOUT.npz" "$D"
rmdir "$C/tmp" 2>/dev/null || true
ln -sfn "$(realpath -m "${REF}_ens_$HOLDOUT.npz")" "$C/ref_ens_$HOLDOUT.npz"
python scripts/score_step13.py --era5 "$E0" --out "$OUT/score_s1_$HOLDOUT.json" --ref "$C/ref" \
    --arms "$C/ref" "$C/sr_base_blend_band" "$C/sr_plus_blend_band" "$C/sr_nopinn_blend_band" --test-years "$HOLDOUT" \
    $([ "$SMOKE" = 1 ] && echo --n-boot 20) > "$OUT/score_s1.log" 2>&1 || { tail -n 20 "$OUT/score_s1.log"; exit 1; }
CL=$(ls runs/blend_ngcm/ngcm$HOLDOUT/climode*/climode_pred_"$HOLDOUT"_valid.npz 2>/dev/null | head -n 1 || true)
if [ -n "$CL" ] && [ "$SMOKE" != 1 ]; then
    python scripts/compare_step13_climode.py --era5 "$E0" --climode "$CL" --ref-name "ClimODE (public checkpoint)" \
        --arms "$C/ref" "$C/sr_base_blend_band" "$C/sr_plus_blend_band" "$C/sr_nopinn_blend_band" --out "$OUT/climode_paired_s1_$HOLDOUT.json" > "$OUT/paired_s1.log" 2>&1 \
        || echo "[sr] ClimODE pairs failed, see $OUT/paired_s1.log"
fi
echo "[sr] finished: $OUT/score_s1_$HOLDOUT.json"
