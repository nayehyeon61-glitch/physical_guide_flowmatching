#!/usr/bin/env bash
# Step 26: blend_band+ (docs/step26_blend_plus.md). The finished blend_band run in $BASE (run_step13_best.sh
# BLEND=1 PROFILE=ngcm: PINN, FM, regression and the small G(x) core, 1979-2015) is reused frozen; only the mean
# gets bigger:
#   1. plus_long   a wide G(x) core (PLUS_HIDDEN channels, PLUS_BLOCKS residual blocks; 48 / "2 2" before)
#                  on 5-day rollouts (K = 20, curriculum 1 .. 20), gradient checkpointing per 6 h step
#   2. plus_short  the same core on 2-day rollouts (K = 8): the base core lost 24 h skill to the 5-day loss
#                  (24 h z500 160 with K = 8 on 10 years, 168 with K = 20 on 37 years)
#   3. blend       mu* = a + b1 plus_long + b2 plus_short + b3 base core + c regression + d x(t) per lead and
#                  variable on 2016 (BlendMBM, closed-form least squares), FM + PINN deviations, band Gamma
#   4. scores      2017-18 (360 ICs) against the base blend_band, the ClimODE pairs of $BASE/climode, and the
#                  2020 hold-out of run_ngcm_vs_neuralgcm.sh ($BASE/ngcm2020) with its ClimODE 2020 forecasts
# The ablation arm plus1 (= the base blend with plus_long in place of the base core) separates the wider core
# from the extra cores. Every step is skipped when its .done marker exists; a step that stops on MAX_HOURS
# (exit 75) resumes when the script is run again.
#
#   bash scripts/run_blend_plus.sh                                   # BASE=runs/blend_ngcm, OUT=runs/blend_plus
#   PLUS_HIDDEN=96 PLUS_BLOCKS="2 2 2" bash scripts/run_blend_plus.sh   # a smaller core+
#   SHORT=0 BASE_CORE=0 ...                                          # only the wide long core in the blend
#   SMOKE=1 DEVICE=cpu ...                                           # 2 iterations of a tiny core, every step once
#   DRYRUN=1 ...                                                     # print the commands only
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-runs/blend_ngcm}
OUT=${OUT:-runs/blend_plus}
DATA=${DATA:-data/era5_5p625}
DEVICE=${DEVICE:-cuda}
TRAIN_YEARS=${TRAIN_YEARS:-"$(seq -s ' ' 1979 2015)"}
VAL_YEAR=${VAL_YEAR:-2016}
TEST_YEARS=${TEST_YEARS:-"2017 2018"}
HOLDOUT=${HOLDOUT-2020}                # HOLDOUT="" skips the hold-out year
PLUS_HIDDEN=${PLUS_HIDDEN:-128}
PLUS_BLOCKS=${PLUS_BLOCKS:-"2 2 2 2"}
LONG_ITERS=${LONG_ITERS:-60000}
SHORT_ITERS=${SHORT_ITERS:-40000}
BATCH=${BATCH:-16}
LR=${LR:-2e-4}
VAL_EVERY=${VAL_EVERY:-1000}
PATIENCE=${PATIENCE:-10}
MAX_HOURS=${MAX_HOURS:-0}
THREADS=${THREADS:-8}
AMP=${AMP:-0}
SHORT=${SHORT:-1}                      # train and blend plus_short
BASE_CORE=${BASE_CORE:-1}              # keep the base core ($BASE/dyn_gfeat) in the blend
SMOKE=${SMOKE:-0}
NGCM_DIR=${NGCM_DIR:-docs/step26/neuralgcm}  # NeuralGCM forecasts of the hold-out year (step 23, CPU)
K=$DATA/kai6h
E=$DATA/era5_data
FM=seqfm_dct_pinn
REG=seqregress_pinn
LAST_TEST=${TEST_YEARS##* }
mkdir -p "$OUT"
abs() { realpath -m "$1"; }
step() {  # step <name> <command...>: skip when $OUT/.done/<name> exists, log to $OUT/<name>.log
    local name=$1; shift
    mkdir -p "$OUT/.done"
    if [ -e "$OUT/.done/$name" ]; then echo "[plus] $name: done, skipped"; return; fi
    if [ "${DRYRUN:-0}" = 1 ]; then echo "[dry] $name: $(printf '%q ' "$@")"; return; fi
    echo "[plus] $name: start $(date +%H:%M), log $OUT/$name.log"
    local rc=0
    "$@" > "$OUT/$name.log" 2>&1 || rc=$?
    if [ "$rc" = 0 ]; then
        touch "$OUT/.done/$name"; echo "[plus] $name: done $(date +%H:%M)"
    elif [ "$rc" = 75 ]; then
        echo "[plus] $name stopped on MAX_HOURS=$MAX_HOURS; run the same command again to resume"; exit 75
    else
        echo "[plus] $name failed, see $OUT/$name.log"; tail -n 20 "$OUT/$name.log"; exit 1
    fi
}
link() {  # link <target> <name in dir> <dir>: absolute symlink, the target must exist
    [ -e "$1" ] || { echo "[plus] missing $1 (finish the base run first)"; exit 1; }
    ln -sfn "$(abs "$1")" "$3/$2"
}

# 0. the frozen base run: stats, members, regression, base core and its blend_band
link "$BASE/stats" stats "$OUT"
for y in $VAL_YEAR $TEST_YEARS; do
    link "$BASE/${FM}_ens_$y.npz" "${FM}_ens_$y.npz" "$OUT"
    link "$BASE/${REG}_ens_$y.npz" "${REG}_ens_$y.npz" "$OUT"
done
for f in dyn_gfeat.json dyn_gfeat.pt dyn_gfeat_pred.npz "dyn_gfeat_pred_$VAL_YEAR.npz"; do link "$BASE/$f" "$f" "$OUT"; done
for y in $TEST_YEARS; do link "$BASE/${FM}_blend_band_ens_$y.npz" "base_blend_band_ens_$y.npz" "$OUT"; done

TRAIN="--kai $K --era5 $E --out $OUT --gfeat --lambda-p 0 --train-years $TRAIN_YEARS --val-year $VAL_YEAR"
TRAIN="$TRAIN --device $DEVICE --threads $THREADS --patience $PATIENCE --max-hours $MAX_HOURS --grad-ckpt --val-chunk 4"
[ -d "$BASE/cache" ] && TRAIN="$TRAIN --cache $(abs "$BASE/cache")"
[ "$AMP" = 1 ] && TRAIN="$TRAIN --amp"
if [ "$SMOKE" = 1 ]; then
    PLUS_HIDDEN=16; PLUS_BLOCKS="1 1"
    NET="--hidden $PLUS_HIDDEN --blocks $PLUS_BLOCKS --batch 2 --n-val 4 --n-val-train 4 --val-K 8 --val-every 1 --proj-K 2"
    LONG="--iters 2 --K 4 --curriculum 1 4"; SHORT_ARGS="--iters 2 --K 2 --curriculum 1 2"
    GUIDE_ARGS="--n-valid 1"; SCORE_ARGS="--n-boot 20"
else
    NET="--hidden $PLUS_HIDDEN --blocks $PLUS_BLOCKS --batch $BATCH --lr $LR --n-val 16 --n-val-train 8 --val-K 24"
    NET="$NET --val-every $VAL_EVERY"
    LONG="--iters $LONG_ITERS --K 20 --curriculum 1 2 4 8 12 16 20"; SHORT_ARGS="--iters $SHORT_ITERS --K 8 --curriculum 1 2 4 8"
    GUIDE_ARGS=""; SCORE_ARGS=""
fi
PRED="--kai $K --era5 $E --run $OUT --val-year $VAL_YEAR --device $DEVICE --threads $THREADS"

# 1-2. the wide cores
step plus_long python scripts/run_step16_dynpinn.py $TRAIN $NET $LONG --tag plus_long
step predict_long python scripts/run_step16_predict.py $PRED --test-years $TEST_YEARS --tag plus_long
CORES="$OUT/plus_long"
EXTRA=""
if [ "$SHORT" = 1 ]; then
    step plus_short python scripts/run_step16_dynpinn.py $TRAIN $NET $SHORT_ARGS --tag plus_short
    step predict_short python scripts/run_step16_predict.py $PRED --test-years $TEST_YEARS --tag plus_short
    EXTRA="$EXTRA $OUT/plus_short"
fi
[ "$BASE_CORE" = 1 ] && EXTRA="$EXTRA $OUT/dyn_gfeat"

# 3. blends on the frozen members (the operator and learned-guide arms are not saved, so 1 epoch each)
G="--era5 $E --reg-mbm --blend --val-year $VAL_YEAR --save-arms blend_band --epochs 1 --guide-epochs 1"
G="$G --threads $THREADS $GUIDE_ARGS"
mkdir -p "$OUT/abl_plus1"
step guide_plus1 python scripts/run_step13_guide.py $G --ens "$OUT/$FM" --reg "$OUT/$REG" --test-years $TEST_YEARS \
    --out "$OUT/abl_plus1" --core "$CORES"
step guide_plus python scripts/run_step13_guide.py $G --ens "$OUT/$FM" --reg "$OUT/$REG" --test-years $TEST_YEARS \
    --out "$OUT" --core "$CORES" --core-extra $EXTRA
for y in $TEST_YEARS; do
    ln -sfn "abl_plus1/${FM}_blend_band_ens_$y.npz" "$OUT/plus1_blend_band_ens_$y.npz"
    ln -sfn "${FM}_blend_band_ens_$y.npz" "$OUT/plus_blend_band_ens_$y.npz"
done

# 4. scores: bootstrap against the base blend_band, energy scores, ClimODE pairs, figure
ARMS="$OUT/$REG $OUT/base_blend_band $OUT/plus1_blend_band $OUT/plus_blend_band"
step score python scripts/score_step13.py --era5 "$E" --out "$OUT/score_plus.json" --ref "$OUT/base_blend_band" \
    --arms $ARMS --test-years $TEST_YEARS $SCORE_ARGS
step energy python scripts/step20_energy_score.py --era5 "$E" --out "$OUT/energy_plus.json" --arms $ARMS \
    --test-years $TEST_YEARS --threads "$THREADS"
PAIRED=""
if [ -e "$BASE/climode/climode_pred.npz" ]; then
    step paired python scripts/compare_step13_climode.py --era5 "$E" --climode "$BASE/climode/climode_pred.npz" \
        --arms $ARMS --core "$OUT/plus_long" --out "$OUT/climode_paired_plus.json"
    PAIRED="--paired $OUT/climode_paired_plus.json"
fi
step figure python scripts/plot_step13.py --era5 "$E" --scores "$OUT/score_plus.json" --years $TEST_YEARS \
    --figures fig_step26 --out "$OUT/fig" --step "Step 26" $PAIRED

# 5. hold-out year: frozen members of run_ngcm_vs_neuralgcm.sh ($BASE/ngcm$HOLDOUT), the wide cores predict it
N=$OUT/y$HOLDOUT
H=$BASE/ngcm$HOLDOUT
if [ -n "$HOLDOUT" ] && [ -e "$H/${FM}_ens_$HOLDOUT.npz" ]; then
    mkdir -p "$N"
    link "$OUT/stats" stats "$N"
    for y in $VAL_YEAR $HOLDOUT; do
        link "$H/${FM}_ens_$y.npz" "${FM}_ens_$y.npz" "$N"
        link "$H/${REG}_ens_$y.npz" "${REG}_ens_$y.npz" "$N"
    done
    for f in dyn_gfeat.json dyn_gfeat.pt dyn_gfeat_pred.npz "dyn_gfeat_pred_$VAL_YEAR.npz"; do link "$H/$f" "$f" "$N"; done
    link "$H/${FM}_blend_band_ens_$HOLDOUT.npz" "base_blend_band_ens_$HOLDOUT.npz" "$N"
    HPRED="--kai $K --era5 $E --run $N --val-year $VAL_YEAR --device $DEVICE --threads $THREADS --test-years $HOLDOUT"
    HCORES="$N/plus_long"
    HEXTRA=""
    for t in plus_long plus_short; do
        [ -e "$OUT/$t.pt" ] || continue
        link "$OUT/$t.pt" "$t.pt" "$N"; link "$OUT/$t.json" "$t.json" "$N"
        step "y${HOLDOUT}_predict_$t" python scripts/run_step16_predict.py $HPRED --tag "$t"
    done
    [ "$SHORT" = 1 ] && HEXTRA="$HEXTRA $N/plus_short"
    [ "$BASE_CORE" = 1 ] && HEXTRA="$HEXTRA $N/dyn_gfeat"
    HG="$G --ens $N/$FM --reg $N/$REG --test-years $HOLDOUT"
    mkdir -p "$N/abl_plus1"
    step "y${HOLDOUT}_guide_plus1" python scripts/run_step13_guide.py $HG --out "$N/abl_plus1" --core "$HCORES"
    step "y${HOLDOUT}_guide_plus" python scripts/run_step13_guide.py $HG --out "$N" --core "$HCORES" --core-extra $HEXTRA
    ln -sfn "abl_plus1/${FM}_blend_band_ens_$HOLDOUT.npz" "$N/plus1_blend_band_ens_$HOLDOUT.npz"
    ln -sfn "${FM}_blend_band_ens_$HOLDOUT.npz" "$N/plus_blend_band_ens_$HOLDOUT.npz"
    HARMS="$N/$REG $N/base_blend_band $N/plus1_blend_band $N/plus_blend_band"
    step "y${HOLDOUT}_score" python scripts/score_step13.py --era5 "$E" --out "$N/score_plus_$HOLDOUT.json" \
        --ref "$N/base_blend_band" --arms $HARMS --test-years "$HOLDOUT" $SCORE_ARGS
    C2=$(ls "$H"/climode*/climode_pred_"$HOLDOUT"_valid.npz 2>/dev/null | head -n 1 || true)
    if [ -n "$C2" ]; then
        step "y${HOLDOUT}_paired" python scripts/compare_step13_climode.py --era5 "$E" --climode "$C2" \
            --ref-name "ClimODE (public checkpoint)" --arms $HARMS --core "$N/plus_long" \
            --out "$N/climode_paired_plus_$HOLDOUT.json"
    fi
    # NeuralGCM 2.8 deg (official deterministic checkpoint, run on CPU in step 23 on 9 / 18 ICs of 2020) on the same
    # ICs: from our 5.625 deg / 12-level IC (the equal-information target) and from its own 0.25 deg / 37-level IC
    for R in coarse full; do
        P=$NGCM_DIR/neuralgcm_${HOLDOUT}_${R}_ic.npz
        [ -e "$P" ] || continue
        step "y${HOLDOUT}_ngcm_$R" python scripts/compare_step13_climode.py --era5 "$E" --climode "$P" \
            --ref-name "NeuralGCM 2.8° ($R IC)" --arms $HARMS --core $HCORES --out "$N/ngcm_${R}_paired_plus.json"
    done
else
    echo "[plus] hold-out $HOLDOUT skipped (no $H/${FM}_ens_$HOLDOUT.npz: run scripts/run_ngcm_vs_neuralgcm.sh first)"
fi
echo "[plus] finished: $OUT/score_plus.json, $OUT/fig/fig_step26.png" \
    "$([ -e "$N/score_plus_$HOLDOUT.json" ] && echo ", $N/score_plus_$HOLDOUT.json")"
