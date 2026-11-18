#!/usr/bin/env bash
# The tendency core INSIDE KAI, in every stage (docs/kai_stage_core.md), against the output mix it replaces.
#
# Needs in $OUT: kai.pt and the frozen tendency core $C.pt (scripts/run_kai_plus.sh at 1.5 deg, or the commands of
# docs/kai_tendency.md sec. 8.3 at 5.625 deg). Nothing of them is changed. Every model below is the SAME rollout
# fine-tuning of kai (2 to 3 days, $RO_ITERS iterations), so the only difference between them is where the core is:
#   kai_ro   no core                                              (the control of the fine-tuning itself; reused if there)
#   stc      the core INSIDE every stage: at the entry of each of the four stages its channels are merged to the 67
#            variables, the core steps from that state, its tendency is split to the stage's channels and added
#   stcg     stc + an output mix with the core forecast, its weight computed by KAI per channel and grid point;
#            starts as the loop (KAI, core) with the loop's fixed weights
#   stg      control of stc: the core is asked at the input x only, the same tendency enters all four stages
#   stgg     control of stcg: stg + the learned output mix
#   outg     the learned output mix alone                         (the control of the stages)
# and the existing connections of the same two models are scored next to them: blend_$C, loop_$C on kai and kai_ro.
#
#   bash scripts/run_kai_stage.sh                                 # 1.5 deg, the folders of SRC=daily run_kai_plus.sh
#   C=core_plus_u64 SFX=_u64 bash scripts/run_kai_stage.sh        # another core: tags stg_u64, stgg_u64, outg_u64
#   GRID=5.625 C=core_gn bash scripts/run_kai_stage.sh            # the CPU test of docs/kai_stage_core.md
#   ARMS="stcg stgg" ...                                          # only some of the new models
#   CORE_GRAD=0 ...                                               # no gradient through the frozen core (less memory, time)
#   DRYRUN=1 ...                                                  # print the commands only
set -euo pipefail
cd "$(dirname "$0")/.."
GRID=${GRID:-1.5}
if [ "$GRID" = 5.625 ]; then
    DATA=${DATA:-data/era5_5p625}; OUT=${OUT:-runs/kaitend}; DEVICE=${DEVICE:-cpu}; THREADS=${THREADS:-4}
    KAI_DEPTHS=${KAI_DEPTHS:-"2 2 4 2"}; KAI_DIMS=${KAI_DIMS:-"24 48 64 96"}
    RO_BATCH=${RO_BATCH:-8}; SCORE_BATCH=${SCORE_BATCH:-16}; VAL_BATCH=${VAL_BATCH:-16}; KAI_CKPT=${KAI_CKPT:-0}
    C=${C:-core_gngmz8}
else
    DATA=${DATA:-data/era5_${GRID//./p}_daily}; OUT=${OUT:-runs/kaiplus_${GRID//./p}_daily}; DEVICE=${DEVICE:-cuda}
    THREADS=${THREADS:-8}
    KAI_DEPTHS=${KAI_DEPTHS:-"3 3 15 3"}; KAI_DIMS=${KAI_DIMS:-"48 96 192 288"}
    RO_BATCH=${RO_BATCH:-4}; SCORE_BATCH=${SCORE_BATCH:-4}; VAL_BATCH=${VAL_BATCH:-4}; KAI_CKPT=${KAI_CKPT:-1}
    C=${C:-core_plus}
fi
TRAIN_YEARS=${TRAIN_YEARS:-"2012 2013 2014 2015"}; VAL_YEAR=${VAL_YEAR:-2016}; TEST_YEAR=${TEST_YEAR:-2017}
RO_ITERS=${RO_ITERS:-1000}; RO_LR=${RO_LR:-3e-4}
GATE_LR=${GATE_LR:-1e-3}              # the 67 base weights of the output mix start at a fitted value: a small step
NEW_LR=${NEW_LR:-1e-3}                # the layers added to KAI start at zero: KAI's own from-scratch learning rate
CORE_GRAD=${CORE_GRAD:-1}             # 0: no gradient through the core between the days (stc / stcg keep the one
                                      #    through the core inside the stages: their merge layers learn through it)
IC_STRIDE=${IC_STRIDE:-3}; LEADS=${LEADS:-10}; LOGK=${LOGK:-250}
ARMS=${ARMS:-"stcg stc stgg stg outg"}
SFX=${SFX:-}                           # suffix of the new tags (another core in the same folder)
SC=stage$SFX
BL=blend_$C; LP=loop_$C; [ "$C" = core ] && { BL=blend; LP=loop; }

step() {  # step <name> <done file> <command...>: skipped when the done file exists, log to $OUT/<name>.log
    local name=$1 mark=$2; shift 2
    if [ -e "$mark" ]; then echo "[stage] $name: done, skipped"; return; fi
    if [ "${DRYRUN:-0}" = 1 ]; then echo "[dry] $name: $(printf '%q ' "$@")"; return; fi
    echo "[stage] $name: start $(date +%H:%M), log $OUT/$name.log"
    "$@" >> "$OUT/$name.log" 2>&1 || { echo "[stage] $name FAILED, see $OUT/$name.log"; exit 1; }
}
for f in kai "$C"; do
    [ -e "$OUT/$f.pt" ] || [ "${DRYRUN:-0}" = 1 ] || { echo "missing $OUT/$f.pt (KAI and the trained core come first)"; exit 1; }
done

D="--data $DATA/kai6h --cache $DATA/kaiday --era5 $DATA/era5_data --out $OUT --train-years $TRAIN_YEARS"
D="$D --val-year $VAL_YEAR --test-year $TEST_YEAR --device $DEVICE --threads $THREADS --val-batch $VAL_BATCH"
R="python scripts/run_kai_tendency.py $D --depths $KAI_DEPTHS --dims $KAI_DIMS --core-tag $C"
KC=""; [ "$KAI_CKPT" = 1 ] && KC="--kai-ckpt --core-ckpt"
CG=""; [ "$CORE_GRAD" = 1 ] || CG="--core-nograd"
FT="--init kai --rollout 2 3 --lr $RO_LR --iters $RO_ITERS --batch $RO_BATCH --log-every $LOGK --resume $KC"

# 0. the weights of the loop (KAI, core): the day-1 weights per channel of the lead-wise blend, validation year
W0="$OUT/score_${SC}0.json"
step ${SC}0 "$W0" bash -c "$R --score kai --score-cores $C --blend-bases kai --leads $LEADS --ic-stride $IC_STRIDE \
    --score-batch $SCORE_BATCH && cp $OUT/score.json $W0"

# 1. the control of the fine-tuning: KAI in its own 2 to 3 day rollout, no core
step kai_ro "$OUT/kai_ro.pt" $R --arm kai --tag kai_ro $FT

# 2. the same fine-tuning with the core inside
for arm in $ARMS; do
    step $arm$SFX "$OUT/$arm$SFX.pt" $R --arm $arm --tag $arm$SFX $FT $CG --gate-lr $GATE_LR --new-lr $NEW_LR \
        --gate-init "$W0:$BL"
done

# 3. score: every new model with its diagnostics (~notau, ~ns1..4: one stage's tau off), the output mixes on both KAIs
TAGS=""; PAIRS="$LP@kai_ro:kai_ro"
for arm in $ARMS; do
    t=$arm$SFX; TAGS="$TAGS $t"
    PAIRS="$PAIRS $t:kai_ro $t:$LP@kai_ro $t:$BL@kai_ro"
    for i in 1 2 3 4; do [ "$arm" = outg ] || PAIRS="$PAIRS $t~ns$i:$t"; done
    [ "$arm" = outg ] || PAIRS="$PAIRS $t~notau:$t"
done
# between the new models: the core inside the stages vs its tendency of x, the stages vs the output mix alone
for ab in stcg:stgg stc:stg stcg:outg stgg:outg stcg:stc stgg:stg; do
    case " $ARMS " in *" ${ab%:*} "*) case " $ARMS " in *" ${ab#*:} "*) PAIRS="$PAIRS ${ab%:*}$SFX:${ab#*:}$SFX";; esac;; esac
done
step $SC "$OUT/score_$SC.json" bash -c "$R --score kai kai_ro $TAGS --score-cores $C --blend-bases kai kai_ro \
    --loop-blend $C --leads $LEADS --ic-stride $IC_STRIDE --score-batch $SCORE_BATCH --pairs $PAIRS \
    && cp $OUT/score.json $OUT/score_$SC.json && cp $OUT/score_raw.npz $OUT/score_${SC}_raw.npz"

# 4. the split of every gain into information and amplitude (a gain that is only a smoother field is not counted)
if [ "${DRYRUN:-0}" != 1 ]; then
    BP=("$LP@kai_ro:kai_ro=output mix fed back (loop) on the fine-tuned KAI vs the fine-tuned KAI")
    for arm in $ARMS; do
        t=$arm$SFX
        BP+=("$t:kai_ro=$t vs the fine-tuned KAI" "$t:$LP@kai_ro=$t vs the loop on the fine-tuned KAI")
    done
    mkdir -p "$OUT/breakdown_$SC"
    python scripts/eval_kai_breakdown.py --stats "$OUT/stats.npz" "$OUT/score_$SC.json" "$OUT/score_${SC}_raw.npz" \
        "$OUT/breakdown_$SC" "${BP[@]}" > "$OUT/breakdown_$SC/breakdown_$SC.txt" 2>&1 \
        || echo "[stage] breakdown failed, see $OUT/breakdown_$SC/breakdown_$SC.txt"
    echo "[stage] done: $OUT/$SC.log, $OUT/score_$SC.json, $OUT/breakdown_$SC/breakdown_$SC.txt"
fi
