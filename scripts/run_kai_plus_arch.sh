#!/usr/bin/env bash
# Tendency core + at 1.5 deg: is the network too small? (docs/kai_tendency_plus_arch.md)
# The first full run (docs/kai_tendency_plus_1p5_daily.md) kept the 5.625 deg network (hidden 48, 259,842 parameters)
# on a grid with 14 times the points. This driver trains variants of the core in the folder of that run, with the
# same data, KAI, schedule and seed, and scores them against its core (core_plus):
#
#   s1    core_plus_s1    the same core, seed 1: the size of a seed difference (what a variant has to beat)
#   w96   core_plus_w96   trunk 48 -> 96 wide, memory 24 -> 48 channels, two-layer encoder of the memory's inputs
#   w128  core_plus_w128  the same with a trunk 128 wide
#   u64   core_plus_u64   multi-scale trunk (1.5 / 3 / 6 deg, widths 64 / 128 / 192), memory as in w96
#   u96   core_plus_u96   the same with widths 96 / 160 / 224
#
#   VARIANTS="w96" bash scripts/run_kai_plus_arch.sh             # train one variant (skipped when its .pt exists)
#   VARIANTS="s1 w96 u64" SCORE=1 bash scripts/run_kai_plus_arch.sh   # train what is missing, then score all of them
#   VARIANTS="s1 w96 u64" TRAIN=0 SCORE=1 WAIT=1 bash ...        # only score, after waiting for the checkpoints
#                                                                #   (the trainings run elsewhere, one per GPU)
#   VARIANTS="w128" TRAIN=0 SCORE=1 SC=arch_w128 bash ...        # a score of its own name (files *_arch_w128.*)
#   DRYRUN=1 ...                                                 # print the commands only
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/kaiplus_1p5_daily}
DATA=${DATA:-data/era5_1p5_daily}
DEVICE=${DEVICE:-cuda}
THREADS=${THREADS:-8}
CORE_ITERS=${CORE_ITERS:-6000}         # the schedule of core_plus in this folder
CORE_BATCH=${CORE_BATCH:-8}
LOGC=${LOGC:-250}
VARIANTS=${VARIANTS:-"s1 w96 u64"}
TRAIN=${TRAIN:-1}; SCORE=${SCORE:-0}; WAIT=${WAIT:-0}
SC=${SC:-arch}                          # the name of the score: score_$SC.json, breakdown_$SC/, variable_table_$SC.md
SCORE_BATCH=${SCORE_BATCH:-4}; IC_STRIDE=${IC_STRIDE:-3}; LEADS=${LEADS:-10}
MEM="--core-mem-ch 48 --core-mem-enc 128"
opts_of() {
    case "$1" in
        s1)   echo "--core-hidden 48 --seed 1" ;;
        w96)  echo "--core-hidden 96 $MEM" ;;
        w128) echo "--core-hidden 128 $MEM" ;;
        u64)  echo "--core-trunk unet --core-unet-dims 64 128 192 $MEM" ;;
        u96)  echo "--core-trunk unet --core-unet-dims 96 160 224 $MEM" ;;
        *)    echo "unknown variant $1" >&2; exit 1 ;;
    esac
}
name_of() {
    case "$1" in
        s1) echo "core + (seed 1)" ;; w96) echo "wide core + (96)" ;; w128) echo "wide core + (128)" ;;
        u64) echo "multi-scale core + (64/128/192)" ;; u96) echo "multi-scale core + (96/160/224)" ;;
    esac
}
step() {  # step <name> <done file> <command...>
    local name=$1 mark=$2; shift 2
    if [ -e "$mark" ]; then echo "[arch] $name: done, skipped"; return; fi
    if [ "${DRYRUN:-0}" = 1 ]; then echo "[dry] $name: $(printf '%q ' "$@")"; return; fi
    echo "[arch] $name: start $(date '+%m-%d %H:%M'), log $OUT/$name.log"
    "$@" >> "$OUT/$name.log" 2>&1 || { echo "[arch] $name FAILED, see $OUT/$name.log"; exit 1; }
    echo "[arch] $name: end $(date '+%m-%d %H:%M')"
}
D="--data $DATA/kai6h --cache $DATA/kaiday --era5 $DATA/era5_data --out $OUT --train-years 2012 2013 2014 2015"
D="$D --val-year 2016 --test-year 2017 --device $DEVICE --threads $THREADS --val-batch 4"
R="python scripts/run_kai_tendency.py $D --depths 3 3 15 3 --dims 48 96 192 288"
CT="--arm core --core-norm gn --core-gfeat --core-mem 8 --core-mz --core-substeps 4 --core-ckpt --core-blocks 2 2"
CT="$CT --iters $CORE_ITERS --batch $CORE_BATCH --log-every $LOGC --resume"

if [ "$TRAIN" = 1 ]; then
    for v in $VARIANTS; do
        step "core_plus_$v" "$OUT/core_plus_$v.pt" $R $CT $(opts_of "$v") --tag "core_plus_$v"
    done
fi
[ "$SCORE" = 1 ] || exit 0

if [ "$WAIT" = 1 ] && [ "${DRYRUN:-0}" != 1 ]; then
    for v in $VARIANTS; do
        until [ -e "$OUT/core_plus_$v.pt" ]; do echo "[arch] waiting for core_plus_$v.pt $(date '+%m-%d %H:%M')"; sleep 600; done
    done
fi
# score: every core alone and in the coupled loop on KAI and on the fine-tuned KAI, against core_plus
C=core_plus
CORES="$C"; PAIRS="$C:kai loop_$C@kai_ro:kai_ro"
BP=("loop_$C@kai_ro:kai_ro=coupled loop(fine-tuned KAI, core +) vs fine-tuned KAI")
VT=("kai=KAI" "kai_ro=KAI (rollout fine-tuned)" "$C=tendency core +" "loop_$C@kai_ro=fine-tuned KAI + core + loop")
VP=()
for v in $VARIANTS; do
    T=${C}_$v; N=$(name_of "$v")
    CORES="$CORES $T"
    PAIRS="$PAIRS $T:$C loop_$T:loop_$C loop_$T@kai_ro:loop_$C@kai_ro loop_$T@kai_ro:kai_ro"
    BP+=("$T:$C=alone: $N vs core +"
         "loop_$T@kai_ro:loop_$C@kai_ro=on the fine-tuned KAI: coupled loop with the $N vs with core +"
         "loop_$T@kai_ro:kai_ro=coupled loop(fine-tuned KAI, $N) vs fine-tuned KAI")
    VT+=("$T=$N" "loop_$T@kai_ro=fine-tuned KAI + $N loop")
    VP+=("loop_$T@kai_ro:loop_$C@kai_ro")
done
step "score_$SC" "$OUT/score_$SC.json" bash -c "$R --score kai kai_ro --score-cores $CORES --blend-bases kai kai_ro \
    --loop-blend $CORES --leads $LEADS --ic-stride $IC_STRIDE --score-batch $SCORE_BATCH --pairs $PAIRS \
    && cp $OUT/score.json $OUT/score_$SC.json && cp $OUT/score_raw.npz $OUT/score_${SC}_raw.npz"
if [ "${DRYRUN:-0}" != 1 ]; then
    mkdir -p "$OUT/breakdown_$SC"
    python scripts/eval_kai_breakdown.py --stats "$OUT/stats.npz" "$OUT/score_$SC.json" "$OUT/score_${SC}_raw.npz" \
        "$OUT/breakdown_$SC" "${BP[@]}" \
        > "$OUT/breakdown_$SC/breakdown_$SC.txt" 2>&1 || echo "[arch] breakdown failed, see $OUT/breakdown_$SC/breakdown_$SC.txt"
    python scripts/kai_variable_table.py "$OUT/score_$SC.json" "$OUT/variable_table_$SC.md" "${VT[@]}" \
        --base kai_ro --grid 1.5 --pairs "${VP[@]}" > "$OUT/variable_table_$SC.log" 2>&1 \
        || echo "[arch] variable table failed, see $OUT/variable_table_$SC.log"
    echo "[arch] done: $OUT/score_$SC.log, $OUT/breakdown_$SC/breakdown_$SC.txt, $OUT/variable_table_$SC.md"
fi
