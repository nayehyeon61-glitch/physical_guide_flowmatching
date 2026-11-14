#!/usr/bin/env bash
# Tendency core ++ (docs/kai_tendency_plusplus.md): the two-core model of docs/kai_tendency.md sec. 7.9 (the rollout
# fine-tuned KAI coupled with TWO separately trained tendency cores, seed 0 and 1) with the extended core of the size
# experiment (scripts/run_kai_plus_arch.sh: wider trunk or multi-scale trunk, memory 48 channels with a two-layer
# encoder) in place of the 5.625 deg network that the + version kept on the 1.5 deg grid.
# It runs in the folder of the finished + run: KAI, the fine-tuned KAI and core + are reused, and so is an extended
# core that the size experiment has trained already (same tag, same options, same seed).
#
#   A = core_plus_$ARCH        the extended core, seed 0   (w96 / w128 / u64 / u96, as in run_kai_plus_arch.sh)
#   B = core_plus_${ARCH}_s1   the same core, seed 1                  -> the ++ model = kai_ro + A + B
#       ARCH2=u96 instead:     B = core_plus_u96 (seed 0), a pair of two different extended cores (no seed twin)
#
#   1. train   A and B, each skipped when its checkpoint exists; a training that was cut off resumes. A core that is
#              being trained elsewhere (the size experiment, another GPU) is never trained a second time: the other
#              core goes first and the driver waits for that checkpoint (FORCE=1 trains it here anyway)
#   2. score   every core alone, the one-core loops, the two-core loop and the lead-wise mix (blend) on KAI and on the
#              fine-tuned KAI, next to the + version (core_plus; with core_plus_s1 in the folder also the + two-core
#              model), and the damping control (KAI mixed with the climatology like a core); paired bootstrap, exact
#              term split, per-variable table. Files *_$SC.* (score_pp.json, breakdown_pp/, variable_table_pp.md)
#
#   bash scripts/run_kai_plusplus.sh                              # ARCH=w128: train what is missing, then score
#   ARCH=u96 bash scripts/run_kai_plusplus.sh                     # the multi-scale core and its seed twin
#   ARCH=w128 ARCH2=u96 bash scripts/run_kai_plusplus.sh          # two different extended cores (nothing to train
#                                                                 #   when the size experiment has both)
#   TRAIN=0 WAIT=1 bash scripts/run_kai_plusplus.sh               # only score, after waiting for the checkpoints
#                                                                 #   (the trainings run elsewhere, one per GPU)
#   SCORE=0 bash scripts/run_kai_plusplus.sh                      # only train
#   PLUS=0 OUT=runs/kaiplus_1p5_daily_32y CORE_ITERS=48000 bash scripts/run_kai_plusplus.sh
#                                                                 # a folder without core + (another training period, made
#                                                                 #   by CORE=0 bash scripts/run_kai_plus.sh): the two extended
#                                                                 #   cores on the years of the folder's KAI, the score without
#                                                                 #   the rows of the + version (docs sec. 7)
#   TIME=0 ...                                                    # time extension off: by default, with more than 4 training
#                                                                 #   years in the folder's KAI, a CORE_ITERS that is not given
#                                                                 #   grows with the years (x years / 4: the epochs of the 4-year
#                                                                 #   run; docs/kai_time_extension.md: more years without more
#                                                                 #   iterations gave nothing)
#   GAP=0 ...                                                     # without step 3 (the generalization gap of KAI and of the two
#                                                                 #   cores: short of data or short of capacity / iterations?)
#   SMOKE=1 OUT=... DATA=... DEVICE=cpu bash ...                  # a few iterations of small variants (CPU test)
#   DRYRUN=1 ...                                                  # print the commands only
set -euo pipefail
cd "$(dirname "$0")/.."
SMOKE=${SMOKE:-0}
OUT=${OUT:-runs/kaiplus_1p5_daily}
DATA=${DATA:-data/era5_1p5_daily}
DEVICE=${DEVICE:-cuda}
THREADS=${THREADS:-8}
YEARS_GIVEN=${TRAIN_YEARS:+1}          # not given: the training years of the folder's KAI (kai.pt), see below
TRAIN_YEARS=${TRAIN_YEARS:-"2012 2013 2014 2015"}
VAL_YEAR=${VAL_YEAR:-2016}
TEST_YEAR=${TEST_YEAR:-2017}
KAI_DEPTHS=${KAI_DEPTHS:-"3 3 15 3"}   # the KAI of the folder (kai.pt, kai_ro.pt)
KAI_DIMS=${KAI_DIMS:-"48 96 192 288"}
ITERS_GIVEN=${CORE_ITERS:+1}
CORE_ITERS=${CORE_ITERS:-6000}         # the schedule of core_plus in this folder (4 training years)
TIME=${TIME:-1}                        # 1 = time extension: more than 4 training years and no CORE_ITERS given ->
                                       #     CORE_ITERS x years / 4 (the same epochs); 0 = the count above as it is
GAP=${GAP:-1}                          # 1 = step 3: one-step loss of KAI and the two cores on training years against
GAP_DAYS=${GAP_DAYS:-30}               #     the validation / test year (GAP_YEARS: other years), GAP_DAYS days a year
CORE_BATCH=${CORE_BATCH:-8}
CORE_MEM=${CORE_MEM:-8}
SUBSTEPS=${SUBSTEPS:-4}
LOGC=${LOGC:-250}
VAL_BATCH=${VAL_BATCH:-4}
ARCH=${ARCH:-w128}                     # the extended core: w96 | w128 | u64 | u96
ARCH2=${ARCH2:-}                       # empty: the second core is the seed twin of the first; else another variant
SEED2=${SEED2:-1}
TRAIN=${TRAIN:-1}; SCORE=${SCORE:-1}; WAIT=${WAIT:-0}
SC=${SC:-pp}                           # the name of the score: score_$SC.json, breakdown_$SC/, variable_table_$SC.md
PLUS=${PLUS:-1}                        # 1 = the folder holds core + (core_plus.pt) and the score compares with the +
                                       #     version; 0 = a folder without core + (CORE=0 bash scripts/run_kai_plus.sh,
                                       #     e.g. another training period): the ++ rows against KAI and the fine-tuned KAI
SCORE_BATCH=${SCORE_BATCH:-4}; IC_STRIDE=${IC_STRIDE:-3}; LEADS=${LEADS:-10}
BUSY_MIN=${BUSY_MIN:-120}              # a .part / .log that changed in the last BUSY_MIN minutes = a training is alive
POLL=${POLL:-600}                      # seconds between two looks at a checkpoint that is waited for
MEM="--core-mem-ch 48 --core-mem-enc 128"
if [ "$SMOKE" = 1 ]; then  # small variants of the same kinds, a few iterations: does every step run?
    TRAIN_YEARS=2015; KAI_DEPTHS="2 2 4 2"; KAI_DIMS="24 48 64 96"; CORE_ITERS=2; CORE_BATCH=1; LOGC=2; VAL_BATCH=1
    SCORE_BATCH=2; IC_STRIDE=6; LEADS=4; THREADS=${THREADS_SMOKE:-2}; MEM="--core-mem-ch 12 --core-mem-enc 24"
    YEARS_GIVEN=1
fi
opts_of() {  # the variants of scripts/run_kai_plus_arch.sh (the same tags give the same cores)
    if [ "$SMOKE" = 1 ]; then
        case "$1" in
            w*) echo "--core-hidden 24 $MEM" ;;
            u*) echo "--core-trunk unet --core-unet-dims 16 24 32 $MEM" ;;
            *)  echo "unknown variant $1" >&2; exit 1 ;;
        esac
        return
    fi
    case "$1" in
        w96)  echo "--core-hidden 96 $MEM" ;;
        w128) echo "--core-hidden 128 $MEM" ;;
        u64)  echo "--core-trunk unet --core-unet-dims 64 128 192 $MEM" ;;
        u96)  echo "--core-trunk unet --core-unet-dims 96 160 224 $MEM" ;;
        *)    echo "unknown variant $1" >&2; exit 1 ;;
    esac
}
name_of() {
    case "$1" in
        w96) echo "wide core (96)" ;; w128) echo "wide core (128)" ;;
        u64) echo "multi-scale core (64/128/192)" ;; u96) echo "multi-scale core (96/160/224)" ;; *) echo "core $1" ;;
    esac
}
step() {  # step <name> <done file> <command...>: skipped when the done file exists, log to $OUT/<name>.log
    local name=$1 mark=$2; shift 2
    if [ -e "$mark" ]; then echo "[++] $name: done, skipped"; return; fi
    if [ "${DRYRUN:-0}" = 1 ]; then echo "[dry] $name: $(printf '%q ' "$@")"; return; fi
    echo "[++] $name: start $(date '+%m-%d %H:%M'), log $OUT/$name.log"
    "$@" >> "$OUT/$name.log" 2>&1 || { echo "[++] $name FAILED, see $OUT/$name.log"; exit 1; }
    echo "[++] $name: end $(date '+%m-%d %H:%M')"
}
busy() {  # is the core $1 being trained by someone else? a process of this machine with --tag $1, or (another
          # machine on the same disk) its training state $1.part or its log changed in the last BUSY_MIN minutes
    pgrep -f -- "--tag $1( |\$)" > /dev/null 2>&1 && return 0
    local f
    for f in "$OUT/$1.part" "$OUT/$1.log"; do
        if [ -e "$f" ] && [ -n "$(find "$f" -mmin "-$BUSY_MIN" 2>/dev/null)" ]; then return 0; fi
    done
    return 1
}
wait_for() {  # two trainings of one tag in one folder would overwrite each other's state: wait while $1 is busy.
              # It ends with the checkpoint there (the step is then skipped) or with a training that stopped
              # (the step then resumes it here)
    if [ "${DRYRUN:-0}" = 1 ] || [ "${FORCE:-0}" = 1 ]; then return 0; fi
    while [ ! -e "$OUT/$1.pt" ] && busy "$1"; do
        echo "[++] $1 is being trained elsewhere: waiting for $1.pt $(date '+%m-%d %H:%M') (FORCE=1 trains it here)"
        sleep "$POLL"
    done
}
NEED="kai.pt kai_ro.pt"; [ "$PLUS" = 1 ] && NEED="$NEED core_plus.pt"
for f in $NEED; do
    [ -e "$OUT/$f" ] || [ "${DRYRUN:-0}" = 1 ] || {
        echo "$OUT/$f is missing: run scripts/run_kai_plus.sh first (a folder without core +: CORE=0 there, PLUS=0 here)"; exit 1; }
done
# The cores belong to the folder's KAI: the same training years (the normalization stats.npz and the climatology are
# theirs). Not given: taken from kai.pt; given and different: stop before a training of days on the wrong years.
if [ "${DRYRUN:-0}" != 1 ]; then
    KY=$(python -c "import sys, torch; print(*torch.load(sys.argv[1], map_location='cpu', weights_only=False)['args']['train_years'])" "$OUT/kai.pt")
    TRAIN_YEARS=$(echo $TRAIN_YEARS)
    if [ -z "$YEARS_GIVEN" ]; then
        TRAIN_YEARS=$KY
    elif [ "$TRAIN_YEARS" != "$KY" ]; then
        echo "TRAIN_YEARS ($TRAIN_YEARS) are not the training years of $OUT/kai.pt ($KY):"
        echo "leave TRAIN_YEARS out (the cores then take the years of the folder's KAI) or give another OUT"; exit 1
    fi
    echo "[++] folder $OUT, training years $(echo $TRAIN_YEARS | awk '{print $1 " to " $NF " (" NF ")"}'), validation $VAL_YEAR, test $TEST_YEAR"
fi
# Time extension (docs/kai_time_extension.md): a longer training period gave nothing at the iterations of the 4-year
# run and its gain came with iterations in proportion to the years, so the count follows the years unless it is given.
NY=$(echo $TRAIN_YEARS | wc -w)
if [ "$TIME" = 1 ] && [ "$SMOKE" != 1 ] && [ -z "$ITERS_GIVEN" ] && [ "$NY" -gt 4 ]; then
    CORE_ITERS=$(( CORE_ITERS * NY / 4 ))
    echo "[++] time extension: $NY training years -> $CORE_ITERS core iterations (x $NY/4, the epochs of the 4-year run; CORE_ITERS=... or TIME=0 to set them by hand)"
fi
D="--data $DATA/kai6h --cache $DATA/kaiday --era5 $DATA/era5_data --out $OUT --train-years $TRAIN_YEARS"
D="$D --val-year $VAL_YEAR --test-year $TEST_YEAR --device $DEVICE --threads $THREADS --val-batch $VAL_BATCH"
R="python scripts/run_kai_tendency.py $D --depths $KAI_DEPTHS --dims $KAI_DIMS"
CT="--arm core --core-norm gn --core-gfeat --core-mem $CORE_MEM --core-mz --core-substeps $SUBSTEPS --core-ckpt"
CT="$CT --core-blocks 2 2 --iters $CORE_ITERS --batch $CORE_BATCH --log-every $LOGC --resume"

C=core_plus
A=${C}_$ARCH; NA=$(name_of "$ARCH")
if [ -n "$ARCH2" ]; then
    [ "$ARCH2" != "$ARCH" ] || { echo "ARCH2 = ARCH: leave ARCH2 empty for the seed twin"; exit 1; }
    B=${C}_$ARCH2; NB=$(name_of "$ARCH2"); OB="$(opts_of "$ARCH2")"; WHAT="two different extended cores"
else
    B=${A}_s$SEED2; NB="$NA, seed $SEED2"; OB="$(opts_of "$ARCH") --seed $SEED2"; WHAT="the extended core and its seed twin"
fi
M="$A+$B"

# 1. the two extended cores
if [ "$TRAIN" = 1 ]; then
    ORDER="A B"  # the core that is being trained elsewhere goes last: the other one is trained here meanwhile
    if [ ! -e "$OUT/$A.pt" ] && [ "${DRYRUN:-0}" != 1 ] && [ "${FORCE:-0}" != 1 ] && busy "$A"; then ORDER="B A"; fi
    for x in $ORDER; do
        if [ "$x" = A ]; then t=$A; o="$(opts_of "$ARCH")"; else t=$B; o="$OB"; fi
        wait_for "$t"
        step "$t" "$OUT/$t.pt" $R $CT $o --tag "$t"
    done
fi
[ "$SCORE" = 1 ] || exit 0
if [ "$WAIT" = 1 ] && [ "${DRYRUN:-0}" != 1 ]; then
    for t in $A $B; do
        until [ -e "$OUT/$t.pt" ]; do echo "[++] waiting for $t.pt $(date '+%m-%d %H:%M')"; sleep 600; done
    done
fi

# 2. score. The ++ model = loop_$M@kai_ro (trajectory) and blend_$M@kai_ro (lowest RMSE).
if [ "$PLUS" = 1 ]; then
CORES="$C $A $B"; MULTI="$M"
PAIRS="loop_$M@kai_ro:kai_ro loop_$M@kai_ro:kai loop_$M@kai_ro:loop_$C@kai_ro loop_$M@kai_ro:loop_$A@kai_ro"
PAIRS="$PAIRS loop_$A@kai_ro:loop_$C@kai_ro loop_$B@kai_ro:loop_$A@kai_ro $A:$C $B:$A"
PAIRS="$PAIRS blend_$M@kai_ro:kai_ro blend_$M@kai_ro:kai blend_$M@kai_ro:blend_$C@kai_ro blend_$M@kai_ro:loop_$M@kai_ro"
PAIRS="$PAIRS blend_$M@kai_ro:blend_climatology@kai_ro blend_$C@kai_ro:blend_climatology@kai_ro"
PAIRS="$PAIRS blend_climatology@kai_ro:kai_ro blend_climatology:kai"
BP=("loop_$M@kai_ro:kai_ro=++ loop (fine-tuned KAI, $WHAT) vs the fine-tuned KAI"
    "loop_$M@kai_ro:loop_$C@kai_ro=++ loop vs the + loop (fine-tuned KAI with one core +)"
    "loop_$M@kai_ro:loop_$A@kai_ro=two extended cores vs one in the loop on the fine-tuned KAI"
    "loop_$A@kai_ro:loop_$C@kai_ro=one core in the loop on the fine-tuned KAI: $NA vs core +"
    "loop_$B@kai_ro:loop_$A@kai_ro=the second extended core vs the first, each in the one-core loop"
    "$A:$C=alone: $NA vs core +"
    "blend_$M@kai_ro:kai_ro=++ blend (fine-tuned KAI mixed with both extended cores) vs the fine-tuned KAI"
    "blend_$M@kai_ro:kai=++ blend vs KAI"
    "blend_$M@kai_ro:blend_$C@kai_ro=++ blend vs the + blend (one core +)"
    "blend_$M@kai_ro:loop_$M@kai_ro=++ blend vs ++ loop"
    "blend_$M@kai_ro:blend_climatology@kai_ro=++ blend vs the fine-tuned KAI damped toward the climatology"
    "blend_$C@kai_ro:blend_climatology@kai_ro=+ blend vs the fine-tuned KAI damped toward the climatology"
    "blend_climatology@kai_ro:kai_ro=fine-tuned KAI damped toward the climatology vs the fine-tuned KAI")
VT=("kai=KAI" "kai_ro=KAI (rollout fine-tuned)" "loop_$C@kai_ro=+ loop: fine-tuned KAI + core +"
    "loop_$A@kai_ro=fine-tuned KAI + $NA loop" "loop_$M@kai_ro=++ loop: fine-tuned KAI + two extended cores"
    "blend_$C@kai_ro=+ blend" "blend_$M@kai_ro=++ blend" "blend_climatology@kai_ro=fine-tuned KAI damped to climatology")
VP=("loop_$M@kai_ro:loop_$C@kai_ro" "loop_$M@kai_ro:kai_ro" "blend_$M@kai_ro:blend_$C@kai_ro" "blend_$M@kai_ro:kai_ro")
else  # PLUS=0, a folder without core +: the ++ rows against KAI, the fine-tuned KAI and the one-core models
CORES="$A $B"; MULTI="$M"
PAIRS="loop_$M@kai_ro:kai_ro loop_$M@kai_ro:kai loop_$M@kai_ro:loop_$A@kai_ro loop_$A@kai_ro:kai_ro"
PAIRS="$PAIRS loop_$B@kai_ro:loop_$A@kai_ro $A:kai $B:$A kai_ro:kai"
PAIRS="$PAIRS blend_$M@kai_ro:kai_ro blend_$M@kai_ro:kai blend_$M@kai_ro:blend_$A@kai_ro blend_$A@kai_ro:kai_ro"
PAIRS="$PAIRS blend_$M@kai_ro:loop_$M@kai_ro blend_$M@kai_ro:blend_climatology@kai_ro"
PAIRS="$PAIRS blend_climatology@kai_ro:kai_ro blend_climatology:kai"
BP=("loop_$M@kai_ro:kai_ro=++ loop (fine-tuned KAI, $WHAT) vs the fine-tuned KAI"
    "loop_$M@kai_ro:kai=++ loop vs KAI"
    "loop_$A@kai_ro:kai_ro=one extended core in the loop on the fine-tuned KAI ($NA) vs the fine-tuned KAI"
    "loop_$M@kai_ro:loop_$A@kai_ro=two extended cores vs one in the loop on the fine-tuned KAI"
    "loop_$B@kai_ro:loop_$A@kai_ro=the second extended core vs the first, each in the one-core loop"
    "$A:kai=alone: $NA vs KAI"
    "kai_ro:kai=KAI with rollout fine-tuning vs KAI"
    "blend_$M@kai_ro:kai_ro=++ blend (fine-tuned KAI mixed with both extended cores) vs the fine-tuned KAI"
    "blend_$M@kai_ro:kai=++ blend vs KAI"
    "blend_$M@kai_ro:blend_$A@kai_ro=blend with two extended cores vs with one"
    "blend_$M@kai_ro:loop_$M@kai_ro=++ blend vs ++ loop"
    "blend_$M@kai_ro:blend_climatology@kai_ro=++ blend vs the fine-tuned KAI damped toward the climatology"
    "blend_climatology@kai_ro:kai_ro=fine-tuned KAI damped toward the climatology vs the fine-tuned KAI")
VT=("kai=KAI" "kai_ro=KAI (rollout fine-tuned)" "$A=$NA alone" "loop_$A@kai_ro=fine-tuned KAI + $NA loop"
    "loop_$M@kai_ro=++ loop: fine-tuned KAI + two extended cores" "blend_$A@kai_ro=blend with one extended core"
    "blend_$M@kai_ro=++ blend" "blend_climatology@kai_ro=fine-tuned KAI damped to climatology")
VP=("loop_$M@kai_ro:kai_ro" "loop_$M@kai_ro:loop_$A@kai_ro" "blend_$M@kai_ro:kai_ro" "blend_$M@kai_ro:kai")
fi
C2=${C}_s1
if [ "$PLUS" = 1 ] && [ -e "$OUT/$C2.pt" ] && [ "$B" != "$C2" ]; then  # the + two-core model (seed 0, 1): the same-structure baseline
    MB="$C+$C2"; CORES="$CORES $C2"; MULTI="$MULTI $MB"
    PAIRS="$PAIRS loop_$M@kai_ro:loop_$MB@kai_ro blend_$M@kai_ro:blend_$MB@kai_ro loop_$MB@kai_ro:loop_$C@kai_ro"
    BP+=("loop_$M@kai_ro:loop_$MB@kai_ro=++ loop vs the + two-core loop (fine-tuned KAI, two cores +)"
         "blend_$M@kai_ro:blend_$MB@kai_ro=++ blend vs the + two-core blend"
         "loop_$MB@kai_ro:loop_$C@kai_ro=+ version: two cores + vs one in the loop on the fine-tuned KAI")
    VT+=("loop_$MB@kai_ro=+ two-core loop: fine-tuned KAI + two cores +")
    VP=("loop_$M@kai_ro:loop_$MB@kai_ro" "${VP[@]}")
fi
step "score_$SC" "$OUT/score_$SC.json" bash -c "$R --score kai kai_ro --score-cores $CORES --blend-bases kai kai_ro \
    --loop-blend $CORES --blend-multi $MULTI --loop-multi $MULTI --blend-with climatology \
    --leads $LEADS --ic-stride $IC_STRIDE --score-batch $SCORE_BATCH --pairs $PAIRS \
    && cp $OUT/score.json $OUT/score_$SC.json && cp $OUT/score_raw.npz $OUT/score_${SC}_raw.npz"
if [ "${DRYRUN:-0}" != 1 ]; then
    mkdir -p "$OUT/breakdown_$SC"
    python scripts/eval_kai_breakdown.py --stats "$OUT/stats.npz" "$OUT/score_$SC.json" "$OUT/score_${SC}_raw.npz" \
        "$OUT/breakdown_$SC" "${BP[@]}" \
        > "$OUT/breakdown_$SC/breakdown_$SC.txt" 2>&1 || echo "[++] breakdown failed, see $OUT/breakdown_$SC/breakdown_$SC.txt"
    python scripts/kai_variable_table.py "$OUT/score_$SC.json" "$OUT/variable_table_$SC.md" "${VT[@]}" \
        --base kai_ro --grid 1.5 --pairs "${VP[@]}" > "$OUT/variable_table_$SC.log" 2>&1 \
        || echo "[++] variable table failed, see $OUT/variable_table_$SC.log"
    echo "[++] done: $OUT/score_$SC.log, $OUT/breakdown_$SC/breakdown_$SC.txt, $OUT/variable_table_$SC.md"
fi

# 3. time extension: are KAI and the cores short of data or short of capacity / iterations? The one-step loss (and the
#    loss of the model's own 3-day rollout) on years of the training period against years the model never saw. A gap
#    of a few % = more years alone will not help (extend ARCH or the iterations); a large gap = more years help.
if [ "$GAP" = 1 ]; then
    if [ -z "${GAP_YEARS:-}" ]; then  # the first, the middle and the last training year, the validation and the test year
        TY=($TRAIN_YEARS); n=${#TY[@]}
        GAP_YEARS=$(echo "${TY[0]} ${TY[$(( (n - 1) / 2 ))]} ${TY[$(( n - 1 ))]} $VAL_YEAR $TEST_YEAR" | tr ' ' '\n' | awk '!s[$0]++' | tr '\n' ' ')
    fi
    step "gap_$SC" "$OUT/diag_gap_$SC.json" python scripts/kai_gap_diag.py $D --depths $KAI_DEPTHS --dims $KAI_DIMS \
        --kai kai kai_ro --cores $A $B --years $GAP_YEARS --days "$GAP_DAYS" --gap-leads 3 --json "$OUT/diag_gap_$SC.json"
    if [ "${DRYRUN:-0}" != 1 ] && [ -e "$OUT/gap_$SC.log" ]; then
        grep -E "trained on|day 1 loss|reading" "$OUT/gap_$SC.log" | sed 's/^/[++] gap  /'
    fi
fi
