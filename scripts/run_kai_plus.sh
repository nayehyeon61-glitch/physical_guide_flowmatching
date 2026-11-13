#!/usr/bin/env bash
# Tendency core + (docs/kai_tendency_plus.md): the KAI + tendency core model of docs/kai_tendency.md at KAI's own
# resolution instead of 5.625 deg. Same pieces, trained separately, on daily means:
#   1. data     daily means of 00/06/12/18 UTC on the fine grid, <DATA>/kaiday/<year>.npy
#               GRID=1.5      KAI's grid, 121 x 240 with the pole rows       (SRC=wb2: WeatherBench2 240x121)
#               SRC=daily     from a local archive of 0.25 deg DAILY MEANS (ERA5_0p25_DAILY): regridded as they are
#               SRC=local     from a local archive of 0.25 deg hourly files: the mean of 00/06/12/18 UTC
#               GRID=1.40625  128 x 256 without poles, 4 x 4 cells = one 5.625 deg cell (SRC=local: 0.25 deg archive)
#   2. kai      KAI-alpha at its own size (3 3 15 3 / 48 96 192 288), KAI's recipe
#   3. kai_ro   the same KAI fine-tuned on 2 to 3 day rollouts (the fair baseline of the coupled model)
#   4. core+    the tendency core: physical terms G(x) + GRU memory + Mori-Zwanzig sequence loss (8-day sequences),
#               SUBSTEPS Heun steps a day (4 at 1.5 deg: the same number of grid cells travelled per solver step as
#               one step a day at 5.625 deg), gradient checkpointing per substep
#   4b. twin    the same core+ trained again with another seed (TWIN=1, the default): the best model of
#               docs/kai_tendency.md sec. 7.9 is KAI coupled with TWO separately trained cores (seed 0, 1)
#   5. score    core+ alone, the coupled loop KAI + core+ on kai and on kai_ro, term split, per-variable table;
#               TWIN=1: also the seed twin, the coupled loop KAI + both cores, and what the second core gives
#               (two cores vs one, next to the seed difference of the one-core loop), files *_plus2.*
#   4c. wide    WIDE=1: one wide core+ with the parameters of the two (hidden WIDE_HIDDEN = 72, memory WIDE_MEM_CH = 34
#               channels: 520,966 vs 2 x 259,842) and its coupled loop: is the gain of two cores an ensemble effect or
#               only twice the parameters? (docs/kai_tendency.md sec. 8.2), files *_plus2w.*
# Every step is skipped when its checkpoint / marker exists; a training that was cut off resumes (--resume).
#
#   bash scripts/run_kai_plus.sh                                  # GRID=1.5, WB2 download, OUT=runs/kaiplus_1p5
#   SRC=daily bash scripts/run_kai_plus.sh                        # from the local 0.25 deg daily means ($ARCHIVE; first
#                                                                 #   python scripts/local_era5_to_kai.py --src-daily --check ...);
#                                                                 #   folders data/era5_1p5_daily, runs/kaiplus_1p5_daily
#   GRID=1.40625 SRC=local bash scripts/run_kai_plus.sh           # from the local 0.25 deg hourly archive ($ARCHIVE)
#   SRC=none DATA=/path/with/kaiday bash scripts/run_kai_plus.sh  # the daily means are there already
#   PLUS_HIDDEN=96 PLUS_BLOCKS="2 2 2" CORE_BATCH=4 ...           # a wider core+ (about 3 GB of activations per sample; default 1.6 GB)
#   KAI_CKPT=0 KAI_BATCH=8 RO_BATCH=2 ...                         # KAI without gradient checkpointing: faster per iteration, but
#                                                                 #   about 6 GB of activations per sample and rollout day at 1.5 deg
#                                                                 #   (with it, the default: about 2 GB; CPU measurements, docs sec. 4)
#   GFEAT=0 ...                                                   # core+ without the physical terms (GRU + MZ only)
#   TWIN=0 ...                                                    # one core only (no seed twin, no two-core loop): files *_plus.*
#   WIDE=1 ...                                                    # also the wide core + of the two cores' size (the same-size control)
#   SMOKE=1 bash scripts/run_kai_plus.sh                          # a few iterations of every step on 24 days a year,
#                                                                 #   in its own folders (data/era5_1p5_smoke, runs/kaiplus_1p5_smoke)
#   KAI_ITERS=200 RO_ITERS=20 CORE_ITERS=40 LOGK=50 LOGC=10 OUT=runs/kaiplus_1p5_test bash scripts/run_kai_plus.sh
#                                                                 # a short training test at the real size on the real data: the
#                                                                 #   logs give seconds per iteration ("sec") and GPU memory ("gpu_gb")
#   CORE=0 TRAIN_YEARS="$(seq -s ' ' 1980 2011)" KAI_ITERS=160000 OUT=runs/kaiplus_1p5_daily_32y SRC=daily bash scripts/run_kai_plus.sh
#                                                                 # only steps 1 to 3 (data, KAI, fine-tuned KAI) in a new
#                                                                 #   folder, here for another training period: no core + is
#                                                                 #   trained and nothing is scored; the cores of the ++ version
#                                                                 #   follow with PLUS=0 bash scripts/run_kai_plusplus.sh
#   TWIN=0 CLIM=1 CORE_ITERS=6000 SRC=daily bash scripts/run_kai_plus.sh
#                                                                 # after a one-core run: the exact controls of the blend (KAI mixed
#                                                                 #   with the climatology, bootstrap, exact term split), step 6
#   DRYRUN=1 ...                                                  # print the commands only
set -euo pipefail
cd "$(dirname "$0")/.."
GRID=${GRID:-1.5}
TAGG=${GRID//./p}
SMOKE=${SMOKE:-0}
SFX=""
if [ "$SMOKE" = 1 ]; then  # a smoke test: its own folders, one training year, 24 days a year
    SFX=_smoke; TRAIN_YEARS=${TRAIN_YEARS:-2015}; DAYS=${DAYS:-24}
fi
SRC=${SRC:-wb2}                        # wb2 | daily | local | none
SRCTAG=""; [ "$SRC" = daily ] && SRCTAG=_daily   # the local daily means get their own folders: sources are not mixed
DATA=${DATA:-data/era5_$TAGG$SRCTAG$SFX}
OUT=${OUT:-runs/kaiplus_$TAGG$SRCTAG$SFX}
DAYS=${DAYS:-}                         # only the first n days of every year (a smoke test)
DEVICE=${DEVICE:-cuda}
if [ "$SRC" = daily ]; then ARCHIVE=${ARCHIVE:-/lustre/home/mahmed/ERA5_0p25_DAILY}; fi
ARCHIVE=${ARCHIVE:-/lustre/home/mahmed/ERA5/hourly}
# PATTERN (file layout under $ARCHIVE), OROG, LSM (0.25 deg constant files): the defaults of local_era5_to_kai.py
LOCAL_OPTS=(${PATTERN:+--pattern "$PATTERN"} ${OROG:+--orog "$OROG"} ${LSM:+--lsm "$LSM"})
WORKERS=${WORKERS:-8}
TRAIN_YEARS=${TRAIN_YEARS:-"2012 2013 2014 2015"}
VAL_YEAR=${VAL_YEAR:-2016}
TEST_YEAR=${TEST_YEAR:-2017}
THREADS=${THREADS:-8}
KAI_DEPTHS=${KAI_DEPTHS:-"3 3 15 3"}
KAI_DIMS=${KAI_DIMS:-"48 96 192 288"}
KAI_ITERS=${KAI_ITERS:-20000}
KAI_BATCH=${KAI_BATCH:-8}
KAI_CKPT=${KAI_CKPT:-1}                # gradient checkpointing per KAI block (KAI keeps the full grid in every stage)
RO_ITERS=${RO_ITERS:-1000}
RO_BATCH=${RO_BATCH:-4}                # the rollout fine-tuning stores 3 days of activations per sample
SUBSTEPS=${SUBSTEPS:-4}                # Heun steps a day: about 5.625 / GRID
PLUS_HIDDEN=${PLUS_HIDDEN:-48}
PLUS_BLOCKS=${PLUS_BLOCKS:-"2 2"}
CORE_ITERS=${CORE_ITERS:-3000}
CORE_BATCH=${CORE_BATCH:-8}
CORE_MEM=${CORE_MEM:-8}
GFEAT=${GFEAT:-1}
SCORE_BATCH=${SCORE_BATCH:-4}
VAL_BATCH=${VAL_BATCH:-4}                # days per forward pass of the validation loss
IC_STRIDE=${IC_STRIDE:-3}
LEADS=${LEADS:-10}
TWIN=${TWIN:-1}                        # 1 = train the core a second time with seed TWIN_SEED and score KAI coupled
TWIN_SEED=${TWIN_SEED:-1}              #     with both cores (docs/kai_tendency.md sec. 7.9); 0 = one core
WIDE=${WIDE:-0}                        # 1 = also one wide core with the parameters of the two (the same-size control)
WIDE_HIDDEN=${WIDE_HIDDEN:-72}; WIDE_MEM_CH=${WIDE_MEM_CH:-34}   # 2 x the default core's parameters (hidden 48, 24)
CORE=${CORE:-1}                        # 0 = stop after the fine-tuned KAI (steps 1 to 3): no core +, no score
C=core_plus; C2=${C}_s$TWIN_SEED; M="$C+$C2"; CW=${C}_w
LOGK=${LOGK:-500}; LOGC=${LOGC:-250}   # iterations between two log lines of the KAI / core training (each runs the validation loss)
if [ "$SMOKE" = 1 ]; then
    KAI_DEPTHS="2 2 4 2"; KAI_DIMS="24 48 64 96"; KAI_ITERS=4; KAI_BATCH=2; RO_ITERS=2; RO_BATCH=2; CORE_ITERS=2; CORE_BATCH=1
    SCORE_BATCH=2; VAL_BATCH=1; IC_STRIDE=6; LEADS=4; LOGK=2; LOGC=1; THREADS=${THREADS_SMOKE:-2}
fi
mkdir -p "$OUT"
step() {  # step <name> <done file> <command...>: skipped when the done file exists, log to $OUT/<name>.log
    local name=$1 mark=$2; shift 2
    if [ -e "$mark" ]; then echo "[plus] $name: done, skipped"; return; fi
    if [ "${DRYRUN:-0}" = 1 ]; then echo "[dry] $name: $(printf '%q ' "$@")"; return; fi
    echo "[plus] $name: start $(date +%H:%M), log $OUT/$name.log"
    "$@" >> "$OUT/$name.log" 2>&1 || { echo "[plus] $name FAILED, see $OUT/$name.log"; exit 1; }
}

# 1. data: one .npy of daily means per year and the constants (orography, land-sea mask) of the grid
YEARS="$TRAIN_YEARS $VAL_YEAR $TEST_YEAR"
need=""
for y in $YEARS; do [ -e "$DATA/kaiday/$y.npy" ] || [ -e "$DATA/kai6h/$y.nc" ] || need="$need $y"; done
if [ -n "$need" ]; then
    case "$SRC" in
        wb2)   [ "$GRID" = 1.5 ] || { echo "SRC=wb2 has GRID=1.5 only (240x121); use SRC=local"; exit 1; }
               step data /nonexistent python scripts/wb2_to_kai.py --out "$DATA" --grid 240x121 --daily-mean \
                   --no-climode --years $need ${DAYS:+--days $DAYS} ;;
        daily) step data /nonexistent python scripts/local_era5_to_kai.py --src "$ARCHIVE" --src-daily --out "$DATA" \
                   --grid "$GRID" --daily-mean --years $need --workers "$WORKERS" ${DAYS:+--days $DAYS} ${LOCAL_OPTS[@]+"${LOCAL_OPTS[@]}"} ;;
        local) step data /nonexistent python scripts/local_era5_to_kai.py --src "$ARCHIVE" --out "$DATA" \
                   --grid "$GRID" --daily-mean --years $need --workers "$WORKERS" ${DAYS:+--days $DAYS} ${LOCAL_OPTS[@]+"${LOCAL_OPTS[@]}"} ;;
        *)     echo "no data for the years$need under $DATA (kaiday/<year>.npy or kai6h/<year>.nc)"; exit 1 ;;
    esac
fi

D="--data $DATA/kai6h --cache $DATA/kaiday --era5 $DATA/era5_data --out $OUT --train-years $TRAIN_YEARS"
D="$D --val-year $VAL_YEAR --test-year $TEST_YEAR --device $DEVICE --threads $THREADS --val-batch $VAL_BATCH"
R="python scripts/run_kai_tendency.py $D --depths $KAI_DEPTHS --dims $KAI_DIMS"

# 2. KAI at its own size, 3. the same KAI fine-tuned in its own 2 to 3 day rollout
KC=""; [ "$KAI_CKPT" = 1 ] && KC="--kai-ckpt"
step kai "$OUT/kai.pt" $R --arm kai --tag kai --iters "$KAI_ITERS" --batch "$KAI_BATCH" --log-every $LOGK --resume $KC
step kai_ro "$OUT/kai_ro.pt" $R --arm kai --tag kai_ro --init kai --rollout 2 3 --lr 3e-4 --iters "$RO_ITERS" \
    --batch "$RO_BATCH" --log-every $LOGK --resume $KC

if [ "$CORE" != 1 ]; then
    echo "[plus] CORE=0: data, KAI and the fine-tuned KAI are in $OUT; no core + is trained and nothing is scored"
    exit 0
fi

# 4. the tendency core +
G=""; [ "$GFEAT" = 1 ] && G="--core-gfeat"
CT="--arm core --core-norm gn $G --core-mem $CORE_MEM --core-mz --core-substeps $SUBSTEPS --core-ckpt"
CT="$CT --core-blocks $PLUS_BLOCKS --iters $CORE_ITERS --batch $CORE_BATCH --log-every $LOGC --resume"
step $C "$OUT/$C.pt" $R $CT --core-hidden "$PLUS_HIDDEN" --tag $C
# 4b. its seed twin: the same core, the same training, another seed (initial weights and the order of the samples)
if [ "$TWIN" = 1 ]; then step $C2 "$OUT/$C2.pt" $R $CT --core-hidden "$PLUS_HIDDEN" --tag $C2 --seed "$TWIN_SEED"; fi
# 4c. one wide core of the two cores' size, the same training and seed as the first core
if [ "$WIDE" = 1 ]; then
    step $CW "$OUT/$CW.pt" $R $CT --core-hidden "$WIDE_HIDDEN" --core-mem-ch "$WIDE_MEM_CH" --tag $CW
fi

# 5. score: the core alone, the coupled loop on KAI and on the fine-tuned KAI; term split; per-variable table.
#    TWIN=1 (score_plus2.*): the same with both cores, KAI coupled with both (weights of KAI / core / twin per channel
#    from the day-1 forecasts of the validation year, fed back every day) and the lead-wise mix of the three
PAIRS="$C:kai loop_$C:kai kai_ro:kai loop_$C@kai_ro:kai_ro loop_$C@kai_ro:loop_$C loop_$C@kai_ro:kai"
BP=("$C:kai=tendency core + alone vs KAI"
    "loop_$C:kai=coupled loop(KAI, core +) vs KAI"
    "kai_ro:kai=KAI with rollout fine-tuning vs KAI"
    "loop_$C@kai_ro:kai_ro=coupled loop(KAI with rollout fine-tuning, core +) vs KAI with rollout fine-tuning"
    "loop_$C@kai_ro:loop_$C=coupled loop on the fine-tuned KAI vs on KAI"
    "$C~w0:$C=memory removed (0 days) vs $CORE_MEM days")
VT=("kai=KAI" "kai_ro=KAI (rollout fine-tuned)" "$C=tendency core +" "loop_$C=KAI + core + loop"
    "loop_$C@kai_ro=fine-tuned KAI + core + loop")
VP=("loop_$C@kai_ro:kai_ro" "loop_$C:kai")
if [ "$TWIN" = 1 ]; then
    SC=plus2; SN=score2; CORES="$C $C2"; MULTI="--blend-multi $M --loop-multi $M"
    PAIRS="$PAIRS $C2:$C loop_$C2:loop_$C loop_$C2@kai_ro:loop_$C@kai_ro loop_$M:loop_$C loop_$M@kai_ro:loop_$C@kai_ro"
    PAIRS="$PAIRS loop_$M@kai_ro:kai_ro blend_$M@kai_ro:blend_$C@kai_ro"
    BP+=("$C2:$C=seed difference, alone: core + trained with seed $TWIN_SEED vs seed 0"
         "loop_$C2@kai_ro:loop_$C@kai_ro=seed difference of the one-core loop on the fine-tuned KAI: core of seed $TWIN_SEED vs seed 0"
         "loop_$M:loop_$C=on KAI: coupled loop with two cores + (seed 0, $TWIN_SEED) vs with one"
         "loop_$M@kai_ro:loop_$C@kai_ro=on the fine-tuned KAI: coupled loop with two cores + (seed 0, $TWIN_SEED) vs with one"
         "loop_$M@kai_ro:kai_ro=coupled loop(fine-tuned KAI, two cores +) vs fine-tuned KAI"
         "blend_$M@kai_ro:blend_$C@kai_ro=on the fine-tuned KAI: lead-wise mix with two cores + vs with one")
    VT+=("$C2=tendency core + (seed $TWIN_SEED)" "loop_$M=KAI + two cores + loop"
         "loop_$M@kai_ro=fine-tuned KAI + two cores + loop")
    VP=("loop_$M@kai_ro:loop_$C@kai_ro" "loop_$M@kai_ro:kai_ro" "${VP[@]}")
else
    SC=plus; SN=score; CORES="$C"; MULTI=""
fi
if [ "$WIDE" = 1 ]; then  # the wide core: alone, in the loop, and against the two-core loop of the same parameters
    SC=${SC}w; SN=${SN}w; CORES="$CORES $CW"
    PAIRS="$PAIRS $CW:$C loop_$CW@kai_ro:loop_$C@kai_ro loop_$CW@kai_ro:kai_ro blend_$CW@kai_ro:blend_$C@kai_ro"
    BP+=("$CW:$C=alone: the wide core + (parameters of two cores) vs core +"
         "loop_$CW@kai_ro:loop_$C@kai_ro=on the fine-tuned KAI: coupled loop with the wide core + vs with core +"
         "loop_$CW@kai_ro:kai_ro=coupled loop(fine-tuned KAI, wide core +) vs fine-tuned KAI")
    VT+=("$CW=wide core +" "loop_$CW@kai_ro=fine-tuned KAI + wide core + loop")
    if [ "$TWIN" = 1 ]; then
        PAIRS="$PAIRS loop_$CW:loop_$M loop_$CW@kai_ro:loop_$M@kai_ro blend_$CW@kai_ro:blend_$M@kai_ro"
        BP+=("loop_$CW:loop_$M=on KAI, same parameters: coupled loop with one wide core + vs with two cores + (seed 0, $TWIN_SEED)"
             "loop_$CW@kai_ro:loop_$M@kai_ro=on the fine-tuned KAI, same parameters: coupled loop with one wide core + vs with two cores +"
             "blend_$CW@kai_ro:blend_$M@kai_ro=on the fine-tuned KAI, same parameters: lead-wise mix with one wide core + vs with two cores +")
        VP=("loop_$CW@kai_ro:loop_$M@kai_ro" "${VP[@]}")
    else
        VP=("loop_$CW@kai_ro:loop_$C@kai_ro" "${VP[@]}")
    fi
fi
step $SN "$OUT/score_$SC.json" bash -c "$R --score kai kai_ro --score-cores $CORES --mem-warm 0 1 \
    --blend-bases kai kai_ro --loop-blend $CORES $MULTI --leads $LEADS --ic-stride $IC_STRIDE --score-batch $SCORE_BATCH \
    --pairs $PAIRS '$C~w0:$C' '$C~w1:$C' \
    && cp $OUT/score.json $OUT/score_$SC.json && cp $OUT/score_raw.npz $OUT/score_${SC}_raw.npz"
if [ "${DRYRUN:-0}" != 1 ]; then
    mkdir -p "$OUT/breakdown"
    python scripts/eval_kai_breakdown.py --stats "$OUT/stats.npz" "$OUT/score_$SC.json" "$OUT/score_${SC}_raw.npz" \
        "$OUT/breakdown" "${BP[@]}" \
        > "$OUT/breakdown/breakdown_$SC.txt" 2>&1 || echo "[plus] breakdown failed, see $OUT/breakdown/breakdown_$SC.txt"
    python scripts/kai_variable_table.py "$OUT/score_$SC.json" "$OUT/variable_table_$SC.md" "${VT[@]}" \
        --base kai_ro --grid "$GRID" --pairs "${VP[@]}" > "$OUT/variable_table_$SC.log" 2>&1 \
        || echo "[plus] variable table failed, see $OUT/variable_table_$SC.log"
    echo "[plus] done: $OUT/$SN.log, $OUT/breakdown/breakdown_$SC.txt, $OUT/variable_table_$SC.md"
fi

# 6. CLIM=1: the exact controls of the lead-wise blend (docs/kai_tendency_plus_1p5_daily.md sec. 8.9). KAI and the
#    fine-tuned KAI mixed with the daily climatology in the same way as with the core (weights per lead and channel
#    from the validation year): how much of the blend's gain is only damping toward the climatology? Also the paired
#    bootstrap of the blend against KAI, the fine-tuned KAI and these damping controls, and the exact term split of
#    every pair. Nothing is trained; files *_${SC}c.*, $OUT/breakdown_clim.
if [ "${CLIM:-0}" = 1 ]; then
    SCC=${SC}c; B1="blend_$C"; B2="blend_$C@kai_ro"
    CP="$B1:kai $B2:kai $B2:kai_ro blend_climatology:kai blend_climatology@kai_ro:kai_ro"
    CP="$CP $B1:blend_climatology $B2:blend_climatology@kai_ro"
    step scoreclim "$OUT/score_$SCC.json" bash -c "$R --score kai kai_ro --score-cores $CORES --mem-warm \
        --blend-bases kai kai_ro --blend-with climatology --leads $LEADS --ic-stride $IC_STRIDE --score-batch $SCORE_BATCH \
        --pairs $CP && cp $OUT/score.json $OUT/score_$SCC.json && cp $OUT/score_raw.npz $OUT/score_${SCC}_raw.npz"
    if [ "${DRYRUN:-0}" != 1 ]; then
        mkdir -p "$OUT/breakdown_clim"
        python scripts/eval_kai_breakdown.py --stats "$OUT/stats.npz" "$OUT/score_$SCC.json" "$OUT/score_${SCC}_raw.npz" \
            "$OUT/breakdown_clim" "$B1:kai=blend vs KAI" "$B2:kai=blend on the fine-tuned KAI vs KAI" \
            "$B2:kai_ro=blend on the fine-tuned KAI vs the fine-tuned KAI" \
            "blend_climatology:kai=KAI damped toward the climatology vs KAI" \
            "blend_climatology@kai_ro:kai_ro=fine-tuned KAI damped toward the climatology vs the fine-tuned KAI" \
            "$B1:blend_climatology=blend vs KAI damped toward the climatology" \
            "$B2:blend_climatology@kai_ro=blend on the fine-tuned KAI vs the fine-tuned KAI damped toward the climatology" \
            > "$OUT/breakdown_clim/breakdown_$SCC.txt" 2>&1 \
            || echo "[plus] breakdown failed, see $OUT/breakdown_clim/breakdown_$SCC.txt"
        echo "[plus] done: $OUT/scoreclim.log, $OUT/breakdown_clim/breakdown_$SCC.txt"
    fi
fi
