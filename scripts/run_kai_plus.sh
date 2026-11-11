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
#   5. score    core+ alone, the coupled loop KAI + core+ on kai and on kai_ro, term split, per-variable table
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
#   SMOKE=1 bash scripts/run_kai_plus.sh                          # a few iterations of every step on 24 days a year,
#                                                                 #   in its own folders (data/era5_1p5_smoke, runs/kaiplus_1p5_smoke)
#   KAI_ITERS=200 RO_ITERS=20 CORE_ITERS=40 LOGK=50 LOGC=10 OUT=runs/kaiplus_1p5_test bash scripts/run_kai_plus.sh
#                                                                 # a short training test at the real size on the real data: the
#                                                                 #   logs give seconds per iteration ("sec") and GPU memory ("gpu_gb")
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
C=core_plus
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

# 4. the tendency core +
G=""; [ "$GFEAT" = 1 ] && G="--core-gfeat"
step $C "$OUT/$C.pt" $R --arm core --tag $C --core-norm gn $G --core-mem "$CORE_MEM" --core-mz \
    --core-substeps "$SUBSTEPS" --core-ckpt --core-hidden "$PLUS_HIDDEN" --core-blocks $PLUS_BLOCKS \
    --iters "$CORE_ITERS" --batch "$CORE_BATCH" --log-every $LOGC --resume

# 5. score: the core alone, the coupled loop on KAI and on the fine-tuned KAI; term split; per-variable table
step score "$OUT/score_plus.json" bash -c "$R --score kai kai_ro --score-cores $C --mem-warm 0 1 \
    --blend-bases kai kai_ro --loop-blend $C --leads $LEADS --ic-stride $IC_STRIDE --score-batch $SCORE_BATCH \
    --pairs $C:kai loop_$C:kai kai_ro:kai loop_$C@kai_ro:kai_ro loop_$C@kai_ro:loop_$C loop_$C@kai_ro:kai \
        '$C~w0:$C' '$C~w1:$C' \
    && cp $OUT/score.json $OUT/score_plus.json && cp $OUT/score_raw.npz $OUT/score_plus_raw.npz"
if [ "${DRYRUN:-0}" != 1 ]; then
    mkdir -p "$OUT/breakdown"
    python scripts/eval_kai_breakdown.py --stats "$OUT/stats.npz" "$OUT/score_plus.json" "$OUT/score_plus_raw.npz" \
        "$OUT/breakdown" \
        "$C:kai=tendency core + alone vs KAI" \
        "loop_$C:kai=coupled loop(KAI, core +) vs KAI" \
        "kai_ro:kai=KAI with rollout fine-tuning vs KAI" \
        "loop_$C@kai_ro:kai_ro=coupled loop(KAI with rollout fine-tuning, core +) vs KAI with rollout fine-tuning" \
        "loop_$C@kai_ro:loop_$C=coupled loop on the fine-tuned KAI vs on KAI" \
        "$C~w0:$C=memory removed (0 days) vs $CORE_MEM days" \
        > "$OUT/breakdown/breakdown_plus.txt" 2>&1 || echo "[plus] breakdown failed, see $OUT/breakdown/breakdown_plus.txt"
    python scripts/kai_variable_table.py "$OUT/score_plus.json" "$OUT/variable_table_plus.md" \
        "kai=KAI" "kai_ro=KAI (rollout fine-tuned)" "$C=tendency core +" "loop_$C=KAI + core + loop" \
        "loop_$C@kai_ro=fine-tuned KAI + core + loop" --base kai_ro --grid "$GRID" \
        --pairs "loop_$C@kai_ro:kai_ro" "loop_$C:kai" > "$OUT/variable_table_plus.log" 2>&1 \
        || echo "[plus] variable table failed, see $OUT/variable_table_plus.log"
    echo "[plus] done: $OUT/score.log, $OUT/breakdown/breakdown_plus.txt, $OUT/variable_table_plus.md"
fi
