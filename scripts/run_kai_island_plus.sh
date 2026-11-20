#!/usr/bin/env bash
# Kalman island + version (docs/kai_island_plus.md; owner 2026-11-20 12:30 KST: "이 칼만 섬에 대해서도 1.5도 기준 gpu에서
# 10, 32년도 학습시킬 수 있도록 + 버전 작성"): the best 5.625 deg model of 2026-11-18 (D3, docs/kai_island_plus.md sec. 1)
# at KAI's 1.5 deg grid on the GPU, for a longer training period. One command runs the whole chain; every step is
# skipped when its result exists, a training that was cut off resumes.
#
#   0. data, KAI, the fine-tuned KAI      scripts/run_kai_plus.sh CORE=0 in the same folder (the measurement of the
#                                         island; a folder whose KAI is trained already is reused as it is)
#   1. first core  $C1                   the tendency core with the 8-day memory, the Mori-Zwanzig sequence loss and
#                                         the dycore's 24 h integrated hint (core_gndmz8 at 5.625 deg); at 1.5 deg the
#                                         dycore integrates on its own T21 grid and the increment is interpolated back
#   2. Kalman table, the first core alone diag_island_kalman.py on the validation year (the gain per channel and
#                                         latitude, K = (S - c) / (S + R - 2c), for day 1-4 and the island) and the
#                                         first core's own 10-day forecasts of both years (the simple blend's base)
#   3. second core $C2                   trained behind the frozen first core and the fixed Kalman island (it takes
#                                         over the first core's memory at the island; core_s2memx at 5.625 deg)
#   4. the island run (B2)               day 1-4 output corrected toward KAI, the island corrected, $C2 to day 10
#   5. matrix island                     island_matrix_gain.py: the gain's shape from the validation year's cases
#                                         (cross-variable x 3 x 3 stencil of points 6 deg apart), then the island run
#                                         again with it (B3)
#   6. late KAI correction               make_full_blend.py: every lead corrected toward KAI with gains of the
#                                         validation year: D2 (on B2), D3 (on B3, the best model), S (on the first
#                                         core alone = the simple blend, the control of the island structure)
#   7. scores                            equal-smoothing check of every model against the fine-tuned KAI, D3 against
#                                         D2 and against S; per-variable table and the 2x2 figure
#
#   YEARS=10 bash scripts/run_kai_island_plus.sh          # 2006-2015, folder runs/kaiplus_1p5_daily_10y
#   YEARS=32 bash scripts/run_kai_island_plus.sh          # 1980-2011, folder runs/kaiplus_1p5_daily_32y (the folder of
#                                                         #   the 32-year ++ command: its KAI is reused when it is there)
#   YEARS=4 ...                                           # 2012-2015, folder runs/kaiplus_1p5_daily (the + run's KAI)
#   TRAIN_YEARS="..." OUT=... ...                         # any other period, in its own folder
#   SMOKE=1 DEVICE=cpu bash scripts/run_kai_island_plus.sh   # every step with a few iterations on 48 days a year
#   STOP=1 ...                                            # stop after the first core (step 1); STOP=3 after the
#                                                         #   second core, ...
#   DRYRUN=1 ...                                          # print the commands only
# Iterations (TIME=1, the default): the 4-year counts below x years / 4 unless given (the epochs of the 4-year run).
# GPU: CUDA_VISIBLE_DEVICES picks the card; the dycore (JAX) runs on the CPU (JAX_PLATFORMS=cpu).
set -euo pipefail
cd "$(dirname "$0")/.."
SMOKE=${SMOKE:-0}
YEARS=${YEARS:-10}
case "$YEARS" in
    4)  TY="2012 2013 2014 2015" ;;
    10) TY="$(seq -s ' ' 2006 2015)" ;;
    32) TY="$(seq -s ' ' 1980 2011)" ;;
    *)  TY="" ;;
esac
SRC=${SRC:-daily}                       # the owner's local 0.25 deg daily means (run_kai_plus.sh SRC=...)
SRCTAG=""; [ "$SRC" = daily ] && SRCTAG=_daily
if [ "$SMOKE" = 1 ]; then
    TRAIN_YEARS=${TRAIN_YEARS:-2015}; DAYS=${DAYS:-48}
    DATA=${DATA:-data/era5_1p5${SRCTAG}_smoke}; OUT=${OUT:-runs/kaiisl_1p5${SRCTAG}_smoke}
fi
TRAIN_YEARS=${TRAIN_YEARS:-$TY}
[ -n "$TRAIN_YEARS" ] || { echo "YEARS=4|10|32 or TRAIN_YEARS=\"...\""; exit 1; }
NY=$(echo $TRAIN_YEARS | wc -w)
if [ "$NY" = 4 ]; then OUT=${OUT:-runs/kaiplus_1p5$SRCTAG}; else OUT=${OUT:-runs/kaiplus_1p5${SRCTAG}_${NY}y}; fi
DATA=${DATA:-data/era5_1p5$SRCTAG}
VAL_YEAR=${VAL_YEAR:-2016}
TEST_YEAR=${TEST_YEAR:-2017}
DEVICE=${DEVICE:-cuda}
THREADS=${THREADS:-8}
KAI_DEPTHS=${KAI_DEPTHS:-"3 3 15 3"}
KAI_DIMS=${KAI_DIMS:-"48 96 192 288"}
MEAS=${MEAS:-kai_ro}                    # the island's measurement: the fine-tuned KAI of the folder
SUBSTEPS=${SUBSTEPS:-4}                 # Heun steps a day of the cores (4 at 1.5 deg, as the + core)
HIDDEN=${HIDDEN:-48}
BLOCKS=${BLOCKS:-"2 2"}
DYC_GRID=${DYC_GRID:-}                  # 64 = the dycore on T42 (128 x 64) instead of T21 (64 x 32)
C1=${C1:-core_m8d}                      # the first core (8-day memory + MZ loss + integrated dycore hint)
C2=${C2:-${C1}_s2}                      # the second core (memory, behind the Kalman island)
C1_ITERS_GIVEN=${C1_ITERS:+1}; C2_ITERS_GIVEN=${C2_ITERS:+1}
C1_ITERS=${C1_ITERS:-3000}; C1_BATCH=${C1_BATCH:-8}; C1_ACCUM=${C1_ACCUM:-1}; C1_LR=${C1_LR:-4e-4}
C2_ITERS=${C2_ITERS:-1200}; C2_BATCH=${C2_BATCH:-4}; C2_ACCUM=${C2_ACCUM:-2}; C2_LR=${C2_LR:-3e-4}
C2_CURRICULUM=${C2_CURRICULUM:-"2 4 5 5"}
ISLAND=5                                # the island day (the second core takes over after it)
LEADS=${LEADS:-10}
IC_STRIDE=${IC_STRIDE:-3}               # 119 start dates a year (one forecast cache: 4.6 GB at 1.5 deg)
LIB_STRIDE=${LIB_STRIDE:-5}             # the Kalman table's library: every 5th day of the validation year
FOLD=${FOLD:-10}                        # blocks of start dates of the matrix island's two folds
CHUNK=${CHUNK:-4}                       # start dates per batch of the island runs (x 8 in the library)
LOGC=${LOGC:-100}
TIME=${TIME:-1}
STOP=${STOP:-99}
if [ "$SMOKE" = 1 ]; then
    HIDDEN=16; BLOCKS="1 1"; C1_ITERS=2; C1_BATCH=1; C2_ITERS=2; C2_BATCH=1; C2_ACCUM=1; C2_CURRICULUM="2"
    LEADS=7; IC_STRIDE=6; LIB_STRIDE=4; FOLD=2; CHUNK=1; LOGC=1; THREADS=${THREADS_SMOKE:-3}
    KAI_DEPTHS="2 2 4 2"; KAI_DIMS="24 48 64 96"
fi
if [ "$TIME" = 1 ] && [ "$SMOKE" != 1 ] && [ "$NY" -gt 4 ]; then
    [ -n "$C1_ITERS_GIVEN" ] || C1_ITERS=$(( C1_ITERS * NY / 4 ))
    [ -n "$C2_ITERS_GIVEN" ] || C2_ITERS=$(( C2_ITERS * NY / 4 ))
fi
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}           # the dycore on the CPU: JAX never takes the GPU from torch
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$OUT"
if [ "${DRYRUN:-0}" != 1 ] && ! python -c "import jax, dinosaur" 2>/dev/null; then  # before days of KAI training
    echo "[island+] the dycore hint needs jax and dinosaur: pip install jax \"git+https://github.com/neuralgcm/dinosaur.git\""
    exit 1
fi
echo "[island+] $NY training years ($(echo $TRAIN_YEARS | cut -d' ' -f1)..$(echo $TRAIN_YEARS | awk '{print $NF}')), validation $VAL_YEAR, test $TEST_YEAR, folder $OUT, data $DATA, device $DEVICE"
echo "[island+] first core $C1: $C1_ITERS it (batch $C1_BATCH x $C1_ACCUM); second core $C2: $C2_ITERS it (batch $C2_BATCH x $C2_ACCUM)"
step() {  # step <name> <done file> <command...>: skipped when the done file exists, log to $OUT/<name>.log
    local name=$1 mark=$2; shift 2
    if [ -e "$mark" ]; then echo "[island+] $name: done, skipped"; return; fi
    if [ "${DRYRUN:-0}" = 1 ]; then echo "[dry] $name: $(printf '%q ' "$@")"; return; fi
    echo "[island+] $name: start $(date +%m-%d_%H:%M), log $OUT/$name.log"
    "$@" >> "$OUT/$name.log" 2>&1 || { echo "[island+] $name FAILED, see $OUT/$name.log"; exit 1; }
}
stop_after() { if [ "$STOP" -le "$1" ]; then echo "[island+] STOP=$STOP: stopped after step $1"; exit 0; fi; }

# 0. data, KAI and the fine-tuned KAI (the + driver without its cores; its own skips and time extension)
if [ "${DRYRUN:-0}" = 1 ]; then
    echo "[dry] kai: SMOKE=$SMOKE CORE=0 SRC=$SRC OUT=$OUT DATA=$DATA TRAIN_YEARS=\"$TRAIN_YEARS\" bash scripts/run_kai_plus.sh"
else
    SMOKE=$SMOKE CORE=0 SRC=$SRC GRID=1.5 OUT="$OUT" DATA="$DATA" TRAIN_YEARS="$TRAIN_YEARS" VAL_YEAR=$VAL_YEAR \
        TEST_YEAR=$TEST_YEAR DEVICE=$DEVICE THREADS=$THREADS KAI_DEPTHS="$KAI_DEPTHS" KAI_DIMS="$KAI_DIMS" \
        DAYS="${DAYS:-}" bash scripts/run_kai_plus.sh
fi
if [ -e "$OUT/kai.pt" ] && [ "${DRYRUN:-0}" != 1 ]; then  # the folder's KAI must be of this training period
    python - "$OUT/kai.pt" $TRAIN_YEARS <<'EOF'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("args")
ty = list(getattr(a, "train_years", []) or [])
want = [int(y) for y in sys.argv[2:]]
if ty and ty != want:
    sys.exit(f"[island+] {sys.argv[1]} was trained on {ty[0]}..{ty[-1]} ({len(ty)} years), not on the {len(want)} years asked for: use another OUT")
EOF
fi

D="--data $DATA/kai6h --cache $DATA/kaiday --era5 $DATA/era5_data --out $OUT --train-years $TRAIN_YEARS"
D="$D --val-year $VAL_YEAR --test-year $TEST_YEAR --device $DEVICE --threads $THREADS --leads $LEADS --ic-stride $IC_STRIDE"
R="python scripts/run_kai_tendency.py $D --depths $KAI_DEPTHS --dims $KAI_DIMS"
K="python scripts/diag_island_kalman.py $D --depths $KAI_DEPTHS --dims $KAI_DIMS --n1 $ISLAND --lib-stride $LIB_STRIDE --chunk $CHUNK"
CT="--arm core --core-norm gn --core-gfeat --core-gdycore ${DYC_GRID:+--core-dycore-grid $DYC_GRID} --core-mem 8 --core-mz"
CT="$CT --core-substeps $SUBSTEPS --core-ckpt --core-blocks $BLOCKS --core-hidden $HIDDEN --log-every $LOGC --resume"
MT=${MEAS//_/}                                       # the measurement's tag in the file names (kai_ro -> kairo)
TAB="$OUT/kal_tab_${C1}_${ISLAND}_${VAL_YEAR}_s${LIB_STRIDE}_d1234o_${MT}.npz"
KAL="--kalman $MEAS --gain corr --amps 0 --kal-days 1 2 3 4 --kal-open"
B2="isn-${C1}+${C2}-Kcorrd1234o${MT}-err0-m1"        # island + second core, day 1-4 corrected toward KAI (output)
B3="isn-${C1}+${C2}-Kcorrd1234o${MT}+incfull-err0-m1"  # the same with the matrix island
D2="fb10-${C1}+${C2}-${MT}"; D3="fb10-${C1}+${C2}inc-${MT}"; S="fb10-${C1}-${MT}"
fc() { echo "$OUT/fc_$1_$2.npy"; }

# 1. the first core
step "$C1" "$OUT/$C1.pt" $R $CT --tag "$C1" --iters "$C1_ITERS" --batch "$C1_BATCH" --accum "$C1_ACCUM" --core-lr "$C1_LR"
stop_after 1

# 2. the Kalman table (validation year) and the first core alone, both years (also KAI's forecasts of every lead)
step table "$(fc $C1 $VAL_YEAR)" $K --tag "$C1" $KAL --year "$VAL_YEAR" --kai-fc --no-runs \
    --json "$OUT/island_${C1}_${VAL_YEAR}.json"
step core1_test "$(fc $C1 $TEST_YEAR)" $K --tag "$C1" $KAL --year "$TEST_YEAR" --kai-fc --no-runs \
    --json "$OUT/island_${C1}_${TEST_YEAR}.json"
stop_after 2

# 3. the second core behind the frozen first core and the fixed Kalman island (memory handed over at the island)
step "$C2" "$OUT/$C2.pt" $R $CT --init "$C1" --joint "$C1" --joint-freeze --joint-kalman "$MEAS" --kal-fixed \
    --kal-table "$TAB" --tag "${C1}_f" --joint-tag "$C2" --iters "$C2_ITERS" --batch "$C2_BATCH" --accum "$C2_ACCUM" \
    --curriculum $C2_CURRICULUM --core-lr "$C2_LR"
stop_after 3

# 4. the island run (B2), both years
for y in $VAL_YEAR $TEST_YEAR; do
    step "B2_$y" "$(fc "$B2" $y)" $K --tag "$C1" --stage2 "$C2" $KAL --year "$y" --no-central \
        --json "$OUT/island_B2_$y.json"
done
stop_after 4

# 5. the matrix island from the validation year's cases, then the island run with it (B3), both years
step matrix "$OUT/isl_inc_full_$TEST_YEAR.npy" python scripts/island_matrix_gain.py $D --core-fc "$C1" --meas "$MEAS" \
    --table "$TAB" --name full --fold "$FOLD" --json "$OUT/island_matrix_full.json"
step "B3_$VAL_YEAR" "$(fc "$B3" $VAL_YEAR)" $K --tag "$C1" --stage2 "$C2" $KAL --year "$VAL_YEAR" --no-central \
    --isl-inc "$OUT/isl_inc_full_${VAL_YEAR}cv.npy" --json "$OUT/island_B3_$VAL_YEAR.json"
step "B3_$TEST_YEAR" "$(fc "$B3" $TEST_YEAR)" $K --tag "$C1" --stage2 "$C2" $KAL --year "$TEST_YEAR" --no-central \
    --isl-inc "$OUT/isl_inc_full_$TEST_YEAR.npy" --json "$OUT/island_B3_$TEST_YEAR.json"
stop_after 5

# 6. the late KAI correction of every lead (gains of the validation year): D2, D3 and the simple blend S
FB="python scripts/make_full_blend.py $D --meas $MEAS"
step D2 "$(fc "$D2" $TEST_YEAR)" $FB --base "$B2" --name "$D2"
step D3 "$(fc "$D3" $TEST_YEAR)" $FB --base "$B3" --name "$D3"
step S "$(fc "$S" $TEST_YEAR)" $FB --base "$C1" --name "$S"
stop_after 6

# 7. scores: every model against the fine-tuned KAI at equal smoothing, D3 against D2 and against the simple blend
F="python scripts/diag_stage_filter_light.py $D"
step score "$OUT/score_island_vs_kai.json" $F --arms "$D3" "$D2" "$S" "$B2" "$C1" --ctrl "$MEAS" --fold "$FOLD" \
    --json "$OUT/score_island_vs_kai.json"
step score_d3 "$OUT/score_island_d3.json" $F --arms "$D3" --ctrl "$D2" --fold "$FOLD" --json "$OUT/score_island_d3.json"
step score_s "$OUT/score_island_vs_s.json" $F --arms "$D3" "$D2" --ctrl "$S" --fold "$FOLD" --json "$OUT/score_island_vs_s.json"
if [ "${DRYRUN:-0}" != 1 ]; then
    python scripts/island_plus_table.py --json "$OUT/score_island_vs_kai.json" --ctrl-label "KAI (fine-tuned, $MEAS)" \
        --title "Kalman island + version, 1.5 deg, $NY training years, test $TEST_YEAR" \
        --rows "D3: island + matrix island + second core + late KAI correction (best)=$D3" \
               "D2: island + second core + late KAI correction=$D2" "S: first core + KAI at every lead (simple blend)=$S" \
               "B2: island + second core, no late correction=$B2" "first core alone=$C1" \
        --gains "D3 against D2 at equal smoothing (the matrix island's share)=$OUT/score_island_d3.json" \
                "D3 and D2 against the simple blend S at equal smoothing (the island structure's share)=$OUT/score_island_vs_s.json" \
        --md "$OUT/table_island_plus.md" > "$OUT/table_island_plus.log" 2>&1 || echo "[island+] table failed, see $OUT/table_island_plus.log"
    python scripts/plot_filter_rows.py --out "$OUT" --ref "fine-tuned KAI ($MEAS)=score_island_vs_kai.json" \
        --series "D3 (best: matrix island)=score_island_vs_kai.json:$D3" "D2 (scalar island)=score_island_vs_kai.json:$D2" \
                 "S (simple blend, control)=score_island_vs_kai.json:$S" "B2 (no late correction)=score_island_vs_kai.json:$B2" \
        --gain-title "nRMSE gain at equal smoothing vs the fine-tuned KAI, % (+ = better)" \
        --title "Kalman island + version, 1.5 deg, $NY training years, test $TEST_YEAR ($IC_STRIDE-day start dates)" \
        --png fig_island_plus.png > "$OUT/fig_island_plus.log" 2>&1 || echo "[island+] figure failed, see $OUT/fig_island_plus.log"
    echo "[island+] done: $OUT/table_island_plus.md, $OUT/fig_island_plus.png, $OUT/score_island_*.json"
    grep -A7 "^## nRMSE" "$OUT/table_island_plus.md" || true
fi
