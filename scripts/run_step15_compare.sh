#!/usr/bin/env bash
# Step 15 comparison on a GPU machine (docs/step15_training_system.md 7): the same chain under
#   climode     ClimODE's data years (train 2006-2015), the broad training settings
#   broad_long  train 1979-2015, FM / regression 120000 it, validation every 1000 steps, patience 10
# then the json report of each run and the side-by-side comparison with the owner's 2026-10-27 broad run.
#
#   DATA=/path/era5_5p625 bash scripts/run_step15_compare.sh
#   BROAD=runs/step13_broad   the earlier broad run to include (skipped when it has no score_best.json)
#   ONLY="climode"            run only these profiles (default "climode broad_long")
#   MAX_HOURS=11              per trainer invocation; on exit 75 rerun this script, it resumes
# Disk: memory-mapped years ~8 GB (climode) and ~30 GB (broad_long), each arm-year ensemble ~228 MB.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data/era5_5p625}
BROAD=${BROAD:-runs/step13_broad}
ONLY=${ONLY:-"climode broad_long"}
OUT_ROOT=${OUT_ROOT:-runs}
for prof in $ONLY; do
    echo "[compare] profile $prof -> $OUT_ROOT/step15_$prof"
    rc=0
    PROFILE=$prof DATA=$DATA OUT=$OUT_ROOT/step15_$prof bash scripts/run_step13_best.sh || rc=$?
    if [ "$rc" = 75 ]; then echo "[compare] $prof: budget reached, rerun this script to resume"; exit 75; fi
    [ "$rc" = 0 ] || { echo "[compare] $prof failed ($rc)"; exit 1; }
    python scripts/report_step13_broad.py --runs "$OUT_ROOT/step15_$prof" --out "docs/step15/$prof"
done
RUNS=""
for prof in $ONLY; do RUNS="$RUNS $prof=$OUT_ROOT/step15_$prof"; done
if [ -e "$BROAD/score_best.json" ]; then RUNS="${RUNS% *} broad=$BROAD ${RUNS##* }"; fi  # order: climode, broad, broad_long
python scripts/compare_step15_runs.py --runs $RUNS --era5 "$DATA/era5_data" --out docs/step15
echo "[compare] done: docs/step15/compare.md, fig_compare.png, fig_compare_training.png, docs/step15/<profile>/step13_broad_report.pdf"
