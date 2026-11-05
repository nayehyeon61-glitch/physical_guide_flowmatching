#!/usr/bin/env bash
# blend_band+ on the CPU at idle priority, detached from any session: the smoke first, the full run only when it passes.
# Threads are set from the cores idle at each start and never above THREADS_MAX. The core training stops every
# MAX_HOURS (exit 75) and resumes from its checkpoint, so the thread count follows the load. Run again after a reboot.
#   setsid nohup bash scripts/run_plus_cpu.sh > logs/plus_main.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv-gaia/bin/activate
THREADS_MAX=${THREADS_MAX:-12}; MAX_HOURS=${MAX_HOURS:-24}
idle() { chrt -i 0 nice -n 19 ionice -c3 "$@"; }
others_cores() {  # cores in use at normal priority (ours all run at idle priority, counted under ni)
    LC_ALL=C top -bn2 -d5 | awk -v n="$(nproc)" '/^%Cpu/{l=$0} END{split(l,c,/[:,]/); for(i in c){split(c[i],d," "); if(d[2]=="id")id=d[1]; if(d[2]=="ni")ni=d[1]} v=(100-id-ni)*n/100; printf "%d", v+0.999}'
}
own_jobs() {  # cores kept for our other jobs
    local r=0
    pgrep -u "$USER" -f fetch_wb2_2p8.sh > /dev/null && r=$((r+4))
    pgrep -u "$USER" -f 'OUT=runs/sr28_smoke|run_blend_sr.sh' > /dev/null && r=$((r+4))
    echo "$r"
}
pick_threads() {
    local t
    while :; do
        t=$(( $(nproc) - $(others_cores) - $(own_jobs) ))
        [ "$t" -gt "$THREADS_MAX" ] && t=$THREADS_MAX
        [ "$t" -ge 2 ] && { echo "$t"; return; }
        echo "[cpu] fewer than 2 idle cores, waiting $(date '+%m-%d %H:%M')" >&2; sleep 600
    done
}
if [ "${SKIP_SMOKE:-0}" != 1 ]; then
    SMOKE=1 BASE=runs/blend_ngcm OUT=runs/blend_plus_smoke DATA=data/era5_5p625 DEVICE=cpu THREADS=4 AMP=0 \
        idle bash scripts/run_blend_plus.sh > logs/plus_smoke.log 2>&1
    rc=$?; echo "[cpu] PLUS_SMOKE_EXIT=$rc $(date '+%m-%d %H:%M')"
    [ "$rc" = 0 ] || { echo "[cpu] smoke failed, the full run is not started; see logs/plus_smoke.log"; exit "$rc"; }
fi
while :; do
    T=$(pick_threads); echo "[cpu] full run: $T threads, start $(date '+%m-%d %H:%M')"
    OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T BASE=runs/blend_ngcm OUT=runs/blend_plus \
        DATA=data/era5_5p625 DEVICE=cpu THREADS=$T AMP=0 MAX_HOURS=$MAX_HOURS idle bash scripts/run_blend_plus.sh
    rc=$?; echo "[cpu] run exit $rc $(date '+%m-%d %H:%M')"
    [ "$rc" = 75 ] || exit "$rc"
done
