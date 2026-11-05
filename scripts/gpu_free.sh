#!/usr/bin/env bash
# Run a command only on a GPU that nobody is using. A GPU counts as free when it has no compute process, utilization
# below GPU_MAX_UTIL % and memory use below GPU_MAX_MEM MiB. An occupied GPU is never used.
#
#   bash scripts/gpu_free.sh bash scripts/run_blend_sr.sh     # exit 3 at once when no GPU is free
#   GPU_WAIT=1 bash scripts/gpu_free.sh <command>             # wait, re-checking every GPU_POLL seconds
#   GPU_ID=2 bash scripts/gpu_free.sh <command>               # only this GPU, still checked
#   GPU_LIST=1 bash scripts/gpu_free.sh                       # print each GPU's state and stop
set -uo pipefail
MAX_UTIL=${GPU_MAX_UTIL:-5}; MAX_MEM=${GPU_MAX_MEM:-500}; POLL=${GPU_POLL:-300}; ONLY=${GPU_ID:-}
command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1 || { echo "[gpu] no working GPU on $(hostname)"; exit 3; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
state() {  # index uuid util mem nproc per line
    local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)
    nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',' |
    while read -r i u g m; do echo "$i $u $g $m $(grep -c "$u" <<< "$apps")"; done
}
pick() {
    state | while read -r i _ g m n; do
        [ -n "$ONLY" ] && [ "$i" != "$ONLY" ] && continue
        [ "$n" = 0 ] && [ "$g" -lt "$MAX_UTIL" ] && [ "$m" -lt "$MAX_MEM" ] && { echo "$i"; break; }
    done
}
if [ "${GPU_LIST:-0}" = 1 ]; then echo "index uuid util% memMiB procs"; state; exit 0; fi
while :; do
    G=$(pick)
    [ -n "$G" ] && break
    [ "${GPU_WAIT:-0}" = 1 ] || { echo "[gpu] no free GPU on $(hostname), not starting:"; state; exit 3; }
    echo "[gpu] no free GPU, waiting $(date '+%m-%d %H:%M')"; sleep "$POLL"
done
echo "[gpu] using free GPU $G on $(hostname) $(date '+%m-%d %H:%M')"
export CUDA_VISIBLE_DEVICES=$G
exec "$@"
