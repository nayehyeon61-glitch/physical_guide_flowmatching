#!/bin/bash
# step 26, CPU experiment B: speed and peak memory of the wide core (128 x 8 blocks) against the step 18 core
# (48 x 4 blocks), with and without gradient checkpointing; 6 iterations, batch 4, K = 8 (no curriculum), 1 year.
set -e
cd "$(dirname "$0")/../.."
G=runs/step26/timing
mkdir -p "$G"
A="--kai /home/claude/era5/kai6h --era5 /home/claude/era5/era5_data --out $G --gfeat --lambda-p 0 --train-years 2015"
A="$A --val-year 2016 --iters 6 --batch 4 --K 8 --curriculum 8 --n-val 4 --n-val-train 4 --val-K 8 --val-every 0 --threads 4"
for c in "base 48 2 2" "base_ck 48 2 2 --grad-ckpt" "plus 128 2 2 2 2" "plus_ck 128 2 2 2 2 --grad-ckpt"; do
    set -- $c
    tag=$1; h=$2; shift 2
    blocks=""; while [ $# -gt 0 ] && [ "$1" != "--grad-ckpt" ]; do blocks="$blocks $1"; shift; done
    [ -f "$G/$tag.json" ] || python runs/step26/maxrss.py scripts/run_step16_dynpinn.py $A --hidden "$h" --blocks $blocks \
        --tag "$tag" "$@" > "$G/$tag.log" 2>&1
    echo "$tag $(grep -E '^  it +(1|6) |maxrss' "$G/$tag.log" | tr '\n' ' ')"
done
echo "QUEUE26B_DONE $(date +%H:%M)"
