#!/bin/bash
# step 17 full42, remaining arms: the first run was OOM-killed in the node arm at 17:17 UTC together with the
# queue d sampling (9.3 GB + 4.7 GB > the 14.3 GB cgroup). Runs alone now; queue d2 waits for its done file.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
python scripts/run_step17_swe.py --out runs/step17 --setting full --res 42 --hidden 16 --iters 1000 \
    --arms node pi pi_only --resume --threads 2 > runs/step17/full42_b.log 2>&1
python scripts/plot_step17.py --runs runs/step17/coarse.json runs/step17/full42.json runs/step17/full.json \
    --out docs/step17/fig_step17.png > runs/step17/plot.log 2>&1
echo QUEUE17B2_DONE > runs/step17/queue17b2.done
