#!/bin/bash
# step 17 full42, remaining arms (the first run was OOM-killed in the node arm while queue d sampled
# ensembles): waits for queue d, then resumes node / pi / pi_only on the saved data arm.
cd "$(dirname "$0")/.."
while [ ! -f runs/step16/queue16d.done ]; do sleep 60; done
export OMP_NUM_THREADS=1
python scripts/run_step17_swe.py --out runs/step17 --setting full --res 42 --hidden 16 --iters 1000 \
    --arms node pi pi_only --resume --threads 2 > runs/step17/full42_b.log 2>&1
python scripts/plot_step17.py --runs runs/step17/coarse.json runs/step17/full42.json runs/step17/full.json \
    --out docs/step17/fig_step17.png > runs/step17/plot.log 2>&1
echo QUEUE17B2_DONE > runs/step17/queue17b2.done
