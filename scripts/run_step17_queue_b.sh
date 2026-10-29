#!/bin/bash
# step 17, third setting: full42 = truth and learners both at T42 (128 x 256), so the physics operator is exact
# AND the flow is the broken Galewsky jet of the coarse setting (at T21 the jet never breaks, docs/step17 3.2).
# Smaller net and fewer iterations (4x the pixels); no Raissi arm (its result does not depend on the grid).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
python scripts/run_step17_swe.py --out runs/step17 --setting full --res 42 --hidden 16 --iters 1000 \
    --arms data node pi pi_only --threads 1 > runs/step17/full42.log 2>&1
python scripts/plot_step17.py --runs runs/step17/coarse.json runs/step17/full42.json runs/step17/full.json \
    --out docs/step17/fig_step17.png > runs/step17/plot.log 2>&1
echo QUEUE17B_DONE > runs/step17/queue17b.done
