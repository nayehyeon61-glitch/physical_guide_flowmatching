#!/bin/bash
# step 17 full42 phys + hybrid rerun with 15 min RK4 substeps (30 min is unstable at T42: gravity-wave
# CFL; the queue_c arms went NaN after 1 h), then replot. 2 threads beside the step 18 queue.
cd "$(dirname "$0")/.."
python scripts/run_step17_swe.py --out runs/step17 --setting full --res 42 --hidden 16 --iters 1000 \
    --arms phys hybrid --hybrid-substeps 4 --resume --threads 2 > runs/step17/full42_d.log 2>&1
python scripts/plot_step17.py --runs runs/step17/coarse.json runs/step17/full42.json runs/step17/full.json \
    --out docs/step17/fig_step17.png > runs/step17/plot_d.log 2>&1
echo QUEUE17D_DONE > runs/step17/queue17d.done
