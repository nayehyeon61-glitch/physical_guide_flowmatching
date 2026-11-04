#!/bin/bash
# step 17 shallow-water test bed (owner 2026-10-27 12:35): coarse setting (T42 truth, T21 learners: closure
# missing) then full (T21 truth = learners' resolution). One thread, beside queue d.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
for S in coarse full; do
    python scripts/run_step17_swe.py --out runs/step17 --setting $S --threads 1 > runs/step17/$S.log 2>&1
done
python scripts/plot_step17.py --runs runs/step17/coarse.json runs/step17/full.json --out docs/step17/fig_step17.png \
    > runs/step17/plot.log 2>&1
echo QUEUE17_DONE > runs/step17/queue17.done
