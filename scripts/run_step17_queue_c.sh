#!/bin/bash
# step 17 hybrid arms (owner 2026-10-27 23:23: physics as structure for the members): the learners' own
# equations integrated for one hour + a learned hourly closure (hybrid), and the equations alone (phys), added
# to the saved coarse (closure gap) and full42 (exact operator, broken jet) runs with --resume.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2
python scripts/run_step17_swe.py --out runs/step17 --setting coarse --arms phys hybrid --resume --threads 2 \
    > runs/step17/coarse_c.log 2>&1
python scripts/run_step17_swe.py --out runs/step17 --setting full --res 42 --hidden 16 --iters 1000 \
    --arms phys hybrid --resume --threads 2 > runs/step17/full42_c.log 2>&1
python scripts/plot_step17.py --runs runs/step17/coarse.json runs/step17/full42.json runs/step17/full.json \
    --out docs/step17/fig_step17.png > runs/step17/plot.log 2>&1
echo QUEUE17C_DONE > runs/step17/queue17c.done
