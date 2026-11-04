#!/bin/bash
# step 21: SR / downscaling as a forecast corrector (scale 2) and the 5.625 deg control (scale 1), then the figure. 2 threads.
cd "$(dirname "$0")/.."
O=runs/step21
[ -f "$O/corr_s2.json" ] || python scripts/run_step21_sr_correct.py --scale 2 --out "$O" --fig docs/step21 --threads 2 > "$O/corr_s2.log" 2>&1
[ -f "$O/corr_s1.json" ] || python scripts/run_step21_sr_correct.py --scale 1 --out "$O" --fig docs/step21 --threads 2 > "$O/corr_s1.log" 2>&1
python scripts/run_step21_sr_correct.py --plot-only --out "$O" --fig docs/step21 > "$O/plot.log" 2>&1
echo QUEUE21_DONE > "$O/queue21.done"
