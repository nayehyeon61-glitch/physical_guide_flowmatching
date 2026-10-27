#!/bin/bash
# step 16 follow-up: waits for queue16a, then the tangent-spectrum diagnostic and the tangent-loss arms
cd "$(dirname "$0")/.."
while [ ! -f runs/step16/queue16a.done ]; do sleep 30; done
python scripts/step16_tangent_spectrum.py --kai /home/claude/era5/kai6h --era5 /home/claude/era5/era5_data --stats runs/step16/stats --out docs/step16 > runs/step16/tangent_spectrum.log 2>&1
COMMON="--kai /home/claude/era5/kai6h --era5 /home/claude/era5/era5_data --out runs/step16 --train-years 2012 2013 2014 2015 --val-year 2016 --K 8 --curriculum 1 2 4 8 --iters 4000 --batch 4 --val-every 500 --n-val-train 8 --n-val 16 --val-K 24 --val-chunk 4"
python scripts/run_step16_dynpinn.py $COMMON --lambda-p 0 --tangent 1 --tag dyn_lp0_tan1 > runs/step16/dyn_lp0_tan1.log 2>&1
python scripts/run_step16_dynpinn.py $COMMON --lambda-p 0.05 --tangent 1 --tag dyn_lp0.05_tan1 > runs/step16/dyn_lp0.05_tan1.log 2>&1
echo QUEUE16B_DONE > runs/step16/queue16b.done
