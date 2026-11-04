#!/bin/bash
# step 16 reduced CPU test: three arms, serial (one 4-thread torch job at a time)
cd "$(dirname "$0")/.."
COMMON="--kai /home/claude/era5/kai6h --era5 /home/claude/era5/era5_data --out runs/step16 --train-years 2012 2013 2014 2015 --val-year 2016 --K 8 --curriculum 1 2 4 8 --iters 4000 --batch 4 --val-every 500 --n-val-train 8 --n-val 16 --val-K 24 --val-chunk 4"
python scripts/run_step16_dynpinn.py $COMMON --lambda-p 0 --tag dyn_lp0 > runs/step16/dyn_lp0.log 2>&1
python scripts/run_step16_dynpinn.py $COMMON --lambda-p 0.05 --tag dyn_lp0.05 > runs/step16/dyn_lp0.05.log 2>&1
python scripts/run_step16_dynpinn.py $COMMON --lambda-p 0.5 --phys-target 0 --tag dyn_lp0.5_t0 > runs/step16/dyn_lp0.5_t0.log 2>&1
echo QUEUE16A_DONE > runs/step16/queue16a.done
