#!/bin/bash
# step 16 two-head PINN arms (A: rollout gradient reaches c, B: detached) after queue b, then the ODE
# trajectories of each core at ClimODE's ICs and the core_band arm of the step 13 guide (the core's mean with
# the FM members' deviations), scored against the regression ensemble and regm_band
cd "$(dirname "$0")/.."
while [ ! -f runs/step16/queue16b.done ]; do sleep 30; done
D="--kai /home/claude/era5/kai6h --era5 /home/claude/era5/era5_data"
COMMON="$D --out runs/step16 --train-years 2012 2013 2014 2015 --val-year 2016 --K 8 --curriculum 1 2 4 8 --iters 4000 --batch 4 --val-every 500 --n-val-train 8 --n-val 16 --val-K 24 --val-chunk 4"
mean_slot() {  # $1 = core tag: ODE trajectories at ClimODE's ICs -> guide core_band -> score -> delete the npz
    [ -f "runs/step16/$1_pred.npz" ] || \
        python scripts/run_step16_predict.py $D --run runs/step16 --tag "$1" > "runs/step16/predict_$1.log" 2>&1
    G="runs/step16/g_$1"; mkdir -p "$G"
    [ -f "$G/seqfm_dct_pinn_core_band_ens_2018.npz" ] || \
        python scripts/run_step13_guide.py --ens runs/step13/seqfm_dct_pinn --era5 /home/claude/era5/era5_data --out "$G" \
            --core "runs/step16/$1" --save-arms core_band > "$G/guide.log" 2>&1
    python scripts/score_step13.py --era5 /home/claude/era5/era5_data --out "$G/score.json" --ref runs/step13/seqregress_pinn \
        --arms runs/step13/seqregress_pinn "$G/seqfm_dct_pinn_core_band" runs/step13/seqfm_dct_pinn_regm_band \
        > "$G/score.log" 2>&1 && rm -f "$G"/seqfm_dct_pinn_core_band_ens_*.npz
}
mean_slot dyn_lp0
[ -f runs/step16/dyn_balA.json ] || python scripts/run_step16_dynpinn.py $COMMON --lambda-p 0 --balance --lambda-hyd 0.1 \
    --tag dyn_balA > runs/step16/dyn_balA.log 2>&1
[ -f runs/step16/dyn_balB.json ] || python scripts/run_step16_dynpinn.py $COMMON --lambda-p 0 --balance --bal-detach \
    --lambda-hyd 0.1 --tag dyn_balB > runs/step16/dyn_balB.log 2>&1
mean_slot dyn_balA
mean_slot dyn_balB
echo QUEUE16C_DONE > runs/step16/queue16c.done
