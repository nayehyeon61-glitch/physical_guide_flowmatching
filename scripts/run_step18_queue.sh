#!/bin/bash
# step 18: the operator-tendency feature arm of the data-only core (G(x) = -R0(x) as 48 input channels),
# trained as step 16's dyn_lp0 (same seed, same validation ICs), then its ODE trajectories at ClimODE's ICs
# (run_step16_predict) and the core_band slot of the step 13 guide, scored as step 16 queue c. 2 threads
# (the step 17 hybrid queue holds the other 2).
cd "$(dirname "$0")/.."
D="--kai /home/claude/era5/kai6h --era5 /home/claude/era5/era5_data"
COMMON="$D --out runs/step16 --train-years 2012 2013 2014 2015 --val-year 2016 --K 8 --curriculum 1 2 4 8 --iters 4000 --batch 4 --val-every 500 --n-val-train 8 --n-val 16 --val-K 24 --val-chunk 4 --threads 2"
[ -f runs/step16/dyn_gfeat.json ] || \
    python scripts/run_step16_dynpinn.py $COMMON --lambda-p 0 --gfeat --tag dyn_gfeat > runs/step16/dyn_gfeat.log 2>&1
[ -f runs/step16/dyn_gfeat_pred.npz ] || \
    python scripts/run_step16_predict.py $D --run runs/step16 --tag dyn_gfeat --threads 2 > runs/step16/predict_dyn_gfeat.log 2>&1
G="runs/step16/g_dyn_gfeat"; mkdir -p "$G"
[ -f "$G/score.json" ] || {
    python scripts/run_step13_guide.py --ens runs/step13/seqfm_dct_pinn --era5 /home/claude/era5/era5_data --out "$G" \
        --core runs/step16/dyn_gfeat --save-arms core_band > "$G/guide.log" 2>&1
    python scripts/score_step13.py --era5 /home/claude/era5/era5_data --out "$G/score.json" --ref runs/step13/seqregress_pinn \
        --arms runs/step13/seqregress_pinn "$G/seqfm_dct_pinn_core_band" runs/step13/seqfm_dct_pinn_regm_band \
        > "$G/score.log" 2>&1 && rm -f "$G"/seqfm_dct_pinn_core_band_ens_*.npz
}
echo QUEUE18_DONE > runs/step18/queue18.done
