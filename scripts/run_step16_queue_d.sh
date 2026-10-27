#!/bin/bash
# step 16 residual flow (owner 2026-10-27 07:25): the sequence FM learns d - Delta_core around the data-only
# core's 6 h rollout, members = x0 + Delta_core + residual -> frozen stage-1 PINN -> band Gamma (mbm_band: on
# its own mean; regm_band: on the step 13 regression mean, isolating the deviation structure) -> scores
# against the regression ensemble, the step 13 FM and its regm_band. After queue c (one 4-thread job at a time).
cd "$(dirname "$0")/.."
while [ ! -f runs/step16/queue16c.done ]; do sleep 30; done
K=/home/claude/era5/kai6h; E=/home/claude/era5/era5_data
python scripts/run_step13_seqfm.py --kai $K --era5 $E --out runs/step13 --core-mean runs/step16/dyn_lp0 \
    --pinn runs/step13/pinn_project.pt --iters 3000 --batch 4 --val-every 500 --n-val-train 8 --val-members 2 \
    --save-ens > runs/step16/resfm.log 2>&1
FM=runs/step13/seqfm_dct_core_pinn
python scripts/run_step13_guide.py --ens $FM --era5 $E --out runs/step13 --reg runs/step13/seqregress_pinn --reg-mbm \
    --save-arms mbm_band regm_band > runs/step16/resfm_guide.log 2>&1
python scripts/score_step13.py --era5 $E --out runs/step16/score_resfm.json --ref runs/step13/seqregress_pinn \
    --arms $FM ${FM}_mbm_band ${FM}_regm_band runs/step13/seqfm_dct_pinn runs/step13/seqfm_dct_pinn_regm_band \
    > runs/step16/score_resfm.log 2>&1
python scripts/diag_step13_flowdep.py --runs runs/step13 --arms seqfm_dct_core_pinn seqfm_dct_core_pinn_regm_band \
    seqfm_dct_pinn seqfm_dct_pinn_regm_band --leads 24 72 144 --out runs/step16/flowdep_resfm.json \
    > runs/step16/flowdep_resfm.log 2>&1
echo QUEUE16D_DONE > runs/step16/queue16d.done
