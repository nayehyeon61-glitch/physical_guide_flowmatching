#!/bin/bash
# full-year 2020 check (180 ClimODE-convention ICs, never used for any choice): frozen step 13/16 weights,
# 2016 constants, members from runs/step23/queue_2020.sh
cd "$(dirname "$0")/../.."
python scripts/score_step13.py --era5 /home/claude/era5/era5_data --out runs/step25/score_2020.json \
    --ref runs/step23/seqregress_pinn --test-years 2020 \
    --arms runs/step23/seqregress_pinn runs/step23/seqfm_dct_pinn_regm_band runs/step23/seqfm_dct_pinn_blend_band
echo done > runs/step25/score_2020.done
