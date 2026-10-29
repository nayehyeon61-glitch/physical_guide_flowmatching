#!/bin/bash
# step 19: SR output stage of the step 13 reported ensemble (regm_sband_clim_flow), 2 threads
cd "$(dirname "$0")/.."
python scripts/run_step19_sr.py --coarse /home/claude/era5/era5_data --fine /home/claude/era5_hres/era5_data \
    --ens runs/step13/seqfm_dct_pinn_regm_sband_clim_flow --out runs/step19 --fig docs/step19 --tag sr --threads 2 \
    > runs/step19/sr.log 2>&1
echo QUEUE19_DONE > runs/step19/queue19.done
