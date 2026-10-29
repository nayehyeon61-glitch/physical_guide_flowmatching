#!/bin/bash
# Step 24: data-efficiency curve. The step 13 chain (PINN, FM, regression, regm_band) retrained with 1 and 2
# training years (2015; 2014-2015), same iterations as the 4-year small profile, scored on the 360 ICs.
# 4 years = runs/step13 (regm_band z500 244/522/778), 37 years = owner's GPU run (170/423/741).
# Restarted 2026-10-28 after a container restart: run_step13_best.sh skips finished steps and reuses weights.
cd /home/claude/physical_guide_flowmatching
for Y in "2015" "2014 2015"; do
  N=y$(echo $Y | wc -w)
  echo "[$(date -u +%H:%M)] train years $Y -> runs/step24/$N"
  TRAIN_YEARS="$Y" OUT=runs/step24/$N DATA=/home/claude/era5 SIGMA=0 SBAND=0 CLIMODE=0 BLEND=0 THREADS=4 \
    bash scripts/run_step13_best.sh >> runs/step24/$N.log 2>&1 || { echo "FAILED $N"; exit 1; }
  # scored: drop the member files (scores are in score_best.json), keep the weights and json
  if [ "$N" = y1 ]; then
    rm -f /home/claude/physical_guide_flowmatching/runs/step24/y1/seqfm_dct_pinn_ens_2016.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y1/seqfm_dct_pinn_ens_2017.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y1/seqfm_dct_pinn_ens_2018.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y1/seqfm_dct_raw_ens_2017.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y1/seqfm_dct_raw_ens_2018.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y1/seqregress_pinn_ens_2016.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y1/seqregress_pinn_ens_2017.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y1/seqregress_pinn_ens_2018.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y1/seqregress_raw_ens_2017.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y1/seqregress_raw_ens_2018.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y1/seqfm_dct_pinn_regm_band_ens_2017.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y1/seqfm_dct_pinn_regm_band_ens_2018.npz
  else
    rm -f /home/claude/physical_guide_flowmatching/runs/step24/y2/seqfm_dct_pinn_ens_2016.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y2/seqfm_dct_pinn_ens_2017.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y2/seqfm_dct_pinn_ens_2018.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y2/seqfm_dct_raw_ens_2017.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y2/seqfm_dct_raw_ens_2018.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y2/seqregress_pinn_ens_2016.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y2/seqregress_pinn_ens_2017.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y2/seqregress_pinn_ens_2018.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y2/seqregress_raw_ens_2017.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y2/seqregress_raw_ens_2018.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y2/seqfm_dct_pinn_regm_band_ens_2017.npz \
          /home/claude/physical_guide_flowmatching/runs/step24/y2/seqfm_dct_pinn_regm_band_ens_2018.npz
  fi
  ls runs/step24/$N/*_ens_*.npz 2>/dev/null | sed 's/^/left: /'
done
echo "[$(date -u +%H:%M)] done"; touch runs/step24/queue_years.done
