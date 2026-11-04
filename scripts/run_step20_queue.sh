#!/bin/bash
# step 20: (2) lead-dependent blend mean (G(x) core + regression mean + x0) with FM deviations + band Gamma,
# scored as the reported model; (3) multivariate energy scores of the saved ensembles. 2 threads.
cd "$(dirname "$0")/.."
G=runs/step20
[ -f "$G/seqfm_dct_pinn_blend_band_ens_2018.npz" ] || \
    python scripts/run_step13_guide.py --ens runs/step13/seqfm_dct_pinn --era5 /home/claude/era5/era5_data --out "$G" \
        --core runs/step16/dyn_gfeat --reg runs/step13/seqregress_pinn --reg-mbm --blend \
        --save-arms blend_band regm_band --threads 2 > "$G/guide.log" 2>&1
[ -f "$G/score.json" ] || \
    python scripts/score_step13.py --era5 /home/claude/era5/era5_data --out "$G/score.json" --ref runs/step13/seqregress_pinn \
        --arms runs/step13/seqregress_pinn "$G/seqfm_dct_pinn_blend_band" "$G/seqfm_dct_pinn_regm_band" \
        runs/step13/seqfm_dct_pinn_regm_sband_clim_flow > "$G/score.log" 2>&1
python scripts/step20_energy_score.py --era5 /home/claude/era5/era5_data --out docs/step20/energy_score.json --threads 2 \
    --arms runs/step13/seqfm_dct_raw runs/step13/seqfm_dct_pinn runs/step13/seqfm_dct_core_raw runs/step13/seqfm_dct_core_hyd \
    runs/step13/seqfm_dct_pinn_regm_band runs/step13/seqfm_dct_pinn_regm_sband_clim_flow "$G/seqfm_dct_pinn_blend_band" \
    > "$G/energy.log" 2>&1
rm -f "$G"/seqfm_dct_pinn_regm_band_ens_*.npz
echo QUEUE20_DONE > "$G/queue20.done"
