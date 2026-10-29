#!/bin/bash
# step 20 (b): energy scores (after the leap-year fix), then the attribution control: the same blend with the
# data-only core dyn_lp0 instead of the G(x)-feature core dyn_gfeat, scored against regm_band. 2 threads.
cd "$(dirname "$0")/.."
G=runs/step20
[ -f docs/step20/energy_score.json ] || \
    python scripts/step20_energy_score.py --era5 /home/claude/era5/era5_data --out docs/step20/energy_score.json --threads 2 \
        --arms runs/step13/seqfm_dct_raw runs/step13/seqfm_dct_pinn runs/step13/seqfm_dct_core_raw runs/step13/seqfm_dct_core_hyd \
        runs/step13/seqfm_dct_pinn_regm_band runs/step13/seqfm_dct_pinn_regm_sband_clim_flow "$G/seqfm_dct_pinn_blend_band" \
        > "$G/energy.log" 2>&1
C=$G/ctrl_lp0
[ -f "$C/seqfm_dct_pinn_blend_band_ens_2018.npz" ] || \
    python scripts/run_step13_guide.py --ens runs/step13/seqfm_dct_pinn --era5 /home/claude/era5/era5_data --out "$C" \
        --core runs/step16/dyn_lp0 --reg runs/step13/seqregress_pinn --reg-mbm --blend \
        --save-arms blend_band --threads 2 > "$C/guide.log" 2>&1
[ -f "$C/score.json" ] || \
    python scripts/score_step13.py --era5 /home/claude/era5/era5_data --out "$C/score.json" --ref runs/step13/seqfm_dct_pinn_regm_band \
        --arms runs/step13/seqfm_dct_pinn_regm_band "$C/seqfm_dct_pinn_blend_band" "$G/seqfm_dct_pinn_blend_band" > "$C/score.log" 2>&1
echo QUEUE20B_DONE > "$G/queue20b.done"
