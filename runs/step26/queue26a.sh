#!/bin/bash
# step 26 (blend_band+), CPU experiment A: does a blend of several cores beat the one-core blend_band?
# 4-year inputs of step 20 (FM + PINN members, regression mean of runs/step13, cores of runs/step16, all K = 8):
#   blend1 = a + b gfeat + c reg + d x0 (step 20's blend_band, reproduced as the check of the --core-extra code)
#   blend2 = + data-only core dyn_lp0          blend4 = + dyn_lp0 + dyn_balA + dyn_balB
# The operator / learned guide epochs are cut to 1: blend_band does not use them.
set -e
cd "$(dirname "$0")/../.."
G=runs/step26
E=/home/claude/era5/era5_data
C=runs/step16
guide() {  # guide <name> <extra cores...>
    local name=$1; shift
    mkdir -p "$G/$name"
    [ -f "$G/$name/seqfm_dct_pinn_blend_band_ens_2018.npz" ] || \
        python scripts/run_step13_guide.py --ens runs/step13/seqfm_dct_pinn --era5 "$E" --out "$G/$name" \
            --core "$C/dyn_gfeat" --core-extra "$@" --reg runs/step13/seqregress_pinn --reg-mbm --blend \
            --save-arms blend_band --epochs 1 --guide-epochs 1 --threads 4 > "$G/$name/guide.log" 2>&1
    for y in 2017 2018; do ln -sfn "$name/seqfm_dct_pinn_blend_band_ens_$y.npz" "$G/${name}_ens_$y.npz"; done
}
echo "start $(date +%H:%M)"
guide blend1
echo "blend1 $(date +%H:%M)"
guide blend2 "$C/dyn_lp0"
echo "blend2 $(date +%H:%M)"
guide blend4 "$C/dyn_lp0" "$C/dyn_balA" "$C/dyn_balB"
echo "blend4 $(date +%H:%M)"
[ -f "$G/score_a.json" ] || python scripts/score_step13.py --era5 "$E" --out "$G/score_a.json" --ref "$G/blend1" \
    --arms "$G/blend1" "$G/blend2" "$G/blend4" > "$G/score_a.log" 2>&1
echo "QUEUE26A_DONE $(date +%H:%M)" > "$G/queue26a.done"
