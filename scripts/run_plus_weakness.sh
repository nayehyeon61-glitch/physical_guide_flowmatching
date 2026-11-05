#!/usr/bin/env bash
# Step 27b: weakness checks of blend_band+ on the 2020 hold-out (docs/step27_recent_models.md, section 5).
# Run after scripts/run_blend_plus.sh has finished its hold-out part ($OUT/y2020/plus_blend_band_ens_2020.npz).
# Nothing is retrained: the frozen members, cores and blend of the plus run are re-blended or re-scored.
#
#   W1 short-lead accuracy (WeatherODE / PINN-Cast beat us at 24-72 h): every core alone vs the blend vs ClimODE,
#      paired on the 179 ICs -> is the 24 h error the mean (cores) or the blend?
#   W2 member realism (members carry 5-19x ERA5 power at wavenumbers 10-32): spectra, spread-skill, rank / Brier
#   W3 PINN contribution (0 so far): the same blend built on the members WITHOUT the PINN projection
#      (seqfm_dct_raw / seqregress_raw, saved by run_step13_seqfm.py next to the pinn ones), paired vs plus
#   W4 recent models: plus and the no-PINN control on the table / figure of step 27 (WB2 forecasts, ClimODE)
#
#   bash scripts/run_plus_weakness.sh                 # OUT=runs/blend_plus BASE=runs/blend_ngcm
#   DRYRUN=1 bash scripts/run_plus_weakness.sh        # print the commands only
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-runs/blend_ngcm}
OUT=${OUT:-runs/blend_plus}
DATA=${DATA:-data/era5_5p625}
HOLDOUT=${HOLDOUT:-2020}
VAL_YEAR=${VAL_YEAR:-2016}
THREADS=${THREADS:-8}
E=$DATA/era5_data
N=$OUT/y$HOLDOUT
H=$BASE/ngcm$HOLDOUT
W=$OUT/weak$HOLDOUT
mkdir -p "$W/nopinn"
step() {
    local name=$1; shift
    mkdir -p "$W/.done"
    if [ -e "$W/.done/$name" ]; then echo "[weak] $name: done, skipped"; return; fi
    if [ "${DRYRUN:-0}" = 1 ]; then echo "[dry] $name: $(printf '%q ' "$@")"; return; fi
    echo "[weak] $name: start $(date +%H:%M), log $W/$name.log"
    "$@" > "$W/$name.log" 2>&1 || { echo "[weak] $name failed, see $W/$name.log"; tail -n 20 "$W/$name.log"; exit 1; }
    touch "$W/.done/$name"; echo "[weak] $name: done $(date +%H:%M)"
}
need() { [ -e "$1" ] || [ "${DRYRUN:-0}" = 1 ] || { echo "[weak] missing $1 ($2)"; exit 1; }; }
need "$N/plus_blend_band_ens_$HOLDOUT.npz" "finish run_blend_plus.sh with HOLDOUT=$HOLDOUT first"
C2=$(ls "$H"/climode*/climode_pred_"$HOLDOUT"_valid.npz 2>/dev/null | head -n 1 || true)
need "$C2" "ClimODE $HOLDOUT forecasts: scripts/run_climode_2020.py, docs/results_2026_11/README.md section 6"
CORES="$N/plus_long"
for t in plus_short dyn_gfeat; do [ -e "$N/${t}_pred.npz" ] && CORES="$CORES $N/$t"; done
ARMS="$N/base_blend_band $N/plus_blend_band"

# W1: cores alone vs blend vs ClimODE (paired, 179 ICs)
step w1_paired python scripts/compare_step13_climode.py --era5 "$E" --climode "$C2" --ref-name "ClimODE (public checkpoint)" \
    --arms $ARMS --core $CORES --out "$W/w1_cores_paired.json"

# W3: the same plus blend on members without the PINN projection (raw FM / raw regression)
RAW_FM=seqfm_dct_raw; RAW_REG=seqregress_raw
for y in $VAL_YEAR $HOLDOUT; do
    for f in "${RAW_FM}_ens_$y.npz" "${RAW_REG}_ens_$y.npz"; do
        if [ -e "$H/$f" ]; then ln -sfn "$(realpath -m "$H/$f")" "$W/nopinn/$f"
        elif [ "${DRYRUN:-0}" != 1 ]; then
            echo "[weak] missing $H/$f: the base run did not save raw members; W3 skipped"; RAW_FM=""; fi
    done
done
ln -sfn "$(realpath -m "$N/stats")" "$W/nopinn/stats"
G="--era5 $E --reg-mbm --blend --val-year $VAL_YEAR --save-arms blend_band --epochs 1 --guide-epochs 1 --threads $THREADS"
EXTRA=""; for c in ${CORES#* }; do [ "$c" != "$N/plus_long" ] && EXTRA="$EXTRA $c"; done
if [ -n "$RAW_FM" ]; then
    step w3_guide_nopinn python scripts/run_step13_guide.py $G --ens "$W/nopinn/$RAW_FM" --reg "$W/nopinn/$RAW_REG" \
        --test-years $HOLDOUT --out "$W/nopinn" --core "$N/plus_long" ${EXTRA:+--core-extra $EXTRA}
    ln -sfn "nopinn/${RAW_FM}_blend_band_ens_$HOLDOUT.npz" "$W/nopinn_blend_band_ens_$HOLDOUT.npz"
    ARMS="$ARMS $W/nopinn_blend_band"
    step w3_score python scripts/score_step13.py --era5 "$E" --out "$W/w3_pinn_vs_nopinn.json" \
        --ref "$W/nopinn_blend_band" --arms $ARMS --test-years $HOLDOUT
    step w3_paired python scripts/compare_step13_climode.py --era5 "$E" --climode "$C2" \
        --ref-name "ClimODE (public checkpoint)" --arms $ARMS --out "$W/w34_paired.json"
fi

# W2: spectra, spread-skill, rank / PIT, tercile Brier on the same ICs
step w2_prob python scripts/step22_prob_quality.py --era5 "$E" --climode "$C2" --arms $ARMS \
    --clim-years $(seq -s ' ' 2006 2015) --spec-leads 24 72 144 --out "$W/w2_prob_quality.json" --fig "$W/fig_w2_prob_quality.png"

# W4: on the step 27 table / figure with the WB2 forecasts (needs only the 2006-2020 ERA5 slice for min / max)
step w4_era5 python scripts/compare_recent/fetch_wb2_era5_5var.py --out data/wb2_5var --years $(seq -s ' ' 2006 2020)
P=$W/w34_paired.json; [ -e "$P" ] || P=$N/climode_paired_plus_$HOLDOUT.json
X="$P:plus_blend_band:ours blend_band+"
[ -n "$RAW_FM" ] && X="$X|$P:nopinn_blend_band:ours blend_band+ without PINN"
IFS='|' read -r -a XA <<< "$X"
step w4_report python scripts/compare_recent/report.py --out "$W/step27" --extra "${XA[@]}"
echo "[weak] finished: send $W/*.json $W/*.png $W/step27/ (small files)"
