#!/usr/bin/env bash
# Step 13 reported ensemble (regm_band) from scratch, in the order the results in docs/step13_sequence.md
# were made: ERA5 download -> stage 1 PINN -> stage 2 sequence FM and sequence regression (frozen PINN on
# every member) -> calibration (mu* = a + b mu_reg + c x(t), FM + PINN deviations, closed-form band Gamma)
# -> scores. Each step is skipped when its output exists, so a stopped run resumes where it stopped (the
# trainers also keep their own .train checkpoints). Step by step: docs/step13_reproduce.md.
#
#   bash scripts/run_step13_best.sh                    # everything: 5-6 h on 4 CPU cores after the data
#   DATA=/path/era5_5p625 OUT=runs/best bash scripts/run_step13_best.sh
#   CONTROL=1 bash scripts/run_step13_best.sh          # + the same mean and Gamma on the regression's deviations
#   CLIMODE=1 bash scripts/run_step13_best.sh          # + ClimODE's checkpoint on its first 16 test ICs, paired
#   SMOKE=1 bash scripts/run_step13_best.sh            # 2 iterations, 6 ICs a year: checks the chain (~15 min)
#
# Trained weights: put pinn_project.pt, pinn_project.json, seqfm_dct.pt, seqregress.pt and stats/ in $OUT
# first and the three trainers load them instead of training (sampling and calibration still run).
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=${SMOKE:-0}
DATA=${DATA:-data/era5_5p625}
OUT=${OUT:-$([ "$SMOKE" = 1 ] && echo runs/step13_smoke || echo runs/step13_best)}
THREADS=${THREADS:-4}
CONTROL=${CONTROL:-0}
CLIMODE=${CLIMODE:-0}
K=$DATA/kai6h
E=$DATA/era5_data
FM=$OUT/seqfm_dct_pinn
REG=$OUT/seqregress_pinn
if [ "$SMOKE" = 1 ]; then
    PINN_ARGS="--iters 2 --n-val 4"; GEN_ARGS="--iters 2 --max-ics 6"
    GUIDE_ARGS="--n-valid 1 --epochs 1 --guide-epochs 1"; SCORE_ARGS="--n-boot 20"
else
    PINN_ARGS=""; GEN_ARGS=""; GUIDE_ARGS=""; SCORE_ARGS=""
fi
mkdir -p "$OUT"
run() {  # run <name> <done-file> <command...>: skip when <done-file> exists, log to $OUT/<name>.log
    local name=$1 done=$2; shift 2
    if [ -e "$done" ]; then echo "[best] $name: $done exists, skipped"; return; fi
    echo "[best] $name: start $(date +%H:%M), log $OUT/$name.log"
    "$@" > "$OUT/$name.log" 2>&1 || { echo "[best] $name failed, see $OUT/$name.log"; tail -n 20 "$OUT/$name.log"; exit 1; }
    echo "[best] $name: done $(date +%H:%M)"
}

# 0. ERA5 5.625 deg from the public WeatherBench2 zarr: 6-hourly KAI files for the model years and
#    WeatherBench1-style files (ClimODE's 5 variables) for scoring; 2006-2011 only for the climatology.
run data_6h "$K/2018.nc" python scripts/wb2_to_kai.py --out "$DATA" --all-hours \
    --years 2012 2013 2014 2015 2016 2017 2018
run data_clim "$E/geopotential_500/geopotential_500_2006_5.625deg.nc" python scripts/wb2_to_kai.py --out "$DATA" \
    --years 2006 2007 2008 2009 2010 2011

# 1. stage 1: PINN projector on the low 8x16 DCT modes (project objective), then frozen. Also writes the
#    normalization statistics $OUT/stats that every later step reads.
run pinn_project "$OUT/pinn_project.json" python scripts/run_step13_pinn.py --kai "$K" --era5 "$E" --out "$OUT" \
    --threads "$THREADS" $PINN_ARGS

# 2. stage 2: sequence FM on the 3-D DCT of the 24 increments (6 h to 144 h), 8 members per IC, the frozen
#    PINN on every member; then the same U-Net as a sequence regression + statistical errors.
#    Ensembles of 2016 (calibration) and 2017-2018 (test) go to <arm>_ens_<year>.npz.
run seqfm_dct "${FM}_ens_2018.npz" python scripts/run_step13_seqfm.py --kai "$K" --era5 "$E" --out "$OUT" \
    --basis dct --pinn "$OUT/pinn_project.pt" --save-ens --resume-ens --threads "$THREADS" $GEN_ARGS
run seqregress "${REG}_ens_2018.npz" python scripts/run_step13_seqfm.py --kai "$K" --era5 "$E" --out "$OUT" \
    --model regress --pinn "$OUT/pinn_project.pt" --save-ens --resume-ens --threads "$THREADS" $GEN_ARGS

# 3. calibration, fitted on 2016 with every generator frozen: mu* = a + b mu_reg + c x(t) per lead, variable
#    and grid point, deviations of the FM + PINN members scaled by the closed-form band Gamma.
run guide_regm "${FM}_regm_band_ens_2018.npz" python scripts/run_step13_guide.py --ens "$FM" --era5 "$E" \
    --out "$OUT" --reg "$REG" --reg-mbm --save-arms regm_band --threads "$THREADS" $GUIDE_ARGS
ARMS="$REG ${FM}_regm_band"
if [ "$CONTROL" = 1 ]; then  # the paper's control: same mu* and Gamma formula on the regression's own deviations
    run guide_control "${REG}_regm_band_ens_2018.npz" python scripts/run_step13_guide.py --ens "$REG" --era5 "$E" \
        --out "$OUT" --reg "$REG" --reg-mbm --save-arms regm_band --threads "$THREADS" $GUIDE_ARGS
    ARMS="$ARMS ${REG}_regm_band"
fi

# 4. scores on ClimODE's test ICs (2017-2018), bootstrap differences against regression + statistical errors
run score "$OUT/score_best.json" python scripts/score_step13.py --era5 "$E" --out "$OUT/score_best.json" \
    --ref "$REG" --arms $ARMS $SCORE_ARGS

# 5. optional: ClimODE's published checkpoint on its first 8 ICs of each test year, then the paired comparison
if [ "$CLIMODE" = 1 ] && [ "$SMOKE" != 1 ]; then
    run climode "$OUT/climode/climode_pred.npz" python scripts/run_climode_eval.py --data "$E" --out "$OUT/climode" \
        --max-ics 8 --lead-steps 24
    run paired "$OUT/climode_paired_best.json" python scripts/compare_step13_climode.py --era5 "$E" \
        --climode "$OUT/climode/climode_pred.npz" --arms $ARMS --out "$OUT/climode_paired_best.json"
fi
echo "[best] finished: $OUT/score_best.json (summary at the end of $OUT/score.log)"
