#!/usr/bin/env bash
# Step 13 reported ensembles from scratch, in the order the results in docs/step13_sequence.md were made:
# ERA5 download -> stage 1 PINN -> stage 2 sequence FM and sequence regression (frozen PINN on every member)
# -> calibration (mu* = a + b mu_reg + c x(t), FM + PINN deviations, closed-form band Gamma = regm_band; then
# the flow-dependent spread field on top: regm_flow with ClimODE's uncertainty net, regm_flowop with the
# operator pretrained on the regm_band members, docs/step13_sequence.md 4.3; then regm_sband_clim_flow, the
# band x cell climatological multipliers times regm_flow's case factor, 4.4) -> scores. Each step is skipped
# when its output exists, so a stopped run resumes where it stopped (the trainers also keep their own .train
# checkpoints). Step by step: docs/step13_reproduce.md.
#
#   bash scripts/run_step13_best.sh                    # everything: 6-7 h on 4 CPU cores after the data
#   DATA=/path/era5_5p625 OUT=runs/best bash scripts/run_step13_best.sh
#   CONTROL=1 bash scripts/run_step13_best.sh          # + the same mean and Gamma on the regression's deviations
#   CLIMODE=1 bash scripts/run_step13_best.sh          # + ClimODE's checkpoint on its first 16 test ICs, paired
#   SIGMA=0 bash scripts/run_step13_best.sh            # regm_band only, without the spread-field arms
#   SBAND=0 bash scripts/run_step13_best.sh            # without regm_sband_clim_flow (saves ~15 min)
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
SIGMA=${SIGMA:-1}
SBAND=${SBAND:-1}
K=$DATA/kai6h
E=$DATA/era5_data
FM=$OUT/seqfm_dct_pinn
REG=$OUT/seqregress_pinn
if [ "$SMOKE" = 1 ]; then
    PINN_ARGS="--iters 2 --n-val 4"; GEN_ARGS="--iters 2 --max-ics 6"
    GUIDE_ARGS="--n-valid 1 --epochs 1 --guide-epochs 1 --sigma-epochs 1"; SCORE_ARGS="--n-boot 20"; SIGMA_FULL=1
    ABAND_ARGS="--epochs 1 --width 8"
else
    PINN_ARGS=""; GEN_ARGS=""; GUIDE_ARGS=""; SCORE_ARGS=""; SIGMA_FULL=30; ABAND_ARGS=""
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
#    and grid point, deviations of the FM + PINN members scaled by the closed-form band Gamma (regm_band).
#    SIGMA=1 also fits the spread field sigma(x) on the error of mu* (ClimODE's Gaussian NLL) in the same run
#    and rescales the regm_band members case by case: regm_flow (ClimODE's uncertainty ResNet) and
#    regm_flowop (EnsembleToDeterministic pretrained on the regm_band members, then a new sigma head).
if [ "$SIGMA" = 1 ]; then
    run guide_regm "${FM}_regm_flowop_ens_2018.npz" python scripts/run_step13_guide.py --ens "$FM" --era5 "$E" \
        --out "$OUT" --reg "$REG" --reg-mbm --flow-sigma --flow-op --sigma-full-epochs "$SIGMA_FULL" \
        --save-arms regm_band regm_flow regm_flowop --threads "$THREADS" $GUIDE_ARGS
    ARMS="$REG ${FM}_regm_band ${FM}_regm_flow ${FM}_regm_flowop"
    SIG_ARMS="${FM}_regm_band ${FM}_regm_flow ${FM}_regm_flowop"
    if [ "$SBAND" = 1 ]; then
        # band x cell multipliers T = sum_b P_b M_b P_b with the per-lead cell climatology of the truth's error
        # energy (about 16 or 32 DCT coefficients per cell, the better on the held-out 2016 ICs), Gamma_max = 1,
        # times regm_flow's case factor sigma / rho. The same run fits the owner's case-adaptive band and
        # band x cell nets and reports them in aband_seqfm_dct_pinn.json (docs/step13_sequence.md 4.4).
        run aband "${FM}_regm_sband_clim_flow_ens_2018.npz" python scripts/run_step13_aband.py --ens "$FM" \
            --reg "$REG" --era5 "$E" --out "$OUT" --save-arms regm_sband_clim_flow --threads "$THREADS" $ABAND_ARGS
        ARMS="$ARMS ${FM}_regm_sband_clim_flow"
        SIG_ARMS="$SIG_ARMS ${FM}_regm_sband_clim_flow"
    fi
else
    run guide_regm "${FM}_regm_band_ens_2018.npz" python scripts/run_step13_guide.py --ens "$FM" --era5 "$E" \
        --out "$OUT" --reg "$REG" --reg-mbm --save-arms regm_band --threads "$THREADS" $GUIDE_ARGS
    ARMS="$REG ${FM}_regm_band"
fi
if [ "$CONTROL" = 1 ]; then  # the paper's control: same mu* and Gamma formula on the regression's own deviations
    run guide_control "${REG}_regm_band_ens_2018.npz" python scripts/run_step13_guide.py --ens "$REG" --era5 "$E" \
        --out "$OUT" --reg "$REG" --reg-mbm --save-arms regm_band --threads "$THREADS" $GUIDE_ARGS
    ARMS="$ARMS ${REG}_regm_band"
fi

# 4. scores on ClimODE's test ICs (2017-2018), bootstrap differences against regression + statistical errors
run score "$OUT/score_best.json" python scripts/score_step13.py --era5 "$E" --out "$OUT/score_best.json" \
    --ref "$REG" --arms $ARMS $SCORE_ARGS
# does the spread follow the case-to-case error? (CV of the spread, error by spread fifth; doc 7.6 and 4.3)
run flowdep "$OUT/flowdep.json" python scripts/diag_step13_flowdep.py --runs "$OUT" \
    --arms $(for q in $ARMS; do basename "$q"; done) --leads 24 48 72 96 120 144
if [ "$SIGMA" = 1 ]; then  # the spread-field arms against regm_band (bootstrap) and docs/step13/fig_flowdep.png
    run score_sigma "$OUT/score_sigma.json" python scripts/score_step13.py --era5 "$E" --out "$OUT/score_sigma.json" \
        --ref "${FM}_regm_band" --arms $SIG_ARMS $SCORE_ARGS
    run fig_flowdep "$OUT/fig/fig_flowdep.png" python scripts/plot_step13_flowdep.py --score "$OUT/score_sigma.json" \
        --flowdep "$OUT/flowdep.json" --out "$OUT/fig"
fi

# 5. optional: ClimODE's published checkpoint on its first 8 ICs of each test year, then the paired comparison
if [ "$CLIMODE" = 1 ] && [ "$SMOKE" != 1 ]; then
    run climode "$OUT/climode/climode_pred.npz" python scripts/run_climode_eval.py --data "$E" --out "$OUT/climode" \
        --max-ics 8 --lead-steps 24
    run paired "$OUT/climode_paired_best.json" python scripts/compare_step13_climode.py --era5 "$E" \
        --climode "$OUT/climode/climode_pred.npz" --arms $ARMS --out "$OUT/climode_paired_best.json"
fi
echo "[best] finished: $OUT/score_best.json (summary at the end of $OUT/score.log), $OUT/flowdep.json"
