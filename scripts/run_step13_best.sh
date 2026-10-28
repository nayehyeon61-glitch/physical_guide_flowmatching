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
#   SR=1 bash scripts/run_step13_best.sh               # + step 19 output-stage super-resolution of every scored arm
#                                                      #   (2.8125 deg truth download, SR net, 2.8 deg scores; docs/RUN_ON_GPU.md 12)
#   BLEND=1 PROFILE=climode bash scripts/run_step13_best.sh   # + step 20 blend_band: G(x)-feature core trained on the
#                                                      #   profile's years, mu* = a + b core + c regression + d x(t), scored
#                                                      #   with the energy scores and the step 20 figure (docs/RUN_ON_GPU.md 13)
#
# Profiles (docs/step15_training_system.md) set the training years, iterations, batches and validation:
#   PROFILE=small (default)   train 2012-2015, the CPU settings of docs/step13_reproduce.md
#   PROFILE=broad             train 1979-2015 on a GPU: PINN 8000 it batch 8, FM / regression 40000 it batch 16,
#                             validation every 1000 steps with 4-member z500 RMSE, memory-mapped years,
#                             DEVICE=auto (the 2026-10-27 broad run, docs/step13_broad)
#   PROFILE=broad_long        broad with FM / regression 120000 it and PATIENCE=10 (stop after 10 validations
#                             without improvement, the best state is selected)
#   PROFILE=climode           ClimODE's years: train 2006-2015 (validation 2016, test 2017-2018), broad settings
#   PROFILE=smoke             = SMOKE=1
#   Comparing profiles: scripts/run_step15_compare.sh (climode + broad_long, then the reports side by side)
# Every profile value can be overridden: TRAIN_YEARS="1990 ... 2015" PINN_ITERS GEN_ITERS GEN_BATCH VAL_EVERY
# VAL_MEMBERS PATIENCE MAX_HOURS (per trainer invocation: saves and exits 75, rerun to resume) DEVICE CACHE
# (1 = memory-mapped years under $OUT/cache) AMP (1 = bf16 autocast on cuda). With SR=1: SR_TRAIN_YEARS (default
# TRAIN_YEARS; 2.8125 deg files are 240 MB a year), SR_ITERS, SR_BATCH, SR_CHUNK (ICs per scoring chunk), SR_SAVE
# (1 = write the super-resolved members, 0.7 GB per arm and year). With BLEND=1: CORE_ITERS, CORE_BATCH (the G(x)
# core; small 4000 x 4, broad / climode 20000 x 16). Each step appends a line to
# $OUT/manifest.jsonl (command, commit, start, end, exit).
#
# Trained weights: put pinn_project.pt, pinn_project.json, seqfm_dct.pt, seqregress.pt and stats/ in $OUT
# first and the three trainers load them instead of training (sampling and calibration still run).
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=${SMOKE:-0}
PROFILE=${PROFILE:-$([ "$SMOKE" = 1 ] && echo smoke || echo small)}
[ "$PROFILE" = smoke ] && SMOKE=1
DATA=${DATA:-data/era5_5p625}
OUT=${OUT:-$([ "$SMOKE" = 1 ] && echo runs/step13_smoke || echo runs/step13_best)}
THREADS=${THREADS:-4}
CONTROL=${CONTROL:-0}
CLIMODE=${CLIMODE:-0}
SIGMA=${SIGMA:-1}
SBAND=${SBAND:-1}
SR=${SR:-0}
SR_SAVE=${SR_SAVE:-0}
BLEND=${BLEND:-0}
VAL_YEAR=${VAL_YEAR:-2016}
TEST_YEARS=${TEST_YEARS:-"2017 2018"}
case "$PROFILE" in
    small|smoke)
        TRAIN_YEARS=${TRAIN_YEARS:-"2012 2013 2014 2015"}
        PINN_ITERS=${PINN_ITERS:-1000}; PINN_BATCH=${PINN_BATCH:-4}
        GEN_ITERS=${GEN_ITERS:-3000}; GEN_BATCH=${GEN_BATCH:-4}
        VAL_EVERY=${VAL_EVERY:-0}; VAL_MEMBERS=${VAL_MEMBERS:-0}
        SR_ITERS=${SR_ITERS:-3000}; SR_BATCH=${SR_BATCH:-16}; SR_CHUNK=${SR_CHUNK:-6}
        CORE_ITERS=${CORE_ITERS:-4000}; CORE_BATCH=${CORE_BATCH:-4}; CORE_VAL_EVERY=${CORE_VAL_EVERY:-500}
        DEVICE=${DEVICE:-cpu}; CACHE=${CACHE:-0} ;;
    broad|broad_long|climode)
        # broad: the owner's 2026-10-27 GPU run (37 years). broad_long: the same data, 3x the FM / regression
        # steps (cosine over 120000) with early stopping on the validation loss. climode: ClimODE's data
        # years (train 2006-2015, validation 2016, test 2017-2018) with the broad training settings, so the
        # comparison with broad isolates the 27 extra years (docs/step15_training_system.md 7).
        case "$PROFILE" in
            climode) TRAIN_YEARS=${TRAIN_YEARS:-"$(seq -s ' ' 2006 2015)"} ;;
            *) TRAIN_YEARS=${TRAIN_YEARS:-"$(seq -s ' ' 1979 2015)"} ;;
        esac
        PINN_ITERS=${PINN_ITERS:-8000}; PINN_BATCH=${PINN_BATCH:-8}
        GEN_ITERS=${GEN_ITERS:-$([ "$PROFILE" = broad_long ] && echo 120000 || echo 40000)}; GEN_BATCH=${GEN_BATCH:-16}
        VAL_EVERY=${VAL_EVERY:-1000}; VAL_MEMBERS=${VAL_MEMBERS:-4}
        PATIENCE=${PATIENCE:-$([ "$PROFILE" = broad_long ] && echo 10 || echo 0)}
        SR_ITERS=${SR_ITERS:-20000}; SR_BATCH=${SR_BATCH:-32}; SR_CHUNK=${SR_CHUNK:-30}
        CORE_ITERS=${CORE_ITERS:-20000}; CORE_BATCH=${CORE_BATCH:-16}; CORE_VAL_EVERY=${CORE_VAL_EVERY:-1000}
        DEVICE=${DEVICE:-auto}; CACHE=${CACHE:-1} ;;
    *) echo "unknown PROFILE $PROFILE (small, broad, broad_long, climode, smoke)"; exit 2 ;;
esac
PATIENCE=${PATIENCE:-0}
MAX_HOURS=${MAX_HOURS:-0}
AMP=${AMP:-0}
SR_TRAIN_YEARS=${SR_TRAIN_YEARS:-$TRAIN_YEARS}
K=$DATA/kai6h
E=$DATA/era5_data
FM=$OUT/seqfm_dct_pinn
REG=$OUT/seqregress_pinn
PINN_YEARS_ARGS="--train-years $TRAIN_YEARS --val-year $VAL_YEAR"
YEARS_ARGS="$PINN_YEARS_ARGS --test-years $TEST_YEARS"
TRAIN_ARGS="--device $DEVICE --val-every $VAL_EVERY --patience $PATIENCE --max-hours $MAX_HOURS --threads $THREADS"
[ "$CACHE" = 1 ] && TRAIN_ARGS="$TRAIN_ARGS --cache $OUT/cache"
[ "$AMP" = 1 ] && TRAIN_ARGS="$TRAIN_ARGS --amp"
if [ "$SMOKE" = 1 ]; then
    PINN_ARGS="--iters 2 --n-val 4"; GEN_ARGS="--iters 2 --max-ics 6"
    GUIDE_ARGS="--n-valid 1 --epochs 1 --guide-epochs 1 --sigma-epochs 1"; SCORE_ARGS="--n-boot 20"; SIGMA_FULL=1
    ABAND_ARGS="--epochs 1 --width 8"; SR_ARGS="--iters 2 --val-every 1 --n-val 4 --max-ics 6"
    CORE_ARGS="--iters 2 --batch 2 --n-val 4 --n-val-train 4 --val-K 8 --val-every 1"; PREDICT_ARGS="--max-ics 6"
else
    PINN_ARGS="--iters $PINN_ITERS --batch $PINN_BATCH"; GEN_ARGS="--iters $GEN_ITERS --batch $GEN_BATCH --val-members $VAL_MEMBERS"
    GUIDE_ARGS=""; SCORE_ARGS=""; SIGMA_FULL=30; ABAND_ARGS=""; SR_ARGS="--iters $SR_ITERS --batch $SR_BATCH"
    CORE_ARGS="--iters $CORE_ITERS --batch $CORE_BATCH --n-val 16 --n-val-train 8 --val-K 24 --val-every $CORE_VAL_EVERY"
    PREDICT_ARGS=""
fi
[ "$AMP" = 1 ] && SR_ARGS="$SR_ARGS --amp"
[ "$SR_SAVE" = 1 ] && SR_ARGS="$SR_ARGS --save-sr"
mkdir -p "$OUT"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "")
echo "[best] profile $PROFILE, train $TRAIN_YEARS, device $DEVICE, out $OUT, commit $COMMIT"
run() {  # run <name> <done-file> <command...>: skip when <done-file> exists, log to $OUT/<name>.log
    local name=$1 done=$2; shift 2
    if [ -e "$done" ]; then echo "[best] $name: $done exists, skipped"; return; fi
    local start; start=$(date -u +%FT%TZ)
    echo "[best] $name: start $(date +%H:%M), log $OUT/$name.log"
    local rc=0
    "$@" > "$OUT/$name.log" 2>&1 || rc=$?
    printf '{"name": "%s", "profile": "%s", "commit": "%s", "start": "%s", "end": "%s", "exit": %d, "cmd": "%s"}\n' \
        "$name" "$PROFILE" "$COMMIT" "$start" "$(date -u +%FT%TZ)" "$rc" "$(printf '%q ' "$@")" >> "$OUT/manifest.jsonl"
    if [ "$rc" = 75 ]; then
        echo "[best] $name: wall-clock budget reached, state saved; rerun the same command to resume"; exit 75
    elif [ "$rc" != 0 ]; then
        echo "[best] $name failed ($rc), see $OUT/$name.log"; tail -n 20 "$OUT/$name.log"; exit 1
    fi
    echo "[best] $name: done $(date +%H:%M)"
}

# 0. ERA5 5.625 deg from the public WeatherBench2 zarr: 6-hourly KAI files for the model years and
#    WeatherBench1-style files (ClimODE's 5 variables) for scoring; 2006-2015 for the tercile climatology
#    (daily files only, when they are not training years already).
LAST_TEST=${TEST_YEARS##* }
run data_6h "$K/$LAST_TEST.nc" python scripts/wb2_to_kai.py --out "$DATA" --all-hours \
    --years $TRAIN_YEARS $VAL_YEAR $TEST_YEARS
CLIM_YEARS=""
for y in $(seq 2006 2015); do case " $TRAIN_YEARS " in *" $y "*) ;; *) CLIM_YEARS="$CLIM_YEARS $y" ;; esac; done
if [ -n "$CLIM_YEARS" ]; then
    run data_clim "$E/geopotential_500/geopotential_500_${CLIM_YEARS##* }_5.625deg.nc" python scripts/wb2_to_kai.py \
        --out "$DATA" --years $CLIM_YEARS
fi

# 1. stage 1: PINN projector on the low 8x16 DCT modes (project objective), then frozen. Also writes the
#    normalization statistics $OUT/stats that every later step reads.
run pinn_project "$OUT/pinn_project.json" python scripts/run_step13_pinn.py --kai "$K" --era5 "$E" --out "$OUT" \
    $PINN_YEARS_ARGS $TRAIN_ARGS $PINN_ARGS

# 2. stage 2: sequence FM on the 3-D DCT of the 24 increments (6 h to 144 h), 8 members per IC, the frozen
#    PINN on every member; then the same U-Net as a sequence regression + statistical errors.
#    Ensembles of 2016 (calibration) and 2017-2018 (test) go to <arm>_ens_<year>.npz.
run seqfm_dct "${FM}_ens_$LAST_TEST.npz" python scripts/run_step13_seqfm.py --kai "$K" --era5 "$E" --out "$OUT" \
    --basis dct --pinn "$OUT/pinn_project.pt" --save-ens --resume-ens $YEARS_ARGS $TRAIN_ARGS $GEN_ARGS
run seqregress "${REG}_ens_$LAST_TEST.npz" python scripts/run_step13_seqfm.py --kai "$K" --era5 "$E" --out "$OUT" \
    --model regress --pinn "$OUT/pinn_project.pt" --save-ens --resume-ens $YEARS_ARGS $TRAIN_ARGS $GEN_ARGS

# 3. calibration, fitted on 2016 with every generator frozen: mu* = a + b mu_reg + c x(t) per lead, variable
#    and grid point, deviations of the FM + PINN members scaled by the closed-form band Gamma (regm_band).
#    SIGMA=1 also fits the spread field sigma(x) on the error of mu* (ClimODE's Gaussian NLL) in the same run
#    and rescales the regm_band members case by case: regm_flow (ClimODE's uncertainty ResNet) and
#    regm_flowop (EnsembleToDeterministic pretrained on the regm_band members, then a new sigma head).
if [ "$SIGMA" = 1 ]; then
    run guide_regm "${FM}_regm_flowop_ens_$LAST_TEST.npz" python scripts/run_step13_guide.py --ens "$FM" --era5 "$E" \
        --out "$OUT" --reg "$REG" --reg-mbm --flow-sigma --flow-op --sigma-full-epochs "$SIGMA_FULL" \
        --val-year "$VAL_YEAR" --test-years $TEST_YEARS \
        --save-arms regm_band regm_flow regm_flowop --threads "$THREADS" $GUIDE_ARGS
    ARMS="$REG ${FM}_regm_band ${FM}_regm_flow ${FM}_regm_flowop"
    SIG_ARMS="${FM}_regm_band ${FM}_regm_flow ${FM}_regm_flowop"
    if [ "$SBAND" = 1 ]; then
        # band x cell multipliers T = sum_b P_b M_b P_b with the per-lead cell climatology of the truth's error
        # energy (about 16 or 32 DCT coefficients per cell, the better on the held-out 2016 ICs), Gamma_max = 1,
        # times regm_flow's case factor sigma / rho. The same run fits the owner's case-adaptive band and
        # band x cell nets and reports them in aband_seqfm_dct_pinn.json (docs/step13_sequence.md 4.4).
        run aband "${FM}_regm_sband_clim_flow_ens_$LAST_TEST.npz" python scripts/run_step13_aband.py --ens "$FM" \
            --reg "$REG" --era5 "$E" --out "$OUT" --val-year "$VAL_YEAR" --test-years $TEST_YEARS \
            --save-arms regm_sband_clim_flow --threads "$THREADS" $ABAND_ARGS
        ARMS="$ARMS ${FM}_regm_sband_clim_flow"
        SIG_ARMS="$SIG_ARMS ${FM}_regm_sband_clim_flow"
    fi
else
    run guide_regm "${FM}_regm_band_ens_$LAST_TEST.npz" python scripts/run_step13_guide.py --ens "$FM" --era5 "$E" \
        --out "$OUT" --reg "$REG" --reg-mbm --val-year "$VAL_YEAR" --test-years $TEST_YEARS \
        --save-arms regm_band --threads "$THREADS" $GUIDE_ARGS
    ARMS="$REG ${FM}_regm_band"
fi
if [ "$CONTROL" = 1 ]; then  # the paper's control: same mu* and Gamma formula on the regression's own deviations
    run guide_control "${REG}_regm_band_ens_$LAST_TEST.npz" python scripts/run_step13_guide.py --ens "$REG" --era5 "$E" \
        --out "$OUT" --reg "$REG" --reg-mbm --val-year "$VAL_YEAR" --test-years $TEST_YEARS \
        --save-arms regm_band --threads "$THREADS" $GUIDE_ARGS
    ARMS="$ARMS ${REG}_regm_band"
fi

# 3c. optional (BLEND=1): the step 20 blended mean (docs/step20_pinn_score.md). The G(x)-feature tendency core
#     (step 18, pgcfm/dyncore.py: the strong-form operator tendency -R0(x) of u, v, T, q as 48 input channels of
#     the ODE core, docs/step18_downscaling.md 5) is trained on the profile's years with the trainer options of
#     step 15, predicts ClimODE's ICs, and the guide fits mu* = a + b core + c regression mean + d x(t) per lead and
#     variable on the validation year under the FM + PINN deviations and the band Gamma (arm blend_band). Scored
#     against the regression next to regm_band (and regm_sband_clim_flow when present), with the energy scores and
#     the step 20 figure: $OUT/score_blend.json, $OUT/energy_blend.json, $OUT/fig/fig_step20.png.
if [ "$BLEND" = 1 ]; then
    CORE=$OUT/dyn_gfeat
    run core_gfeat "$CORE.json" python scripts/run_step16_dynpinn.py --kai "$K" --era5 "$E" --out "$OUT" --tag dyn_gfeat \
        --gfeat --lambda-p 0 $PINN_YEARS_ARGS --K 8 --curriculum 1 2 4 8 --val-chunk 4 $TRAIN_ARGS $CORE_ARGS
    run core_predict "${CORE}_pred.npz" python scripts/run_step16_predict.py --kai "$K" --era5 "$E" --run "$OUT" \
        --tag dyn_gfeat --val-year "$VAL_YEAR" --test-years $TEST_YEARS --threads "$THREADS" $PREDICT_ARGS
    run guide_blend "${FM}_blend_band_ens_$LAST_TEST.npz" python scripts/run_step13_guide.py --ens "$FM" --era5 "$E" \
        --out "$OUT" --core "$CORE" --reg "$REG" --reg-mbm --blend --val-year "$VAL_YEAR" --test-years $TEST_YEARS \
        --save-arms blend_band --threads "$THREADS" $GUIDE_ARGS
    BLEND_ARMS="$REG ${FM}_regm_band ${FM}_blend_band"
    [ -e "${FM}_regm_sband_clim_flow_ens_$LAST_TEST.npz" ] && BLEND_ARMS="$BLEND_ARMS ${FM}_regm_sband_clim_flow"
    run score_blend "$OUT/score_blend.json" python scripts/score_step13.py --era5 "$E" --out "$OUT/score_blend.json" \
        --ref "$REG" --arms $BLEND_ARMS --test-years $TEST_YEARS $SCORE_ARGS
    run energy_blend "$OUT/energy_blend.json" python scripts/step20_energy_score.py --era5 "$E" \
        --out "$OUT/energy_blend.json" --arms $BLEND_ARMS --test-years $TEST_YEARS --threads "$THREADS"
    run fig_blend "$OUT/fig/fig_step20.png" python scripts/plot_step13.py --era5 "$E" --scores "$OUT/score_blend.json" \
        --years $TEST_YEARS --figures fig_step20 --out "$OUT/fig" --step "Step 20"
fi

# 4. scores on ClimODE's test ICs (2017-2018), bootstrap differences against regression + statistical errors
run score "$OUT/score_best.json" python scripts/score_step13.py --era5 "$E" --out "$OUT/score_best.json" \
    --ref "$REG" --arms $ARMS --test-years $TEST_YEARS $SCORE_ARGS
# does the spread follow the case-to-case error? (CV of the spread, error by spread fifth; doc 7.6 and 4.3)
run flowdep "$OUT/flowdep.json" python scripts/diag_step13_flowdep.py --runs "$OUT" \
    --arms $(for q in $ARMS; do basename "$q"; done) --leads 24 48 72 96 120 144
if [ "$SIGMA" = 1 ]; then  # the spread-field arms against regm_band (bootstrap) and docs/step13/fig_flowdep.png
    run score_sigma "$OUT/score_sigma.json" python scripts/score_step13.py --era5 "$E" --out "$OUT/score_sigma.json" \
        --ref "${FM}_regm_band" --arms $SIG_ARMS --test-years $TEST_YEARS $SCORE_ARGS
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
# 6. optional: step 19 output-stage super-resolution (docs/step19_sr_output.md). The 2.8125 deg truth of the 5
#    ClimODE variables (WB2 128x64) lands next to the 5.625 deg files in $E; the SR net (pgcfm/sr.py) learns
#    analysis -> analysis pairs on the SR years; then every scored arm is super-resolved member by member and
#    scored at 2.8 deg against the interpolation of the same members (RMSE, spread, CRPS; $OUT/sr/sr.json,
#    $OUT/fig/fig_sr_sr_<arm>.png). SR_SAVE=1 also writes the super-resolved members.
if [ "$SR" = 1 ]; then
    run data_fine "$E/geopotential_500/geopotential_500_${LAST_TEST}_2.8125deg.nc" python scripts/wb2_to_kai.py \
        --out "$DATA" --grid 128x64 --climode-only --years $SR_TRAIN_YEARS $VAL_YEAR $TEST_YEARS
    run sr "$OUT/sr/sr.json" python scripts/run_step19_sr.py --coarse "$E" --fine "$E" --device "$DEVICE" \
        --train-years $SR_TRAIN_YEARS --val-year "$VAL_YEAR" --test-years $TEST_YEARS --chunk "$SR_CHUNK" \
        --ens $ARMS --out "$OUT/sr" --fig "$OUT/fig" --threads "$THREADS" $SR_ARGS
fi
echo "[best] finished: $OUT/score_best.json (summary at the end of $OUT/score.log), $OUT/flowdep.json"
