#!/usr/bin/env bash
# BEVPlace++ with the official BEVPlace2 code on split v2 (2026-10-11): the
# released code's actual behaviour; only protocol changes and run fixes
# (logs/baseline_official_20261011/bevplace.md section 8).
#   selftest  CPU checks (BEV adapter == official generator; dataset adapter,
#             chunked mining distance, batched val recall == official/previous)
#   bev       index + official BEV images for train / val / test_odom keyframes
#             (NPROC parallel shards, resumable)
#   smoke     1 epoch on the first 200 frames of every sequence (GPU timing and
#             memory check; separate run/cache dirs, not used for results)
#   train     official training (Adam 1e-4, no decay/schedule, 40 epochs, hard
#             mining every epoch, image-rotation augmentation), seed 1, best
#             epoch by v2-val family-equal R@1; resumes an unfinished run from
#             train_selection.json
#   dump      past-only pr-dump-v1 ledgers on the 46 odometry test sequences
#             (official 8192-D descriptor, official L2 distance, prefix only)
#   aggregate PR summary (results/pdv2_bevplace_official/odom/pr_summary_baselines.json;
#             SEED=N != 1: results/pdv2_bevplace_official_sN/odom/...)
#   paths     print the resolved locations and commands for this SEED (runs nothing)
# Every stage skips finished work. Run inside ring_baseline, e.g.
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_bevplace_official_run.sh bev smoke
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_bevplace_official_run.sh train dump aggregate
# Official main.py flags can be appended to the training call through
# TRAIN_FLAGS (e.g. TRAIN_FLAGS="--threads 12").
# SEED (default 1) selects the training seed.  SEED=1 (or unset) uses the
# original locations unchanged; SEED=N != 1 trains with --seed N and keeps
# everything training-dependent apart:
#   _external/bevplace_official_pdv2/sN/{runs,cache,smoke}/
#   results/pdv2_bevplace_official_sN/{train_selection.json,odom/}
#   logs/bevplace_official_pdv2_sN/
# BEV images, index and poses (_external/bevplace_official_pdv2/bev/) are shared.
# TRAIN_NEEDS_TEST_BEV=0 lets `train` start when only the train/val BEV images
# are complete (default 1: all roles must be complete).
#   docker exec -e SEED=2 -e CUDA_VISIBLE_DEVICES=0 -w /workspace/Neural-Spectral-Codec \
#     ring_baseline bash scripts/_pdv2_bevplace_official_run.sh paths train
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
SEED=${SEED:-1}
case "$SEED" in ''|*[!0-9]*) echo "SEED must be a positive integer, got '$SEED'"; exit 1 ;; esac
SEED=$((10#$SEED))
[ "$SEED" -ge 1 ] || { echo "SEED must be >= 1"; exit 1; }
W=_external/bevplace_official_pdv2
if [ "$SEED" -eq 1 ]; then          # seed 1: original locations, original commands
  LOG=logs/bevplace_official_pdv2
  WS=$W
  RES=results/pdv2_bevplace_official
  SEED_ARG=""
  SEL_ARG=""
  RES_ARG=""
else
  LOG=logs/bevplace_official_pdv2_s$SEED
  WS=$W/s$SEED
  RES=results/pdv2_bevplace_official_s$SEED
  SEED_ARG="--seed $SEED"
  SEL_ARG="--selection-out $RES/train_selection.json"
  RES_ARG="--results-dir $RES"
fi
SEL=$RES/train_selection.json
B="python3 scripts/_bevplace_official_bev.py"
T="python3 scripts/_bevplace_official_train.py"
D="python3 scripts/_bevplace_official_dump.py"
NPROC=${NPROC:-12}
TRAIN_FLAGS=${TRAIN_FLAGS:-}
TRAIN_NEEDS_TEST_BEV=${TRAIN_NEEDS_TEST_BEV:-1}
mkdir -p "$LOG"
say() { echo "[bevplace-official] $* ($(date '+%F %T'))"; }

stage_paths() {   # print the resolved locations and commands (nothing is run)
  echo "SEED=$SEED TRAIN_NEEDS_TEST_BEV=$TRAIN_NEEDS_TEST_BEV"
  echo "LOG=$LOG"
  echo "runsPath=$WS/runs/ cachePath=$WS/cache/ smoke=$WS/smoke/"
  echo "SEL=$SEL"
  echo "results=$RES ledgers=$RES/odom"
  echo "train(start):  " $T $SEED_ARG $SEL_ARG --runsPath "$WS/runs/" --cachePath "$WS/cache/" $TRAIN_FLAGS
  echo "train(resume): " $T --resume '<logdir from SEL>' $SEED_ARG $SEL_ARG --runsPath "$WS/runs/" --cachePath "$WS/cache/" $TRAIN_FLAGS
  echo "smoke:         " $T $SEED_ARG --smoke 200 --nEpochs 1 --runsPath "$WS/smoke/runs/" --cachePath "$WS/smoke/cache/" --selection-out "$WS/smoke/train_selection.json" $TRAIN_FLAGS
  if [ "$TRAIN_NEEDS_TEST_BEV" = 1 ]; then echo "train check:   " $B status; else echo "train check:   " $B status --role train val; fi
  echo "dump:          " $D dump $RES_ARG
  echo "aggregate:     " $D aggregate $RES_ARG
}

stage_selftest() {
  say selftest
  CUDA_VISIBLE_DEVICES= $B selftest --n-scans 2 --out-json "$LOG/selftest_bev.json" \
    > "$LOG/selftest_bev.log" 2>&1
  CUDA_VISIBLE_DEVICES= $T --selftest > "$LOG/selftest_train.log" 2>&1
  CUDA_VISIBLE_DEVICES= $D selftest > "$LOG/selftest_dump.log" 2>&1
  tail -1 "$LOG/selftest_bev.log"; tail -1 "$LOG/selftest_train.log"; tail -1 "$LOG/selftest_dump.log"
}

stage_bev() {
  say "bev index"
  $B index > "$LOG/bev_index.log" 2>&1
  say "bev gen ($NPROC parallel)"
  $B jobs | xargs -r -P "$NPROC" -L 1 bash -c \
    'python3 scripts/_bevplace_official_bev.py gen --role "$0" --seq "$1" --shard "$2" --nshards "$3" \
       >> "'"$LOG"'/bev_gen_$0_${1//\//_}.log" 2>&1'
  $B status | tail -4
}

stage_smoke() {
  say smoke
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $T $SEED_ARG --smoke 200 --nEpochs 1 --runsPath "$WS/smoke/runs/" --cachePath "$WS/smoke/cache/" \
     --selection-out "$WS/smoke/train_selection.json" $TRAIN_FLAGS > "$LOG/smoke_train.log" 2>&1
  python3 -c "import json; e=json.load(open('$WS/smoke/train_selection.json'))['epochs'][-1]; \
print('smoke epoch:', {k: e[k] for k in ('train_s', 'val_s', 'cuda_peak_gb', 'family_macro_r1')})"
}

stage_train() {
  if [ "$TRAIN_NEEDS_TEST_BEV" = 1 ]; then
    $B status > /dev/null || { echo "BEV images incomplete"; exit 1; }
  else
    $B status --role train val > /dev/null || { echo "train/val BEV images incomplete"; exit 1; }
  fi
  if [ -f "$SEL" ]; then
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('$SEL')).get('finished') else 1)"; then
      say "train: finished, skip"; return
    fi
    RUN=$(python3 -c "import json; print(json.load(open('$SEL'))['_meta']['logdir'])")
    say "train: resume $RUN"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $T --resume "$RUN" $SEED_ARG $SEL_ARG --runsPath "$WS/runs/" --cachePath "$WS/cache/" $TRAIN_FLAGS \
      >> "$LOG/train.log" 2>&1
  else
    say "train: start"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $T $SEED_ARG $SEL_ARG --runsPath "$WS/runs/" --cachePath "$WS/cache/" $TRAIN_FLAGS > "$LOG/train.log" 2>&1
  fi
  tail -1 "$LOG/train.log"
}

stage_dump() {
  say dump
  $D dump $RES_ARG >> "$LOG/dump.log" 2>&1
  tail -1 "$LOG/dump.log"
}

stage_aggregate() {
  say aggregate
  $D aggregate $RES_ARG > "$LOG/aggregate.log" 2>&1
  tail -2 "$LOG/aggregate.log"
}

[ $# -gt 0 ] || set -- selftest bev train dump aggregate
for s in "$@"; do
  "stage_$s"
done
say "DONE: $*"
