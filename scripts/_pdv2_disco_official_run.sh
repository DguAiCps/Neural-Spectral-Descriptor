#!/usr/bin/env bash
# DiSCO with the official DiSCO-pytorch code/recipe on split v2 (2026-10-11).
#   build     build the official gputransform CUDA extension for sm_120 into
#             _external/disco_env/pylib (only needed once per environment)
#   selftest  CPU checks: polar input == official generator (6 sensors, NCLT raw
#             files), heading convention, FFT shims, query pools == official
#             construct_query_dict, official train_one_epoch on CPU, Euclidean
#             R@1 == KDTree, L2 ledger causality / exactness, ledger format
#   gpucheck  official CUDA extension == CPU transcription (few scans; GPU)
#   prep      index + polar occupancy stores for train / val / test_odom
#             keyframes with the official CUDA extension (NPROC parallel)
#   smoke     official loop on the first 200 keyframes per sequence, epochs 0
#             and 6 (GPU timing / memory; separate dirs, not used for results)
#   train     official training, 20 epochs; resumes an unfinished run
#   dump      past-only pr-dump-v1 ledgers on the 46 odometry test sequences
#   aggregate PR summary (results/pdv2_disco_official/odom/pr_summary_baselines.json)
# Every stage skips finished work. Run inside ring_baseline, e.g.
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_disco_official_run.sh selftest gpucheck prep smoke
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_disco_official_run.sh train dump aggregate
# The official code trains on one GPU (train_DiSCO.py:26); GPU=<id> picks it.
# TRAIN_FLAGS defaults to --fast-hard-negatives (exact k-NN in place of the
# per-step KDTree of get_random_hard_negatives: same hard negatives, speed
# only; user decision 2026-10-11).  TRAIN_FLAGS="" gives the official KDTree.
# Official train_DiSCO.py flags can be appended through TRAIN_FLAGS.
# The test dump uses the val-best checkpoint (train_selection.json 'chosen').
# SEED (default 1) = training seed.  SEED=1 keeps every location below; any
# other seed uses its own logs / results / runs / smoke dirs (suffix _s<SEED>)
# and shares the polar input stores.  TRAIN_NEEDS_TEST=0 lets `train` start
# once the train and val stores are complete (default 1: all stores).
#   paths     print the resolved locations for this SEED
#   verify    integrity of every complete polar store (failed-CUDA signature
#             rows + 16 rows per store recomputed on CPU); read-only, exits 1 on
#             failure.  VERIFY_RESET=1 moves failing stores aside and marks them
#             incomplete so that `prep` regenerates them.  `gen` itself now
#             compares every CUDA image with the CPU transcription and aborts
#             on disagreement (the extension ignores CUDA errors).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
ulimit -s 81920 || true          # official README: gputransform keeps per-point arrays on the stack
SEED=${SEED:-1}
[[ "$SEED" =~ ^[1-9][0-9]*$ ]] || { echo "SEED must be a positive integer: $SEED"; exit 1; }
export SEED
SFX=""; [ "$SEED" = 1 ] || SFX="_s$SEED"
LOG=logs/disco_official_pdv2$SFX
W=_external/disco_official_pdv2            # shared inputs (polar stores)
SEL=results/pdv2_disco_official$SFX/train_selection.json
SMOKE_SEL=$W/smoke$SFX/train_selection.json
TRAIN_NEEDS_TEST=${TRAIN_NEEDS_TEST:-1}
P="python3 scripts/_disco_official_prep.py"
T="python3 scripts/_disco_official_train.py"
D="python3 scripts/_disco_official_dump.py"
NPROC=${NPROC:-8}
TRAIN_FLAGS=${TRAIN_FLAGS---fast-hard-negatives}
mkdir -p "$LOG"
say() { echo "[disco-official] $* ($(date '+%F %T'))"; }

stage_build() {
  say build
  bash _external/disco_env/build_gputransform.sh > "$LOG/build_gputransform.log" 2>&1
  tail -1 "$LOG/build_gputransform.log"
}

stage_selftest() {
  say selftest
  CUDA_VISIBLE_DEVICES= $P selftest --n-scans 2 --out-json "$LOG/selftest_prep.json" \
    > "$LOG/selftest_prep.log" 2>&1
  CUDA_VISIBLE_DEVICES= $T --selftest --out-json "$LOG/selftest_train.json" \
    > "$LOG/selftest_train.log" 2>&1
  CUDA_VISIBLE_DEVICES= $D selftest > "$LOG/selftest_dump.log" 2>&1
  CUDA_VISIBLE_DEVICES= $D dry-run-cpu --seq nclt/2013-01-10 --n 80 --out-dir "$W/dryrun$SFX" \
    > "$LOG/selftest_dryrun.log" 2>&1
  rm -rf "$W/dryrun$SFX"
  for f in prep train dump dryrun; do tail -1 "$LOG/selftest_$f.log"; done
}

stage_gpucheck() {
  say gpucheck
  $P gpu-check --n-scans 3 --out-json "$LOG/gpu_check.json" > "$LOG/gpu_check.log" 2>&1
  tail -1 "$LOG/gpu_check.log"
}

stage_prep() {
  say "prep index"
  $P index > "$LOG/prep_index.log" 2>&1
  say "prep gen ($NPROC parallel)"
  $P jobs | xargs -r -P "$NPROC" -L 1 bash -c \
    'python3 scripts/_disco_official_prep.py gen --role "$0" --seq "$1" \
       >> "'"$LOG"'/prep_gen_$0_${1//\//_}.log" 2>&1'
  $P status | tail -4
}

stage_smoke() {
  say smoke
  $T --smoke 200 --smoke-epochs 0 6 $TRAIN_FLAGS > "$LOG/smoke_train.log" 2>&1
  python3 -c "import json; r=json.load(open('$SMOKE_SEL')); \
[print('smoke epoch', e['epoch'], {k: e[k] for k in ('train_s', 'trained_queries', 's_per_query', 'val_s', 'cuda_peak_gb', 'family_macro_r1')}) for e in r['epochs']]"
}

stage_train() {
  if [ "$TRAIN_NEEDS_TEST" = 1 ]; then
    $P status > /dev/null || { echo "polar stores incomplete"; exit 1; }
  else
    $P status --role train val > /dev/null || { echo "train/val polar stores incomplete"; exit 1; }
  fi
  if [ -f "$SEL" ]; then
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('$SEL')).get('finished') else 1)"; then
      say "train: finished, skip"; return
    fi
    RUN=$(python3 -c "import json; print(json.load(open('$SEL'))['_meta']['run_dir'])")
    say "train: resume $RUN"
    $T --resume "$RUN" $TRAIN_FLAGS >> "$LOG/train_stdout.log" 2>&1
  else
    say "train: start"
    $T $TRAIN_FLAGS > "$LOG/train_stdout.log" 2>&1
  fi
  tail -1 "$LOG/train_stdout.log"
}

stage_dump() {
  say dump
  $D dump >> "$LOG/dump.log" 2>&1
  tail -1 "$LOG/dump.log"
}

stage_verify() {
  say verify
  if [ "${VERIFY_RESET:-0}" = 1 ]; then
    $P verify --reset > "$LOG/verify_reset.log" 2>&1; tail -1 "$LOG/verify_reset.log"
  else
    $P verify > "$LOG/verify.log" 2>&1 || { tail -1 "$LOG/verify.log"; exit 1; }
    tail -1 "$LOG/verify.log"
  fi
}

stage_paths() {
  echo "SEED=$SEED LOG=$LOG SEL=$SEL SMOKE_SEL=$SMOKE_SEL TRAIN_NEEDS_TEST=$TRAIN_NEEDS_TEST"
  python3 -c "import sys; sys.path[:0] = ['scripts']
import _disco_official_prep as P, _disco_official_train as T, _disco_official_dump as D
r = lambda p: str(p.relative_to(P.REPO))
print('python SEED', P.SEED, 'RESULTS', r(P.RESULTS), 'RUNS', r(T.RUNS), 'SMOKE', r(T.SMOKE),
      'SELECTION', r(T.SELECTION), 'OUT', r(D.OUT), 'POLAR (shared)', r(P.POLAR))"
}

stage_aggregate() {
  say aggregate
  $D aggregate > "$LOG/aggregate.log" 2>&1
  tail -2 "$LOG/aggregate.log"
}

[ $# -gt 0 ] || set -- selftest gpucheck prep train dump aggregate
for s in "$@"; do
  "stage_$s"
done
say "DONE: $*"
