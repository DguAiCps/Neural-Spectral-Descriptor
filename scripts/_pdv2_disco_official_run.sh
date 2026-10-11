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
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
ulimit -s 81920 || true          # official README: gputransform keeps per-point arrays on the stack
LOG=logs/disco_official_pdv2
W=_external/disco_official_pdv2
SEL=results/pdv2_disco_official/train_selection.json
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
  CUDA_VISIBLE_DEVICES= $D dry-run-cpu --seq nclt/2013-01-10 --n 80 --out-dir "$W/dryrun" \
    > "$LOG/selftest_dryrun.log" 2>&1
  rm -rf "$W/dryrun"
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
  $P jobs | xargs -P "$NPROC" -L 1 bash -c \
    'python3 scripts/_disco_official_prep.py gen --role "$0" --seq "$1" \
       >> "'"$LOG"'/prep_gen_$0_${1//\//_}.log" 2>&1'
  $P status | tail -4
}

stage_smoke() {
  say smoke
  $T --smoke 200 --smoke-epochs 0 6 $TRAIN_FLAGS > "$LOG/smoke_train.log" 2>&1
  python3 -c "import json; r=json.load(open('$W/smoke/train_selection.json')); \
[print('smoke epoch', e['epoch'], {k: e[k] for k in ('train_s', 'trained_queries', 's_per_query', 'val_s', 'cuda_peak_gb', 'family_macro_r1')}) for e in r['epochs']]"
}

stage_train() {
  $P status > /dev/null || { echo "polar stores incomplete"; exit 1; }
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
