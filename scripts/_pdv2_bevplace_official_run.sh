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
#   aggregate PR summary (results/pdv2_bevplace_official/odom/pr_summary_baselines.json)
# Every stage skips finished work. Run inside ring_baseline, e.g.
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_bevplace_official_run.sh bev smoke
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_bevplace_official_run.sh train dump aggregate
# Official main.py flags can be appended to the training call through
# TRAIN_FLAGS (e.g. TRAIN_FLAGS="--threads 12").
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
LOG=logs/bevplace_official_pdv2
W=_external/bevplace_official_pdv2
SEL=results/pdv2_bevplace_official/train_selection.json
B="python3 scripts/_bevplace_official_bev.py"
T="python3 scripts/_bevplace_official_train.py"
D="python3 scripts/_bevplace_official_dump.py"
NPROC=${NPROC:-12}
TRAIN_FLAGS=${TRAIN_FLAGS:-}
mkdir -p "$LOG"
say() { echo "[bevplace-official] $* ($(date '+%F %T'))"; }

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
  $B jobs | xargs -P "$NPROC" -L 1 bash -c \
    'python3 scripts/_bevplace_official_bev.py gen --role "$0" --seq "$1" --shard "$2" --nshards "$3" \
       >> "'"$LOG"'/bev_gen_$0_${1//\//_}.log" 2>&1'
  $B status | tail -4
}

stage_smoke() {
  say smoke
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $T --smoke 200 --nEpochs 1 --runsPath "$W/smoke/runs/" --cachePath "$W/smoke/cache/" \
     --selection-out "$W/smoke/train_selection.json" $TRAIN_FLAGS > "$LOG/smoke_train.log" 2>&1
  python3 -c "import json; e=json.load(open('$W/smoke/train_selection.json'))['epochs'][-1]; \
print('smoke epoch:', {k: e[k] for k in ('train_s', 'val_s', 'cuda_peak_gb', 'family_macro_r1')})"
}

stage_train() {
  $B status > /dev/null || { echo "BEV images incomplete"; exit 1; }
  if [ -f "$SEL" ]; then
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('$SEL')).get('finished') else 1)"; then
      say "train: finished, skip"; return
    fi
    RUN=$(python3 -c "import json; print(json.load(open('$SEL'))['_meta']['logdir'])")
    say "train: resume $RUN"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $T --resume "$RUN" --runsPath "$W/runs/" --cachePath "$W/cache/" $TRAIN_FLAGS \
      >> "$LOG/train.log" 2>&1
  else
    say "train: start"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $T --runsPath "$W/runs/" --cachePath "$W/cache/" $TRAIN_FLAGS > "$LOG/train.log" 2>&1
  fi
  tail -1 "$LOG/train.log"
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

[ $# -gt 0 ] || set -- selftest bev train dump aggregate
for s in "$@"; do
  "stage_$s"
done
say "DONE: $*"
