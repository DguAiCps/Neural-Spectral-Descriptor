#!/usr/bin/env bash
# DeepRING on split v2 -- PAPER-BASED RE-IMPLEMENTATION (no public official
# code exists). Prepared 2026-10-11, NOT yet run on GPU.
# Report: logs/baseline_official_20261011/deepring.md
#   check      CPU checks (scorer, UNet import, preprocessing vs official class,
#              episode sampler, cost, tiny CPU dry run incl. resume and ledger format)
#   gpu-check  official GPU kernels vs CPU paths, fp32 scorer, one 168-scan episode
#   bev        occupancy BEVs for train / val / test_odom keyframes (NPROC parallel
#              sequences, resumable). test_odom BEVs are copied from the RING
#              pipeline cache (_external/ring_official_cache/odom) when it is
#              complete; run its ring-bev first to save that work (optional)
#   train      paper recipe, 20 epochs x 60 episodes, seed SEED (default 1), per-epoch v2-val
#              selection; resumes from the last epoch checkpoint
#   dump       past-only pr-dump-v1 ledgers on the 46 odometry test sequences
#   aggregate  PR summary (results/pdv2_deepring_reimpl/odom/pr_summary_baselines.json)
# Every stage skips finished work. Run inside ring_baseline, e.g.
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_deepring_reimpl_run.sh gpu-check bev train dump aggregate
# Environment: NPROC (parallel BEV sequences, default 8), GPU (default 0),
# SEED (training seed, default 1).  SEED=1 keeps every location below; any
# other seed uses its own logs / results / run dir (_s<SEED>, runs/s<SEED>)
# and shares the BEV caches.
#   paths      print the resolved locations for this SEED
set -euo pipefail
cd "$(dirname "$0")/.."
ulimit -s 81920            # RING README: voxelocc keeps per-point arrays on the stack
export PYTHONDONTWRITEBYTECODE=1
SEED=${SEED:-1}
[[ "$SEED" =~ ^[1-9][0-9]*$ ]] || { echo "SEED must be a positive integer: $SEED"; exit 1; }
export SEED
SFX=""; [ "$SEED" = 1 ] || SFX="_s$SEED"
LOG=logs/deepring_reimpl$SFX
SEL=results/pdv2_deepring_reimpl$SFX/train_selection.json
P="python3 scripts/_deepring_reimpl.py"
NPROC=${NPROC:-8}
GPU=${GPU:-0}
mkdir -p "$LOG"
say() { echo "[deepring-reimpl] $* ($(date '+%F %T'))"; }

stage_check() {
  say check
  CUDA_VISIBLE_DEVICES= $P check > "$LOG/check_cpu.log" 2>&1
  tail -1 "$LOG/check_cpu.log"
}

stage_gpu-check() {
  say gpu-check
  CUDA_VISIBLE_DEVICES=$GPU $P gpu-check > "$LOG/check_gpu.log" 2>&1
  tail -1 "$LOG/check_gpu.log"
}

stage_bev() {
  say "bev ($NPROC parallel sequences)"
  $P jobs | CUDA_VISIBLE_DEVICES=$GPU xargs -r -P "$NPROC" -L 1 bash -c \
    'python3 scripts/_deepring_reimpl.py bev --role "$0" --seq "$1" \
       >> "'"$LOG"'/bev_$0_${1//\//_}.log" 2>&1 || echo "BEV_FAILED $0 $1"'
  $P status | tail -1
}

stage_train() {
  $P status | grep -q BEV_COMPLETE || { echo "BEV caches incomplete"; exit 1; }
  if [ -f "$SEL" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$SEL')).get('finished') else 1)"; then
    say "train: finished, skip"; return
  fi
  say "train (resumes from the last epoch checkpoint if any)"
  CUDA_VISIBLE_DEVICES=$GPU $P train >> "$LOG/train.log" 2>&1
  tail -1 "$LOG/train.log"
}

stage_dump() {
  say dump
  CUDA_VISIBLE_DEVICES=$GPU $P dump >> "$LOG/dump.log" 2>&1
  tail -1 "$LOG/dump.log"
}

stage_paths() {
  echo "SEED=$SEED LOG=$LOG SEL=$SEL"
  python3 -c "import sys; sys.path[:0] = ['scripts']
import _deepring_reimpl as M
r = lambda p: str(p.relative_to(M.REPO))
print('python SEED', M.SEED, 'RUN_DIR', r(M.RUN_DIR), 'RESULTS', r(M.RESULTS),
      'SELECTION', r(M.SELECTION), 'BEV_ROOT (shared)', r(M.BEV_ROOT))"
}

stage_aggregate() {
  say aggregate
  $P aggregate > "$LOG/aggregate.log" 2>&1
  tail -2 "$LOG/aggregate.log"
}

[ $# -gt 0 ] || set -- gpu-check bev train dump aggregate
for s in "$@"; do
  "stage_$s"
done
say "DONE: $*"
