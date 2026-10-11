#!/usr/bin/env bash
# C3 baselines on split v2 (2026-10-08): SC++ (training-free) and SeqOT
# (retrained on v2 train, seed 1) under the past-only protocol, on the
# odometry-keyframe (odom) and GT-keyframe (gt) caches of scripts/_pdv2_c3.py.
#   GPU (background): SeqOT train data -> SeqOT train
#   CPU            : SC++ ledgers (odom, gt) and SeqOT test range stacks
#   then           : SeqOT ledgers (odom, gt) -> aggregate per condition
# Every step skips finished work. Run inside ring_baseline:
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_c3_baselines.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs
P="python3 scripts/_pdv2_c3.py"
TEST="$($P roster --role test)"

run() {
  local log="$1"; shift
  echo "[c3b] === $* ($(date '+%F %T'))"
  "$@" 2>&1 | tee "logs/${log}.log"
}
tag() { echo "$1" | tr / _; }

run c3b_seqot_train_data $P seqot-train-data
( $P seqot-train > logs/c3b_seqot_train.log 2>&1 \
    && echo "[c3b] SEQOT TRAIN DONE" || echo "[c3b] SEQOT TRAIN FAILED" ) &
TRAIN_PID=$!

for cond in odom gt; do
  for sq in $TEST; do
    run "c3b_${cond}_scpp_$(tag "$sq")" $P scpp-dump --cond "$cond" --seq "$sq"
  done
done
for cond in odom gt; do
  echo $TEST | tr ' ' '\n' | xargs -P 2 -I{} sh -c \
    "python3 scripts/_pdv2_c3.py seqot-prep --cond $cond --seq {} > logs/c3b_${cond}_seqotprep_\$(echo {} | tr / _).log 2>&1 || { echo PREP_FAILED $cond {}; exit 1; }"
done

wait "$TRAIN_PID"
grep -q "gem_latest" <(ls _external/seqot_runs_pdv2/s1) || { echo "[c3b] SeqOT weights missing"; exit 1; }
for cond in odom gt; do
  for sq in $TEST; do
    run "c3b_${cond}_seqot_$(tag "$sq")" $P seqot-dump --cond "$cond" --seq "$sq"
  done
  run "c3b_${cond}_aggregate" $P aggregate-baselines --cond "$cond"
done
echo "[c3b] C3 BASELINES COMPLETE ($(date '+%F %T'))"
