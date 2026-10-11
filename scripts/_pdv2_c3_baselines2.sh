#!/usr/bin/env bash
# C3 baselines, second batch (2026-10-08): BEVPlace++ and RING#-L retrained
# on split v2 (seed 1), past-only ledgers on the odom and gt keyframe caches.
#   branch BEV  (background): train (ImageNet init, 40 epochs, best v2-val
#                             epoch) -> ledgers odom, gt
#   branch RING             : train (official trainer, 5 epochs) -> v2-val per
#                             epoch -> selection -> ledgers odom, gt
#   then                    : aggregate per condition
# Every step skips finished work. Run inside ring_baseline:
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_c3_baselines2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs
P="python3 scripts/_pdv2_c3.py"
TEST="$($P roster --role test)"
VAL="$($P roster --role val)"

run() {
  local log="$1"; shift
  echo "[c3b2] === $* ($(date '+%F %T'))"
  "$@" 2>&1 | tee "logs/${log}.log"
}
tag() { echo "$1" | tr / _; }

(
  set -euo pipefail
  echo "[c3b2-bev] train ($(date '+%F %T'))"
  $P bev-train > logs/c3b2_bev_train.log 2>&1
  for cond in odom gt; do
    for sq in $TEST; do
      $P bev-dump --cond "$cond" --seq "$sq" > "logs/c3b2_${cond}_bev_$(tag "$sq").log" 2>&1
    done
    echo "[c3b2-bev] dumps $cond done ($(date '+%F %T'))"
  done
  echo "[c3b2-bev] BEV BRANCH DONE ($(date '+%F %T'))"
) > logs/c3b2_bev_branch.log 2>&1 &
BEV_PID=$!

run c3b2_ring_prep $P ring-prep
run c3b2_ring_train $P ring-train
for e in 1 2 3 4 5; do
  for sq in $VAL; do
    run "c3b2_ring_val_e${e}_$(tag "$sq")" $P ring-val --epoch "$e" --seq "$sq"
  done
done
run c3b2_ring_select $P ring-select
for cond in odom gt; do
  for sq in $TEST; do
    run "c3b2_${cond}_ring_$(tag "$sq")" $P ring-dump --cond "$cond" --seq "$sq"
  done
done

wait "$BEV_PID" || { echo "[c3b2] BEV BRANCH FAILED (logs/c3b2_bev_branch.log)"; exit 1; }
tail -1 logs/c3b2_bev_branch.log
for cond in odom gt; do
  run "c3b2_${cond}_aggregate" $P aggregate-baselines --cond "$cond"
done
echo "[c3b2] C3 BASELINES2 COMPLETE ($(date '+%F %T'))"
