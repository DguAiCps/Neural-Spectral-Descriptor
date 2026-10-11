#!/usr/bin/env bash
# C3 chain on split v2 (2026-10-07): odometry-keyframe, past-only NSD/NSD-H.
#   GPU (background): causal retrain seed 1 -> causal edge classifier
#   CPU            : KISS-ICP odometry (val + test) -> odometry-keyframe caches
#   then, per condition (odom = odometry keyframes, gt = GT keyframes):
#   val tuple grid -> tuple selection -> val ledger -> confidence + tau*
#   -> test ledger -> aggregate.
# Every step skips finished work. Run inside ring_baseline:
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_c3_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs results/pdv2_c3
P="python3 scripts/_pdv2_c3.py"
TRAIN_SEQS="kitti/01 kitti/02 kitti/05 kitti/06 helipr/Bridge01 helipr/Bridge02 helipr/Bridge03 helipr/Bridge04"
VAL_TAGS="K09 K3604 K3606 RA1 RA2 RA3 SJ1 SJ2 SJ3"

run() {
  local log="$1"; shift
  echo "[c3] === $* ($(date '+%F %T'))"
  "$@" 2>&1 | tee "logs/${log}.log"
}
tag() { echo "$1" | tr / _; }

# ---- GPU branch (background): causal retrain + causal edge classifier ----
(
  set -euo pipefail
  if [ ! -f checkpoints/pdv2_causal_s1/best_model.pth ]; then
    echo "[c3-gpu] train ($(date '+%F %T'))"
    python3 train_multi_dataset.py \
      --config configs/training_place_disjoint_v2_causal.yaml \
      --data-root /workspace/data \
      --encoder-preset no_interdiff \
      --use-gated-context \
      --gate-initial-alpha 0.0625 \
      --checkpoint-dir checkpoints/pdv2_causal_s1 \
      --seed 1 > logs/retrain_pdv2_causal_s1.log 2>&1
  fi
  if [ ! -f artifacts/edge_classifier_pdv2_causal.pt ]; then
    for sq in $TRAIN_SEQS; do
      echo "[c3-gpu] clf features $sq ($(date '+%F %T'))"
      $P clf --only "$sq" > "logs/c3_clf_$(tag "$sq").log" 2>&1
    done
    echo "[c3-gpu] clf train ($(date '+%F %T'))"
    $P clf --train-only > logs/c3_clf.log 2>&1
  fi
  echo "[c3-gpu] GPU BRANCH DONE ($(date '+%F %T'))"
) > logs/c3_gpu_branch.log 2>&1 &
GPU_PID=$!

# ---- CPU branch: odometry, then odometry-keyframe caches ----
SEQS_ALL="$($P roster --role val) $($P roster --role test)"
echo "[c3] odometry for $(echo $SEQS_ALL | wc -w) sequences ($(date '+%F %T'))"
echo $SEQS_ALL | tr ' ' '\n' | xargs -P 6 -I{} sh -c \
  'python3 scripts/_pdv2_c3.py odometry --seq {} > logs/c3_odom_$(echo {} | tr / _).log 2>&1 || { echo "ODOM_FAILED {}"; exit 1; }'
run c3_odom_report $P odom-report

echo "[c3] odometry-keyframe caches ($(date '+%F %T'))"
{ for sq in $($P roster --role val); do echo "val $sq"; done
  for sq in $($P roster --role test); do echo "test $sq"; done; } | \
  xargs -P 4 -L 1 sh -c \
  'python3 scripts/_pdv2_c3.py caches --role "$0" --seq "$1" > logs/c3_cache_$0_$(echo $1 | tr / _).log 2>&1 || { echo "CACHE_FAILED $0 $1"; exit 1; }'

echo "[c3] waiting for GPU branch ($(date '+%F %T'))"
wait "$GPU_PID" || { echo "[c3] GPU BRANCH FAILED (logs/c3_gpu_branch.log)"; exit 1; }
tail -1 logs/c3_gpu_branch.log

# ---- selection and test, per condition ----
for cond in odom gt; do
  for t in $VAL_TAGS; do
    run "c3_${cond}_valtuples_${t}" $P val-tuples --cond "$cond" --tag "$t"
  done
  run "c3_${cond}_select" $P select-tuple --cond "$cond"
  for t in $VAL_TAGS; do
    run "c3_${cond}_valconf_${t}" $P val-conf --cond "$cond" --tag "$t"
  done
  run "c3_${cond}_confselect" $P conf-select --cond "$cond"
  for sq in $($P roster --role test); do
    run "c3_${cond}_dump_$(tag "$sq")" $P test-dump --cond "$cond" --seq "$sq"
  done
  run "c3_${cond}_aggregate" $P aggregate --cond "$cond"
done
echo "[c3] C3 CHAIN COMPLETE ($(date '+%F %T'))"
