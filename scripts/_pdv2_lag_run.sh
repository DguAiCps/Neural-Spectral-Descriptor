#!/usr/bin/env bash
# Fixed-lag (15 keyframes) chain on split v2 (2026-10-09), odometry condition:
#   train (configs/training_place_disjoint_v2_lag15.yaml, seed 1)
#   -> lag edge classifier on GT train caches
#   -> val tuple grid -> tuple selection -> val ledger -> confidence + tau*
#   -> test ledger (46) -> aggregate.
# Same steps and rules as scripts/_pdv2_c3_run.sh; only the graph differs.
# Every step skips finished work. Run inside ring_baseline:
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_lag_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs/lag15 results/pdv2_lag15
P="python3 scripts/_pdv2_lag.py"
TRAIN_SEQS="kitti/01 kitti/02 kitti/05 kitti/06 helipr/Bridge01 helipr/Bridge02 helipr/Bridge03 helipr/Bridge04"
VAL_TAGS="K09 K3604 K3606 RA1 RA2 RA3 SJ1 SJ2 SJ3"
COND=odom

run() {
  local log="$1"; shift
  echo "[lag] === $* ($(date '+%F %T'))"
  "$@" > "logs/lag15/${log}.log" 2>&1
}
tag() { echo "$1" | tr / _; }

if [ ! -f checkpoints/pdv2_lag15_s1/best_model.pth ] || \
   ! grep -q "TRAINING COMPLETE" logs/lag15/train.log 2>/dev/null; then
  run train python3 train_multi_dataset.py \
    --config configs/training_place_disjoint_v2_lag15.yaml \
    --data-root /workspace/data \
    --encoder-preset no_interdiff \
    --use-gated-context \
    --gate-initial-alpha 0.0625 \
    --checkpoint-dir checkpoints/pdv2_lag15_s1 \
    --seed 1
fi
if [ ! -f artifacts/edge_classifier_pdv2_lag15.pt ]; then
  for sq in $TRAIN_SEQS; do
    run "clf_$(tag "$sq")" $P clf --only "$sq"
  done
  run clf_train $P clf --train-only
fi
for t in $VAL_TAGS; do
  run "${COND}_valtuples_${t}" $P val-tuples --cond $COND --tag "$t"
done
run "${COND}_select" $P select-tuple --cond $COND
for t in $VAL_TAGS; do
  run "${COND}_valconf_${t}" $P val-conf --cond $COND --tag "$t"
done
run "${COND}_confselect" $P conf-select --cond $COND
for sq in $(python3 scripts/_pdv2_c3.py roster --role test); do
  run "${COND}_dump_$(tag "$sq")" $P test-dump --cond $COND --seq "$sq"
done
run "${COND}_aggregate" $P aggregate --cond $COND
echo "[lag] LAG15 CHAIN COMPLETE ($(date '+%F %T'))"
