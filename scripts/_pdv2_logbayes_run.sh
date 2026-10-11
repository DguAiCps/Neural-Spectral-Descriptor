#!/usr/bin/env bash
# C3 chain with the refitted key weights (log1p + Bayes diagonal, 2026-10-10),
# odometry condition: edge classifier -> val tuples -> tuple selection ->
# val ledger -> confidence (+ alignment candidates) -> test ledger -> aggregate.
# Refiner checkpoint unchanged (checkpoints/pdv2_causal_s1). Every step skips
# finished work. Run inside ring_baseline:
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_logbayes_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs/logbayes
P="python3 scripts/_pdv2_logbayes.py"
TRAIN_SEQS="kitti/01 kitti/02 kitti/05 kitti/06 helipr/Bridge01 helipr/Bridge02 helipr/Bridge03 helipr/Bridge04"
VAL_TAGS="K09 K3604 K3606 RA1 RA2 RA3 SJ1 SJ2 SJ3"
COND=odom

run() {
  local log="$1"; shift
  echo "[lb] === $* ($(date '+%F %T'))"
  "$@" > "logs/logbayes/${log}.log" 2>&1
}
tag() { echo "$1" | tr / _; }

if [ ! -f artifacts/edge_classifier_pdv2_causal_logbayes.pt ]; then
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
run "${COND}_align_channels" $P align-channels --cond $COND
run "${COND}_confselect_align" $P conf-select-align --cond $COND
run "${COND}_aggregate_align" $P aggregate-align --cond $COND
echo "[lb] LOGBAYES CHAIN COMPLETE ($(date '+%F %T'))"
