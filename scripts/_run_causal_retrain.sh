#!/usr/bin/env bash
# Causal retrain chain (pre-registered, docs/CAUSAL_PR_PREREGISTRATION.md §2.2):
# 3 seeds of the full-causal pd configuration (temporal past_to_current with
# past half-window 10 + prefix-only similarity edges emitted past->current).
# Mirrors scripts/_run_place_disjoint_retrain.sh; resumable via best_model.pth.
# Run inside the ring_baseline container:
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_run_causal_retrain.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs

for seed in 1 2 3; do
  ckdir="checkpoints/pd_causal_s${seed}"
  if [ -f "${ckdir}/best_model.pth" ]; then
    echo "[causal-chain] skip ${ckdir} (best_model.pth exists)"
    continue
  fi
  echo "[causal-chain] === causal seed ${seed} -> ${ckdir} ($(date '+%F %T'))"
  python3 train_multi_dataset.py \
    --config configs/training_place_disjoint_causal.yaml \
    --data-root /workspace/data \
    --encoder-preset no_interdiff \
    --use-gated-context \
    --gate-initial-alpha 0.0625 \
    --checkpoint-dir "${ckdir}" \
    --seed "${seed}" 2>&1 | tee "logs/retrain_causal_s${seed}.log"
done
echo "[causal-chain] ALL CAUSAL RUNS COMPLETE ($(date '+%F %T'))"
