#!/usr/bin/env bash
# Place-disjoint retrain chain (2026-08-04): 2 configs x 3 seeds, sequential
# (single 16GB GPU). Skips runs whose best_model.pth already exists, so the
# chain is resumable. Run inside the ring_baseline container:
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_run_place_disjoint_retrain.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs

for cfg_pair in "training_place_disjoint:pd" "training_place_disjoint_no_nclt:pdnn"; do
  cfg="${cfg_pair%%:*}"
  tag="${cfg_pair##*:}"
  for seed in 1 2 3; do
    ckdir="checkpoints/${tag}_s${seed}"
    if [ -f "${ckdir}/best_model.pth" ]; then
      echo "[chain] skip ${ckdir} (best_model.pth exists)"
      continue
    fi
    echo "[chain] === ${cfg} seed ${seed} -> ${ckdir} ($(date '+%F %T'))"
    python3 train_multi_dataset.py \
      --config "configs/${cfg}.yaml" \
      --data-root /workspace/data \
      --encoder-preset no_interdiff \
      --use-gated-context \
      --gate-initial-alpha 0.0625 \
      --checkpoint-dir "${ckdir}" \
      --seed "${seed}" 2>&1 | tee "logs/retrain_${tag}_s${seed}.log"
  done
done
echo "[chain] ALL RUNS COMPLETE ($(date '+%F %T'))"
