#!/usr/bin/env bash
# v1/v3 seed-2/3 replication for the deployed-config ablation (sign confirmation).
set -euo pipefail
mkdir -p logs
for s in 2 3; do
  for v in v1 v3; do
    ckpt="checkpoints/abl_deployed_${v}_s${s}/best_model.pth"
    if [[ -f "${ckpt}" ]]; then echo "[skip] ${v} seed${s}"; continue; fi
    echo "=== [train] ${v} seed ${s} ==="
    python3 train_multi_dataset.py \
      --config "configs/_abl_deployed_${v}.yaml" \
      --encoder-preset no_interdiff --use-gated-context --gate-initial-alpha 0.0625 \
      --checkpoint-dir "checkpoints/abl_deployed_${v}_s${s}" \
      --seed "${s}" 2>&1 | tee "logs/abl_deployed_${v}_s${s}.log"
  done
done
echo "seed replication training done"
