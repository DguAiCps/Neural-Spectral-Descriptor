#!/usr/bin/env bash
set -euo pipefail

# tab:transfer (leave-one-sensor-in) campaign, seed 1.
#
# Trains three single-sensor models at the deployed configuration (only
# data.datasets.train/val differ from _abl_deployed_v0.yaml; early stopping on
# own-sensor val), then evaluates the raw 416D key of the three models plus the
# frozen joint seed-1 baseline on the 6-sequence KITTI+NCLT+HeLiPR subset via
# scripts/_eval_deployed_ablation.py (results merge into
# results/transfer_eval.json).
#
# Run inside nvcr.io/nvidia/pyg:26.01-py3 from the repo root. Resume-safe:
# existing best_model.pth is skipped (FORCE=1 to retrain).

SEED="${SEED:-1}"
SENSORS="${SENSORS:-kitti nclt helipr}"
FORCE="${FORCE:-0}"

mkdir -p logs results

for s in ${SENSORS}; do
  ckpt="checkpoints/transfer_${s}/best_model.pth"
  if [[ -f "${ckpt}" && "${FORCE}" != "1" ]]; then
    echo "[skip] transfer_${s}: ${ckpt} already exists (FORCE=1 to retrain)"
    continue
  fi
  echo "=== [train] transfer_${s} (seed ${SEED}) ==="
  python3 train_multi_dataset.py \
    --config "configs/_transfer_${s}.yaml" \
    --encoder-preset no_interdiff \
    --use-gated-context \
    --gate-initial-alpha 0.0625 \
    --checkpoint-dir "checkpoints/transfer_${s}" \
    --seed "${SEED}" \
    2>&1 | tee "logs/transfer_${s}.log"
done

echo "=== [eval] baseline + transfer rows ==="
python3 scripts/_eval_deployed_ablation.py baseline t_kitti t_nclt t_helipr \
  --out results/transfer_eval.json \
  2>&1 | tee logs/transfer_eval.log

echo "Done. Results: results/transfer_eval.json"
