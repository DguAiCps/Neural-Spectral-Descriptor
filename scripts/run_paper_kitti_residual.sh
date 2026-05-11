#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1}"
CHECKPOINT="${CHECKPOINT:-results/train_no_interdiff_288_gate00625_seed1/best_model.pth}"
CONFIG="${CONFIG:-configs/training_kitti_only.yaml}"
OUTPUT="${OUTPUT:-results/kitti_learned_reranker_bev384_residual_seed${SEED}.json}"
RERANK_CKPT="${RERANK_CKPT:-results/kitti_learned_reranker_bev384_residual_seed${SEED}.pth}"
KITTI_CACHE="${KITTI_CACHE:-data/preprocessed_kitti_encoder_ablation_no_interdiff}"
KITTI_BEV_CACHE="${KITTI_BEV_CACHE:-data/preprocessed_kitti_bev_layout}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint not found: ${CHECKPOINT}" >&2
  echo "Restore it with the command in artifacts/MANIFEST.md or set CHECKPOINT=/path/to/best_model.pth" >&2
  exit 1
fi

python3 scripts/train_kitti_learned_reranker.py \
  --config "${CONFIG}" \
  --encoder-preset no_interdiff \
  --use-gated-context --gate-initial-alpha 0.0625 \
  --checkpoint "${CHECKPOINT}" \
  --train-sequences 01 02 06 07 \
  --val-sequences 00 05 08 \
  --cache-dir "${KITTI_CACHE}" \
  --bev-cache-dir "${KITTI_BEV_CACHE}" \
  --device "${DEVICE}" \
  --bev-height-encoding max \
  --bev-row-pool 16 \
  --bev-freqs 12 \
  --max-candidates 800 \
  --epochs 20 \
  --seed "${SEED}" \
  --output "${OUTPUT}" \
  --checkpoint-out "${RERANK_CKPT}"

