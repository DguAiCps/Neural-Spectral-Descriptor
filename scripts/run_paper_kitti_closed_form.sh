#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-results/train_no_interdiff_288_gate00625_seed1/best_model.pth}"
CONFIG="${CONFIG:-configs/training_kitti_only.yaml}"
OUTPUT="${OUTPUT:-results/kitti_checkpoint_eval_nointerdiff288_gate00625_bevonly384_sketch_fft_n800.json}"
KITTI_CACHE="${KITTI_CACHE:-data/preprocessed_kitti_encoder_ablation_no_interdiff}"
KITTI_BEV_CACHE="${KITTI_BEV_CACHE:-data/preprocessed_kitti_bev_layout}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint not found: ${CHECKPOINT}" >&2
  echo "Restore it with the command in artifacts/MANIFEST.md or set CHECKPOINT=/path/to/best_model.pth" >&2
  exit 1
fi

python3 scripts/evaluate_kitti_checkpoint.py \
  --config "${CONFIG}" \
  --encoder-preset no_interdiff \
  --use-gated-context --gate-initial-alpha 0.0625 \
  --checkpoint "${CHECKPOINT}" \
  --sequences 00 05 08 \
  --device "${DEVICE}" \
  --cache-dir "${KITTI_CACHE}" \
  --bev-cache-dir "${KITTI_BEV_CACHE}" \
  --enable-bev-layout --bev-row-pool 16 --bev-height-encoding max \
  --enable-phase-sketch --phase-sketch-only \
  --phase-range-freqs 0 --phase-bev-freqs 12 \
  --phase-sketch-range-weights 0.0 \
  --phase-sketch-bev-weights 0.5 1.0 2.0 4.0 8.0 \
  --phase-rerank-mode sketch_fft \
  --n-coarse 800 \
  --output "${OUTPUT}"

