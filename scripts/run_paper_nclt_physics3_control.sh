#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-results/train_kitti_nclt_nointerdiff288_gate00625_seed1/best_model.pth}"
CONFIG="${CONFIG:-configs/training_kitti_nclt_compact_fast.yaml}"
NCLT_CACHE="${NCLT_CACHE:-data/preprocessed_nclt_kitti_nclt_compact}"
NCLT_BEV_CACHE="${NCLT_BEV_CACHE:-data/preprocessed_nclt_bev_kitti_nclt_compact}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint not found: ${CHECKPOINT}" >&2
  echo "Set CHECKPOINT=/path/to/KITTI+NCLT best_model.pth" >&2
  exit 1
fi

python3 scripts/evaluate_nclt_checkpoint.py \
  --config "${CONFIG}" \
  --encoder-preset no_interdiff \
  --use-gated-context --gate-initial-alpha 0.0625 \
  --checkpoint "${CHECKPOINT}" \
  --dates 2012-01-08 2013-01-10 \
  --device "${DEVICE}" \
  --cache-dir "${NCLT_CACHE}" \
  --bev-cache-dir "${NCLT_BEV_CACHE}" \
  --sensor-key nclt \
  --elevation-range -30.67 10.67 \
  --scan-stride 5 \
  --skip-frames 6 \
  --n-coarse 800 \
  --bev-height-encoding max \
  --bev-row-pool 16 \
  --phase-range-freqs 0 \
  --phase-bev-freqs 12 \
  --output results/nclt_kitti_nclt_maxbev384_n800.json

python3 scripts/evaluate_nclt_checkpoint.py \
  --config "${CONFIG}" \
  --encoder-preset no_interdiff \
  --use-gated-context --gate-initial-alpha 0.0625 \
  --checkpoint "${CHECKPOINT}" \
  --dates 2012-01-08 2013-01-10 \
  --device "${DEVICE}" \
  --cache-dir "${NCLT_CACHE}" \
  --bev-cache-dir "${NCLT_BEV_CACHE}_physics3" \
  --sensor-key nclt \
  --elevation-range -30.67 10.67 \
  --scan-stride 5 \
  --skip-frames 6 \
  --n-coarse 800 \
  --bev-height-encoding physics3 \
  --bev-row-pool 48 \
  --phase-range-freqs 0 \
  --phase-bev-freqs 4 \
  --output results/nclt_kitti_nclt_physics3_384_n800.json

