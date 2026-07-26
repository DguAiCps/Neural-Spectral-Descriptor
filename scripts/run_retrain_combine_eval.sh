#!/usr/bin/env bash
set -euo pipefail

# ABLATION-ONLY RUNNER (not the paper-main configuration).
#
# This script reproduces the sensor-aware GAT + physics3 BEV ablation chain
# reported as negative results in the paper appendix:
#   - "Sensor-aware GAT ablation (negative result)"
#   - "Physics-aware BEV ablation (KITTI 08 trade-off)"
#
# The PAPER-MAIN configuration is:
#   - configs/training_multi_dataset.yaml
#   - fixed-alpha gated GAT (gate_initial_alpha=0.0625, NO sensor-aware gate)
#   - max-only BEV phase-alignment sketch (NOT physics3)
#   - closed-form cyclic-shift cosine reranker (Eq. 6)
#
# Sequential NSD upgrade experiment (ablation):
#   1. retrain 416D sensor-aware GAT retrieval key
#   2. evaluate GAT-only and GAT + physics3 phase sketch
#
# Run from repo root:
#   bash scripts/run_retrain_combine_eval.sh

SEED="${SEED:-1}"
DEVICE="${DEVICE:-cuda}"
RUN_TESTS="${RUN_TESTS:-0}"
RUN_NCLT="${RUN_NCLT:-1}"

ROOT_KITTI="${ROOT_KITTI:-/workspace/data/kitti/dataset}"
ROOT_NCLT="${ROOT_NCLT:-/workspace/data/nclt}"

CONFIG="${CONFIG:-configs/training_multi_dataset_sensor_gat_absdiff.yaml}"
ENCODER_PRESET="${ENCODER_PRESET:-no_interdiff}"
GATE_ALPHA="${GATE_ALPHA:-0.0625}"

RUN_ID="${RUN_ID:-sensor_gat_absdiff_physics3_seed${SEED}}"
GAT_DIR="${GAT_DIR:-results/train_${RUN_ID}}"
GAT_CKPT="${GAT_CKPT:-${GAT_DIR}/best_model.pth}"

KITTI_CACHE="${KITTI_CACHE:-data/preprocessed_kitti_${RUN_ID}}"
KITTI_BEV_CACHE="${KITTI_BEV_CACHE:-data/preprocessed_kitti_bev_${RUN_ID}}"
NCLT_CACHE="${NCLT_CACHE:-data/preprocessed_nclt_${RUN_ID}}"
NCLT_BEV_CACHE="${NCLT_BEV_CACHE:-data/preprocessed_nclt_bev_${RUN_ID}}"

mkdir -p results logs "${GAT_DIR}"

if [[ "${RUN_TESTS}" == "1" ]]; then
  python3 -m py_compile \
    train_multi_dataset.py \
    scripts/evaluate_kitti_checkpoint.py \
    scripts/evaluate_nclt_checkpoint.py \
    src/encoding/bev_image.py \
    src/encoding/spectral_encoder.py \
    src/gnn/model.py \
    src/keyframe/graph_manager.py
  pytest -q tests/test_gnn_gate.py tests/test_cross_spectrum.py tests/test_phase_coherence.py
fi

echo "[1/5] Retrain 416D sensor-aware GAT retrieval key"
python3 train_multi_dataset.py \
  --config "${CONFIG}" \
  --checkpoint-dir "${GAT_DIR}" \
  --encoder-preset "${ENCODER_PRESET}" \
  --use-gated-context \
  --gate-initial-alpha "${GATE_ALPHA}" \
  --seed "${SEED}" \
  2>&1 | tee "logs/train_${RUN_ID}.log"

echo "[2/5] Evaluate GAT-only retrieval key on KITTI 00/05/08"
python3 scripts/evaluate_kitti_checkpoint.py \
  --config "${CONFIG}" \
  --encoder-preset "${ENCODER_PRESET}" \
  --checkpoint "${GAT_CKPT}" \
  --use-gated-context \
  --gate-initial-alpha "${GATE_ALPHA}" \
  --root "${ROOT_KITTI}" \
  --sequences 00 05 08 \
  --cache-dir "${KITTI_CACHE}" \
  --device "${DEVICE}" \
  --n-coarse 800 \
  --output "results/kitti_gat_only_${RUN_ID}.json" \
  2>&1 | tee "logs/eval_kitti_gat_only_${RUN_ID}.log"

echo "[3/5] Evaluate 800D analytic state: 416D GAT key + 384D physics3 phase sketch"
python3 scripts/evaluate_kitti_checkpoint.py \
  --config "${CONFIG}" \
  --encoder-preset "${ENCODER_PRESET}" \
  --checkpoint "${GAT_CKPT}" \
  --use-gated-context \
  --gate-initial-alpha "${GATE_ALPHA}" \
  --root "${ROOT_KITTI}" \
  --sequences 00 05 08 \
  --cache-dir "${KITTI_CACHE}" \
  --bev-cache-dir "${KITTI_BEV_CACHE}" \
  --device "${DEVICE}" \
  --n-coarse 800 \
  --enable-bev-layout \
  --bev-height-encoding physics3 \
  --bev-row-pool 48 \
  --enable-phase-sketch \
  --phase-sketch-only \
  --phase-range-freqs 0 \
  --phase-bev-freqs 4 \
  --phase-rerank-mode sketch_fft \
  --phase-sketch-bev-weights 0.5 1.0 2.0 4.0 8.0 \
  --phase-sketch-range-weights 0.0 \
  --output "results/kitti_physics3_sketch_${RUN_ID}.json" \
  2>&1 | tee "logs/eval_kitti_physics3_sketch_${RUN_ID}.log"

python3 scripts/summarize_retrain_combine_eval.py \
  --run-id "${RUN_ID}" \
  --kitti-gat "results/kitti_gat_only_${RUN_ID}.json" \
  --kitti-sketch "results/kitti_physics3_sketch_${RUN_ID}.json" \
  --output "results/summary_${RUN_ID}.json"

echo "Done. Summary: results/summary_${RUN_ID}.json"
