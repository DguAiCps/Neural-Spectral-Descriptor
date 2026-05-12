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
#   - configs/training_multi_dataset.yaml (or training_kitti_only.yaml)
#   - fixed-alpha gated GAT (gate_initial_alpha=0.0625, NO sensor-aware gate)
#   - max-only BEV phase-alignment sketch (NOT physics3)
#   - closed-form cyclic-shift cosine rerank, plus zero-init residual reranker
#     trained via scripts/train_kitti_learned_reranker.py
#
# Sequential NSD upgrade experiment (ablation):
#   1. retrain 416D sensor-aware GAT retrieval key
#   2. evaluate GAT-only and GAT + physics3 phase sketch
#   3. train learned residual reranker on physics3 phase sketch
#   4. evaluate the combined 800D state on KITTI and NCLT zero-shot
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
RERANK_CKPT="${RERANK_CKPT:-results/kitti_learned_reranker_${RUN_ID}.pth}"
RERANK_JSON="${RERANK_JSON:-results/kitti_learned_reranker_${RUN_ID}.json}"

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
    scripts/evaluate_nclt_learned_reranker.py \
    scripts/train_kitti_learned_reranker.py \
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

echo "[4/5] Train learned residual reranker on physics3 phase sketch"
python3 scripts/train_kitti_learned_reranker.py \
  --config "${CONFIG}" \
  --encoder-preset "${ENCODER_PRESET}" \
  --checkpoint "${GAT_CKPT}" \
  --use-gated-context \
  --gate-initial-alpha "${GATE_ALPHA}" \
  --root "${ROOT_KITTI}" \
  --train-sequences 01 02 06 07 \
  --val-sequences 00 05 08 \
  --cache-dir "${KITTI_CACHE}" \
  --bev-cache-dir "${KITTI_BEV_CACHE}" \
  --device "${DEVICE}" \
  --bev-height-encoding physics3 \
  --bev-row-pool 48 \
  --bev-freqs 4 \
  --n-coarse 800 \
  --max-candidates 800 \
  --include-phase-candidates \
  --epochs "${RERANK_EPOCHS:-20}" \
  --batch-size "${RERANK_BATCH_SIZE:-8}" \
  --seed "${SEED}" \
  --output "${RERANK_JSON}" \
  --checkpoint-out "${RERANK_CKPT}" \
  2>&1 | tee "logs/train_reranker_${RUN_ID}.log"

if [[ "${RUN_NCLT}" == "1" ]]; then
  echo "[5/5] Evaluate combined 800D model on NCLT zero-shot"
  python3 scripts/evaluate_nclt_learned_reranker.py \
    --config "${CONFIG}" \
    --encoder-preset "${ENCODER_PRESET}" \
    --encoder-checkpoint "${GAT_CKPT}" \
    --reranker-checkpoint "${RERANK_CKPT}" \
    --use-gated-context \
    --gate-initial-alpha "${GATE_ALPHA}" \
    --root "${ROOT_NCLT}" \
    --dates 2012-01-08 2013-01-10 \
    --cache-dir "${NCLT_CACHE}" \
    --bev-cache-dir "${NCLT_BEV_CACHE}" \
    --device "${DEVICE}" \
    --bev-height-encoding physics3 \
    --bev-row-pool 48 \
    --bev-freqs 4 \
    --n-coarse 800 \
    --max-candidates 800 \
    --include-phase-candidates \
    --sensor-key nclt \
    --output "results/nclt_zero_shot_${RUN_ID}.json" \
    2>&1 | tee "logs/eval_nclt_zero_shot_${RUN_ID}.log"
else
  echo "[5/5] Skipped NCLT because RUN_NCLT=${RUN_NCLT}"
fi

python3 scripts/summarize_retrain_combine_eval.py \
  --run-id "${RUN_ID}" \
  --kitti-gat "results/kitti_gat_only_${RUN_ID}.json" \
  --kitti-sketch "results/kitti_physics3_sketch_${RUN_ID}.json" \
  --kitti-reranker "${RERANK_JSON}" \
  --nclt "results/nclt_zero_shot_${RUN_ID}.json" \
  --output "results/summary_${RUN_ID}.json"

echo "Done. Summary: results/summary_${RUN_ID}.json"
