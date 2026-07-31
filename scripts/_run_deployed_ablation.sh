#!/usr/bin/env bash
set -euo pipefail

# Deployed-config component-ablation campaign (tab:ablation re-run).
#
# Sequentially retrains the 5 component variants at the deployed 288D-key /
# 800D-state configuration (100 epochs each, ~1.6 h/run on the 16 GB GPU),
# then evaluates every variant's raw 416D key PLUS the frozen seed-1 baseline
# through the identical eval path (cosine R@1, 6-sequence KITTI+NCLT+HeLiPR
# subset, 4,845 queries, 5 m / 30-frame skip).
#
#   V1 standard attention (keys/values from absolute h_j)
#   V2 no edge bias (EdgeEncoder disabled)
#   V3 temporal edges only (similarity_max_k=0)
#   V4 similarity edges only (edge_type_filter=similarity_only)
#   V5 mining on refined f instead of raw d (mine_on_refined=true)
#
# Baseline row: checkpoints/800d_4sensor_20260511_161726 (NO retrain).
#
# Run inside nvcr.io/nvidia/pyg:26.01-py3 from the repo root:
#   docker run --rm --gpus all --shm-size=16g \
#     -v /mnt/d/NSD_datasets:/workspace/data \
#     -v <repo-root>:/workspace/Neural-Spectral-Codec \
#     -w /workspace/Neural-Spectral-Codec \
#     nvcr.io/nvidia/pyg:26.01-py3 \
#     bash scripts/_run_deployed_ablation.sh
#
# Already-trained variants (existing best_model.pth) are skipped, so the
# campaign is resume-safe; set FORCE=1 to retrain everything.

SEED="${SEED:-1}"
VARIANTS="${VARIANTS:-v1 v2 v3 v4 v5}"
FORCE="${FORCE:-0}"

mkdir -p logs results

for v in ${VARIANTS}; do
  ckpt="checkpoints/abl_deployed_${v}/best_model.pth"
  if [[ -f "${ckpt}" && "${FORCE}" != "1" ]]; then
    echo "[skip] ${v}: ${ckpt} already exists (FORCE=1 to retrain)"
    continue
  fi
  echo "=== [train] ${v} (seed ${SEED}) ==="
  python3 train_multi_dataset.py \
    --config "configs/_abl_deployed_${v}.yaml" \
    --encoder-preset no_interdiff \
    --use-gated-context \
    --gate-initial-alpha 0.0625 \
    --checkpoint-dir "checkpoints/abl_deployed_${v}" \
    --seed "${SEED}" \
    2>&1 | tee "logs/abl_deployed_${v}.log"
done

echo "=== [eval] baseline + ${VARIANTS} ==="
# shellcheck disable=SC2086
python3 scripts/_eval_deployed_ablation.py baseline ${VARIANTS} \
  2>&1 | tee logs/abl_deployed_eval.log

echo "Done. Results: results/deployed_ablation.json"
