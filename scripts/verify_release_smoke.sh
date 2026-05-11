#!/usr/bin/env bash
set -euo pipefail

cleanup_bytecode() {
  find . -type d -name "__pycache__" -prune -exec rm -rf {} +
  rm -rf .pytest_cache
}

trap cleanup_bytecode EXIT

required_files=(
  RELEASE_800D.md
  DATA.md
  EXPERIMENT_HANDOFF.md
  README.md
  RELEASE_CHECKLIST.md
  setup.py
  configs/README.md
  artifacts/MANIFEST.md
  src/data/kitti_loader.py
  src/data/mulran_loader.py
  src/data/multi_dataset_loader.py
  src/data/pose_utils.py
  src/gnn/learned_reranker.py
  scripts/evaluate_kitti_checkpoint.py
  scripts/evaluate_nclt_checkpoint.py
  scripts/train_kitti_learned_reranker.py
  scripts/run_paper_kitti_closed_form.sh
  scripts/run_paper_kitti_residual.sh
  scripts/run_paper_nclt_physics3_control.sh
  scripts/restore_release_artifacts.sh
  scripts/make_release_bundle.sh
)

for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing required release file: ${path}" >&2
    exit 1
  fi
done

for script in \
  scripts/run_paper_kitti_closed_form.sh \
  scripts/run_paper_kitti_residual.sh \
  scripts/run_paper_nclt_physics3_control.sh \
  scripts/restore_release_artifacts.sh \
  scripts/make_release_bundle.sh; do
  bash -n "${script}"
done

if grep -R -n -E "0\\.9496|zero-shot performance|older README|Current Sequential|current main upgrade|GAT learns phase|NSD full|full NSD" \
  README.md RELEASE_800D.md DATA.md EXPERIMENT_HANDOFF.md configs/README.md artifacts/MANIFEST.md >/tmp/nsd_release_stale.txt; then
  cat /tmp/nsd_release_stale.txt >&2
  echo "stale release wording found" >&2
  exit 1
fi

python3 -m py_compile \
  train_multi_dataset.py \
  scripts/evaluate_kitti_checkpoint.py \
  scripts/evaluate_nclt_checkpoint.py \
  scripts/evaluate_nclt_learned_reranker.py \
  scripts/train_kitti_learned_reranker.py \
  src/data/kitti_loader.py \
  src/data/mulran_loader.py \
  src/data/multi_dataset_loader.py \
  src/data/pose_utils.py \
  src/encoding/bev_image.py \
  src/encoding/spectral_encoder.py \
  src/gnn/model.py \
  src/gnn/learned_reranker.py \
  src/keyframe/graph_manager.py

if [[ "${RUN_TESTS:-0}" == "1" ]]; then
  PYTHONPATH=src pytest -q \
    tests/test_cross_spectrum.py \
    tests/test_gnn_gate.py \
    tests/test_phase_alignment.py \
    tests/test_phase_coherence.py
fi

echo "release smoke check passed"
