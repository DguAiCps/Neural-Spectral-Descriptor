#!/usr/bin/env bash
# Place-disjoint selection-stack chain (2026-08-04). Run inside ring_baseline:
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline bash scripts/_pd_run_stack.sh
# Waits for any in-flight val spec-cache build, then: train layouts ->
# edge classifiers -> B3 rerank -> branch fusion -> tuple selection.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs results/pd2026

echo "[stack] waiting for in-flight val spec-cache builds ($(date '+%T'))"
while ps aux | grep -q "[_]pd_build_val_spec_caches"; do sleep 60; done
for f in valspec_kitti_09 valspec_nclt_2012-01-08 valspec_helipr_Roundabout03 valspec_mulran_Sejong03; do
  [ -f "data/preprocessed_pd2026/${f}.npz" ] || { echo "[stack] MISSING ${f}.npz"; exit 1; }
done
echo "[stack] val spec caches complete ($(date '+%T'))"

run() {
  local log="$1"; shift
  echo "[stack] === $* ($(date '+%T'))"
  "$@" 2>&1 | tee "logs/${log}.log"
}

run pd_layouts        python3 scripts/_pd_build_layouts.py

# Edge-classifier features: one process per sequence (a monolithic run
# fragments its way past the ~28 GB memory cgroup and gets OOM-killed).
PD_SEQS="kitti/01 kitti/02 kitti/06 kitti/07 \
  nclt/2012-05-11 nclt/2012-08-04 nclt/2012-11-04 nclt/2012-11-16 nclt/2013-02-23 \
  helipr/Bridge01 helipr/Bridge02 helipr/Bridge03 helipr/Roundabout01 helipr/Roundabout02 \
  mulran/Sejong01 mulran/Sejong02"
for sq in ${PD_SEQS}; do
  run "pd_clf_pd_$(echo "$sq" | tr / _)" \
    python3 scripts/_pd_build_edge_classifier.py --variant pd --only "$sq"
done
run pd_clf_pd         python3 scripts/_pd_build_edge_classifier.py --variant pd --train-only
for sq in ${PD_SEQS}; do
  case "$sq" in nclt/*) continue;; esac
  run "pd_clf_pdnn_$(echo "$sq" | tr / _)" \
    python3 scripts/_pd_build_edge_classifier.py --variant pdnn --only "$sq"
done
run pd_clf_pdnn       python3 scripts/_pd_build_edge_classifier.py --variant pdnn --train-only

for s in s1 s2 s3; do
  run "pd_rerank_pd_${s}"   python3 scripts/_pd_b3_rerank.py --variant pd --seeds "$s"
  run "pd_rerank_pdnn_${s}" python3 scripts/_pd_b3_rerank.py --variant pdnn --seeds "$s"
done
for v in pd pdnn; do
  for s in 1 2 3; do
    run "pd_fusion_${v}_s${s}" python3 scripts/_pd_branch_fusion.py --variant "$v" --seed "$s"
  done
done
run pd_tuple_select   python3 scripts/_pd_tuple_select.py
echo "[stack] STACK COMPLETE ($(date '+%F %T'))"
