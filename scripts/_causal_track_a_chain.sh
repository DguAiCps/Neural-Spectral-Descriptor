#!/usr/bin/env bash
# Track A chain for the pre-registered causal PR campaign
# (docs/CAUSAL_PR_PREREGISTRATION.md §2.3, §3-A). Run AFTER the causal
# retrain chain completes (checkpoints/pd_causal_s{1,2,3}). Resumable:
# per-sequence artifacts/JSONs are skipped when present; one heavy sequence
# per process (memory-cgroup discipline, as in _pd_run_stack.sh).
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_causal_track_a_chain.sh
# Order: causal edge-classifier features+train for BOTH tracks (pd_causal =
# retrained, pd = Frozen-GNN adaptation) -> val tuple grid (2 variants x 3
# seeds) -> tuple selection freeze -> confidence/tau* selection per variant.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs results/pd2026_pr

for s in 1 2 3; do
  [ -f "checkpoints/pd_causal_s${s}/best_model.pth" ] || {
    echo "[trackA] missing checkpoints/pd_causal_s${s}/best_model.pth — run scripts/_run_causal_retrain.sh first"; exit 1; }
done

run() {
  local log="$1"; shift
  echo "[trackA] === $* ($(date '+%F %T'))"
  "$@" 2>&1 | tee "logs/${log}.log"
}

PD_SEQS="kitti/01 kitti/02 kitti/06 kitti/07 \
  nclt/2012-05-11 nclt/2012-08-04 nclt/2012-11-04 nclt/2012-11-16 nclt/2013-02-23 \
  helipr/Bridge01 helipr/Bridge02 helipr/Bridge03 helipr/Roundabout01 helipr/Roundabout02 \
  mulran/Sejong01 mulran/Sejong02"

echo "[trackA] ---- stage 1: causal edge-classifier (variant pd_causal) ----"
for sq in ${PD_SEQS}; do
  run "causal_clf_pdc_$(echo "$sq" | tr / _)" \
    python3 scripts/_causal_build_edge_classifier.py --variant pd_causal --only "$sq"
done
run causal_clf_pdc_train python3 scripts/_causal_build_edge_classifier.py --variant pd_causal --train-only

echo "[trackA] ---- stage 2: causal edge-classifier (variant pd, Frozen-GNN) ----"
for sq in ${PD_SEQS}; do
  run "causal_clf_pd_$(echo "$sq" | tr / _)" \
    python3 scripts/_causal_build_edge_classifier.py --variant pd --only "$sq"
done
run causal_clf_pd_train python3 scripts/_causal_build_edge_classifier.py --variant pd --train-only

echo "[trackA] ---- stage 3: val tuple grid (2 variants x 3 seeds) ----"
for v in pd_causal pd; do
  for s in 1 2 3; do
    run "causal_tuples_${v}_s${s}" \
      python3 scripts/_causal_val_select.py tuples --variant "$v" --seed "$s"
  done
done

echo "[trackA] ---- stage 4: tuple selection (freeze) ----"
run causal_select_tuple python3 scripts/_causal_val_select.py select-tuple

echo "[trackA] ---- stage 5: confidence statistic + tau* (per variant) ----"
for v in pd_causal pd; do
  for s in 1 2 3; do
    run "causal_conf_${v}_s${s}" \
      python3 scripts/_causal_val_select.py confidence --variant "$v" --seed "$s"
  done
done
echo "[trackA] TRACK A CHAIN COMPLETE ($(date '+%F %T'))"
