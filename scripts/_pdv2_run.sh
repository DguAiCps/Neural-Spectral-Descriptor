#!/usr/bin/env bash
# Split v2 PILOT chain (2026-10-07): NSD / NSD-H, seed 1 only.
# prepare -> val/test caches -> train -> omega -> edge classifier ->
# val branch fusion -> tuple selection -> frozen test apply -> aggregate.
# Every step skips finished work, so the chain is resumable. Run inside
# ring_baseline:
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs results/pdv2
P="python3 scripts/_pdv2_pilot.py"

run() {
  local log="$1"; shift
  echo "[pdv2] === $* ($(date '+%F %T'))"
  "$@" 2>&1 | tee "logs/${log}.log"
}
tag() { echo "$1" | tr / _; }

run pdv2_prepare $P prepare

# Checkpoint-free caches, one sequence per process (memory cgroup).
for sq in helipr/Roundabout01 helipr/Roundabout02 mulran/Sejong01 mulran/Sejong02; do
  run "pdv2_cache_val_$(tag "$sq")" $P caches --role val --seq "$sq"
done
for sq in kitti/07 nclt/2012-05-11 nclt/2012-08-04 nclt/2012-11-04 \
          nclt/2012-11-16 nclt/2013-01-10 nclt/2013-02-23; do
  run "pdv2_cache_test_$(tag "$sq")" $P caches --role test --seq "$sq"
done

if [ -f checkpoints/pdv2_s1/best_model.pth ]; then
  echo "[pdv2] skip training (checkpoints/pdv2_s1/best_model.pth exists)"
else
  run retrain_pdv2_s1 python3 train_multi_dataset.py \
    --config configs/training_place_disjoint_v2.yaml \
    --data-root /workspace/data \
    --encoder-preset no_interdiff \
    --use-gated-context \
    --gate-initial-alpha 0.0625 \
    --checkpoint-dir checkpoints/pdv2_s1 \
    --seed 1
fi

# Frozen once written: val fusion already done with them must not be
# re-scored against a retrained (GPU-nondeterministic) classifier.
if [ -f artifacts/key_remetrize_r_pdv2.npy ]; then
  echo "[pdv2] skip remetrize (artifacts/key_remetrize_r_pdv2.npy exists)"
else
  run pdv2_remetrize $P remetrize
fi
if [ -f artifacts/edge_classifier_pdv2.pt ]; then
  echo "[pdv2] skip edge classifier (artifacts/edge_classifier_pdv2.pt exists)"
else
  for sq in $($P roster --role train); do
    run "pdv2_clf_$(tag "$sq")" $P edgecls --only "$sq"
  done
  run pdv2_clf $P edgecls --train-only
fi

for t in K09 K3604 K3606 RA1 RA2 RA3 SJ1 SJ2 SJ3; do
  run "pdv2_fusion_${t}" $P fusion --tag "$t"
done
run pdv2_select $P select

for sq in $($P roster --role test); do
  run "pdv2_apply_$(tag "$sq")" $P apply --seq "$sq"
done
run pdv2_aggregate $P aggregate
echo "[pdv2] PDV2 CHAIN COMPLETE ($(date '+%F %T'))"
