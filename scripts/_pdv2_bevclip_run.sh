#!/usr/bin/env bash
# Fix A chain (2026-10-10): clipped-BEV NSD auxiliary block, seeds 1-3.
#   1. clipped spectrum caches for the 55 val/test sequences, both keyframe
#      conditions (scripts/_bevclip_spec.py, CPU, 4 at a time, largest first)
#   2. wait for the seed 2 and 3 trainings
#   3. per seed, odometry condition: [edge classifier for seeds 2, 3] ->
#      val tuples -> tuple selection -> val ledger -> confidence + tau*
#      -> test ledger -> aggregate -> alignment statistics
#      (seed 1, then 2, then 3); then seed 1 in the GT-keyframe condition.
# Heavy chain steps run one at a time (MulRan Sejong uses most of the
# container memory). Every step skips finished work. Run inside ring_baseline:
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_bevclip_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs/bevclip
TRAIN_SEQS="kitti/01 kitti/02 kitti/05 kitti/06 helipr/Bridge01 helipr/Bridge02 helipr/Bridge03 helipr/Bridge04"
VAL_TAGS="K09 K3604 K3606 RA1 RA2 RA3 SJ1 SJ2 SJ3"

run() {
  local log="$1"; shift
  echo "[bevclip] === $* ($(date '+%F %T'))"
  "$@" > "logs/bevclip/${log}.log" 2>&1
}
tag() { echo "$1" | tr / _; }

echo "[bevclip] clipped spectra ($(date '+%F %T'))"
python3 - <<'EOF' | xargs -P 4 -I{} sh -c \
  'python3 scripts/_bevclip_spec.py --seq {} > logs/bevclip/spec_$(echo {} | tr / _).log 2>&1 || { echo "SPEC_FAILED {}"; exit 1; }'
import glob, os
import numpy as np
sizes = []
for f in glob.glob("data/preprocessed_v2_odom/*spec_*.npz"):
    _, sensor, q = os.path.basename(f)[:-4].split("_", 2)
    with np.load(f) as z:
        sizes.append((len(z["scan_ids"]), f"{sensor}/{q}"))
for _, seq in sorted(sizes, reverse=True):
    print(seq)
EOF
echo "[bevclip] clipped spectra done ($(date '+%F %T'))"

for s in 2 3; do
  until grep -q "Training complete" "logs/retrain_pdv2_causal_s${s}.log" 2>/dev/null; do
    sleep 60
  done
  test -f "checkpoints/pdv2_causal_s${s}/best_model.pth"
done
echo "[bevclip] trainings complete ($(date '+%F %T'))"

chain() {
  local s="$1" cond="$2"
  local P="python3 scripts/_pdv2_bevclip.py --seed $s"
  if [ "$s" != 1 ] && [ ! -f "artifacts/edge_classifier_pdv2_causal_s${s}.pt" ]; then
    for sq in $TRAIN_SEQS; do
      run "s${s}_clf_$(tag "$sq")" $P clf --only "$sq"
    done
    run "s${s}_clf_train" $P clf --train-only
  fi
  for t in $VAL_TAGS; do
    run "s${s}_${cond}_valtuples_${t}" $P val-tuples --cond "$cond" --tag "$t"
  done
  run "s${s}_${cond}_select" $P select-tuple --cond "$cond"
  for t in $VAL_TAGS; do
    run "s${s}_${cond}_valconf_${t}" $P val-conf --cond "$cond" --tag "$t"
  done
  run "s${s}_${cond}_confselect" $P conf-select --cond "$cond"
  for sq in $(python3 scripts/_pdv2_c3.py roster --role test); do
    run "s${s}_${cond}_dump_$(tag "$sq")" $P test-dump --cond "$cond" --seq "$sq"
  done
  run "s${s}_${cond}_aggregate" $P aggregate --cond "$cond"
  run "s${s}_${cond}_align_channels" $P align-channels --cond "$cond"
  run "s${s}_${cond}_confselect_align" $P conf-select-align --cond "$cond"
  run "s${s}_${cond}_aggregate_align" $P aggregate-align --cond "$cond"
  echo "[bevclip] SEED $s $cond COMPLETE ($(date '+%F %T'))"
}

chain 1 odom
chain 2 odom
chain 3 odom
chain 1 gt
echo "[bevclip] BEVCLIP CHAIN COMPLETE ($(date '+%F %T'))"
