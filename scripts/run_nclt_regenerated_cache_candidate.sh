#!/usr/bin/env bash
# Build a fresh NCLT operating-cache candidate after the active single-GPU queue.
# This deliberately writes to a new directory: it never replaces a committed or
# historical cache.  A later Table-1/query-index validation decides whether the
# candidate can be used for paired statistics.
set -euo pipefail

repo=/workspace/Neural-Spectral-Codec
queue_pid=${1:?usage: $0 <active-gpu-queue-pid>}
candidate_tag=regenerated_candidate_20260730
cache_dir="data/preprocessed_nclt_${candidate_tag}"
bev_cache_dir="data/preprocessed_nclt_bev_${candidate_tag}"
result="results/supplementary/nclt_${candidate_tag}_seed1.json"
log="results/supplementary/logs/nclt_${candidate_tag}.log"

cd "$repo"
export NSD_DATA_ROOT=/workspace/data
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$(dirname "$log")"
if [[ -e "$cache_dir" || -e "$bev_cache_dir" || -e "$result" ]]; then
    echo "Refusing to reuse candidate paths; inspect or choose a new tag." >&2
    exit 2
fi

echo "Waiting for GPU queue PID ${queue_pid} at $(date -Is)" | tee -a "$log"
while kill -0 "$queue_pid" 2>/dev/null; do
    sleep 60
done

echo "Starting fresh NCLT cache candidate at $(date -Is)" | tee -a "$log"
python3 scripts/evaluate_nclt_checkpoint.py \
    --root "$NSD_DATA_ROOT/nclt" \
    --dates 2012-01-08 2013-01-10 \
    --config configs/training_multi_dataset.yaml \
    --encoder-preset no_interdiff \
    --checkpoint checkpoints/800d_4sensor_20260511_161726/best_model.pth \
    --use-gated-context \
    --gate-initial-alpha 0.0625 \
    --cache-dir "$cache_dir" \
    --bev-cache-dir "$bev_cache_dir" \
    --sensor-key nclt \
    --scan-stride 1 \
    --n-coarse 800 \
    --output "$result" \
    2>&1 | tee -a "$log"

touch "results/supplementary/nclt_${candidate_tag}.complete"
echo "Candidate complete at $(date -Is); Table-1/query-index validation required." | tee -a "$log"
