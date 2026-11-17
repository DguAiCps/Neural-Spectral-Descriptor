#!/usr/bin/env bash
# Test stage of the SeqOT paper-recipe chain (run by the chain master, which
# holds the GPU0 lock): input reproduction check -> 3 seeds x 46 ledgers ->
# per-seed PR summaries -> 3-seed aggregation and verification.
set -euo pipefail
REPO=/home/qkrrlans00/Neural-Spectral-Codec
cd "$REPO"
T=scripts/_review_seqot_paper_recipe_20261107_test.py
L=_external/seqot_paper_recipe_20261107/logs
dx() {  # dx <seed> <args...>
  local s=$1; shift
  docker exec -w /workspace/Neural-Spectral-Codec -e CUDA_VISIBLE_DEVICES=0 -e SEED="$s" \
    -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e PYTHONUNBUFFERED=1 ring_baseline python3 "$T" "$@"
}
echo "$(date -u +%FT%TZ) test stage start"
dx 1 reproduce-old > "$L/test_reproduce_old.log" 2>&1
pids=()
for s in 1 2 3; do dx "$s" dump > "$L/test_dump_s$s.log" 2>&1 & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
for s in 1 2 3; do dx "$s" aggregate > "$L/test_aggregate_s$s.log" 2>&1; done
dx 1 final > "$L/test_final.log" 2>&1
echo "$(date -u +%FT%TZ) test stage done"
