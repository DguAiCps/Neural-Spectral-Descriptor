#!/usr/bin/env bash
# Frozen test phase for the place-disjoint protocol (2026-08-04).
# Stage 1: build test caches (checkpoint-free encoder preprocessing),
#          one sequence per process (memory-cgroup discipline).
# Stage 2: frozen-tuple APPLY evaluation, 2 variants x 3 seeds x all sequences,
#          one sequence per process. No selection happens anywhere here.
# Run inside ring_baseline:
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pd_run_test_phase.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs results/pd2026

SEQS=$(python3 -c "
import json
m = json.load(open('scripts/_pd_test_cache_manifest.json'))
print(' '.join(f\"{s['sensor']}/{s['sequence']}\" for s in m['sequences']))
")

run() {
  local log="$1"; shift
  echo "[test-phase] === $* ($(date '+%T'))"
  "$@" 2>&1 | tee "logs/${log}.log"
}

echo "[test-phase] stage 1: caches for $(echo "$SEQS" | wc -w) sequences"
for sq in $SEQS; do
  run "testcache_$(echo "$sq" | tr / _)" \
    python3 scripts/_pd_build_test_caches.py --seqs "$sq"
done

echo "[test-phase] stage 2: frozen apply (pd,pdnn x seeds 1-3)"
for v in pd pdnn; do
  for s in 1 2 3; do
    for sq in $SEQS; do
      run "testapply_${v}_s${s}_$(echo "$sq" | tr / _)" \
        python3 scripts/_pd_apply_frozen.py --variant "$v" --seed "$s" --seqs "$sq"
    done
  done
done
echo "[test-phase] TEST PHASE COMPLETE ($(date '+%F %T'))"
