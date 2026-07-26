#!/usr/bin/env bash
# GPU-side baseline latency bench: BEVPlace++ (ms-ft) + SeqOT.
# Run when the GPU is EXCLUSIVE (no training). The SC++ (CPU) part is run
# separately without --gpus; results merge into the same JSON:
#   results/baseline_latency_bench.json
#
# Usage:  bash scripts/_bench_baseline_latency.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker run --rm --gpus all --cpus 6 --memory 12g --shm-size 8g \
  -v "$REPO":/workspace/Neural-Spectral-Codec \
  -v /mnt/d/NSD_datasets:/workspace/data \
  -w /workspace/Neural-Spectral-Codec \
  nvcr.io/nvidia/pyg:26.01-py3 \
  sh -c 'python scripts/_bench_baseline_latency.py --methods bevplace seqot; s=$?; \
         chown 1000:1000 results/baseline_latency_bench.json 2>/dev/null; exit $s'
