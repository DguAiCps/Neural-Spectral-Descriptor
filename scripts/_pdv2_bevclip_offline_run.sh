#!/usr/bin/env bash
# Offline split-v2 pilot with the clipped-BEV NSD auxiliary block (fix A,
# 2026-10-10), seed 1: waits for scripts/_pdv2_bevclip_run.sh to finish
# (heavy steps run one at a time), then links the caches -> val fusion grid
# -> tuple selection -> test application -> aggregate. Every step skips
# finished work. Run inside ring_baseline:
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_bevclip_offline_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs/bevclip
P="python3 scripts/_pdv2_bevclip_offline.py"
VAL_TAGS="K09 K3604 K3606 RA1 RA2 RA3 SJ1 SJ2 SJ3"

run() {
  local log="$1"; shift
  echo "[bevclip-off] === $* ($(date '+%F %T'))"
  "$@" > "logs/bevclip/${log}.log" 2>&1
}
tag() { echo "$1" | tr / _; }

until grep -q "BEVCLIP CHAIN COMPLETE" logs/bevclip_chain.log 2>/dev/null; do
  sleep 120
done
run off_link $P link
for t in $VAL_TAGS; do
  run "off_fusion_${t}" $P fusion --tag "$t"
done
run off_select $P select
for sq in $(python3 scripts/_pdv2_pilot.py roster --role test); do
  run "off_apply_$(tag "$sq")" $P apply --seq "$sq"
done
run off_aggregate $P aggregate
echo "[bevclip-off] OFFLINE CHAIN COMPLETE ($(date '+%F %T'))"
