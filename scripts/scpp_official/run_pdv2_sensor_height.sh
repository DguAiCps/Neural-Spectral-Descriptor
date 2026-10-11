#!/usr/bin/env bash
# Official SC++ (A-PC, search_ratio 0) on the 46 split-v2 test sequences with a
# per-sensor LiDAR height instead of the official constant 1.9 m (user decision
# 2026-10-11): the official code adds the LiDAR height so the ground sits near
# z = 0; here that rule uses each rig's measured height.
#   heights: median over sequences of the 'lowest' ground estimate in
#   results/pdv2_ring_official/sensor_height.json (scripts/_ring_official_sensor_height.py).
#   NCLT: the upside-down HDL-32E sees too little near ground for that estimate
#   (0.14-0.35 m, implausible), so NCLT keeps the official 1.9 m.
# Runs on the HOST (MATLAB); calls run_pdv2.sh once per rig. Resumable.
#   bash scripts/scpp_official/run_pdv2_sensor_height.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export VARIANTS=apc SEARCH_RATIO=0 OUT=results/pdv2_scpp_official_hsens COND=odom
DOCKER="docker exec -w /workspace/Neural-Spectral-Codec ring_baseline"
ROSTER=$($DOCKER python3 scripts/_pdv2_c3.py roster --role test)
pick() { for s in $ROSTER; do case "$s" in "$1"/*) echo "$s";; esac; done; }
declare -A H=([kitti]=1.774 [kitti360]=1.735 [helipr]=1.596 [mulran]=1.693 [mcd]=1.938 [nclt]=1.9)
for sensor in kitti kitti360 helipr mulran mcd nclt; do
  echo "[scpp-hsens] $sensor: LIDAR_HEIGHT=${H[$sensor]} ($(date '+%F %T'))"
  LIDAR_HEIGHT=${H[$sensor]} bash scripts/scpp_official/run_pdv2.sh $(pick "$sensor")
done
echo "[scpp-hsens] ALL DONE ($(date '+%F %T'))"
