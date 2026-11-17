#!/usr/bin/env bash
# 로컬 ring_baseline(RTX 5080)에서 (a) 인코딩만 시퀀스별로 하나씩 실행한다. GPU 사용률을 5초 간격으로 함께 기록한다.
# 사용: bash scripts/_train_timing_20261117_local.sh kitti/05 helipr/Bridge01 ...
set -uo pipefail
cd "$(dirname "$0")/.."
L=results/pdv2_train_timing_20261117/local_logs
mkdir -p "$L"
for q in "$@"; do
  n=${q/\//_}
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 5 > "$L/gpu_$n.csv" &
  smi=$!
  t0=$(date +%s.%N)
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_train_timing_20261117.py encode --sequence "$q" > "$L/encode_$n.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  kill $smi
  python3 -c "import json;print(json.dumps(dict(step='encode_$n', host='local', rc=$rc, wall_s=$t1-$t0, loadavg=open('/proc/loadavg').read().split()[:3])))" >> "$L/wall.jsonl"
done
echo DONE "$@"
