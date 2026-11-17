#!/usr/bin/env bash
# 서버 호스트에서 실행. 스테이징 폴더의 ring_baseline 안에서 한 번에 프로세스 하나만 돈다.
# 사용: bash scripts/_train_timing_20261117.sh <GPU번호(PCI 순서)> <단계...>
#   단계: train1 train2 train3 | encode | stats | pools
set -euo pipefail
GPU=$1; shift
STAGE=/workspace/Neural-Spectral-Codec/_nsd_stage_20261116
OUT=/home/qkrrlans00/Neural-Spectral-Codec/_nsd_stage_20261116/results/pdv2_train_timing_20261117
mkdir -p "$OUT/logs"
TRAIN_SEQ="${TRAIN_SEQ:-kitti/01 kitti/02 kitti/05 kitti/06 helipr/Bridge01 helipr/Bridge02 helipr/Bridge03 helipr/Bridge04}"
step() {  # 이름, 인자...
  local name=$1; shift
  local t0 t1
  t0=$(date +%s.%N)
  docker exec -w "$STAGE" ring_baseline env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    python3 scripts/_train_timing_20261117.py "$@" > "$OUT/logs/$name.log" 2>&1
  t1=$(date +%s.%N)
  echo "{\"step\": \"$name\", \"gpu\": $GPU, \"wall_s\": $(echo "$t1 - $t0" | bc), \"start\": \"$(date -d @${t0%.*} -Iseconds)\"}" >> "$OUT/logs/wall.jsonl"
}
for s in "$@"; do
  case $s in
    train*) step "$s" train --rep "${s#train}" ;;
    encode) for q in $TRAIN_SEQ; do step "encode_${q/\//_}" encode --sequence "$q"; done ;;
    stats) step stats stats ;;
    pools) step pools pools ;;
  esac
done
echo DONE "$@"
