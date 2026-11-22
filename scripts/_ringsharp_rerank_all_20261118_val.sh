#!/bin/bash
# 검증9 3시드 순차 실행(컨테이너 안에서 nohup으로 띄움). 끝난 출력은 건너뛴다.
cd /workspace/Neural-Spectral-Codec
L=logs/ringsharp_rerank_all_20261118
while pgrep -f "rerank_all_20261118.py val --seed 1 --seq kitti/09" >/dev/null; do sleep 5; done
for seed in 1 2 3; do for seq in kitti/09 kitti360/0004 kitti360/0006 helipr/Roundabout01 helipr/Roundabout02 helipr/Roundabout03 mulran/Sejong01 mulran/Sejong02 mulran/Sejong03; do
  n=$(echo $seq | tr / _)
  python3 scripts/_ringsharp_rerank_all_20261118.py val --seed $seed --seq $seq > $L/val_s${seed}_${n}.log 2>&1
  echo "$(date +%T) s$seed $seq rc=$?" >> $L/val_driver.log
done; done
echo ALL_VAL_END >> $L/val_driver.log
