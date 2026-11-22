#!/bin/bash
# 시험46 전체 재정렬 서버 실행: 작업자 8개(GPU 2·3에 4개씩), 큰 시퀀스부터 배정(LPT). 컨테이너 안에서 실행.
cd /workspace/Neural-Spectral-Codec
L=logs/ringsharp_rerank_all_20261118
mkdir -p $L
w=$1
case $w in
  0) G=2; SEQS="helipr/Town03 helipr/KAIST06 helipr/Riverside05 kitti/08";;
  1) G=2; SEQS="kitti360/0000 mulran/Riverside03 nclt/2012-11-04 kitti/00 nclt/2013-01-10 mcd/tuhh_night_07";;
  2) G=2; SEQS="helipr/Town02 mulran/KAIST01 nclt/2012-11-16 helipr/DCC06 kitti/07";;
  3) G=2; SEQS="helipr/Town01 mulran/KAIST03 mulran/DCC03 mulran/DCC02 mcd/kth_day_09 mcd/tuhh_day_04";;
  4) G=3; SEQS="nclt/2012-01-08 helipr/KAIST04 mulran/Riverside02 mulran/DCC01 mcd/kth_day_06 mcd/tuhh_day_02";;
  5) G=3; SEQS="nclt/2012-05-11 nclt/2012-08-04 mulran/KAIST02 mulran/Riverside01 mcd/kth_night_01 mcd/kth_day_10";;
  6) G=3; SEQS="helipr/Riverside06 nclt/2012-02-04 helipr/Riverside04 helipr/DCC04 mcd/tuhh_day_03 mcd/kth_night_05";;
  7) G=3; SEQS="nclt/2012-03-17 helipr/KAIST05 nclt/2013-02-23 helipr/DCC05 mcd/tuhh_night_08 mcd/kth_night_04";;
esac
for s in $SEQS; do
  n=$(echo $s | tr / _)
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G python3 scripts/_ringsharp_rerank_all_20261118.py test --seq $s --seeds 1 2 3 > $L/test_$n.log 2>&1
  echo "$(date +%F_%T) w$w $s rc=$?" >> $L/driver.log
done
echo "$(date +%F_%T) w$w END" >> $L/driver.log
