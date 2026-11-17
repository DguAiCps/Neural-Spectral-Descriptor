#!/usr/bin/env bash
# RING 08-20→02-04 날짜 쌍 진단 서버 실행기. 서버 호스트에서 백그라운드로 실행한다.
#   setsid nohup bash scripts/_review_ring_followup_datepair_run.sh \
#       > results/pdv2_reproduction_orchestration_20261106/ring/date_pair_20261107/runner.log 2>&1 < /dev/null &
# 순서: coverage(CPU) -> protocol 고정 -> 누락 스캔 범위 수신(CPU, 상한 판정) -> 공식 evaluate() 4조건(GPU0 flock).
# 완료 단계(파일 존재)는 건너뛰므로 같은 명령으로 재개한다. protocol.json은 다시 쓰지 않는다.
set -euo pipefail
cd /home/qkrrlans00/Neural-Spectral-Codec
root=results/pdv2_reproduction_orchestration_20261106/ring/date_pair_20261107
lock=/tmp/nsd_native_reproduction_gpu0.lock
script=scripts/_review_ring_followup_datepair.py
echo "$$" > "$root/runner.pid"
env_args=(-e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e LOKY_MAX_CPU_COUNT=4
          -e NUMEXPR_NUM_THREADS=4 -w /workspace/Neural-Spectral-Codec)
cpu() { docker exec "${env_args[@]}" -e CUDA_VISIBLE_DEVICES= ring_baseline bash -lc "ulimit -s 81920; $*"; }
gpu() { flock -x "$lock" docker exec "${env_args[@]}" -e CUDA_VISIBLE_DEVICES=0 ring_baseline \
            bash -lc "ulimit -s 81920; $*"; }
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "$(stamp) start"
if [[ ! -f "$root/protocol.json" ]]; then
    echo "$(stamp) coverage"
    cpu python3 -u "$script" coverage > "$root/coverage.log" 2>&1
    echo "$(stamp) protocol"
    cpu python3 -u "$script" protocol > "$root/protocol.log" 2>&1
fi
if [[ ! -f "$root/fetch_complete.json" ]]; then
    echo "$(stamp) fetch"
    cpu python3 -u "$script" fetch > "$root/fetch.log" 2>&1
fi
for cond in "all occ official" "all feat maxfix" "test occ official" "test feat maxfix"; do
    read -r split bev impl <<< "$cond"
    name="${split}_${bev}_${impl}"
    if [[ -f "$root/$name/report.json" ]]; then continue; fi
    mkdir -p "$root/$name"
    echo "$(stamp) wait_gpu_lock $name"
    gpu "echo \$(date -u +%Y-%m-%dT%H:%M:%SZ) lock_acquired; python3 -u $script run --split $split --bev $bev --impl $impl" \
        > "$root/$name/evaluate.log" 2>&1
    echo "$(stamp) done $name"
done
echo "$(stamp) complete"
echo complete > "$root/runner.done"
