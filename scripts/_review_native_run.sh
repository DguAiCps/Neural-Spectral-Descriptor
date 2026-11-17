#!/usr/bin/env bash
# 서버에서 실행한다. GPU 1의 배정된 MIG 인스턴스만 사용한다.
set -euo pipefail
cd /home/qkrrlans00/Neural-Spectral-Codec
variant="${1:-official}"
out=results/pdv2_review_native_20261105
entry=scripts/_review_native_ring.py
if [[ "$variant" == "maxfix" ]]; then
    out="$out/maxfix"
    entry=scripts/_review_native_ring_maxfix.py
elif [[ "$variant" != "official" ]]; then
    exit 2
fi
echo "$$" > "$out/run.pid"
run_worker() {
    docker exec -e CUDA_VISIBLE_DEVICES=1 -e OMP_NUM_THREADS=4 \
        -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e LOKY_MAX_CPU_COUNT=4 \
        ring_baseline bash -lc \
        "ulimit -s 81920; cd /workspace/Neural-Spectral-Codec && python3 -u $entry run $*"
}
run_worker --shards 2 --shard-index 0 > "$out/generate_0.log" 2>&1 &
first_pid=$!
run_worker --shards 2 --shard-index 1 > "$out/generate_1.log" 2>&1 &
second_pid=$!
printf '%s\n%s\n' "$first_pid" "$second_pid" > "$out/generate.pids"
wait "$first_pid"
wait "$second_pid"
run_worker > "$out/retrieval.log" 2>&1
