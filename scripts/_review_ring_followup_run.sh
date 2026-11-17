#!/usr/bin/env bash
# RING 후속 진단 서버 실행기. 서버 호스트에서 백그라운드로 실행한다.
#   nohup bash scripts/_review_ring_followup_run.sh > results/pdv2_reproduction_orchestration_20261106/ring/runner.log 2>&1 &
# 순서: C1 점수(CPU 4스레드) -> C1 검산 -> 커널 대조(GPU0) -> 공식 evaluate() 6조건(GPU0) -> 요약·검산.
# GPU 구간은 조건마다 호스트 /tmp/nsd_native_reproduction_gpu0.lock 을 flock -x 로 잡는다.
# 완료된 단계(report.json/summary.json 존재)는 건너뛰므로 같은 명령으로 재개한다.
set -euo pipefail
cd /home/qkrrlans00/Neural-Spectral-Codec
root=results/pdv2_reproduction_orchestration_20261106/ring
lock=/tmp/nsd_native_reproduction_gpu0.lock
echo "$$" > "$root/runner.pid"
env_args=(-e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e LOKY_MAX_CPU_COUNT=4
          -e NUMEXPR_NUM_THREADS=4 -w /workspace/Neural-Spectral-Codec)
cpu() { docker exec "${env_args[@]}" -e CUDA_VISIBLE_DEVICES= ring_baseline bash -lc "ulimit -s 81920; $*"; }
gpu() { flock -x "$lock" docker exec "${env_args[@]}" -e CUDA_VISIBLE_DEVICES=0 ring_baseline \
            bash -lc "ulimit -s 81920; $*"; }
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "$(stamp) start"
if ! grep -q '"stage": "complete"' "$root/score_c1/status.json" 2>/dev/null; then
    echo "$(stamp) score_c1 run"
    cpu python3 -u scripts/_review_ring_followup_score.py run > "$root/score_c1/run.log" 2>&1
fi
if [[ ! -f "$root/score_c1/verification.json" ]]; then
    echo "$(stamp) score_c1 verify"
    cpu python3 -u scripts/_review_ring_followup_verify.py score > "$root/score_c1/verify.log" 2>&1
fi
mkdir -p "$root/kernels" "$root/official_eval"
if [[ ! -f "$root/kernels/report.json" ]]; then
    echo "$(stamp) kernels"
    gpu python3 -u scripts/_review_ring_followup_kernels.py > "$root/kernels/run.log" 2>&1
fi
for cond in "test occ official" "test feat maxfix" "test feat official" \
            "all occ official" "all feat maxfix" "all feat official"; do
    read -r split bev impl <<< "$cond"
    name="${split}_${bev}_${impl}"
    if [[ -f "$root/official_eval/$name/report.json" ]]; then continue; fi
    mkdir -p "$root/official_eval/$name"
    echo "$(stamp) official $name"
    gpu python3 -u scripts/_review_ring_followup_official.py run --split "$split" --bev "$bev" --impl "$impl" \
        > "$root/official_eval/$name/evaluate.log" 2>&1
done
cpu python3 -u scripts/_review_ring_followup_official.py summarize > "$root/official_eval/summarize.log" 2>&1
cpu python3 -u scripts/_review_ring_followup_verify.py official > "$root/official_eval/verify.log" 2>&1
echo "$(stamp) complete"
echo complete > "$root/runner.done"
