#!/usr/bin/env bash
# 정확max RING++ 로컬 비용 단계를 순서대로 하나씩 실행한다(GPU 프로세스 1개).
# 종료코드75(MemAvailable<8GiB)는 기다렸다 같은 단계를 잇는다. 다른 실패는 멈춘다.
set -u
REPO=/home/pkm/Neural-Spectral-Codec
WORK=$REPO/results/pdv2_baseline_revision_20261107/ringpp
LOG=$WORK/supervisor.log
STATUS=$WORK/supervisor_status.json
SCRIPT=scripts/_ringpp_exactmax_local_cost_20261107.py
STAGES=${STAGES:-"equiv cache measure memtrace finalize"}
MAX_PAUSES=12

note() {  # stage state extra
  python3 - "$STATUS" "$1" "$2" "${3:-}" <<'EOF'
import json, os, sys, time
path, stage, state, extra = sys.argv[1:]
data = {"supervisor_pid": os.getppid(), "stage": stage, "state": state, "extra": extra,
        "updated_unix": time.time(), "updated_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
tmp = path + ".tmp"
open(tmp, "w").write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
os.replace(tmp, path)
EOF
  echo "$(date -Is) $1 $2 ${3:-}" >> "$LOG"
}

c_free_gib() { df -B1 --output=avail /mnt/c 2>/dev/null | tail -1 | awk '{print $1/2^30}'; }
mem_avail_gib() { awk '/MemAvailable/{print $2/2^20}' /proc/meminfo; }

run_stage() {
  local stage=$1
  if [ "$stage" = finalize ]; then
    python3 scripts/_ringpp_exactmax_finalize_20261107.py >> "$WORK/stage_finalize.log" 2>&1
    return $?
  fi
  local env_args=(-e HOME=/tmp)
  if [ "$stage" = memtrace ]; then
    env_args+=(-e LD_PRELOAD=/workspace/Neural-Spectral-Codec/results/pdv2_table3_20261029/native_trace/libnsd_cuda_trace.so
               -e NSD_CUDA_TRACE_PATH=/workspace/Neural-Spectral-Codec/results/pdv2_baseline_revision_20261107/ringpp/cost/memtrace/trace.jsonl)
    mkdir -p "$WORK/cost/memtrace"
    # 중단된 추적은 이어 쓰지 않는다. 기존 파일은 지우지 않고 이름을 바꿔 보존한다.
    [ -f "$WORK/cost/memtrace/trace.jsonl" ] && [ ! -f "$WORK/cost/memtrace/summary.json" ] && \
      mv "$WORK/cost/memtrace/trace.jsonl" "$WORK/cost/memtrace/trace_interrupted_$(date +%s).jsonl"
  fi
  timeout 4h docker exec -u 1000:1000 "${env_args[@]}" -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 "$SCRIPT" "$stage" >> "$WORK/stage_$stage.log" 2>&1
}

cd "$REPO" || exit 2
echo $$ > "$WORK/supervisor.pid"
for stage in $STAGES; do
  pauses=0
  while true; do
    cf=$(c_free_gib); ma=$(mem_avail_gib)
    if awk "BEGIN{exit !($cf < 1.0)}"; then
      note "$stage" "blocked_c_drive" "C free ${cf}GiB < 1GiB"; exit 3
    fi
    if awk "BEGIN{exit !($ma < 8.0)}"; then
      pauses=$((pauses+1)); note "$stage" "waiting_memory" "MemAvailable ${ma}GiB, pause $pauses"
      [ $pauses -gt $MAX_PAUSES ] && { note "$stage" "failed" "memory stayed low"; exit 4; }
      sleep 300; continue
    fi
    note "$stage" "running" "C free ${cf}GiB MemAvailable ${ma}GiB"
    run_stage "$stage"; code=$?
    if [ $code -eq 0 ]; then note "$stage" "done" ""; break; fi
    if [ $code -eq 75 ]; then
      pauses=$((pauses+1)); note "$stage" "paused_low_memory" "exit75, pause $pauses"
      [ $pauses -gt $MAX_PAUSES ] && { note "$stage" "failed" "memory stayed low"; exit 4; }
      sleep 300; continue
    fi
    note "$stage" "failed" "exit $code; see stage_$stage.log"; exit 1
  done
done
note all complete ""
