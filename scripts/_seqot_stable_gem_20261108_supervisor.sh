#!/usr/bin/env bash
# SeqOT stage-2 recovery with the stabilized GeM (2026-11-08) -- server supervisor (joms-gpu host).
# train (CPU, seeds serial) -> VAL selection (CPU) -> val freeze -> GPU gate -> test (GPU0, seeds
# serial, existing GPU0 flock) -> aggregate -> final -> COMPUTE_DONE.json | COMPUTE_FAILED.json.
# Every step is idempotent (flags/<step>.done); a restart continues at the first missing step.
# Start: cd $REPO && nohup setsid bash _external/seqot_stable_gem_20261108/code/_seqot_stable_gem_20261108_supervisor.sh \
#          >> _external/seqot_stable_gem_20261108/logs/supervisor.log 2>&1 &
set -uo pipefail
REPO=/home/qkrrlans00/Neural-Spectral-Codec
NEW=$REPO/_external/seqot_stable_gem_20261108
CODE=/workspace/Neural-Spectral-Codec/_external/seqot_stable_gem_20261108/code/_seqot_stable_gem_20261108.py
F=$NEW/flags
L=$NEW/logs
HB=$REPO/_external/overnight_repro_20261108
GPU_LOCK=/tmp/nsd_native_reproduction_gpu0.lock
MIN_FREE_ROOT_GIB=35
MAX_NEW_GIB=2
MIN_GPU_FREE_MIB=25600
mkdir -p "$F" "$L"

exec 8>"$NEW/supervisor.lock"
if ! flock -n 8; then echo "$(date -u +%FT%TZ) supervisor already running"; exit 0; fi
echo $$ > "$NEW/supervisor.pid"

status() {  # status <status> <stage> <note>
  python3 - "$NEW/server_status.json" "$1" "$2" "${3:-}" <<'EOF'
import json, os, sys, time
p, st, stage, note = sys.argv[1:5]
t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
try:
    d = json.load(open(p))
except Exception:
    d = {"history": []}
h = d.get("history", [])
if not h or h[-1][1:3] != [stage, st] or h[-1][3] != note:
    h.append([t, stage, st, note])
d.update(history=h[-200:], status=st, stage=stage, note=note, time=t, supervisor_pid=os.getppid())
json.dump(d, open(p + ".tmp", "w"), indent=1)
os.replace(p + ".tmp", p)
EOF
  echo "$(date -u +%FT%TZ) [$1] $2 $3"
}

failed() {  # failed <stage> <rc>
  python3 - "$NEW" "$1" "$2" <<'EOF'
import glob, json, os, sys, time
new, stage, rc = sys.argv[1:4]
flags = {os.path.basename(f): json.load(open(f)) for f in sorted(glob.glob(f"{new}/flags/FAILED_*.json"))}
d = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "failed_stage": stage, "rc": int(rc),
     "failure_records": flags, "log": f"{new}/logs/{stage}.log",
     "note": "stopped at the first failure; no later stage ran; a failed model is not a candidate"}
json.dump(d, open(f"{new}/COMPUTE_FAILED.json.tmp", "w"), indent=1)
os.replace(f"{new}/COMPUTE_FAILED.json.tmp", f"{new}/COMPUTE_FAILED.json")
EOF
  status failed "$1" "rc=$2"
  exit 1
}

disk_ok() {
  while :; do
    free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
    used=$(du -s -BG "$NEW" | cut -f1 | tr -dc 0-9)
    if [ "$used" -gt "$MAX_NEW_GIB" ]; then status failed disk "new output ${used}G > ${MAX_NEW_GIB}G"; failed disk 4; fi
    if [ "$free" -ge "$MIN_FREE_ROOT_GIB" ]; then return 0; fi
    status waiting disk "root free ${free}G < ${MIN_FREE_ROOT_GIB}G"
    sleep 600
  done
}

dx_cpu() {  # dx_cpu <seed> <cmd>
  docker exec -w /tmp -e SEED="$1" -e CUDA_VISIBLE_DEVICES= -e TORCH_THREADS=1 -e OMP_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    ring_baseline nice -n 10 ionice -c 3 python3 "$CODE" "$2"
}

dx_gpu() {  # dx_gpu <seed> <cmd>
  docker exec -w /tmp -e SEED="$1" -e CUDA_VISIBLE_DEVICES=0 -e TORCH_THREADS=2 -e OMP_NUM_THREADS=2 \
    -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    ring_baseline nice -n 10 python3 "$CODE" "$2"
}

step_cpu() {  # step_cpu <name> <seed> <cmd>
  [ -f "$F/$1.done" ] && return 0
  disk_ok
  status running "$1" "cpu seed $2"
  if dx_cpu "$2" "$3" >> "$L/$1.log" 2>&1; then date -u +%FT%TZ > "$F/$1.done"; else failed "$1" $?; fi
}

helios_slot() {  # prints the reason a HeLiOS GPU slot is free (server completion), or nothing
  for r in helios_finetune helios_scratch; do
    if [ -f "$HB/$r/COMPUTE_DONE.json" ]; then echo "${r}_COMPUTE_DONE"; return; fi
  done
  for r in helios_finetune helios_scratch; do
    pid=$(cat "$HB/$r/supervisor.pid" 2>/dev/null)
    if [ -f "$HB/$r/COMPUTE_FAILED.json" ] && [ -n "$pid" ] && [ ! -d "/proc/$pid" ]; then
      echo "${r}_COMPUTE_FAILED_supervisor_gone"; return
    fi
  done
}

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -dc 0-9; }

step_gpu() {  # step_gpu <name> <seed> <cmd>: gate -> existing GPU0 flock -> re-check -> run
  [ -f "$F/$1.done" ] && return 0
  while :; do
    disk_ok
    slot=$(helios_slot)
    free=$(gpu_free)
    if [ -z "$slot" ] || [ "${free:-0}" -lt "$MIN_GPU_FREE_MIB" ]; then
      status waiting gpu_gate "$1: helios slot='${slot:-none}' gpu0 free=${free}MiB (need slot and >= ${MIN_GPU_FREE_MIB})"
      sleep 300
      continue
    fi
    status waiting gpu_lock "$1: slot=$slot free=${free}MiB; waiting for $GPU_LOCK"
    exec 9<"$GPU_LOCK"
    flock -x 9
    free=$(gpu_free)
    if [ "${free:-0}" -lt "$MIN_GPU_FREE_MIB" ]; then
      flock -u 9; exec 9<&-
      status waiting gpu_gate "$1: lock held but gpu0 free=${free}MiB; released"
      sleep 300
      continue
    fi
    echo "$(date -u +%FT%TZ) $1 gate slot=$slot free=${free}MiB lock=$GPU_LOCK" >> "$L/gpu_gate.log"
    status running "$1" "gpu0 seed $2 slot=$slot free=${free}MiB (lock held)"
    dx_gpu "$2" "$3" >> "$L/$1.log" 2>&1
    rc=$?
    flock -u 9; exec 9<&-
    if [ $rc -eq 0 ]; then date -u +%FT%TZ > "$F/$1.done"; return 0; fi
    failed "$1" $rc
  done
}

[ -f "$NEW/COMPUTE_DONE.json" ] && { status done final "already complete"; exit 0; }
rm -f "$NEW/COMPUTE_FAILED.json.tmp"
[ -f "$NEW/COMPUTE_FAILED.json" ] && { status failed restart_refused "COMPUTE_FAILED.json present; mechanical recovery must move it aside first"; exit 1; }
status running start "supervisor pid $$"
step_cpu freeze 1 freeze
step_cpu selftest 1 selftest
step_cpu valaudit 1 valaudit
for s in 1 2 3; do
  step_cpu "train2_s$s" "$s" train2
  step_cpu "select2_s$s" "$s" select2
done
step_cpu valfreeze 1 valfreeze
status running val_frozen "selection frozen; test waits for the GPU gate"
for s in 1 2 3; do step_gpu "test_s$s" "$s" test; done
for s in 1 2 3; do step_cpu "aggregate_s$s" "$s" aggregate; done
step_cpu final 1 final
python3 - "$NEW" <<'EOF'
import hashlib, json, os, sys, time
new = sys.argv[1]
rec = f"{new}/records"
def h(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
v = json.load(open(f"{rec}/verification.json"))
s = json.load(open(f"{rec}/summary.json"))
d = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True, "verify_passed": v["passed"],
     "label": s["label"], "mean_std": s["mean_std"], "selected_epochs": s["selected_epochs"],
     "sha256": {n: h(f"{rec}/{n}") for n in ("protocol.json", "val_freeze.json", "val_audit.json", "selftest.json",
                                              "summary.json", "verification.json")}}
json.dump(d, open(f"{new}/COMPUTE_DONE.json.tmp", "w"), indent=1)
os.replace(f"{new}/COMPUTE_DONE.json.tmp", f"{new}/COMPUTE_DONE.json")
EOF
status done final "COMPUTE_DONE.json written"
