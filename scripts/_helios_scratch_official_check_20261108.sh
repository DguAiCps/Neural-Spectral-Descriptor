#!/usr/bin/env bash
# HeLiOS scratch pilot official single-scan (B1) VAL9 confirmation supervisor (2026-11-08), host side.
# scratch_seed1_pilot_diagnostic.  Fixed order, every step resumable (flags/<step>.done; eval per sequence):
#   preflight (CPU) -> wait public-FT B1 check COMPUTE_DONE.json (complete=true; FAILED -> stop, root decides)
#   -> B1 old_e10 (GPU0) -> B1 candidate lr5e-4 e10 (GPU0) -> verify (CPU) -> COMPUTE_DONE.json | COMPUTE_FAILED.json
# Each GPU step: root free>=35G, own output<=2G, GPU0 free>=25600MiB -> the existing global GPU0 flock
# /tmp/nsd_native_reproduction_gpu0.lock (same open/flock -x as other native jobs; lock file never changed)
# -> re-check free -> run -> release.  Own role lock /tmp/nsd_ovn_helios_scratch_official_check.lock.
# Never stops/changes other processes.  No training, no reselection, no test.
# Start: cd $REPO && mkdir -p _external/overnight_repro_20261108/helios_scratch_official_check/logs && \
#        nohup setsid bash scripts/_helios_scratch_official_check_20261108.sh \
#          >> _external/overnight_repro_20261108/helios_scratch_official_check/logs/supervisor.log 2>&1 < /dev/null &
set -uo pipefail
REPO=/home/qkrrlans00/Neural-Spectral-Codec
W=$REPO/_external/overnight_repro_20261108/helios_scratch_official_check
FTB1=$REPO/_external/overnight_repro_20261108/helios_ft_official_check
PY=scripts/_helios_scratch_official_check_20261108.py
GPU_LOCK=/tmp/nsd_native_reproduction_gpu0.lock
MIN_ROOT_FREE_GIB=35
MAX_OUT_GIB=2
MIN_GPU_FREE_MIB=25600
MAX_TRY=3
WAIT_HOURS=24          # max waiting (FT B1/gates/lock) from first start; a running evaluation is never killed
cd "$REPO" || exit 1
mkdir -p "$W/logs" "$W/flags"
exec 8>/tmp/nsd_ovn_helios_scratch_official_check.lock
flock -n 8 || { echo "$(date -u +%FT%TZ) supervisor already running"; exit 0; }
echo $$ > "$W/supervisor.pid"
[ -f "$W/start_epoch" ] || date +%s > "$W/start_epoch"
DEADLINE=$(( $(cat "$W/start_epoch") + WAIT_HOURS * 3600 ))
ENVS=(-e SEED=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
DX_GPU=(docker exec -w /workspace/Neural-Spectral-Codec -e CUDA_VISIBLE_DEVICES=0 -e OMP_NUM_THREADS=2
        -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 "${ENVS[@]}" ring_baseline
        nice -n 10 ionice -c 2 -n 7 python3 "$PY")
DX_CPU=(docker exec -w /workspace/Neural-Spectral-Codec -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=1
        -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 "${ENVS[@]}" ring_baseline
        nice -n 10 ionice -c 3 python3 "$PY")

log() { echo "$(date -u +%FT%TZ) $*"; }
status() {  # status <status> <stage> <note> [job_pid]
  python3 - "$W/server_status.json" "$1" "$2" "$3" "${4:-}" <<'EOF'
import json, os, sys, time
p, st, stage, note, job = sys.argv[1:6]
t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
try:
    d = json.load(open(p))
except Exception:
    d = {"history": []}
h = d.get("history", [])
if not h or h[-1][1:4] != [stage, st, note]:
    h.append([t, stage, st, note])
d.update(history=h[-300:], status=st, stage=stage, note=note, time=t, supervisor_pid=os.getppid(),
         job_host_pid=int(job) if job else None, label="scratch_seed1_pilot_diagnostic")
json.dump(d, open(p + ".tmp", "w"), indent=1)
os.replace(p + ".tmp", p)
EOF
  log "[$1] $2 $3 ${4:-}"
}
fail() {  # fail <stage> <reason>
  python3 - "$W" "$1" "$2" <<'EOF'
import json, os, sys, time
w, stage, reason = sys.argv[1:4]
d = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "failed", "failed_stage": stage,
     "reason": reason, "log": f"{w}/logs/{stage}.log", "label": "scratch_seed1_pilot_diagnostic",
     "note": "stopped at the first failure; resumable after root review (move this file aside, rerun supervisor)"}
json.dump(d, open(f"{w}/COMPUTE_FAILED.json.tmp", "w"), indent=1)
os.replace(f"{w}/COMPUTE_FAILED.json.tmp", f"{w}/COMPUTE_FAILED.json")
EOF
  status failed "$1" "$2"
  exit 1
}
past_deadline() { [ "$(date +%s)" -ge "$DEADLINE" ]; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -dc 0-9; }
gates_ok() {  # sets GATE note; 0 if disk and GPU0 gates pass
  local free out gpu
  free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  out=$(du -s -BG "$W" | cut -f1 | tr -dc 0-9)
  gpu=$(gpu_free)
  GATE="root_free=${free}G out=${out}G gpu0_free=${gpu}MiB"
  [ "$out" -gt "$MAX_OUT_GIB" ] && fail disk "own output ${out}G > ${MAX_OUT_GIB}G"
  [ "$free" -ge "$MIN_ROOT_FREE_GIB" ] && [ "${gpu:-0}" -ge "$MIN_GPU_FREE_MIB" ]
}
run_logged() {  # run_logged <logname> <cmd...>: background + wait, records host PID of the docker exec client
  local name=$1; shift
  "$@" >> "$W/logs/$name.log" 2>&1 &
  JOB=$!
  echo "$JOB" > "$W/job.pid"
  wait "$JOB"
  local rc=$?
  rm -f "$W/job.pid"
  return $rc
}
step_cpu() {  # step_cpu <name> <args...>
  local name=$1; shift
  [ -f "$W/flags/$name.done" ] && return 0
  for t in $(seq 1 $MAX_TRY); do
    status running "$name" "cpu try $t"
    if run_logged "$name" "${DX_CPU[@]}" "$@"; then date -u +%FT%TZ > "$W/flags/$name.done"; return 0; fi
    status retry "$name" "cpu try $t failed"; sleep 60
  done
  fail "$name" "failed $MAX_TRY times"
}
wait_ftb1() {  # the public-FT B1 check must be finished (its GPU steps done) before this role uses GPU0
  [ -f "$W/flags/ftb1_done.seen" ] && return 0
  while :; do
    if [ -f "$FTB1/COMPUTE_DONE.json" ] && python3 -c "import json,sys;sys.exit(0 if json.load(open('$FTB1/COMPUTE_DONE.json')).get('complete') is True else 1)"; then
      { date -u +%FT%TZ; sha256sum "$FTB1/COMPUTE_DONE.json"; } > "$W/flags/ftb1_done.seen"
      status running wait_ftb1 "FT B1 COMPUTE_DONE complete=true seen"
      return 0
    fi
    if [ -f "$FTB1/COMPUTE_FAILED.json" ]; then
      fail wait_ftb1 "FT B1 COMPUTE_FAILED.json present; scratch B1 not started (root decides; FT may be resumed)"
    fi
    past_deadline && fail wait_ftb1 "timeout ${WAIT_HOURS}h waiting for FT B1 COMPUTE_DONE"
    status waiting wait_ftb1 "FT B1 stage=$(python3 -c "import json;d=json.load(open('$FTB1/server_status.json'));print(d.get('stage'),d.get('status'))" 2>/dev/null)"
    sleep 300
  done
}
step_gpu() {  # step_gpu <name> <args...>
  local name=$1; shift
  [ -f "$W/flags/$name.done" ] && return 0
  local t=0
  while [ $t -lt $MAX_TRY ]; do
    if ! gates_ok; then
      past_deadline && fail "$name" "timeout ${WAIT_HOURS}h waiting for gates ($GATE)"
      status waiting gpu_gate "$name: $GATE"
      sleep 300; continue
    fi
    status waiting gpu_lock "$name: $GATE; waiting for $GPU_LOCK"
    exec 9<"$GPU_LOCK"
    until flock -x -w 600 9; do
      past_deadline && { exec 9<&-; fail "$name" "timeout ${WAIT_HOURS}h waiting for $GPU_LOCK"; }
      status waiting gpu_lock "$name: still waiting for $GPU_LOCK"
    done
    if ! gates_ok; then
      flock -u 9; exec 9<&-
      status waiting gpu_gate "$name: lock acquired but $GATE; released"
      sleep 300; continue
    fi
    t=$((t + 1))
    echo "$(date -u +%FT%TZ) $name try $t gate $GATE lock=$GPU_LOCK" >> "$W/logs/gpu_gate.log"
    "${DX_GPU[@]}" "$@" >> "$W/logs/$name.log" 2>&1 &
    JOB=$!
    echo "$JOB" > "$W/job.pid"
    status running "$name" "gpu0 try $t (lock held) $GATE" "$JOB"
    wait "$JOB"
    local rc=$?
    rm -f "$W/job.pid"
    flock -u 9; exec 9<&-
    echo "$(date -u +%FT%TZ) $name try $t rc=$rc lock released" >> "$W/logs/gpu_gate.log"
    if [ $rc -eq 0 ]; then date -u +%FT%TZ > "$W/flags/$name.done"; return 0; fi
    status retry "$name" "gpu try $t rc=$rc; resumes per sequence"
    sleep 120
  done
  fail "$name" "failed $MAX_TRY times"
}

[ -f "$W/COMPUTE_DONE.json" ] && { status done final "already complete"; exit 0; }
[ -f "$W/COMPUTE_FAILED.json" ] && { status failed restart_refused "COMPUTE_FAILED.json present; root moves it aside first"; exit 1; }
status running start "supervisor pid $$"
step_cpu preflight preflight
wait_ftb1
step_gpu b1_old_e10 eval --which old_e10
step_gpu b1_candidate eval --which candidate
step_cpu verify verify
python3 - "$W" <<'EOF'
import hashlib, json, os, sys, time
w = sys.argv[1]
v = json.load(open(f"{w}/verification.json"))
s = json.load(open(f"{w}/summary.json"))
def h(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
files = ["preflight.json", "verification.json", "summary.json",
         "old_seed1/val/epoch010.json", "lr5e-4/val/epoch010.json"]
d = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "complete" if v["all_ok"] else "verify_failed",
     "complete": bool(v["all_ok"]), "label": "scratch_seed1_pilot_diagnostic", "summary": s,
     "sha256": {f: h(f"{w}/{f}") for f in files}}
name = "COMPUTE_DONE.json" if v["all_ok"] else "COMPUTE_FAILED.json"
json.dump(d, open(f"{w}/{name}.tmp", "w"), indent=1)
os.replace(f"{w}/{name}.tmp", f"{w}/{name}")
EOF
if [ -f "$W/COMPUTE_DONE.json" ]; then status done final "COMPUTE_DONE.json written"; else status failed final "verify not all_ok"; fi
log "supervisor finished"
