#!/usr/bin/env bash
# HeLiOS from-scratch pilot supervisor v3 (2026-11-08), server host side.
# One GPU job at a time for this role (root slot rule): train to the next
# evaluation epoch -> evaluate that checkpoint -> continue (resume.pth).
# Own lock /tmp/nsd_overnight_helios_scratch_v2.lock; the SeqOT lock
# /tmp/nsd_native_reproduction_gpu0.lock is never touched.
#   adopt    a v1 training process still running is stopped right after an
#            even epoch is recorded (checkpoint boundary) and resumed here
#   lr5e-4   epochs 2,4,...,20: train segment, then val (B=32)
#   calib    existing seed1 epoch-10 checkpoint, same encoder (after first val)
#   lr1e-4   same
#   decide   pre-registered extension rule -> optional segments to 40
#   verify   ledgers -> family R@1; finalize: COMPUTE_DONE.json (complete +
#            verified) or COMPUTE_FAILED.json (reasons)
#   official one-scan-per-forward recheck: NOT started; official_pending.json
#            lists it for root scheduling
# Usage: nohup setsid bash scripts/_helios_scratch_pilot_20261108.sh >> <W>/logs/supervisor.log 2>&1 &
set -u
REPO=/home/qkrrlans00/Neural-Spectral-Codec
W=$REPO/_external/overnight_repro_20261108/helios_scratch
CW=/workspace/Neural-Spectral-Codec/_external/overnight_repro_20261108/helios_scratch
PY=scripts/_helios_scratch_pilot_20261108.py
MAX_TRY=3
MIN_ROOT_FREE_GIB=35
MAX_OUT_GIB=5          # this role's share of the 12 GiB new-output budget
MIN_GPU_FREE_GIB=25
EVAL_EVERY=2
cd "$REPO" || exit 1
mkdir -p "$W/logs" "$W/flags"
exec 9>/tmp/nsd_overnight_helios_scratch_v2.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) v2 lock held, exit"; exit 1; }
echo $$ > "$W/supervisor.pid"
DX=(docker exec -w /workspace/Neural-Spectral-Codec -e CUDA_VISIBLE_DEVICES=0 -e SEED=1
    -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ring_baseline python3 "$PY")

log() { echo "$(date -u +%FT%TZ) $*"; }
state() {  # stage status note
  python3 - "$W/server_status.json" "$1" "$2" "$3" "$$" <<'EOF'
import json, sys, os, datetime
p, stage, status, note, pid = sys.argv[1:]
d = json.load(open(p)) if os.path.exists(p) else {"history": []}
d.update(stage=stage, status=status, note=note, supervisor_pid=int(pid), supervisor="v3 serialized",
         time=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z")
d["history"] = (d.get("history", []) + [[d["time"], stage, status, note]])[-300:]
json.dump(d, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
EOF
}
guards() {  # wait until disk / output / GPU memory allow a new start
  while :; do
    free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
    out=$(du -sBG "$W" | cut -f1 | tr -dc 0-9)
    gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -dc 0-9)
    gpu=$((gpu / 1024))
    if [ "$free" -ge $MIN_ROOT_FREE_GIB ] && [ "$out" -le $MAX_OUT_GIB ] && [ "$gpu" -ge $MIN_GPU_FREE_GIB ]; then
      return 0
    fi
    state "${STAGE:-guard}" waiting "root_free=${free}G out=${out}G gpu0_free=${gpu}G"
    log "guard wait: root_free=${free}G out=${out}G gpu0_free=${gpu}G"
    sleep 300
  done
}
run_retry() {  # name logfile cmd...
  local name=$1 lf=$2; shift 2
  for t in $(seq 1 $MAX_TRY); do
    guards
    log "start $name try $t"
    "$@" >> "$lf" 2>&1 && { log "ok $name"; return 0; }
    log "FAIL $name try $t"; state "$name" retry "try $t failed; see $lf"; sleep 120
  done
  return 1
}
trained() {  # cand -> last recorded epoch
  python3 -c "import json,os;p='$W/$1/train_record.json';print(max([e['epoch'] for e in json.load(open(p))['epochs']] or [0]) if os.path.exists(p) else 0)"
}
evaluated() {  # cand epoch -> 0 if finished
  grep -q '"finished"' "$W/$1/val/epoch$(printf %03d "$2").json" 2>/dev/null
}

adopt() {  # stop a v1 training process at the next even-epoch boundary
  local c
  for c in lr5e-4 lr1e-4; do
    while pgrep -f "_helios_scratch_pilot_20261108.py train --cand $c" > /dev/null; do
      local ep; ep=$(trained "$c")
      if [ "$ep" -ge $EVAL_EVERY ] && [ $((ep % EVAL_EVERY)) -eq 0 ] && ! [ -f "$W/flags/adopt_$c.$ep" ]; then
        touch "$W/flags/adopt_$c.$ep"
        log "adopt: stopping v1 training of $c after recorded epoch $ep"
        docker exec ring_baseline pkill -f "_helios_scratch_pilot_20261108.py train --cand $c"
        sleep 30
        break
      fi
      state "adopt_$c" running "v1 training running, recorded epoch $ep; waiting for even-epoch boundary"
      sleep 20
    done
  done
  rm -f "$W/flags/"*.TRAINING
}

segment_loop() {  # cand last_epoch
  local c=$1 n=$2 e
  for e in $(seq $EVAL_EVERY $EVAL_EVERY "$n"); do
    if [ "$(trained "$c")" -lt "$e" ]; then
      STAGE="train_${c}_to$e"; state "$STAGE" running "segment to epoch $e"
      local ok=1
      for t in $(seq 1 $MAX_TRY); do
        guards
        local res=""
        [ -f "$W/$c/run/resume.pth" ] && res="--resume"
        log "train $c to $e try $t $res"
        if "${DX[@]}" train --cand "$c" --epochs "$e" $res >> "$W/logs/train_$c.log" 2>&1 \
           && [ "$(trained "$c")" -ge "$e" ]; then ok=0; break; fi
        log "train $c to $e FAIL try $t"; state "$STAGE" retry "try $t failed"; sleep 120
      done
      [ $ok -eq 0 ] || { state "$STAGE" failed "training failed $MAX_TRY times; next candidate"; return 1; }
    fi
    if ! evaluated "$c" "$e"; then
      STAGE="eval_${c}_e$e"; state "$STAGE" running "val9 B=32"
      run_retry "$STAGE" "$W/logs/eval_$c.log" "${DX[@]}" evaluate --cand "$c" --epochs "$e" \
        || log "eval $c e$e failed"
    fi
    if [ "$c" = lr5e-4 ] && [ "$e" -eq $EVAL_EVERY ] && ! grep -q '"finished"' "$W/old_seed1_e10_b32.json" 2>/dev/null; then
      STAGE=calib; state calib running "old seed1 e10 rescoring (B=32)"
      run_retry calib "$W/logs/calib.log" "${DX[@]}" eval-old --batch 32 || log "calib failed"
    fi
  done
}

state start running "supervisor v3 up"
adopt
segment_loop lr5e-4 20 || log "lr5e-4 failed, continue"
segment_loop lr1e-4 20 || log "lr1e-4 failed, continue"
grep -q '"finished"' "$W/old_seed1_e10_b32.json" 2>/dev/null || \
  run_retry calib "$W/logs/calib.log" "${DX[@]}" eval-old --batch 32

state decide running ""
if "${DX[@]}" decide >> "$W/logs/decide.log" 2>&1; then
  ext=$(python3 -c "import json;print(json.load(open('$W/decision_20.json'))['extend'] or '')")
  if [ -n "$ext" ]; then
    log "extend $ext to 40"; segment_loop "$ext" 40 || log "extension failed"
  fi
else
  log "decide failed (incomplete 20-epoch evaluation?)"; state decide failed "see logs/decide.log"
fi
"${DX[@]}" verify >> "$W/logs/verify.log" 2>&1
if python3 scripts/_helios_scratch_pilot_20261108_finalize.py "$W" >> "$W/logs/verify.log" 2>&1; then
  state compute_done done "COMPUTE_DONE.json; official one-scan recheck pending for root scheduling"
else
  state compute_failed failed "COMPUTE_FAILED.json; see reasons"
fi
# The official one-scan-per-forward recheck is not started here; root schedules it
# from official_pending.json when GPU0 is free (python side also needs flags/OFFICIAL_APPROVED).
log "supervisor finished (official recheck pending, not started)"
