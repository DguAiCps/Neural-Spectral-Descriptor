#!/usr/bin/env bash
# SeqOT paper-recipe chain (review revision 2026-11-07), host side of joms-gpu.
#   master: holds the shared GPU0 lock for the whole run, runs the three seed
#           chains in parallel on GPU0, then (test stage) waits for the test
#           inputs (READY marker) and runs scripts/_review_seqot_paper_recipe_20261107_test.sh.
#   seed S: train1 + select1 --follow -> subdesc -> train2 -> select2, each step
#           resumable (the wrapper resumes / skips finished work), marker per step.
# Usage: setsid nohup bash scripts/_review_seqot_paper_recipe_20261107_chain.sh master &
# Markers: _external/seqot_paper_recipe_20261107/chain/{s1,s2,s3}/<step>.done,
#          chain/TRAIN_DONE, chain/TEST_DONE, chain/FAILED (text = reason), chain/master.pid
set -u
REPO=/home/qkrrlans00/Neural-Spectral-Codec
W=_external/seqot_paper_recipe_20261107
C=$REPO/$W/chain
LOG=$REPO/$W/logs
WRAP=scripts/_review_seqot_paper_recipe_20261107.py
LOCK=/tmp/nsd_native_reproduction_gpu0.lock
mkdir -p "$C" "$LOG"
cd "$REPO" || exit 1
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(ts) $*" >> "$LOG/chain.log"; }

run() {  # run <seed> <logname> <args...>: wrapper step inside ring_baseline on GPU0
  local seed=$1 name=$2; shift 2
  docker exec -w /workspace/Neural-Spectral-Codec -e CUDA_VISIBLE_DEVICES=0 -e SEED="$seed" \
    -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e PYTHONUNBUFFERED=1 ring_baseline \
    python3 "$WRAP" "$@" >> "$LOG/s${seed}_${name}.log" 2>&1
}

step() {  # step <seed> <name> <tries> <args...>
  local seed=$1 name=$2 tries=$3; shift 3
  local m="$C/s$seed/$name.done"
  [ -f "$m" ] && { say "s$seed $name: done before, skip"; return 0; }
  local k
  for k in $(seq 1 "$tries"); do
    say "s$seed $name: start try $k (wrapper sha $(sha256sum $WRAP | cut -c1-16))"
    run "$seed" "$name" "$@"
    local rc=$?
    if [ $rc = 0 ]; then ts > "$m"; say "s$seed $name: ok"; return 0; fi
    say "s$seed $name: rc=$rc try $k"
    sleep 60
  done
  return 1
}

seed_chain() {
  local s=$1
  mkdir -p "$C/s$s"
  if [ ! -f "$C/s$s/stage1.done" ]; then
    ( step "$s" select1 3 select1 --follow ) &
    local sp=$!
    step "$s" train1 3 train1 || { kill "$sp" 2>/dev/null; echo "s$s train1" >> "$C/FAILED"; return 1; }
    wait "$sp" || { echo "s$s select1" >> "$C/FAILED"; return 1; }
    ts > "$C/s$s/stage1.done"
  fi
  step "$s" subdesc 3 subdesc || { echo "s$s subdesc" >> "$C/FAILED"; return 1; }
  step "$s" train2 3 train2 || { echo "s$s train2" >> "$C/FAILED"; return 1; }
  step "$s" select2 3 select2 || { echo "s$s select2" >> "$C/FAILED"; return 1; }
  ts > "$C/s$s/SEED_DONE"
}

master() {
  exec 9>"$LOCK"
  if ! flock -n 9; then say "master: GPU0 lock held by another job; exit"; echo "gpu0 lock held" >> "$C/FAILED"; exit 3; fi
  echo $$ > "$C/master.pid"
  rm -f "$C/FAILED"
  say "master: start pid $$ (lock $LOCK held)"
  local pids=()
  for s in 1 2 3; do
    if [ -f "$C/s$s/SEED_DONE" ]; then say "s$s: SEED_DONE before"; continue; fi
    ( seed_chain "$s" ) & pids+=($!)
    sleep 20
  done
  local bad=0
  for p in "${pids[@]}"; do wait "$p" || bad=1; done
  for s in 1 2 3; do [ -f "$C/s$s/SEED_DONE" ] || bad=1; done
  if [ $bad = 1 ]; then say "master: training failed: $(cat "$C/FAILED" 2>/dev/null)"; exit 1; fi
  ts > "$C/TRAIN_DONE"
  say "master: TRAIN_DONE; waiting for test inputs ($W/test_inputs/READY) and test stage script"
  local waited=0
  while [ ! -f "$REPO/$W/test_inputs/READY" ] || [ ! -f scripts/_review_seqot_paper_recipe_20261107_test.sh ]; do
    if [ -f "$REPO/$W/test_inputs/FAILED" ]; then echo "test inputs failed" >> "$C/FAILED"; say "master: test inputs FAILED"; exit 1; fi
    sleep 300; waited=$((waited + 300))
    if [ $waited -ge 259200 ]; then echo "test inputs not ready after 72 h" >> "$C/FAILED"; say "master: test wait timeout"; exit 1; fi
  done
  say "master: test stage start (sha $(sha256sum scripts/_review_seqot_paper_recipe_20261107_test.sh | cut -c1-16))"
  if bash scripts/_review_seqot_paper_recipe_20261107_test.sh >> "$LOG/test_stage.log" 2>&1; then
    ts > "$C/TEST_DONE"; say "master: TEST_DONE"
  else
    echo "test stage rc=$?" >> "$C/FAILED"; say "master: test stage failed"; exit 1
  fi
}

case "${1:-}" in
  master) master ;;
  *) echo "usage: $0 master"; exit 2 ;;
esac
