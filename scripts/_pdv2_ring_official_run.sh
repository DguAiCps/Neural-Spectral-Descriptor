#!/usr/bin/env bash
# Official RING / RING++ / RING#-L on split v2 (prepared 2026-10-11, NOT yet run).
# Report: logs/baseline_official_20261011/ring_family.md (section 10: decisions)
#
# Run inside docker ring_baseline from the repo root, one stage at a time or
# several in order (every step is resumable: finished outputs are skipped):
#   bash scripts/_pdv2_ring_official_run.sh checks
#   bash scripts/_pdv2_ring_official_run.sh ring-bev ring-dump ring-agg
#   bash scripts/_pdv2_ring_official_run.sh rs-prep rs-microcheck rs-train rs-valbev rs-val rs-select rs-dump rs-agg
# Two GPUs: RS_VAL_DEVICE=cuda:1 makes rs-train validate every epoch on
# GPU 1 while training runs on RS_TRAIN_DEVICE (GPU 0): a watcher validates
# each checkpoint as it appears; rs-valbev / rs-val are then not needed.
#   RS_VAL_DEVICE=cuda:1 bash scripts/_pdv2_ring_official_run.sh rs-prep rs-train rs-select rs-dump rs-agg
# Environment:
#   COND=odom|gt          keyframe condition (default odom = KISS-ICP keyframes)
#   EPOCHS="5 10 ..."     RING#-L epochs to validate / select over (default 1..30)
#   RS_TRAIN_DEVICE       GPU for RING#-L training (default cuda:0; always pinned
#                         with CUDA_VISIBLE_DEVICES so the trainer's DataParallel
#                         stays on one GPU as on the 1-GPU machine)
#   RS_VAL_DEVICE         e.g. cuda:1 -> concurrent per-epoch validation (above)
#   NSC_MICRO_BATCH       RING#-L gradient accumulation (default 2 for 16 GB;
#                         1 = upstream full-batch step, fine on 96 GB cards)
#   JOBS                  RING/RING++ BEV sequences built in parallel (default 1);
#                         each job gets nproc/JOBS threads for k-NN / eigen
# GPU stages: ring-bev (small CUDA kernels + CPU k-NN), ring-dump, rs-microcheck,
# rs-train, rs-val, rs-dump.  CPU only: checks (first two lines), rs-prep,
# rs-valbev, rs-select, *-agg.
set -u
cd "$(dirname "$0")/.."
ulimit -s 81920            # RING README: voxelocc / voxelfeat keep per-point arrays on the stack
COND=${COND:-odom}
EPOCHS=${EPOCHS:-$(seq 1 30)}
JOBS=${JOBS:-1}
RS_TRAIN_GPU=${RS_TRAIN_DEVICE:-cuda:0}; RS_TRAIN_GPU=${RS_TRAIN_GPU#cuda:}
RS_VAL_GPU=${RS_VAL_DEVICE:-}; RS_VAL_GPU=${RS_VAL_GPU#cuda:}
LOG=logs/ring_official
mkdir -p "$LOG"
TEST=$(python3 scripts/_ring_official_pr.py roster)
VALSEQ=$(python3 scripts/_pdv2_pilot.py roster --role val)
FAILED=()

step() {  # step <log-name> <command...>
  local name=$1; shift
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG/chain.log"
  if ! "$@" > "$LOG/$name.log" 2>&1; then
    echo "  FAILED ($LOG/$name.log)" | tee -a "$LOG/chain.log"
    FAILED+=("$name")
  fi
}

ring_bev_parallel() {  # JOBS > 1: several sequences at once, threads split evenly
  local threads=$(( $(nproc) / JOBS )); [ "$threads" -lt 1 ] && threads=1
  echo "[$(date '+%F %T')] ring-bev: $JOBS jobs x $threads threads" | tee -a "$LOG/chain.log"
  local out
  out=$(printf '%s\n' $TEST | xargs -P "$JOBS" -I{} env OMP_NUM_THREADS="$threads" \
      MKL_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" LOKY_MAX_CPU_COUNT="$threads" \
      bash -c 'n=${2//\//_}; python3 scripts/_ring_official_pr.py bev --cond "$1" --seq "$2" \
               > "$0/bev_$1_$n.log" 2>&1 && echo "OK $2" || echo "FAILED bev_$1_$n"' \
      "$LOG" "$COND" {})
  echo "$out" | tee -a "$LOG/chain.log"
  for f in $(echo "$out" | awk '/^FAILED/ {print $2}'); do FAILED+=("$f"); done
}

rs_train_with_val_watch() {  # training on RS_TRAIN_GPU, per-epoch validation on RS_VAL_GPU
  local exitf="$LOG/rs_train.exit"
  rm -f "$exitf"
  echo "[$(date '+%F %T')] rs-train on GPU $RS_TRAIN_GPU, validation watcher on GPU $RS_VAL_GPU" | tee -a "$LOG/chain.log"
  ( CUDA_VISIBLE_DEVICES=$RS_TRAIN_GPU python3 scripts/_ringsharp_official.py train > "$LOG/rs_train.log" 2>&1
    echo $? > "$exitf" ) &
  local tpid=$!
  for s in $VALSEQ; do   # CPU, overlaps with the first training epoch
    step "rs_valbev_${s//\//_}" env CUDA_VISIBLE_DEVICES="$RS_VAL_GPU" python3 scripts/_ringsharp_official.py val-bev --seq "$s"
  done
  # shellcheck disable=SC2086
  step rs_val_watch env CUDA_VISIBLE_DEVICES="$RS_VAL_GPU" python3 scripts/_ringsharp_official.py \
      val-watch --exit-file "$exitf" --epochs $EPOCHS --log-dir "$LOG"
  wait "$tpid"
  if [ "$(cat "$exitf" 2>/dev/null)" != "0" ]; then
    echo "  FAILED rs_train (exit $(cat "$exitf" 2>/dev/null), $LOG/rs_train.log)" | tee -a "$LOG/chain.log"
    FAILED+=(rs_train)
  fi
}

for stage in "$@"; do
  case $stage in
    checks)
      step check_ring_cpu env CUDA_VISIBLE_DEVICES= python3 scripts/_ring_official_pr.py check
      step check_rs_cpu env CUDA_VISIBLE_DEVICES= python3 scripts/_ringsharp_official.py check
      step check_ring_gpu python3 scripts/_ring_official_pr.py gpu-check ;;
    ring-bev)
      if [ "$JOBS" -gt 1 ]; then ring_bev_parallel; else
        for s in $TEST; do step "bev_${COND}_${s//\//_}" python3 scripts/_ring_official_pr.py bev --cond "$COND" --seq "$s"; done
      fi ;;
    ring-dump)
      for s in $TEST; do step "dump_ring_${COND}_${s//\//_}" python3 scripts/_ring_official_pr.py dump --cond "$COND" --seq "$s"; done ;;
    ring-agg|rs-agg)
      step "aggregate_${COND}" python3 scripts/_ring_official_pr.py aggregate --cond "$COND" ;;
    rs-prep)
      step rs_prep python3 scripts/_ringsharp_official.py prep ;;
    rs-microcheck)
      # gradient equivalence of the memory patches (full batch vs NSC_MICRO_BATCH=2);
      # needs about 15 GB VRAM for the full-batch reference, so run with the GPU free
      step rs_microcheck env CUDA_VISIBLE_DEVICES="$RS_TRAIN_GPU" python3 _external/ringsharp_env/check_micro.py ;;
    rs-train)
      if [ -n "$RS_VAL_GPU" ]; then rs_train_with_val_watch; else
        step rs_train env CUDA_VISIBLE_DEVICES="$RS_TRAIN_GPU" python3 scripts/_ringsharp_official.py train
      fi ;;
    rs-valbev)
      for s in $VALSEQ; do step "rs_valbev_${s//\//_}" python3 scripts/_ringsharp_official.py val-bev --seq "$s"; done ;;
    rs-val)
      for e in $EPOCHS; do for s in $VALSEQ; do
        step "rs_val_e${e}_${s//\//_}" python3 scripts/_ringsharp_official.py val --epoch "$e" --seq "$s"
      done; done ;;
    rs-select)
      # shellcheck disable=SC2086
      step rs_select python3 scripts/_ringsharp_official.py select --epochs $EPOCHS ;;
    rs-dump)
      for s in $TEST; do step "rs_dump_${COND}_${s//\//_}" python3 scripts/_ringsharp_official.py dump --cond "$COND" --seq "$s"; done ;;
    *) echo "unknown stage $stage"; exit 2 ;;
  esac
done
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED steps: ${FAILED[*]}" | tee -a "$LOG/chain.log"; exit 1
fi
echo "RING_OFFICIAL_RUN_DONE $*" | tee -a "$LOG/chain.log"
