#!/usr/bin/env bash
# SeqOT with the official BIT-MJY/SeqOT code/recipe on split v2 (2026-10-11).
#   config    our config copy (official config.yml @ 6e10697, only paths changed)
#   selftest  CPU checks: range images == official gen_depth_data / load_vertex,
#             overlap == official com_overlap_yaw, labels == official
#             gen_training_index.py (run unmodified), both training loops ==
#             official trainHandlers (bit-identical weights), sub-descriptors /
#             GeM eval == official scripts, trailing windows causal, resume,
#             ledger format (dry run)
#   prep      range images of train (all raw scans) and val (raw scans behind
#             each keyframe), 5th-scan frames, overlap labels (NPROC parallel)
#   smoke     GPU timing / memory on small subsets (separate dirs, not results)
#   train1    phase 1 (featureExtracter), paper recipe: lr 5e-6, epochs 1..20,
#             1 + 6 + 6, past windows; resumes
#   select1   score phase-1 epochs on v2 VAL (FOLLOW=1: wait for new epochs,
#             can run on a second GPU while train1 runs)
#   subdesc   official gen_sub_descriptors with the chosen phase-1 weights
#   train2    phase 2 (GeM), paper recipe: lr 5e-5, epochs 1..20; resumes
#   select2   score GeM epochs on v2 VAL
#   dump      per test sequence: range images (NPROC parallel) -> past-only
#             ledgers (seqotl2 = official L2, REPORTED; seqot = cosine record) -> range store
#             deleted unless KEEP_TEST_RANGE=1
#   aggregate PR summary (results/pdv2_seqot_official/odom/pr_summary_baselines.json)
# Every stage skips finished work. Run inside ring_baseline, e.g.
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_seqot_official_run.sh config selftest prep smoke
#   docker exec -e GPU=0 -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_seqot_official_run.sh train1 select1 subdesc train2 select2 dump aggregate
# Decisions 2026-10-11: paper recipe wherever code and paper differ (lr, epochs,
# Npos/Nneg, past training windows, negative rule, Eq. 13 overlap); official
# label counts per sequence; L2 for selection and the reported ledger.
# Optional: SEQOT_SELECT_METRIC=l2|cos (default l2), SEQOT_RANGE_ROOT (range
# store location, same value for every stage), NPROC (default 16).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=logs/seqot_official_pdv2
P="python3 scripts/_seqot_official_prep.py"
T="python3 scripts/_seqot_official_train.py"
D="python3 scripts/_seqot_official_dump.py"
NPROC=${NPROC:-16}
TRAIN1_FLAGS=${TRAIN1_FLAGS:-}
TRAIN2_FLAGS=${TRAIN2_FLAGS:-}
W=_external/seqot_official_pdv2
SEL=results/pdv2_seqot_official/train_selection.json
mkdir -p "$LOG"
say() { echo "[seqot-official] $* ($(date '+%F %T'))"; }

stage_config() {
  say config
  $P config > "$LOG/config.log" 2>&1
  tail -1 "$LOG/config.log"
}

stage_selftest() {
  say selftest
  CUDA_VISIBLE_DEVICES= $P selftest --n-scans 2 --out-json "$LOG/selftest_prep.json" \
    > "$LOG/selftest_prep.log" 2>&1
  CUDA_VISIBLE_DEVICES= $T selftest --parts recall train1 train2 resume descriptors \
    --out-json "$LOG/selftest_train.json" > "$LOG/selftest_train.log" 2>&1
  CUDA_VISIBLE_DEVICES= $T selftest --parts paper --out-json "$LOG/selftest_train_paper.json" \
    > "$LOG/selftest_train_paper.log" 2>&1   # separate process: keeps CPU RAM < 3 GB
  CUDA_VISIBLE_DEVICES= $D dry-run-cpu --seq nclt/2013-01-10 --n 80 --out-dir "$W/dryrun" \
    > "$LOG/selftest_dryrun.log" 2>&1
  rm -rf "$W/dryrun"
  for f in prep train train_paper dryrun; do tail -1 "$LOG/selftest_$f.log"; done
}

stage_prep() {
  say "prep index (train, val)"
  $P index --role train val > "$LOG/prep_index.log" 2>&1
  say "prep range images ($NPROC parallel)"
  $P jobs --role train val | xargs -P "$NPROC" -L 1 bash -c \
    'python3 scripts/_seqot_official_prep.py gen --role "$0" --seq "$1" --shard "$2" --nshards "$3" \
       >> "'"$LOG"'/prep_gen_$0_${1//\//_}.log" 2>&1'
  say "prep frames"
  python3 -c "import sys; sys.path.insert(0, 'scripts'); from _pdv2_pilot import TRAIN; [print(f'{s}/{q}') for s, q in TRAIN]" \
    | xargs -P "$(( NPROC < 8 ? NPROC : 8 ))" -L 1 bash -c \
      'python3 scripts/_seqot_official_prep.py frames --seq "$0" >> "'"$LOG"'/prep_frames_${0//\//_}.log" 2>&1'
  say "prep overlap ($NPROC parallel)"
  $P ovjobs --nshards "$NPROC" | xargs -P "$NPROC" -L 1 bash -c \
    'python3 scripts/_seqot_official_prep.py overlap --seq "$0" --shard "$1" --nshards "$2" \
       >> "'"$LOG"'/prep_overlap_${0//\//_}.log" 2>&1'
  say "prep labels"
  $P labels > "$LOG/prep_labels.log" 2>&1
  $P status | tail -5
}

stage_smoke() {
  say smoke
  $T smoke > "$LOG/smoke.log" 2>&1
  grep '^SMOKE' "$LOG/smoke.log"
}

stage_train1() {
  $P status > /dev/null || { echo "prep incomplete"; exit 1; }
  if [ -f "$W/runs/stage1/finished" ]; then say "train1: finished, skip"; return; fi
  say "train1 $TRAIN1_FLAGS"
  $T train1 $TRAIN1_FLAGS >> "$LOG/train1.log" 2>&1
  tail -1 "$LOG/train1.log"
}

stage_select1() {
  say select1
  if [ "${FOLLOW:-0}" = 1 ]; then
    $T select1 --follow >> "$LOG/select1.log" 2>&1
  else
    $T select1 >> "$LOG/select1.log" 2>&1
  fi
  tail -1 "$LOG/select1.log"
}

stage_subdesc() {
  say subdesc
  $T subdesc >> "$LOG/subdesc.log" 2>&1
  tail -1 "$LOG/subdesc.log"
}

stage_train2() {
  if [ -f "$W/runs/stage2/finished" ]; then say "train2: finished, skip"; return; fi
  say "train2 $TRAIN2_FLAGS"
  $T train2 $TRAIN2_FLAGS >> "$LOG/train2.log" 2>&1
  tail -1 "$LOG/train2.log"
}

stage_select2() {
  say select2
  $T select2 >> "$LOG/select2.log" 2>&1
  tail -1 "$LOG/select2.log"
}

stage_dump() {
  python3 -c "import json,sys; sys.exit(0 if json.load(open('$SEL')).get('finished') else 1)" \
    || { echo "selection not finished"; exit 1; }
  for key in $(python3 -c "import sys; sys.path.insert(0, 'scripts'); from _pdv2_pilot import TEST; [print(f'{s}/{q}') for s, q in TEST]"); do
    s=${key%%/*}; q=${key#*/}
    if [ -f "results/pdv2_seqot_official/odom/dump_seqot_${s}_${q}.npz" ] && \
       [ -f "results/pdv2_seqot_official/odom/dump_seqotl2_${s}_${q}.npz" ]; then
      continue
    fi
    say "dump $key: range images"
    $P index --role test_odom --seq "$key" >> "$LOG/dump_index.log" 2>&1
    $P jobs --role test_odom --seq "$key" --chunk 1000 | xargs -P "$NPROC" -L 1 bash -c \
      'python3 scripts/_seqot_official_prep.py gen --role "$0" --seq "$1" --shard "$2" --nshards "$3" \
         >> "'"$LOG"'/dump_gen_${1//\//_}.log" 2>&1'
    say "dump $key: ledgers"
    $D dump --seqs "$key" >> "$LOG/dump.log" 2>&1
    tail -1 "$LOG/dump.log"
    if [ "${KEEP_TEST_RANGE:-0}" != 1 ]; then
      rm -rf "${SEQOT_RANGE_ROOT:-$W/range}/test_odom/${s}_${q}"
    fi
  done
  say "dump: all sequences done"
}

stage_aggregate() {
  say aggregate
  $D aggregate > "$LOG/aggregate.log" 2>&1
  tail -3 "$LOG/aggregate.log"
}

[ $# -gt 0 ] || set -- config selftest prep train1 select1 subdesc train2 select2 dump aggregate
for s in "$@"; do
  "stage_$s"
done
say "DONE: $*"
