#!/usr/bin/env bash
# HeLiOS with the official HeLiOS code/recipe, retrained on split v2 (2026-10-11).
#   build     CPU reference builds: conda-forge PCL prefix, official toolbox and
#             overlap binaries, libhelios_toolbox.so (official utility.cpp),
#             pytorch-metric-learning 1.1.2 (official pin) -> _external/helios_env
#             (MinkowskiEngine / sptr / torch_scatter / torch_cluster for sm_120:
#             run scripts/_helios_port_setup.sh on the HOST first)
#   selftest  CPU checks: toolbox adapter == official toolbox binary (6 sensors,
#             HeLiPR with undistortion); overlap replica == official overlap
#             binary + construct_query_dict; official training code on CPU
#             (imports, losses with pml 1.1.2, collate, multistage step, resume);
#             batched val R@1 == compute_recall_multi_k; ledger format
#   prep      index (TRAIN_SPACING, DESKEW) + official toolbox scans for
#             train / val / test_odom keyframes (NPROC parallel, resumable)
#   overlap   official overlap labels -> data/{training,validation}.pickle
#   smoke     1 epoch on the first 300 scans per folder (GPU time / memory)
#   train     official training, 80 epochs; resumes an unfinished run
#   select    v2-VAL family-equal R@1 of every saved checkpoint; chosen =
#             val_best (user decision 2026-10-11); official _best also recorded
#   dump      past-only pr-dump-v1 ledgers on the 46 odometry test sequences
#   aggregate PR summary (results/pdv2_helios_official/odom/pr_summary_baselines.json)
# Every stage skips finished work. Run inside the container from the repo root:
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_helios_official_run.sh build selftest prep overlap smoke
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_helios_official_run.sh train select dump aggregate
# GPU=<id> picks the GPU (the official trainer uses one).
# SEED=<n> (default 1) is the training seed.  SEED=1 keeps every path of the
# first run.  Other seeds use their own logs / smoke / runs / config / results
# (suffix _s<n>) and SHARE the prepared scans, pickles and overlap files of seed
# 1: for SEED != 1 the build / prep / overlap stages only check that those exist
# (no rebuild, no re-index), e.g.
#   docker exec -e SEED=2 -e GPU=1 -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_helios_official_run.sh smoke train select dump aggregate
#   paths     print the resolved locations for the current SEED (read-only)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
SEED=${SEED:-1}
export SEED
if [ "$SEED" = 1 ]; then SFX=""; else SFX="_s$SEED"; fi
LOG=logs/helios_official_pdv2$SFX
W=_external/helios_official_pdv2            # shared by all seeds (scans, pickles, overlap)
SMOKE=$W/smoke$SFX
REC=results/pdv2_helios_official$SFX/train_record.json
Pp="python3 scripts/_helios_official_prep.py"
O="python3 scripts/_helios_official_overlap.py"
T="python3 scripts/_helios_official_train.py"
D="python3 scripts/_helios_official_dump.py"
NPROC=${NPROC:-12}
TRAIN_SPACING=${TRAIN_SPACING:-3}     # official toolbox distanceThreshold; 0 = every keyframe
DESKEW=${DESKEW:-helipr}              # helipr | none
CHOSEN=${CHOSEN:-val_best}            # val_best | official_best
mkdir -p "$LOG"
say() { echo "[helios-official] $* ($(date '+%F %T'))"; }

shared_only() {   # SEED != 1: seed-1 artifacts are reused, never rewritten
  [ "$SEED" != 1 ]
}

stage_build() {
  if shared_only; then
    say "build: shared with seed 1, check only"
    for f in _external/helios_env/build/libhelios_toolbox.so _external/helios_env/pml112/pytorch_metric_learning; do
      [ -e "$f" ] || { echo "missing $f (run the build stage with SEED=1)"; exit 1; }
    done
    python3 -c "import MinkowskiEngine, sptr_cuda, torch_scatter, torch_cluster; print('ME', MinkowskiEngine.__version__)"
    return
  fi
  say build
  bash _external/helios_env/build_reference.sh > "$LOG/build_reference.log" 2>&1
  [ -d _external/helios_env/pml112/pytorch_metric_learning ] || \
    pip install -q --no-deps --target _external/helios_env/pml112 pytorch-metric-learning==1.1.2 \
      >> "$LOG/build_reference.log" 2>&1
  python3 -c "import MinkowskiEngine, sptr_cuda, torch_scatter, torch_cluster; print('ME', MinkowskiEngine.__version__)"
  tail -1 "$LOG/build_reference.log"
}

oj() {   # selftest JSON: default location for seed 1, the seed's log dir otherwise
  shared_only && echo "--out-json $LOG/selftest_$1.json" || true
}

stage_selftest() {
  say selftest
  CUDA_VISIBLE_DEVICES= $Pp selftest $(oj prep) > "$LOG/selftest_prep.log" 2>&1
  CUDA_VISIBLE_DEVICES= $O selftest $(oj overlap) > "$LOG/selftest_overlap.log" 2>&1
  CUDA_VISIBLE_DEVICES= $T --selftest $(oj train) > "$LOG/selftest_train.log" 2>&1
  CUDA_VISIBLE_DEVICES= $D selftest $(oj dump) > "$LOG/selftest_dump.log" 2>&1
  for f in prep overlap train dump; do tail -1 "$LOG/selftest_$f.log"; done
}

stage_prep() {
  if shared_only; then
    say "prep: shared with seed 1, status only"
    $Pp status | tail -4
    return
  fi
  say "prep index (train spacing $TRAIN_SPACING m, deskew $DESKEW)"
  $Pp index --train-spacing "$TRAIN_SPACING" --deskew "$DESKEW" > "$LOG/prep_index.log" 2>&1
  say "prep gen ($NPROC parallel)"
  $Pp jobs | xargs -r -P "$NPROC" -L 1 bash -c \
    'python3 scripts/_helios_official_prep.py gen --role "$0" --seq "$1" --shard "$2" --nshards "$3" \
       >> "'"$LOG"'/prep_gen_$0_${1//\//_}.log" 2>&1'
  $Pp status | tail -4
}

stage_overlap() {
  if shared_only; then
    say "overlap: shared with seed 1, check only"
    for f in "$W/data/training.pickle" "$W/data/validation.pickle"; do
      [ -f "$f" ] || { echo "missing $f (run prep overlap with SEED=1)"; exit 1; }
    done
    return
  fi
  $Pp status > /dev/null || { echo "scans incomplete"; exit 1; }
  for role in train val; do
    say "overlap $role"
    [ -f "$W/overlap/$role/vox.npz" ] || $O vox --role "$role" > "$LOG/overlap_vox_$role.log" 2>&1
    seq 0 $((NPROC - 1)) | xargs -r -P "$NPROC" -I{} bash -c \
      "python3 scripts/_helios_official_overlap.py pairs --role $role --shard {} --nshards $NPROC \
         >> $LOG/overlap_pairs_${role}_{}.log 2>&1"
    $O pickle --role "$role" > "$LOG/overlap_pickle_$role.log" 2>&1
    tail -3 "$LOG/overlap_pickle_$role.log"
  done
}

stage_smoke() {
  say smoke
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $T --smoke 300 > "$LOG/smoke_train.log" 2>&1
  python3 -c "import json; e=json.load(open('$SMOKE/train_record.json'))['epochs'][-1]; \
print('smoke epoch:', {k: e.get(k) for k in ('train', 'train_batches', 'val', 'val_batches', 'cuda_peak_gb', 'val_ap')})"
}

stage_train() {
  if [ -f "$REC" ]; then
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('$REC')).get('finished') else 1)"; then
      say "train: finished, skip"; return
    fi
    RUN=$(python3 -c "import json; print(json.load(open('$REC'))['_meta']['run_dir'])")
    say "train: resume $RUN"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $T --resume "$RUN" >> "$LOG/train.log" 2>&1
  else
    say "train: start"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $T > "$LOG/train.log" 2>&1
  fi
  tail -1 "$LOG/train.log"
}

stage_select() {
  say "select (chosen=$CHOSEN)"
  $D select --chosen "$CHOSEN" >> "$LOG/select.log" 2>&1
  tail -1 "$LOG/select.log"
}

stage_dump() {
  say dump
  $D dump >> "$LOG/dump.log" 2>&1
  tail -1 "$LOG/dump.log"
}

stage_aggregate() {
  say aggregate
  $D aggregate > "$LOG/aggregate.log" 2>&1
  tail -2 "$LOG/aggregate.log"
}

stage_paths() {   # read-only: resolved locations for this SEED
  echo "SEED=$SEED LOG=$LOG W=$W SMOKE=$SMOKE REC=$REC"
  python3 - <<'PY'
import sys
sys.path[:0] = ["scripts"]
import _helios_official_prep as P, _helios_official_overlap as O
import _helios_official_train as T, _helios_official_dump as D
for k, v in (("prep.DATA", P.DATA), ("prep.FILL_SEED", P.FILL_SEED), ("overlap.OVL", O.OVL),
             ("train.SEED", T.SEED), ("train.CONFIG", T.CONFIG), ("train.RUNS", T.RUNS),
             ("train.SMOKE", T.SMOKE), ("train.RECORD", T.RECORD), ("dump.RESULTS", D.RESULTS),
             ("dump.OUT", D.OUT), ("dump.RECORD", D.RECORD), ("dump.SELECTION", D.SELECTION)):
    print(f"{k} = {v}")
PY
}

[ $# -gt 0 ] || set -- build selftest prep overlap train select dump aggregate
for s in "$@"; do
  "stage_$s"
done
say "DONE: $*"
