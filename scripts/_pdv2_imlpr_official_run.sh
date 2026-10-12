#!/usr/bin/env bash
# ImLPR with the official ImLPR code/recipe on split v2 (2026-10-11).
#   selftest  CPU checks: RIV adapter == official generator (incl. 5 s submaps,
#             MulRan end-to-end), tuples == official construct_query_dict,
#             patched collate == official collate, CPU dry run of the trainer
#             with resume, ledger format on 80 test keyframes
#   hub       pin DINOv2 ViT-S/14 (torch.hub code + dinov2_vits14_pretrain.pth)
#             under $W/torch_hub (copied from TORCH_HOME_SRC or downloaded)
#   index     keyframe lists / poses for train (3 m thinning), val, test_odom
#   riv       RIVs of train and val (NPROC parallel shards, resumable)
#   pickle    official training tuples (PICKLE_MODE=global|per_sequence) + config
#   smoke     2 full-size training steps + val on 300 frames/sequence
#             (GPU time and memory; separate run dir, not used for results)
#   train     100 epochs, `_best` by v2-val family-equal R@1 every 5 epochs;
#             resumes an unfinished run
#   test      per test sequence: RIVs (NPROC) -> descriptors + pr-dump-v1
#             ledger (GPU) -> RIVs deleted unless KEEP_TEST_RIV=1
#   aggregate PR summary (results/pdv2_imlpr_official/odom/pr_summary_baselines.json)
# Every stage skips finished work.  Run inside the container, e.g.
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_imlpr_official_run.sh hub index selftest riv pickle smoke
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdv2_imlpr_official_run.sh train test aggregate
# SEED (default 1): seed 1 uses the original locations; SEED=k writes everything
# training-dependent to *_s{k} (config, runs, smoke, logs, results, descriptors,
# ledgers) and shares RIVs / index / pickles / hub with seed 1 (for k != 1:
# `index` skips if present, `pickle` only writes the seed's config, `test`
# keeps the shared test RIVs unless KEEP_TEST_RIV=0).
# Knobs: NPROC (RIV processes, default 12), NUM_WORKERS (official DataLoader
# workers, default 4; official 16 needs ~110 GB of shared memory),
# PICKLE_MODE (default global = official), CUDA_VISIBLE_DEVICES (official
# trainer uses DataParallel over all visible GPUs), IMLPR_NCLT_INTENSITY
# (invert255 default | x255raw; set for index and test), IMLPR_PAIR_GEOMETRY
# (sensor default | config = official collate geometry; set for smoke/train),
# IMLPR_AGGREGATOR (code default = 64 clusters x 128, as the released weights |
# paper = 128 x 64), IMLPR_BACKBONE (code default = code path, as the released
# weights | paper = last two DINOv2 blocks only).
# Rule: released code with its default behaviour; exceptions = protocol, fixes
# needed to run, and lr 5e-4 (proven by the released weights);
# logs/baseline_official_20261011/imlpr.md section 12.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
W=_external/imlpr_official_pdv2
export TORCH_HOME="$PWD/$W/torch_hub"
DINO_SHA=b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9
SEED=${SEED:-1}
export SEED
SFX=""
[ "$SEED" -eq 1 ] || SFX="_s$SEED"
LOG=logs/imlpr_official_pdv2$SFX
SEL=results/pdv2_imlpr_official$SFX/train_selection.json
R="python3 scripts/_imlpr_official_riv.py"
T="python3 scripts/_imlpr_official_train.py"
D="python3 scripts/_imlpr_official_dump.py"
NPROC=${NPROC:-12}
NUM_WORKERS=${NUM_WORKERS:-4}
KEEP_DEFAULT=0
[ "$SEED" -eq 1 ] || KEEP_DEFAULT=1   # shared test RIVs are kept for seeds != 1
PICKLE_MODE=${PICKLE_MODE:-global}
export IMLPR_PICKLE_MODE=$PICKLE_MODE
mkdir -p "$LOG"
say() { echo "[imlpr-official] $* ($(date '+%F %T'))"; }

gen_jobs() {  # stdin: "role seq shard nshards" lines -> parallel single-thread workers
  xargs -r -P "$NPROC" -L 1 bash -c \
    'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     python3 scripts/_imlpr_official_riv.py gen --role "$0" --seq "$1" --shard "$2" --nshards "$3" \
       >> "'"$LOG"'/riv_$0_${1//\//_}.log" 2>&1'
}

stage_selftest() {
  say selftest
  [ -f "$W/data/test_odom/nclt_2013-01-10/index.npz" ] || stage_index
  CUDA_VISIBLE_DEVICES= $R selftest --n-scans 2 \
    --out-json "$LOG/selftest_riv.json" > "$LOG/selftest_riv.log" 2>&1
  tail -1 "$LOG/selftest_riv.log"
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 $T selftest \
    --out-json "$LOG/selftest_train.json" > "$LOG/selftest_train.log" 2>&1
  tail -1 "$LOG/selftest_train.log"
  rm -rf "$W/dryrun$SFX"
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 $D dry-run-cpu --seq nclt/2013-01-10 --n 80 \
    --out-dir "$W/dryrun$SFX" > "$LOG/selftest_dryrun.log" 2>&1
  tail -1 "$LOG/selftest_dryrun.log"
}

stage_hub() {
  H="$TORCH_HOME/hub"
  if [ ! -f "$H/checkpoints/dinov2_vits14_pretrain.pth" ]; then
    SRC="${TORCH_HOME_SRC:-$HOME/.cache/torch}/hub"
    if [ -f "$SRC/checkpoints/dinov2_vits14_pretrain.pth" ]; then
      say "hub: copy from $SRC"
      mkdir -p "$H/checkpoints"
      cp -r "$SRC/facebookresearch_dinov2_main" "$H/"
      cp "$SRC/checkpoints/dinov2_vits14_pretrain.pth" "$H/checkpoints/"
    else
      say "hub: download (needs internet)"
      python3 -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"
    fi
  fi
  echo "$DINO_SHA  $H/checkpoints/dinov2_vits14_pretrain.pth" | sha256sum -c -
  (cd "$H/facebookresearch_dinov2_main" && find . -name "*.py" | sort | xargs cat | sha256sum) \
    | tee "$LOG/dinov2_hub_code.sha256"
}

stage_index() {
  if [ "$SEED" -ne 1 ] && [ -f "$W/data/test_odom/nclt_2013-02-23/index.npz" ]; then
    say "index: shared with seed 1, skip"; return
  fi
  say index
  $R index > "$LOG/index.log" 2>&1
  tail -3 "$LOG/index.log"
}

stage_riv() {
  say "riv train+val ($NPROC parallel)"
  $R jobs --role train val | gen_jobs
  $R status --role train val | tail -3
}

stage_pickle() {
  $R status --role train > /dev/null || { echo "train RIVs incomplete"; exit 1; }
  if [ "$SEED" -ne 1 ]; then   # shared pickle of seed 1, only the seed's config
    [ -f "$W/data/training_pdv2_$PICKLE_MODE.pickle" ] || { echo "shared pickle missing"; exit 1; }
    say "pickle: shared training_pdv2_$PICKLE_MODE.pickle; config only (num_workers $NUM_WORKERS)"
  else
    say "pickle ($PICKLE_MODE) + config (num_workers $NUM_WORKERS)"
    $T pickle --pickle-mode "$PICKLE_MODE" > "$LOG/pickle_$PICKLE_MODE.log" 2>&1
    tail -1 "$LOG/pickle_$PICKLE_MODE.log"
  fi
  $T config --pickle-mode "$PICKLE_MODE" --num-workers "$NUM_WORKERS" > "$LOG/config.log" 2>&1
  tail -1 "$LOG/config.log"
}

stage_config() {   # rewrite only the seed's config (e.g. NUM_WORKERS=2 before train)
  say "config ($PICKLE_MODE, num_workers $NUM_WORKERS)"
  $T config --pickle-mode "$PICKLE_MODE" --num-workers "$NUM_WORKERS" > "$LOG/config.log" 2>&1
  tail -1 "$LOG/config.log"
}

stage_paths() {    # resolved locations for this SEED (no side effects)
  echo "SEED=$SEED LOG=$LOG SEL=$SEL"
  echo "smoke=$W/smoke$SFX dryrun=$W/dryrun$SFX test_ledgers=results/pdv2_imlpr_official$SFX/odom"
  python3 -c "import sys; sys.path.insert(0, 'scripts'); import _imlpr_official_train as T, _imlpr_official_dump as D; \
r = lambda p: str(p.relative_to(T.REPO)); \
print('py: SEED', T.SEED, D.SEED, '| RESULTS', r(T.RESULTS), r(D.RESULTS), '| SELECTION', r(T.SELECTION), \
'| CONFIG', r(T.CONFIG), '| RUNS', r(T.RUNS), '| SMOKE', r(T.SMOKE), '| OUT', r(D.OUT), '| DESC', r(D.DESC), \
'| shared: DATA', r(T.DATA), 'MODEL_CONFIG', r(T.MODEL_CONFIG))"
}

stage_smoke() {
  say smoke
  rm -rf "$W/smoke$SFX"
  $T smoke --pickle-mode "$PICKLE_MODE" > "$LOG/smoke.log" 2>&1
  grep -E "collate|step|val |family|cuda" "$LOG/smoke.log" | tail -20
}

stage_train() {
  $R status --role train val > /dev/null || { echo "train/val RIVs incomplete"; exit 1; }
  if [ -f "$SEL" ]; then
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('$SEL')).get('finished') else 1)"; then
      say "train: finished, skip"; return
    fi
    say "train: resume"
    $T train --pickle-mode "$PICKLE_MODE" --resume >> "$LOG/train.log" 2>&1
  else
    say "train: start"
    $T train --pickle-mode "$PICKLE_MODE" > "$LOG/train.log" 2>&1
  fi
  tail -1 "$LOG/train.log"
}

stage_test() {
  local keys key name
  keys=$(python3 -c "import sys; sys.path.insert(0, 'scripts'); from _pdv2_pilot import TEST; \
print(' '.join(f'{s}/{q}' for s, q in TEST))")
  for key in $keys; do
    name=${key//\//_}
    if [ -f "results/pdv2_imlpr_official$SFX/odom/dump_imlpr_${name}.npz" ]; then
      continue
    fi
    say "test $key: riv"
    $R jobs --role test_odom --seq "$key" | gen_jobs
    say "test $key: dump"
    $D dump --seqs "$key" >> "$LOG/dump.log" 2>&1
    if [ "${KEEP_TEST_RIV:-$KEEP_DEFAULT}" != 1 ]; then
      find "$W/data/test_odom/$name" -name '[0-9]*.npy' -delete
    fi
  done
  say "test: all ledgers written"
}

stage_aggregate() {
  say aggregate
  $D aggregate > "$LOG/aggregate.log" 2>&1
  tail -2 "$LOG/aggregate.log"
}

[ $# -gt 0 ] || set -- hub index selftest riv pickle train test aggregate
for s in "$@"; do
  "stage_$s"
done
say "DONE: $*"
