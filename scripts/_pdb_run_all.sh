#!/usr/bin/env bash
# Baseline evaluation chain for the place-disjoint 42-seq test roster.
# Order: fast methods first, the ~20 h RING/RING++ block last. Resumable:
# per-sequence JSONs are skipped when present. One sequence per process.
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdb_run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs results/pd2026_baselines

SEQS=$(python3 -c "
import json
m = json.load(open('scripts/_pd_test_cache_manifest.json'))
print(' '.join(f\"{s['sensor']}/{s['sequence']}\" for s in m['sequences']))
")
PD_TRAIN_VAL="kitti/01 kitti/02 kitti/06 kitti/07 kitti/09 \
  nclt/2012-05-11 nclt/2012-08-04 nclt/2012-11-04 nclt/2012-11-16 nclt/2013-02-23 nclt/2012-01-08 \
  helipr/Bridge01 helipr/Bridge02 helipr/Bridge03 helipr/Roundabout01 helipr/Roundabout02 helipr/Roundabout03 \
  mulran/Sejong01 mulran/Sejong02 mulran/Sejong03"

run() {
  local log="$1"; shift
  echo "[pdb] === $* ($(date '+%F %T'))"
  "$@" 2>&1 | tee "logs/${log}.log"
}
tag() { echo "$1" | tr / _; }

echo "[pdb] ---- stage 1: SC++ ----"
for sq in $SEQS; do
  s=${sq%%/*}; q=${sq##*/}
  [ -f "results/pd2026_baselines/scpp_${s}_${q}.json" ] && continue
  run "pdb_scpp_$(tag "$sq")" python3 scripts/_pdb_scpp_eval.py --seqs "$sq"
done

echo "[pdb] ---- stage 2: RING# (official-retrain weights, zero-shot) ----"
for sq in $SEQS; do
  s=${sq%%/*}; q=${sq##*/}
  [ -f "results/pd2026_baselines/ringsharp_${s}_${q}.json" ] && continue
  run "pdb_ringsharp_$(tag "$sq")" python3 scripts/_pdb_ringsharp_eval.py "$sq"
done

echo "[pdb] ---- stage 3: BEVPlace++ pd retrain + eval ----"
if [ ! -f baselines/weights/bevplace_pd2026.pth ]; then
  run pdb_bevplace_train python3 scripts/_pdb_bevplace_train.py \
    --output baselines/weights/bevplace_pd2026.pth
fi
for sq in $SEQS; do
  run "pdb_bevplace_$(tag "$sq")" python3 scripts/_pdb_bevplace_eval.py \
    --weights baselines/weights/bevplace_pd2026.pth --seqs "$sq"
done

echo "[pdb] ---- stage 4: SeqOT prepare + pd retrain + eval ----"
for sq in $PD_TRAIN_VAL $SEQS; do
  run "pdb_seqot_prep_$(tag "$sq")" python3 scripts/_pdb_seqot_prepare.py "$sq"
done
if [ ! -f _external/seqot_runs_pd2026/full/gem_latest.pth.tar ]; then
  run pdb_seqot_train python3 scripts/_pdb_seqot_train.py \
    --run-dir _external/seqot_runs_pd2026/full
fi
for sq in $SEQS; do
  s=${sq%%/*}; q=${sq##*/}
  [ -f "results/pd2026_baselines/seqot_${s}_${q}.json" ] && continue
  run "pdb_seqot_$(tag "$sq")" python3 scripts/_pdb_seqot_eval.py "$sq" \
    --seqot-checkpoint _external/seqot_runs_pd2026/full/seqot_latest.pth.tar \
    --gem-checkpoint _external/seqot_runs_pd2026/full/gem_latest.pth.tar
done

echo "[pdb] ---- stage 5: RING/RING++ verification gate (helipr/Town01) ----"
if [ ! -f "results/pd2026_baselines/ringpp_helipr_Town01.json" ]; then
  run pdb_ring_verify python3 scripts/_pdb_ring_eval.py helipr/Town01 --variant both
fi
python3 - <<'PYEOF'
import json, sys
new = json.load(open('results/pd2026_baselines/ringpp_helipr_Town01.json'))
ref = json.load(open('results/ringpp_eval.json'))['helipr_Town01']
new_r1 = new['recall@1']
ref_r1 = ref['recall@1']
delta = abs(new_r1 - ref_r1)
print(f'[pdb] RING++ Town01 verification: new={new_r1:.4f} ref={ref_r1:.4f} delta={delta:.4f}')
sys.exit(0 if delta <= 0.02 else 1)
PYEOF

echo "[pdb] ---- stage 6: RING/RING++ full roster (~20 h) ----"
for sq in $SEQS; do
  s=${sq%%/*}; q=${sq##*/}
  [ -f "results/pd2026_baselines/ringpp_${s}_${q}.json" ] && \
  [ -f "results/pd2026_baselines/ring_${s}_${q}.json" ] && continue
  run "pdb_ring_$(tag "$sq")" python3 scripts/_pdb_ring_eval.py "$sq" --variant both
done
echo "[pdb] ALL BASELINES COMPLETE ($(date '+%F %T'))"
