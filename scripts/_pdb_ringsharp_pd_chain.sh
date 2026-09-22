#!/usr/bin/env bash
# RING# pd2026 post-training chain: (1) pd-val evaluation of every per-epoch
# checkpoint, (2) pre-declared best-checkpoint selection + sha256 freeze,
# (3) frozen 42-seq test roster evaluation with the selected weight only.
# Resumable: per-sequence JSONs are skipped when present.
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_pdb_ringsharp_pd_chain.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs

WDIR=_external/RINGSharp/tools/results/weights/ring_sharp_l_nsc_pd2026
VAL_SEQS="kitti/09 nclt/2012-01-08 helipr/Roundabout03 mulran/Sejong03"

echo "[rspd] ---- stage 1: pd-val eval of per-epoch checkpoints ----"
for W in "$WDIR"/model_*_[1-5].pth; do
  ep="e${W##*_}"; ep="${ep%.pth}"
  for sq in $VAL_SEQS; do
    s=${sq%%/*}; q=${sq##*/}
    [ -f "results/pd2026_baselines/_val_ringsharp_pd_${ep}_${s}_${q}.json" ] && continue
    echo "[rspd] === val $ep $sq ($(date '+%F %T'))"
    python3 scripts/_pdb_ringsharp_pd_eval.py --weight "$W" --val --tag "$ep" "$sq" \
      2>&1 | tee "logs/rspd_val_${ep}_${s}_${q}.log"
  done
done

echo "[rspd] ---- stage 2: checkpoint selection (pre-declared rule) ----"
python3 scripts/_pdb_ringsharp_select.py 2>&1 | tee logs/rspd_select.log

SEL=$(python3 -c "
import json
print(json.load(open('results/pd2026_baselines/ringsharp_pd_ckpt_selection.json'))['selected_weight'])
")
echo "[rspd] selected weight: $SEL"

echo "[rspd] ---- stage 3: frozen 42-seq test roster ----"
SEQS=$(python3 -c "
import json
m = json.load(open('scripts/_pd_test_cache_manifest.json'))
print(' '.join(f\"{s['sensor']}/{s['sequence']}\" for s in m['sequences']))
")
for sq in $SEQS; do
  s=${sq%%/*}; q=${sq##*/}
  [ -f "results/pd2026_baselines/ringsharp_pd_${s}_${q}.json" ] && continue
  echo "[rspd] === test $sq ($(date '+%F %T'))"
  python3 scripts/_pdb_ringsharp_pd_eval.py --weight "$SEL" "$sq" \
    2>&1 | tee "logs/rspd_test_${s}_${q}.log"
done
echo "[rspd] RINGSHARP PD CHAIN COMPLETE ($(date '+%F %T'))"
