#!/usr/bin/env bash
# Test-phase NSD dump chain for the pre-registered causal PR campaign
# (docs/CAUSAL_PR_PREREGISTRATION.md §3 track B). Requires Track A frozen
# (results/pd2026_pr/causal_tuple_selection.json + confidence_selection_*).
# Dumps the ALL-FRAMES per-query ledger for the Strict-v2 37-sequence roster,
# both tracks (pd_causal = causal-retrained, pd = Frozen-GNN causal
# adaptation), seeds 1-3, both ops, at the FROZEN causal tuples (passed
# explicitly; never the bidirectional placeholder default).
# Resumable per (variant,seed,seq). One sequence per worker subprocess.
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_causal_test_chain.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs results/pd2026_pr

SEL=results/pd2026_pr/causal_tuple_selection.json
[ -f "$SEL" ] || { echo "[testB] missing $SEL — run Track A first"; exit 1; }

SEQS=$(python3 - <<'PYEOF'
import json
m = json.load(open('scripts/_pd_test_cache_manifest.json'))
EXCL = {("kitti","00"),("kitti360","0004"),("helipr","Bridge04"),
        ("nclt","2012-02-04"),("nclt","2012-03-17")}
seqs = [(e["sensor"], e["sequence"]) for e in m["sequences"]]
strict = [f"{s}/{q}" for s, q in seqs if (s, q) not in EXCL]
assert len(strict) == 37, len(strict)
print(" ".join(strict))
PYEOF
)

tuple_for() {  # variant op -> "w0 w1 w2 w3" (space-separated for argparse nargs=4)
  python3 - "$1" "$2" <<'PYEOF'
import json, sys
sel = json.load(open('results/pd2026_pr/causal_tuple_selection.json'))
print(" ".join(str(x) for x in sel["selections"][f"{sys.argv[1]}/{sys.argv[2]}"]["tuple"]))
PYEOF
}

for v in pd_causal pd; do
  TP=$(tuple_for "$v" pbev128_main)
  TH=$(tuple_for "$v" hspec12)
  echo "[testB] variant=$v tuples pbev=$TP hspec=$TH"
  for s in 1 2 3; do
    for sq in $SEQS; do
      sn=${sq%%/*}; qn=${sq##*/}
      if [ -f "results/pd2026_pr/dump_${v}_pbev128_main_s${s}_${sn}_${qn}.npz" ] \
         && [ -f "results/pd2026_pr/dump_${v}_hspec12_s${s}_${sn}_${qn}.npz" ]; then
        continue
      fi
      echo "[testB] === $v s$s $sq ($(date '+%F %T'))"
      python3 scripts/_causal_pr_dump.py --variant "$v" --seed "$s" \
        --seqs "$sq" --tuple-pbev $TP --tuple-hspec $TH \
        2>&1 | tee "logs/prdump_${v}_s${s}_${sn}_${qn}.log" | tail -2
    done
  done
done
echo "[testB] NSD PR DUMP CHAIN COMPLETE ($(date '+%F %T'))"
