#!/usr/bin/env bash
# Baseline dump chain for the pre-registered causal PR eval (Track C of
# docs/CAUSAL_PR_PREREGISTRATION.md): ALL-FRAMES per-query ledgers, one npz
# per (method, sensor/seq) in results/pd2026_pr/ (schema pr-dump-v1).
# Roster = Strict-v2 37 sequences. Cost order: SC++ (CPU) -> BEVPlace++ ->
# SeqOT (trailing window) -> RING# -> RING/RING++ (O(N^2) correlations) last.
# Resumable: per-sequence dump npz skipped when present. One heavy sequence
# per invocation, mirroring scripts/_pdb_run_all.sh.
# Evaluator scripts default their contract-fixed artifacts (weights /
# checkpoints per prereg §1.5) — no artifact flags passed here.
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_causal_baseline_chain.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs results/pd2026_pr

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

run() {
  local log="$1"; shift
  echo "[causalC] === $* ($(date '+%F %T'))"
  "$@" 2>&1 | tee "logs/${log}.log" | tail -3
}
tag() { echo "$1" | tr / _; }
MISSING=0

# stage <name> <script> <dump-prefixes...>: loop the roster, skip sequences
# whose dump npz(s) all exist, one sequence per invocation.
stage() {
  local name="$1" script="$2"; shift 2
  echo "[causalC] ---- ${name} ----"
  if [ ! -f "$script" ]; then
    echo "[causalC] WARNING: $script not present yet — stage skipped"
    MISSING=1
    return 0
  fi
  local sq s q p done_all
  for sq in $SEQS; do
    s=${sq%%/*}; q=${sq##*/}
    done_all=1
    for p in "$@"; do
      [ -f "results/pd2026_pr/dump_${p}_${s}_${q}.npz" ] || done_all=0
    done
    [ "$done_all" = 1 ] && continue
    run "causal_pr_${name}_$(tag "$sq")" python3 "$script" --seqs "$sq"
  done
}

stage scpp      scripts/_causal_scpp_pr.py      scpp
stage bevplace  scripts/_causal_bevplace_pr.py  bevplace
stage seqot     scripts/_causal_seqot_pr.py     seqot
stage ringsharp scripts/_causal_ringsharp_pr.py ringsharp
stage ring      scripts/_causal_ring_pr.py      ring ringpp

if [ "$MISSING" = 1 ]; then
  echo "[causalC] INCOMPLETE: at least one evaluator script was missing"
  exit 1
fi
echo "[causalC] CAUSAL BASELINE DUMPS COMPLETE ($(date '+%F %T'))"
