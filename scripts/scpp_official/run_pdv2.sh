#!/usr/bin/env bash
# Official Scan Context++ (MATLAB, on the HOST) on split-v2 test sequences ->
# pr-dump-v1 ledgers in $OUT/<variant>/$COND/dump_scpp_<sensor>_<seq>.npz.
#
# Per sequence: container exports keyframe clouds in chunks of $CHUNK ->
# host MATLAB computes official descriptors per chunk (chunk deleted right
# after) -> host MATLAB online k=1 matching per variant -> host Python ledger.
# Resumable: finished descriptor chunks, match CSVs and ledgers are skipped.
#
#   bash scripts/scpp_official/run_pdv2.sh kitti/07          # one sequence
#   bash scripts/scpp_official/run_pdv2.sh                   # all 46 test
# Env (defaults): COND=odom VARIANTS="apc pc cc acc" SEARCH_RATIO=0
#   LIDAR_HEIGHT=1.9 CHUNK=2000 KEEP_DESC=0 OUT=results/pdv2_scpp_official
# Changing LIDAR_HEIGHT or SEARCH_RATIO needs a different OUT (nothing in the
# file names encodes them; the ledger meta does).
# Aggregate afterwards:
#   python3 scripts/scpp_official/write_ledger.py --aggregate --variant apc
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
DOCKER="docker exec -w /workspace/Neural-Spectral-Codec ring_baseline"
COND=${COND:-odom}
VARIANTS=${VARIANTS:-"apc pc cc acc"}
SEARCH_RATIO=${SEARCH_RATIO:-0}
LIDAR_HEIGHT=${LIDAR_HEIGHT:-1.9}
CHUNK=${CHUNK:-2000}
KEEP_DESC=${KEEP_DESC:-0}
OUT=${OUT:-results/pdv2_scpp_official}
MIN_GAP=30
case "$OUT" in /*) echo "OUT must be repo-relative (the container sees the repo at another path)"; exit 2;; esac
export PYTHONDONTWRITEBYTECODE=1
M="addpath('$REPO/scripts/scpp_official');"

SEQS="$*"
[ -n "$SEQS" ] || SEQS=$($DOCKER python3 scripts/_pdv2_c3.py roster --role test)

for S in $SEQS; do
  TAG=${S/\//_}
  WORK=$OUT/_work/$COND/$TAG
  todo=""
  for V in $VARIANTS; do
    [ -f "$OUT/$V/$COND/dump_scpp_${TAG}.npz" ] || todo="$todo $V"
  done
  if [ -z "$todo" ]; then echo "[scpp-official] $S: all ledgers exist, skip"; continue; fi
  echo "[scpp-official] === $S ($(date '+%F %T')) variants:$todo"
  mkdir -p "$WORK"   # host-owned, so root-owned chunk files can be removed here
  [ -f "$WORK/seq_info.json" ] || \
    $DOCKER python3 scripts/scpp_official/export_keyframes.py --seq "$S" --cond "$COND" --work "$WORK" --info
  N=$(python3 -c "import json;print(json.load(open('$WORK/seq_info.json'))['n_frames'])")
  need_match=0
  for V in $todo; do [ -f "$OUT/$V/$COND/raw/match_${TAG}.csv" ] || need_match=1; done

  for ((s = 0; need_match && s < N; s += CHUNK)); do
    D=$(printf "%s/desc_%06d.mat" "$WORK" "$s")
    [ -f "$D" ] && continue
    C=$(printf "%s/chunk_%06d" "$WORK" "$s")
    [ -f "$C.cnt" ] || \
      $DOCKER python3 scripts/scpp_official/export_keyframes.py --seq "$S" --cond "$COND" --work "$WORK" --chunk "$s" $((s + CHUNK))
    matlab -batch "$M scpp_official_describe('$REPO/$WORK', $s, $LIDAR_HEIGHT)"
    rm -f "$C.xyz" "$C.cnt"
  done

  CALLS=""
  for V in $todo; do
    CSV=$OUT/$V/$COND/raw/match_${TAG}.csv
    mkdir -p "$(dirname "$CSV")"
    [ -f "$CSV" ] || CALLS="$CALLS scpp_official_match('$REPO/$WORK', $N, '$V', $SEARCH_RATIO, $MIN_GAP, '$REPO/$CSV');"
  done
  [ -z "$CALLS" ] || matlab -batch "$M $CALLS"
  for V in $todo; do
    python3 scripts/scpp_official/write_ledger.py --seq "$S" --variant "$V" --cond "$COND" \
      --work "$WORK" --csv "$OUT/$V/$COND/raw/match_${TAG}.csv" --out-root "$OUT"
  done
  [ "$KEEP_DESC" = 1 ] || rm -f "$WORK"/desc_*.mat
done
echo "SCPP_OFFICIAL_DONE"
