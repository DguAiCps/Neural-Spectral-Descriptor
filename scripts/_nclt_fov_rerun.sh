#!/usr/bin/env bash
# NCLT elevation-window fix (2026-10-11): re-record the nine NCLT test sessions
# with the invariant-key caches of data/nclt_fov/ (scripts/_nclt_fov_fix.py)
# and re-aggregate. NCLT is in no validation set, so tuples, confidence
# statistics and tau* stay as selected. Pre-fix NCLT records and summaries
# move to <result dir>/_pre_nclt_fov/.
#   online : results/pdv2_c3_bevclip_s{1,2,3}/odom, results/pdv2_c3_bevclip_s1/gt
#   offline: results/pdv2_bevclip
# Heavy steps run one at a time. Run inside ring_baseline:
#   docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
#     bash scripts/_nclt_fov_rerun.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs/nclt_fov
NCLT="2012-01-08 2012-02-04 2012-03-17 2012-05-11 2012-08-04 2012-11-04 2012-11-16 2013-01-10 2013-02-23"

run() {
  local log="$1"; shift
  echo "[nclt-fov] === $* ($(date '+%F %T'))"
  "$@" > "logs/nclt_fov/${log}.log" 2>&1
}

for c in odom gt; do
  test "$(ls data/nclt_fov/$c/cache_*_nclt_test_*.npz | wc -l)" = 9
done

archive() {   # archive <dir> <glob>...
  local dir="$1"; shift
  mkdir -p "$dir/_pre_nclt_fov"
  for g in "$@"; do
    for f in $dir/$g; do
      [ -e "$f" ] && mv "$f" "$dir/_pre_nclt_fov/"
    done
  done
  return 0
}

online() {
  local s="$1" cond="$2"
  local dir="results/pdv2_c3_bevclip_s${s}/${cond}"
  local P="python3 scripts/_pdv2_bevclip.py --seed $s"
  if [ ! -d "$dir/_pre_nclt_fov" ]; then
    archive "$dir" "dump_*_nclt_*.npz" "align_dump_*_nclt_*.npz"
    cp "$dir/pr_summary.json" "$dir/pr_summary_align.json" "$dir/_pre_nclt_fov/"
  fi
  for q in $NCLT; do
    run "s${s}_${cond}_dump_nclt_${q}" $P test-dump --cond "$cond" --seq "nclt/$q"
  done
  run "s${s}_${cond}_align_channels" $P align-channels --cond "$cond"
  run "s${s}_${cond}_aggregate" $P aggregate --cond "$cond"
  run "s${s}_${cond}_aggregate_align" $P aggregate-align --cond "$cond"
  echo "[nclt-fov] SEED $s $cond COMPLETE ($(date '+%F %T'))"
}

online 1 odom
online 2 odom
online 3 odom
online 1 gt

# offline pilot (seed 1)
O=results/pdv2_bevclip
P="python3 scripts/_pdv2_bevclip_offline.py"
if [ ! -d "$O/_pre_nclt_fov" ]; then
  archive "$O" "test_apply_pdv2_*_nclt_*.json"
  cp "$O/summary.json" "$O/_pre_nclt_fov/"
fi
run off_link $P link
for q in $NCLT; do
  run "off_apply_nclt_${q}" $P apply --seq "nclt/$q"
done
run off_aggregate $P aggregate
echo "[nclt-fov] NCLT FOV RERUN COMPLETE ($(date '+%F %T'))"
