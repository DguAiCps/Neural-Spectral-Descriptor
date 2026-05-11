#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/rise/RISE1/workspace/data}"
NCLT_ROOT="${DATA_ROOT}/nclt"
LOG_DIR="${DATA_ROOT}/download_logs"
DATES="${NSD_NCLT_DATES:-2012-01-08 2013-01-10}"

mkdir -p "${NCLT_ROOT}" "${LOG_DIR}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

download_date() {
  local date="$1"
  local base="https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu"
  local date_dir="${NCLT_ROOT}/${date}"

  log "NCLT ${date}: download started"
  cd "${NCLT_ROOT}"
  wget -c --progress=dot:giga \
    "${base}/velodyne_data/${date}_vel.tar.gz" \
    "${base}/ground_truth/groundtruth_${date}.csv"

  mkdir -p "${date_dir}"
  cp -n "${NCLT_ROOT}/groundtruth_${date}.csv" "${date_dir}/groundtruth_${date}.csv" || true

  if [ ! -d "${date_dir}/velodyne_sync" ] || ! find "${date_dir}/velodyne_sync" -name '*.bin' -print -quit | grep -q .; then
    log "NCLT ${date}: extracting velodyne archive"
    tar -xzf "${NCLT_ROOT}/${date}_vel.tar.gz" -C "${date_dir}"
  else
    log "NCLT ${date}: velodyne archive already extracted"
  fi

  log "NCLT ${date}: ready"
}

{
  log "NCLT validation download started at ${NCLT_ROOT}"
  for date in ${DATES}; do
    download_date "${date}"
  done
  cd "${NCLT_ROOT}"
  wget -c --progress=dot:giga \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/laser_angles.csv
  log "NCLT validation download finished"
} 2>&1 | tee -a "${LOG_DIR}/nsd_nclt_val_data.log"
