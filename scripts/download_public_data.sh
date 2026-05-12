#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/workspace/data}"
LOG_DIR="${DATA_ROOT}/download_logs"

mkdir -p "${DATA_ROOT}/kitti" "${DATA_ROOT}/nclt" "${LOG_DIR}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

download_kitti() {
  log "KITTI odometry download started"
  cd "${DATA_ROOT}/kitti"
  wget -c --progress=dot:giga \
    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip \
    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip \
    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
  log "KITTI odometry download finished"
}

download_nclt() {
  log "NCLT download started"
  cd "${DATA_ROOT}/nclt"
  wget -c --progress=dot:giga \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/2012-01-08_vel.tar.gz \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_2012-01-08.csv \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/2012-05-11_vel.tar.gz \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_2012-05-11.csv \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/2012-08-04_vel.tar.gz \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_2012-08-04.csv \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/2012-11-04_vel.tar.gz \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_2012-11-04.csv \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/2012-11-16_vel.tar.gz \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_2012-11-16.csv \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/2013-01-10_vel.tar.gz \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_2013-01-10.csv \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/2013-02-23_vel.tar.gz \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_2013-02-23.csv \
    https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/laser_angles.csv
  log "NCLT download finished"
}

{
  log "NSD public data download started at ${DATA_ROOT}"
  download_kitti
  download_nclt
  log "NSD public data download finished"
} 2>&1 | tee -a "${LOG_DIR}/nsd_public_data.log"
