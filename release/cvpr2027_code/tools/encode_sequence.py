#!/usr/bin/env python3
"""Encode one raw sequence into the per-sequence keyframe file used by the rest of NSD.

    python tools/encode_sequence.py --dataset kitti --root <kitti dir> --sequence 07 \
        --out encoded/kitti_07.npz
    python tools/encode_sequence.py --dataset helipr --root <helipr dir> --sequence Bridge02 \
        --reference-poses --out train/helipr_Bridge02.npz

Steps: read scans and reference poses (nsd.data) -> KISS-ICP odometry over every scan
(nsd.odometry) -> keyframe selection on the odometry poses (nsd.keyframes) -> spectral
encoding of each keyframe (nsd.encoding).

Output fields (keyframes only, in order): scan_ids int64 [N], timestamps float64 [N],
poses float64 [N, 4, 4] (odometry), poses_gt float64 [N, 4, 4] (reference, evaluation only),
d float32 [N, 288], z complex64 [N, 16, 8].

Training sequences: `--reference-poses` skips KISS-ICP and uses the dataset reference poses as
`poses` (so poses = poses_gt), which is how the training files of the reported model were made.
For KITTI it also writes `lidar_to_pose` [4, 4] (calibration Tr; KITTI reference poses are camera
poses), which training needs to obtain LiDAR yaw.

Odometry is the slow step. `--save-poses` stores the per-scan odometry poses [S, 4, 4] as .npy
and `--poses` reuses such a file instead of running KISS-ICP again.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsd.data import DATASETS, open_sequence  # noqa: E402
from nsd.encoding import encode_scan  # noqa: E402
from nsd.keyframes import select_keyframes  # noqa: E402
from nsd.odometry import run_kiss_icp  # noqa: E402


def encode_sequence(dataset: str, root: str, sequence: str, poses: np.ndarray | None = None,
                    log_every: int = 2000, reference_poses: bool = False):
    """Returns (per-sequence dict, per-scan poses used by the method [S, 4, 4], timings in seconds).
    reference_poses=True uses the dataset reference poses instead of odometry (training files)."""
    timings = {}
    t = time.time()
    seq = open_sequence(dataset, root, sequence)
    timings["read_index"] = time.time() - t

    t = time.time()
    if reference_poses:
        poses = seq.poses_gt
    elif poses is None:
        poses = run_kiss_icp(seq, log_every=log_every)
        timings["odometry"] = time.time() - t
    elif len(poses) != len(seq):
        raise ValueError(f"--poses has {len(poses)} poses but the sequence has {len(seq)} scans")

    t = time.time()
    scan_ids = select_keyframes(poses, seq.timestamps)
    timings["keyframes"] = time.time() - t

    t = time.time()
    d = np.empty((len(scan_ids), 288), dtype=np.float32)
    z = np.empty((len(scan_ids), 16, 8), dtype=np.complex64)
    for j, i in enumerate(scan_ids):
        d[j], z[j] = encode_scan(seq.points(int(i)), dataset)
        if log_every and j % log_every == 0:
            print(f"  encoding keyframe {j}/{len(scan_ids)}", flush=True)
    timings["encoding"] = time.time() - t

    out = dict(scan_ids=scan_ids, timestamps=seq.timestamps[scan_ids].astype(np.float64),
               poses=poses[scan_ids].astype(np.float64),
               poses_gt=seq.poses_gt[scan_ids].astype(np.float64), d=d, z=z)
    if reference_poses and seq.lidar_to_pose is not None:
        out["lidar_to_pose"] = seq.lidar_to_pose.astype(np.float64)
    timings["n_scans"], timings["n_keyframes"] = len(seq), len(scan_ids)
    return out, poses, timings


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--root", required=True, help="dataset directory (layouts in nsd/data.py)")
    ap.add_argument("--sequence", required=True, help="e.g. 07, 0000, 2012-02-04, DCC01")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--poses", help="per-scan odometry poses (.npy [S, 4, 4]) to reuse")
    ap.add_argument("--save-poses", help="write the per-scan odometry poses to this .npy")
    ap.add_argument("--reference-poses", action="store_true",
                    help="use the dataset reference poses instead of odometry (training sequences)")
    a = ap.parse_args()
    if a.reference_poses and (a.poses or a.save_poses):
        ap.error("--reference-poses cannot be combined with --poses / --save-poses")

    poses = np.load(a.poses) if a.poses else None
    out, scan_poses, timings = encode_sequence(a.dataset, a.root, a.sequence, poses,
                                               reference_poses=a.reference_poses)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    if a.save_poses:
        np.save(a.save_poses, scan_poses)
    print(f"{a.dataset}/{a.sequence}: {timings['n_keyframes']} keyframes of "
          f"{timings['n_scans']} scans -> {a.out}  "
          + "  ".join(f"{k} {v:.1f}s" for k, v in timings.items() if k not in ("n_scans", "n_keyframes")),
          flush=True)


if __name__ == "__main__":
    main()
