"""LiDAR odometry: KISS-ICP over every scan of a sequence.

The method never sees reference poses. Keyframe selection and the relative geometry of the
temporal graph use these odometry poses; reference poses are used only to score retrieval.

Settings of the evaluated configuration (KISS-ICP 1.3.0 defaults otherwise): no deskewing
(the readers provide no per-point times), points kept between 1 m and 100 m, two registration
threads.
"""
from __future__ import annotations

import time

import numpy as np

MIN_RANGE_M = 1.0
MAX_RANGE_M = 100.0
REGISTRATION_THREADS = 2


def run_kiss_icp(sequence, log_every: int = 2000) -> np.ndarray:
    """Register every scan of `sequence` in order; returns the odometry poses [N, 4, 4]."""
    from kiss_icp.config import load_config
    from kiss_icp.kiss_icp import KissICP

    cfg = load_config(None)
    cfg.data.deskew = False
    cfg.data.min_range = MIN_RANGE_M
    cfg.data.max_range = MAX_RANGE_M
    cfg.registration.max_num_threads = REGISTRATION_THREADS
    odometry = KissICP(cfg)

    n = len(sequence)
    poses = np.empty((n, 4, 4))
    no_point_times = np.empty(0)
    start = time.time()
    for i in range(n):
        odometry.register_frame(sequence.points(i).astype(np.float64), no_point_times)
        poses[i] = odometry.last_pose
        if log_every and i % log_every == 0:
            print(f"  odometry scan {i}/{n} ({(i + 1) / (time.time() - start):.1f} Hz)", flush=True)
    return poses
