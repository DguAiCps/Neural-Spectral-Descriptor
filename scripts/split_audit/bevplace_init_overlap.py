#!/usr/bin/env python3
"""Does the BEVPlace++ author checkpoint's training data overlap the strict test places?

The vendored official trainer trains on the first 3,000 frames of KITTI 00
(baselines/_bevplace_official/kitti_dataset.py:200-202) and keeps the best epoch
by mean NCLT recall over sessions that include 2012-02-04 and 2012-03-17
(baselines/_bevplace_official/main.py:332, 345-350). KITTI 00 and the NCLT pair are
excluded from the strict-37 roster; this script checks whether KITTI 00[:3000]
shares places with the remaining KITTI-family strict test sequences.

Input: results/split_audit/_traj_cache.pkl (UTM 32N trajectories from geo_audit.py).
Output: results/split_audit/bevplace_init_overlap.json.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results' / 'split_audit'
TESTS = ['kitti/05', 'kitti/08', 'kitti360/0000', 'kitti360/0006']


def main():
    tr = pickle.load(open(OUT / '_traj_cache.pkl', 'rb'))[0]
    train_xy = tr['kitti/00']['xy'][:3000]
    tree = cKDTree(train_xy)
    res = {'train_frames': 'kitti/00 frames 0-2999',
           'train_path_km': float(np.linalg.norm(np.diff(train_xy, axis=0), axis=1).sum() / 1000),
           'pairs': {}}
    for t in TESTS:
        dist, _ = tree.query(tr[t]['xy'])
        res['pairs'][t] = {'min_m': float(dist.min()),
                           'frac_within_25m': float((dist < 25).mean()),
                           'frac_within_50m': float((dist < 50).mean())}
        print(t, res['pairs'][t])
    (OUT / 'bevplace_init_overlap.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
