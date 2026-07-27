#!/usr/bin/env python3
"""SC++ (ring-key retrieval + cyclic column-shift rerank) on the corrected
8-byte NCLT clouds. Scan contexts are built directly from the fixed loader at
the corrected caches' keyframe scan_ids (the old contaminated values stand in
Table 1 as the reference)."""
import sys
from pathlib import Path

import numpy as np

REPO = Path('/rise/RISE1/workspace/impl/NSD-paper')
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from data.nclt_loader import NCLTLoader
from run_kitti_operating_point import (
    build_scan_context, _ring_key, _normalize, _find_queries, _topk_cosine,
    cyclic_column_cosine_distance,
)

CASES = [
    ('N12', '2012-01-08', 'data/preprocessed_nclt_fix8/nclt_operating_2012-01-08_layout60_stride1.npz'),
    ('N13', '2013-01-10', 'data/preprocessed_nclt_fix8/nclt_operating_2013-01-10_layout60_stride1.npz'),
]

for tag, date, rel in CASES:
    cache = np.load(REPO / rel)
    sids = cache['scan_ids']
    poses = cache['poses']
    loader = NCLTLoader('/rise/RISE1/workspace/data/nclt', date, lazy_load=True)
    scs, rks = [], []
    for i, s in enumerate(sids):
        if i % 1000 == 0:
            print(f'[{tag}] sc {i}/{len(sids)}', flush=True)
        sc = build_scan_context(loader[int(s)]['points'], n_rings=20,
                                n_sectors=60, max_range=80.0)
        scs.append(sc.astype(np.float32))
        rks.append(_ring_key(sc).astype(np.float32))
    scs = np.asarray(scs)
    rks = _normalize(np.asarray(rks))
    queries = _find_queries(poses, 5.0, 30)
    positions = poses[:, :3, 3]
    correct = 0
    for qi, (q, _) in enumerate(queries):
        cand = _topk_cosine(rks, q, 200, 30)
        dists = np.asarray([
            cyclic_column_cosine_distance(scs[q], scs[int(c)]) for c in cand
        ], dtype=np.float32)
        top = cand[int(np.argmin(dists))]
        correct += np.linalg.norm(positions[top] - positions[q]) < 5.0
        if qi and qi % 500 == 0:
            print(f'[{tag}] q {qi}/{len(queries)}', flush=True)
    print(f'{tag}: q={len(queries)} scpp_fixed_R@1={correct/len(queries):.4f}',
          flush=True)
print('SCPP_FIX8_DONE', flush=True)
