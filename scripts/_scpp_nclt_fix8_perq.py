#!/usr/bin/env python3
"""Per-query dump of the Table-1 SC++ NCLT result on the corrected 8-byte NCLT
clouds. Identical pipeline to scripts/_scpp_nclt_fix8.py (which reproduces the
Table-1 numbers N12=0.9149, N13=0.9558), but records [query_idx, success_at_k1]
so the two NCLT sequences can replace the pre-z-fix entries in the stored SC++
per-query dump and yield a Table-1-consistent paired significance test.

Query set is _find_queries(poses,5.0,30) on the SAME corrected caches the NSD
headline run uses, so query_idx aligns 1:1 with results/per_query_headline/nsd*.
Output: results/per_query_headline/scpp_zfix/<DATASET>.json
"""
import os, sys, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from data.nclt_loader import NCLTLoader
from run_kitti_operating_point import (
    build_scan_context, _ring_key, _normalize, _find_queries, _topk_cosine,
    cyclic_column_cosine_distance,
)

CASES = [
    ('N12', '2012-01-08', 'NCLT_2012-01-08',
     'data/preprocessed_nclt_fix8/nclt_operating_2012-01-08_layout60_stride1.npz'),
    ('N13', '2013-01-10', 'NCLT_2013-01-10',
     'data/preprocessed_nclt_fix8/nclt_operating_2013-01-10_layout60_stride1.npz'),
]
OUT = REPO / "results/per_query_headline/scpp_zfix"
OUT.mkdir(parents=True, exist_ok=True)

for tag, date, ds, rel in CASES:
    cache = np.load(REPO / rel)
    sids = cache['scan_ids']
    poses = cache['poses']
    loader = NCLTLoader(str(Path(os.environ.get("NSD_DATA_ROOT", REPO / "data")) / "nclt"),
                        date, lazy_load=True)
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
    records, correct = [], 0
    for qi, (q, _) in enumerate(queries):
        cand = _topk_cosine(rks, q, 200, 30)
        dists = np.asarray([
            cyclic_column_cosine_distance(scs[q], scs[int(c)]) for c in cand
        ], dtype=np.float32)
        top = cand[int(np.argmin(dists))]
        hit = int(np.linalg.norm(positions[top] - positions[q]) < 5.0)
        correct += hit
        records.append({"query_idx": int(q), "success_at_k1": hit})
        if qi and qi % 500 == 0:
            print(f'[{tag}] q {qi}/{len(queries)}', flush=True)
    r1 = correct / len(queries)
    (OUT / f"{ds}.json").write_text(json.dumps(
        {"method": "sc++_zfix", "dataset": ds, "distance_threshold_m": 5.0,
         "skip_frames": 30, "n_queries": len(queries), "recall_at_1": r1,
         "records": records}, indent=1))
    print(f'{tag}: q={len(queries)} scpp_fixed_R@1={r1:.4f}  (Table-1: '
          f'{"0.9149" if tag=="N12" else "0.9558"})', flush=True)
print('SCPP_FIX8_PERQ_DONE', flush=True)
