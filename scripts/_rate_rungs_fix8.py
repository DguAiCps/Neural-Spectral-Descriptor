#!/usr/bin/env python3
"""Intermediate frequency-truncation rungs (16xF floats, F=90/45/11/5) for the
appendix rate sweep, under the exact Table-2 protocol of _tables23_fix8.py
(operating caches, corrected parser, deterministic keys)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path('/rise/RISE1/workspace/impl/NSD-paper')
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from _height_candidate_fusion import JOBS, l2
from run_kitti_operating_point import _find_queries, _topk_cosine

VAL8 = ['K00', 'K05', 'K08', 'N12', 'N13', 'DCC', 'KAI', 'RIV']
FREQS = {1440: 90, 720: 45, 176: 11, 80: 5}
DTH, SKIP = 5.0, 30
OUT = REPO / 'results/rate_rungs_fix8.json'


def r1(keys, poses):
    positions = poses[:, :3, 3]
    queries = _find_queries(poses, DTH, SKIP)
    c = 0
    for q, _ in queries:
        top = _topk_cosine(keys, q, 1 + 2 * SKIP, SKIP)[0]
        c += np.linalg.norm(positions[top] - positions[q]) < DTH
    return c / max(len(queries), 1), len(queries)


out = json.loads(OUT.read_text()) if OUT.exists() else {}
for tag in VAL8:
    if tag in out:
        continue
    sensor, sequence, cache_path = JOBS[tag]
    cache = np.load(REPO / cache_path)
    mags = cache['fft_magnitudes'].astype(np.float64)
    poses = cache['poses']
    rec = {}
    for floats, F in FREQS.items():
        v, nq = r1(l2(mags[:, :, :F].reshape(len(mags), -1)), poses)
        rec[str(floats)] = v
        rec['n_queries'] = nq
    out[tag] = rec
    OUT.write_text(json.dumps(out, indent=2))
    print(f'[{tag}] ' + ' '.join(f'{k}={rec[k]:.4f}' for k in ('1440', '720', '176', '80')), flush=True)

nq = np.array([out[t]['n_queries'] for t in VAL8], float)
for fl in ('1440', '720', '176', '80'):
    r = np.array([out[t][fl] for t in VAL8], float)
    print(f'AGG {fl}: Rq={float((r * nq).sum() / nq.sum()):.4f}', flush=True)
print('RATE_RUNGS_DONE', flush=True)
