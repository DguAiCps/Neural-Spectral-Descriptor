#!/usr/bin/env python3
"""Headline matching latency v3: matching latency from PRECOMPUTED query keys
largest real map (Riverside03): true 416D retrieval key (frozen seed-1 model),
two exact GPU similarity scans (top-100 each) with temporal exclusion, per-query min--max
fusion with each variant's full frozen score set:
  NSD   (416/128): 0.5*s_f + 1*s_m + 0.5*s_col
  NSD-H (416/240): 1*s_f + 1*s_m + 1*s_align + 2*s_col
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path('/rise/RISE1/workspace/impl/NSD-paper')
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from _height_candidate_fusion import (
    JOBS, bin288, minmax, keyframes_from, column_align_bank,
    column_align_scores, align_scores,
)
from _hspec_branch_fusion import hspec_inputs
from _pbev_branch_fusion import variant_inputs
from evaluate_kitti_checkpoint import _apply_encoder_preset, _build_eval_graph, _make_model
from run_kitti_operating_point import _find_queries

K, SKIP = 100, 30
TAG = 'RIV'
DEV = 'cuda'

sensor, sequence, cache_path = JOBS[TAG]
cache = np.load(REPO / cache_path)
d288 = bin288(cache['fft_magnitudes'].astype(np.float64)).astype(np.float32)
poses = cache['poses']

cfg = _apply_encoder_preset(
    yaml.safe_load(open(REPO / 'configs/training_multi_dataset.yaml')),
    'no_interdiff')
cfg['gnn']['use_residual_gate'] = True
cfg['gnn']['gate_initial_alpha'] = 0.0625
cfg = copy.deepcopy(cfg)
model = _make_model(cfg, REPO / 'checkpoints/800d_4sensor_20260511_161726/best_model.pth', DEV)
graph = _build_eval_graph(
    keyframes=keyframes_from(cache, d288), poses=poses, descriptors=d288,
    cache=cache, config=cfg, device=DEV, temporal_edge_mode='bidirectional',
    temporal_direction_mode='none', similarity_min_k=0, phase_features=None,
    sensor_key=sensor)
with torch.no_grad():
    f416 = model(graph.to(DEV)).detach().cpu().numpy().astype(np.float32)
f416 /= np.clip(np.linalg.norm(f416, axis=1, keepdims=True), 1e-8, None)

hc = np.load(REPO / 'results/height_sota/cache' / f'{TAG}_height120.npz')
raw_height = hc['raw_height'].astype(np.float32)
pc = np.load(REPO / 'results/height_sota/cache' / f'{TAG}_pbev.npz')
sk_pbev = pc['sk_re'].astype(np.float32) + 1j * pc['sk_im'].astype(np.float32)

hk128, _, _, col128 = variant_inputs(sk_pbev, 8)
hk240, hfft240, hnorm240, col240 = hspec_inputs(raw_height, 12)

n = len(f416)
F = torch.from_numpy(f416).to(DEV)
_allq = [q for q, _ in _find_queries(poses, 5.0, SKIP)]
_rngq = np.random.default_rng(7)
queries = list(_rngq.choice(_allq, size=min(400, len(_allq)), replace=False))
report = {}
for name, hk, use_align, wts in (
        ('NSD (416/128, 800f)', hk128, False, (0.5, 1.0, 0.0, 0.5)),
        ('NSD-H (416/240, 1024f)', hk240, True, (1.0, 1.0, 1.0, 2.0))):
    Hgpu = torch.from_numpy(np.ascontiguousarray(hk)).to(DEV)
    colbank = col128 if not use_align else col240
    for _warm in queries[:20]:      # full-path warm-up (excluded from timing)
        _sfw = F @ F[_warm]
        _shw = Hgpu @ Hgpu[_warm]
        _cw = torch.unique(torch.cat([torch.topk(_sfw, K).indices,
                                      torch.topk(_shw, K).indices])).cpu().numpy()
        _ = minmax(column_align_scores(colbank, int(_warm), _cw))
        if use_align:
            _ = minmax(align_scores(hfft240, hnorm240, int(_warm), _cw))
    torch.cuda.synchronize()
    _lat = []
    for q in queries:
        torch.cuda.synchronize()
        _tq = time.perf_counter()
        sf_all = F @ F[q]
        sh_all = Hgpu @ Hgpu[q]
        lo, hi = max(0, q - SKIP), min(n, q + SKIP + 1)
        sf_all[lo:hi] = -9; sh_all[lo:hi] = -9
        cf = torch.topk(sf_all, K).indices
        ch = torch.topk(sh_all, K).indices
        cand = torch.unique(torch.cat([cf, ch])).cpu().numpy()
        sf = minmax(sf_all[cand].cpu().numpy())
        sh = minmax(sh_all[cand].cpu().numpy())
        scol = minmax(column_align_scores(colbank, int(q), cand))
        s = wts[0] * sf + wts[1] * sh + wts[3] * scol
        if use_align:
            s = s + wts[2] * minmax(align_scores(hfft240, hnorm240, int(q), cand))
        int(cand[int(np.argmax(s))])
        torch.cuda.synchronize()
        _lat.append((time.perf_counter() - _tq) * 1000)
    _lat = np.asarray(_lat)
    report[name] = {'n_stored_keyframes': int(n), 'n_queries': len(queries),
                    'matching_ms_mean': round(float(_lat.mean()), 3),
                    'matching_ms_median': round(float(np.median(_lat)), 3),
                    'matching_ms_p95': round(float(np.percentile(_lat, 95)), 3),
                    'note': 'matching latency from precomputed query keys; '
                            'two exact GPU similarity scans + union fusion'}
    print(f'{name}: stored={n}, matching mean {_lat.mean():.2f} ms, '
          f'median {np.median(_lat):.2f}, p95 {np.percentile(_lat, 95):.2f}', flush=True)

(REPO / 'results/headline_latency_fix8.json').write_text(json.dumps(report, indent=2))
print('LATENCY_DONE', flush=True)
