#!/usr/bin/env python3
"""Recompute manuscript Tables 2 and 3 on corrected-parser caches.

Table 3 (context recovery): per-sequence cosine R@1 of the 288D invariant key
vs the plain 416D retrieval key f=[d; gated c] -- BEFORE re-metrization,
edge-classifier graph refinement, and alignment (3 seeds for the 416D column;
the 288D key is deterministic).

Table 2 (rate/aggregation ladder): R@1 of full magnitude spectrum (2,896D),
frequency-truncated (16x22=352D), and octave mean+std (288D) keys.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path('/rise/RISE1/workspace/impl/NSD-paper')
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from _height_candidate_fusion import JOBS, bin288, l2, keyframes_from
from evaluate_kitti_checkpoint import _apply_encoder_preset, _build_eval_graph, _make_model
from run_kitti_operating_point import _find_queries, _topk_cosine

VAL9 = ['K00', 'K05', 'K08', 'N12', 'N13', 'TOWN', 'DCC', 'KAI', 'RIV']
CKPTS = {
    1: REPO / 'checkpoints/800d_4sensor_20260511_161726/best_model.pth',
    2: REPO / 'checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth',
    3: REPO / 'checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth',
}
DTH, SKIP = 5.0, 30


def r1(keys, poses):
    positions = poses[:, :3, 3]
    queries = _find_queries(poses, DTH, SKIP)
    c = 0
    for q, _ in queries:
        top = _topk_cosine(keys, q, 1 + 2 * SKIP, SKIP)[0]
        c += np.linalg.norm(positions[top] - positions[q]) < DTH
    return c / max(len(queries), 1), len(queries)


def plain_model(device, seed):
    cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / 'configs/training_multi_dataset.yaml')),
        'no_interdiff')
    cfg['gnn']['use_residual_gate'] = True
    cfg['gnn']['gate_initial_alpha'] = 0.0625
    cfg = copy.deepcopy(cfg)          # NO key_remetrize, NO edge classifier
    model = _make_model(cfg, CKPTS[seed], device)
    return cfg, model


def trunc352(mags):
    return l2(mags[:, :, :22].reshape(len(mags), -1))


def full_key(mags):
    return l2(mags.reshape(len(mags), -1))


out = {'table2': {}, 'table3': {}}
device = 'cuda'
models = {s: plain_model(device, s) for s in (1, 2, 3)}
for tag in VAL9:
    sensor, sequence, cache_path = JOBS[tag]
    cache = np.load(REPO / cache_path)
    mags = cache['fft_magnitudes'].astype(np.float64)
    poses = cache['poses']
    d288 = bin288(mags)
    r288, nq = r1(l2(d288.astype(np.float64)), poses)
    rfull, _ = r1(full_key(mags), poses)
    rtr, _ = r1(trunc352(mags), poses)
    out['table2'][tag] = {'full2896': rfull, 'trunc352': rtr,
                          'octave288': r288, 'n_queries': nq}
    f416 = []
    for s in (1, 2, 3):
        cfg, model = models[s]
        graph = _build_eval_graph(
            keyframes=keyframes_from(cache, d288), poses=poses,
            descriptors=d288, cache=cache, config=cfg, device=device,
            temporal_edge_mode='bidirectional', temporal_direction_mode='none',
            similarity_min_k=0, phase_features=None, sensor_key=sensor)
        with torch.no_grad():
            emb = model(graph.to(device)).detach().cpu().numpy()
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
        r, _ = r1(emb.astype(np.float64), poses)
        f416.append(r)
    out['table3'][tag] = {'inv288': r288, 'key416_seeds': f416,
                          'key416_mean': float(np.mean(f416)), 'n_queries': nq}
    print(f'[{tag}] q={nq} 288={r288:.4f} full={rfull:.4f} trunc={rtr:.4f} '
          f'416={np.mean(f416):.4f}±{np.std(f416):.4f}', flush=True)

(REPO / 'results/tables23_fix8.json').write_text(json.dumps(out, indent=2))
print('TABLES23_DONE', flush=True)
