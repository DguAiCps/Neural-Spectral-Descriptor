#!/usr/bin/env python3
"""Context-permutation retrieval control (Table 3 extension):
compare d (288D), [d; pi(c)] (permuted context, 10 draws), and [d; c]
under the plain 416D protocol on corrected caches. Distinguishes
dimensionality increase from genuine trajectory information."""
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

from _height_candidate_fusion import JOBS, bin288, keyframes_from
from evaluate_kitti_checkpoint import _apply_encoder_preset, _build_eval_graph, _make_model
from run_kitti_operating_point import _find_queries, _topk_cosine

VAL9 = ['K00', 'K05', 'K08', 'N12', 'N13', 'TOWN', 'DCC', 'KAI', 'RIV']
CKPTS = {
    1: REPO / 'checkpoints/800d_4sensor_20260511_161726/best_model.pth',
    2: REPO / 'checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth',
    3: REPO / 'checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth',
}
DTH, SKIP = 5.0, 30
N_PERM = 10


def r1(keys, poses):
    positions = poses[:, :3, 3]
    queries = _find_queries(poses, DTH, SKIP)
    c = 0
    for q, _ in queries:
        top = _topk_cosine(keys, q, 1 + 2 * SKIP, SKIP)[0]
        c += np.linalg.norm(positions[top] - positions[q]) < DTH
    return c / max(len(queries), 1), len(queries)


def norm(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


out = {}
device = 'cuda'
for seed in (1, 2, 3):
    cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / 'configs/training_multi_dataset.yaml')),
        'no_interdiff')
    cfg['gnn']['use_residual_gate'] = True
    cfg['gnn']['gate_initial_alpha'] = 0.0625
    cfg = copy.deepcopy(cfg)
    model = _make_model(cfg, CKPTS[seed], device)
    for tag in VAL9:
        sensor, sequence, cache_path = JOBS[tag]
        cache = np.load(REPO / cache_path)
        d288 = bin288(cache['fft_magnitudes'].astype(np.float64))
        poses = cache['poses']
        graph = _build_eval_graph(
            keyframes=keyframes_from(cache, d288), poses=poses,
            descriptors=d288, cache=cache, config=cfg, device=device,
            temporal_edge_mode='bidirectional', temporal_direction_mode='none',
            similarity_min_k=0, phase_features=None, sensor_key=sensor)
        with torch.no_grad():
            emb = model(graph.to(device)).detach().cpu().numpy().astype(np.float64)
        rec = out.setdefault(tag, {'d288': None, 'f416': {}, 'perm': {}})
        if rec['d288'] is None:
            v, nq = r1(norm(d288.astype(np.float64)), poses)
            rec['d288'] = v
            rec['n_queries'] = nq
        v, _ = r1(norm(emb), poses)
        rec['f416'][seed] = v
        rng = np.random.default_rng(1000 + seed)
        perms = []
        for _ in range(N_PERM):
            e = emb.copy()
            idx = rng.permutation(len(e))
            e[:, 288:] = e[idx, 288:]
            pv, _ = r1(norm(e), poses)
            perms.append(pv)
        rec['perm'][seed] = perms
        print(f'[s{seed} {tag}] d={rec["d288"]:.4f} f={v:.4f} '
              f'perm={np.mean(perms):.4f}±{np.std(perms):.4f}', flush=True)

(REPO / 'results/ctx_perm_fix8.json').write_text(
    json.dumps(out, indent=2, default=float))
print('CTX_PERM_DONE', flush=True)
