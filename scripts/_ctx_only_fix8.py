#!/usr/bin/env python3
"""Context-only column for the appendix context-isolation table: cosine R@1
of the 128D context block alone, same plain-416D protocol and corrected
caches as tables23/ctx_perm (3 seeds, 9 sequences). Saves per-sequence
checkpoints incrementally."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

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
OUT = REPO / 'results/ctx_only_fix8.json'
DTH, SKIP = 5.0, 30


def r1(keys, poses):
    positions = poses[:, :3, 3]
    queries = _find_queries(poses, DTH, SKIP)
    c = 0
    for q, _ in queries:
        top = _topk_cosine(keys, q, 1 + 2 * SKIP, SKIP)[0]
        c += np.linalg.norm(positions[top] - positions[q]) < DTH
    return c / max(len(queries), 1), len(queries)


out = json.loads(OUT.read_text()) if OUT.exists() else {}
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
        if str(seed) in out.get(tag, {}):
            continue
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
        ctx = emb[:, 288:]
        ctx = ctx / np.clip(np.linalg.norm(ctx, axis=1, keepdims=True), 1e-12, None)
        v, nq = r1(ctx, poses)
        out.setdefault(tag, {})[str(seed)] = {'ctx_only': v, 'n_queries': nq}
        OUT.write_text(json.dumps(out, indent=2))     # per-sequence checkpoint
        print(f'[s{seed} {tag}] ctx_only={v:.4f} q={nq}', flush=True)
print('CTX_ONLY_DONE', flush=True)
