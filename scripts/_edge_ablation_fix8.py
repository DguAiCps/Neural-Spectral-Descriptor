#!/usr/bin/env python3
"""Headline-protocol edge-selection ablation (the one remaining experiment):
Pass-1 only vs two-pass cosine top-10 vs two-pass learned top-10, everything
else fixed to the headline NSD configuration (dual top-100 union with the
128D BEV magnitudes, per-query min-max fusion, frozen tuple (0.5,1,0,0.5),
corrected caches, 3 seeds). Per-sequence incremental checkpointing."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path('/rise/RISE1/workspace/impl/NSD-paper')
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

from _height_candidate_fusion import (
    JOBS, CACHE_OUT, bin288, l2, minmax, keyframes_from, make_model,
    column_align_bank, column_align_scores,
)
from _pbev_branch_fusion import variant_inputs
from evaluate_kitti_checkpoint import _build_eval_graph
from run_kitti_operating_point import _find_queries, _normalize, _topk_cosine
from build_and_train_edge_classifier import edge_features, range_phase_flat

VAL9 = ['K00', 'K05', 'K08', 'N12', 'N13', 'TOWN', 'DCC', 'KAI', 'RIV']
K, SKIP, DTH = 100, 30, 5.0
W = (0.5, 1.0, 0.0, 0.5)
OUT = REPO / 'results/edge_ablation_fix8.json'


def embeddings_three_ways(cache, sensor, d288, layouts, cfg, model, clf,
                          clf_blob, device):
    graph = _build_eval_graph(
        keyframes=keyframes_from(cache, d288), poses=cache['poses'],
        descriptors=d288, cache=cache, config=cfg, device=device,
        temporal_edge_mode='bidirectional', temporal_direction_mode='none',
        similarity_min_k=0, phase_features=None, sensor_key=sensor)
    with torch.no_grad():
        emb_a = model(graph.to(device)).detach().cpu().numpy()

    r = np.load(REPO / 'artifacts/key_remetrize_r.npy').astype(np.float32)
    d_rm = _normalize(d288 * r[None])
    src, dst, feats, _ = edge_features(
        emb_a, d_rm, range_phase_flat(layouts), cache['poses'],
        want_labels=False)
    mu = clf_blob['mu'].astype(np.float32)
    sd = clf_blob['sd'].astype(np.float32)
    with torch.no_grad():
        logits = clf(torch.from_numpy((feats - mu) / sd).float().to(device))
        probs = torch.sigmoid(logits).cpu().numpy()

    n = len(emb_a)
    keep_k = 10
    dst2d = dst.reshape(n, -1)
    cos2d = feats[:, 0].reshape(n, -1)
    prob2d = probs.reshape(n, -1)

    def second_pass(score2d):
        pick = np.argsort(-score2d, axis=1)[:, :keep_k]
        rows = np.arange(n)[:, None]
        src2 = np.repeat(np.arange(n), keep_k)
        dst2 = dst2d[rows, pick].reshape(-1)
        cos2 = cos2d[rows, pick].reshape(-1)
        prob2 = prob2d[rows, pick].reshape(-1)
        l2_2 = np.linalg.norm(emb_a[src2] - emb_a[dst2], axis=1)
        cfg_t = copy.deepcopy(cfg)
        cfg_t['keyframe'].setdefault('graph', {})['similarity_threshold'] = 2.0
        graph_b = _build_eval_graph(
            keyframes=keyframes_from(cache, d288), poses=cache['poses'],
            descriptors=d288, cache=cache, config=cfg_t, device=device,
            temporal_edge_mode='bidirectional', temporal_direction_mode='none',
            similarity_min_k=0, phase_features=None, sensor_key=sensor)
        dev = graph_b.edge_index.device
        ei = torch.from_numpy(np.stack([src2, dst2])).long().to(dev)
        ea = torch.from_numpy(np.stack([
            np.zeros_like(cos2), np.zeros_like(cos2), cos2,
            np.log1p(l2_2) / 5.0, prob2], axis=1).astype(np.float32)).to(dev)
        graph_b.edge_index = torch.cat([graph_b.edge_index, ei], dim=1)
        graph_b.edge_attr = torch.cat([graph_b.edge_attr, ea], dim=0)
        graph_b.edge_type = torch.cat([
            graph_b.edge_type,
            torch.ones(ei.shape[1], dtype=torch.long, device=dev)])
        with torch.no_grad():
            return _normalize(model(graph_b.to(device)).detach().cpu().numpy())

    return {'pass1': _normalize(emb_a),
            'cosine10': second_pass(cos2d),
            'learned10': second_pass(prob2d)}


def eval_frozen(f, hkey, colbank, poses):
    positions = poses[:, :3, 3]
    queries = _find_queries(poses, DTH, SKIP)
    c = 0
    for q, _ in queries:
        cf = _topk_cosine(f, q, K + 2 * SKIP, SKIP)[:K]
        chh = _topk_cosine(hkey, q, K + 2 * SKIP, SKIP)[:K]
        cand = np.unique(np.concatenate([cf, chh]))
        s = (W[0] * minmax(f[cand] @ f[q]) + W[1] * minmax(hkey[cand] @ hkey[q])
             + W[3] * minmax(column_align_scores(colbank, int(q), cand)))
        c += np.linalg.norm(positions[cand[int(np.argmax(s))]] - positions[q]) < DTH
    return c / max(len(queries), 1), len(queries)


out = json.loads(OUT.read_text()) if OUT.exists() else {}
device = 'cuda'
for seed in (1, 2, 3):
    cfg, model, clf, clf_blob = make_model(device, seed)
    for tag in VAL9:
        if str(seed) in out.get(tag, {}):
            continue
        sensor, sequence, cache_path = JOBS[tag]
        cache = np.load(REPO / cache_path)
        d288 = bin288(cache['fft_magnitudes'].astype(np.float32))
        hc = np.load(CACHE_OUT / f'{tag}_pbev.npz')
        sk = hc['sk_re'].astype(np.float32) + 1j * hc['sk_im'].astype(np.float32)
        hkey, _, _, colbank = variant_inputs(sk, 8)
        layouts = np.load(CACHE_OUT / f'{tag}_height120.npz')['nsd_layout'].astype(np.float32)
        embs = embeddings_three_ways(cache, sensor, d288, layouts, cfg, model,
                                     clf, clf_blob, device)
        rec = {}
        for mode, f in embs.items():
            v, nq = eval_frozen(f.astype(np.float64), hkey.astype(np.float64),
                                colbank, cache['poses'])
            rec[mode] = v
            rec['n_queries'] = nq
        out.setdefault(tag, {})[str(seed)] = rec
        OUT.write_text(json.dumps(out, indent=2))
        print(f'[s{seed} {tag}] ' + ' '.join(
            f'{m}={rec[m]:.4f}' for m in ('pass1', 'cosine10', 'learned10')),
            flush=True)
print('EDGE_ABLATION_DONE', flush=True)
