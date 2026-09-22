#!/usr/bin/env python3
"""Causal edge-classifier contract module (pre-registered causal PR protocol).

Frozen API for the causal two-pass pipeline
(docs/CAUSAL_PR_PREREGISTRATION.md §1.1, §2.1):

  causal_edge_features -- the 14 B3 edge-classifier features of
      scripts/build_and_train_edge_classifier.edge_features, causally
      redefined: candidates come from each query's PREFIX only (j <= i - skip,
      per-node gemv via keyframe.graph_manager._causal_prefix_candidates --
      never a global matrix filtered afterwards), reciprocal rank / dst
      density read candidate j's OWN cached prefix list (the list j had at
      its insertion time), and the index-gap feature uses the FIXED
      normalization min(|i-j|, d_max)/d_max instead of |i-j|/n (sequence
      length is not prefix-invariant).  Every quantity attached to a pair
      (i, j) is computable from frames <= i, so truncating the sequence at t
      leaves all features for nodes <= t bit-identical (per-node gemv dots,
      fixed constants, no population statistics).

  causal_pass1_graph -- the pass-1 CAUSAL eval graph.
      evaluate_kitti_checkpoint._build_eval_graph has no similarity_causal
      passthrough, so this calls build_graph_from_keyframes_batch directly
      with the exact argument set _build_eval_graph uses plus
      temporal_edge_mode='past_to_current', similarity_causal=True, and
      replicates its x_fft / sensor_id / beam_count attachment
      (phase_features is None at every deployed pd call site, so no
      x_phase).  Every emitted edge is past -> current; asserted
      (pre-registration §4.5).

(src, dst) NAMING: src = query node i, dst = past candidate j -- the
bidirectional edge_features convention, kept so downstream per-node
grouping keeps working.  GRAPH INJECTION direction is the consumer's
responsibility: inject as (dst, src) = (j, i), past -> current.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from keyframe.graph_manager import (  # noqa: E402
    _causal_prefix_candidates, build_graph_from_keyframes_batch,
)
from build_and_train_edge_classifier import (  # noqa: E402
    N_FEAT, NEG_D, POS_D, phase_alignment_edge_features,
)
from run_kitti_operating_point import _normalize  # noqa: E402
from _height_candidate_fusion import keyframes_from  # noqa: E402


def causal_edge_features(emb_a, d_rm, phase_flat, poses, *, skip=30, m_cand=30,
                         d_max=5000.0, want_labels=True):
    """Prefix-only B3 edge features. Returns (src, dst, feats(E,14), labels|None).

    emb_a: (n, 416) float32 pass-1 causal GNN embeddings; d_rm: (n, 288)
    re-metrized normalized keys; phase_flat: range_phase_flat output; poses
    (n, 4, 4).  For each node i (i >= skip): candidates j = top-m_cand of
    GNN-embedding cosine over the prefix j <= i - skip (fewer when the prefix
    is small; nodes with empty prefix emit nothing).  Labels: pos < 5 m
    planar, neg >= 25 m, ambiguous band -1 (identical to the bidirectional
    rule).
    """
    n = len(emb_a)
    f_final = _normalize(emb_a)
    ctx = _normalize(emb_a[:, 288:])
    xy = poses[:, :2, 3]

    # Prefix j <= i - skip  <=>  graph_manager gap = skip - 1 (its prefix is
    # [0, i - gap - 1]).  Bit-stable per-node gemv, similarity-descending.
    cand_idx, cand_sims = _causal_prefix_candidates(f_final, None, skip - 1, m_cand)

    # Per-node prefix statistics.  Each node's candidate list IS the cached
    # list it had at insertion time (the sweep only ever looks backwards);
    # empty prefix -> margin/density 0.0.
    margin = np.zeros(n, dtype=np.float32)
    density = np.zeros(n, dtype=np.float32)
    for i in range(n):
        cs = cand_sims[i]
        if len(cs) >= 2:
            margin[i] = cs[0] - cs[1]
        if len(cs):
            density[i] = cs[:min(10, len(cs))].mean()

    src_parts, dst_parts, sim_parts, rank_parts, rr_parts = [], [], [], [], []
    for i in range(n):
        cj, cs = cand_idx[i], cand_sims[i]
        k = len(cj)
        if k == 0:
            continue  # empty prefix emits nothing
        src_parts.append(np.full(k, i, dtype=np.int64))
        dst_parts.append(cj)
        sim_parts.append(cs)
        rank_parts.append(np.arange(k, dtype=np.float32))
        # Causal reciprocal rank: insert position of sim(i, j) in candidate
        # j's cached descending prefix list = count of j's cached sims
        # strictly greater than sim(i, j), capped at m_cand.
        rr = np.empty(k, dtype=np.float32)
        for t in range(k):
            cached = cand_sims[int(cj[t])]
            pos = (int(np.searchsorted(-cached, -cs[t], side="left"))
                   if len(cached) else 0)
            rr[t] = min(pos, m_cand)
        rr_parts.append(rr)

    if not src_parts:
        src = np.empty(0, dtype=np.int64)
        dst = np.empty(0, dtype=np.int64)
        feats = np.empty((0, N_FEAT), dtype=np.float32)
        return src, dst, feats, (np.empty(0, dtype=np.int8) if want_labels else None)

    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    feats = np.empty((len(src), N_FEAT), dtype=np.float32)
    feats[:, 0] = np.concatenate(sim_parts)                    # cos final (pairwise)
    feats[:, 1] = np.einsum("ij,ij->i", d_rm[src], d_rm[dst])  # cos re-metrized key
    feats[:, 2] = np.einsum("ij,ij->i", ctx[src], ctx[dst])    # cos context
    feats[:, 3] = np.concatenate(rank_parts) / m_cand          # prefix rank
    feats[:, 4] = np.concatenate(rr_parts) / m_cand            # causal reciprocal rank
    feats[:, 5] = margin[src]                                  # query prefix margin
    feats[:, 6] = density[src]                                 # src prefix density
    feats[:, 7] = density[dst]                                 # dst CACHED prefix density

    ei = torch.from_numpy(np.stack([src, dst])).long()
    ph = phase_alignment_edge_features(
        torch.from_numpy(phase_flat), ei, None, n_rows=16, n_freqs=4, n_sectors=60,
        include_score=True, similarity_only=False).numpy()
    feats[:, 8:8 + ph.shape[1]] = ph[:, :5]
    feats[:, 13] = np.minimum(np.abs(src - dst).astype(np.float32), d_max) / d_max

    if not want_labels:
        return src, dst, feats, None
    d = np.linalg.norm(xy[src] - xy[dst], axis=1)
    labels = np.full(len(src), -1, dtype=np.int8)
    labels[d < POS_D] = 1
    labels[d >= NEG_D] = 0
    return src, dst, feats, labels


def causal_pass1_graph(cache, d288, cfg, device, sensor):
    """Pass-1 CAUSAL graph: _pd_branch_fusion's construction, made causal.

    Same argument set as evaluate_kitti_checkpoint._build_eval_graph (which
    has no similarity_causal passthrough) with temporal_edge_mode=
    'past_to_current' and similarity_causal=True; temporal_neighbors is read
    from cfg (configs/training_place_disjoint_causal.yaml uses 20 = past-only
    half-window 10).  x_fft / sensor_id / beam_count attachment replicated.
    """
    graph_cfg = cfg["keyframe"].get("graph", {})
    similarity_max_k = graph_cfg.get("similarity_max_k", 10)
    similarity_min_k = 0
    if graph_cfg.get("sensor_similarity", {}).get("enabled", False):
        entry = graph_cfg["sensor_similarity"].get(sensor, {})
        similarity_max_k = int(entry.get("max_k", similarity_max_k))
        similarity_min_k = int(entry.get("min_k", similarity_min_k))
    graph = build_graph_from_keyframes_batch(
        keyframes_from(cache, d288),
        temporal_neighbors=cfg["keyframe"].get("temporal_neighbors", 10),
        device=device,
        poses=cache["poses"],
        descriptors=d288,
        similarity_threshold=graph_cfg.get("similarity_threshold", 0.993),
        similarity_max_k=similarity_max_k,
        similarity_min_k=similarity_min_k,
        similarity_exclude_temporal=graph_cfg.get("similarity_exclude_temporal", True),
        similarity_dist=None,
        similarity_metric=graph_cfg.get("similarity_metric", "cosine"),
        temporal_edge_mode="past_to_current",
        temporal_direction_mode="none",
        similarity_causal=True,
    )
    # Edge audit (pre-registration §4.5): every edge must be past -> current.
    assert bool((graph.edge_index[0] < graph.edge_index[1]).all()), "non-causal edge"
    if "fft_magnitudes" in cache.files:
        fft = cache["fft_magnitudes"].astype(np.float32)
        graph.x_fft = torch.from_numpy(fft.reshape(len(fft), -1)).float().to(device)
    sensor_id_by_key = {"kitti": 0, "nclt": 1, "helipr": 2, "mulran": 3}
    beam_count_by_key = {"kitti": 64.0, "mulran": 64.0, "nclt": 32.0, "helipr": 16.0}
    sensor_id = sensor_id_by_key.get(sensor, len(sensor_id_by_key))
    beam_count = beam_count_by_key.get(sensor, 64.0)
    n = len(d288)
    graph.sensor_id = torch.full((n,), sensor_id, dtype=torch.long, device=device)
    graph.beam_count = torch.full((n, 1), beam_count, dtype=torch.float32, device=device)
    return graph
