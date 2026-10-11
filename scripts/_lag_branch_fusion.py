#!/usr/bin/env python3
"""Fixed-lag (15 keyframes) two-pass NSD embedding (2026-10-09).

Same pipeline as scripts/_causal_branch_fusion.nsd_embedding_causal except
the temporal edges: every node receives from its +-5 temporal neighbours
(temporal_neighbors 10, bidirectional), while similarity edges (pass 1) and
classifier-selected edges (pass 2) stay past -> current with prefix-only
candidates. With 2 GNN layers the embedding of node k then depends on frames
<= k + 15 (pass 2: +10 through temporal hops, plus +5 more because the
pass-2 edge attributes / selection of node n <= k + 5 use the pass-1
embedding of n, which reaches n + 10). The decision for keyframe k is
therefore available at keyframe k + 15; `prefix_check` verifies this bound
by re-running the pipeline on truncated caches.

Library module: scripts/_pdv2_lag.py patches it into the causal drivers.
"""
from __future__ import annotations
import copy
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from keyframe.graph_manager import build_graph_from_keyframes_batch  # noqa: E402
from run_kitti_operating_point import _normalize  # noqa: E402
from build_and_train_edge_classifier import range_phase_flat  # noqa: E402
from _height_candidate_fusion import keyframes_from  # noqa: E402
from _causal_edge_features import causal_edge_features  # noqa: E402
from _causal_branch_fusion import select_edges_per_node  # noqa: E402

LAG = 15


def _audit(graph, half_window: int) -> None:
    """Temporal edges within the +-half window; every other edge past -> current."""
    src, dst = graph.edge_index[0], graph.edge_index[1]
    temporal = graph.edge_type == 0
    assert bool(((src[temporal] - dst[temporal]).abs() <= half_window).all()), \
        "temporal edge outside the +-half window"
    assert bool((src[~temporal] < dst[~temporal]).all()), \
        "non-temporal edge is not past -> current"


def lag_pass1_graph(cache, d288, cfg, device, sensor):
    """scripts/_causal_edge_features.causal_pass1_graph with bidirectional
    temporal edges (similarity edges keep similarity_causal=True)."""
    graph_cfg = cfg["keyframe"].get("graph", {})
    similarity_max_k = graph_cfg.get("similarity_max_k", 10)
    similarity_min_k = 0
    if graph_cfg.get("sensor_similarity", {}).get("enabled", False):
        entry = graph_cfg["sensor_similarity"].get(sensor, {})
        similarity_max_k = int(entry.get("max_k", similarity_max_k))
        similarity_min_k = int(entry.get("min_k", similarity_min_k))
    temporal_neighbors = cfg["keyframe"].get("temporal_neighbors", 10)
    graph = build_graph_from_keyframes_batch(
        keyframes_from(cache, d288),
        temporal_neighbors=temporal_neighbors,
        device=device,
        poses=cache["poses"],
        descriptors=d288,
        similarity_threshold=graph_cfg.get("similarity_threshold", 0.993),
        similarity_max_k=similarity_max_k,
        similarity_min_k=similarity_min_k,
        similarity_exclude_temporal=graph_cfg.get("similarity_exclude_temporal", True),
        similarity_dist=None,
        similarity_metric=graph_cfg.get("similarity_metric", "cosine"),
        temporal_edge_mode="bidirectional",
        temporal_direction_mode="none",
        similarity_causal=True,
    )
    _audit(graph, temporal_neighbors // 2)
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


def nsd_embedding_lag(cache, sensor: str, d288: np.ndarray, layouts: np.ndarray,
                      cfg, model, clf, clf_blob, r: np.ndarray, device: str):
    """nsd_embedding_causal with lag_pass1_graph. Returns the L2-normalized
    416D embedding; row k is final once keyframe k + LAG exists."""
    half_window = cfg["keyframe"].get("temporal_neighbors", 10) // 2
    graph = lag_pass1_graph(cache, d288, cfg, device, sensor)
    with torch.no_grad():
        emb_a = model(graph.to(device)).detach().cpu().numpy()

    d_rm = _normalize(d288 * r[None])
    src, dst, feats, _ = causal_edge_features(
        emb_a, d_rm, range_phase_flat(layouts), cache["poses"], want_labels=False,
    )
    if len(src):
        mu = clf_blob["mu"].astype(np.float32)
        sd = clf_blob["sd"].astype(np.float32)
        with torch.no_grad():
            logits = clf(torch.from_numpy((feats - mu) / sd).float().to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
        sel = select_edges_per_node(src, probs, keep_k=10)
        src2, dst2 = src[sel], dst[sel]
        cos2 = feats[sel, 0]
        prob2 = probs[sel].astype(np.float32)
        l2_2 = np.linalg.norm(emb_a[src2] - emb_a[dst2], axis=1)
    else:
        src2 = dst2 = np.empty(0, dtype=np.int64)
        cos2 = prob2 = l2_2 = np.empty(0, dtype=np.float32)
    assert np.all(dst2 < src2), "selected edge candidate is not strictly past"

    cfg_t = copy.deepcopy(cfg)
    cfg_t["keyframe"].setdefault("graph", {})["similarity_threshold"] = 2.0
    graph_b = lag_pass1_graph(cache, d288, cfg_t, device, sensor)
    assert int((graph_b.edge_type == 1).sum()) == 0, "pass-2 base must be temporal-only"
    dev = graph_b.edge_index.device
    edge_index = torch.from_numpy(np.stack([dst2, src2])).long().to(dev)
    edge_attr = torch.from_numpy(np.stack([
        np.zeros_like(cos2), np.zeros_like(cos2), cos2,
        np.log1p(l2_2) / 5.0, prob2,
    ], axis=1).astype(np.float32)).to(dev)
    graph_b.edge_index = torch.cat([graph_b.edge_index, edge_index], dim=1)
    graph_b.edge_attr = torch.cat([graph_b.edge_attr, edge_attr], dim=0)
    graph_b.edge_type = torch.cat([
        graph_b.edge_type,
        torch.ones(edge_index.shape[1], dtype=torch.long, device=dev),
    ])
    _audit(graph_b, half_window)
    with torch.no_grad():
        emb_b = model(graph_b.to(device)).detach().cpu().numpy()
    return _normalize(emb_b)


class TruncatedCache:
    """First n keyframes of an operating cache (npz-like: .files + [key])."""

    def __init__(self, cache, n: int):
        self.n_total = len(cache["poses"])
        self._d = {k: (cache[k][:n] if np.ndim(cache[k]) and len(cache[k]) == self.n_total
                       else cache[k]) for k in cache.files}
        self.files = list(self._d)

    def __getitem__(self, key):
        return self._d[key]


def prefix_check(cache, sensor, d288, layouts, cfg, model, clf, clf_blob, r, device,
                 ks, lag: int = LAG, embed=nsd_embedding_lag):
    """For each k: embedding of node k from the prefix [0, k+lag] vs the full
    sequence (must match), and from [0, k+lag-1] (expected to differ)."""
    full = embed(cache, sensor, d288, layouts, cfg, model, clf, clf_blob, r, device)
    rows = []
    for k in ks:
        out = {"k": int(k)}
        for name, n in (("at_lag", k + lag + 1), ("before_lag", k + lag)):
            tc = TruncatedCache(cache, n)
            e = embed(tc, sensor, d288[:n], layouts[:n], cfg, model, clf, clf_blob,
                      r, device)
            out[name] = float(np.abs(e[k] - full[k]).max())
            # rows 0..k must also be unchanged when the prefix ends at k + lag
            if name == "at_lag":
                out["at_lag_rows_le_k"] = float(np.abs(e[:k + 1] - full[:k + 1]).max())
        rows.append(out)
    return rows
