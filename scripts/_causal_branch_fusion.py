#!/usr/bin/env python3
"""Causal two-pass NSD embedding for the pre-registered causal PR evaluation.

Mirror of scripts/_pd_branch_fusion.make_model_pd / nsd_embedding_pd
(docs/CAUSAL_PR_PREREGISTRATION.md §2.1(3)) with the causal replacements:

  pass-1 graph   scripts/_causal_edge_features.causal_pass1_graph
                 (past_to_current temporal + similarity_causal prefix edges)
  features       scripts/_causal_edge_features.causal_edge_features
                 (prefix-only candidates; frozen 14-column contract)
  edge selection top-10 per node by classifier prob with PER-NODE GROUPING —
                 candidate counts are ragged near the sequence start
                 (k_i = min(30, i - 29) for i >= 30), so the bidirectional
                 fixed (n, 30) reshape does not apply
  pass-2 graph   temporal-only causal (similarity_threshold=2.0 trick +
                 past_to_current + similarity_causal), selected edges injected
                 as (j, i) = edge_index rows [dst_candidates, src_nodes]
                 (past -> current; the (src, dst) arrays keep the
                 bidirectional NAMING src = query i, dst = candidate j),
                 same 5D edge_attr layout [0, 0, cos, log1p(l2)/5, prob]

Two tracks (pre-registration §0), both re-using the pd omega
artifacts/key_remetrize_r_pd2026.npy (checkpoint-free, §2.3):

  --variant pd_causal  checkpoints/pd_causal_s{seed}  (causal-retrained, main)
                       + artifacts/edge_classifier_pd2026_causal.pt
  --variant pd         checkpoints/pd_s{seed}  (Frozen-GNN causal adaptation)
                       + artifacts/edge_classifier_pd2026_frozengnn.pt

Both variants build the model/graph from configs/training_place_disjoint_causal.yaml
so the eval-time causal graph condition is identical across tracks
(temporal_neighbors 20 = past 10, matching the ~10 incoming temporal degree
both checkpoints were trained with; the yaml is byte-identical to
training_place_disjoint.yaml outside the three causal graph keys, so model
construction for the frozen pd checkpoints is unchanged).

Library module (no CLI): the per-query PR ledger driver is
scripts/_causal_pr_dump.py.  The bidirectional pipeline files stay untouched.
"""
from __future__ import annotations
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model  # noqa: E402
from run_kitti_operating_point import _normalize  # noqa: E402
from build_and_train_edge_classifier import (  # noqa: E402
    EdgeMLP, N_FEAT, range_phase_flat,
)
from _causal_edge_features import (  # noqa: E402
    causal_edge_features, causal_pass1_graph,
)

VARIANTS = {
    "pd_causal": dict(  # causal-retrained track (main row)
        ckpt_dir="pd_causal_s{seed}",
        rvec=REPO / "artifacts/key_remetrize_r_pd2026.npy",
        cls=REPO / "artifacts/edge_classifier_pd2026_causal.pt",
        config=REPO / "configs/training_place_disjoint_causal.yaml",
    ),
    "pd": dict(  # Frozen-GNN causal adaptation track
        ckpt_dir="pd_s{seed}",
        rvec=REPO / "artifacts/key_remetrize_r_pd2026.npy",
        cls=REPO / "artifacts/edge_classifier_pd2026_frozengnn.pt",
        config=REPO / "configs/training_place_disjoint_causal.yaml",
    ),
}


def make_model_causal(variant: str, seed: int, device: str):
    """_pd_branch_fusion.make_model_pd wired to the causal variant table."""
    V = VARIANTS[variant]
    cfg = _apply_encoder_preset(yaml.safe_load(open(V["config"])), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    cfg = copy.deepcopy(cfg)
    cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(V["rvec"])}
    ckpt = REPO / "checkpoints" / V["ckpt_dir"].format(seed=seed) / "best_model.pth"
    model = _make_model(cfg, ckpt, device)
    blob = torch.load(V["cls"], map_location="cpu", weights_only=False)
    clf = EdgeMLP(N_FEAT)
    clf.load_state_dict(blob["state_dict"])
    clf.eval().to(device)
    r = np.load(V["rvec"]).astype(np.float32)
    return cfg, model, clf, blob, r, ckpt


def select_edges_per_node(src: np.ndarray, probs: np.ndarray,
                          keep_k: int = 10) -> np.ndarray:
    """Ragged per-node top-keep_k by classifier prob.

    Candidate counts vary near the sequence start (k_i = min(30, i - 29)), so
    the bidirectional fixed (n, M) reshape does not apply: src is
    non-decreasing by construction (the feature sweep emits nodes in ascending
    order), group by src runs and take top-keep_k inside each run (stable sort;
    reads only the node's own candidate block -> prefix-invariant).  Returns
    the selected indices into src/probs (grouped, node-ascending)."""
    assert len(src) == len(probs)
    if not len(src):
        return np.empty(0, dtype=np.int64)
    assert np.all(np.diff(src) >= 0), "src must be grouped non-decreasing"
    starts = np.flatnonzero(np.r_[True, src[1:] != src[:-1]])
    bounds = np.r_[starts, len(src)]
    sel = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        pick = np.argsort(-probs[a:b], kind="stable")[:keep_k]
        sel.append(a + pick)
    return np.concatenate(sel)


def nsd_embedding_causal(cache, sensor: str, d288: np.ndarray, layouts: np.ndarray,
                         cfg, model, clf, clf_blob, r: np.ndarray, device: str):
    """scripts/_pd_branch_fusion.nsd_embedding_pd with the causal replacements
    listed in the module docstring.  Returns the L2-normalized 416D embedding."""
    graph = causal_pass1_graph(cache, d288, cfg, device, sensor)
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
    graph_b = causal_pass1_graph(cache, d288, cfg_t, device, sensor)
    assert int((graph_b.edge_type == 1).sum()) == 0, "pass-2 base must be temporal-only"
    dev = graph_b.edge_index.device
    # R3 injection direction: (j, i) = [dst_candidates, src_nodes].
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
    assert bool((graph_b.edge_index[0] < graph_b.edge_index[1]).all()), \
        "pass-2 graph contains a non past->current edge"
    with torch.no_grad():
        emb_b = model(graph_b.to(device)).detach().cpu().numpy()
    return _normalize(emb_b)
