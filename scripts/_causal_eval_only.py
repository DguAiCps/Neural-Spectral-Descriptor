#!/usr/bin/env python3
"""Causal-deployment gap for the frozen deployed pipeline (no retraining).

Reruns the deployed B3 coarse path (frozen 800D checkpoint + fixed diag_wccn
key re-metrization A + classifier-selected similarity edges) twice per
sequence:
  bidir:     deployed default temporal graph (5 past + 5 future, M=10)
  past_only: temporal edges restricted to the PAST only, same total k=10
             (edges j -> i for i-10 <= j < i)

The past-only intervention is applied to BOTH forwards of the two-pass B3
path (pass-1 candidate embedding and the final temporal-only + classifier-edge
graph), since a causal deployment has no future frames at either stage.
Everything else — cosine floor 0.993 pass-1 similarity edges, top-30
candidate pool, edge-classifier top-10 reselection, 5D edge-attr layout —
is byte-identical to scripts/run_b3_rerank.py's coarse computation, so the
bidir numbers must reproduce results/_remetrize_twopass/b3_rerank_<seed>.json
"coarse" values exactly (validation gate).

Output (merged incrementally): results/_remetrize_twopass/causal_eval_only.json
  {seed: {"sensor/seq": {"bidir": r1, "past_only": r1}}}

Usage:
  python scripts/_causal_eval_only.py seed1                      # all 9 seqs
  python scripts/_causal_eval_only.py seed1 seed2 seed3
  python scripts/_causal_eval_only.py seed1 --seqs nclt/2013-01-10   # smoke
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import (_apply_encoder_preset, _make_model,
    _build_eval_graph, _cache_to_keyframes)
from run_kitti_operating_point import _find_queries, _topk_cosine, _score, _normalize
from build_and_train_edge_classifier import edge_features, range_phase_flat, EdgeMLP, N_FEAT

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
RVEC = REPO / "artifacts/key_remetrize_r.npy"
CLS = REPO / "artifacts/edge_classifier.pt"
OUT = REPO / "results/_remetrize_twopass"
CKPTS = {
    "seed1": REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth",
    "seed2": REPO / "checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth",
    "seed3": REPO / "checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth",
}
SENSORS = {
    "kitti": ["00", "05", "08"], "nclt": ["2012-01-08", "2013-01-10"],
    "helipr": ["Town01"], "mulran": ["DCC03", "KAIST03", "Riverside03"],
}
DTH, SKIP, KEEP_K, PAST_K = 5.0, 30, 10, 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def r1(emb, poses):
    normed = _normalize(emb)
    q = _find_queries(poses, DTH, SKIP)
    ranked = [_topk_cosine(normed, i, 1 + 2 * SKIP, SKIP)[:1] for i, _ in q]
    return _score(poses, q, ranked, [1], DTH)["R@1"]


def _replace_temporal_past_only(graph, poses, k=PAST_K):
    """Drop the builder's bidirectional temporal edges (edge_type==0) and
    replace them with past-only edges j -> i for i-k <= j < i. Edge attrs
    mirror the builder's _append_temporal_edge with direction_mode='none':
    [log1p(dist)/5, angle/pi, 0, 0, 1]. Similarity edges are untouched."""
    ei, ea, et = graph.edge_index, graph.edge_attr, graph.edge_type
    keep = et != 0
    n = int(graph.num_nodes)
    dst = np.concatenate([np.full(min(k, i), i, dtype=np.int64) for i in range(1, n)])
    src = np.concatenate([np.arange(max(0, i - k), i, dtype=np.int64) for i in range(1, n)])
    pos = poses[:, :3, 3]
    norm_dist = np.log1p(np.linalg.norm(pos[src] - pos[dst], axis=1)) / 5.0
    R_rel = np.einsum("nij,nkj->nik", poses[dst, :3, :3], poses[src, :3, :3])
    trace = np.clip(np.trace(R_rel, axis1=1, axis2=2), -1.0, 3.0)
    norm_rot = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)) / np.pi
    attr = np.stack([norm_dist, norm_rot, np.zeros_like(norm_dist),
                     np.zeros_like(norm_dist), np.ones_like(norm_dist)],
                    axis=1).astype(np.float32)
    dev = ei.device
    ei_t = torch.from_numpy(np.stack([src, dst])).long().to(dev)
    graph.edge_index = torch.cat([ei_t, ei[:, keep]], dim=1)
    graph.edge_attr = torch.cat([torch.from_numpy(attr).to(dev), ea[keep]], dim=0)
    graph.edge_type = torch.cat(
        [torch.zeros(ei_t.shape[1], dtype=torch.long, device=dev), et[keep]])
    return graph


def _b3_coarse(model, clf, mu, sd, r, rm_cfg, cache, sensor, poses, d_raw, past_only):
    # pass 1: paper graph (cosine floor 0.993 similarity edges) -> candidates
    graph = _build_eval_graph(
        keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
        cache=cache, config=rm_cfg, device=DEVICE, temporal_edge_mode="bidirectional",
        temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
        sensor_key=sensor)
    if past_only:
        graph = _replace_temporal_past_only(graph, poses)
    with torch.no_grad():
        embA = model(graph.to(DEVICE)).cpu().numpy()
    d_rm = _normalize(d_raw * r[None, :])
    x_phase = range_phase_flat(cache["nsd_layouts"])
    src, dst, feats, _ = edge_features(embA, d_rm, x_phase, poses, want_labels=False)
    with torch.no_grad():
        logits = clf(torch.from_numpy((feats - mu) / sd).float().to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()

    # per-node top-K by classifier probability (identical to run_b3_rerank)
    n = len(embA)
    probs2 = probs.reshape(n, -1)
    pick = np.argsort(-probs2, axis=1)[:, :KEEP_K]
    src2 = np.repeat(np.arange(n), KEEP_K)
    dst2 = dst.reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
    cos2 = feats[:, 0].reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
    prob2 = probs2[np.arange(n)[:, None], pick].reshape(-1)
    l2_2 = np.linalg.norm(embA[src2] - embA[dst2], axis=1)

    # pass 2: temporal-only base graph + injected classifier edges
    cfg_t = copy.deepcopy(rm_cfg)
    cfg_t["keyframe"].setdefault("graph", {})["similarity_threshold"] = 2.0
    g = _build_eval_graph(
        keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
        cache=cache, config=cfg_t, device=DEVICE, temporal_edge_mode="bidirectional",
        temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
        sensor_key=sensor)
    if past_only:
        g = _replace_temporal_past_only(g, poses)
    dev = g.edge_index.device
    ei_new = torch.from_numpy(np.stack([src2, dst2])).long().to(dev)
    attr_new = torch.from_numpy(np.stack([
        np.zeros_like(cos2), np.zeros_like(cos2), cos2,
        np.log1p(l2_2) / 5.0, prob2], axis=1).astype(np.float32)).to(dev)
    g.edge_index = torch.cat([g.edge_index, ei_new], dim=1)
    g.edge_attr = torch.cat([g.edge_attr, attr_new], dim=0)
    g.edge_type = torch.cat(
        [g.edge_type, torch.ones(ei_new.shape[1], dtype=torch.long, device=dev)])
    with torch.no_grad():
        embB3 = model(g.to(DEVICE)).cpu().numpy()
    return r1(embB3, poses)


def main(seeds, seq_filter=None):
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "causal_eval_only.json"
    report = json.load(open(out_path)) if out_path.exists() else {}

    cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    rm_cfg = copy.deepcopy(cfg)
    rm_cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(RVEC)}
    r = np.load(RVEC).astype(np.float32)

    blob = torch.load(CLS, map_location="cpu", weights_only=False)
    clf = EdgeMLP(N_FEAT)
    clf.load_state_dict(blob["state_dict"])
    clf.eval().to(DEVICE)
    mu, sd = blob["mu"].astype(np.float32), blob["sd"].astype(np.float32)

    for seed in seeds:
        model = _make_model(rm_cfg, CKPTS[seed], DEVICE)
        report.setdefault(seed, {})
        for sensor, seqs in SENSORS.items():
            for seq in seqs:
                name = f"{sensor}/{seq}"
                if seq_filter is not None and name not in seq_filter:
                    continue
                cache = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
                d_raw = cache["descriptors"].astype(np.float32)
                poses = cache["poses"]
                bidir = _b3_coarse(model, clf, mu, sd, r, rm_cfg, cache, sensor,
                                   poses, d_raw, past_only=False)
                past = _b3_coarse(model, clf, mu, sd, r, rm_cfg, cache, sensor,
                                  poses, d_raw, past_only=True)
                report[seed][name] = {"bidir": bidir, "past_only": past}
                print(f"[causal:{seed}] {name:<20} bidir={bidir:.6f} "
                      f"past_only={past:.6f} gap={bidir - past:+.4f}", flush=True)
                json.dump(report, open(out_path, "w"), indent=1)
    print("saved", out_path, flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    seq_filter = None
    if "--seqs" in args:
        i = args.index("--seqs")
        seq_filter = set(args[i + 1].split(","))
        args = args[:i] + args[i + 2:]
    main(args or ["seed1"], seq_filter)
