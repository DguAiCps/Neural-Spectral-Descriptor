#!/usr/bin/env python3
"""B3 evaluation: similarity edges selected by the post-hoc edge classifier.

Frozen seed-1 checkpoint (fixed diag_wccn key). Per validation sequence:
pass-1 embeddings on the paper graph -> top-30 cosine candidates per node ->
classifier probability -> keep top-10 per node -> inject as similarity edges
(same 5D attr layout as the builder) on a temporal-only graph -> forward.
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes
from run_kitti_operating_point import _find_queries, _topk_cosine, _score, _normalize
from build_and_train_edge_classifier import edge_features, range_phase_flat, EdgeMLP, N_FEAT

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
RVEC = REPO / "artifacts/key_remetrize_r.npy"
CLS = REPO / "results/_edge_cls/classifier.pt"
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
DTH, SKIP, KEEP_K = 5.0, 30, 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def r1(emb, poses):
    normed = _normalize(emb)
    q = _find_queries(poses, DTH, SKIP)
    ranked = [_topk_cosine(normed, i, 1 + 2 * SKIP, SKIP)[:1] for i, _ in q]
    return _score(poses, q, ranked, [1], DTH)["R@1"]


def main(seed='seed1'):
    cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    rm_cfg = copy.deepcopy(cfg)
    rm_cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(RVEC)}
    r = np.load(RVEC).astype(np.float32)
    model = _make_model(rm_cfg, CKPTS[seed], DEVICE)

    blob = torch.load(CLS, map_location="cpu", weights_only=False)
    clf = EdgeMLP(N_FEAT)
    clf.load_state_dict(blob["state_dict"])
    clf.eval().to(DEVICE)
    mu, sd = blob["mu"].astype(np.float32), blob["sd"].astype(np.float32)

    report = {}
    for sensor, seqs in SENSORS.items():
        for seq in seqs:
            name = f"{sensor}/{seq}"
            cache = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
            d_raw = cache["descriptors"].astype(np.float32)
            poses = cache["poses"]
            graph = _build_eval_graph(
                keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
                cache=cache, config=rm_cfg, device=DEVICE, temporal_edge_mode="bidirectional",
                temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
                sensor_key=sensor)
            with torch.no_grad():
                embA = model(graph.to(DEVICE)).cpu().numpy()
            d_rm = _normalize(d_raw * r[None, :])
            x_phase = range_phase_flat(cache["nsd_layouts"])
            src, dst, feats, _ = edge_features(embA, d_rm, x_phase, poses, want_labels=False)
            with torch.no_grad():
                logits = clf(torch.from_numpy((feats - mu) / sd).float().to(DEVICE))
                probs = torch.sigmoid(logits).cpu().numpy()

            # per-node top-K by classifier probability
            n = len(embA)
            probs2 = probs.reshape(n, -1)
            pick = np.argsort(-probs2, axis=1)[:, :KEEP_K]
            src2 = np.repeat(np.arange(n), KEEP_K)
            dst2 = dst.reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
            cos2 = feats[:, 0].reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
            prob2 = probs2[np.arange(n)[:, None], pick].reshape(-1)
            l2_2 = np.linalg.norm(embA[src2] - embA[dst2], axis=1)

            # temporal-only base graph, then inject classifier edges (attr layout
            # mirrors the builder: [0, 0, cos, log1p(l2)/5, posterior])
            cfg_t = copy.deepcopy(rm_cfg)
            cfg_t["keyframe"].setdefault("graph", {})["similarity_threshold"] = 2.0
            g = _build_eval_graph(
                keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
                cache=cache, config=cfg_t, device=DEVICE, temporal_edge_mode="bidirectional",
                temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
                sensor_key=sensor)
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
            report[name] = r1(embB3, poses)
            print(f"[b3] {name:<20} B3={report[name]:.3f}", flush=True)

    json.dump(report, open(OUT / f"b3_{seed}.json", "w"), indent=1)
    print("saved", OUT / f"b3_{seed}.json", flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["seed1"]):
        main(s)
