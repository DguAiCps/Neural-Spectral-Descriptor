#!/usr/bin/env python3
"""Dump builder vs classifier-selected similarity edges per val sequence (viz용).

Per sequence saves: poses, builder graph edge_index/edge_type, classifier
top-10 edges (src, dst, prob). Mirrors run_b3_eval.py pass-1 exactly.
Output: results/_edge_dump/{sensor}_{seq}.npz
"""
from __future__ import annotations
import copy, sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes
from run_kitti_operating_point import _normalize
from build_and_train_edge_classifier import edge_features, range_phase_flat, EdgeMLP, N_FEAT

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
RVEC = REPO / "artifacts/key_remetrize_r.npy"
CLS = REPO / "artifacts/edge_classifier.pt"
OUT = REPO / "results/_edge_dump"
CKPT = REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth"
SENSORS = {"kitti": ["00", "05", "08"], "nclt": ["2012-01-08", "2013-01-10"],
           "helipr": ["Town01"], "mulran": ["DCC03", "KAIST03", "Riverside03"]}
KEEP_K = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    rm_cfg = copy.deepcopy(cfg)
    rm_cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(RVEC)}
    r = np.load(RVEC).astype(np.float32)
    model = _make_model(rm_cfg, CKPT, DEVICE)
    blob = torch.load(CLS, map_location="cpu", weights_only=False)
    clf = EdgeMLP(N_FEAT); clf.load_state_dict(blob["state_dict"]); clf.eval().to(DEVICE)
    mu, sd = blob["mu"].astype(np.float32), blob["sd"].astype(np.float32)

    for sensor, seqs in SENSORS.items():
        for seq in seqs:
            cache = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
            d_raw = cache["descriptors"].astype(np.float32); poses = cache["poses"]
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
            n = len(embA)
            probs2 = probs.reshape(n, -1)
            pick = np.argsort(-probs2, axis=1)[:, :KEEP_K]
            src2 = np.repeat(np.arange(n), KEEP_K)
            dst2 = dst.reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
            prob2 = probs2[np.arange(n)[:, None], pick].reshape(-1)
            np.savez_compressed(OUT / f"{sensor}_{seq}.npz",
                poses=poses.astype(np.float32),
                builder_edge_index=graph.edge_index.cpu().numpy(),
                builder_edge_type=graph.edge_type.cpu().numpy(),
                cls_src=src2.astype(np.int32), cls_dst=dst2.astype(np.int32),
                cls_prob=prob2.astype(np.float32))
            print(f"[dump] {sensor}/{seq} n={n}", flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
