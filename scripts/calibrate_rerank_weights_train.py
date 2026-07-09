#!/usr/bin/env python3
"""Per-sensor rerank-weight calibration on the TRAINING split.

Runs the full B3 two-pass + phase-rerank grid on the training sequences (the 23
non-validation sequences), so that the fusion weights (w_b, w_r) can be selected
per sensor family WITHOUT touching the evaluation sequences. Mirrors
run_b3_rerank.py exactly except: (i) the sequence list is the training split,
(ii) BEV layouts are read from data/preprocessed_cross_sensor_bev_layout (which
holds every sequence, unlike the val-only per-sensor dirs), and (iii) only the
seed-1 model is used (weights are structural and shared across seeds).

Output: results/_remetrize_twopass/b3_train_calib_seed1.json
        {sensor/seq: {coarse, grid:{phase_sketch_fusion_bevW_rangeW: R@1}}}
Downstream: summarize_rerank_calibration.py picks the best per-sensor weight on
train and applies it to the committed val grids.
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import (_apply_encoder_preset, _make_model, _build_eval_graph,
    _cache_to_keyframes, _recall_with_phase_sketch_fusion, _pool_rows)
from run_kitti_operating_point import _find_queries, _topk_cosine, _score, _normalize
from build_and_train_edge_classifier import edge_features, range_phase_flat, EdgeMLP, N_FEAT

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
BEVDIR = REPO / "data/preprocessed_cross_sensor_bev_layout"
RVEC = REPO / "artifacts/key_remetrize_r.npy"
CLS = REPO / "artifacts/edge_classifier.pt"
OUT = REPO / "results/_remetrize_twopass"
CKPT = REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth"  # seed1
TRAIN = {
    "kitti": ["01", "02", "06", "07"],
    "nclt": ["2012-05-11", "2012-08-04", "2012-11-04", "2012-11-16", "2013-02-23"],
    "helipr": ["Bridge01", "DCC04", "KAIST04", "Riverside04", "Roundabout01", "Town02"],
    "mulran": ["DCC01", "DCC02", "KAIST01", "KAIST02", "Riverside01", "Riverside02",
               "Sejong01", "Sejong02"],
}
DTH, SKIP, KEEP_K = 5.0, 30, 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def r1(emb, poses):
    normed = _normalize(emb)
    q = _find_queries(poses, DTH, SKIP)
    ranked = [_topk_cosine(normed, i, 1 + 2 * SKIP, SKIP)[:1] for i, _ in q]
    return _score(poses, q, ranked, [1], DTH)["R@1"]


def main():
    cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    rm_cfg = copy.deepcopy(cfg)
    rm_cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(RVEC)}
    r = np.load(RVEC).astype(np.float32)
    model = _make_model(rm_cfg, CKPT, DEVICE)

    blob = torch.load(CLS, map_location="cpu", weights_only=False)
    clf = EdgeMLP(N_FEAT)
    clf.load_state_dict(blob["state_dict"])
    clf.eval().to(DEVICE)
    mu, sd = blob["mu"].astype(np.float32), blob["sd"].astype(np.float32)

    report = {}
    for sensor, seqs in TRAIN.items():
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

            n = len(embA)
            probs2 = probs.reshape(n, -1)
            pick = np.argsort(-probs2, axis=1)[:, :KEEP_K]
            src2 = np.repeat(np.arange(n), KEEP_K)
            dst2 = dst.reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
            cos2 = feats[:, 0].reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
            prob2 = probs2[np.arange(n)[:, None], pick].reshape(-1)
            l2_2 = np.linalg.norm(embA[src2] - embA[dst2], axis=1)

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
            coarse = r1(embB3, poses)
            bev = _pool_rows(np.load(BEVDIR /
                f"{sensor}_bev_layout_{seq}_s60_max_r1-80_z-3-5_h8.npz")["bev_layouts"].astype(np.float32), 16, "max")
            res = _recall_with_phase_sketch_fusion(
                embeddings=embB3, range_layouts=cache["nsd_layouts"].astype(np.float32),
                bev_layouts=bev, poses=poses, k_values=[1], distance_threshold=DTH,
                skip_frames=SKIP, n_coarse=800, range_freqs=4, bev_freqs=8, n_sectors=60,
                sketch_bev_weights=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
                sketch_range_weights=[0.0, 0.5, 1.0, 2.0, 4.0],
                rerank_mode="sketch_fft", fusion_norm="raw")
            grid = {k: v["R@1"] for k, v in res.items() if not k.startswith("_")}
            report[name] = {"coarse": coarse, "grid": grid, "n_q": int(len(_find_queries(poses, DTH, SKIP)))}
            print(f"[calib] {name:<22} coarse={coarse:.3f} best={max(max(grid.values()),coarse):.3f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT / "b3_train_calib_seed1.json", "w"), indent=1)
    print("saved", OUT / "b3_train_calib_seed1.json", flush=True)


if __name__ == "__main__":
    main()
