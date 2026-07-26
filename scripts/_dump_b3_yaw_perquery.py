#!/usr/bin/env python3
"""Per-query dumps for the deployed B3 pipeline (TODO(student) items, 2026-07-23).

Same pipeline as eval_rank_k.py (pass-1 -> classifier edges -> pass-2 -> 416D
coarse -> phase-sketch rerank at the deployed KITTI grid point (w_b,w_r)=
(0.5,0)), but emits per-query records in the trainer dump schema so that
compute_yaw_recall.py and the paired NSD-vs-SC++ tests can consume them:

  results/per_query_b3/{seed}/coarse/{DATASET}.json   416D cosine top-1
  results/per_query_b3/{seed}/rerank/{DATASET}.json   phase-rerank top-1

Record fields: query_idx, true_match_idx, top1_idx, success_at_k1,
delta_yaw_deg (signed, wrapped to [-180, 180], from the pose rotation block --
identical formula to src/gnn/trainer.py validate dumps).

Run inside nvcr.io/nvidia/pyg:26.01-py3 with the repo at /workspace/...:
  python scripts/_dump_b3_yaw_perquery.py seed1 seed2 seed3
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
    _cache_to_keyframes, _pool_rows, _phase_sketch, _phase_sketch_keys,
    _phase_sketch_distances_fft)
from run_kitti_operating_point import _find_queries, _topk_cosine, _normalize
from build_and_train_edge_classifier import edge_features, range_phase_flat, EdgeMLP, N_FEAT

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
RVEC = REPO / "artifacts/key_remetrize_r.npy"
CLS = REPO / "artifacts/edge_classifier.pt"
OUT = REPO / "results/per_query_b3"
CKPTS = {
    "seed1": REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth",
    "seed2": REPO / "checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth",
    "seed3": REPO / "checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth",
}
SENSORS = {
    "kitti": ["00", "05", "08"], "nclt": ["2012-01-08", "2013-01-10"],
    "helipr": ["Town01"], "mulran": ["DCC03", "KAIST03", "Riverside03"],
}
DATASET_NAMES = {
    ("kitti", "00"): "KITTI_00", ("kitti", "05"): "KITTI_05", ("kitti", "08"): "KITTI_08",
    ("nclt", "2012-01-08"): "NCLT_2012-01-08", ("nclt", "2013-01-10"): "NCLT_2013-01-10",
    ("helipr", "Town01"): "HeLiPR_Town01",
    ("mulran", "DCC03"): "MulRan_DCC03", ("mulran", "KAIST03"): "MulRan_KAIST03",
    ("mulran", "Riverside03"): "MulRan_Riverside03",
}
DTH, SKIP, KEEP_K = 5.0, 30, 10
N_COARSE = 800
WB, WR = 0.5, 0.0  # deployed KITTI grid point (phase_sketch_fusion_bev0.5_range0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _delta_yaw_deg(poses: np.ndarray, qi: int, mi: int) -> float:
    R_q, R_m = poses[qi, :3, :3], poses[mi, :3, :3]
    yaw_q = np.arctan2(R_q[1, 0], R_q[0, 0])
    yaw_m = np.arctan2(R_m[1, 0], R_m[0, 0])
    return float(np.degrees(np.arctan2(np.sin(yaw_q - yaw_m), np.cos(yaw_q - yaw_m))))


def _dump(path: Path, name: str, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"dataset": name, "distance_threshold_m": DTH, "skip_frames": SKIP,
               "n_queries": len(records), "records": records}, open(path, "w"))


def main(seed: str) -> None:
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

    for sensor, seqs in SENSORS.items():
        for seq in seqs:
            name = DATASET_NAMES[(sensor, seq)]
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

            # --- per-query coarse + rerank top-1, trainer dump schema ---
            normed = _normalize(embB3)
            bev_dirs = {"kitti": "preprocessed_kitti_bev_layout", "nclt": "preprocessed_nclt_bev_layout",
                        "helipr": "preprocessed_helipr_bev_layout", "mulran": "preprocessed_mulran_bev_layout"}
            stride = "_stride1" if sensor == "nclt" else ""
            bev = _pool_rows(np.load(REPO / "data" / bev_dirs[sensor] /
                f"{sensor}_bev_layout_{seq}_s60_max_r1-80_z-3-5_h8{stride}.npz")["bev_layouts"].astype(np.float32), 16, "max")
            range_sketch = _phase_sketch(cache["nsd_layouts"].astype(np.float32), 4)
            bev_sketch = _phase_sketch(bev, 8)
            range_keys = _phase_sketch_keys(range_sketch)
            bev_keys = _phase_sketch_keys(bev_sketch)
            positions = poses[:, :3, 3]

            queries = _find_queries(poses, DTH, SKIP)
            rec_coarse, rec_rerank = [], []
            for qi, mi in queries:
                dyaw = _delta_yaw_deg(poses, qi, mi)
                cand_emb = _topk_cosine(normed, qi, N_COARSE, SKIP)
                coarse_top1 = int(cand_emb[0])
                candidates = np.unique(np.concatenate([
                    cand_emb,
                    _topk_cosine(range_keys, qi, N_COARSE, SKIP),
                    _topk_cosine(bev_keys, qi, N_COARSE, SKIP)]))
                emb_d = 1.0 - (normed[candidates] @ normed[qi])
                bev_d = _phase_sketch_distances_fft(bev_sketch[qi], bev_sketch[candidates], n_sectors=60)
                rng_d = _phase_sketch_distances_fft(range_sketch[qi], range_sketch[candidates], n_sectors=60)
                fused = emb_d + WB * bev_d + WR * rng_d          # fusion_norm="raw"
                rerank_top1 = int(candidates[np.argmin(fused)])

                for top1, out in ((coarse_top1, rec_coarse), (rerank_top1, rec_rerank)):
                    hit = bool(np.linalg.norm(positions[top1] - positions[qi]) < DTH)
                    out.append({"query_idx": int(qi), "true_match_idx": int(mi),
                                "top1_idx": top1, "success_at_k1": hit,
                                "delta_yaw_deg": dyaw})

            _dump(OUT / seed / "coarse" / f"{name}.json", name, rec_coarse)
            _dump(OUT / seed / "rerank" / f"{name}.json", name, rec_rerank)
            r1c = np.mean([x["success_at_k1"] for x in rec_coarse]) if rec_coarse else 0.0
            r1r = np.mean([x["success_at_k1"] for x in rec_rerank]) if rec_rerank else 0.0
            print(f"[{seed}] {name:<20} n={len(rec_coarse):<5} coarse R@1={r1c:.4f}  "
                  f"rerank(0.5,0) R@1={r1r:.4f}", flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["seed1"]):
        main(s)
