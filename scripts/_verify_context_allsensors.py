#!/usr/bin/env python3
"""Context-isolation ablation at the MAIN 800D operating point, all 4 sensors, 3 seeds.

Reports 288D raw magnitude key vs 128D context-only vs 416D retrieval key R@1
(cosine retrieval, before phase rerank) plus deployed gate alpha per sensor.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes
from run_kitti_operating_point import _find_queries, _topk_cosine, _score, _normalize

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
CKPTS = {
    "seed1": REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth",
    "seed2": REPO / "checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth",
    "seed3": REPO / "checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth",
}
SENSORS = {
    "kitti":  ["00", "05", "08"],
    "nclt":   ["2012-01-08", "2013-01-10"],
    "helipr": ["Town01"],
    "mulran": ["DCC03", "KAIST03", "Riverside03"],
}
DTH, SKIP, KVALS = 5.0, 30, [1, 5, 10]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def r1(emb, poses):
    normed = _normalize(emb)
    q = _find_queries(poses, DTH, SKIP)
    ranked = [_topk_cosine(normed, i, max(KVALS) + 2 * SKIP, SKIP)[:max(KVALS)] for i, _ in q]
    return _score(poses, q, ranked, KVALS, DTH)["R@1"], len(q)


def main():
    cfg = _apply_encoder_preset(yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625

    report = {}
    for seed, ckpt in CKPTS.items():
        if not ckpt.exists():
            print("SKIP", seed); continue
        model = _make_model(cfg, ckpt, DEVICE)
        core = model.gnn if hasattr(model, "gnn") else model
        rep = {}
        for sensor, seqs in SENSORS.items():
            for seq in seqs:
                cpath = CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz"
                if not cpath.exists():
                    print("MISS", cpath); continue
                cache = np.load(cpath)
                desc = cache["descriptors"].astype(np.float32); poses = cache["poses"]
                graph = _build_eval_graph(
                    keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=desc,
                    cache=cache, config=cfg, device=DEVICE, temporal_edge_mode="bidirectional",
                    temporal_direction_mode="none", similarity_min_k=0, phase_features=None, sensor_key=sensor)
                with torch.no_grad():
                    emb = model(graph.to(DEVICE)).detach().cpu().numpy()
                alpha = getattr(core, "_last_alpha", None)
                a_mean = float(alpha.detach().cpu().numpy().mean()) if alpha is not None else None
                raw_r1, nq = r1(desc, poses)
                ctx_r1, _ = r1(emb[:, 288:416], poses)
                fin_r1, _ = r1(emb, poses)
                rep[f"{sensor}/{seq}"] = {
                    "n_q": nq, "raw288": round(raw_r1, 4), "ctx128": round(ctx_r1, 4),
                    "final416": round(fin_r1, 4), "delta": round(fin_r1 - raw_r1, 4),
                    "alpha_mean": round(a_mean, 4) if a_mean is not None else None,
                }
                print(f"[{seed}] {sensor}/{seq:<12} raw={raw_r1:.4f} ctx={ctx_r1:.4f} "
                      f"final={fin_r1:.4f} d={fin_r1-raw_r1:+.4f} a={a_mean:.3f}", flush=True)
        report[seed] = rep
    out = REPO / "results/_verify_context_allsensors.json"
    out.write_text(json.dumps(report, indent=2))
    print("WROTE", out)


if __name__ == "__main__":
    main()
