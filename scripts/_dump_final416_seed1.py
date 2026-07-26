#!/usr/bin/env python3
"""Dump seed-1 416D retrieval-key embeddings for the 9 validation sequences.

Adapted from scripts/_verify_context_allsensors.py (same graph build + forward);
saves one npz per sequence for host-side analysis/plotting (no torch needed there).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
CKPT = REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth"  # seed 1
OUT = REPO / "results/_final416_seed1"
SENSORS = {
    "kitti":  ["00", "05", "08"],
    "nclt":   ["2012-01-08", "2013-01-10"],
    "helipr": ["Town01"],
    "mulran": ["DCC03", "KAIST03", "Riverside03"],
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _apply_encoder_preset(yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    model = _make_model(cfg, CKPT, DEVICE)

    for sensor, seqs in SENSORS.items():
        for seq in seqs:
            out_path = OUT / f"{sensor}_{seq}.npz"
            if out_path.exists():
                print("SKIP (exists)", out_path, flush=True)
                continue
            cpath = CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz"
            if not cpath.exists():
                print("MISS", cpath, flush=True)
                continue
            cache = np.load(cpath)
            desc = cache["descriptors"].astype(np.float32)
            poses = cache["poses"]
            graph = _build_eval_graph(
                keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=desc,
                cache=cache, config=cfg, device=DEVICE, temporal_edge_mode="bidirectional",
                temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
                sensor_key=sensor)
            with torch.no_grad():
                emb = model(graph.to(DEVICE)).detach().cpu().numpy().astype(np.float32)
            np.savez_compressed(out_path, emb416=emb, poses=poses)
            print(f"OK {sensor}/{seq}: emb {emb.shape} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
