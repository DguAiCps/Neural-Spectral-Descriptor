#!/usr/bin/env python3
"""Reconstruct 544D encoder keys and 672D (544D+ctx128) embeddings for the 9 val sequences.

544D: closed-form re-bin of the cached FFT magnitudes with the default (inter-diff)
encoder policy; validated against the cached 288D no_interdiff descriptors.
672D: forward of the surviving 544D-input GNN checkpoint
results/ctx128_cosine_bayesian/best_model.pth (NOT one of the pruned paper seeds).
"""
from __future__ import annotations
import copy, sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes
from run_kitti_operating_point import _make_encoder, _find_queries, _topk_cosine, _score, _normalize

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
CKPT = REPO / "results/ctx128_cosine_bayesian/best_model.pth"
OUT = REPO / "results/_bench544672"
SENSORS = {
    "kitti": ["00", "05", "08"], "nclt": ["2012-01-08", "2013-01-10"],
    "helipr": ["Town01"], "mulran": ["DCC03", "KAIST03", "Riverside03"],
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def descriptors_from_mags(encoder, mags: np.ndarray) -> np.ndarray:
    """Replicates encode_projected_image after the (cached) FFT-magnitude stage."""
    edges = encoder._compute_bin_edges(encoder.alpha)
    out = []
    with torch.no_grad():
        for m in mags:
            hist = encoder._bin_fft_magnitudes(torch.from_numpy(m).float().to(DEVICE), edges)
            out.append(hist.detach().cpu().numpy())
    return np.asarray(out, dtype=np.float32)


def r1(emb, poses):
    normed = _normalize(emb)
    q = _find_queries(poses, 5.0, 30)
    ranked = [_topk_cosine(normed, i, 61, 30)[:1] for i, _ in q]
    return _score(poses, q, ranked, [1], 5.0)["R@1"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml"))
    enc544 = _make_encoder(cfg, DEVICE)
    cfg288 = _apply_encoder_preset(copy.deepcopy(cfg), "no_interdiff")
    enc288 = _make_encoder(cfg288, DEVICE)

    cfg672 = copy.deepcopy(cfg)
    cfg672["gnn"]["use_residual_gate"] = False
    model = _make_model(cfg672, CKPT, DEVICE)

    for sensor, seqs in SENSORS.items():
        for seq in seqs:
            out_path = OUT / f"{sensor}_{seq}.npz"
            if out_path.exists():
                print("SKIP", out_path, flush=True); continue
            cache = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
            mags, poses = cache["fft_magnitudes"], cache["poses"]

            d288 = descriptors_from_mags(enc288, mags)
            err = float(np.abs(d288 - cache["descriptors"]).max())
            d544 = descriptors_from_mags(enc544, mags)
            assert d544.shape[1] == 544, d544.shape

            graph = _build_eval_graph(
                keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d544,
                cache=cache, config=cfg672, device=DEVICE, temporal_edge_mode="bidirectional",
                temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
                sensor_key=sensor)
            with torch.no_grad():
                emb = model(graph.to(DEVICE)).detach().cpu().numpy().astype(np.float32)
            np.savez_compressed(out_path, desc544=d544, emb672=emb, poses=poses)
            print(f"OK {sensor}/{seq}: 288D-recon max|err|={err:.2e}  emb{emb.shape[1]}  "
                  f"R@1 raw544={r1(d544, poses):.3f} final672={r1(emb, poses):.3f}", flush=True)


if __name__ == "__main__":
    main()
