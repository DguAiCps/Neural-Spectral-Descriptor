#!/usr/bin/env python3
"""Synthetic beam-downsampling on KITTI 00 (64 -> 32 -> 16 beams), tab:downsample.

Re-encodes the exact release evaluation keyframes (cache
data/preprocessed_cross_sensor_operating/kitti_operating_00_layout60_stride1.npz)
from the raw velodyne scans with the DEPLOYED encoder (288D no_interdiff key),
after synthetic beam decimation: each point gets an elevation index from 64
percentile bins of atan2(z, sqrt(x^2+y^2)); the 32-beam variant keeps
even-indexed bins, 16-beam keeps every 4th bin.

Per density (64/32/16):
  - 288D key R@1 / R@5 (release protocol: positive <5 m, candidates >=30 frames
    apart, cosine; mirrors scripts/_verify_alias_source_ladder.py)
  - AliasRate per scripts/_aliasrate_544672_recompute.py (eps=0.1, gamma=25 m,
    200k pairs, rng(0), 2D-xy far pairs, percent)
  - 416D retrieval key (frozen seed-1 checkpoint, pattern of
    scripts/_verify_aliasrate_main.py) with the same three metrics.

Validation: the 64-beam re-encode must reproduce the cache descriptors
(cosine ~1.0) and the known 64-beam R@1 (288D 0.921, 416D ~0.945).

Run inside the pyg container:
  docker exec ring_baseline python /workspace/Neural-Spectral-Codec/scripts/_downsample_kitti00_eval.py
Output: results/_alias_source/downsample_kitti00.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from evaluate_kitti_checkpoint import (  # noqa: E402
    _apply_encoder_preset,
    _make_model,
    _build_eval_graph,
    _cache_to_keyframes,
)
from evaluate_multi_sensor_phase import make_encoder  # noqa: E402
from data.kitti_loader import KITTILoader  # noqa: E402

CACHE = REPO / "data/preprocessed_cross_sensor_operating/kitti_operating_00_layout60_stride1.npz"
CKPT = REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth"
KITTI_ROOT = "/data/kitti/dataset"
OUT = REPO / "results/_alias_source/downsample_kitti00.json"
DESC_CACHE = REPO / "results/_alias_source/downsample_kitti00_desc.npz"

DENSITIES = {64: 1, 32: 2, 16: 4}  # beams -> keep every Nth of 64 elevation bins
POS_M, SKIP = 5.0, 30
EPS, GAMMA, NPAIRS = 0.1, 25.0, 200_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def decimate(points: np.ndarray, keep_every: int) -> np.ndarray:
    """Assign each point one of 64 percentile elevation bins, keep every Nth bin."""
    if keep_every == 1:
        return points
    xyz = points[:, :3]
    elev = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))
    edges = np.quantile(elev, np.linspace(0.0, 1.0, 65))
    idx = np.clip(np.searchsorted(edges, elev, side="right") - 1, 0, 63)
    return points[(idx % keep_every) == 0]


def aliasrate(desc, poses, rng):
    """Verbatim scripts/_aliasrate_544672_recompute.py (percent)."""
    d = desc.astype(np.float64)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    xy = poses[:, :2, 3]
    n = d.shape[0]
    i = rng.integers(0, n, NPAIRS)
    j = rng.integers(0, n, NPAIRS)
    keep = i != j
    i, j = i[keep], j[keep]
    geo = np.linalg.norm(xy[i] - xy[j], axis=1)
    dd = np.linalg.norm(d[i] - d[j], axis=1)
    far = geo >= GAMMA
    coll = (dd <= EPS) & far
    return 100.0 * coll.sum() / max(int(far.sum()), 1)


def find_queries(pos):
    q = []
    for j in range(SKIP, len(pos)):
        d = np.linalg.norm(pos[: j - SKIP + 1] - pos[j], axis=1)
        if (d < POS_M).any():
            q.append(j)
    return np.asarray(q, dtype=int)


def recall_1_5(emb, pos, queries):
    """R@1/R@5, candidates = all frames with |i-j| >= SKIP, cosine (unit-norm emb)."""
    sims = emb[queries] @ emb.T
    idx = np.arange(len(emb))
    hit1, hit5 = [], []
    for qi, j in enumerate(queries):
        cand = np.abs(idx - j) >= SKIP
        s = sims[qi, cand]
        geo = np.linalg.norm(pos[cand] - pos[j], axis=1)
        same = geo < POS_M
        top5 = np.argsort(-s)[:5]
        hit1.append(bool(same[top5[0]]))
        hit5.append(bool(same[top5].any()))
    return float(np.mean(hit1)), float(np.mean(hit5))


class _DictCache:
    """npz-like shim so _build_eval_graph/_cache_to_keyframes accept per-density arrays."""

    def __init__(self, d):
        self._d = d
        self.files = list(d.keys())

    def __getitem__(self, k):
        return self._d[k]


OCTAVE = [0, 1, 2, 4, 8, 16, 32, 64, 128, 181]


def octave_desc(fft):
    """(n,16,181) FFT magnitudes -> deployed 288D [mu(144)|sigma(144)] key.

    Exact recon of SpectralEncoder.encode_points with the no_interdiff preset
    (normalize_channels=False); verified cos=1.0 / max abs diff ~6e-5 vs
    encode_points on raw and decimated scans, and vs the release cache.
    """
    n = fft.shape[0]
    mus, sds = [], []
    for b in range(len(OCTAVE) - 1):
        seg = fft[:, :, OCTAVE[b]:OCTAVE[b + 1]]
        mu = seg.mean(axis=2)
        mus.append(mu)
        sds.append(np.sqrt(((seg - mu[:, :, None]) ** 2).mean(axis=2) + 1e-8))
    return np.concatenate(
        [np.stack(mus, 2).reshape(n, -1), np.stack(sds, 2).reshape(n, -1)], axis=1
    ).astype(np.float32)


def encode_all(cache):
    """Re-encode all cached keyframes from raw scans at each density."""
    cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff"
    )
    encoder = make_encoder(cfg, "kitti", DEVICE)
    loader = KITTILoader(KITTI_ROOT, "00", lazy_load=True)
    scan_ids = cache["scan_ids"]
    n = len(scan_ids)
    fft = {b: np.zeros((n, 16, 181), dtype=np.float32) for b in DENSITIES}
    ratios = {b: [] for b in DENSITIES}
    for k, sid in enumerate(scan_ids):
        pts = loader[int(sid)]["points"]
        for beams, keep_every in DENSITIES.items():
            p = decimate(pts, keep_every)
            ratios[beams].append(len(p) / len(pts))
            fft[beams][k] = encoder.compute_fft_magnitudes(p)
        if k % 250 == 0:
            print(f"encode {k}/{n}", flush=True)
    desc = {b: octave_desc(fft[b].astype(np.float64)) for b in DENSITIES}
    ratios = {b: float(np.mean(r)) for b, r in ratios.items()}
    return desc, fft, ratios


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cache = np.load(CACHE)
    poses = cache["poses"]
    pos = poses[:, :3, 3]

    if DESC_CACHE.exists():
        z = np.load(DESC_CACHE)
        desc = {b: z[f"desc{b}"] for b in DENSITIES}
        fft = {b: z[f"fft{b}"] for b in DENSITIES}
        ratios = {b: float(z[f"ratio{b}"]) for b in DENSITIES}
        print(f"[cache hit] {DESC_CACHE.name}")
    else:
        desc, fft, ratios = encode_all(cache)
        np.savez_compressed(
            DESC_CACHE,
            **{f"desc{b}": desc[b] for b in DENSITIES},
            **{f"fft{b}": fft[b] for b in DENSITIES},
            **{f"ratio{b}": ratios[b] for b in DENSITIES},
        )

    # --- validation: 64-beam re-encode must reproduce the cache descriptors ---
    ref = cache["descriptors"].astype(np.float32)
    cos = np.sum(l2(desc[64].astype(np.float64)) * l2(ref.astype(np.float64)), axis=1)
    val = {
        "encode_vs_cache_cos_mean": float(cos.mean()),
        "encode_vs_cache_cos_min": float(cos.min()),
        "point_keep_ratio": ratios,
    }
    print(f"64-beam vs cache: cos mean={cos.mean():.6f} min={cos.min():.6f}")
    print(f"point keep ratios: {ratios}")

    queries = find_queries(pos)
    print(f"n={len(pos)} n_q={len(queries)}")

    # --- 416D model (frozen seed-1), pattern of _verify_aliasrate_main.py ---
    cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff"
    )
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    model = _make_model(cfg, CKPT, DEVICE)

    results = {}
    for beams in DENSITIES:
        d288 = desc[beams].astype(np.float32)
        shim = _DictCache(
            {
                "descriptors": d288,
                "poses": poses,
                "timestamps": cache["timestamps"],
                "scan_ids": cache["scan_ids"],
                "keyframe_ids": cache["keyframe_ids"],
                "fft_magnitudes": fft[beams].astype(np.float32),
            }
        )
        graph = _build_eval_graph(
            keyframes=_cache_to_keyframes(shim),
            poses=poses,
            descriptors=d288,
            cache=shim,
            config=cfg,
            device=DEVICE,
            temporal_edge_mode="bidirectional",
            temporal_direction_mode="none",
            similarity_min_k=0,
            phase_features=None,
            sensor_key="kitti",
        )
        with torch.no_grad():
            emb416 = model(graph.to(DEVICE)).detach().cpu().numpy()

        e288 = l2(d288.astype(np.float64))
        e416 = l2(emb416.astype(np.float64))
        r1_288, r5_288 = recall_1_5(e288, pos, queries)
        r1_416, r5_416 = recall_1_5(e416, pos, queries)
        results[str(beams)] = {
            "key288": {
                "r1": r1_288,
                "r5": r5_288,
                "aliasrate_pct": aliasrate(d288, poses, np.random.default_rng(0)),
            },
            "key416_seed1": {
                "r1": r1_416,
                "r5": r5_416,
                "aliasrate_pct": aliasrate(emb416, poses, np.random.default_rng(0)),
            },
        }
        print(
            f"{beams:>2} beams: 288D R@1={r1_288:.4f} R@5={r5_288:.4f} "
            f"AR={results[str(beams)]['key288']['aliasrate_pct']:.4f}%  | "
            f"416D R@1={r1_416:.4f} R@5={r5_416:.4f} "
            f"AR={results[str(beams)]['key416_seed1']['aliasrate_pct']:.4f}%",
            flush=True,
        )

    val["r1_288_64beam"] = results["64"]["key288"]["r1"]
    val["r1_288_64beam_expected"] = 0.921
    val["r1_288_64beam_ok"] = bool(abs(results["64"]["key288"]["r1"] - 0.921) <= 0.005)
    val["r1_416_64beam"] = results["64"]["key416_seed1"]["r1"]
    val["r1_416_64beam_expected"] = 0.945
    val["encode_ok"] = bool(val["encode_vs_cache_cos_min"] > 0.999)

    out = {
        "sequence": "kitti/00",
        "protocol": {
            "positive_m": POS_M,
            "skip_frames": SKIP,
            "aliasrate": {"eps": EPS, "gamma_m": GAMMA, "n_pairs": NPAIRS, "rng_seed": 0},
            "decimation": "64 percentile elevation bins of atan2(z, sqrt(x^2+y^2)); "
                          "32-beam keeps even bins, 16-beam keeps every 4th",
            "checkpoint_416": str(CKPT.relative_to(REPO)),
        },
        "validation": val,
        "results": results,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"WROTE {OUT}")
    if not (val["encode_ok"] and val["r1_288_64beam_ok"]):
        print("VALIDATION FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
