#!/usr/bin/env python3
"""Run KITTI NSD operating-point experiments without GNN retraining.

This script is designed for the urgent KITTI gap analysis:
  - NSD raw cosine retrieval
  - NSD-native layout/phase column-shift reranking
  - NSD raw/layout score fusion and confidence-gated adaptive fusion
  - SC++ baseline retrieval, for comparison only
  - optional NSD+SC++ hybrid diagnostics, disabled by default

The SC++ rows are never NSD method results. The main NSD-native rows retain NSD
as the compact invariant descriptor, then recover top-N azimuth layout evidence
from NSD's own pre-FFT projection instead of depending on Scan Context matrices.
Do not merge any row from this script into the paper Table 2 NSD+GNN row; use
these outputs only as diagnostic/appendix ablations unless the paper method is
explicitly extended.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from baselines.scan_context import (  # noqa: E402
    _ring_key,
    build_scan_context,
)
from data.kitti_loader import KITTILoader  # noqa: E402
from encoding.bev_image import interpolate_bev_image  # noqa: E402
from encoding.range_image import interpolate_range_image  # noqa: E402
from encoding.spectral_encoder import SpectralEncoder  # noqa: E402
from keyframe.selector import KeyframeSelector  # noqa: E402
from utils.cyclic_shift_distance import cyclic_column_cosine_distance  # noqa: E402


def _load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _make_encoder(config: Dict, device: str) -> SpectralEncoder:
    enc = config["encoding"]
    bev = enc.get("bev", {})
    return SpectralEncoder(
        n_elevation=enc.get("n_elevation", 16),
        n_azimuth=enc.get("n_azimuth", 360),
        n_bins=enc.get("n_bins", 16),
        alpha=enc.get("alpha", 2.0),
        learnable_alpha=enc.get("learnable_alpha", False),
        epsilon=enc.get("epsilon", 1e-8),
        target_elevation_bins=enc.get("target_elevation_bins", 16),
        elevation_range=tuple(enc.get("elevation_range", [-24.8, 2.0])),
        bin_statistics=enc.get("bin_statistics", ["mean", "std"]),
        inter_bin_statistics=enc.get("inter_bin_statistics", ["diff"]),
        device=device,
        projection_type=enc.get("projection_type", "range_image"),
        max_range=enc.get("max_range", 80.0),
        min_range=enc.get("min_range", 1.0),
        z_min=bev.get("z_min", -3.0),
        height_encoding=bev.get("height_encoding", "iris"),
        n_height_layers=bev.get("n_height_layers", 8),
        z_max=bev.get("z_max", 5.0),
        zero_center=enc.get("zero_center", False),
        log_magnitude=enc.get("log_magnitude", False),
        binning_strategy=enc.get("binning_strategy", "octave"),
        normalize_channels=enc.get("normalize_channels", False),
        cross_spectrum_enabled=enc.get("cross_spectrum", {}).get("enabled", False),
        cross_spectrum_n_freqs=enc.get("cross_spectrum", {}).get("n_freqs", 0),
    ).to(device)


def _pool_columns(image: np.ndarray, n_cols: int) -> np.ndarray:
    """Average-pool azimuth columns to a fixed sector count for fast reranking."""
    if image.shape[1] == n_cols:
        return image.astype(np.float32)
    image_tensor = torch.from_numpy(image).float()
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        image_tensor.unsqueeze(0).unsqueeze(0),
        (image.shape[0], n_cols),
    ).squeeze()
    return pooled.detach().cpu().numpy().astype(np.float32)


def _project_nsd_layout(
    encoder: SpectralEncoder,
    points: np.ndarray,
    n_layout_sectors: int = 60,
) -> np.ndarray:
    """Project points with NSD's own projector before FFT magnitude compression.

    This preserves azimuth layout/phase information that the compact NSD
    descriptor intentionally discards. It is used only as a test-time reranking
    signal so we can replace SC++ reranking with an NSD-native layout path.
    """
    image_2d, _ = encoder.projector.project(points, keep_intensity=False)
    if encoder.interpolate_empty:
        if encoder.projection_type == "bev":
            image_2d = interpolate_bev_image(image_2d, method="linear")
        else:
            image_2d = interpolate_range_image(image_2d, method="linear")

    if encoder.projection_type != "bev" and image_2d.shape[0] != encoder.target_elevation_bins:
        image_tensor = torch.from_numpy(image_2d).float().to(encoder.alpha.device)
        image_tensor = torch.nn.functional.adaptive_avg_pool2d(
            image_tensor.unsqueeze(0).unsqueeze(0),
            (encoder.target_elevation_bins, image_tensor.shape[1]),
        ).squeeze()
        image_2d = image_tensor.detach().cpu().numpy()

    return _pool_columns(image_2d.astype(np.float32), n_layout_sectors)


def _build_sequence_cache(
    root: Path,
    sequence: str,
    config: Dict,
    cache_dir: Path,
    device: str,
    max_scans: int | None = None,
    layout_sectors: int = 60,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{sequence}_{max_scans}" if max_scans else sequence
    cache_path = cache_dir / f"kitti_operating_{suffix}_layout{layout_sectors}.npz"
    if cache_path.exists():
        return cache_path

    loader = KITTILoader(root, sequence, lazy_load=True)
    encoder = _make_encoder(config, device)
    key_cfg = config["keyframe"]
    selector = KeyframeSelector(
        distance_threshold=key_cfg.get("distance_threshold", 0.8),
        rotation_threshold=key_cfg.get("rotation_threshold", 20.0),
        overlap_threshold=key_cfg.get("overlap_threshold", 0.65),
        temporal_threshold=key_cfg.get("temporal_threshold", 30.0),
        voxel_size=key_cfg.get("voxel_size", 0.2),
        max_keyframes=key_cfg.get("max_keyframes", 10000000),
    )

    n_scans = len(loader) if max_scans is None else min(len(loader), max_scans)
    descriptors, poses, timestamps, scan_ids, keyframe_ids = [], [], [], [], []
    sc_matrices, ring_keys, fft_magnitudes, nsd_layouts = [], [], [], []

    for scan_id in range(n_scans):
        if scan_id % 250 == 0:
            print(f"[{sequence}] scan {scan_id}/{n_scans}, keyframes={len(scan_ids)}", flush=True)
        item = loader[scan_id]
        selected, keyframe, _ = selector.process_scan(
            scan_id=scan_id,
            points=item["points"],
            pose=item["pose"],
            timestamp=item["timestamp"],
        )
        if not selected:
            continue

        desc = encoder.encode_points(item["points"]).detach().cpu().numpy().astype(np.float32)
        fft = encoder.compute_fft_magnitudes(item["points"]).astype(np.float32)
        nsd_layout = _project_nsd_layout(
            encoder,
            item["points"],
            n_layout_sectors=layout_sectors,
        )
        sc = build_scan_context(item["points"], n_rings=20, n_sectors=60, max_range=80.0)
        rk = _ring_key(sc).astype(np.float32)
        rk_norm = np.linalg.norm(rk)
        if rk_norm > 1e-8:
            rk = rk / rk_norm

        descriptors.append(desc)
        poses.append(keyframe.pose.astype(np.float64))
        timestamps.append(float(keyframe.timestamp))
        scan_ids.append(int(scan_id))
        keyframe_ids.append(int(keyframe.keyframe_id))
        sc_matrices.append(sc.astype(np.float32))
        ring_keys.append(rk.astype(np.float32))
        fft_magnitudes.append(fft)
        nsd_layouts.append(nsd_layout)

    np.savez_compressed(
        cache_path,
        descriptors=np.asarray(descriptors, dtype=np.float32),
        poses=np.asarray(poses, dtype=np.float64),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        scan_ids=np.asarray(scan_ids, dtype=np.int64),
        keyframe_ids=np.asarray(keyframe_ids, dtype=np.int64),
        sc_matrices=np.asarray(sc_matrices, dtype=np.float32),
        ring_keys=np.asarray(ring_keys, dtype=np.float32),
        fft_magnitudes=np.asarray(fft_magnitudes, dtype=np.float32),
        nsd_layouts=np.asarray(nsd_layouts, dtype=np.float32),
    )
    return cache_path


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32).copy()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def _topk_cosine(descs: np.ndarray, query_idx: int, search_k: int, skip_frames: int) -> np.ndarray:
    sims = descs @ descs[query_idx]
    order = np.argsort(-sims)
    valid = order[np.abs(order - query_idx) >= skip_frames]
    valid = valid[valid != query_idx]
    return valid[:search_k]


def _find_queries(poses: np.ndarray, distance_threshold: float, skip_frames: int) -> List[Tuple[int, int]]:
    positions = poses[:, :3, 3]
    queries = []
    for j in range(skip_frames, len(poses)):
        prior = np.arange(0, j - skip_frames + 1)
        if len(prior) == 0:
            continue
        dists = np.linalg.norm(positions[prior] - positions[j], axis=1)
        hits = np.where(dists < distance_threshold)[0]
        if len(hits) > 0:
            queries.append((j, int(prior[hits[0]])))
    return queries


def _score(
    poses: np.ndarray,
    queries: List[Tuple[int, int]],
    ranked_lists: Iterable[np.ndarray],
    k_values: List[int],
    distance_threshold: float,
) -> Dict[str, float]:
    positions = poses[:, :3, 3]
    correct = {k: 0 for k in k_values}
    for (query_idx, _), ranked in zip(queries, ranked_lists):
        if len(ranked) == 0:
            continue
        dists = np.linalg.norm(positions[ranked[:max(k_values)]] - positions[query_idx], axis=1)
        for k in k_values:
            if np.any(dists[:k] < distance_threshold):
                correct[k] += 1
    denom = max(len(queries), 1)
    return {f"R@{k}": correct[k] / denom for k in k_values}


def _distance_columnwise(mat1: np.ndarray, mat2: np.ndarray) -> float:
    """Column-shift cosine distance for NSD layout images.

    This uses the shared cyclic shift primitive and operates on NSD's own
    projected range/BEV image rather than Scan Context matrices.
    """
    return cyclic_column_cosine_distance(mat1, mat2)


def _rank_fusion(raw_candidates: np.ndarray, reranked: np.ndarray, raw_weight: float) -> np.ndarray:
    """Fuse raw NSD rank with a reranker rank without depending on score scale."""
    if len(raw_candidates) == 0:
        return raw_candidates
    raw_rank = {int(c): i for i, c in enumerate(raw_candidates)}
    rerank_rank = {int(c): i for i, c in enumerate(reranked)}
    fused = sorted(
        raw_candidates.tolist(),
        key=lambda c: rerank_rank[int(c)] + raw_weight * raw_rank[int(c)],
    )
    return np.asarray(fused, dtype=raw_candidates.dtype)


def _minmax01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def _score_fusion(
    candidates: np.ndarray,
    raw_distances: np.ndarray,
    layout_distances: np.ndarray,
    layout_weight: float,
) -> np.ndarray:
    """Fuse raw descriptor and layout distances after per-query normalization.

    Raw NSD cosine keeps first-rank discriminability, while the layout distance
    recovers the azimuth phase/layout evidence discarded by FFT magnitudes.
    """
    if len(candidates) == 0:
        return candidates
    raw_norm = _minmax01(raw_distances)
    layout_norm = _minmax01(layout_distances)
    fused_scores = raw_norm + layout_weight * layout_norm
    return candidates[np.argsort(fused_scores)]


def _layout_row_keys(nsd_layouts: np.ndarray) -> np.ndarray:
    """Rotation-invariant coarse keys from NSD layout rows.

    This mirrors SC++'s ring-key idea but uses NSD's own range/elevation layout:
    average over azimuth sectors, then cosine-normalize for coarse candidate
    retrieval before column-shift layout reranking.
    """
    if nsd_layouts.ndim != 3:
        raise ValueError(f"Expected layout tensor (N, rows, cols), got {nsd_layouts.shape}")
    return _normalize(nsd_layouts.mean(axis=2))


def _adaptive_score_fusion(
    candidates: np.ndarray,
    raw_distances: np.ndarray,
    layout_distances: np.ndarray,
    low_weight: float,
    high_weight: float,
    gap_scale: float,
) -> np.ndarray:
    """Confidence-gated fusion using only test-time descriptor score margins."""
    if len(candidates) == 0:
        return candidates
    raw_norm = _minmax01(raw_distances)
    if len(raw_norm) < 2 or gap_scale <= 1e-8:
        layout_weight = high_weight
    else:
        sorted_raw = np.sort(raw_norm)
        margin = float(sorted_raw[1] - sorted_raw[0])
        ambiguity = np.clip(1.0 - margin / gap_scale, 0.0, 1.0)
        layout_weight = low_weight + (high_weight - low_weight) * ambiguity
    return _score_fusion(candidates, raw_distances, layout_distances, layout_weight)


def _evaluate_sequence(
    cache_path: Path,
    n_coarse: int,
    k_values: List[int],
    distance_threshold: float,
    skip_frames: int,
    layout_fusion_weights: List[float],
    layout_score_weights: List[float],
    adaptive_gap_scales: List[float],
    adaptive_low_weight: float,
    adaptive_high_weight: float,
    include_scpp_hybrid: bool,
) -> Dict:
    data = np.load(cache_path)
    descs = _normalize(data["descriptors"])
    ring_keys = _normalize(data["ring_keys"])
    sc_matrices = data["sc_matrices"]
    nsd_layouts = data["nsd_layouts"]
    layout_keys = _layout_row_keys(nsd_layouts)
    poses = data["poses"]
    queries = _find_queries(poses, distance_threshold, skip_frames)

    raw_ranked, nsd_layout_ranked, sc_ranked = [], [], []
    nsd_scpp_ranked, hybrid_ranked = [], []
    nsd_layout_key_ranked, nsd_layout_union_ranked = [], []
    nsd_layout_fused_ranked = {w: [] for w in layout_fusion_weights}
    nsd_layout_score_fused_ranked = {w: [] for w in layout_score_weights}
    nsd_layout_union_score_fused_ranked = {w: [] for w in layout_score_weights}
    nsd_layout_adaptive_ranked = {g: [] for g in adaptive_gap_scales}
    for query_idx, _ in queries:
        raw_candidates = _topk_cosine(descs, query_idx, n_coarse, skip_frames)
        sc_candidates = _topk_cosine(ring_keys, query_idx, n_coarse, skip_frames)
        layout_candidates = _topk_cosine(layout_keys, query_idx, n_coarse, skip_frames)
        layout_union_candidates = np.unique(np.concatenate([raw_candidates, layout_candidates]))
        raw_distances = 1.0 - (descs[raw_candidates] @ descs[query_idx])

        raw_ranked.append(raw_candidates[:max(k_values)])

        def rerank_scpp(candidates: np.ndarray) -> np.ndarray:
            if len(candidates) == 0:
                return candidates
            dists = np.asarray([
                cyclic_column_cosine_distance(sc_matrices[query_idx], sc_matrices[int(c)])
                for c in candidates
            ], dtype=np.float32)
            return candidates[np.argsort(dists)]

        sc_ranked.append(rerank_scpp(sc_candidates)[:max(k_values)])
        if include_scpp_hybrid:
            nsd_scpp_ranked.append(rerank_scpp(raw_candidates)[:max(k_values)])
        layout_distances = np.asarray([
            _distance_columnwise(nsd_layouts[query_idx], nsd_layouts[int(c)])
            for c in raw_candidates
        ], dtype=np.float32)
        layout_ranked = raw_candidates[np.argsort(layout_distances)]
        nsd_layout_ranked.append(layout_ranked[:max(k_values)])
        layout_key_distances = np.asarray([
            _distance_columnwise(nsd_layouts[query_idx], nsd_layouts[int(c)])
            for c in layout_candidates
        ], dtype=np.float32)
        nsd_layout_key_ranked.append(layout_candidates[np.argsort(layout_key_distances)][:max(k_values)])
        layout_union_distances = np.asarray([
            _distance_columnwise(nsd_layouts[query_idx], nsd_layouts[int(c)])
            for c in layout_union_candidates
        ], dtype=np.float32)
        nsd_layout_union_ranked.append(
            layout_union_candidates[np.argsort(layout_union_distances)][:max(k_values)]
        )
        for weight in layout_fusion_weights:
            fused = _rank_fusion(raw_candidates, layout_ranked, raw_weight=weight)
            nsd_layout_fused_ranked[weight].append(fused[:max(k_values)])
        for weight in layout_score_weights:
            fused = _score_fusion(raw_candidates, raw_distances, layout_distances, layout_weight=weight)
            nsd_layout_score_fused_ranked[weight].append(fused[:max(k_values)])
            union_raw_distances = 1.0 - (descs[layout_union_candidates] @ descs[query_idx])
            union_fused = _score_fusion(
                layout_union_candidates,
                union_raw_distances,
                layout_union_distances,
                layout_weight=weight,
            )
            nsd_layout_union_score_fused_ranked[weight].append(union_fused[:max(k_values)])
        for gap_scale in adaptive_gap_scales:
            fused = _adaptive_score_fusion(
                raw_candidates,
                raw_distances,
                layout_distances,
                low_weight=adaptive_low_weight,
                high_weight=adaptive_high_weight,
                gap_scale=gap_scale,
            )
            nsd_layout_adaptive_ranked[gap_scale].append(fused[:max(k_values)])
        if include_scpp_hybrid:
            union = np.unique(np.concatenate([raw_candidates, sc_candidates]))
            hybrid_ranked.append(rerank_scpp(union)[:max(k_values)])

    results = {
        "n_keyframes": int(len(poses)),
        "n_queries": int(len(queries)),
        "nsd_raw": _score(poses, queries, raw_ranked, k_values, distance_threshold),
        "nsd_coarse_layout_rerank": _score(
            poses, queries, nsd_layout_ranked, k_values, distance_threshold
        ),
        "nsd_layout_key_rerank": _score(
            poses, queries, nsd_layout_key_ranked, k_values, distance_threshold
        ),
        "nsd_raw_layoutkey_union_rerank": _score(
            poses, queries, nsd_layout_union_ranked, k_values, distance_threshold
        ),
        "scpp": _score(poses, queries, sc_ranked, k_values, distance_threshold),
    }
    if include_scpp_hybrid:
        results["diagnostic_nsd_coarse_scpp_rerank"] = _score(
            poses, queries, nsd_scpp_ranked, k_values, distance_threshold
        )
        results["diagnostic_nsd_scpp_union_rerank"] = _score(
            poses, queries, hybrid_ranked, k_values, distance_threshold
        )
    for weight, ranked_lists in nsd_layout_fused_ranked.items():
        key = f"nsd_coarse_layout_rank_fusion_w{weight:g}"
        results[key] = _score(poses, queries, ranked_lists, k_values, distance_threshold)
    for weight, ranked_lists in nsd_layout_score_fused_ranked.items():
        key = f"nsd_coarse_layout_score_fusion_w{weight:g}"
        results[key] = _score(poses, queries, ranked_lists, k_values, distance_threshold)
    for weight, ranked_lists in nsd_layout_union_score_fused_ranked.items():
        key = f"nsd_raw_layoutkey_union_score_fusion_w{weight:g}"
        results[key] = _score(poses, queries, ranked_lists, k_values, distance_threshold)
    for gap_scale, ranked_lists in nsd_layout_adaptive_ranked.items():
        key = (
            "nsd_coarse_layout_adaptive_score_"
            f"lo{adaptive_low_weight:g}_hi{adaptive_high_weight:g}_gap{gap_scale:g}"
        )
        results[key] = _score(poses, queries, ranked_lists, k_values, distance_threshold)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_kitti_only.yaml")
    parser.add_argument("--root", default=None, help="KITTI dataset root containing poses/ and sequences/")
    parser.add_argument("--sequences", nargs="+", default=["00", "05", "08"])
    parser.add_argument("--cache-dir", default="data/preprocessed_kitti_operating")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-coarse", type=int, default=200)
    parser.add_argument("--skip-frames", type=int, default=30)
    parser.add_argument("--distance-threshold", type=float, default=5.0)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--layout-sectors", type=int, default=60)
    parser.add_argument(
        "--layout-fusion-weights",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    )
    parser.add_argument(
        "--layout-score-weights",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 3.2, 4.0],
    )
    parser.add_argument(
        "--adaptive-gap-scales",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2, 0.4],
    )
    parser.add_argument(
        "--adaptive-low-weight",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--adaptive-high-weight",
        type=float,
        default=3.2,
    )
    parser.add_argument(
        "--include-scpp-hybrid",
        action="store_true",
        help="Report diagnostic NSD+SC++ hybrid rows. Disabled by default.",
    )
    parser.add_argument("--output", default="results/kitti_operating_point.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config(config_path)
    if args.root is None:
        root = Path(config["data"]["datasets"]["val"][0]["root"])
    else:
        root = Path(args.root)

    results = {}
    for seq in args.sequences:
        cache_path = _build_sequence_cache(
            root=root,
            sequence=seq,
            config=config,
            cache_dir=Path(args.cache_dir),
            device=args.device,
            max_scans=args.max_scans,
            layout_sectors=args.layout_sectors,
        )
        results[seq] = _evaluate_sequence(
            cache_path=cache_path,
            n_coarse=args.n_coarse,
            k_values=args.k_values,
            distance_threshold=args.distance_threshold,
            skip_frames=args.skip_frames,
            layout_fusion_weights=args.layout_fusion_weights,
            layout_score_weights=args.layout_score_weights,
            adaptive_gap_scales=args.adaptive_gap_scales,
            adaptive_low_weight=args.adaptive_low_weight,
            adaptive_high_weight=args.adaptive_high_weight,
            include_scpp_hybrid=args.include_scpp_hybrid,
        )
        print(seq, json.dumps(results[seq], indent=2), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
