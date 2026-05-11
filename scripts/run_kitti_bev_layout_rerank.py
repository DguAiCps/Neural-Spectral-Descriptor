#!/usr/bin/env python3
"""Evaluate an NSD-owned BEV layout reranker on KITTI caches.

This is a diagnostic for KITTI 08: range/elevation layout reranking recovers
some phase information, but SC++ remains much stronger because it reranks in a
BEV polar layout. This script uses NSD's own BEVProjector, not SC++'s matrix
builder, to test whether an auxiliary BEV layout should become an NSD rerank
head.

These outputs are not paper Table 2 NSD+GNN results. They belong only in an
explicit diagnostic/appendix ablation unless the paper architecture is extended
to include this BEV layout rerank head and its extra storage/latency cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from encoding.bev_image import BEVProjector, interpolate_bev_image  # noqa: E402
from run_kitti_operating_point import (  # noqa: E402
    _distance_columnwise,
    _find_queries,
    _layout_row_keys,
    _minmax01,
    _normalize,
    _score,
    _score_fusion,
    _topk_cosine,
)


def _load_points(path: Path) -> np.ndarray:
    points = np.fromfile(path, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"Invalid KITTI scan: {path}")
    return points.reshape(-1, 4)


def _build_bev_layout_cache(
    root: Path,
    sequence: str,
    base_cache: np.lib.npyio.NpzFile,
    output_path: Path,
    n_sectors: int,
    max_range: float,
    min_range: float,
    z_min: float,
    z_max: float,
    n_height_layers: int,
    height_encoding: str,
) -> np.ndarray:
    if output_path.exists():
        return np.load(output_path)["bev_layouts"].astype(np.float32)

    projector = BEVProjector(
        n_sectors=n_sectors,
        max_range=max_range,
        min_range=min_range,
        z_min=z_min,
        height_encoding=height_encoding,
        n_height_layers=n_height_layers,
        z_max=z_max,
    )
    layouts = []
    scan_ids = base_cache["scan_ids"].astype(np.int64)
    velodyne_dir = root / "sequences" / sequence / "velodyne"
    for i, scan_id in enumerate(scan_ids):
        if i % 250 == 0:
            print(f"[{sequence}] BEV layout {i}/{len(scan_ids)}", flush=True)
        points = _load_points(velodyne_dir / f"{int(scan_id):06d}.bin")
        bev, _ = projector.project(points, keep_intensity=False)
        bev = interpolate_bev_image(
            bev,
            method="linear",
            n_channels=3 if height_encoding == "physics3" else 1,
        )
        layouts.append(bev.astype(np.float32))

    bev_layouts = np.asarray(layouts, dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, bev_layouts=bev_layouts)
    return bev_layouts


def _pool_rows(
    layouts: np.ndarray,
    n_rows: int,
    mode: str,
    n_channels: int = 1,
) -> np.ndarray:
    if n_rows <= 0 or layouts.shape[1] == n_rows:
        return layouts.astype(np.float32)
    n_channels = max(1, int(n_channels))
    if layouts.shape[1] % n_channels != 0 or n_rows % n_channels != 0:
        raise ValueError(
            f"Cannot channel-pool rows={layouts.shape[1]} to n_rows={n_rows} "
            f"with n_channels={n_channels}"
        )
    rows_per_channel = layouts.shape[1] // n_channels
    out_rows_per_channel = n_rows // n_channels
    pooled = []
    for ch in range(n_channels):
        start = ch * rows_per_channel
        end = start + rows_per_channel
        groups = np.array_split(np.arange(start, end), out_rows_per_channel)
        for group in groups:
            chunk = layouts[:, group, :]
            if mode == "max":
                pooled.append(chunk.max(axis=1))
            elif mode == "mean":
                pooled.append(chunk.mean(axis=1))
            else:
                raise ValueError(f"Unknown row pool mode: {mode}")
    return np.stack(pooled, axis=1).astype(np.float32)


def _fused_order(
    candidates: np.ndarray,
    raw_distances: np.ndarray,
    layout_distances: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    score = _minmax01(raw_distances)
    for name, distance in layout_distances.items():
        score = score + weights.get(name, 0.0) * _minmax01(distance)
    return candidates[np.argsort(score)]


def _tag_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def _evaluate(
    base_cache: np.lib.npyio.NpzFile,
    bev_layouts: np.ndarray,
    n_coarse: int,
    k_values: List[int],
    distance_threshold: float,
    skip_frames: int,
    score_weights: List[float],
    dual_bev_weights: List[float],
    dual_range_weights: List[float],
) -> Dict:
    poses = base_cache["poses"]
    descs = _normalize(base_cache["descriptors"])
    scpp_keys = _normalize(base_cache["ring_keys"])
    scpp_mats = base_cache["sc_matrices"]
    range_layouts = base_cache["nsd_layouts"].astype(np.float32)
    bev_keys = _layout_row_keys(bev_layouts)
    range_keys = _layout_row_keys(range_layouts)
    queries = _find_queries(poses, distance_threshold, skip_frames)

    raw_bev_ranked, bev_key_ranked, raw_bev_union_ranked = [], [], []
    score_ranked = {w: [] for w in score_weights}
    dual_ranked = {(wb, wr): [] for wb in dual_bev_weights for wr in dual_range_weights}
    for query_idx, _ in queries:
        raw_candidates = _topk_cosine(descs, query_idx, n_coarse, skip_frames)
        bev_candidates = _topk_cosine(bev_keys, query_idx, n_coarse, skip_frames)
        range_candidates = _topk_cosine(range_keys, query_idx, n_coarse, skip_frames)
        union_candidates = np.unique(np.concatenate([
            raw_candidates,
            bev_candidates,
            range_candidates,
        ]))

        def rerank_bev(candidates: np.ndarray) -> np.ndarray:
            distances = np.asarray([
                _distance_columnwise(bev_layouts[query_idx], bev_layouts[int(c)])
                for c in candidates
            ], dtype=np.float32)
            return candidates[np.argsort(distances)]

        raw_bev_ranked.append(rerank_bev(raw_candidates)[:max(k_values)])
        bev_key_ranked.append(rerank_bev(bev_candidates)[:max(k_values)])
        raw_bev_union_ranked.append(rerank_bev(union_candidates)[:max(k_values)])

        union_layout_distances = np.asarray([
            _distance_columnwise(bev_layouts[query_idx], bev_layouts[int(c)])
            for c in union_candidates
        ], dtype=np.float32)
        union_range_distances = np.asarray([
            _distance_columnwise(range_layouts[query_idx], range_layouts[int(c)])
            for c in union_candidates
        ], dtype=np.float32)
        union_raw_distances = 1.0 - (descs[union_candidates] @ descs[query_idx])
        for weight in score_weights:
            fused = _score_fusion(
                union_candidates,
                union_raw_distances,
                union_layout_distances,
                layout_weight=weight,
            )
            score_ranked[weight].append(fused[:max(k_values)])
        for bev_weight in dual_bev_weights:
            for range_weight in dual_range_weights:
                fused = _fused_order(
                    union_candidates,
                    union_raw_distances,
                    {"bev": union_layout_distances, "range": union_range_distances},
                    {"bev": bev_weight, "range": range_weight},
                )
                dual_ranked[(bev_weight, range_weight)].append(fused[:max(k_values)])

    scpp_ranked = []
    for query_idx, _ in queries:
        scpp_candidates = _topk_cosine(scpp_keys, query_idx, n_coarse, skip_frames)
        distances = np.asarray([
            _distance_columnwise(scpp_mats[query_idx], scpp_mats[int(c)])
            for c in scpp_candidates
        ], dtype=np.float32)
        scpp_ranked.append(scpp_candidates[np.argsort(distances)][:max(k_values)])

    results = {
        "n_keyframes": int(len(poses)),
        "n_queries": int(len(queries)),
        "nsd_raw_bev_rerank": _score(
            poses, queries, raw_bev_ranked, k_values, distance_threshold
        ),
        "nsd_bev_key_rerank": _score(
            poses, queries, bev_key_ranked, k_values, distance_threshold
        ),
        "nsd_raw_bevkey_union_rerank": _score(
            poses, queries, raw_bev_union_ranked, k_values, distance_threshold
        ),
        "scpp": _score(poses, queries, scpp_ranked, k_values, distance_threshold),
    }
    for weight, ranked in score_ranked.items():
        results[f"nsd_raw_bevkey_union_score_fusion_w{weight:g}"] = _score(
            poses, queries, ranked, k_values, distance_threshold
        )
    for (bev_weight, range_weight), ranked in dual_ranked.items():
        results[
            f"nsd_dual_layout_score_fusion_bev{bev_weight:g}_range{range_weight:g}"
        ] = _score(poses, queries, ranked, k_values, distance_threshold)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--sequences", nargs="+", default=["08"])
    parser.add_argument("--base-cache-dir", default="data/preprocessed_kitti_full_layout")
    parser.add_argument("--bev-cache-dir", default="data/preprocessed_kitti_bev_layout")
    parser.add_argument("--layout-sectors", type=int, default=60)
    parser.add_argument("--n-coarse", type=int, default=200)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--distance-threshold", type=float, default=5.0)
    parser.add_argument("--skip-frames", type=int, default=30)
    parser.add_argument("--score-weights", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--dual-bev-weights", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--dual-range-weights", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--max-range", type=float, default=80.0)
    parser.add_argument("--min-range", type=float, default=1.0)
    parser.add_argument("--z-min", type=float, default=-3.0)
    parser.add_argument("--z-max", type=float, default=5.0)
    parser.add_argument("--n-height-layers", type=int, default=8)
    parser.add_argument("--height-encoding", default="iris", choices=["iris", "max", "physics3"])
    parser.add_argument("--row-pool", type=int, default=0)
    parser.add_argument("--row-pool-mode", default="max", choices=["max", "mean"])
    parser.add_argument("--output", default="results/kitti_bev_layout_rerank.json")
    args = parser.parse_args()

    root = Path(args.root)
    results = {}
    for seq in args.sequences:
        base_path = Path(args.base_cache_dir) / f"kitti_operating_{seq}_layout{args.layout_sectors}.npz"
        base_cache = np.load(base_path)
        cache_tag = (
            f"s{args.layout_sectors}_{args.height_encoding}"
            f"_r{_tag_float(args.min_range)}-{_tag_float(args.max_range)}"
            f"_z{_tag_float(args.z_min)}-{_tag_float(args.z_max)}"
            f"_h{args.n_height_layers}"
        )
        bev_path = Path(args.bev_cache_dir) / (
            f"kitti_bev_layout_{seq}_{cache_tag}.npz"
        )
        bev_layouts = _build_bev_layout_cache(
            root=root,
            sequence=seq,
            base_cache=base_cache,
            output_path=bev_path,
            n_sectors=args.layout_sectors,
            max_range=args.max_range,
            min_range=args.min_range,
            z_min=args.z_min,
            z_max=args.z_max,
            n_height_layers=args.n_height_layers,
            height_encoding=args.height_encoding,
        )
        bev_layouts = _pool_rows(
            bev_layouts,
            args.row_pool,
            args.row_pool_mode,
            n_channels=3 if args.height_encoding == "physics3" else 1,
        )
        results[seq] = _evaluate(
            base_cache=base_cache,
            bev_layouts=bev_layouts,
            n_coarse=args.n_coarse,
            k_values=args.k_values,
            distance_threshold=args.distance_threshold,
            skip_frames=args.skip_frames,
            score_weights=args.score_weights,
            dual_bev_weights=args.dual_bev_weights,
            dual_range_weights=args.dual_range_weights,
        )
        print(seq, json.dumps(results[seq], indent=2), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
