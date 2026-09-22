#!/usr/bin/env python3
"""Fast KITTI-360 validation for NSD checkpoint/phase-sketch diagnostics.

Mirrors evaluate_mulran_checkpoint.py: builds an operating cache of encoder
descriptors + NSD layouts per drive, then runs evaluate_cache from
evaluate_kitti_checkpoint.py with the standard KITTI protocol (HDL-64E,
sensor key 'kitti', 5 m threshold, skip_frames 30).

Poses come from each drive's poses.txt (IMU/GNSS world pose), linearly
interpolated per velodyne frame — see data.kitti360_loader for the frame
convention and the lever-arm note.

--dry-run needs no torch/checkpoint: it loads a few scans per sequence,
verifies bin shapes, and prints pose-coverage stats.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from data.kitti360_loader import Kitti360Loader  # noqa: E402


def _short_id(sequence: str) -> str:
    """'2013_05_28_drive_0002_sync' -> '0002'; '0002' stays '0002'."""
    seq = str(sequence)
    if seq.startswith("2013_05_28_drive"):
        return seq.split("_")[4]
    return f"{int(seq):04d}"


def _build_kitti360_cache(
    root: Path,
    sequence: str,
    config: Dict,
    cache_dir: Path,
    device: str,
    layout_sectors: int,
    scan_stride: int,
) -> Path:
    from keyframe.selector import KeyframeSelector
    from run_kitti_operating_point import _make_encoder, _project_nsd_layout

    seq_id = _short_id(sequence)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"kitti360_operating_{seq_id}_layout{layout_sectors}_stride{scan_stride}.npz"
    if cache_path.exists():
        return cache_path

    loader = Kitti360Loader(root, sequence, lazy_load=True)
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

    descriptors, poses, timestamps, scan_ids, keyframe_ids = [], [], [], [], []
    fft_magnitudes, nsd_layouts = [], []
    for scan_id in range(0, len(loader), scan_stride):
        if scan_id % 250 == 0:
            print(f"[{seq_id}] scan {scan_id}/{len(loader)}, keyframes={len(scan_ids)}", flush=True)
        item = loader[scan_id]
        selected, keyframe, _ = selector.process_scan(
            scan_id=scan_id,
            points=item["points"],
            pose=item["pose"],
            timestamp=item["timestamp"],
        )
        if not selected:
            continue
        descriptors.append(encoder.encode_points(item["points"]).detach().cpu().numpy().astype(np.float32))
        fft_magnitudes.append(encoder.compute_fft_magnitudes(item["points"]).astype(np.float32))
        nsd_layouts.append(_project_nsd_layout(encoder, item["points"], n_layout_sectors=layout_sectors))
        poses.append(keyframe.pose.astype(np.float64))
        timestamps.append(float(keyframe.timestamp))
        scan_ids.append(int(scan_id))
        keyframe_ids.append(int(keyframe.keyframe_id))

    np.savez_compressed(
        cache_path,
        descriptors=np.asarray(descriptors, dtype=np.float32),
        poses=np.asarray(poses, dtype=np.float64),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        scan_ids=np.asarray(scan_ids, dtype=np.int64),
        keyframe_ids=np.asarray(keyframe_ids, dtype=np.int64),
        fft_magnitudes=np.asarray(fft_magnitudes, dtype=np.float32),
        nsd_layouts=np.asarray(nsd_layouts, dtype=np.float32),
    )
    return cache_path


def _build_kitti360_bev_cache(
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
    from encoding.bev_image import BEVProjector, interpolate_bev_image

    if output_path.exists():
        return np.load(output_path)["bev_layouts"].astype(np.float32)

    loader = Kitti360Loader(root, sequence, lazy_load=True)
    projector = BEVProjector(
        n_sectors=n_sectors,
        max_range=max_range,
        min_range=min_range,
        z_min=z_min,
        height_encoding=height_encoding,
        n_height_layers=n_height_layers,
        z_max=z_max,
    )
    seq_id = _short_id(sequence)
    layouts = []
    for i, scan_id in enumerate(base_cache["scan_ids"].astype(np.int64)):
        if i % 250 == 0:
            print(f"[{seq_id}] BEV layout {i}/{len(base_cache['scan_ids'])}", flush=True)
        points = loader[int(scan_id)]["points"]
        bev, _ = projector.project(points, keep_intensity=False)
        layouts.append(
            interpolate_bev_image(
                bev,
                method="linear",
                n_channels=3 if height_encoding == "physics3" else 1,
            ).astype(np.float32)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bev_layouts = np.asarray(layouts, dtype=np.float32)
    np.savez_compressed(output_path, bev_layouts=bev_layouts)
    return bev_layouts


def _dry_run(root: Path, sequences: list, n_scans: int) -> None:
    """CPU-only sanity path: bin shapes + pose-interpolation coverage."""
    summary = {}
    for sequence in sequences:
        loader = Kitti360Loader(root, sequence, lazy_load=True)
        cov = dict(loader.coverage)
        cov["n_loader_scans"] = len(loader)
        idxs = np.linspace(0, len(loader) - 1, num=min(n_scans, len(loader))).astype(int)
        shapes = []
        for i in idxs:
            item = loader[int(i)]
            points = item["points"]
            if points.ndim != 2 or points.shape[1] != 4 or points.dtype != np.float32:
                raise ValueError(
                    f"Bad scan shape/dtype at loader idx {int(i)}: {points.shape} {points.dtype}"
                )
            shapes.append(
                {"loader_idx": int(i), "frame": int(loader.scan_frames[int(i)]),
                 "n_points": int(points.shape[0])}
            )
        cov["checked_scans"] = shapes
        summary[_short_id(sequence)] = cov
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_multi_dataset.yaml")
    parser.add_argument("--encoder-preset", default="no_interdiff")
    parser.add_argument("--checkpoint", default=None,
                        help="Required unless --dry-run")
    parser.add_argument("--use-gated-context", action="store_true")
    parser.add_argument("--gate-initial-alpha", type=float, default=0.0625)
    parser.add_argument(
        "--root",
        default=str(Path(os.environ.get("NSD_DATA_ROOT", "data")) / "kitti360"),
    )
    parser.add_argument("--sequences", nargs="+", default=["0002", "0009"])
    parser.add_argument("--cache-dir", default="data/preprocessed_kitti360_checkpoint_eval")
    parser.add_argument("--bev-cache-dir", default="data/preprocessed_kitti360_bev_layout")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-coarse", type=int, default=800)
    parser.add_argument("--scan-stride", type=int, default=1)
    parser.add_argument("--skip-frames", type=int, default=30)
    parser.add_argument("--distance-threshold", type=float, default=5.0)
    parser.add_argument("--phase-range-freqs", type=int, default=4)
    parser.add_argument("--phase-bev-freqs", type=int, default=8)
    parser.add_argument("--bev-height-encoding", default="max", choices=["iris", "max", "physics3"])
    parser.add_argument("--bev-row-pool", type=int, default=16)
    parser.add_argument("--disable-phase-sketch", action="store_true",
                        help="Skip 384D/512D phase-sketch reranking and evaluate only requested compact paths")
    parser.add_argument("--enable-phase-token", action="store_true",
                        help="Evaluate compact PCA phase-token proxy")
    parser.add_argument("--phase-token-dim", type=int, default=64)
    parser.add_argument("--phase-token-method", default="mag_cross",
                        choices=["mag", "cross", "mag_cross"])
    parser.add_argument("--phase-token-weights", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument(
        "--sensor-key",
        default="kitti",
        help="Sensor key used to select config.encoding.sensor_elevation_ranges (HDL-64E, same as KITTI).",
    )
    parser.add_argument(
        "--elevation-range",
        nargs=2,
        type=float,
        default=(-24.8, 2.0),
        metavar=("MIN_DEG", "MAX_DEG"),
        help="Override range-image elevation range for this dataset (default: HDL-64E).",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU-only: load a few scans per sequence, check bin shapes, "
                             "print pose-interpolation coverage stats, then exit")
    parser.add_argument("--dry-run-scans", type=int, default=5)
    parser.add_argument("--output", default="results/kitti360_checkpoint_eval_nointerdiff288_gate00625_phase384_n800.json")
    args = parser.parse_args()

    if args.dry_run:
        _dry_run(Path(args.root), args.sequences, args.dry_run_scans)
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --dry-run")

    # Torch-dependent imports stay below the dry-run exit so the sanity path
    # runs on hosts without torch.
    from evaluate_kitti_checkpoint import (
        _apply_encoder_preset,
        _load_config,
        evaluate_cache,
    )
    from run_kitti_bev_layout_rerank import _pool_rows

    config = _apply_encoder_preset(_load_config(Path(args.config)), args.encoder_preset)
    enc = config["encoding"]
    if args.elevation_range is not None:
        enc["elevation_range"] = [float(args.elevation_range[0]), float(args.elevation_range[1])]
    else:
        sensor_ranges = enc.get("sensor_elevation_ranges", {})
        if args.sensor_key in sensor_ranges:
            enc["elevation_range"] = sensor_ranges[args.sensor_key]
    print(f"Using elevation_range={enc.get('elevation_range')}", flush=True)
    if args.use_gated_context:
        config["gnn"]["use_residual_gate"] = True
        config["gnn"]["gate_initial_alpha"] = args.gate_initial_alpha

    results = {}
    for sequence in args.sequences:
        seq_id = _short_id(sequence)
        cache_path = _build_kitti360_cache(
            Path(args.root), sequence, config, Path(args.cache_dir), args.device,
            layout_sectors=60, scan_stride=args.scan_stride
        )
        base_cache = np.load(cache_path)
        bev_height_encoding = args.bev_height_encoding
        bev_path = Path(args.bev_cache_dir) / (
            f"kitti360_bev_layout_{seq_id}_s60_{bev_height_encoding}_r1-80_z-3-5_h8"
            f"_stride{args.scan_stride}.npz"
        )
        bev_layouts = _build_kitti360_bev_cache(
            Path(args.root), sequence, base_cache, bev_path,
            n_sectors=60, max_range=80.0, min_range=1.0,
            z_min=-3.0, z_max=5.0, n_height_layers=8,
            height_encoding=bev_height_encoding,
        )
        bev_layouts = _pool_rows(
            bev_layouts,
            args.bev_row_pool,
            "max",
            n_channels=3 if args.bev_height_encoding == "physics3" else 1,
        )
        results[seq_id] = evaluate_cache(
            cache_path=cache_path,
            config=config,
            checkpoint_path=Path(args.checkpoint),
            device=args.device,
            k_values=[1, 5, 10],
            distance_threshold=args.distance_threshold,
            skip_frames=args.skip_frames,
            temporal_edge_mode="bidirectional",
            temporal_direction_mode="none",
            similarity_min_k=0,
            causal_twin=False,
            n_coarse=args.n_coarse,
            layout_score_weights=[2.0, 3.2, 4.0],
            adaptive_gap_scales=[0.05, 0.2, 0.8],
            adaptive_low_weight=0.2,
            adaptive_high_weight=3.2,
            bev_layouts=bev_layouts,
            dual_bev_weights=[1.0, 2.0, 4.0],
            dual_range_weights=[0.0, 0.25, 0.5, 1.0],
            enable_phase_sketch=not args.disable_phase_sketch,
            phase_sketch_only=True,
            phase_range_freqs=args.phase_range_freqs,
            phase_bev_freqs=args.phase_bev_freqs,
            phase_sketch_bev_weights=[0.5, 1.0, 2.0, 4.0, 8.0],
            phase_sketch_range_weights=[0.0, 0.125, 0.25, 0.5, 1.0],
            enable_phase_token=args.enable_phase_token,
            phase_token_dim=args.phase_token_dim,
            phase_token_method=args.phase_token_method,
            phase_token_weights=args.phase_token_weights,
            layout_sectors=60,
            skip_checkpoint=False,
            ctx_weights=[0.0, 0.0625, 0.125, 0.25, 0.5, 1.0],
            sensor_key=args.sensor_key,
        )
        print(seq_id, json.dumps(results[seq_id], indent=2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
