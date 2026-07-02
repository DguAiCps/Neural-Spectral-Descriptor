"""Cross-sensor scan-level + BEV-layout cache builders.

Wraps the dataset-specific loaders (KITTI/NCLT/HeLiPR/MulRan) behind a uniform
`_build_dataset_cache` / `_build_dataset_bev_cache` dispatcher so the
cross-sensor learned reranker training script can treat all four sensors the
same way.

The output cache schema matches the existing KITTI/NCLT operating caches:
descriptors, poses, timestamps, scan_ids, keyframe_ids, fft_magnitudes,
nsd_layouts — all aligned along the keyframe axis.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.helipr_loader import HeLiPRLoader  # noqa: E402
from data.kitti_loader import KITTILoader  # noqa: E402
from data.mulran_loader import MulRanLoader  # noqa: E402
from data.nclt_loader import NCLTLoader  # noqa: E402
from encoding.bev_image import BEVProjector, interpolate_bev_image  # noqa: E402
from keyframe.selector import KeyframeSelector  # noqa: E402

# Lazy imports from sibling scripts (avoid circular when this module is loaded
# from train_cross_sensor_learned_reranker.py before evaluate_kitti_checkpoint
# is fully imported).
from run_kitti_operating_point import _make_encoder, _project_nsd_layout  # noqa: E402


def _make_loader(dataset_type: str, root: str, sequence: str):
    if dataset_type == "kitti":
        return KITTILoader(root, sequence, lazy_load=True)
    if dataset_type == "nclt":
        return NCLTLoader(root, sequence, lazy_load=True)
    if dataset_type == "helipr":
        seq_path = Path(root) / sequence / sequence
        return HeLiPRLoader(str(seq_path), lazy_load=True)
    if dataset_type == "mulran":
        return MulRanLoader(root, sequence, lazy_load=True)
    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def _sensor_elevation(config: Dict, dataset_type: str):
    ranges = config["encoding"].get("sensor_elevation_ranges", {}) or {}
    return ranges.get(dataset_type)


def _build_operating_cache(
    dataset_type: str,
    root: str,
    sequence: str,
    config: Dict,
    cache_dir: Path,
    device: str,
    layout_sectors: int,
    scan_stride: int,
) -> Path:
    """Build (or reuse) the scan-level operating cache for one sequence.

    Cache file naming: `{dataset_type}_operating_{sequence}_layout{S}_stride{T}.npz`.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_seq = sequence.replace("/", "_")
    cache_path = cache_dir / f"{dataset_type}_operating_{safe_seq}_layout{layout_sectors}_stride{scan_stride}.npz"
    if cache_path.exists():
        return cache_path

    loader = _make_loader(dataset_type, root, sequence)
    encoder = _make_encoder(config, device)
    elev = _sensor_elevation(config, dataset_type)
    if elev is not None and hasattr(encoder, "set_elevation_range"):
        encoder.set_elevation_range(tuple(elev))

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
    n = len(loader)
    for scan_id in range(0, n, scan_stride):
        if scan_id % 500 == 0:
            print(
                f"[{dataset_type}:{sequence}] scan {scan_id}/{n}, keyframes={len(scan_ids)}",
                flush=True,
            )
        item = loader[scan_id]
        selected, keyframe, _ = selector.process_scan(
            scan_id=scan_id,
            points=item["points"],
            pose=item["pose"],
            timestamp=item["timestamp"],
        )
        if not selected:
            continue
        pts = item["points"]
        descriptors.append(encoder.encode_points(pts).detach().cpu().numpy().astype(np.float32))
        fft_magnitudes.append(encoder.compute_fft_magnitudes(pts).astype(np.float32))
        nsd_layouts.append(_project_nsd_layout(encoder, pts, n_layout_sectors=layout_sectors))
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


def _build_bev_layout_cache(
    dataset_type: str,
    root: str,
    sequence: str,
    operating_cache_path: Path,
    bev_cache_dir: Path,
    *,
    n_sectors: int = 60,
    max_range: float = 80.0,
    min_range: float = 1.0,
    z_min: float = -3.0,
    z_max: float = 5.0,
    n_height_layers: int = 8,
    height_encoding: str = "max",
) -> Path:
    """Build per-keyframe BEV layout cache aligned with operating cache scan_ids."""
    bev_cache_dir.mkdir(parents=True, exist_ok=True)
    safe_seq = sequence.replace("/", "_")
    cache_path = bev_cache_dir / (
        f"{dataset_type}_bev_layout_{safe_seq}_s{n_sectors}_{height_encoding}_"
        f"r{int(min_range)}-{int(max_range)}_z{int(z_min)}-{int(z_max)}_h{n_height_layers}.npz"
    )
    if cache_path.exists():
        return cache_path

    cache = np.load(operating_cache_path)
    scan_ids = cache["scan_ids"].astype(np.int64)
    loader = _make_loader(dataset_type, root, sequence)

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
    for i, scan_id in enumerate(scan_ids):
        if i % 500 == 0:
            print(f"[{dataset_type}:{sequence}] BEV layout {i}/{len(scan_ids)}", flush=True)
        points = loader[int(scan_id)]["points"]
        bev, _ = projector.project(points, keep_intensity=False)
        layouts.append(
            interpolate_bev_image(
                bev,
                method="linear",
                n_channels=3 if height_encoding == "physics3" else 1,
            ).astype(np.float32)
        )

    np.savez_compressed(cache_path, bev_layouts=np.asarray(layouts, dtype=np.float32))
    return cache_path


def build_cache(
    dataset_type: str,
    root: str,
    sequence: str,
    *,
    config: Dict,
    cache_dir: Path,
    bev_cache_dir: Path,
    device: str,
    layout_sectors: int = 60,
    scan_stride: int = 1,
    bev_height_encoding: str = "max",
    bev_n_sectors: int = 60,
    bev_max_range: float = 80.0,
    bev_min_range: float = 1.0,
    bev_z_min: float = -3.0,
    bev_z_max: float = 5.0,
    bev_n_height_layers: int = 8,
):
    """Build operating + BEV layout caches for one (dataset, sequence) pair.

    Returns (operating_cache_path, bev_cache_path).
    """
    op_path = _build_operating_cache(
        dataset_type=dataset_type,
        root=root,
        sequence=sequence,
        config=config,
        cache_dir=cache_dir,
        device=device,
        layout_sectors=layout_sectors,
        scan_stride=scan_stride,
    )
    bev_path = _build_bev_layout_cache(
        dataset_type=dataset_type,
        root=root,
        sequence=sequence,
        operating_cache_path=op_path,
        bev_cache_dir=bev_cache_dir,
        n_sectors=bev_n_sectors,
        max_range=bev_max_range,
        min_range=bev_min_range,
        z_min=bev_z_min,
        z_max=bev_z_max,
        n_height_layers=bev_n_height_layers,
        height_encoding=bev_height_encoding,
    )
    return op_path, bev_path
