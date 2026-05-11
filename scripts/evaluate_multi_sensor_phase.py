"""Multi-sensor phase fusion evaluation.

Generalizes evaluate_kitti_checkpoint.py to all 4 val sensors (KITTI/NCLT/
HeLiPR/MulRan). For each val sequence:
  1. Pick the right loader and per-sensor elevation range
  2. Build a cache with range_layouts using the encoder
  3. Run phase fusion eval (range phase sketch only — BEV optional)

Cache is per-sensor and contains: descriptors (288D), poses, scan_ids,
nsd_layouts (16×60 range image projection used by phase sketch).

Usage:
    python3 scripts/evaluate_multi_sensor_phase.py \
        --config configs/training_multi_dataset.yaml \
        --checkpoint results/ablation_no_interdiff_gate_seed42/best_model.pth \
        --output results/phase_eval_all_sensors.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data.kitti_loader import KITTILoader  # noqa: E402
from data.nclt_loader import NCLTLoader  # noqa: E402
from data.helipr_loader import HeLiPRLoader  # noqa: E402
from data.mulran_loader import MulRanLoader  # noqa: E402
from keyframe.selector import KeyframeSelector  # noqa: E402
from encoding.spectral_encoder import SpectralEncoder  # noqa: E402
from encoding.bev_image import BEVProjector, interpolate_bev_image  # noqa: E402

from run_kitti_operating_point import _project_nsd_layout  # noqa: E402
from run_kitti_bev_layout_rerank import _pool_rows  # noqa: E402
from evaluate_kitti_checkpoint import (  # noqa: E402
    _make_model,
    _build_eval_graph,
    _cache_to_keyframes,
    _scale_context_embeddings,
    _recall_with_phase_sketch_fusion,
    _recall_cosine,
    _apply_encoder_preset,
    _load_config,
)


def make_loader(dataset_type: str, root: str, sequence: str):
    if dataset_type == "kitti":
        return KITTILoader(root, sequence, lazy_load=True)
    if dataset_type == "nclt":
        return NCLTLoader(root, sequence, lazy_load=True)
    if dataset_type == "helipr":
        seq_path = os.path.join(root, sequence, sequence)
        return HeLiPRLoader(seq_path, lazy_load=True)
    if dataset_type == "mulran":
        return MulRanLoader(root, sequence, lazy_load=True)
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def make_encoder(config: Dict, dataset_type: str, device: str) -> SpectralEncoder:
    enc_cfg = config["encoding"]
    elev_ranges = enc_cfg.get("sensor_elevation_ranges", {})
    elev_range = elev_ranges.get(dataset_type, enc_cfg["elevation_range"])
    encoder = SpectralEncoder(
        n_elevation=enc_cfg["n_elevation"],
        n_azimuth=enc_cfg["n_azimuth"],
        elevation_range=tuple(elev_range),
        max_range=enc_cfg["max_range"],
        min_range=enc_cfg["min_range"],
        target_elevation_bins=enc_cfg.get("target_elevation_bins", 16),
        binning_strategy=enc_cfg.get("binning_strategy", "octave"),
        n_bins=enc_cfg.get("n_bins", 16),
        alpha=enc_cfg.get("alpha", 2.0),
        learnable_alpha=enc_cfg.get("learnable_alpha", False),
        epsilon=enc_cfg.get("epsilon", 1e-8),
        bin_statistics=enc_cfg.get("bin_statistics", ["mean", "std"]),
        inter_bin_statistics=enc_cfg.get("inter_bin_statistics", []),
        zero_center=enc_cfg.get("zero_center", False),
        log_magnitude=enc_cfg.get("log_magnitude", False),
        normalize_channels=enc_cfg.get("normalize_channels", False),
    )
    encoder = encoder.to(device).eval()
    return encoder


def build_cache_for_sequence(
    dataset_type: str,
    root: str,
    sequence: str,
    config: Dict,
    cache_dir: Path,
    device: str,
    layout_sectors: int = 60,
    build_bev: bool = True,
    bev_row_pool: int = 16,
    bev_max_range: float = 80.0,
    bev_min_range: float = 1.0,
    bev_z_min: float = -3.0,
    bev_z_max: float = 5.0,
    bev_height_layers: int = 8,
    bev_height_encoding: str = "max",
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_seq = sequence.replace("/", "_")
    suffix = f"_bev{bev_row_pool}" if build_bev else "_norebev"
    cache_path = cache_dir / f"phase_eval_{dataset_type}_{safe_seq}_layout{layout_sectors}{suffix}.npz"
    if cache_path.exists():
        print(f"  [cache hit] {cache_path.name}")
        return cache_path

    loader = make_loader(dataset_type, root, sequence)
    encoder = make_encoder(config, dataset_type, device)
    key_cfg = config["keyframe"]
    selector = KeyframeSelector(
        distance_threshold=key_cfg.get("distance_threshold", 0.8),
        rotation_threshold=key_cfg.get("rotation_threshold", 20.0),
        overlap_threshold=key_cfg.get("overlap_threshold", 0.65),
        temporal_threshold=key_cfg.get("temporal_threshold", 30.0),
        voxel_size=key_cfg.get("voxel_size", 0.2),
        max_keyframes=key_cfg.get("max_keyframes", 10000000),
    )

    bev_projector = None
    if build_bev:
        bev_projector = BEVProjector(
            n_sectors=layout_sectors,
            max_range=bev_max_range,
            min_range=bev_min_range,
            z_min=bev_z_min,
            height_encoding=bev_height_encoding,
            n_height_layers=bev_height_layers,
            z_max=bev_z_max,
        )

    n_scans = len(loader)
    descriptors, poses, timestamps, scan_ids, keyframe_ids = [], [], [], [], []
    nsd_layouts, bev_layouts = [], []

    for scan_id in range(n_scans):
        if scan_id % 500 == 0:
            print(f"    [{dataset_type}/{sequence}] {scan_id}/{n_scans}, kf={len(scan_ids)}", flush=True)
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
        nsd_layout = _project_nsd_layout(encoder, item["points"], n_layout_sectors=layout_sectors)
        descriptors.append(desc)
        poses.append(keyframe.pose.astype(np.float64))
        timestamps.append(float(keyframe.timestamp))
        scan_ids.append(int(scan_id))
        keyframe_ids.append(int(keyframe.keyframe_id))
        nsd_layouts.append(nsd_layout.astype(np.float32))
        if bev_projector is not None:
            bev, _ = bev_projector.project(item["points"], keep_intensity=False)
            bev = interpolate_bev_image(bev, method="linear")
            bev_layouts.append(bev.astype(np.float32))

    save_kwargs = dict(
        descriptors=np.asarray(descriptors, dtype=np.float32),
        poses=np.asarray(poses, dtype=np.float64),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        scan_ids=np.asarray(scan_ids, dtype=np.int64),
        keyframe_ids=np.asarray(keyframe_ids, dtype=np.int64),
        nsd_layouts=np.asarray(nsd_layouts, dtype=np.float32),
    )
    if bev_layouts:
        bev_arr = np.asarray(bev_layouts, dtype=np.float32)
        bev_arr = _pool_rows(bev_arr, bev_row_pool, "max")
        save_kwargs["bev_layouts"] = bev_arr

    np.savez_compressed(cache_path, **save_kwargs)
    return cache_path


def evaluate_sequence(
    cache_path: Path,
    config: Dict,
    checkpoint_path: Path,
    device: str,
    n_coarse: int,
    distance_threshold: float,
    skip_frames: int,
    phase_range_freqs: int,
    phase_bev_freqs: int,
    layout_sectors: int,
    sketch_bev_weights: List[float],
    sketch_range_weights: List[float],
    ctx_weights: List[float],
) -> Dict:
    cache = np.load(cache_path)
    keyframes = _cache_to_keyframes(cache)
    poses = cache["poses"]
    descriptors = cache["descriptors"].astype(np.float32)
    nsd_layouts = cache["nsd_layouts"].astype(np.float32)
    has_bev = "bev_layouts" in cache.files
    bev_layouts = cache["bev_layouts"].astype(np.float32) if has_bev else nsd_layouts

    # Forward pass: 288D raw → 672D refined
    model = _make_model(config, checkpoint_path, device)
    with torch.no_grad():
        graph = _build_eval_graph(
            keyframes=keyframes,
            poses=poses,
            descriptors=descriptors,
            cache=cache,
            config=config,
            device=device,
            temporal_edge_mode="bidirectional",
            temporal_direction_mode="none",
            similarity_min_k=0,
        )
        embeddings = model(graph.to(device)).detach().cpu().numpy()

    raw_dim = descriptors.shape[1]
    k_values = [1, 5, 10]

    results = {
        "n_keyframes": int(len(descriptors)),
        "n_queries": 0,
        "has_bev": has_bev,
        "raw": _recall_cosine(descriptors, poses, k_values, distance_threshold, skip_frames),
        "final": _recall_cosine(embeddings, poses, k_values, distance_threshold, skip_frames),
    }
    bev_weights = sketch_bev_weights if has_bev else [0.0]

    results["raw_phase_sketch"] = _recall_with_phase_sketch_fusion(
        descriptors, nsd_layouts, bev_layouts, poses, k_values,
        distance_threshold, skip_frames, n_coarse,
        phase_range_freqs, phase_bev_freqs, layout_sectors,
        bev_weights, sketch_range_weights,
    )
    results["final_phase_sketch"] = _recall_with_phase_sketch_fusion(
        embeddings, nsd_layouts, bev_layouts, poses, k_values,
        distance_threshold, skip_frames, n_coarse,
        phase_range_freqs, phase_bev_freqs, layout_sectors,
        bev_weights, sketch_range_weights,
    )
    results["final_phase_sketch_ctx_weight"] = {
        f"w{w:g}": _recall_with_phase_sketch_fusion(
            _scale_context_embeddings(embeddings, raw_dim, w),
            nsd_layouts, bev_layouts, poses, k_values,
            distance_threshold, skip_frames, n_coarse,
            phase_range_freqs, phase_bev_freqs, layout_sectors,
            bev_weights, sketch_range_weights,
        )
        for w in ctx_weights
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_multi_dataset.yaml")
    parser.add_argument("--encoder-preset", default="no_interdiff")
    parser.add_argument("--use-gated-context", action="store_true", default=True)
    parser.add_argument("--gate-initial-alpha", type=float, default=0.0625)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", default="data/preprocessed_phase_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-coarse", type=int, default=800)
    parser.add_argument("--distance-threshold", type=float, default=5.0)
    parser.add_argument("--skip-frames", type=int, default=30)
    parser.add_argument("--layout-sectors", type=int, default=60)
    parser.add_argument("--phase-range-freqs", type=int, default=4)
    parser.add_argument("--phase-bev-freqs", type=int, default=8)
    parser.add_argument("--phase-sketch-bev-weights", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--phase-sketch-range-weights", nargs="+", type=float,
                        default=[0.0, 0.125, 0.25, 0.5, 1.0])
    parser.add_argument("--ctx-weights", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--no-bev", action="store_true",
                        help="Skip BEV layout computation (range phase only)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Filter to specific dataset types (e.g., 'kitti nclt')")
    parser.add_argument("--output", default="results/phase_eval_all_sensors.json")
    args = parser.parse_args()

    config = _apply_encoder_preset(_load_config(Path(args.config)), args.encoder_preset)
    if args.use_gated_context:
        config["gnn"]["use_residual_gate"] = True
        config["gnn"]["gate_initial_alpha"] = args.gate_initial_alpha

    val_datasets = config["data"]["datasets"]["val"]
    if args.datasets:
        val_datasets = [d for d in val_datasets if d["type"] in args.datasets]

    all_results = {}
    cache_dir = Path(args.cache_dir)

    for ds in val_datasets:
        dataset_type = ds["type"]
        root = ds["root"]
        for seq in ds["sequences"]:
            name = f"{dataset_type.upper()}_{seq}"
            print(f"\n=== {name} ===", flush=True)
            cache_path = build_cache_for_sequence(
                dataset_type=dataset_type,
                root=root,
                sequence=seq,
                config=config,
                cache_dir=cache_dir,
                device=args.device,
                layout_sectors=args.layout_sectors,
                build_bev=not args.no_bev,
            )
            results = evaluate_sequence(
                cache_path=cache_path,
                config=config,
                checkpoint_path=Path(args.checkpoint),
                device=args.device,
                n_coarse=args.n_coarse,
                distance_threshold=args.distance_threshold,
                skip_frames=args.skip_frames,
                phase_range_freqs=args.phase_range_freqs,
                phase_bev_freqs=args.phase_bev_freqs,
                layout_sectors=args.layout_sectors,
                sketch_bev_weights=args.phase_sketch_bev_weights,
                sketch_range_weights=args.phase_sketch_range_weights,
                ctx_weights=args.ctx_weights,
            )
            all_results[name] = results
            r1_raw = results["raw"]["R@1"]
            r1_final = results["final"]["R@1"]
            # find best phase R@1
            best_r1 = 0.0
            best_key = ""
            for ctx_key, ctx_block in results.get("final_phase_sketch_ctx_weight", {}).items():
                for fk, fv in ctx_block.items():
                    if isinstance(fv, dict) and "R@1" in fv and fv["R@1"] > best_r1:
                        best_r1 = fv["R@1"]
                        best_key = f"{ctx_key}/{fk}"
            print(f"  raw={r1_raw:.4f}  final={r1_final:.4f}  best_phase={best_r1:.4f} ({best_key})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
