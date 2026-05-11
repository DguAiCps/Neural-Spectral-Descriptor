#!/usr/bin/env python3
"""Zero-shot NCLT evaluation for the KITTI-trained learned phase reranker.

This script intentionally reuses the KITTI learned reranker without fine-tuning.
It evaluates whether the learned residual over the analytic BEV phase-alignment
score transfers to NCLT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_kitti_checkpoint import (  # noqa: E402
    _apply_encoder_preset,
    _load_config,
    _phase_sketch,
    _phase_sketch_keys,
)
from evaluate_nclt_checkpoint import _build_nclt_bev_cache, _build_nclt_cache  # noqa: E402
from gnn.learned_reranker import PhaseCorrelationReranker  # noqa: E402
from run_kitti_bev_layout_rerank import _pool_rows  # noqa: E402
from train_kitti_learned_reranker import _evaluate, _load_embeddings, _query_refs  # noqa: E402


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _make_reranker(
    checkpoint_path: Path,
    device: str,
    residual_scale: float = 1.0,
) -> Tuple[PhaseCorrelationReranker, Dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_args = ckpt.get("args", {})
    adaptive_residual_gate = _as_bool(saved_args.get("adaptive_residual_gate", False))
    state_dict = ckpt["model_state_dict"]
    checkpoint_has_gate_weights = any(k.startswith("gate_net.") for k in state_dict)
    if adaptive_residual_gate and not checkpoint_has_gate_weights:
        raise RuntimeError(
            "Checkpoint metadata requests adaptive_residual_gate=True, but "
            "gate_net weights are absent."
        )
    model = PhaseCorrelationReranker(
        n_shifts=int(saved_args.get("layout_sectors", 60)),
        hidden_dim=int(saved_args.get("hidden_dim", 128)),
        dropout=float(saved_args.get("dropout", 0.1)),
        base_phase_weight=float(saved_args.get("base_phase_weight", 10.0)),
        base_embedding_weight=float(saved_args.get("base_embedding_weight", 1.0)),
        adaptive_residual_gate=adaptive_residual_gate,
        gate_hidden_dim=int(saved_args.get("gate_hidden_dim", 16)),
        gate_initial_alpha=float(saved_args.get("residual_gate_initial_alpha", 0.25)),
        residual_scale=residual_scale,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ignored_missing = [k for k in missing if k.startswith("gate_net.") and not adaptive_residual_gate]
    real_missing = [k for k in missing if k not in ignored_missing]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Reranker checkpoint mismatch: missing={real_missing}, "
            f"unexpected={unexpected}"
        )
    model.eval()
    metadata = {
        "reranker_checkpoint": str(checkpoint_path),
        "adaptive_residual_gate": adaptive_residual_gate,
        "saved_adaptive_residual_gate_raw": saved_args.get("adaptive_residual_gate", None),
        "checkpoint_has_gate_weights": checkpoint_has_gate_weights,
        "ignored_missing_keys": ignored_missing,
        "n_shifts": int(saved_args.get("layout_sectors", 60)),
        "hidden_dim": int(saved_args.get("hidden_dim", 128)),
        "base_phase_weight": float(saved_args.get("base_phase_weight", 10.0)),
        "base_embedding_weight": float(saved_args.get("base_embedding_weight", 1.0)),
        "residual_scale": float(residual_scale),
    }
    return model, metadata


def _prepare_nclt_sequence(
    date: str,
    root: Path,
    config: Dict,
    encoder_checkpoint: Path,
    cache_dir: Path,
    bev_cache_dir: Path,
    device: str,
    layout_sectors: int,
    bev_freqs: int,
    bev_row_pool: int,
    scan_stride: int,
    temporal_edge_mode: str,
    temporal_direction_mode: str,
    similarity_min_k: int,
    sensor_key: str,
    bev_height_encoding: str,
) -> Dict:
    cache_path = _build_nclt_cache(
        root=root,
        date=date,
        config=config,
        cache_dir=cache_dir,
        device=device,
        layout_sectors=layout_sectors,
        scan_stride=scan_stride,
    )
    cache = np.load(cache_path)
    bev_path = bev_cache_dir / (
        f"nclt_bev_layout_{date}_s{layout_sectors}_{bev_height_encoding}_r1-80_z-3-5_h8.npz"
    )
    bev_layouts = _build_nclt_bev_cache(
        root=root,
        date=date,
        base_cache=cache,
        output_path=bev_path,
        n_sectors=layout_sectors,
        max_range=80.0,
        min_range=1.0,
        z_min=-3.0,
        z_max=5.0,
        n_height_layers=8,
        height_encoding=bev_height_encoding,
    )
    bev_layouts = _pool_rows(
        bev_layouts,
        bev_row_pool,
        "max",
        n_channels=3 if bev_height_encoding == "physics3" else 1,
    )
    embeddings = _load_embeddings(
        cache=cache,
        config=config,
        checkpoint=encoder_checkpoint,
        device=device,
        temporal_edge_mode=temporal_edge_mode,
        temporal_direction_mode=temporal_direction_mode,
        similarity_min_k=similarity_min_k,
        sensor_key=sensor_key,
    )
    sketch = _phase_sketch(bev_layouts.astype(np.float32), bev_freqs)
    return {
        "sequence": date,
        "poses": cache["poses"].astype(np.float64),
        "embeddings": embeddings.astype(np.float32),
        "emb_norm": embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8),
        "phase_sketch": sketch,
        "phase_keys": _phase_sketch_keys(sketch),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_kitti_only.yaml")
    parser.add_argument("--encoder-preset", default="no_interdiff")
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--reranker-checkpoint", required=True)
    parser.add_argument("--use-gated-context", action="store_true")
    parser.add_argument("--gate-initial-alpha", type=float, default=0.0625)
    parser.add_argument("--root", default="/rise/RISE1/workspace/data/nclt")
    parser.add_argument("--dates", nargs="+", default=["2012-01-08", "2013-01-10"])
    parser.add_argument("--cache-dir", default="data/preprocessed_nclt_learned_reranker")
    parser.add_argument("--bev-cache-dir", default="data/preprocessed_nclt_bev_layout_reranker")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layout-sectors", type=int, default=60)
    parser.add_argument("--bev-freqs", type=int, default=12)
    parser.add_argument("--bev-row-pool", type=int, default=16)
    parser.add_argument("--bev-height-encoding", default="max", choices=["iris", "max", "physics3"])
    parser.add_argument("--scan-stride", type=int, default=5)
    parser.add_argument("--skip-frames", type=int, default=6)
    parser.add_argument("--distance-threshold", type=float, default=5.0)
    parser.add_argument("--temporal-edge-mode", default="bidirectional")
    parser.add_argument("--temporal-direction-mode", default="none")
    parser.add_argument("--similarity-min-k", type=int, default=0)
    parser.add_argument("--n-coarse", type=int, default=400)
    parser.add_argument("--max-candidates", type=int, default=800)
    parser.add_argument("--include-phase-candidates", action="store_true")
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="Scale learned residual at evaluation time. Use 0 for base-only transfer.",
    )
    parser.add_argument("--sensor-key", default="nclt")
    parser.add_argument("--output", default="results/nclt_learned_reranker_zero_shot.json")
    args = parser.parse_args()

    config = _apply_encoder_preset(_load_config(Path(args.config)), args.encoder_preset)
    # The zero-shot reranker reuses only the frozen 288D+128D embedding model.
    # Drop phase-only experimental heads from richer configs; phase alignment is
    # handled explicitly by the reranker below, not inside the GNN forward pass.
    for key in (
        "phase_token",
        "phase_edge",
        "phase_alignment_edge",
        "phase_coherence",
        "dual_stream",
    ):
        config.get("gnn", {}).pop(key, None)
    enc = config["encoding"]
    sensor_ranges = enc.get("sensor_elevation_ranges", {})
    if args.sensor_key in sensor_ranges:
        enc["elevation_range"] = sensor_ranges[args.sensor_key]
    if args.use_gated_context:
        config["gnn"]["use_residual_gate"] = True
        config["gnn"]["gate_initial_alpha"] = args.gate_initial_alpha

    seqs = [
        _prepare_nclt_sequence(
            date=date,
            root=Path(args.root),
            config=config,
            encoder_checkpoint=Path(args.encoder_checkpoint),
            cache_dir=Path(args.cache_dir),
            bev_cache_dir=Path(args.bev_cache_dir),
            device=args.device,
            layout_sectors=args.layout_sectors,
            bev_freqs=args.bev_freqs,
            bev_row_pool=args.bev_row_pool,
            scan_stride=args.scan_stride,
            temporal_edge_mode=args.temporal_edge_mode,
            temporal_direction_mode=args.temporal_direction_mode,
            similarity_min_k=args.similarity_min_k,
            sensor_key=args.sensor_key,
            bev_height_encoding=args.bev_height_encoding,
        )
        for date in args.dates
    ]
    refs = _query_refs(seqs, args.distance_threshold, args.skip_frames)
    model, reranker_metadata = _make_reranker(
        Path(args.reranker_checkpoint),
        args.device,
        residual_scale=args.residual_scale,
    )
    metrics = _evaluate(
        model=model,
        seqs=seqs,
        refs=refs,
        n_coarse=args.n_coarse,
        max_candidates=args.max_candidates,
        skip_frames=args.skip_frames,
        distance_threshold=args.distance_threshold,
        include_phase_candidates=args.include_phase_candidates,
        device=args.device,
        n_sectors=args.layout_sectors,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "metadata": {
                    "encoder_checkpoint": args.encoder_checkpoint,
                    "config": args.config,
                    "encoder_preset": args.encoder_preset,
                    "reranker": reranker_metadata,
                },
                "args": vars(args),
            },
            indent=2,
        )
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
