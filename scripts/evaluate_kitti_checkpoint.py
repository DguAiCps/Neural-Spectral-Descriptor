#!/usr/bin/env python3
"""Evaluate an NSD checkpoint on KITTI cache files.

This is a lightweight checkpoint path for fast KITTI ablations. It reuses the
cache produced by `run_kitti_operating_point.py`, builds the trajectory graph,
loads `best_model.pth`, and reports raw/context/final Recall@K plus NSD-native
layout/phase fusion operating points.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gnn.model import create_spectral_gnn  # noqa: E402
from keyframe.graph_manager import build_graph_from_keyframes_batch  # noqa: E402
from keyframe.selector import Keyframe  # noqa: E402
from run_kitti_operating_point import (  # noqa: E402
    _adaptive_score_fusion,
    _build_sequence_cache,
    _distance_columnwise,
    _find_queries,
    _layout_row_keys,
    _minmax01,
    _normalize,
    _score,
    _score_fusion,
    _topk_cosine,
)
from run_kitti_bev_layout_rerank import (  # noqa: E402
    _build_bev_layout_cache,
    _pool_rows,
)


def _load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _apply_encoder_preset(config: Dict, preset: str) -> Dict:
    """Apply compact closed-form encoder presets for quick ablations."""
    enc = config["encoding"]
    gnn = config["gnn"]
    if preset == "full":
        return config

    # These presets test closed-form octave encoders. Disable spectral_policy so
    # the measured dimension is exactly the hand-designed encoder dimension.
    enc.setdefault("spectral_policy", {})["enabled"] = False
    enc["binning_strategy"] = "octave"
    enc["zero_center"] = False

    enc["cross_spectrum"] = {"enabled": False, "n_freqs": 0}

    if preset == "no_interdiff":
        enc["target_elevation_bins"] = 16
        enc["bin_statistics"] = ["mean", "std"]
        enc["inter_bin_statistics"] = []
        gnn["input_dim"] = 16 * 9 * 2
    elif preset == "cross4_no_interdiff":
        enc["target_elevation_bins"] = 16
        enc["bin_statistics"] = ["mean", "std"]
        enc["inter_bin_statistics"] = []
        enc["cross_spectrum"] = {"enabled": True, "n_freqs": 4}
        gnn["input_dim"] = 16 * 9 * 2 + 15 * 4 * 2
    elif preset == "cross8_no_interdiff":
        enc["target_elevation_bins"] = 16
        enc["bin_statistics"] = ["mean", "std"]
        enc["inter_bin_statistics"] = []
        enc["cross_spectrum"] = {"enabled": True, "n_freqs": 8}
        gnn["input_dim"] = 16 * 9 * 2 + 15 * 8 * 2
    elif preset == "mean_diff":
        enc["target_elevation_bins"] = 16
        enc["bin_statistics"] = ["mean"]
        enc["inter_bin_statistics"] = ["diff"]
        gnn["input_dim"] = 16 * (9 + 8)
    elif preset == "rows12_full":
        enc["target_elevation_bins"] = 12
        enc["bin_statistics"] = ["mean", "std"]
        enc["inter_bin_statistics"] = ["diff"]
        gnn["input_dim"] = 12 * (9 * 2 + 8 * 2)
    else:
        raise ValueError(f"Unknown encoder preset: {preset}")

    return config


def _make_model(config: Dict, checkpoint_path: Path, device: str) -> torch.nn.Module:
    spectral_policy = None
    policy_cfg = config["encoding"].get("spectral_policy", {})
    if policy_cfg.get("enabled", False):
        from encoding.spectral_policy import create_spectral_policy

        spectral_policy = create_spectral_policy(
            policy_cfg,
            n_rings=config["encoding"].get("target_elevation_bins", 16),
            n_freqs=config["encoding"].get("n_azimuth", 360) // 2 + 1,
        )

    gnn_cfg = config["gnn"]
    model = create_spectral_gnn(
        input_dim=gnn_cfg["input_dim"],
        hidden_dim=gnn_cfg["hidden_dim"],
        context_dim=gnn_cfg["context_dim"],
        n_layers=gnn_cfg["n_layers"],
        n_heads=gnn_cfg.get("n_heads", 4),
        dropout=gnn_cfg.get("dropout", 0.1),
        use_local_updates=gnn_cfg.get("use_local_updates", True),
        local_update_hops=gnn_cfg.get("local_update_hops", 3),
        edge_encoder_config=gnn_cfg.get("edge_encoding"),
        spectral_policy=spectral_policy,
        norm_type=gnn_cfg.get("norm_type", "batch_norm"),
        use_residual_gate=gnn_cfg.get("use_residual_gate", False),
        gate_hidden_dim=gnn_cfg.get("gate_hidden_dim", 64),
        gate_initial_alpha=gnn_cfg.get("gate_initial_alpha", 0.5),
        use_edge_confidence_gate=gnn_cfg.get("use_edge_confidence_gate", False),
        edge_gate_hidden_dim=gnn_cfg.get("edge_gate_hidden_dim", 16),
        phase_token_config=gnn_cfg.get("phase_token"),
        phase_edge_config=gnn_cfg.get("phase_edge"),
        phase_alignment_config=gnn_cfg.get("phase_alignment_edge"),
        phase_coherence_config=gnn_cfg.get("phase_coherence"),
        dual_stream_config=gnn_cfg.get("dual_stream"),
        sensor_gate_config=gnn_cfg.get("sensor_gate"),
        diffattn_value_source=gnn_cfg.get("diffattn_value_source", "diff"),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"missing_keys={missing}")
    if unexpected:
        print(f"unexpected_keys={unexpected}")
    print(
        f"loaded_checkpoint={checkpoint_path} epoch={ckpt.get('epoch')} "
        f"best_val_metric={ckpt.get('best_val_metric')}"
    )
    model.eval()
    return model


def _cache_to_keyframes(cache: np.lib.npyio.NpzFile) -> List[Keyframe]:
    keyframes = []
    for i, desc in enumerate(cache["descriptors"]):
        keyframes.append(
            Keyframe(
                keyframe_id=int(cache["keyframe_ids"][i]),
                scan_id=int(cache["scan_ids"][i]),
                points=np.empty((0, 3), dtype=np.float32),
                pose=cache["poses"][i],
                timestamp=float(cache["timestamps"][i]),
                descriptor=desc.astype(np.float32),
            )
        )
    return keyframes


def _recall_cosine(
    embeddings: np.ndarray,
    poses: np.ndarray,
    k_values: List[int],
    distance_threshold: float,
    skip_frames: int,
) -> Dict[str, float]:
    normed = _normalize(embeddings)
    queries = _find_queries(poses, distance_threshold, skip_frames)
    ranked = [
        _topk_cosine(normed, q_idx, max(k_values) + 2 * skip_frames, skip_frames)[: max(k_values)]
        for q_idx, _ in queries
    ]
    return _score(poses, queries, ranked, k_values, distance_threshold)


def _recall_with_layout_fusion(
    embeddings: np.ndarray,
    nsd_layouts: np.ndarray,
    poses: np.ndarray,
    k_values: List[int],
    distance_threshold: float,
    skip_frames: int,
    n_coarse: int,
    layout_score_weights: List[float],
    adaptive_gap_scales: List[float],
    adaptive_low_weight: float,
    adaptive_high_weight: float,
) -> Dict[str, Dict[str, float]]:
    """Evaluate NSD-native phase/layout reranking on top of an embedding."""
    normed = _normalize(embeddings)
    queries = _find_queries(poses, distance_threshold, skip_frames)
    score_ranked = {w: [] for w in layout_score_weights}
    adaptive_ranked = {g: [] for g in adaptive_gap_scales}

    for query_idx, _ in queries:
        candidates = _topk_cosine(normed, query_idx, n_coarse, skip_frames)
        raw_distances = 1.0 - (normed[candidates] @ normed[query_idx])
        layout_distances = np.asarray([
            _distance_columnwise(nsd_layouts[query_idx], nsd_layouts[int(c)])
            for c in candidates
        ], dtype=np.float32)
        for weight in layout_score_weights:
            fused = _score_fusion(
                candidates,
                raw_distances,
                layout_distances,
                layout_weight=weight,
            )
            score_ranked[weight].append(fused[:max(k_values)])
        for gap_scale in adaptive_gap_scales:
            fused = _adaptive_score_fusion(
                candidates,
                raw_distances,
                layout_distances,
                low_weight=adaptive_low_weight,
                high_weight=adaptive_high_weight,
                gap_scale=gap_scale,
            )
            adaptive_ranked[gap_scale].append(fused[:max(k_values)])

    results: Dict[str, Dict[str, float]] = {}
    for weight, ranked in score_ranked.items():
        results[f"layout_score_fusion_w{weight:g}"] = _score(
            poses,
            queries,
            ranked,
            k_values,
            distance_threshold,
        )
    for gap_scale, ranked in adaptive_ranked.items():
        results[
            "layout_adaptive_score_"
            f"lo{adaptive_low_weight:g}_hi{adaptive_high_weight:g}_gap{gap_scale:g}"
        ] = _score(
            poses,
            queries,
            ranked,
            k_values,
            distance_threshold,
        )
    return results


def _recall_with_dual_layout_fusion(
    embeddings: np.ndarray,
    range_layouts: np.ndarray,
    bev_layouts: np.ndarray,
    poses: np.ndarray,
    k_values: List[int],
    distance_threshold: float,
    skip_frames: int,
    n_coarse: int,
    dual_bev_weights: List[float],
    dual_range_weights: List[float],
) -> Dict[str, Dict[str, float]]:
    """Evaluate GNN embeddings with NSD-owned BEV/range layout reranking.

    This is the candidate method extension, not an SC++ hybrid: candidates come
    from the supplied NSD embedding and from NSD-owned layout row keys; reranking
    uses cyclic shifts on NSD's projected layouts only.
    """
    normed = _normalize(embeddings)
    range_layouts = range_layouts.astype(np.float32)
    bev_layouts = bev_layouts.astype(np.float32)
    range_keys = _layout_row_keys(range_layouts)
    bev_keys = _layout_row_keys(bev_layouts)
    queries = _find_queries(poses, distance_threshold, skip_frames)
    ranked = {(wb, wr): [] for wb in dual_bev_weights for wr in dual_range_weights}

    for query_idx, _ in queries:
        embedding_candidates = _topk_cosine(normed, query_idx, n_coarse, skip_frames)
        range_candidates = _topk_cosine(range_keys, query_idx, n_coarse, skip_frames)
        bev_candidates = _topk_cosine(bev_keys, query_idx, n_coarse, skip_frames)
        candidates = np.unique(np.concatenate([
            embedding_candidates,
            range_candidates,
            bev_candidates,
        ]))

        embedding_distances = 1.0 - (normed[candidates] @ normed[query_idx])
        bev_distances = np.asarray([
            _distance_columnwise(bev_layouts[query_idx], bev_layouts[int(c)])
            for c in candidates
        ], dtype=np.float32)
        range_distances = np.asarray([
            _distance_columnwise(range_layouts[query_idx], range_layouts[int(c)])
            for c in candidates
        ], dtype=np.float32)
        base_score = _minmax01(embedding_distances)
        bev_score = _minmax01(bev_distances)
        range_score = _minmax01(range_distances)

        for bev_weight in dual_bev_weights:
            for range_weight in dual_range_weights:
                fused_score = base_score + bev_weight * bev_score + range_weight * range_score
                fused = candidates[np.argsort(fused_score)]
                ranked[(bev_weight, range_weight)].append(fused[:max(k_values)])

    results: Dict[str, Dict[str, float]] = {}
    for (bev_weight, range_weight), ranked_lists in ranked.items():
        key = f"dual_layout_score_fusion_bev{bev_weight:g}_range{range_weight:g}"
        results[key] = _score(
            poses,
            queries,
            ranked_lists,
            k_values,
            distance_threshold,
        )
    return results


def _phase_sketch(layouts: np.ndarray, n_freqs: int) -> np.ndarray:
    """Low-frequency complex phase sketch along azimuth columns.

    A yaw shift is a phase rotation of each retained Fourier coefficient. The
    sketch keeps only the first non-DC frequencies, so storage is
    rows * n_freqs * 2 floats instead of rows * sectors layout values.
    """
    if n_freqs <= 0:
        raise ValueError("n_freqs must be positive")
    if layouts.ndim != 3:
        raise ValueError(f"Expected layout tensor (N, rows, sectors), got {layouts.shape}")
    coeffs = np.fft.rfft(layouts.astype(np.float32), axis=2)
    max_freqs = coeffs.shape[2] - 1
    if n_freqs > max_freqs:
        raise ValueError(f"n_freqs={n_freqs} exceeds available non-DC frequencies {max_freqs}")
    return coeffs[:, :, 1 : n_freqs + 1].astype(np.complex64)


def _phase_sketch_keys(sketch: np.ndarray) -> np.ndarray:
    """Rotation-invariant keys derived from compact phase-sketch magnitudes."""
    return _normalize(np.abs(sketch).reshape(sketch.shape[0], -1).astype(np.float32))


def _phase_sketch_distances(
    query_sketch: np.ndarray,
    candidate_sketches: np.ndarray,
    n_sectors: int,
) -> np.ndarray:
    """Minimum cyclic-shift cosine distance in compact Fourier-sketch space."""
    if candidate_sketches.size == 0:
        return np.empty((0,), dtype=np.float32)
    n_freqs = query_sketch.shape[1]
    freqs = np.arange(1, n_freqs + 1, dtype=np.float32)
    shifts = np.arange(n_sectors, dtype=np.float32)
    phase = np.exp(-2j * np.pi * shifts[:, None] * freqs[None, :] / float(n_sectors)).astype(
        np.complex64
    )
    corr = np.einsum(
        "rk,nrk,sk->ns",
        np.conj(query_sketch),
        candidate_sketches,
        phase,
        optimize=True,
    ).real
    q_norm = np.linalg.norm(query_sketch.reshape(-1))
    c_norm = np.linalg.norm(candidate_sketches.reshape(candidate_sketches.shape[0], -1), axis=1)
    denom = np.maximum(q_norm * c_norm[:, None], 1e-8)
    sims = corr / denom
    return (1.0 - sims.max(axis=1)).astype(np.float32)


def _phase_sketch_distances_fft(
    query_sketch: np.ndarray,
    candidate_sketches: np.ndarray,
    n_sectors: int,
) -> np.ndarray:
    """Math-equivalent FFT acceleration of :func:`_phase_sketch_distances`.

    The cyclic-shift einsum

        corr[n, s] = Σ_{r,k} conj(F_q[r,k]) · F_c[n,r,k] · exp(-2πi·k·s/S)

    factors as

        cross[n, k] = Σ_r conj(F_q[r,k]) · F_c[n,r,k]      # complex (N, K)
        corr[n, s]  = Σ_k cross[n, k] · exp(-2πi·k·s/S)    # length-S DFT

    Zero-padding ``cross`` to length ``S`` (with the DC bin = 0 since the
    sketch excludes DC) turns the inner summation into a single
    ``np.fft.fft`` of length ``S``. Output is bit-equivalent to the
    legacy ``_phase_sketch_distances`` modulo float rounding.

    Complexity: ``O(N·R·K + N·S log S)`` versus the explicit
    ``O(N·S·R·K)`` einsum. Typical KITTI shape (N=800, S=360, R=16, K=8)
    yields ~14× fewer ops without any change to the score function.
    """
    if candidate_sketches.size == 0:
        return np.empty((0,), dtype=np.float32)
    n_freqs = query_sketch.shape[1]
    if n_sectors < n_freqs + 1:
        raise ValueError(
            f"n_sectors={n_sectors} must be >= n_freqs+1={n_freqs + 1} for shift sweep"
        )

    # Row-summed cross spectrum (small contraction first).
    cross = np.einsum(
        "rk,nrk->nk",
        np.conj(query_sketch),
        candidate_sketches,
        optimize=True,
    )                                                           # (N, K) complex

    # Zero-pad to length S along the frequency axis. The sketch starts at
    # k=1 (DC excluded), so place cross at indices 1..n_freqs.
    padded = np.zeros((cross.shape[0], n_sectors), dtype=np.complex64)
    padded[:, 1 : n_freqs + 1] = cross.astype(np.complex64)

    # Length-S DFT along last axis.
    # np.fft.fft uses the convention X[s] = Σ_k x[k] · exp(-2πi·k·s/S),
    # which matches the explicit phase factor exactly.
    corr = np.fft.fft(padded, axis=-1).real                     # (N, S)

    q_norm = np.linalg.norm(query_sketch.reshape(-1))
    c_norm = np.linalg.norm(
        candidate_sketches.reshape(candidate_sketches.shape[0], -1), axis=1
    )
    denom = np.maximum(q_norm * c_norm[:, None], 1e-8)
    sims = corr / denom
    return (1.0 - sims.max(axis=1)).astype(np.float32)


def _phase_coherence_distances(
    query_sketch: np.ndarray,
    candidate_sketches: np.ndarray,
    pad_factor: int = 4,
    mode: str = "poc",
) -> np.ndarray:
    """Phase-only correlation distance via row-wise IFFT peak.

    Drop-in replacement for :func:`_phase_sketch_distances` that exploits the
    closed-form coherence used inside ``ClosedFormPhaseEdgeBias``. Compared to
    the cyclic-shift einsum, this scales as ``N · rows · L log L`` instead of
    ``N · sectors · rows · freqs`` and is typically 10-20× faster on CPU.

    Args:
        query_sketch: ``(rows, freqs)`` complex Fourier coefficients.
        candidate_sketches: ``(N, rows, freqs)`` complex Fourier coefficients.
        pad_factor: zero-pad multiple for IFFT peak resolution.
        mode: ``"poc"`` (phase-only) or ``"ncc"`` (energy-weighted).

    Returns:
        Real ``(N,)`` distances ``∈ [0, 1]`` (``1 − peak coherence``).
    """
    if candidate_sketches.size == 0:
        return np.empty((0,), dtype=np.float32)
    eps = 1e-6
    cross = candidate_sketches * np.conj(query_sketch)[None, ...]      # (N, rows, K)

    if mode == "poc":
        cross = cross / (np.abs(cross) + eps)
    elif mode != "ncc":
        raise ValueError(f"Unknown mode {mode!r}; expected 'poc' or 'ncc'.")

    K = cross.shape[-1]
    L = max(K * max(int(pad_factor), 1), K)
    if L > K:
        pad_width = [(0, 0)] * (cross.ndim - 1) + [(0, L - K)]
        cross = np.pad(cross, pad_width)
    corr = np.fft.ifft(cross, axis=-1)                                  # (N, rows, L)
    peak = np.abs(corr).max(axis=-1)                                    # (N, rows)

    if mode == "poc":
        peak = peak * (L / max(K, 1))
        sim = np.clip(peak, 0.0, 1.0)
    else:  # ncc
        norm_q = np.linalg.norm(query_sketch, axis=-1)                  # (rows,)
        norm_c = np.linalg.norm(candidate_sketches, axis=-1)            # (N, rows)
        sim = (peak * L) / (norm_q[None, :] * norm_c + eps)
        sim = np.clip(sim, 0.0, 1.0)

    return (1.0 - sim.mean(axis=-1)).astype(np.float32)                 # (N,)


def _recall_with_phase_sketch_fusion(
    embeddings: np.ndarray,
    range_layouts: np.ndarray,
    bev_layouts: np.ndarray,
    poses: np.ndarray,
    k_values: List[int],
    distance_threshold: float,
    skip_frames: int,
    n_coarse: int,
    range_freqs: int,
    bev_freqs: int,
    n_sectors: int,
    sketch_bev_weights: List[float],
    sketch_range_weights: List[float],
    rerank_mode: str = "sketch",
    coherence_pad_factor: int = 4,
    coherence_score_mode: str = "poc",
) -> Dict[str, Dict[str, float]]:
    """Evaluate embeddings with <=512D NSD compact phase sketches.

    Args:
        rerank_mode: ``"sketch"`` (legacy cyclic-shift cosine),
            ``"sketch_fft"`` (the same score evaluated by FFT), or
            ``"coherence"`` (POC-peak ablation; faster/closed-form but less
            discriminative because it discards energy weighting and cross-row
            shift coherence).
        coherence_pad_factor / coherence_score_mode: forwarded to
            :func:`_phase_coherence_distances` when ``rerank_mode='coherence'``.
    """
    normed = _normalize(embeddings)
    if range_freqs <= 0 and bev_freqs <= 0:
        raise ValueError("At least one of range_freqs or bev_freqs must be positive")
    if rerank_mode not in ("sketch", "sketch_fft", "coherence"):
        raise ValueError(f"Unknown rerank_mode {rerank_mode!r}")
    range_sketch = _phase_sketch(range_layouts, range_freqs) if range_freqs > 0 else None
    bev_sketch = _phase_sketch(bev_layouts, bev_freqs) if bev_freqs > 0 else None
    range_keys = _phase_sketch_keys(range_sketch) if range_sketch is not None else None
    bev_keys = _phase_sketch_keys(bev_sketch) if bev_sketch is not None else None
    queries = _find_queries(poses, distance_threshold, skip_frames)
    ranked = {(wb, wr): [] for wb in sketch_bev_weights for wr in sketch_range_weights}

    def _distances(query_sk: np.ndarray, cand_sk: np.ndarray) -> np.ndarray:
        if rerank_mode == "coherence":
            return _phase_coherence_distances(
                query_sk, cand_sk,
                pad_factor=coherence_pad_factor,
                mode=coherence_score_mode,
            )
        if rerank_mode == "sketch_fft":
            return _phase_sketch_distances_fft(query_sk, cand_sk, n_sectors=n_sectors)
        return _phase_sketch_distances(query_sk, cand_sk, n_sectors=n_sectors)

    for query_idx, _ in queries:
        embedding_candidates = _topk_cosine(normed, query_idx, n_coarse, skip_frames)
        candidate_sets = [embedding_candidates]
        if range_keys is not None:
            candidate_sets.append(_topk_cosine(range_keys, query_idx, n_coarse, skip_frames))
        if bev_keys is not None:
            candidate_sets.append(_topk_cosine(bev_keys, query_idx, n_coarse, skip_frames))
        candidates = np.unique(np.concatenate(candidate_sets))

        embedding_distances = 1.0 - (normed[candidates] @ normed[query_idx])
        if range_sketch is not None:
            range_distances = _distances(range_sketch[query_idx], range_sketch[candidates])
        else:
            range_distances = np.zeros_like(embedding_distances, dtype=np.float32)
        if bev_sketch is not None:
            bev_distances = _distances(bev_sketch[query_idx], bev_sketch[candidates])
        else:
            bev_distances = np.zeros_like(embedding_distances, dtype=np.float32)
        base_score = _minmax01(embedding_distances)
        range_score = _minmax01(range_distances)
        bev_score = _minmax01(bev_distances)

        for bev_weight in sketch_bev_weights:
            for range_weight in sketch_range_weights:
                fused_score = base_score + bev_weight * bev_score + range_weight * range_score
                fused = candidates[np.argsort(fused_score)]
                ranked[(bev_weight, range_weight)].append(fused[:max(k_values)])

    aux_dim = 0
    if range_sketch is not None:
        aux_dim += range_sketch.shape[1] * range_sketch.shape[2] * 2
    if bev_sketch is not None:
        aux_dim += bev_sketch.shape[1] * bev_sketch.shape[2] * 2
    results: Dict[str, Dict[str, float]] = {"_aux_dim": {"D": int(aux_dim)}}
    for (bev_weight, range_weight), ranked_lists in ranked.items():
        key = f"phase_sketch_fusion_bev{bev_weight:g}_range{range_weight:g}"
        results[key] = _score(
            poses,
            queries,
            ranked_lists,
            k_values,
            distance_threshold,
        )
    return results


def _phase_token_features(
    range_layouts: np.ndarray,
    bev_layouts: np.ndarray,
    range_freqs: int,
    bev_freqs: int,
    token_dim: int,
    method: str,
) -> tuple[np.ndarray, int]:
    """Compact sequence-local phase token from low-frequency sketch features.

    The full phase sketch stores complex coefficients for shift-aware reranking.
    This diagnostic token stores a much smaller yaw-invariant summary, allowing
    us to test whether 384D/512D hand-crafted aux can be replaced by a compact
    learned-token target. PCA is unsupervised and uses no place labels.
    """
    if token_dim <= 0:
        raise ValueError("token_dim must be positive")
    range_sketch = _phase_sketch(range_layouts, range_freqs)
    bev_sketch = _phase_sketch(bev_layouts, bev_freqs)

    def magnitude_features(sketch: np.ndarray) -> np.ndarray:
        return np.abs(sketch).reshape(sketch.shape[0], -1).astype(np.float32)

    def cross_features(sketch: np.ndarray) -> np.ndarray:
        pair = sketch[:, :-1, :] * np.conj(sketch[:, 1:, :])
        denom = (
            np.abs(sketch[:, :-1, :]) * np.abs(sketch[:, 1:, :])
            + 1e-6
        )
        coherence = (pair / denom).astype(np.complex64)
        return np.concatenate(
            [
                coherence.real.reshape(coherence.shape[0], -1),
                coherence.imag.reshape(coherence.shape[0], -1),
            ],
            axis=1,
        ).astype(np.float32)

    if method == "mag":
        parts = [magnitude_features(range_sketch), magnitude_features(bev_sketch)]
    elif method == "cross":
        parts = [cross_features(range_sketch), cross_features(bev_sketch)]
    elif method == "mag_cross":
        parts = [
            magnitude_features(range_sketch),
            magnitude_features(bev_sketch),
            cross_features(range_sketch),
            cross_features(bev_sketch),
        ]
    else:
        raise ValueError(f"Unknown phase token method: {method}")

    features = np.concatenate(parts, axis=1).astype(np.float32)
    source_dim = int(features.shape[1])
    features = np.sign(features) * np.log1p(np.abs(features))
    features = features - features.mean(axis=0, keepdims=True)
    features = features / np.maximum(features.std(axis=0, keepdims=True), 1e-6)

    if token_dim >= source_dim:
        token = features
    else:
        _, _, vt = np.linalg.svd(features, full_matrices=False)
        n_components = min(token_dim, vt.shape[0])
        token = features @ vt[:n_components].T
        if n_components < token_dim:
            token = np.pad(token, ((0, 0), (0, token_dim - n_components)))
    return _normalize(token.astype(np.float32)), source_dim


def _recall_with_phase_token_fusion(
    embeddings: np.ndarray,
    range_layouts: np.ndarray,
    bev_layouts: np.ndarray,
    poses: np.ndarray,
    k_values: List[int],
    distance_threshold: float,
    skip_frames: int,
    n_coarse: int,
    range_freqs: int,
    bev_freqs: int,
    token_dim: int,
    token_method: str,
    token_weights: List[float],
) -> Dict[str, Dict[str, float]]:
    """Evaluate embeddings with a compact PCA phase token.

    This is a fast proxy for a learned phase-token head: it keeps the retrieval
    interface descriptor-like (cosine token + GNN embedding) and reports the
    effective aux dimension explicitly.
    """
    normed = _normalize(embeddings)
    token, source_dim = _phase_token_features(
        range_layouts,
        bev_layouts,
        range_freqs=range_freqs,
        bev_freqs=bev_freqs,
        token_dim=token_dim,
        method=token_method,
    )
    queries = _find_queries(poses, distance_threshold, skip_frames)
    ranked = {w: [] for w in token_weights}

    for query_idx, _ in queries:
        embedding_candidates = _topk_cosine(normed, query_idx, n_coarse, skip_frames)
        token_candidates = _topk_cosine(token, query_idx, n_coarse, skip_frames)
        candidates = np.unique(np.concatenate([embedding_candidates, token_candidates]))

        embedding_distances = 1.0 - (normed[candidates] @ normed[query_idx])
        token_distances = 1.0 - (token[candidates] @ token[query_idx])
        base_score = _minmax01(embedding_distances)
        token_score = _minmax01(token_distances)

        for weight in token_weights:
            fused_score = base_score + weight * token_score
            fused = candidates[np.argsort(fused_score)]
            ranked[weight].append(fused[:max(k_values)])

    results: Dict[str, Dict[str, float]] = {
        "_aux_dim": {
            "D": int(token_dim),
            "source_D": int(source_dim),
            "method": f"pca_{token_method}",
        }
    }
    for weight, ranked_lists in ranked.items():
        key = f"phase_token_fusion_w{weight:g}"
        results[key] = _score(
            poses,
            queries,
            ranked_lists,
            k_values,
            distance_threshold,
        )
    return results


def _build_eval_graph(
    keyframes: List[Keyframe],
    poses: np.ndarray,
    descriptors: np.ndarray,
    cache: np.lib.npyio.NpzFile,
    config: Dict,
    device: str,
    temporal_edge_mode: str,
    temporal_direction_mode: str,
    similarity_min_k: int,
    phase_features: np.ndarray | None = None,
    sensor_key: str = "kitti",
):
    graph_cfg = config["keyframe"].get("graph", {})
    similarity_max_k = graph_cfg.get("similarity_max_k", 10)
    if graph_cfg.get("sensor_similarity", {}).get("enabled", False):
        entry = graph_cfg["sensor_similarity"].get(sensor_key, {})
        similarity_max_k = int(entry.get("max_k", similarity_max_k))
        similarity_min_k = int(entry.get("min_k", similarity_min_k))
    graph = build_graph_from_keyframes_batch(
        keyframes,
        temporal_neighbors=config["keyframe"].get("temporal_neighbors", 10),
        device=device,
        poses=poses,
        descriptors=descriptors,
        similarity_threshold=graph_cfg.get("similarity_threshold", 0.993),
        similarity_max_k=similarity_max_k,
        similarity_min_k=similarity_min_k,
        similarity_exclude_temporal=graph_cfg.get("similarity_exclude_temporal", True),
        similarity_dist=None,
        similarity_metric=graph_cfg.get("similarity_metric", "cosine"),
        temporal_edge_mode=temporal_edge_mode,
        temporal_direction_mode=temporal_direction_mode,
    )
    if "fft_magnitudes" in cache.files:
        fft = cache["fft_magnitudes"].astype(np.float32)
        graph.x_fft = torch.from_numpy(fft.reshape(len(fft), -1)).float().to(device)
    if phase_features is not None:
        graph.x_phase = torch.from_numpy(phase_features.astype(np.float32)).float().to(device)
    sensor_id_by_key = {"kitti": 0, "nclt": 1, "helipr": 2, "mulran": 3}
    beam_count_by_key = {"kitti": 64.0, "mulran": 64.0, "nclt": 32.0, "helipr": 16.0}
    sensor_id = sensor_id_by_key.get(sensor_key, len(sensor_id_by_key))
    beam_count = beam_count_by_key.get(sensor_key, 64.0)
    graph.sensor_id = torch.full((len(keyframes),), sensor_id, dtype=torch.long, device=device)
    graph.beam_count = torch.full((len(keyframes), 1), beam_count, dtype=torch.float32, device=device)
    return graph


def _average_embeddings(branches: List[np.ndarray]) -> np.ndarray:
    normed = [_normalize(emb) for emb in branches]
    return _normalize(np.mean(np.stack(normed, axis=0), axis=0))


def _scale_context_embeddings(
    embeddings: np.ndarray,
    raw_dim: int,
    ctx_weight: float,
    context_dim: int | None = None,
) -> np.ndarray:
    """Scale context half of cat(raw, ctx) before retrieval."""
    if ctx_weight == 1.0:
        return embeddings
    scaled = embeddings.copy()
    ctx_end = raw_dim + context_dim if context_dim is not None else embeddings.shape[1]
    scaled[:, raw_dim:ctx_end] *= float(ctx_weight)
    return scaled


def evaluate_cache(
    cache_path: Path,
    config: Dict,
    checkpoint_path: Path,
    device: str,
    k_values: List[int],
    distance_threshold: float,
    skip_frames: int,
    temporal_edge_mode: str,
    temporal_direction_mode: str,
    similarity_min_k: int,
    causal_twin: bool,
    n_coarse: int,
    layout_score_weights: List[float],
    adaptive_gap_scales: List[float],
    adaptive_low_weight: float,
    adaptive_high_weight: float,
    bev_layouts: np.ndarray | None = None,
    dual_bev_weights: List[float] | None = None,
    dual_range_weights: List[float] | None = None,
    enable_phase_sketch: bool = False,
    phase_sketch_only: bool = False,
    phase_range_freqs: int = 8,
    phase_bev_freqs: int = 8,
    phase_sketch_bev_weights: List[float] | None = None,
    phase_sketch_range_weights: List[float] | None = None,
    phase_rerank_mode: str = "sketch",
    coherence_pad_factor: int = 4,
    coherence_score_mode: str = "poc",
    enable_phase_token: bool = False,
    phase_token_dim: int = 64,
    phase_token_method: str = "mag_cross",
    phase_token_weights: List[float] | None = None,
    layout_sectors: int = 60,
    skip_checkpoint: bool = False,
    ctx_weights: List[float] | None = None,
    sensor_key: str = "kitti",
) -> Dict:
    cache = np.load(cache_path)
    keyframes = _cache_to_keyframes(cache)
    poses = cache["poses"]
    descriptors = cache["descriptors"].astype(np.float32)
    phase_features = None
    phase_token_cfg = config.get("gnn", {}).get("phase_token", {})
    phase_edge_cfg = config.get("gnn", {}).get("phase_edge", {})
    phase_alignment_cfg = config.get("gnn", {}).get("phase_alignment_edge", {})
    phase_coherence_cfg = config.get("gnn", {}).get("phase_coherence", {})
    dual_stream_cfg = config.get("gnn", {}).get("dual_stream", {})
    need_phase_features = (
        phase_token_cfg.get("enabled", False)
        or phase_edge_cfg.get("enabled", False)
        or phase_alignment_cfg.get("enabled", False)
        or phase_coherence_cfg.get("enabled", False)
        or dual_stream_cfg.get("enabled", False)
    )
    if need_phase_features:
        from encoding.phase_features import (
            phase_feature_dim,
            phase_features_from_layouts,
            prepare_raw_complex_phase_config,
        )

        if "nsd_layouts" not in cache.files:
            raise RuntimeError("phase features enabled but cache has no nsd_layouts")
        phase_cfg = config.get("encoding", {}).get("phase_features", {})
        raw_complex_consumers = {
            "phase_alignment_edge": phase_alignment_cfg,
            "phase_coherence": phase_coherence_cfg,
            "dual_stream": dual_stream_cfg,
        }
        if any(cfg.get("enabled", False) for cfg in raw_complex_consumers.values()):
            phase_cfg = prepare_raw_complex_phase_config(
                phase_cfg,
                raw_complex_consumers,
            )
            config.setdefault("encoding", {})["phase_features"] = phase_cfg
            computed_phase_dim = phase_feature_dim(phase_cfg)
            if phase_alignment_cfg.get("enabled", False):
                phase_alignment_cfg["n_rows"] = int(
                    phase_alignment_cfg.get(
                        "n_rows",
                        phase_cfg.get("bev_rows", phase_cfg.get("range_rows", 16)),
                    )
                )
                phase_alignment_cfg["n_freqs"] = int(
                    phase_alignment_cfg.get(
                        "n_freqs",
                        phase_cfg.get("bev_freqs", phase_cfg.get("range_freqs", 0)),
                    )
                )
                phase_alignment_cfg["n_sectors"] = int(
                    phase_alignment_cfg.get("n_sectors", phase_cfg.get("layout_sectors", layout_sectors))
                )
                config["gnn"]["phase_alignment_edge"] = phase_alignment_cfg
            if phase_token_cfg.get("enabled", False):
                phase_token_cfg["input_dim"] = phase_token_cfg.get("input_dim", computed_phase_dim)
                config["gnn"]["phase_token"] = phase_token_cfg
            if phase_edge_cfg.get("enabled", False):
                phase_edge_cfg["input_dim"] = phase_edge_cfg.get("input_dim", computed_phase_dim)
                config["gnn"]["phase_edge"] = phase_edge_cfg
        if "bev" in phase_cfg.get("source", "bev_complex") and bev_layouts is None:
            raise RuntimeError(
                "BEV phase features enabled but bev_layouts is None; pass --enable-bev-layout"
            )
        nsd_layouts_for_phase = cache["nsd_layouts"].astype(np.float32)
        phase_features = np.asarray(
            [
                phase_features_from_layouts(nsd_layouts_for_phase[i], bev_layouts[i], phase_cfg)
                for i in range(len(nsd_layouts_for_phase))
            ],
            dtype=np.float32,
        )

    embeddings = None
    raw_dim = descriptors.shape[1]
    if not skip_checkpoint:
        model = _make_model(config, checkpoint_path, device)
        with torch.no_grad():
            if causal_twin:
                branches = []
                for mode in ("past_to_current", "future_to_current"):
                    graph = _build_eval_graph(
                        keyframes=keyframes,
                        poses=poses,
                        descriptors=descriptors,
                        cache=cache,
                        config=config,
                        device=device,
                        temporal_edge_mode=mode,
                        temporal_direction_mode=temporal_direction_mode,
                        similarity_min_k=similarity_min_k,
                        phase_features=phase_features,
                        sensor_key=sensor_key,
                    )
                    branches.append(model(graph.to(device)).detach().cpu().numpy())
                embeddings = _average_embeddings(branches)
            else:
                graph = _build_eval_graph(
                    keyframes=keyframes,
                    poses=poses,
                    descriptors=descriptors,
                    cache=cache,
                    config=config,
                    device=device,
                    temporal_edge_mode=temporal_edge_mode,
                    temporal_direction_mode=temporal_direction_mode,
                    similarity_min_k=similarity_min_k,
                    phase_features=phase_features,
                    sensor_key=sensor_key,
                )
                embeddings = model(graph.to(device)).detach().cpu().numpy()
        raw_dim = config["gnn"]["input_dim"]

    raw_metrics = _recall_cosine(descriptors, poses, k_values, distance_threshold, skip_frames)

    results = {
        "n_keyframes": int(len(keyframes)),
        "n_queries": int(len(_find_queries(poses, distance_threshold, skip_frames))),
        "descriptor_dim": int(raw_dim),
        "graph_policy": {
            "temporal_edge_mode": "causal_twin" if causal_twin else temporal_edge_mode,
            "temporal_direction_mode": temporal_direction_mode,
            "similarity_min_k": int(similarity_min_k),
        },
        "raw": raw_metrics,
    }
    if embeddings is not None:
        context_dim = int(config["gnn"].get("context_dim", embeddings.shape[1] - raw_dim))
        ctx_end = raw_dim + context_dim
        results["ctx"] = _recall_cosine(
            embeddings[:, raw_dim:ctx_end],
            poses,
            k_values,
            distance_threshold,
            skip_frames,
        )
        if embeddings.shape[1] > ctx_end:
            results["phase"] = _recall_cosine(
                embeddings[:, ctx_end:],
                poses,
                k_values,
                distance_threshold,
                skip_frames,
            )
        results["final"] = _recall_cosine(
            embeddings,
            poses,
            k_values,
            distance_threshold,
            skip_frames,
        )
        if ctx_weights:
            results["final_ctx_weight"] = {
                f"w{w:g}": _recall_cosine(
                    _scale_context_embeddings(embeddings, raw_dim, w, context_dim),
                    poses,
                    k_values,
                    distance_threshold,
                    skip_frames,
                )
                for w in ctx_weights
            }
    if "nsd_layouts" in cache.files:
        nsd_layouts = cache["nsd_layouts"].astype(np.float32)
        if not phase_sketch_only:
            results["raw_layout"] = _recall_with_layout_fusion(
                descriptors,
                nsd_layouts,
                poses,
                k_values,
                distance_threshold,
                skip_frames,
                n_coarse,
                layout_score_weights,
                adaptive_gap_scales,
                adaptive_low_weight,
                adaptive_high_weight,
            )
            if embeddings is not None:
                results["final_layout"] = _recall_with_layout_fusion(
                    embeddings,
                    nsd_layouts,
                    poses,
                    k_values,
                    distance_threshold,
                    skip_frames,
                    n_coarse,
                    layout_score_weights,
                    adaptive_gap_scales,
                    adaptive_low_weight,
                    adaptive_high_weight,
                )
        if bev_layouts is not None:
            dual_bev_weights = dual_bev_weights or [1.0, 2.0, 4.0]
            dual_range_weights = dual_range_weights or [0.0, 0.5, 1.0]
            if not phase_sketch_only:
                results["raw_dual_layout"] = _recall_with_dual_layout_fusion(
                    descriptors,
                    nsd_layouts,
                    bev_layouts,
                    poses,
                    k_values,
                    distance_threshold,
                    skip_frames,
                    n_coarse,
                    dual_bev_weights,
                    dual_range_weights,
                )
                if embeddings is not None:
                    results["final_dual_layout"] = _recall_with_dual_layout_fusion(
                        embeddings,
                        nsd_layouts,
                        bev_layouts,
                        poses,
                        k_values,
                        distance_threshold,
                        skip_frames,
                        n_coarse,
                        dual_bev_weights,
                        dual_range_weights,
                    )
            if enable_phase_sketch:
                phase_sketch_bev_weights = phase_sketch_bev_weights or [0.5, 1.0, 2.0]
                phase_sketch_range_weights = phase_sketch_range_weights or [0.0, 0.5, 1.0]
                _sketch_kwargs = dict(
                    rerank_mode=phase_rerank_mode,
                    coherence_pad_factor=coherence_pad_factor,
                    coherence_score_mode=coherence_score_mode,
                )
                results["raw_phase_sketch"] = _recall_with_phase_sketch_fusion(
                    descriptors,
                    nsd_layouts,
                    bev_layouts,
                    poses,
                    k_values,
                    distance_threshold,
                    skip_frames,
                    n_coarse,
                    phase_range_freqs,
                    phase_bev_freqs,
                    layout_sectors,
                    phase_sketch_bev_weights,
                    phase_sketch_range_weights,
                    **_sketch_kwargs,
                )
                if embeddings is not None:
                    results["final_phase_sketch"] = _recall_with_phase_sketch_fusion(
                        embeddings,
                        nsd_layouts,
                        bev_layouts,
                        poses,
                        k_values,
                        distance_threshold,
                        skip_frames,
                        n_coarse,
                        phase_range_freqs,
                        phase_bev_freqs,
                        layout_sectors,
                        phase_sketch_bev_weights,
                        phase_sketch_range_weights,
                        **_sketch_kwargs,
                    )
                    if ctx_weights:
                        results["final_phase_sketch_ctx_weight"] = {
                            f"w{w:g}": _recall_with_phase_sketch_fusion(
                                _scale_context_embeddings(embeddings, raw_dim, w, context_dim),
                                nsd_layouts,
                                bev_layouts,
                                poses,
                                k_values,
                                distance_threshold,
                                skip_frames,
                                n_coarse,
                                phase_range_freqs,
                                phase_bev_freqs,
                                layout_sectors,
                                phase_sketch_bev_weights,
                                phase_sketch_range_weights,
                                **_sketch_kwargs,
                            )
                            for w in ctx_weights
                        }
            if enable_phase_token:
                phase_token_weights = phase_token_weights or [0.25, 0.5, 1.0, 2.0, 4.0]
                results["raw_phase_token"] = _recall_with_phase_token_fusion(
                    descriptors,
                    nsd_layouts,
                    bev_layouts,
                    poses,
                    k_values,
                    distance_threshold,
                    skip_frames,
                    n_coarse,
                    phase_range_freqs,
                    phase_bev_freqs,
                    phase_token_dim,
                    phase_token_method,
                    phase_token_weights,
                )
                if embeddings is not None:
                    results["final_phase_token"] = _recall_with_phase_token_fusion(
                        embeddings,
                        nsd_layouts,
                        bev_layouts,
                        poses,
                        k_values,
                        distance_threshold,
                        skip_frames,
                        n_coarse,
                        phase_range_freqs,
                        phase_bev_freqs,
                        phase_token_dim,
                        phase_token_method,
                        phase_token_weights,
                    )
                    if ctx_weights:
                        results["final_phase_token_ctx_weight"] = {
                            f"w{w:g}": _recall_with_phase_token_fusion(
                                _scale_context_embeddings(embeddings, raw_dim, w, context_dim),
                                nsd_layouts,
                                bev_layouts,
                                poses,
                                k_values,
                                distance_threshold,
                                skip_frames,
                                n_coarse,
                                phase_range_freqs,
                                phase_bev_freqs,
                                phase_token_dim,
                                phase_token_method,
                                phase_token_weights,
                            )
                            for w in ctx_weights
                        }
    else:
        results["layout_warning"] = "cache has no nsd_layouts; rebuild with run_kitti_operating_point.py"
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_kitti_only.yaml")
    parser.add_argument(
        "--encoder-preset",
        default="full",
        choices=[
            "full",
            "no_interdiff",
            "cross4_no_interdiff",
            "cross8_no_interdiff",
            "mean_diff",
            "rows12_full",
        ],
        help="Closed-form encoder compression preset for fast raw/phase ablations",
    )
    parser.add_argument("--checkpoint", default="results/ctx128_cosine_bayesian/best_model.pth")
    parser.add_argument("--skip-checkpoint", action="store_true",
                        help="Evaluate encoder-only descriptors and optional phase sketch without loading GNN")
    parser.add_argument("--use-gated-context", action="store_true",
                        help="Build checkpoint model with learned context gate enabled")
    parser.add_argument("--gate-initial-alpha", type=float, default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--sequences", nargs="+", default=["00"])
    parser.add_argument("--cache-dir", default="data/preprocessed_kitti_operating")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--distance-threshold", type=float, default=5.0)
    parser.add_argument("--skip-frames", type=int, default=30)
    parser.add_argument("--temporal-edge-mode", default=None,
                        choices=["bidirectional", "past_to_current", "future_to_current"])
    parser.add_argument("--temporal-direction-mode", default=None,
                        choices=["none", "signed_distance"])
    parser.add_argument("--similarity-min-k", type=int, default=None)
    parser.add_argument("--causal-twin", action="store_true",
                        help="Average past-to-current and future-to-current GNN outputs")
    parser.add_argument("--n-coarse", type=int, default=800)
    parser.add_argument("--layout-sectors", type=int, default=60)
    parser.add_argument("--bev-cache-dir", default="data/preprocessed_kitti_bev_layout")
    parser.add_argument("--enable-bev-layout", action="store_true", default=True,
                        help="Evaluate true GNN+NSD BEV/range layout reranking (default: enabled for BEV-only 384D)")
    parser.add_argument("--bev-max-range", type=float, default=80.0)
    parser.add_argument("--bev-min-range", type=float, default=1.0)
    parser.add_argument("--bev-z-min", type=float, default=-3.0)
    parser.add_argument("--bev-z-max", type=float, default=5.0)
    parser.add_argument("--bev-height-layers", type=int, default=8)
    parser.add_argument("--bev-height-encoding", default="max", choices=["iris", "max", "physics3"])
    parser.add_argument("--bev-row-pool", type=int, default=16)
    parser.add_argument("--bev-row-pool-mode", default="max", choices=["max", "mean"])
    parser.add_argument("--dual-bev-weights", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--dual-range-weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--enable-phase-sketch", action="store_true", default=True,
                        help="Evaluate compact Fourier phase sketch aux (default: enabled for BEV-only 384D)")
    parser.add_argument("--phase-sketch-only", action="store_true",
                        help="Skip expensive full-layout rerank metrics when evaluating phase sketches")
    parser.add_argument("--phase-range-freqs", type=int, default=0)
    parser.add_argument("--phase-bev-freqs", type=int, default=12)
    parser.add_argument("--phase-sketch-bev-weights", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--phase-sketch-range-weights", nargs="+", type=float, default=[0.0])
    parser.add_argument("--phase-rerank-mode", default="sketch_fft",
                        choices=["sketch", "sketch_fft", "coherence"],
                        help="Phase-based rerank backend. "
                             "'sketch' = legacy cyclic-shift einsum (paper main); "
                             "'sketch_fft' = math-equivalent FFT acceleration (~14× fewer ops, "
                             "bit-equivalent score, recommended drop-in replacement); "
                             "'coherence' = POC peak (parameter-free but discriminatively weaker, "
                             "ablation-only — see KITTI 08 0.55 vs sketch 0.85).")
    parser.add_argument("--coherence-pad-factor", type=int, default=4,
                        help="Zero-pad multiple for the IFFT used in coherence mode.")
    parser.add_argument("--coherence-score-mode", default="poc", choices=["poc", "ncc"],
                        help="POC = phase-only (sharpest, default); NCC = energy-weighted.")
    parser.add_argument("--enable-phase-token", action="store_true",
                        help="Evaluate a compact PCA phase-token proxy for learned phase fusion")
    parser.add_argument("--phase-token-dim", type=int, default=64)
    parser.add_argument("--phase-token-method", default="mag_cross",
                        choices=["mag", "cross", "mag_cross"])
    parser.add_argument("--phase-token-weights", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--enable-learned-phase-token", action="store_true",
                        help="Attach x_phase and evaluate checkpoint with trainable phase token head")
    parser.add_argument("--learned-phase-source", default="bev_complex",
                        choices=["bev_complex", "bev_mag", "bev_cross", "bev_mag_cross",
                                 "range_complex", "range_mag", "range_cross", "range_mag_cross",
                                 "range_bev_complex", "range_bev_mag", "range_bev_cross",
                                 "range_bev_mag_cross"])
    parser.add_argument("--learned-phase-token-dim", type=int, default=64)
    parser.add_argument("--learned-phase-hidden-dim", type=int, default=128)
    parser.add_argument("--ctx-weights", nargs="+", type=float, default=[0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--layout-score-weights",
        nargs="+",
        type=float,
        default=[2.0, 3.2, 4.0],
    )
    parser.add_argument(
        "--adaptive-gap-scales",
        nargs="+",
        type=float,
        default=[0.05, 0.2, 0.8],
    )
    parser.add_argument("--adaptive-low-weight", type=float, default=0.2)
    parser.add_argument("--adaptive-high-weight", type=float, default=3.2)
    parser.add_argument("--output", default="results/kitti_checkpoint_eval.json")
    args = parser.parse_args()

    config = _apply_encoder_preset(_load_config(Path(args.config)), args.encoder_preset)
    if args.use_gated_context:
        config["gnn"]["use_residual_gate"] = True
        if args.gate_initial_alpha is not None:
            config["gnn"]["gate_initial_alpha"] = args.gate_initial_alpha
    if not args.enable_learned_phase_token:
        # Some training configs contain a phase_token block. Evaluation should
        # only instantiate that learned head when explicitly requested;
        # phase-sketch reranking is a separate non-parametric diagnostic.
        config.get("gnn", {}).pop("phase_token", None)
    if args.enable_learned_phase_token:
        from encoding.phase_features import phase_feature_dim

        phase_cfg = config.setdefault("encoding", {}).setdefault("phase_features", {})
        phase_bev_rows = args.bev_row_pool if args.bev_row_pool > 0 else 16
        phase_cfg.update({
            "source": args.learned_phase_source,
            "layout_sectors": args.layout_sectors,
            "bev_rows": phase_bev_rows,
            "range_rows": config["encoding"].get("target_elevation_bins", 16),
            "bev_freqs": args.phase_bev_freqs,
            "range_freqs": args.phase_range_freqs,
            "bev_height_encoding": args.bev_height_encoding,
            "bev_height_layers": args.bev_height_layers,
            "bev_min_range": args.bev_min_range,
            "bev_max_range": args.bev_max_range,
            "bev_z_min": args.bev_z_min,
            "bev_z_max": args.bev_z_max,
        })
        config["gnn"]["phase_token"] = {
            "enabled": True,
            "input_dim": phase_feature_dim(phase_cfg),
            "token_dim": args.learned_phase_token_dim,
            "hidden_dim": args.learned_phase_hidden_dim,
            "initial_alpha": config["gnn"].get("phase_token", {}).get("initial_alpha", 0.25),
            "dropout": config["gnn"].get("dropout", 0.1),
        }
        if args.bev_row_pool <= 0:
            args.bev_row_pool = phase_bev_rows
    phase_edge_cfg = config.get("gnn", {}).get("phase_edge", {})
    if phase_edge_cfg.get("enabled", False):
        from encoding.phase_features import phase_feature_dim

        phase_cfg = config.setdefault("encoding", {}).setdefault("phase_features", {})
        if args.bev_row_pool <= 0:
            args.bev_row_pool = int(phase_cfg.get("bev_rows", 16))
        phase_cfg["bev_rows"] = args.bev_row_pool
        phase_cfg["layout_sectors"] = args.layout_sectors
        phase_edge_cfg["input_dim"] = phase_edge_cfg.get(
            "input_dim",
            phase_feature_dim(phase_cfg),
        )
        config["gnn"]["phase_edge"] = phase_edge_cfg
    root = Path(args.root or config["data"]["datasets"]["val"][0]["root"])
    graph_cfg = config["keyframe"].get("graph", {})
    temporal_edge_mode = args.temporal_edge_mode or graph_cfg.get("temporal_edge_mode", "bidirectional")
    temporal_direction_mode = args.temporal_direction_mode or graph_cfg.get("temporal_direction_mode", "none")
    similarity_min_k = (
        args.similarity_min_k
        if args.similarity_min_k is not None
        else graph_cfg.get("similarity_min_k", 0)
    )

    results = {}
    for seq in args.sequences:
        cache_path = _build_sequence_cache(
            root=root,
            sequence=seq,
            config=config,
            cache_dir=Path(args.cache_dir),
            device=args.device,
            layout_sectors=args.layout_sectors,
        )
        bev_layouts = None
        if args.enable_bev_layout:
            base_cache = np.load(cache_path)
            bev_path = Path(args.bev_cache_dir) / (
                f"kitti_bev_layout_{seq}_s{args.layout_sectors}_"
                f"{args.bev_height_encoding}_r{args.bev_min_range:g}-{args.bev_max_range:g}_"
                f"z{args.bev_z_min:g}-{args.bev_z_max:g}_h{args.bev_height_layers}.npz"
            )
            bev_layouts = _build_bev_layout_cache(
                root=root,
                sequence=seq,
                base_cache=base_cache,
                output_path=bev_path,
                n_sectors=args.layout_sectors,
                max_range=args.bev_max_range,
                min_range=args.bev_min_range,
                z_min=args.bev_z_min,
                z_max=args.bev_z_max,
                n_height_layers=args.bev_height_layers,
                height_encoding=args.bev_height_encoding,
            )
            bev_layouts = _pool_rows(
                bev_layouts,
                args.bev_row_pool,
                args.bev_row_pool_mode,
                n_channels=3 if args.bev_height_encoding == "physics3" else 1,
            )
        results[seq] = evaluate_cache(
            cache_path=cache_path,
            config=config,
            checkpoint_path=Path(args.checkpoint),
            device=args.device,
            k_values=args.k_values,
            distance_threshold=args.distance_threshold,
            skip_frames=args.skip_frames,
            temporal_edge_mode=temporal_edge_mode,
            temporal_direction_mode=temporal_direction_mode,
            similarity_min_k=similarity_min_k,
            causal_twin=args.causal_twin,
            n_coarse=args.n_coarse,
            layout_score_weights=args.layout_score_weights,
            adaptive_gap_scales=args.adaptive_gap_scales,
            adaptive_low_weight=args.adaptive_low_weight,
            adaptive_high_weight=args.adaptive_high_weight,
            bev_layouts=bev_layouts,
            dual_bev_weights=args.dual_bev_weights,
            dual_range_weights=args.dual_range_weights,
            enable_phase_sketch=args.enable_phase_sketch,
            phase_sketch_only=args.phase_sketch_only,
            phase_range_freqs=args.phase_range_freqs,
            phase_bev_freqs=args.phase_bev_freqs,
            phase_sketch_bev_weights=args.phase_sketch_bev_weights,
            phase_sketch_range_weights=args.phase_sketch_range_weights,
            phase_rerank_mode=args.phase_rerank_mode,
            coherence_pad_factor=args.coherence_pad_factor,
            coherence_score_mode=args.coherence_score_mode,
            enable_phase_token=args.enable_phase_token,
            phase_token_dim=args.phase_token_dim,
            phase_token_method=args.phase_token_method,
            phase_token_weights=args.phase_token_weights,
            layout_sectors=args.layout_sectors,
            skip_checkpoint=args.skip_checkpoint,
            ctx_weights=args.ctx_weights,
        )
        print(seq, json.dumps(results[seq], indent=2))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
