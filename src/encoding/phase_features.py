"""Compact phase-feature inputs for learned NSD phase tokens.

These features are inputs to a trainable projector, not final retrieval scores.
They keep low-frequency azimuth phase evidence that the closed-form magnitude
encoder discards.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from encoding.bev_image import BEVProjector, interpolate_bev_image
from encoding.range_image import interpolate_range_image

RAW_COMPLEX_PHASE_SOURCES = frozenset({"bev_complex", "range_complex"})


def _pool_rows(image: np.ndarray, n_rows: int, n_channels: int = 1) -> np.ndarray:
    if n_rows <= 0 or image.shape[0] == n_rows:
        return image.astype(np.float32)
    n_channels = max(1, int(n_channels))
    if image.shape[0] % n_channels != 0 or n_rows % n_channels != 0:
        raise ValueError(
            f"Cannot channel-pool rows={image.shape[0]} to n_rows={n_rows} "
            f"with n_channels={n_channels}"
        )
    if n_channels > 1:
        rows_per_channel = image.shape[0] // n_channels
        out_rows_per_channel = n_rows // n_channels
        pooled_channels = []
        for ch in range(n_channels):
            start = ch * rows_per_channel
            end = start + rows_per_channel
            pooled_channels.append(_pool_rows(image[start:end], out_rows_per_channel, n_channels=1))
        return np.concatenate(pooled_channels, axis=0).astype(np.float32)
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0),
        (n_rows, image.shape[1]),
    ).squeeze()
    return pooled.detach().cpu().numpy().astype(np.float32)


def _pool_columns(image: np.ndarray, n_cols: int) -> np.ndarray:
    if n_cols <= 0 or image.shape[1] == n_cols:
        return image.astype(np.float32)
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0),
        (image.shape[0], n_cols),
    ).squeeze()
    return pooled.detach().cpu().numpy().astype(np.float32)


def _phase_sketch(layout: np.ndarray, n_freqs: int) -> np.ndarray:
    if n_freqs <= 0:
        return np.empty((layout.shape[0], 0), dtype=np.complex64)
    coeffs = np.fft.rfft(layout.astype(np.float32), axis=1, norm="ortho")
    max_freqs = coeffs.shape[1] - 1
    if n_freqs > max_freqs:
        raise ValueError(f"n_freqs={n_freqs} exceeds available non-DC frequencies {max_freqs}")
    return coeffs[:, 1 : n_freqs + 1].astype(np.complex64)


def _complex_features(sketch: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [sketch.real.reshape(-1), sketch.imag.reshape(-1)]
    ).astype(np.float32)


def _magnitude_features(sketch: np.ndarray) -> np.ndarray:
    return np.abs(sketch).reshape(-1).astype(np.float32)


def _cross_features(sketch: np.ndarray) -> np.ndarray:
    if sketch.shape[0] < 2 or sketch.shape[1] == 0:
        return np.empty((0,), dtype=np.float32)
    pair = sketch[:-1, :] * np.conj(sketch[1:, :])
    denom = np.abs(sketch[:-1, :]) * np.abs(sketch[1:, :]) + 1e-6
    coherence = (pair / denom).astype(np.complex64)
    return np.concatenate(
        [coherence.real.reshape(-1), coherence.imag.reshape(-1)]
    ).astype(np.float32)


def phase_feature_dim(config: Dict) -> int:
    """Return feature dimension implied by phase feature config."""
    source = config.get("source", "bev_complex")
    bev_rows = int(config.get("bev_rows", 16))
    range_rows = int(config.get("range_rows", 16))
    bev_freqs = int(config.get("bev_freqs", 12))
    range_freqs = int(config.get("range_freqs", 0))

    def dim_for(prefix: str, rows: int, freqs: int) -> int:
        if freqs <= 0:
            return 0
        if prefix == "complex":
            return rows * freqs * 2
        if prefix == "mag":
            return rows * freqs
        if prefix == "cross":
            return max(rows - 1, 0) * freqs * 2
        if prefix == "mag_cross":
            return rows * freqs + max(rows - 1, 0) * freqs * 2
        raise ValueError(f"Unknown phase feature family: {prefix}")

    total = 0
    if source.startswith("range_bev_"):
        family = source[len("range_bev_"):]
        total += dim_for(family, range_rows, range_freqs)
        total += dim_for(family, bev_rows, bev_freqs)
    elif source.startswith("bev_"):
        total += dim_for(source[len("bev_"):], bev_rows, bev_freqs)
    elif source.startswith("range_"):
        total += dim_for(source[len("range_"):], range_rows, range_freqs)
    else:
        raise ValueError(f"Unknown phase feature source: {source}")
    return int(total)


def prepare_raw_complex_phase_config(
    config: Dict,
    consumer_configs: Dict[str, Dict],
) -> Dict:
    """Return a phase-feature config safe for complex phase consumers.

    ``ClosedFormPhaseEdgeBias`` and the dual-stream phase branch interpret
    ``x_phase`` as a flat complex grid ``[real || imag]`` with a single
    ``(n_rows, n_freqs)`` layout. Sources such as ``bev_cross`` already contain
    row-pair cross spectra and cannot be reinterpreted as raw Fourier
    coefficients. This helper enforces that contract and disables scalar
    log-compression, which would otherwise destroy phase geometry.
    """
    active = [
        (name, cfg)
        for name, cfg in consumer_configs.items()
        if cfg is not None and cfg.get("enabled", False)
    ]
    if not active:
        return dict(config)

    n_rows = int(active[0][1]["n_rows"])
    n_freqs = int(active[0][1]["n_freqs"])
    for name, cfg in active:
        rows = int(cfg["n_rows"])
        freqs = int(cfg["n_freqs"])
        if rows != n_rows or freqs != n_freqs:
            raise ValueError(
                "Raw complex phase consumers must agree on shape: "
                f"{active[0][0]}=({n_rows}, {n_freqs}), "
                f"{name}=({rows}, {freqs})"
            )

    if n_rows <= 0 or n_freqs <= 0:
        raise ValueError(f"Raw complex phase shape must be positive, got ({n_rows}, {n_freqs})")

    cfg = dict(config)
    source = cfg.get("source", "bev_complex")
    if source not in RAW_COMPLEX_PHASE_SOURCES:
        allowed = ", ".join(sorted(RAW_COMPLEX_PHASE_SOURCES))
        consumers = ", ".join(name for name, _ in active)
        raise ValueError(
            f"{consumers} requires raw complex phase features; got source={source!r}. "
            f"Set encoding.phase_features.source to one of: {allowed}."
        )

    if source == "bev_complex":
        row_key, freq_key = "bev_rows", "bev_freqs"
    else:
        row_key, freq_key = "range_rows", "range_freqs"

    if row_key in cfg and int(cfg[row_key]) != n_rows:
        raise ValueError(
            f"{source} {row_key}={cfg[row_key]} does not match raw complex consumer n_rows={n_rows}"
        )
    if freq_key in cfg and int(cfg[freq_key]) != n_freqs:
        raise ValueError(
            f"{source} {freq_key}={cfg[freq_key]} does not match raw complex consumer n_freqs={n_freqs}"
        )

    cfg["source"] = source
    cfg[row_key] = n_rows
    cfg[freq_key] = n_freqs
    cfg["apply_log_compression"] = False
    return cfg


def phase_features_from_layouts(
    range_layout: np.ndarray | None,
    bev_layout: np.ndarray | None,
    config: Dict,
) -> np.ndarray:
    """Build one phase-feature vector from projected range/BEV layouts."""
    source = config.get("source", "bev_complex")
    range_freqs = int(config.get("range_freqs", 0))
    bev_freqs = int(config.get("bev_freqs", 12))

    def features_for(layout: np.ndarray, n_freqs: int, family: str) -> np.ndarray:
        sketch = _phase_sketch(layout, n_freqs)
        if family == "complex":
            return _complex_features(sketch)
        if family == "mag":
            return _magnitude_features(sketch)
        if family == "cross":
            return _cross_features(sketch)
        if family == "mag_cross":
            return np.concatenate([_magnitude_features(sketch), _cross_features(sketch)])
        raise ValueError(f"Unknown phase feature family: {family}")

    parts = []
    if source.startswith("range_bev_"):
        family = source[len("range_bev_"):]
        if range_layout is None or bev_layout is None:
            raise ValueError("range_bev phase features need both range and BEV layouts")
        parts.append(features_for(range_layout, range_freqs, family))
        parts.append(features_for(bev_layout, bev_freqs, family))
    elif source.startswith("bev_"):
        if bev_layout is None:
            raise ValueError("BEV phase features requested but bev_layout is None")
        parts.append(features_for(bev_layout, bev_freqs, source[len("bev_"):]))
    elif source.startswith("range_"):
        if range_layout is None:
            raise ValueError("Range phase features requested but range_layout is None")
        parts.append(features_for(range_layout, range_freqs, source[len("range_"):]))
    else:
        raise ValueError(f"Unknown phase feature source: {source}")

    feats = np.concatenate(parts).astype(np.float32)
    if config.get("apply_log_compression", True):
        feats = np.sign(feats) * np.log1p(np.abs(feats))
    return feats


def compute_phase_features_from_points(points: np.ndarray, encoder, config: Dict) -> np.ndarray:
    """Project points and compute one phase-feature vector."""
    layout_sectors = int(config.get("layout_sectors", 60))
    range_rows = int(config.get("range_rows", getattr(encoder, "target_elevation_bins", 16)))
    bev_rows = int(config.get("bev_rows", 16))

    range_layout = None
    if config.get("source", "bev_complex").startswith(("range_", "range_bev_")):
        image_2d, _ = encoder.projector.project(points, keep_intensity=False)
        if encoder.interpolate_empty:
            image_2d = interpolate_range_image(image_2d, method="linear")
        range_layout = _pool_columns(_pool_rows(image_2d, range_rows), layout_sectors)

    bev_layout = None
    if "bev" in config.get("source", "bev_complex"):
        bev_projector = BEVProjector(
            n_sectors=layout_sectors,
            max_range=float(config.get("bev_max_range", 80.0)),
            min_range=float(config.get("bev_min_range", 1.0)),
            z_min=float(config.get("bev_z_min", -3.0)),
            z_max=float(config.get("bev_z_max", 5.0)),
            n_height_layers=int(config.get("bev_height_layers", 8)),
            height_encoding=config.get("bev_height_encoding", "max"),
        )
        bev, _ = bev_projector.project(points, keep_intensity=False)
        bev_layout = _pool_rows(
            interpolate_bev_image(
                bev,
                method="linear",
                n_channels=3 if config.get("bev_height_encoding", "max") == "physics3" else 1,
            ),
            bev_rows,
            n_channels=3 if config.get("bev_height_encoding", "max") == "physics3" else 1,
        )

    return phase_features_from_layouts(range_layout, bev_layout, config)
