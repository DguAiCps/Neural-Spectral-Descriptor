"""Cyclic column-shift distances for polar LiDAR layouts.

This module provides a shared geometric primitive used by rotation-robust
LiDAR place-recognition methods such as SC++, DiSCO, RING#, and NSD auxiliary
layout ablations. It is not an SC++-specific model component.
"""

from __future__ import annotations

import numpy as np


def cyclic_column_cosine_distance(mat1: np.ndarray, mat2: np.ndarray, eps: float = 1e-8) -> float:
    """Minimum column-shift cosine distance between two polar layout matrices.

    For every cyclic shift tau, columns of ``mat2`` are shifted and compared to
    ``mat1`` by per-column cosine distance. Empty columns are ignored, and the
    minimum average distance over shifts is returned.
    """
    if mat1.ndim != 2 or mat2.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got {mat1.shape} and {mat2.shape}")
    if mat1.shape != mat2.shape:
        raise ValueError(f"Shape mismatch: {mat1.shape} vs {mat2.shape}")

    _, n_sectors = mat1.shape
    norm_q = np.linalg.norm(mat1, axis=0)
    valid_q = norm_q > eps
    if not valid_q.any():
        return 1.0

    idx = (np.arange(n_sectors)[None, :] - np.arange(n_sectors)[:, None]) % n_sectors
    shifted = mat2[:, idx].transpose(1, 0, 2)

    dots = (mat1[None] * shifted).sum(axis=1)
    norms = np.linalg.norm(shifted, axis=1)
    valid = valid_q[None, :] & (norms > eps)

    sim = dots / (norm_q[None, :] * norms + eps)
    distances = np.where(valid, 1.0 - sim, 0.0)
    counts = np.maximum(valid.sum(axis=1).astype(np.float64), 1.0)
    per_shift = distances.sum(axis=1) / counts
    return float(per_shift.min())
