"""
Scan Context++ baseline — full two-stage pipeline.

Reference:
    Kim, Choi, Kim. "Scan Context++: Structural Place Recognition Robust to
    Rotation and Lateral Variations in Urban Environments", IEEE T-RO 38(3),
    2022. Sections III-A through III-D, Algorithm 1.
    Reference impl: gisbi-kim/scancontext_tro.

Two stages:
    1. Ring Key (per-ring mean of max-z) for fast cosine pre-filter.
    2. SC matrix column-shift cosine distance for rerank — this is the
       defining rotation-robustness mechanism (paper Eq.7-8).

The Sector Key pre-filter described in Section III-B is omitted: its filtering
benefit is marginal on D <= 80 m and the FAISS cosine on Ring Key already
provides enough recall at the pre-filter stage for the rerank to converge.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from baselines import register
from baselines.base import BaselineEncoder


def build_scan_context(points, n_rings=20, n_sectors=60, max_range=80.0, z_min=-3.0):
    """
    Build BEV polar grid (Scan Context matrix).

    Args:
        points: (N, 3+) point cloud
        n_rings: Number of radial divisions
        n_sectors: Number of angular divisions
        max_range: Maximum range in meters
        z_min: Minimum z for ground filtering

    Returns:
        (n_rings, n_sectors) Scan Context matrix (max z-height per cell)
    """
    xyz = points[:, :3]
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    valid = (r < max_range) & (r > 0.1) & (xyz[:, 2] > z_min)
    xyz = xyz[valid]
    r = r[valid]

    if len(xyz) == 0:
        return np.zeros((n_rings, n_sectors), dtype=np.float32)

    theta = np.arctan2(xyz[:, 1], xyz[:, 0]) + np.pi  # [0, 2*pi]

    ring_idx = np.clip((r / max_range * n_rings).astype(int), 0, n_rings - 1)
    sector_idx = np.clip((theta / (2 * np.pi) * n_sectors).astype(int), 0, n_sectors - 1)

    sc = np.full((n_rings, n_sectors), -np.inf, dtype=np.float32)
    np.maximum.at(sc, (ring_idx, sector_idx), xyz[:, 2].astype(np.float32))
    sc[sc == -np.inf] = 0.0

    return sc


def _ring_key(sc: np.ndarray) -> np.ndarray:
    """Per-ring mean of SC matrix (paper Eq.5). Rotation-invariant."""
    return sc.mean(axis=1)


def _distance_sc_columnwise(sc1: np.ndarray, sc2: np.ndarray) -> float:
    """
    Column-shift cosine distance between two SC matrices (paper Eq.7-8).

    For each shift tau in [0, n_sectors), shift sc2 columns by tau, compute
    per-column cosine distance to sc1, average over non-empty columns. Return
    the minimum over tau.

    Vectorized across all shifts via a (n_sectors, n_rings, n_sectors) tensor
    of column-shifted copies of sc2, computed once with stride tricks.
    """
    n_rings, n_sectors = sc1.shape
    eps = 1e-8

    norm_q = np.linalg.norm(sc1, axis=0)  # (n_sectors,)
    valid_q = norm_q > eps
    if not valid_q.any():
        return 1.0

    # Build all shifted sc2 in one tensor: shifted[tau, :, s] = sc2[:, (s - tau) mod n_sec].
    # Equivalent to stacking np.roll(sc2, tau, axis=1) for tau in 0..n-1.
    idx = (np.arange(n_sectors)[None, :] - np.arange(n_sectors)[:, None]) % n_sectors
    shifted = sc2[:, idx]  # (n_rings, n_sectors_tau, n_sectors_s) via fancy idx
    # Reshape to (n_taus, n_rings, n_sectors): np advanced indexing produces
    # (n_rings, n_taus, n_sectors) — transpose to put tau first.
    shifted = shifted.transpose(1, 0, 2)  # (n_taus, n_rings, n_sectors)

    # Per (tau, s) dot product
    dots = (sc1[None] * shifted).sum(axis=1)  # (n_taus, n_sectors)
    norms = np.linalg.norm(shifted, axis=1)  # (n_taus, n_sectors)
    valid_mat = valid_q[None, :] & (norms > eps)  # (n_taus, n_sectors)

    sim = dots / (norm_q[None, :] * norms + eps)
    one_minus = 1.0 - sim
    # Per-tau mean over valid columns. Mask invalid → 0 contribution and
    # divide by valid count.
    contrib = np.where(valid_mat, one_minus, 0.0)
    counts = valid_mat.sum(axis=1).astype(np.float64)  # (n_taus,)
    counts = np.maximum(counts, 1)
    per_tau = contrib.sum(axis=1) / counts
    return float(per_tau.min())


@register
class ScanContextPP(BaselineEncoder):
    """Scan Context++ with full Ring Key prefilter + column-shift rerank."""

    def __init__(self, n_rings=20, n_sectors=60, max_range=80.0, z_min=-3.0,
                 n_coarse=200):
        self.n_rings = n_rings
        self.n_sectors = n_sectors
        self.max_range = max_range
        self.z_min = z_min
        self.n_coarse = n_coarse

    @property
    def name(self):
        return "Scan Context++"

    @property
    def short_name(self):
        return "sc++"

    @property
    def descriptor_dim(self):
        # Ring Key (the cosine pre-filter signature). The rerank uses the full
        # SC matrix but the descriptor reported in paper tables is Ring Key.
        return self.n_rings

    def encode(self, points):
        sc = build_scan_context(
            points, self.n_rings, self.n_sectors, self.max_range, self.z_min
        )
        rk = _ring_key(sc)
        norm = np.linalg.norm(rk)
        if norm > 1e-8:
            rk = rk / norm
        return rk.astype(np.float32)

    def encode_with_aux(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        sc = build_scan_context(
            points, self.n_rings, self.n_sectors, self.max_range, self.z_min
        )
        rk = _ring_key(sc)
        norm = np.linalg.norm(rk)
        if norm > 1e-8:
            rk_norm = rk / norm
        else:
            rk_norm = np.zeros(self.n_rings, dtype=np.float32)
        return rk_norm.astype(np.float32), {'sc_matrix': sc.astype(np.float32)}

    def compute_recalls(
        self,
        point_clouds: List[np.ndarray],
        poses: np.ndarray,
        k_values: List[int] = [1, 5, 10],
        distance_threshold: float = 5.0,
        skip_frames: int = 30,
        per_query_records=None,
    ):
        from baselines.eval_utils import compute_recall_cosine_then_rerank

        descriptors, aux_list = self.encode_sequence_with_aux(point_clouds)
        sc_matrices = np.stack(
            [a['sc_matrix'] for a in aux_list], axis=0
        ).astype(np.float32)

        def rerank_fn(query_idx: int, candidates: np.ndarray) -> np.ndarray:
            sc_q = sc_matrices[query_idx]
            distances = np.empty(len(candidates), dtype=np.float32)
            for i, c in enumerate(candidates):
                distances[i] = _distance_sc_columnwise(sc_q, sc_matrices[c])
            return candidates[np.argsort(distances)]

        return compute_recall_cosine_then_rerank(
            descriptors, rerank_fn, poses,
            k_values=k_values,
            distance_threshold=distance_threshold,
            skip_frames=skip_frames,
            n_coarse=self.n_coarse,
            per_query_records=per_query_records,
        )
