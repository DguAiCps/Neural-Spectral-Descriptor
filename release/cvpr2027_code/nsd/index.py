"""Place indexing (Sec. 3.1): the remetrised key r, the auxiliary magnitude index m and candidate retrieval.

Each place stores two unit vectors used only for proposing candidates:

    r = normalize(omega * d)      288-D, d is the invariant spectral key, omega a fixed positive remetric weight
    m = normalize(|z|)            128-D, magnitudes of the *unrefined* 16x8 complex DFT features z

For a query keyframe q, candidates are restricted to the eligible past j <= q - 30. Each index returns its
top-100 by inner product (exact brute-force search over the past prefix), and the two lists are merged
into a sorted union of at most 200 ids (duplicates are not back-filled).

All arithmetic is float32 numpy. The search touches only rows [0, q - 29), so the result for q does not
depend on later keyframes.
"""
from __future__ import annotations

import numpy as np

MIN_GAP = 30          # candidates j satisfy j <= q - MIN_GAP
TOP_K = 100           # per-index proposal budget


def remetric_key(d: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """r = omega * d / ||omega * d||  (float32, denominator floored at 1e-12)."""
    r = d.astype(np.float32) * omega.astype(np.float32)[None]
    return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)


def aux_magnitude_index(z: np.ndarray) -> np.ndarray:
    """m = |z| flattened to 128-D and L2-normalised (float32, denominator floored at 1e-12)."""
    m = np.abs(z).reshape(len(z), -1).astype(np.float32)
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)


def prefix_topk(bank: np.ndarray, q: int, n_prefix: int, k: int) -> np.ndarray:
    """Exact top-k of bank[:n_prefix] by inner product with bank[q], best first.

    Ordering of equal scores follows argpartition followed by a stable sort.
    """
    score = bank[:n_prefix] @ bank[q]
    k = min(k, n_prefix)
    top = np.argpartition(-score, k - 1)[:k] if k < n_prefix else np.arange(n_prefix)
    return top[np.argsort(-score[top], kind="stable")]


def candidates(r: np.ndarray, m: np.ndarray, q: int, k: int = TOP_K) -> np.ndarray:
    """Sorted union of the top-k past candidates of the two indices for query q."""
    n_prefix = q - MIN_GAP + 1
    return np.unique(np.r_[prefix_topk(r, q, n_prefix, k), prefix_topk(m, q, n_prefix, k)])
