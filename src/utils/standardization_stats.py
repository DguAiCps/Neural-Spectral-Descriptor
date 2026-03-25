"""
Standardization Statistics for Diagonal Mahalanobis Distance

Computes per-dimension mean and std from training descriptors (unnormalized).
Used to z-score descriptors before L2 distance computation in FAISS.

Standardized Euclidean distance = L2 on z-scored data = diagonal Mahalanobis:
    d(a, b) = sqrt(sum_i ((a_i - b_i) / sigma_i)^2)

Reference: De Maesschalck et al. (2000) "The Mahalanobis Distance"
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class StandardizationStats:
    """Per-dimension mean and std for standardized Euclidean distance."""

    def __init__(self, epsilon: float = 1e-8):
        self.mean: np.ndarray = None  # (D,)
        self.std: np.ndarray = None   # (D,)
        self.fitted: bool = False
        self.epsilon = epsilon

    def fit(self, descriptors: np.ndarray) -> 'StandardizationStats':
        """Compute per-dimension mean and std from training descriptors.

        Args:
            descriptors: (N, D) unnormalized training descriptors

        Returns:
            self (for chaining)
        """
        self.mean = descriptors.mean(axis=0).astype(np.float32)
        raw_std = descriptors.std(axis=0).astype(np.float32)

        # Clamp near-zero std to prevent division by zero
        n_clamped = int((raw_std < self.epsilon).sum())
        self.std = np.maximum(raw_std, self.epsilon)
        self.fitted = True

        if n_clamped > 0:
            logger.warning(
                f"StandardizationStats: {n_clamped}/{len(self.std)} dimensions "
                f"have near-zero std (clamped to {self.epsilon})"
            )
        logger.info(
            f"StandardizationStats fitted: D={len(self.mean)}, "
            f"mean range=[{self.mean.min():.4f}, {self.mean.max():.4f}], "
            f"std range=[{self.std.min():.6f}, {self.std.max():.4f}]"
        )
        return self

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        """Z-score descriptors: (x - mu) / sigma.

        Args:
            descriptors: (N, D) or (D,) raw descriptors

        Returns:
            Same shape, standardized descriptors (float32)
        """
        if not self.fitted:
            raise RuntimeError("Not fitted. Call fit() first.")
        return ((descriptors - self.mean) / self.std).astype(np.float32)

    def save(self, path: str) -> None:
        """Save statistics to .npz file."""
        np.savez(path, mean=self.mean, std=self.std, fitted=self.fitted)
        logger.info(f"Saved standardization stats to {path}")

    def load(self, path: str) -> 'StandardizationStats':
        """Load statistics from .npz file."""
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']
        self.fitted = bool(data['fitted'])
        logger.info(f"Loaded standardization stats from {path} (D={len(self.mean)})")
        return self
