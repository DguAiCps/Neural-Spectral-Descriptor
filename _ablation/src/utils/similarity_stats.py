"""
Bayesian Similarity Distribution for Statistical Edge Construction

Replaces arbitrary similarity thresholds with a principled Bayesian
posterior test:

    P(same_place | similarity) >= confidence_level

Supports two metrics:
  - 'cosine': Fisher z-transform → Gaussian fit (original pipeline)
  - 'l2': Standardized Euclidean distance → direct Gaussian fit
    (diagonal Mahalanobis; high-D L2 is ~Gaussian by CLT)

Reference: De Maesschalck et al. (2000) "The Mahalanobis Distance"

Density/entropy-adaptive prior adjusts P(same_place) per-node,
making the test stricter in aliasing-prone regions.
"""

import logging
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from pathlib import Path
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class SimilarityDistribution:
    """Bayesian model for same/different place similarity distributions.

    Fits two Gaussian distributions from training data:
    - metric='cosine': Fisher z-space (arctanh of cosine similarity)
    - metric='l2': L2 distance space (no transform needed)

    Then computes posterior P(same | observation) via Bayes' rule.
    """

    def __init__(self, metric: str = 'cosine'):
        """
        Args:
            metric: 'cosine' (Fisher z-transform) or 'l2' (direct L2 distance)
        """
        if metric not in ('cosine', 'l2'):
            raise ValueError(f"metric must be 'cosine' or 'l2', got '{metric}'")
        self.metric = metric
        self.mu_same: float = 0.0
        self.sigma_same: float = 1.0
        self.mu_diff: float = 0.0
        self.sigma_diff: float = 1.0
        self.fitted: bool = False

    def fit(
        self,
        descriptors: np.ndarray,
        poses: np.ndarray,
        sequence_ids: Optional[np.ndarray] = None,
        pos_dist: float = 5.0,
        neg_dist: float = 10.0,
        min_temporal_gap: int = 30,
        n_samples: int = 100000,
        seed: int = 42,
    ) -> "SimilarityDistribution":
        """Fit class-conditional distributions from training data.

        Uses random pair sampling to avoid O(N^2) cost.

        Args:
            descriptors: (N, D) descriptors.
                metric='cosine': L2-normalized descriptors
                metric='l2': z-scored descriptors (via StandardizationStats)
            poses: (N, 4, 4) SE(3) poses or (N, 3) XYZ positions
            sequence_ids: Optional (N,) sequence ID per descriptor. When provided,
                fit only samples pairs from the same sequence because pose frames are
                sequence-local in the multi-dataset training graph.
            pos_dist: Maximum GT distance for same-place pairs (meters)
            neg_dist: Minimum GT distance for different-place pairs (meters)
            min_temporal_gap: Minimum index gap between pairs
            n_samples: Number of random pairs to sample
            seed: Random seed for reproducibility

        Returns:
            self (for chaining)
        """
        rng = np.random.RandomState(seed)
        n = len(descriptors)

        # Extract XYZ from poses
        if poses.ndim == 3 and poses.shape[1:] == (4, 4):
            positions = poses[:, :3, 3]
        elif poses.ndim == 2 and poses.shape[1] >= 3:
            positions = poses[:, :3]
        else:
            raise ValueError(f"Unexpected pose shape: {poses.shape}")

        # Sample random pairs
        idx_i = rng.randint(0, n, size=n_samples)
        idx_j = rng.randint(0, n, size=n_samples)

        # Filter: different indices + temporal gap. With multi-dataset training,
        # poses from different trajectories do not share a coordinate frame.
        valid = (idx_i != idx_j) & (np.abs(idx_i - idx_j) >= min_temporal_gap)
        if sequence_ids is not None:
            sequence_ids = np.asarray(sequence_ids)
            if len(sequence_ids) != n:
                raise ValueError(
                    f"sequence_ids length ({len(sequence_ids)}) != descriptors length ({n})"
                )
            valid &= (sequence_ids[idx_i] == sequence_ids[idx_j])
        idx_i = idx_i[valid]
        idx_j = idx_j[valid]

        # Compute GT distances
        gt_dists = np.linalg.norm(positions[idx_i] - positions[idx_j], axis=1)

        if self.metric == 'cosine':
            # L2-normalize descriptors for cosine similarity via dot product
            norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            descs_normed = descriptors / norms
            # Cosine similarities
            observations = np.sum(descs_normed[idx_i] * descs_normed[idx_j], axis=1)
        else:
            # L2 distances on z-scored descriptors (= standardized Euclidean)
            observations = np.linalg.norm(
                descriptors[idx_i] - descriptors[idx_j], axis=1
            )

        # Classify pairs
        same_mask = gt_dists < pos_dist
        diff_mask = gt_dists > neg_dist

        same_obs = observations[same_mask]
        diff_obs = observations[diff_mask]

        logger.info(
            f"SimilarityDistribution.fit(metric={self.metric}): "
            f"{len(same_obs):,} same-place pairs, "
            f"{len(diff_obs):,} diff-place pairs (from {valid.sum():,} valid samples)"
        )

        if len(same_obs) < 10:
            logger.warning(
                f"Too few same-place pairs ({len(same_obs)}). "
                f"Falling back to unfitted state."
            )
            return self

        if len(diff_obs) < 10:
            logger.warning(
                f"Too few diff-place pairs ({len(diff_obs)}). "
                f"Falling back to unfitted state."
            )
            return self

        if self.metric == 'cosine':
            # Fisher z-transform: z = atanh(cos_sim)
            same_vals = np.arctanh(np.clip(same_obs, -0.9999, 0.9999))
            diff_vals = np.arctanh(np.clip(diff_obs, -0.9999, 0.9999))
        else:
            # L2 distances are already ~Gaussian in high-D (CLT), no transform
            same_vals = same_obs
            diff_vals = diff_obs

        # Fit Gaussians
        self.mu_same = float(np.mean(same_vals))
        self.sigma_same = float(np.std(same_vals))
        self.mu_diff = float(np.mean(diff_vals))
        self.sigma_diff = float(np.std(diff_vals))
        self.fitted = True

        # Prevent degenerate distributions
        self.sigma_same = max(self.sigma_same, 1e-6)
        self.sigma_diff = max(self.sigma_diff, 1e-6)

        if self.metric == 'cosine':
            space_label = 'z'
        else:
            space_label = 'L2'

        logger.info(
            f"  Same-place {space_label}-distribution: "
            f"N({self.mu_same:.4f}, {self.sigma_same:.4f})"
        )
        logger.info(
            f"  Diff-place {space_label}-distribution: "
            f"N({self.mu_diff:.4f}, {self.sigma_diff:.4f})"
        )

        # Report equivalent threshold at 95% confidence
        equiv_thresh = self.confidence_threshold(0.95)
        if self.metric == 'cosine':
            logger.info(
                f"  Equivalent cos_sim threshold at 95% confidence: {equiv_thresh:.6f}"
            )
        else:
            logger.info(
                f"  Equivalent L2 distance threshold at 95% confidence: {equiv_thresh:.4f}"
            )

        return self

    def posterior(
        self,
        observation: Union[float, np.ndarray],
        prior: Union[float, np.ndarray] = 0.01,
    ) -> Union[float, np.ndarray]:
        """Compute P(same_place | observation) via Bayes' rule.

        Args:
            observation: metric='cosine': cosine similarity in [-1, 1]
                         metric='l2': L2 distance (>= 0)
            prior: P(same_place) prior probability. Can be scalar or
                   per-element array (for density/entropy-adaptive prior).

        Returns:
            Posterior probability P(same | observation)
        """
        if not self.fitted:
            raise RuntimeError("Distribution not fitted. Call fit() first.")

        if self.metric == 'cosine':
            val = np.arctanh(np.clip(observation, -0.9999, 0.9999))
        else:
            # L2 distance used directly (already ~Gaussian in high-D)
            val = observation

        p_val_same = norm.pdf(val, self.mu_same, self.sigma_same)
        p_val_diff = norm.pdf(val, self.mu_diff, self.sigma_diff)

        numerator = p_val_same * prior
        denominator = numerator + p_val_diff * (1 - prior)

        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-30)

        return numerator / denominator

    def confidence_threshold(
        self, confidence: float = 0.95, prior: float = 0.01
    ) -> float:
        """Find observation threshold for given confidence level.

        Solves: P(same | observation) = confidence.

        Args:
            confidence: Target posterior probability (e.g., 0.95)
            prior: P(same_place) prior

        Returns:
            metric='cosine': minimum cosine similarity (higher = more similar)
            metric='l2': maximum L2 distance (lower = more similar)
        """
        if not self.fitted:
            raise RuntimeError("Distribution not fitted. Call fit() first.")

        if self.metric == 'cosine':
            return self._confidence_threshold_cosine(confidence, prior)
        else:
            return self._confidence_threshold_l2(confidence, prior)

    def _confidence_threshold_cosine(
        self, confidence: float, prior: float
    ) -> float:
        """Find cosine similarity threshold in z-space."""
        def objective(z):
            cs = np.tanh(z)
            return float(self.posterior(cs, prior)) - confidence

        # Search bounds: from mu_diff to mu_same + 5*sigma
        z_lo = self.mu_diff
        z_hi = self.mu_same + 5 * self.sigma_same

        if objective(z_lo) >= 0:
            return float(np.tanh(z_lo))
        if objective(z_hi) <= 0:
            logger.warning(
                f"Cannot reach confidence={confidence} with prior={prior}. "
                f"Max posterior at z={z_hi:.4f}: "
                f"{float(self.posterior(np.tanh(z_hi), prior)):.4f}"
            )
            return float(np.tanh(z_hi))

        z_threshold = brentq(objective, z_lo, z_hi, xtol=1e-8)
        return float(np.tanh(z_threshold))

    def _confidence_threshold_l2(
        self, confidence: float, prior: float
    ) -> float:
        """Find L2 distance threshold.

        For L2: μ_same < μ_diff (same-place pairs have smaller distances).
        We search from μ_same toward μ_diff for the crossing point.
        Returns max L2 distance where P(same | dist) >= confidence.
        """
        def objective(d):
            return float(self.posterior(d, prior)) - confidence

        # Search from near-zero to well past mu_diff
        d_lo = max(0.0, self.mu_same - 5 * self.sigma_same)
        d_hi = self.mu_diff + 5 * self.sigma_diff

        if objective(d_lo) <= 0:
            # Even at smallest distance, can't reach confidence
            logger.warning(
                f"Cannot reach confidence={confidence} with prior={prior}. "
                f"Max posterior at d={d_lo:.4f}: "
                f"{float(self.posterior(d_lo, prior)):.4f}"
            )
            return d_lo
        if objective(d_hi) >= 0:
            # Even at largest distance, still above confidence
            return d_hi

        d_threshold = brentq(objective, d_lo, d_hi, xtol=1e-6)
        return float(d_threshold)

    def compute_adaptive_priors(
        self,
        local_densities: np.ndarray,
        base_prior: float = 0.01,
        beta: float = 10.0,
    ) -> np.ndarray:
        """Compute density-adaptive priors for each node.

        Higher local density → lower prior (stricter edge creation).
        Uses sigmoid mapping centered at median density.

        Args:
            local_densities: (N,) per-node density values
                (e.g., mean cosine similarity to k-NN)
            base_prior: Maximum prior (for low-density nodes)
            beta: Sensitivity of density → prior mapping

        Returns:
            (N,) adaptive priors in (0, base_prior]
        """
        median_density = np.median(local_densities)
        # Negative sign: higher density → lower prior
        scale = _sigmoid(-beta * (local_densities - median_density))
        return base_prior * scale

    def compute_entropy_adaptive_priors(
        self,
        spectral_entropies: np.ndarray,
        base_prior: float = 0.01,
        beta: float = 10.0,
    ) -> np.ndarray:
        """Compute entropy-adaptive priors for each node.

        Spectral entropy directly measures how much information the
        compressed descriptor retains. Low entropy = energy concentrated
        in few frequencies = repetitive structure = high aliasing risk
        → stricter prior. High entropy = unique spectral signature
        = low aliasing risk → more permissive prior.

        Unlike density (which is an indirect proxy computed from k-NN
        distances in descriptor space), entropy is an intrinsic property
        of each descriptor computed during the encoding stage.

        Args:
            spectral_entropies: (N,) per-node spectral entropy (nats)
            base_prior: Maximum prior (for high-entropy nodes)
            beta: Sensitivity of entropy → prior mapping

        Returns:
            (N,) adaptive priors in (0, base_prior]
        """
        median_entropy = np.median(spectral_entropies)
        # Positive sign: higher entropy → higher prior (more permissive)
        # Lower entropy → lower prior (stricter, aliasing-prone)
        scale = _sigmoid(beta * (spectral_entropies - median_entropy))
        return base_prior * scale

    def save(self, path: Union[str, Path]) -> None:
        """Save distribution parameters to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mu_same=self.mu_same,
            sigma_same=self.sigma_same,
            mu_diff=self.mu_diff,
            sigma_diff=self.sigma_diff,
            fitted=self.fitted,
            metric=self.metric,
        )
        logger.info(f"Saved similarity distribution (metric={self.metric}) to {path}")

    def load(self, path: Union[str, Path]) -> "SimilarityDistribution":
        """Load distribution parameters from file."""
        data = np.load(path, allow_pickle=True)
        self.mu_same = float(data["mu_same"])
        self.sigma_same = float(data["sigma_same"])
        self.mu_diff = float(data["mu_diff"])
        self.sigma_diff = float(data["sigma_diff"])
        self.fitted = bool(data["fitted"])
        # Backward compat: old files without metric field default to 'cosine'
        self.metric = str(data["metric"]) if "metric" in data else "cosine"
        logger.info(
            f"Loaded similarity distribution (metric={self.metric}) from {path}: "
            f"same=N({self.mu_same:.4f}, {self.sigma_same:.4f}), "
            f"diff=N({self.mu_diff:.4f}, {self.sigma_diff:.4f})"
        )
        return self
