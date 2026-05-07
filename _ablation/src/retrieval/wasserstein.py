"""
1D Wasserstein Distance for Spectral Histograms

Implements efficient O(n) computation of 1D Wasserstein distance
(also known as Earth Mover's Distance for 1D distributions).

For normalized histograms p and q with n bins:
    W_1(p, q) = ∫|CDF_p(x) - CDF_q(x)| dx
              = Σ|CDF_p[i] - CDF_q[i]|

This is more discriminative than simple L2 distance and captures
distributional differences in frequency space.
"""

import numpy as np
import torch
from typing import Union, Optional


def wasserstein_distance_1d_numpy(
    hist1: np.ndarray,
    hist2: np.ndarray,
    epsilon: float = 1e-8
) -> float:
    """
    Compute 1D Wasserstein distance between two histograms (NumPy)

    Args:
        hist1: (n_bins,) normalized histogram
        hist2: (n_bins,) normalized histogram
        epsilon: Small constant for numerical stability

    Returns:
        Wasserstein distance (scalar)
    """
    # Ensure normalized
    sum1 = hist1.sum()
    sum2 = hist2.sum()

    if sum1 > epsilon:
        hist1 = hist1 / sum1
    if sum2 > epsilon:
        hist2 = hist2 / sum2

    # Compute CDFs
    cdf1 = np.cumsum(hist1)
    cdf2 = np.cumsum(hist2)

    # Wasserstein distance = L1 distance between CDFs
    distance = np.abs(cdf1 - cdf2).sum()

    return float(distance)


def wasserstein_distance_1d_torch(
    hist1: torch.Tensor,
    hist2: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute 1D Wasserstein distance between two histograms (PyTorch)

    Args:
        hist1: (n_bins,) normalized histogram
        hist2: (n_bins,) normalized histogram
        epsilon: Small constant for numerical stability

    Returns:
        Wasserstein distance (scalar tensor)
    """
    # Ensure normalized
    sum1 = hist1.sum()
    sum2 = hist2.sum()

    if sum1 > epsilon:
        hist1 = hist1 / sum1
    if sum2 > epsilon:
        hist2 = hist2 / sum2

    # Compute CDFs
    cdf1 = torch.cumsum(hist1, dim=0)
    cdf2 = torch.cumsum(hist2, dim=0)

    # Wasserstein distance = L1 distance between CDFs
    distance = torch.abs(cdf1 - cdf2).sum()

    return distance


def wasserstein_distance_batch_numpy(
    query_hist: np.ndarray,
    database_hists: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Compute 1D Wasserstein distances from query to database (NumPy)

    Efficient batch computation for retrieval.

    Args:
        query_hist: (n_bins,) normalized query histogram
        database_hists: (n_database, n_bins) database histograms
        epsilon: Numerical stability constant

    Returns:
        (n_database,) Wasserstein distances
    """
    n_database = len(database_hists)

    # Normalize query
    query_sum = query_hist.sum()
    if query_sum > epsilon:
        query_hist = query_hist / query_sum

    # Normalize database (row-wise)
    database_sums = database_hists.sum(axis=1, keepdims=True)
    database_hists = np.where(
        database_sums > epsilon,
        database_hists / (database_sums + epsilon),
        database_hists
    )

    # Compute CDFs
    query_cdf = np.cumsum(query_hist)  # (n_bins,)
    database_cdfs = np.cumsum(database_hists, axis=1)  # (n_database, n_bins)

    # Broadcast and compute L1 distances
    # |CDF_q - CDF_db| summed over bins
    distances = np.abs(database_cdfs - query_cdf[None, :]).sum(axis=1)

    return distances


def precompute_cdfs_numpy(hists: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Precompute normalized CDFs for a batch of histograms.

    Useful when the same histograms will be queried multiple times,
    avoiding redundant normalization and cumsum per query.

    Args:
        hists: (N, n_bins) histograms
        epsilon: Numerical stability constant

    Returns:
        (N, n_bins) precomputed CDFs
    """
    sums = hists.sum(axis=1, keepdims=True)
    normalized = np.where(sums > epsilon, hists / (sums + epsilon), hists)
    return np.cumsum(normalized, axis=1)


def wasserstein_from_cdfs_numpy(
    query_cdf: np.ndarray,
    database_cdfs: np.ndarray,
) -> np.ndarray:
    """
    Compute Wasserstein distances from precomputed CDFs.

    Args:
        query_cdf: (n_bins,) single precomputed CDF
        database_cdfs: (n_database, n_bins) precomputed CDFs

    Returns:
        (n_database,) Wasserstein distances
    """
    return np.abs(database_cdfs - query_cdf[None, :]).sum(axis=1)


def wasserstein_distance_batch_torch(
    query_hist: torch.Tensor,
    database_hists: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute 1D Wasserstein distances from query to database (PyTorch)

    Efficient batch computation for retrieval with GPU support.

    Args:
        query_hist: (n_bins,) normalized query histogram
        database_hists: (n_database, n_bins) database histograms
        epsilon: Numerical stability constant

    Returns:
        (n_database,) Wasserstein distances
    """
    # Normalize query
    query_sum = query_hist.sum()
    if query_sum > epsilon:
        query_hist = query_hist / query_sum

    # Normalize database (row-wise)
    database_sums = database_hists.sum(dim=1, keepdim=True)
    database_hists = torch.where(
        database_sums > epsilon,
        database_hists / (database_sums + epsilon),
        database_hists
    )

    # Compute CDFs
    query_cdf = torch.cumsum(query_hist, dim=0)  # (n_bins,)
    database_cdfs = torch.cumsum(database_hists, dim=1)  # (n_database, n_bins)

    # Broadcast and compute L1 distances
    distances = torch.abs(database_cdfs - query_cdf.unsqueeze(0)).sum(dim=1)

    return distances


def wasserstein_distance_matrix_numpy(
    hists1: np.ndarray,
    hists2: Optional[np.ndarray] = None,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Compute pairwise Wasserstein distance matrix (NumPy)

    Args:
        hists1: (n1, n_bins) first set of histograms
        hists2: (n2, n_bins) second set of histograms (optional, defaults to hists1)
        epsilon: Numerical stability constant

    Returns:
        (n1, n2) distance matrix where D[i, j] = W_1(hists1[i], hists2[j])
    """
    if hists2 is None:
        hists2 = hists1

    n1 = len(hists1)
    n2 = len(hists2)

    # Normalize
    hists1_sums = hists1.sum(axis=1, keepdims=True)
    hists1 = np.where(
        hists1_sums > epsilon,
        hists1 / (hists1_sums + epsilon),
        hists1
    )

    hists2_sums = hists2.sum(axis=1, keepdims=True)
    hists2 = np.where(
        hists2_sums > epsilon,
        hists2 / (hists2_sums + epsilon),
        hists2
    )

    # Compute CDFs
    cdfs1 = np.cumsum(hists1, axis=1)  # (n1, n_bins)
    cdfs2 = np.cumsum(hists2, axis=1)  # (n2, n_bins)

    # Compute distance matrix
    # D[i, j] = Σ_k |CDF1[i, k] - CDF2[j, k]|
    distance_matrix = np.zeros((n1, n2))

    for i in range(n1):
        # Broadcast: (n2, n_bins)
        diff = np.abs(cdfs2 - cdfs1[i:i+1, :])
        distance_matrix[i, :] = diff.sum(axis=1)

    return distance_matrix


def wasserstein_distance_matrix_torch(
    hists1: torch.Tensor,
    hists2: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute pairwise Wasserstein distance matrix (PyTorch)

    Args:
        hists1: (n1, n_bins) first set of histograms
        hists2: (n2, n_bins) second set of histograms (optional, defaults to hists1)
        epsilon: Numerical stability constant

    Returns:
        (n1, n2) distance matrix where D[i, j] = W_1(hists1[i], hists2[j])
    """
    if hists2 is None:
        hists2 = hists1

    # Normalize
    hists1_sums = hists1.sum(dim=1, keepdim=True)
    hists1 = torch.where(
        hists1_sums > epsilon,
        hists1 / (hists1_sums + epsilon),
        hists1
    )

    hists2_sums = hists2.sum(dim=1, keepdim=True)
    hists2 = torch.where(
        hists2_sums > epsilon,
        hists2 / (hists2_sums + epsilon),
        hists2
    )

    # Compute CDFs
    cdfs1 = torch.cumsum(hists1, dim=1)  # (n1, n_bins)
    cdfs2 = torch.cumsum(hists2, dim=1)  # (n2, n_bins)

    # Compute distance matrix using broadcasting
    # (n1, 1, n_bins) - (1, n2, n_bins) -> (n1, n2, n_bins)
    diff = torch.abs(cdfs1.unsqueeze(1) - cdfs2.unsqueeze(0))

    # Sum over bins
    distance_matrix = diff.sum(dim=2)  # (n1, n2)

    return distance_matrix


class WassersteinRetriever:
    """
    Efficient Wasserstein-based retrieval for spectral histograms
    """

    def __init__(
        self,
        use_torch: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize retriever

        Args:
            use_torch: If True, use PyTorch (GPU-accelerated); else NumPy
            device: Device for PyTorch ('cpu' or 'cuda')
        """
        self.use_torch = use_torch
        self.device = device

        # Database
        self.database_hists = None
        self.database_size = 0

    def add_to_database(self, histograms: Union[np.ndarray, torch.Tensor]):
        """
        Add histograms to database

        Args:
            histograms: (n, n_bins) histograms
        """
        if self.use_torch:
            if isinstance(histograms, np.ndarray):
                histograms = torch.from_numpy(histograms).float()

            histograms = histograms.to(self.device)

            if self.database_hists is None:
                self.database_hists = histograms
            else:
                self.database_hists = torch.cat([self.database_hists, histograms], dim=0)
        else:
            if isinstance(histograms, torch.Tensor):
                histograms = histograms.cpu().numpy()

            if self.database_hists is None:
                self.database_hists = histograms
            else:
                self.database_hists = np.vstack([self.database_hists, histograms])

        self.database_size = len(self.database_hists)

    def query(
        self,
        query_hist: Union[np.ndarray, torch.Tensor],
        top_k: int = 10
    ) -> tuple:
        """
        Query database and return top-K matches

        Args:
            query_hist: (n_bins,) query histogram
            top_k: Number of top matches to return

        Returns:
            indices: (top_k,) indices of top matches
            distances: (top_k,) Wasserstein distances
        """
        if self.database_size == 0:
            return np.array([]), np.array([])

        # Compute distances
        if self.use_torch:
            if isinstance(query_hist, np.ndarray):
                query_hist = torch.from_numpy(query_hist).float()

            query_hist = query_hist.to(self.device)

            distances = wasserstein_distance_batch_torch(
                query_hist,
                self.database_hists
            )

            # Get top-K
            top_k = min(top_k, self.database_size)
            distances_sorted, indices = torch.topk(
                distances,
                k=top_k,
                largest=False  # Smallest distances
            )

            indices = indices.cpu().numpy()
            distances_sorted = distances_sorted.cpu().numpy()
        else:
            if isinstance(query_hist, torch.Tensor):
                query_hist = query_hist.cpu().numpy()

            distances = wasserstein_distance_batch_numpy(
                query_hist,
                self.database_hists
            )

            # Get top-K
            top_k = min(top_k, self.database_size)
            indices = np.argpartition(distances, top_k)[:top_k]
            indices = indices[np.argsort(distances[indices])]
            distances_sorted = distances[indices]

        return indices, distances_sorted

    def clear_database(self):
        """Clear database"""
        self.database_hists = None
        self.database_size = 0
