"""RING++ baseline (Xu et al., T-RO 2023) — portable reimplementation.

Follows the official code (github.com/lus6-Jenny/RING, utils/core.py +
evaluation/evaluate.py): point filter (|x|,|y| < 70 m) -> single-height BEV
occupancy on a num_ring x num_sector = 120x120 grid -> Radon sinogram over
120 angles in [0, 360) -> TIRING = row-wise |FFT| (the translation-invariant
RING++ retrieval descriptor) -> ranking by circular cross-correlation of
TIRINGs along the angle axis (``fast_corr``), served here as a rerank over a
TIRING-cosine pre-filter (same two-stage harness as SC++/LiDAR-Iris).

Portability differences from the official repo (results-neutral):
  * Radon via batched rotate-and-project (torch ``grid_sample``) instead of
    the compiled ``torch-radon`` CUDA extension.
  * BEV occupancy via histogram instead of the CUDA voxelizer.
  * The official z-filter [1, 20] m assumes a ground-at-zero frame; our
    loaders are sensor-centric (ground at z ~ -2 m), so the equivalent
    "about 1 m above ground" bound is z in [-1, 20] m.
"""

from __future__ import annotations

from typing import List

import numpy as np

from baselines import register
from baselines.base import BaselineEncoder

try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:  # pragma: no cover
    _TORCH_OK = False

NUM_RING = 120     # number of Radon angles (rows of the sinogram)
NUM_SECTOR = 120   # samples along each projection (columns)
XY_BOUND = 70.0
Z_BOUND = (-1.0, 20.0)


class RINGPlusPlus(BaselineEncoder):
    """RING++ translation-invariant Radon descriptor with correlation rerank."""

    def __init__(self, n_coarse: int = 200, device: str | None = None):
        self.n_coarse = n_coarse
        if _TORCH_OK:
            self._device = torch.device(
                device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self._grids = None  # lazy (A, H, W, 2) rotation grids

    # -- metadata ---------------------------------------------------------
    @property
    def name(self):
        return "RING++"

    @property
    def short_name(self):
        return "ring++"

    @property
    def descriptor_dim(self):
        return NUM_RING * NUM_SECTOR

    def is_available(self):
        return _TORCH_OK

    # -- encoding ---------------------------------------------------------
    def _rotation_grids(self):
        if self._grids is None:
            angles = torch.arange(NUM_RING, dtype=torch.float32,
                                  device=self._device) * (2 * np.pi / NUM_RING)
            cos, sin = torch.cos(angles), torch.sin(angles)
            theta = torch.zeros(NUM_RING, 2, 3, device=self._device)
            theta[:, 0, 0] = cos
            theta[:, 0, 1] = -sin
            theta[:, 1, 0] = sin
            theta[:, 1, 1] = cos
            self._grids = F.affine_grid(
                theta, (NUM_RING, 1, NUM_SECTOR, NUM_SECTOR), align_corners=False)
        return self._grids

    def _bev_occupancy(self, points: np.ndarray) -> np.ndarray:
        p = points[:, :3]
        m = ((np.abs(p[:, 0]) < XY_BOUND) & (np.abs(p[:, 1]) < XY_BOUND)
             & (p[:, 2] > Z_BOUND[0]) & (p[:, 2] < Z_BOUND[1]))
        p = p[m]
        ix = np.clip(((p[:, 0] + XY_BOUND) / (2 * XY_BOUND)
                      * NUM_SECTOR).astype(np.int64), 0, NUM_SECTOR - 1)
        iy = np.clip(((p[:, 1] + XY_BOUND) / (2 * XY_BOUND)
                      * NUM_SECTOR).astype(np.int64), 0, NUM_SECTOR - 1)
        bev = np.zeros((NUM_SECTOR, NUM_SECTOR), dtype=np.float32)
        bev[iy, ix] = 1.0
        return bev

    def _sinogram(self, bev: np.ndarray) -> np.ndarray:
        img = torch.from_numpy(bev).to(self._device)
        img = img[None, None].expand(NUM_RING, 1, NUM_SECTOR, NUM_SECTOR)
        rot = F.grid_sample(img, self._rotation_grids(), mode="bilinear",
                            padding_mode="zeros", align_corners=False)
        return rot.sum(dim=2).squeeze(1).cpu().numpy()  # (A, NUM_SECTOR)

    def encode(self, points: np.ndarray) -> np.ndarray:
        desc, _ = self.encode_with_aux(points)
        return desc

    def encode_with_aux(self, points: np.ndarray):
        bev = self._bev_occupancy(points)
        sino = self._sinogram(bev)
        tiring = np.abs(np.fft.fft(sino, axis=-1, norm="ortho")).astype(np.float32)
        desc = tiring.reshape(-1)
        n = np.linalg.norm(desc)
        if n > 0:
            desc = desc / n
        return desc.astype(np.float32), {"tiring": tiring}

    # -- retrieval --------------------------------------------------------
    def _corr_dists(self, tirings: "torch.Tensor", q_idx: int,
                    cand: np.ndarray) -> np.ndarray:
        """fast_corr of utils/core.py: circular corr along the angle axis."""
        a = tirings[q_idx][None]           # (1, A, S)
        b = tirings[torch.from_numpy(cand).to(tirings.device).long()]
        a = (a - a.mean()) / (a.std() + 1e-8)
        b = (b - b.mean(dim=(-2, -1), keepdim=True)) / (
            b.std(dim=(-2, -1), keepdim=True) + 1e-8)
        af = torch.fft.fft(a, dim=-2, norm="ortho")
        bf = torch.fft.fft(b, dim=-2, norm="ortho")
        corr = torch.fft.ifft(af * bf.conj(), dim=-2, norm="ortho")
        corr = torch.sqrt(corr.real ** 2 + corr.imag ** 2).sum(dim=-1)  # (B, A)
        max_corr = corr.max(dim=-1)[0]
        dist = 1.0 - max_corr / (0.15 * NUM_RING * NUM_SECTOR)
        return dist.cpu().numpy()

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
        tirings = torch.from_numpy(
            np.stack([a["tiring"] for a in aux_list], axis=0)).to(self._device)

        def rerank_fn(query_idx: int, candidates: np.ndarray) -> np.ndarray:
            d = self._corr_dists(tirings, query_idx, candidates)
            return candidates[np.argsort(d)]

        return compute_recall_cosine_then_rerank(
            descriptors, rerank_fn, poses,
            k_values=k_values,
            distance_threshold=distance_threshold,
            skip_frames=skip_frames,
            n_coarse=self.n_coarse,
            per_query_records=per_query_records,
        )


register(RINGPlusPlus)
