"""PointNetVLAD baseline (Uy & Lee, CVPR'18).

Architecture
------------
PointNet shared-MLP point feature extractor (1024D per-point) followed by
NetVLAD aggregation with K=64 clusters and a final FC compressing to 256D.

Forward pass:
    (N, 3) -> normalize to unit cube -> downsample to 4096 points
            -> per-point MLP (3 -> 64 -> 128 -> 1024)
            -> NetVLAD aggregation (1024 x 64 = 65536) -> L2 normalize
            -> FC 65536 -> 256 -> L2 normalize.

Weights
-------
Public PyTorch reference: github.com/cattaneod/PointNetVlad-Pytorch
Download `pretrained_models/checkpoint.pth.tar` and place at:
    baselines/weights/pointnetvlad_kitti.pth
or set environment variable NSD_PNV_WEIGHTS to the absolute path.

Cross-sensor caveat
-------------------
Public weights are trained on the Oxford RobotCar / KITTI benchmark. Performance
on NCLT/HeLiPR/MulRan reflects training-distribution mismatch and is reported
as such; this is a deliberate design choice for the comparison in Table 2.
"""

import os
from typing import Optional

import numpy as np

from baselines import register
from baselines.base import BaselineEncoder


_DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "weights", "pointnetvlad_kitti.pth"
)
_WEIGHTS_PATH = os.environ.get("NSD_PNV_WEIGHTS", _DEFAULT_WEIGHTS)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class _PointNetMLP:
    """Lazy holder; built only when encode() is called and torch is present."""

    _module = None

    @classmethod
    def get(cls, device: str):
        if cls._module is not None:
            return cls._module

        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class NetVLAD(nn.Module):
            def __init__(self, num_clusters: int = 64, dim: int = 1024):
                super().__init__()
                self.num_clusters = num_clusters
                self.dim = dim
                self.conv = nn.Conv1d(dim, num_clusters, kernel_size=1, bias=True)
                self.centroids = nn.Parameter(torch.randn(num_clusters, dim))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B, dim, N)
                soft_assign = F.softmax(self.conv(x), dim=1)  # (B, K, N)
                B, _, N = x.shape
                residual = x.unsqueeze(1) - self.centroids[None, :, :, None]
                vlad = (residual * soft_assign.unsqueeze(2)).sum(dim=-1)
                vlad = F.normalize(vlad, p=2, dim=2)
                vlad = vlad.reshape(B, -1)
                return F.normalize(vlad, p=2, dim=1)

        class PointNetVLAD(nn.Module):
            def __init__(self, output_dim: int = 256):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                    nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                    nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                    nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                    nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024),
                )
                self.netvlad = NetVLAD(num_clusters=64, dim=1024)
                self.fc = nn.Linear(1024 * 64, output_dim)

            def forward(self, pts: torch.Tensor) -> torch.Tensor:
                # pts: (B, 3, N)
                feats = self.mlp(pts)
                vlad = self.netvlad(feats)
                emb = self.fc(vlad)
                return F.normalize(emb, p=2, dim=1)

        model = PointNetVLAD(output_dim=256).to(device).eval()
        if os.path.exists(_WEIGHTS_PATH):
            ckpt = torch.load(_WEIGHTS_PATH, map_location=device)
            state = ckpt.get("state_dict", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                # Print to stderr so the eval log records the mismatch.
                import sys
                print(f"[PointNetVLAD] state_dict missing={len(missing)} "
                      f"unexpected={len(unexpected)} (proceeding with partial load)",
                      file=sys.stderr)
        cls._module = model
        return model


def _to_unit_cube(points: np.ndarray, n: int = 4096) -> np.ndarray:
    pts = np.asarray(points[:, :3], dtype=np.float32)
    if pts.shape[0] >= n:
        idx = np.random.default_rng(0).choice(pts.shape[0], size=n, replace=False)
    else:
        idx = np.random.default_rng(0).choice(pts.shape[0], size=n, replace=True)
    pts = pts[idx]
    centroid = pts.mean(axis=0, keepdims=True)
    pts = pts - centroid
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-8
    return pts / scale


@register
class PointNetVLADBaseline(BaselineEncoder):
    """PointNetVLAD with KITTI-pretrained weights."""

    def __init__(self, n_points: int = 4096, output_dim: int = 256):
        self.n_points = n_points
        self._output_dim = output_dim

    @property
    def name(self) -> str:
        return "PointNetVLAD"

    @property
    def short_name(self) -> str:
        return "pointnetvlad"

    @property
    def descriptor_dim(self) -> int:
        return self._output_dim

    def is_available(self) -> bool:
        return _torch_available() and os.path.exists(_WEIGHTS_PATH)

    def encode(self, points: np.ndarray) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError(
                f"PointNetVLAD weights missing at {_WEIGHTS_PATH}; "
                "set NSD_PNV_WEIGHTS or download per module docstring."
            )
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _PointNetMLP.get(device)
        pts = _to_unit_cube(points, self.n_points)
        with torch.no_grad():
            x = torch.from_numpy(pts).t().unsqueeze(0).to(device)  # (1, 3, N)
            emb = model(x).squeeze(0).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        return emb
