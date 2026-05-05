"""BEVPlace++ baseline (Luo et al., TRO'25).

Architecture
------------
Vendored from the official repo (github.com/zjuluolun/BEVPlace2):
  REM (rotation-equivariant module): warps the BEV by 8 angles, runs a shared
  ResNet-34 backbone (truncated, ImageNet-pretrained), unwarps each output and
  takes the per-pixel max over rotations -- yielding rotation-equivariant
  feature maps -> NetVLAD (64 clusters x 128 dim = 8192D global descriptor).

The architecture, weights, and per-channel normalization match the official
release; the only piece that depends on this codebase is the BEV image
generation, which we mirror as closely as the reference repo allows.

Weights
-------
Place `model_best.pth.tar` from
  github.com/zjuluolun/BEVPlace2/runs/Aug08_10-17-29
at `baselines/weights/bevplace_kitti.pth.tar`, or set
`NSD_BEVPLACE_WEIGHTS` to its absolute path.

Cross-sensor caveat
-------------------
The released checkpoint is KITTI-trained; cross-sensor results in Table 2 are
reported as released (no fine-tuning), matching the paper's honest-comparison
policy. NCLT/HeLiPR/MulRan numbers therefore reflect training-distribution
mismatch and are reported as such.
"""

import os
import sys

import numpy as np

from baselines import register
from baselines.base import BaselineEncoder


_DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "weights", "bevplace_kitti.pth.tar"
)
_WEIGHTS_PATH = os.environ.get("NSD_BEVPLACE_WEIGHTS", _DEFAULT_WEIGHTS)
_OFFICIAL_DIR = os.path.join(os.path.dirname(__file__), "_bevplace_official")


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


def _bev_image(points: np.ndarray,
               img_size: int = 200,
               max_range: float = 50.0,
               z_min: float = -3.0,
               z_max: float = 5.0) -> np.ndarray:
    """Cartesian BEV image: max-z height per cell, normalized to uint8 then /256.

    The output convention matches the BEVPlace2 dataset loader: a 3-channel
    float32 image in [0, 1) range, with the channel dim broadcast from a single
    BEV. Cells outside [-max_range, +max_range] are zero.
    """
    pts = np.asarray(points[:, :3], dtype=np.float32)
    keep = (np.abs(pts[:, 0]) <= max_range) & (np.abs(pts[:, 1]) <= max_range) & \
           (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    pts = pts[keep]
    if pts.shape[0] == 0:
        return np.zeros((3, img_size, img_size), dtype=np.float32)

    res = (2.0 * max_range) / img_size
    xi = np.clip(((pts[:, 0] + max_range) / res).astype(np.int32), 0, img_size - 1)
    yi = np.clip(((pts[:, 1] + max_range) / res).astype(np.int32), 0, img_size - 1)
    z_norm = np.clip((pts[:, 2] - z_min) / max(z_max - z_min, 1e-6) * 255.0, 0, 255)

    grid = np.zeros((img_size, img_size), dtype=np.float32)
    # Use np.maximum.at for max-z aggregation (vectorized scatter-max).
    np.maximum.at(grid, (yi, xi), z_norm)
    grid = grid / 256.0  # match dataset.py: img.astype(float32) / 256
    return np.broadcast_to(grid[None, :, :], (3, img_size, img_size)).copy()


class _REINModule:
    """Lazy holder for the vendored REIN model + KITTI-pretrained weights."""

    _module = None

    @classmethod
    def get(cls, device: str):
        if cls._module is not None:
            return cls._module
        if _OFFICIAL_DIR not in sys.path:
            sys.path.insert(0, _OFFICIAL_DIR)
        import torch
        from REIN import REIN  # vendored from BEVPlace2/REIN.py

        model = REIN().to(device).eval()
        ckpt = torch.load(_WEIGHTS_PATH, map_location=device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[BEVPlace++] state_dict missing={len(missing)} "
                  f"unexpected={len(unexpected)}", file=sys.stderr)
        cls._module = model
        return model


@register
class BEVPlaceBaseline(BaselineEncoder):
    """BEVPlace++ with KITTI-pretrained weights via vendored REIN."""

    def __init__(self, img_size: int = 200, max_range: float = 50.0):
        self.img_size = img_size
        self.max_range = max_range

    @property
    def name(self) -> str:
        return "BEVPlace++"

    @property
    def short_name(self) -> str:
        return "bevplace"

    @property
    def descriptor_dim(self) -> int:
        return 64 * 128  # NetVLAD: 64 clusters x 128 dim = 8192

    def is_available(self) -> bool:
        return (
            _torch_available()
            and os.path.exists(_WEIGHTS_PATH)
            and os.path.exists(os.path.join(_OFFICIAL_DIR, "REIN.py"))
        )

    def encode(self, points: np.ndarray) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError(
                f"BEVPlace++ unavailable; weights={_WEIGHTS_PATH}, "
                f"official_dir={_OFFICIAL_DIR}"
            )
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _REINModule.get(device)
        bev = _bev_image(points, self.img_size, self.max_range)
        with torch.no_grad():
            x = torch.from_numpy(bev).unsqueeze(0).to(device)
            _, _, global_desc = model(x)  # (1, 8192)
            emb = global_desc.squeeze(0).cpu().numpy().astype(np.float32)
        # REIN already L2-normalizes inside NetVLAD; renormalize for safety.
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        return emb
