"""Abstract base class for LiDAR place recognition baseline encoders."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import time


class BaselineEncoder(ABC):
    """All baseline methods must implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable method name."""

    @property
    @abstractmethod
    def short_name(self) -> str:
        """Short name for tables and CLI."""

    @property
    @abstractmethod
    def descriptor_dim(self) -> int:
        """Output descriptor dimensionality."""

    @abstractmethod
    def encode(self, points: np.ndarray) -> np.ndarray:
        """
        Encode a single point cloud to a descriptor vector.

        Args:
            points: (N, 3) or (N, 4) numpy array [x,y,z] or [x,y,z,intensity]

        Returns:
            (D,) numpy float32 descriptor, L2-normalized.
        """

    def encode_sequence(self, point_clouds: List[np.ndarray],
                        progress_interval: int = 200) -> np.ndarray:
        """Encode a sequence of point clouds with timing."""
        descriptors = []
        times = []
        for i, pc in enumerate(point_clouds):
            t0 = time.perf_counter()
            desc = self.encode(pc)
            times.append(time.perf_counter() - t0)
            descriptors.append(desc)
            if progress_interval and (i + 1) % progress_interval == 0:
                avg_ms = np.mean(times[-progress_interval:]) * 1000
                print(f"    [{i+1}/{len(point_clouds)}] avg {avg_ms:.1f} ms/scan")

        self.last_encode_time_ms = np.mean(times) * 1000 if times else 0.0
        if progress_interval:
            print(f"    Done: {len(point_clouds)} scans, "
                  f"avg {self.last_encode_time_ms:.1f} ms/scan")
        return np.array(descriptors, dtype=np.float32)

    def is_available(self) -> bool:
        """Check if this method's dependencies are satisfied."""
        return True
