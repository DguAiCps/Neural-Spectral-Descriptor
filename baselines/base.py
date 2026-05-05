"""Abstract base class for LiDAR place recognition baseline encoders."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np


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

    def encode_with_aux(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Encode and optionally return auxiliary data needed for reranking.

        Default: returns (encode(points), {}). Methods that need to keep a 2D
        SC matrix or binary template stack for rerank should override this and
        populate the aux dict.
        """
        return self.encode(points), {}

    def encode_sequence(
        self, point_clouds: List[np.ndarray], progress_interval: int = 200
    ) -> np.ndarray:
        """Encode a sequence of point clouds with timing."""
        descriptors, _ = self.encode_sequence_with_aux(
            point_clouds, progress_interval=progress_interval
        )
        return descriptors

    def encode_sequence_with_aux(
        self, point_clouds: List[np.ndarray], progress_interval: int = 200
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Encode a sequence and return both descriptors and per-scan aux dicts.

        Sets self.last_encode_time_ms.
        """
        descriptors = []
        aux_list = []
        times = []
        for i, pc in enumerate(point_clouds):
            t0 = time.perf_counter()
            desc, aux = self.encode_with_aux(pc)
            times.append(time.perf_counter() - t0)
            descriptors.append(desc)
            aux_list.append(aux)
            if progress_interval and (i + 1) % progress_interval == 0:
                avg_ms = np.mean(times[-progress_interval:]) * 1000
                print(f"    [{i+1}/{len(point_clouds)}] avg {avg_ms:.1f} ms/scan")

        self.last_encode_time_ms = np.mean(times) * 1000 if times else 0.0
        if progress_interval:
            print(
                f"    Done: {len(point_clouds)} scans, "
                f"avg {self.last_encode_time_ms:.1f} ms/scan"
            )
        return np.array(descriptors, dtype=np.float32), aux_list

    def compute_recalls(
        self,
        point_clouds: List[np.ndarray],
        poses: np.ndarray,
        k_values: List[int] = [1, 5, 10],
        distance_threshold: float = 5.0,
        skip_frames: int = 30,
        per_query_records=None,
    ) -> Tuple[Dict[int, float], int]:
        """
        Default retrieval: encode all point clouds, then run cosine FAISS.

        Methods that need column-shift rerank or Hamming distance should
        override this; see ScanContextPP and LiDARIris.
        """
        from baselines.eval_utils import compute_recall_multi_k

        descriptors = self.encode_sequence(point_clouds)
        return compute_recall_multi_k(
            descriptors, poses,
            k_values=k_values,
            distance_threshold=distance_threshold,
            skip_frames=skip_frames,
            per_query_records=per_query_records,
        )

    def is_available(self) -> bool:
        """Check if this method's dependencies are satisfied."""
        return True
