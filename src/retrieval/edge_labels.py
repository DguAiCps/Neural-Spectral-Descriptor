"""좌표계의 모든 공간 축을 사용하는 간선 분류기 학습 라벨."""
from __future__ import annotations

import numpy as np


def edge_labels_3d(positions: np.ndarray, query: np.ndarray,
                   neighbor: np.ndarray, positive_distance: float = 5.0,
                   negative_distance: float = 25.0) -> np.ndarray:
    """3D 거리로 양성(1), 음성(0), 제외(-1)를 정한다. 경계는 기존 학습과 같다."""
    positions = np.asarray(positions)
    query, neighbor = np.asarray(query), np.asarray(neighbor)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions는 N×3 좌표여야 한다")
    if query.ndim != 1 or query.shape != neighbor.shape:
        raise ValueError("query와 neighbor는 길이가 같은 1차원 배열이어야 한다")
    if not 0 < positive_distance < negative_distance:
        raise ValueError("거리 경계는 0 < positive < negative여야 한다")
    if not np.isfinite(positions).all():
        raise ValueError("좌표에 유한하지 않은 값이 있다")
    distance = np.linalg.norm(positions[query] - positions[neighbor], axis=1)
    labels = np.full(len(query), -1, dtype=np.int8)
    labels[distance < positive_distance] = 1
    labels[distance >= negative_distance] = 0
    return labels
