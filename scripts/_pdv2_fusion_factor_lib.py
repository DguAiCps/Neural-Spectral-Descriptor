"""후보 집합과 점수 정규화 범위를 독립적으로 다루는 진단 연산."""
from typing import Tuple

import numpy as np


def fuse_with_range(channels: np.ndarray, weights: tuple,
                    reference: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """별도 후보 범위로 네 채널을 min-max 정규화한다. 자르기는 하지 않는다."""
    if reference is None:
        reference = channels
    if channels.ndim != 2 or channels.shape[1] != 4 or not len(reference):
        raise ValueError("비어 있지 않은 네 채널 후보 점수가 필요하다")
    if not np.isfinite(channels).all() or not np.isfinite(reference).all():
        raise ValueError("점수는 유한해야 한다")
    pieces = []
    ranges = []
    for i, weight in enumerate(weights):
        low, high = float(reference[:, i].min()), float(reference[:, i].max())
        width = max(high - low, 1e-8)
        pieces.append(float(weight) * ((channels[:, i] - low) / width))
        ranges.append((low, high, width))
    return sum(pieces), np.asarray(ranges, dtype=np.float64)


def pair_components(positive: np.ndarray, negative: np.ndarray,
                    ranges: np.ndarray, weights: tuple) -> np.ndarray:
    """고정 정답·오답 쌍의 네 채널 융합 점수 차이를 계산한다."""
    return np.asarray(weights, dtype=np.float64) * (
        positive.astype(np.float64) - negative.astype(np.float64)) / ranges[:, 2]
