"""키프레임별 원래 양성 표본과 포함 관계를 지키는 예산 표본을 만든다."""
from typing import Dict, List

import numpy as np
from scipy.spatial import cKDTree


def positive_pools(positions: np.ndarray, sequences: np.ndarray) -> List[np.ndarray]:
    """같은 시퀀스의 5 m 이내·프레임 번호 차이 30 이상 후보를 정렬한다."""
    pools = [np.empty(0, np.int64) for _ in positions]
    for sequence in np.unique(sequences):
        ids = np.flatnonzero(sequences == sequence)
        tree = cKDTree(positions[ids])
        for i, neighbors in enumerate(tree.query_ball_tree(tree, 5.)):
            candidates = np.asarray(neighbors, np.int64)
            pools[ids[i]] = ids[candidates[np.abs(candidates-i) >= 30]]
    return pools


def nested_rows(total: int, large: int, small: int, seed: int) -> Dict[str, np.ndarray]:
    """큰 예산 안에 작은 예산이 들어가도록 원 행을 비복원 선택한다."""
    if not 0 < small <= large <= total:
        raise ValueError("0 < small <= large <= total 이어야 한다")
    rng = np.random.default_rng(seed)
    high = rng.choice(total, large, replace=False)
    low = high[rng.choice(large, small, replace=False)]
    return {"large": high, "small": low}
