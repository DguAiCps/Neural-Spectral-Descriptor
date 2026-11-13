"""고정된 최종 매칭 오답 풀과 기존 삼중쌍을 같은 예산으로 섞는다."""
from typing import List, Tuple

import numpy as np


def paired_triplets(positives: List[np.ndarray], original_negative: np.ndarray,
                    matching_negative: np.ndarray, seed: int,
                    fraction: float = 0.5, count: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """정답 쌍·순서를 공유하고 지정 비율의 적격 오답만 바꾼 두 조건을 만든다."""
    assert 0 <= fraction <= 1 and count > 0
    assert len(positives) == len(original_negative) == len(matching_negative)
    rng = np.random.RandomState(seed)
    rows = [(a, int(rng.choice(pool)), int(original_negative[a]))
            for a, pool in enumerate(positives) if len(pool) and original_negative[a] >= 0
            for _ in range(count)]
    original = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    changed = original.copy()
    eligible = np.flatnonzero(np.any(matching_negative[original[:, 0]] >= 0, axis=1))
    chosen = rng.choice(eligible, size=int(len(eligible) * fraction), replace=False)
    for index in chosen:
        pool = matching_negative[original[index, 0]]
        changed[index, 2] = rng.choice(pool[pool >= 0])
    return original, changed
