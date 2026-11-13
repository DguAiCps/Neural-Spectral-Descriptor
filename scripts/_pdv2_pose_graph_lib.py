"""시간 포즈 그래프를 독립 SE(3) 계산과 대조하는 진단 도구."""
from typing import Dict, Tuple

import numpy as np


def relative_geometry(poses: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """목적 노드 좌표계에서 상대 이동 벡터와 부호 없는 회전각을 계산한다."""
    source, target = edges
    target_inverse = np.linalg.inv(poses[target].astype(np.float64))
    relative = target_inverse @ poses[source].astype(np.float64)
    cosine = (np.trace(relative[:, :3, :3], axis1=1, axis2=2) - 1.) / 2.
    return relative[:, :3, 3], np.arccos(np.clip(cosine, -1., 1.))


def quantiles(values: np.ndarray) -> Dict[str, float]:
    """고정 분위수와 평균·최댓값을 유한한 실수로 반환한다."""
    values = np.asarray(values)
    assert values.size and np.isfinite(values).all()
    result = dict(zip(('p10', 'p50', 'p90', 'p99'),
                      map(float, np.quantile(values, [.1, .5, .9, .99]))))
    return dict(result, mean=float(values.mean()), max=float(values.max()))


def ordered_windows(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """경계 없는 중심과 시간순 열 이웃의 실제 거리·회전 크기를 구한다."""
    centers = np.arange(5, len(poses) - 5)
    offsets = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    source = (centers[:, None] + offsets).reshape(-1)
    target = np.repeat(centers, 10)
    translation, angle = relative_geometry(poses, np.stack((source, target)))
    return centers, np.linalg.norm(translation, axis=1).reshape(-1, 10), angle.reshape(-1, 10)


def window_difference(distance: np.ndarray, angle: np.ndarray,
                      first: np.ndarray, second: np.ndarray) -> dict:
    """시간 정순·역순 중 기하 오차가 작은 창 비교를 두 성분별로 반환한다.

    두 성분은 각각 역순을 최소화한다. 학습 점수나 공동 최적 정렬이 아니다.
    """
    d1, d2 = distance[first - 5], distance[second - 5]
    a1, a2 = angle[first - 5], angle[second - 5]
    dr = np.minimum(np.sqrt(np.mean((d1 - d2) ** 2, axis=1)),
                    np.sqrt(np.mean((d1 - d2[:, ::-1]) ** 2, axis=1)))
    ar = np.minimum(np.sqrt(np.mean((a1 - a2) ** 2, axis=1)),
                    np.sqrt(np.mean((a1 - a2[:, ::-1]) ** 2, axis=1)))
    return {'distance_rms_m': dr, 'rotation_rms_deg': np.rad2deg(ar),
            'relative_distance_rms': dr / np.maximum((d1.mean(1) + d2.mean(1)) / 2, 1e-12)}
