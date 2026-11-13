"""같은 행별 크기 스펙트럼에서 낮은 주파수만 남기는 비교용 키."""

import numpy as np


def frequency_key(magnitudes: np.ndarray, frequencies: int) -> np.ndarray:
    """DC부터 지정 개수의 계수를 행 우선으로 펴고 float32로 반환한다."""
    values = np.asarray(magnitudes)
    if values.ndim not in (2, 3) or values.shape[-2] != 16:
        raise ValueError("16개 높이 행의 단일 또는 배치 스펙트럼이 필요하다")
    if not 1 <= frequencies <= values.shape[-1]:
        raise ValueError("남길 주파수 개수가 범위를 벗어났다")
    if not np.isfinite(values).all():
        raise ValueError("유한하지 않은 스펙트럼")
    return np.ascontiguousarray(values[..., :frequencies], dtype=np.float32).reshape(
        *values.shape[:-2], 16 * frequencies
    )
