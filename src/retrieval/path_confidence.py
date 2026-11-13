"""후보 경로의 중심 거절과 점수 상승을 분리하는 고정 판정 대조."""
from typing import Dict

import numpy as np


def confidence_variants(original_score: np.ndarray, selected_score: np.ndarray,
                        null_margin: np.ndarray) -> Dict[str, np.ndarray]:
    """선택 후보를 유지하며 경로 벌점과 기존 점수 상한을 따로 적용한다.

    점수는 확률이 아니다. 음의 경로 여유만 exp(margin)으로 감쇠한다.
    원래 선택의 점수 상한은 점수 상승을 막지만, 바뀐 승자가 정답인지
    보장하지 않는다. 양의 경로 여유로 점수를 추가 상승시키지 않는다.
    """
    original = np.asarray(original_score, dtype=np.float64)
    selected = np.asarray(selected_score, dtype=np.float64)
    margin = np.asarray(null_margin, dtype=np.float64)
    if original.ndim != 1 or selected.shape != original.shape or margin.shape != original.shape:
        raise ValueError("세 입력은 같은 길이의 1차원 배열이어야 한다")
    if not all(np.isfinite(a).all() for a in (original, selected, margin)):
        raise ValueError("선택된 후보의 점수와 경로 여유는 유한해야 한다")
    if (original < 0).any() or (selected < 0).any():
        raise ValueError("이 실험의 상관 최대점 점수는 음수가 아니어야 한다")
    attenuation = np.exp(np.minimum(margin, 0.))
    capped = np.minimum(original, selected)
    return {"baseline_score": original.copy(),
            "soft_null": selected*attenuation,
            "capped": capped,
            "soft_capped": capped*attenuation}
