"""과거 전용 검색의 충돌·퍼짐 진단과 검증셋 거리 보정."""
from __future__ import annotations

import numpy as np

CATEGORIES = ("correct", "spread", "collision", "near_field", "no_revisit")


def rank_prefix(bank: np.ndarray, query: int, positions: np.ndarray,
                gap: int = 30) -> dict:
    """과거 후보만 검색하고 가장 높은 정답의 순위와 거리를 반환한다."""
    end = query - gap + 1
    if end <= 0:
        raise ValueError("검색 가능한 과거 후보가 없다")
    scores = bank[:end] @ bank[query]
    geo = np.linalg.norm(positions[:end] - positions[query], axis=1)
    positive = np.flatnonzero(geo <= 5.0)
    top = int(np.argmax(scores))
    result = {"top1_idx": top, "top1_geo": float(geo[top]),
              "min_geo": float(geo.min()), "best_pos_idx": -1,
              "best_pos_rank": -1, "best_pos_dist": float("nan")}
    if positive.size:
        best = int(positive[np.argmax(scores[positive])])
        rank = 1 + int(np.count_nonzero(scores > scores[best]))
        rank += int(np.count_nonzero(scores[:best] == scores[best]))
        result.update(best_pos_idx=best, best_pos_rank=rank,
                      best_pos_dist=float(np.linalg.norm(bank[query] - bank[best])))
    return result


def miss_categories(min_geo: np.ndarray, top_geo: np.ndarray,
                    best_dist: np.ndarray, epsilon: float) -> np.ndarray:
    """정답, 퍼짐, 원거리 충돌, 근거리 오답, 비재방문으로 완전히 나눈다."""
    positive = np.asarray(min_geo) <= 5.0
    if not np.isfinite(np.asarray(best_dist)[positive]).all():
        raise ValueError("재방문 질의의 정답 특징 거리가 유한하지 않다")
    result = np.full(len(positive), 4, dtype=np.int8)
    result[positive] = 0
    miss = positive & (np.asarray(top_geo) > 5.0)
    spread = miss & (np.asarray(best_dist) > epsilon)
    result[spread] = 1
    close_true = miss & ~spread
    result[close_true & (np.asarray(top_geo) >= 25.0)] = 2
    result[close_true & (np.asarray(top_geo) < 25.0)] = 3
    return result


def matched_threshold(values: np.ndarray, weights: np.ndarray,
                      target: float) -> tuple[float, float]:
    """검증 질의의 가중 정답 포괄률을 맞추는 최소 거리 기준을 고른다."""
    values, weights = np.asarray(values), np.asarray(weights)
    if values.ndim != 1 or values.shape != weights.shape or not values.size:
        raise ValueError("거리·가중치 배열의 모양이 맞지 않는다")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("거리·가중치는 유한해야 한다")
    if np.any(weights <= 0) or not 0 <= target <= 1:
        raise ValueError("잘못된 가중치 또는 목표 포괄률")
    order = np.argsort(values, kind="stable")
    weights = weights / weights.sum()
    if target == 0:
        threshold = float(np.nextafter(values.min(), -np.inf))
    else:
        index = min(int(np.searchsorted(np.cumsum(weights[order]), target)),
                    len(values) - 1)
        threshold = float(values[order[index]])
    return threshold, float(weights[values <= threshold].sum())


def transition_counts(before: np.ndarray, after: np.ndarray,
                      positive: np.ndarray) -> dict:
    """고정 재방문 질의에서 회복과 새 오류를 함께 센다."""
    before, after = np.asarray(before, bool), np.asarray(after, bool)
    positive = np.asarray(positive, bool)
    return {"n_positive": int(positive.sum()),
            "recovered": int((positive & ~before & after).sum()),
            "regressed": int((positive & before & ~after).sum()),
            "both_correct": int((positive & before & after).sum()),
            "both_wrong": int((positive & ~before & ~after).sum())}
