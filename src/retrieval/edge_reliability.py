"""고정 후보에서 간선 신뢰도의 순위·연결 거절 능력을 검사한다."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """양성 비율을 함께 기록하여 AP의 분모를 명시한다."""
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores)
    assert labels.shape == scores.shape and np.isfinite(scores).all()
    n, pos = len(labels), int(labels.sum())
    return {"n": n, "positive": pos, "prevalence": pos / n if n else None,
            "AP": float(average_precision_score(labels, scores)) if pos else None,
            "AUC": float(roc_auc_score(labels, scores)) if 0 < pos < n else None}


def candidate_matrices(query: np.ndarray, values: np.ndarray, nodes: int) -> tuple:
    """오름차순 질의의 최대30개 후보를 패딩하고 원래 순서를 보존한다."""
    query = np.asarray(query, np.int64)
    assert np.all(query[1:] >= query[:-1])
    counts = np.bincount(query, minlength=nodes)
    assert counts.max(initial=0) <= 30
    offset = np.r_[0, np.cumsum(counts)[:-1]]
    slot = np.arange(len(query)) - offset[query]
    shape = (nodes, 30, *values.shape[1:])
    matrix = np.full(shape, np.nan, np.float64)
    matrix[query, slot] = values
    return matrix, counts


def query_metrics(scores: np.ndarray, distances: np.ndarray, valid: np.ndarray) -> Dict[str, np.ndarray]:
    """질의 내부 AUC와 정답 보존을 계산한다. 정답 없는 질의도 남긴다."""
    assert scores.shape == distances.shape and scores.ndim == 2
    present = np.isfinite(distances)
    assert np.isfinite(scores[present]).all()
    positive, negative = present & (distances < 5), present & (distances >= 25)
    correct = present & (distances <= 5)
    labeled = positive | negative
    npos, nneg = positive.sum(1), negative.sum(1)
    ranked = rankdata(np.where(labeled, scores, -np.inf), axis=1, method="average")
    ranked -= (~labeled).sum(1)[:, None]
    auc = np.full(len(scores), np.nan)
    mixed = valid & (npos > 0) & (nneg > 0)
    u = (ranked * positive).sum(1) - npos * (npos + 1) / 2
    auc[mixed] = u[mixed] / (npos[mixed] * nneg[mixed])
    order = np.argsort(-np.where(present, scores, -np.inf), axis=1, kind="stable")
    ordered_correct = np.take_along_axis(correct, order, axis=1)
    nselected = np.minimum(present.sum(1), 10)
    return {"within_query_AUC": auc, "has_correct": correct.any(1),
            "top1_correct": ordered_correct[:, 0], "top10_correct_count": ordered_correct[:, :10].sum(1),
            "top10_has_correct": ordered_correct[:, :10].any(1), "selected_count": nselected,
            "max_score": np.max(np.where(present, scores, -np.inf), axis=1)}


def summarize_queries(metrics: dict, mask: np.ndarray) -> dict:
    """서로 다른 분모인 조건부 순위와 전체 질의의 정답 보존을 구분한다."""
    n = int(mask.sum())
    has = mask & metrics["has_correct"]
    mixed = mask & np.isfinite(metrics["within_query_AUC"])
    return {"queries": n, "candidate_has_correct": int(has.sum()), "mixed_queries": int(mixed.sum()),
            "within_query_AUC": float(metrics["within_query_AUC"][mixed].mean()) if mixed.any() else None,
            "hit1_given_available": float(metrics["top1_correct"][has].mean()) if has.any() else None,
            "hit10_given_available": float(metrics["top10_has_correct"][has].mean()) if has.any() else None,
            "hit10_all_queries": float(metrics["top10_has_correct"][mask].mean()) if n else None,
            "selected_edges": int(metrics["selected_count"][mask].sum()),
            "selected_correct": int(metrics["top10_correct_count"][mask].sum())}


def retention_threshold(positive_scores: np.ndarray, retention: float = .95) -> float:
    """확인용 양성 질의를 최소 지정 비율만큼 보존하는 점수 경계를 정한다."""
    values = np.sort(np.asarray(positive_scores))
    assert len(values) and np.isfinite(values).all() and 0 < retention <= 1
    discard = int(np.floor((1 - retention) * len(values) + 1e-10))
    return float(values[discard])


def confidence_bins(scores: np.ndarray, distances: np.ndarray) -> list:
    """출력 점수와 실제 정답 비율을 고정된 열 구간에서 비교한다."""
    assert scores.shape == distances.shape and np.isfinite(scores).all()
    ids = np.minimum((scores * 10).astype(np.int64), 9)
    assert np.all((ids >= 0) & (ids <= 9))
    rows = []
    for i in range(10):
        m = ids == i
        n = int(m.sum())
        rows.append({"lower": i / 10, "upper": (i + 1) / 10, "edges": n,
                     "score_sum": float(scores[m].astype(np.float64).sum()),
                     "correct": int((distances[m] <= 5).sum()), "far": int((distances[m] > 25).sum()),
                     "ambiguous": int(((distances[m] > 5) & (distances[m] <= 25)).sum())})
    return rows
