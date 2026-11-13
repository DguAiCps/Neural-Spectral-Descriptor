"""관측 범위를 고정한 시퀀스 대응과 중복 후보 가중치를 비교한다."""
from typing import Dict, Tuple

import numpy as np

from .spectral_evidence import correlation_profiles


def distance_indices(path: np.ndarray, center: np.ndarray,
                     distance: np.ndarray, radius: int = 10) -> np.ndarray:
    """제한 창에서 지정한 누적 이동 거리와 가장 가까운 프레임을 고른다."""
    target = path[center] + distance
    left, right = np.maximum(0, center-radius), np.minimum(len(path)-1, center+radius)
    lower = np.clip(np.searchsorted(path, target, side="left")-1, left, right)
    upper = np.clip(lower+1, left, right)
    return np.where(np.abs(path[lower]-target) <= np.abs(path[upper]-target), lower, upper)


def neighbor_plans(path: np.ndarray, query: np.ndarray,
                   candidate: np.ndarray) -> Dict[str, tuple]:
    """각 규칙의 질의·후보 이웃과 유효 마스크를 만든다."""
    q, j = np.asarray(query, np.int64), np.asarray(candidate, np.int64)
    dense = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    result = {}
    for style in ("arc_sparse", "frame_dense", "arc_dense", "linear_dense", "velocity_dense"):
        offsets = np.array([-5, 5]) if style == "arc_sparse" else dense
        directions = (1, -1) if style != "velocity_dense" else (0.5, -0.5, 1., -1., 1.5, -1.5, 2., -2.)
        qs, js, masks = [], [], []
        for direction in directions:
            qn = q[:, None] + offsets[None]
            qsafe = np.clip(qn, 0, len(path)-1)
            if style.startswith("arc"):
                jn = distance_indices(path, j[:, None], direction*(path[qsafe]-path[q, None]))
            elif style == "linear_dense":
                ends = q[:, None] + np.array([-5, 5])[None]
                ends = np.clip(ends, 0, len(path)-1)
                endpoints = distance_indices(path, j[:, None], direction*(path[ends]-path[q, None]))
                delta = np.where(offsets[None] < 0, endpoints[:, :1]-j[:, None], endpoints[:, 1:]-j[:, None])
                jn = j[:, None] + np.rint(delta*np.abs(offsets)[None]/5).astype(np.int64)
            else:
                jn = j[:, None] + np.rint(direction*offsets)[None].astype(np.int64)
            valid = (qn >= 0) & (qn < len(path)) & (jn >= 0) & (jn < len(path)) & (jn != j[:, None])
            qs.append(qn)
            js.append(jn)
            masks.append(valid)
        result[style] = np.stack(qs), np.stack(js), np.stack(masks)
    result["arc_unique"] = result["arc_dense"]
    return result


def repeated_candidate_weights(indices: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """같은 후보 프레임에 연결된 쌍들의 가중치 합을 1로 맞춘다."""
    same = (indices[..., :, None] == indices[..., None, :]) & valid[..., None, :]
    count = same.sum(-1)
    return valid / np.maximum(count, 1)


def path_means(center: np.ndarray, peaks: np.ndarray, valid: np.ndarray,
               weights: np.ndarray = None) -> np.ndarray:
    """중심을 한 번 포함한 경로별 평균을 계산한다."""
    total = np.broadcast_to(center[None], peaks.shape[:2]).copy()
    if weights is None:
        count = np.ones(total.shape, dtype=np.float64)
        # 기존 3쌍 판정의 FP32 덧셈 순서를 유지한다.
        for k in range(peaks.shape[-1]):
            total += np.where(valid[..., k], peaks[..., k], 0)
            count += valid[..., k]
    else:
        total = total.astype(np.float64) + (peaks*weights).sum(-1)
        count = 1+weights.sum(-1)
    return total/count


def sequence_scores(bank: tuple, path: np.ndarray, query: np.ndarray,
                    candidate: np.ndarray, channel: str,
                    raw: np.ndarray = None) -> Tuple[dict, dict]:
    """중심·이웃의 동일 스펙트럼으로 고정된 21개 판정 조건을 계산한다."""
    q, j = np.asarray(query, np.int64), np.asarray(candidate, np.int64)
    plans = neighbor_plans(path, q, j)
    n = len(path)
    pair_ids = [q*n+j]
    for name, (qn, jn, valid) in plans.items():
        if name != "arc_unique":
            pair_ids.append((qn*n+jn)[valid])
    unique = np.unique(np.concatenate(pair_ids))
    peaks = correlation_profiles(bank, unique//n, unique % n, channel).max(1)
    center = peaks[np.searchsorted(unique, q*n+j)]
    raw = center if raw is None else raw
    scores, cost = {"raw": raw}, {}
    for name, (qn, jn, valid) in plans.items():
        values = np.zeros(qn.shape, dtype=peaks.dtype)
        values[valid] = peaks[np.searchsorted(unique, (qn*n+jn)[valid])]
        weights = repeated_candidate_weights(jn, valid) if name == "arc_unique" else None
        means = path_means(center, values, valid, weights)
        # 전체 이웃이 있으면 세 쌍 대조와 중심 이외의 총비중이 같다.
        scale = 1. if name == "arc_sparse" else (2./3.)*(11./10.)
        for base_beta in (.25, .5, 1.):
            beta = base_beta*scale
            key = f"{name}_b{int(base_beta*100)}"
            scores[key] = (1-beta)*raw+beta*means.max(0)
            if name == "arc_dense" and base_beta == .25:
                scores["arc_forward_b25"] = (1-beta)*raw+beta*means[0]
                scores["arc_reverse_b25"] = (1-beta)*raw+beta*means[1]
        pair_count = np.zeros(len(q), np.int64)
        for b in range(len(q)):
            ids = (qn[:, b]*n+jn[:, b])[valid[:, b]]
            pair_count[b] = len(np.unique(np.r_[q[b]*n+j[b], ids]))
        cost[name] = {"pair_count": pair_count, "support": valid.sum(-1),
                      "effective_neighbors": weights.sum(-1) if weights is not None else valid.sum(-1)}
    return scores, cost
