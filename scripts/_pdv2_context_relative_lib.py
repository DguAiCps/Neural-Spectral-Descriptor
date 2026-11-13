"""질의마다 허용된 과거 평균만 사용하는 문맥 점수 보정이다."""
from typing import Dict

import numpy as np

from retrieval.sequence_evidence import neighbor_plans, repeated_candidate_weights

STYLES = ("excess", "centered_dot", "centered_cos")
BETA = (2. / 3.) * (11. / 10.)
RESIDUAL_EPS = 1e-6


def unit64(value: np.ndarray) -> np.ndarray:
    """길이가 0인 벡터를 보존하며 FP64 단위 벡터를 만든다."""
    value = np.asarray(value, dtype=np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def weighted_pairs(path: np.ndarray, query: np.ndarray, winner: np.ndarray,
                   direction: np.ndarray) -> tuple:
    """기존 스펙트럼 방향과 반복 보정에 따른 중심·이웃 가중치를 반환한다."""
    query, winner = np.asarray(query, np.int64), np.asarray(winner, np.int64)
    direction = np.asarray(direction, np.int64)
    assert np.all(query >= 30) and np.all(query + 15 < len(path))
    assert np.all((winner >= 0) & (winner <= query - 30))
    assert np.all((direction == 0) | (direction == 1))
    qn, jn, valid = neighbor_plans(path, query, winner)["arc_unique"]
    weights = repeated_candidate_weights(jn, valid)
    ids = np.arange(len(query))
    qn, jn = qn[direction, ids], jn[direction, ids]
    valid, weights = valid[direction, ids], weights[direction, ids]
    denominator = 1. + weights.sum(1)
    mass = np.column_stack((1. - BETA + BETA / denominator,
                           BETA * weights / denominator[:, None]))
    # 무효 이웃은 중심으로 치환하되 가중치가 0이다.
    qn = np.where(valid, qn, query[:, None])
    jn = np.where(valid, jn, winner[:, None])
    qa = np.column_stack((query, qn))
    ja = np.column_stack((winner, jn))
    assert np.all(qa + 10 <= query[:, None] + 15)
    assert np.all(ja + 10 <= query[:, None] + 15)
    np.testing.assert_allclose(mass.sum(1), 1., atol=1e-14)
    return qa, ja, mass


def relative_features(key: np.ndarray, path: np.ndarray, row: dict,
                      chunk_size: int = 256) -> Dict[str, np.ndarray]:
    """과거 0..q−30 평균을 사용해 같은 승자의 보정 점수 세 가지를 구한다."""
    q = np.asarray(row["query_idx"], np.int64)
    winner = np.asarray(row["j_star"], np.int64)
    direction = np.asarray(row["direction"], np.int64)
    assert key.ndim == 2 and key.shape[1] == 416 and len(key) == len(path)
    assert np.isfinite(key).all() and np.isfinite(path).all()
    assert np.all(np.diff(path) >= 0)
    output = {}
    for name, columns in (("d", slice(0, 288)), ("context", slice(288, 416))):
        vectors = unit64(key[:, columns])
        cumulative = np.cumsum(vectors, axis=0, dtype=np.float64)
        parts = {kind: [] for kind in ("arc_check", *STYLES, "background", "mean_norm", "residual_min")}
        for start in range(0, len(q), chunk_size):
            stop = min(len(q), start + chunk_size)
            query = q[start:stop]
            qa, ja, mass = weighted_pairs(path, query, winner[start:stop], direction[start:stop])
            count = query - 30 + 1
            mean = cumulative[count - 1] / count[:, None]
            left, right = vectors[qa], vectors[ja]
            center_dot = np.einsum("nkd,nkd->nk", left, right)
            background = np.einsum("nkd,nd->nk", left, mean)
            left_residual, right_residual = left - mean[:, None], right - mean[:, None]
            dot = np.einsum("nkd,nkd->nk", left_residual, right_residual)
            left_norm = np.linalg.norm(left_residual, axis=2)
            right_norm = np.linalg.norm(right_residual, axis=2)
            cosine = dot / (np.maximum(left_norm, RESIDUAL_EPS) *
                            np.maximum(right_norm, RESIDUAL_EPS))
            for kind, value in (("arc_check", center_dot), ("excess", center_dot - background),
                                ("centered_dot", dot), ("centered_cos", cosine),
                                ("background", background)):
                parts[kind].append(np.sum(mass * value, axis=1))
            parts["mean_norm"].append(np.linalg.norm(mean, axis=1))
            residual_min = np.minimum(left_norm, right_norm)
            parts["residual_min"].append(np.where(mass > 0, residual_min, np.inf).min(1))
        for kind, values in parts.items():
            output[name + "_" + kind] = np.concatenate(values)
    assert all(v.shape == q.shape and np.isfinite(v).all() for v in output.values())
    return output
