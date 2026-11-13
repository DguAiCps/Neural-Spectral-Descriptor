"""고정 승자의 기존 문맥 특징과 학습셋 전용 수락 점수를 계산한다."""
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from retrieval.sequence_evidence import (
    neighbor_plans, path_means, repeated_candidate_weights,
)
from retrieval.spectral_evidence import correlation_profiles

FEATURES = ("baseline", "d_center", "context_center", "key_center",
            "d_arc", "context_arc", "gate_product", "gated_context")
ARMS = {
    "baseline_fit": ("baseline",),
    "add_d_center": ("baseline", "d_center"),
    "add_context_center": ("baseline", "context_center"),
    "add_key_center": ("baseline", "key_center"),
    "add_d_arc": ("baseline", "d_arc"),
    "add_context_arc": ("baseline", "context_arc"),
    "add_d_context_arc": ("baseline", "d_arc", "context_arc"),
}


def unit(value: np.ndarray) -> np.ndarray:
    """영벡터를 보존하면서 행 단위 정규화한다."""
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def pair_features(key: np.ndarray, bank: tuple, path: np.ndarray,
                  query: np.ndarray, winner: np.ndarray, channel: str,
                  raw: np.ndarray) -> Dict[str, np.ndarray]:
    """같은 스펙트럼 방향·대응에서 중심과 시간 평균 특징을 추출한다."""
    q, j = query, winner
    d, c = unit(key[:, :288]), unit(key[:, 288:])
    qn, jn, valid = neighbor_plans(path, q, j)["arc_unique"]
    n = len(path)
    unique, inverse = np.unique((qn*n+jn)[valid], return_inverse=True)
    center = correlation_profiles(bank, q, j, channel).max(1)
    values = np.zeros(qn.shape, dtype=center.dtype)
    values[valid] = correlation_profiles(bank, unique//n, unique % n, channel).max(1)[inverse]
    weights = repeated_candidate_weights(jn, valid)
    means = path_means(center, values, valid, weights)
    direction = np.argmax(means, axis=0)
    beta = (2./3.)*(11./10.)
    result = {"baseline": (1-beta)*raw+beta*means.max(0), "direction": direction}
    for name, vector in (("d", d), ("context", c)):
        center_dot = np.einsum("nd,nd->n", vector[q], vector[j])
        neighbor_dot = np.zeros(qn.shape, dtype=np.float64)
        neighbor_dot[valid] = np.einsum("nd,nd->n", vector[qn[valid]], vector[jn[valid]])
        average = path_means(center_dot, neighbor_dot, valid, weights)
        result[name+"_center"] = center_dot
        result[name+"_arc"] = (1-beta)*center_dot+beta*average[direction, np.arange(len(q))]
    result["key_center"] = np.einsum("nd,nd->n", key[q], key[j])
    alpha = np.linalg.norm(key[:, 288:], axis=1) / np.maximum(np.linalg.norm(key[:, :288], axis=1), 1e-12)
    result["gate_product"] = alpha[q]*alpha[j]
    result["gated_context"] = np.einsum("nd,nd->n", key[q, 288:], key[j, 288:])
    assert all(np.isfinite(v).all() for v in result.values())
    # 이웃의 2층 시간 문맥도 q+15 도착 시점에 모두 계산 가능하다.
    assert np.all(qn[valid]+10 <= np.broadcast_to(q[None, :, None]+15, qn.shape)[valid])
    assert np.all(jn[valid]+10 <= np.broadcast_to(q[None, :, None]+15, qn.shape)[valid])
    return result


def labels(ledger: dict) -> Tuple[np.ndarray, np.ndarray]:
    """모호 질의를 제외하고 현재 승자의 수락 정답을 정한다."""
    distance = ledger["min_past_gt_dist"]
    eligible = (distance <= 5) | (distance > 25)
    correct = (distance <= 5) & (ledger["top1_geo_dist"] <= 5)
    return eligible, correct.astype(np.float64)


def fit_score(x: np.ndarray, y: np.ndarray, weight: np.ndarray,
              ridge: float = 1e-3) -> dict:
    """학습 분포만으로 양의 가중합·편향을 적합한다. 확률 보장은 없다."""
    w = np.asarray(weight, np.float64)
    w = w/w.sum()
    mean = w @ x
    scale = np.maximum(np.sqrt(w @ ((x-mean)**2)), 1e-6)
    z = (x-mean)/scale

    def objective(theta: np.ndarray) -> tuple:
        score = z @ theta[:-1]+theta[-1]
        residual = w*(expit(score)-y)
        loss = w @ (np.logaddexp(0, score)-y*score)+.5*ridge*(theta[:-1] @ theta[:-1])
        grad = np.r_[z.T @ residual+ridge*theta[:-1], residual.sum()]
        return float(loss), grad

    rate = np.clip(w @ y, 1e-6, 1-1e-6)
    initial = np.r_[np.ones(x.shape[1]), np.log(rate/(1-rate))]
    fit = minimize(objective, initial, method="L-BFGS-B", jac=True,
                   bounds=[(1e-6, None)]+[(0., None)]*(x.shape[1]-1)+[(None, None)],
                   options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000})
    if not fit.success:
        raise RuntimeError(str(fit.message))
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": fit.x[:-1].tolist(), "intercept": float(fit.x[-1]),
            "loss": float(fit.fun), "iterations": int(fit.nit), "ridge": ridge}


def predict_score(x: np.ndarray, fitted: dict) -> np.ndarray:
    """포화와 동점을 피하려고 시그모이드 전 점수를 반환한다."""
    return ((x-np.asarray(fitted["mean"]))/np.asarray(fitted["scale"])) @ np.asarray(fitted["coefficients"])+fitted["intercept"]


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """동점에 절반을 주는 정답·오답 쌍의 순서 정확도를 계산한다."""
    positive = y.astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    ranks = rankdata(score, method="average")
    return float((ranks[positive].sum()-n_pos*(n_pos+1)/2)/(n_pos*n_neg))


def discrimination(features: dict, ledger: dict, tau: float) -> dict:
    """전체와 기존 수락 영역의 구분력·분포를 따로 기록한다."""
    eligible, y = labels(ledger)
    result = {}
    for cohort, mask in (("eligible", eligible), ("baseline_accepted", eligible & (features["baseline"] >= tau))):
        rows = {"correct": int(y[mask].sum()), "incorrect": int((1-y[mask]).sum()), "features": {}}
        for name in FEATURES:
            value = features[name][mask]
            rows["features"][name] = {"auc": auc(value, y[mask]), "distribution": {}}
            for kind, use in (("correct", y[mask] == 1), ("incorrect", y[mask] == 0)):
                selected = value[use]
                rows["features"][name]["distribution"][kind] = (
                    {"mean": float(selected.mean()), "q10_q50_q90": np.quantile(selected, [.1,.5,.9]).tolist()}
                    if len(selected) else None)
        result[cohort] = rows
    return result
