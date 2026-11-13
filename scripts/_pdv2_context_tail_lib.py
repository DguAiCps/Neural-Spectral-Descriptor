"""학습 내부의 높은 점수 오답을 낮추는 두 점수 판정 손실이다."""
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from _pdv2_context_accept_lib import labels
from _pdv2_context_place_lib import fit_fold, score

ARMS = {
    "top1pct_l1": {"fraction": .01, "strength": 1.},
    "top1pct_l4": {"fraction": .01, "strength": 4.},
    "worst_l1": {"fraction": 0., "strength": 1.},
    "worst_l4": {"fraction": 0., "strength": 4.},
}


def training_arrays(data: Dict[str, dict], heldout: Tuple[str, ...],
                    fraction: float) -> dict:
    """제외 장소를 읽지 않고 BCE 기준·오답 집합·균등 가중치를 만든다."""
    initial = fit_fold(data, heldout, "context_arc")
    names = initial["training"]
    families = sorted({s.split("/")[0] for s in names})
    xs, ys, weights, pieces = [], [], [], []
    offset = 0
    for name in names:
        row = data[name]
        keep, target = labels(row)
        ids = np.flatnonzero(keep)
        x = np.column_stack((row["baseline"][keep], row["context_arc"][keep]))
        y = target[keep]
        family = name.split("/")[0]
        count = sum(n.split("/")[0] == family for n in names)
        w = 1. / (len(families) * count)
        xs.append(x)
        ys.append(y)
        weights.append(np.full(len(x), w / len(x)))
        negative = np.flatnonzero(y == 0)
        positive = np.flatnonzero(y == 1)
        ranking = score(row, initial, 1.)[keep]
        k = max(1, int(np.ceil(fraction * len(negative))))
        hard = negative[np.argsort(-ranking[negative], kind="stable")[:k]]
        pieces.append({"sequence": name, "weight": w,
                       "positive": positive + offset, "hard": hard + offset,
                       "hard_query_idx": row["query_idx"][ids[hard]].tolist(),
                       "negative_count": len(negative), "positive_count": len(positive)})
        offset += len(x)
    x = np.concatenate(xs)
    z = (x - np.asarray(initial["mean"])) / np.asarray(initial["scale"])
    # 순위 손실은 정답·오답이 모두 있는 시퀀스에서 계열→시퀀스 균등이다.
    usable = [p for p in pieces if len(p["positive"]) and len(p["hard"])]
    groups = sorted({p["sequence"].split("/")[0] for p in usable})
    pairs = []
    for p in usable:
        family = p["sequence"].split("/")[0]
        count = sum(q["sequence"].split("/")[0] == family for q in usable)
        pairs.append((z[p["hard"]].mean(0) - z[p["positive"]],
                      1. / (len(groups) * count)))
    return {"initial": initial, "z": z, "y": np.concatenate(ys),
            "weight": np.concatenate(weights), "pairs": pairs,
            "mining": [{k: v for k, v in p.items() if k not in ("positive", "hard")}
                       for p in pieces]}


def objective(theta: np.ndarray, arrays: dict, strength: float,
              ridge: float = .001) -> tuple:
    """BCE와 고정 상위 오답 평균보다 정답을 높이는 순위 손실을 계산한다."""
    z, y, weight = arrays["z"], arrays["y"], arrays["weight"]
    value = z @ theta[:2] + theta[2]
    residual = weight * (expit(value) - y)
    loss = weight @ (np.logaddexp(0, value) - y * value)
    loss += .5 * ridge * (theta[:2] @ theta[:2])
    gradient = np.r_[z.T @ residual + ridge * theta[:2], residual.sum()]
    for difference, mass in arrays["pairs"]:
        gap = difference @ theta[:2]
        loss += strength * mass * np.logaddexp(0, gap).mean()
        gradient[:2] += strength * mass * (difference.T @ expit(gap)) / len(gap)
    return float(loss), gradient


def fit_tail(data: Dict[str, dict], heldout: Tuple[str, ...], fraction: float,
             strength: float) -> dict:
    """두 양수 계수만 학습하며 최종 검색·인코더를 고정한다."""
    arrays = training_arrays(data, heldout, fraction)
    initial = arrays["initial"]
    theta = np.r_[initial["coefficients"], initial["intercept"]]
    fitted = minimize(objective, theta, args=(arrays, strength), jac=True,
                      method="L-BFGS-B", bounds=[(1e-6, None), (0., None), (None, None)],
                      options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000})
    if not fitted.success:
        raise RuntimeError(str(fitted.message))
    value, gradient = objective(fitted.x, arrays, strength)
    projected = gradient.copy()
    for i, lower in ((0, 1e-6), (1, 0.)):
        if fitted.x[i] <= lower + 1e-10:
            projected[i] = min(0., gradient[i])
    if np.max(np.abs(projected)) > 2e-5:
        raise RuntimeError("수렴 기울기 확인 실패")
    raw = fitted.x[:2] / np.asarray(initial["scale"])
    return {**initial, "coefficients": fitted.x[:2].tolist(),
            "intercept": float(fitted.x[2]), "loss": value,
            "iterations": int(fitted.nit), "fraction": fraction, "strength": strength,
            "gradient": gradient.tolist(), "projected_gradient_max": float(abs(projected).max()),
            "raw_context_ratio": float(raw[1] / raw[0]), "mining": arrays["mining"],
            "mining_model": initial}
