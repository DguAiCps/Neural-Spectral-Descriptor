"""장소를 통째로 뺀 점수 적합과 문맥 비중의 보수적 선택을 지원한다."""
from typing import Dict, List, Tuple

import numpy as np

from _pdv2_context_accept_lib import fit_score, labels

PLACES = {
    "kitti_highway": ("kitti/01",),
    "kitti_route02": ("kitti/02",),
    "kitti_urban": ("kitti/05", "kitti/06"),
    "helipr_bridge": tuple(f"helipr/Bridge0{i}" for i in (1, 2, 3, 4)),
}
FACTORS = (0., .05, .1, .25, .5, 1.)
FEATURES = ("context_arc", "d_arc")


def fit_fold(data: Dict[str, dict], heldout: Tuple[str, ...], feature: str) -> dict:
    """제외한 장소의 라벨·통계를 쓰지 않고 두 점수의 가중치를 맞춘다."""
    training = [s for s in data if s not in heldout]
    groups = sorted({s.split("/")[0] for s in training})
    xs, ys, weights = [], [], []
    for sequence in training:
        row = data[sequence]
        keep, y = labels(row)
        family = sequence.split("/")[0]
        n_seq = sum(s.split("/")[0] == family for s in training)
        xs.append(np.column_stack((row["baseline"][keep], row[feature][keep])))
        ys.append(y[keep])
        weights.append(np.full(int(keep.sum()), 1/(len(groups)*n_seq*keep.sum())))
    result = fit_score(np.concatenate(xs), np.concatenate(ys), np.concatenate(weights))
    return {"fields": ["baseline", feature], "training": training, "heldout": list(heldout), **result}


def score(row: dict, fitted: dict, factor: float) -> np.ndarray:
    """학습한 문맥 또는 원래 특징의 비중만 지정 비율로 줄인다."""
    x = np.column_stack([row[f] for f in fitted["fields"]])
    coefficients = np.array(fitted["coefficients"], np.float64)
    coefficients[1] *= factor
    return ((x-np.asarray(fitted["mean"]))/np.asarray(fitted["scale"])) @ coefficients+fitted["intercept"]


def false_rate_threshold(values: np.ndarray, budget: float = .01) -> float:
    """학습 오답의 수락률이 예산 이하인 유한 임계값을 보수적으로 정한다."""
    ordered = np.sort(np.asarray(values, np.float64))[::-1]
    if not len(ordered):
        raise ValueError("학습 오답이 필요하다")
    allowed = int(np.floor(budget*len(ordered)))
    return float(np.nextafter(ordered[allowed], np.inf))


def robust_pass(deltas: List[dict], mean_ap_gain: float) -> bool:
    """전체 제외 장소·시드의 AP·R100·오답 수락률 조건을 적용한다."""
    if mean_ap_gain < .001:
        return False
    for row in deltas:
        if row["n_pos"]:
            if row["AP"] < -.001 or row["R@100P"] < -.001:
                return False
        elif row["incorrect_accept_rate"] > .001:
            return False
    return True


def choose(grid: List[dict]) -> dict:
    """동률이면 문맥 비중이 작은 조건을 고르고 통과가 없으면 0으로 둔다."""
    mean_ap = max(grid, key=lambda r: (r["AP"], -r["factor"]))
    eligible = [r for r in grid if r["pass"]]
    robust = max(eligible, key=lambda r: (r["AP"], -r["factor"])) if eligible else next(r for r in grid if r["factor"] == 0)
    return {"mean_ap_factor": mean_ap["factor"], "robust_factor": robust["factor"],
            "eligible_factors": [r["factor"] for r in eligible]}
