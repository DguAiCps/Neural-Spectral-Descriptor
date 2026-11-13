"""중심항 진단의 고정 삼중쌍과 코사인 여유 계산."""
from typing import Dict

import numpy as np
import torch
from scipy.spatial import cKDTree


def fixed_triplets(raw: np.ndarray, xyz: np.ndarray, queries: np.ndarray,
                   exclusion: int = 30) -> np.ndarray:
    """가장 가까운 GT 정답과 원 특징으로 가장 어려운 먼 오답을 고른다.

    정답은 5 m 이내, 오답은 25 m 초과이며 둘 다 q-exclusion 이전이다.
    같은 점수에서는 가장 작은 번호를 택한다. 정답 없는 질의는 제외한다.
    원 특징은 호출자가 재계량한 입력이며 모델 중간 특징을 사용하지 않는다.
    """
    tree = cKDTree(xyz)
    anchors, positives = [], []
    for q in queries:
        candidates = np.asarray(sorted(tree.query_ball_point(xyz[q], 5.)), dtype=np.int64)
        candidates = candidates[candidates <= q-exclusion]
        if len(candidates):
            distances = np.linalg.norm(xyz[candidates]-xyz[q], axis=1)
            anchors.append(int(q))
            positives.append(int(candidates[np.argmin(distances)]))
    if not anchors:
        return np.empty((0, 3), dtype=np.int64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vectors = torch.as_tensor(raw, dtype=torch.float64, device=device)
    vectors = torch.nn.functional.normalize(vectors, dim=1)
    indices = torch.arange(len(raw), device=device)
    negatives = []
    for start in range(0, len(anchors), 128):
        q = np.asarray(anchors[start:start+128])
        scores = vectors[q] @ vectors.T
        # 거리는 원 평가와 같은 3D·float64이며 가까운 후보 목록만 가린다.
        scores[indices[None] > torch.as_tensor(q-exclusion, device=device)[:, None]] = -torch.inf
        for row, anchor in enumerate(q):
            near = tree.query_ball_point(xyz[anchor], 25.)
            scores[row, near] = -torch.inf
        best = scores.argmax(dim=1)
        if not bool(torch.isfinite(scores.gather(1, best[:, None])).all()):
            raise ValueError("25 m 초과 과거 오답이 없는 양성 질의")
        negatives.extend(best.cpu().tolist())
    return np.asarray([anchors, positives, negatives], dtype=np.int64).T


def triplet_statistics(features: np.ndarray, triplets: np.ndarray) -> Dict[str, np.ndarray]:
    """동일 삼중쌍에서 정답·오답 코사인과 차이 및 영벡터 여부를 구한다."""
    out = {k: [] for k in ("positive", "negative", "margin", "valid")}
    for start in range(0, len(triplets), 2048):
        x = features[triplets[start:start+2048]].astype(np.float64)
        norms = np.linalg.norm(x, axis=2)
        x = x / np.maximum(norms[..., None], 1e-12)
        pos = np.einsum("ij,ij->i", x[:, 0], x[:, 1])
        neg = np.einsum("ij,ij->i", x[:, 0], x[:, 2])
        for k, v in {"positive": pos, "negative": neg, "margin": pos-neg,
                     "valid": (norms > 1e-12).all(1)}.items():
            out[k].append(v)
    return {k: np.concatenate(v) if v else np.empty(0, bool if k == "valid" else float)
            for k, v in out.items()}
