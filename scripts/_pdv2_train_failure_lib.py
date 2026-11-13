"""과거 지도 오답 쌍의 배치 노출과 고정 임베딩 손실을 계산한다."""
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F


def wrong_place_neighbors(positions: np.ndarray, sequences: np.ndarray,
                          winner: np.ndarray, radius: float = 5.0) -> csr_matrix:
    """승자와 5m 이내이며 질의보다 30프레임 이상 과거인 동일 세션 점이다."""
    rows, columns = [], []
    for sequence in np.unique(sequences):
        ids = np.flatnonzero(sequences == sequence)
        assert np.all(np.diff(ids) == 1)
        queries = ids[winner[ids] >= 0]
        assert np.all(sequences[winner[queries]] == sequence)
        neighbors = cKDTree(positions[ids]).query_ball_point(positions[winner[queries]], radius)
        for query, local in zip(queries, neighbors):
            candidates = ids[np.asarray(local, dtype=np.int64)]
            candidates = candidates[candidates <= query - 30]
            rows.extend([query] * len(candidates))
            columns.extend(candidates)
    return csr_matrix((np.ones(len(rows), dtype=bool), (rows, columns)),
                      shape=(len(positions), len(positions)))


def batch_membership(batch: np.ndarray, winner: np.ndarray,
                     nearby: csr_matrix) -> dict:
    """각 앵커의 실제 승자 및 같은 오답 장소가 손실 항에 있는지 센다."""
    a, p, n = batch.T
    exact = winner[a, None] == p[None]
    np.fill_diagonal(exact, False)
    close = nearby[a][:, p].toarray()
    np.fill_diagonal(close, False)
    hard_exact = winner[a] == n
    hard_close = np.asarray(nearby[a, n]).ravel()
    reverse = winner[p][None] == a[:, None]
    np.fill_diagonal(reverse, False)
    return {"exact": exact, "nearby": close, "hard_exact": hard_exact,
            "hard_nearby": hard_close, "reverse": reverse,
            "hard_reverse": winner[n] == a}


def batch_geometry(unit: torch.Tensor, batch: torch.Tensor,
                   temperature: float = 0.1) -> Tuple[torch.Tensor, ...]:
    """InfoNCE의 행별 손실·확률과 정규화 임베딩에 대한 세 역할 기울기다."""
    a, p, n = batch.T
    x, y, z = unit[a], unit[p], unit[n]
    logits = torch.cat((x @ y.T, (x * z).sum(1, keepdim=True)), dim=1) / temperature
    probability = logits.softmax(1)
    size = len(a)
    loss = torch.logsumexp(logits, dim=1) - logits[:, :size].diag()
    cross = probability[:, :size].clone()
    cross[torch.arange(size, device=unit.device), torch.arange(size, device=unit.device)] -= 1
    hard = probability[:, -1:]
    return (loss, probability, (cross @ y + hard * z) / temperature,
            cross.T @ x / temperature, hard * x / temperature)


def normalization_gradient(unit: torch.Tensor, norm: torch.Tensor,
                           gradient: torch.Tensor) -> torch.Tensor:
    """L2 정규화의 미분으로 정규화 전 출력의 기울기를 구한다."""
    return (gradient - unit * (gradient * unit).sum(1, keepdim=True)) / norm


def cosine_descent_response(raw: torch.Tensor, gradient: torch.Tensor,
                            q: torch.Tensor, j: torch.Tensor,
                            context_start: int = 288) -> torch.Tensor:
    """문맥 출력만 독립적으로 갱신할 때 오답 코사인의 1차 변화율이다."""
    norm = raw.norm(dim=1, keepdim=True)
    unit = F.normalize(raw, dim=1)
    cosine = (unit[q] * unit[j]).sum(1, keepdim=True)
    dq = (unit[j] - cosine * unit[q]) / norm[q]
    dj = (unit[q] - cosine * unit[j]) / norm[j]
    return -((gradient[q, context_start:] * dq[:, context_start:]).sum(1)
             + (gradient[j, context_start:] * dj[:, context_start:]).sum(1))
