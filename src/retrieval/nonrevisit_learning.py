"""정답 없는 질의를 포함한 문맥 최종 판정의 미분 가능한 연산."""
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .sequence_evidence import neighbor_plans, repeated_candidate_weights


def context_plan(path: np.ndarray, query: np.ndarray,
                 candidate: np.ndarray) -> Dict[str, np.ndarray]:
    """현재 거리 대응을 중심 포함 가중 내적 계획으로 변환한다."""
    q, j = np.asarray(query, np.int64), np.asarray(candidate, np.int64)
    qn, jn, valid = neighbor_plans(path, q, j)["arc_unique"]
    weights = repeated_candidate_weights(jn, valid)
    weights = weights / (1.0 + weights.sum(-1, keepdims=True))
    center_weight = 1.0 - weights.sum(-1, keepdims=True)
    weights = np.concatenate((center_weight, weights), axis=-1) * (11.0 / 15.0)
    weights[..., 0] += 4.0 / 15.0
    shape = qn.shape[:2] + (1,)
    qi = np.concatenate((np.broadcast_to(q[None, :, None], shape), qn), axis=-1)
    ji = np.concatenate((np.broadcast_to(j[None, :, None], shape), jn), axis=-1)
    return {"query": np.clip(qi, 0, len(path) - 1).transpose(1, 0, 2).copy(),
            "candidate": np.clip(ji, 0, len(path) - 1).transpose(1, 0, 2).copy(),
            "weight": weights.transpose(1, 0, 2).astype(np.float32)}


def context_score(embedding: torch.Tensor, plan: Dict[str, torch.Tensor],
                  descriptor_dim: int = 288) -> torch.Tensor:
    """정규화한 문맥의 앞뒤 경로 점수 중 큰 값을 반환한다."""
    context = F.normalize(embedding[:, descriptor_dim:], dim=-1, eps=1e-12)
    dots = (context[plan["query"]] * context[plan["candidate"]]).sum(-1)
    return (dots * plan["weight"]).sum(-1).max(-1).values


def acceptance_loss(logits: torch.Tensor, correct: torch.Tensor,
                    positive_query: torch.Tensor, include_nonrevisit: bool) -> torch.Tensor:
    """현재 승자의 정오를 학습하며 양성 질의 한정 대조도 지원한다."""
    value = F.binary_cross_entropy_with_logits(logits, correct.to(logits.dtype), reduction="none")
    mask = torch.ones_like(positive_query) if include_nonrevisit else positive_query
    return (value * mask).sum() / mask.sum().clamp_min(1)
