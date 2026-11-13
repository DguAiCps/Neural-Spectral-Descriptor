"""정렬 전 검색 키에 위치 정답과 신뢰 가능한 정렬 순서를 전달한다."""
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from .candidate_learning import masked_minmax
from .fixed_fusion_learning import positive_ranking_loss


def alignment_ranking_loss(
    key: torch.Tensor,
    auxiliary: torch.Tensor,
    teacher: torch.Tensor,
    correct: torch.Tensor,
    present: torch.Tensor,
    eligible: torch.Tensor,
    weights: Sequence[float],
    mode: str,
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """키·정렬 전 융합의 정답 순위와 위치 정답에 맞는 교사 분포를 학습한다.

    present는 실제 후보, eligible은 5m 이내 정답 또는 25m 밖 오답이다.
    회색 구간은 추론과 같은 minmax에는 참여하지만 정답/오답 손실에서 제외한다.
    교사의 전체 후보 중 최고 점수가 위치 정답일 때만 증류를 허용한다.
    """
    if mode not in ('key', 'joint', 'teacher') or temperature <= 0:
        raise ValueError('지원하지 않는 순위 손실 또는 온도')
    if not bool(present.any(1).all()) or not bool(eligible.any(1).all()):
        raise ValueError('각 행에 실제 후보와 손실 후보가 필요하다')
    if key.shape != teacher.shape or key.shape != auxiliary.shape:
        raise ValueError('후보 점수의 모양이 같아야 한다')
    eligible = eligible & present
    k = masked_minmax(key, present)
    a = masked_minmax(auxiliary.detach(), present)
    fused = (float(weights[0]) * k + float(weights[1]) * a) / float(weights[0] + weights[1])
    key_loss, active = positive_ranking_loss(k, correct, eligible, temperature)
    joint_loss, joint_active = positive_ranking_loss(fused, correct, eligible, temperature)
    assert torch.equal(active, joint_active)
    count = active.sum().clamp_min(1)
    label = key_loss if mode == 'key' else .5 * (key_loss + joint_loss)
    t = masked_minmax(teacher.detach(), present)
    winner = teacher.detach().masked_fill(~present, -torch.inf).argmax(1)
    trusted = correct.gather(1, winner[:, None]).squeeze(1) & active
    target_logits = (t / temperature).masked_fill(~eligible, -1e9)
    target = F.softmax(target_logits, dim=1).detach()
    target_log = F.log_softmax(target_logits, dim=1).detach()
    divergences = []
    for student in (k, fused):
        log_student = F.log_softmax((student / temperature).masked_fill(~eligible, -1e9), dim=1)
        divergences.append((target * (target_log - log_student)).sum(1))
    divergence = .5 * (divergences[0] + divergences[1])
    distill = (divergence * trusted).sum() / count
    loss = label.sum() / count + (distill if mode == 'teacher' else 0.)
    return loss, dict(key=key_loss.sum() / count, joint=joint_loss.sum() / count,
                      distill=distill, active=active.sum(), trusted=trusted.sum())
