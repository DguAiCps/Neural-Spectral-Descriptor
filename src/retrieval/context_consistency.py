"""두 시간 입력에서 후보 간 문맥 유사도 차이를 유지하는 학습 손실."""
import torch
import torch.nn.functional as F


def context_pair_scores(embedding: torch.Tensor, queries: torch.Tensor,
                        candidates: torch.Tensor) -> torch.Tensor:
    """기존 키의 문맥128차원을 단위 길이로 만들어 후보 내적을 구한다."""
    context = F.normalize(embedding[:, 288:], dim=1)
    return (context[queries, None] * context[candidates.clamp_min(0)]).sum(-1)


def relational_consistency(clean: torch.Tensor, changed: torch.Tensor,
                           valid: torch.Tensor, temperature: float = .1) -> torch.Tensor:
    """현재 정상 입력의 상대 점수를 목표로 하는 행별 제곱 손실이다.

    정상 입력 목표는 미분하지 않는다. 각 질의의 후보 평균을 빼 공통 이동은
    제약하지 않고 후보 간 차이를 보존한다. 정답 라벨은 사용하지 않는다.
    """
    if temperature <= 0 or not bool(valid.any(1).all()):
        raise ValueError('양수 온도와 질의별 유효 후보가 필요하다')
    count = valid.sum(1).clamp_min(1)

    def center(value: torch.Tensor) -> torch.Tensor:
        safe = value.masked_fill(~valid, 0.)
        mean = safe.sum(1) / count
        return (safe - mean[:, None]).masked_fill(~valid, 0.)

    difference = (center(changed) - center(clean.detach())) / temperature
    return difference.square().sum(1) / count
