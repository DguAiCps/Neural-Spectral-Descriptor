"""기존 키가 맞게 구분한 학습 삼중쌍의 점수 차이를 보호한다."""
import torch


def triplet_margin(key: torch.Tensor, triplets: torch.Tensor) -> torch.Tensor:
    """정규화된 키에서 정답 코사인과 오답 코사인의 차이를 돌려준다."""
    query, positive, negative = triplets.unbind(dim=1)
    return (key[query] * (key[positive] - key[negative])).sum(dim=1)


def teacher_margin_loss(
    current: torch.Tensor, reference: torch.Tensor, temperature: float = 0.1,
) -> torch.Tensor:
    """기준 모델의 양수 마진을 줄인 경우에만 제곱 벌점을 준다."""
    if temperature <= 0:
        raise ValueError("온도는 양수여야 한다")
    if current.shape != reference.shape:
        raise ValueError("현재와 기준 마진의 모양이 같아야 한다")
    target = reference.detach()
    penalty = ((target - current).relu() / temperature).square()
    return (penalty * (target > 0)).sum() / max(current.numel(), 1)
