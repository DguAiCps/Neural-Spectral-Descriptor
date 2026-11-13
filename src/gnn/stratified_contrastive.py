"""배치 오답과 명시적 어려운 오답의 대조 손실을 따로 정규화한다."""
import torch
import torch.nn.functional as F
from torch import nn


class StratifiedContrastiveLoss(nn.Module):
    """두 오답 출처의 교차 엔트로피를 같은 비중으로 평균한다.

    0.5 * CE(in-batch) + 0.5 * softplus((hard-positive)/temperature).
    학습 표본과 추론 모델은 바꾸지 않으며 상호정보량 하한을 주장하지 않는다.
    """

    def __init__(self, temperature: float = .1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        """배치 크기에 묻히지 않는 명시적 오답 항을 계산한다."""
        a, p, n = (F.normalize(x, dim=1) for x in (anchors, positives, negatives))
        cross = (a @ p.T) / self.temperature
        hard = (a * n).sum(1) / self.temperature
        positive = cross.diagonal()
        labels = torch.arange(len(a), device=a.device)
        return .5 * (F.cross_entropy(cross, labels) + F.softplus(hard-positive).mean())
