"""같은 대조 후보에서 계열별 평균 손실의 비중만 조정한다."""
import torch
import torch.nn.functional as F


def infonce_rows(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """기존 배치 내 정답 후보와 채굴 오답을 유지한 행별 InfoNCE다."""
    anchors = F.normalize(anchors, dim=1)
    positives = F.normalize(positives, dim=1)
    negatives = F.normalize(negatives, dim=1)
    cross = (anchors @ positives.T) / temperature
    hard = (anchors * negatives).sum(dim=1, keepdim=True) / temperature
    logits = torch.cat([cross, hard], dim=1)
    return F.cross_entropy(
        logits, torch.arange(len(anchors), device=anchors.device), reduction="none"
    )


def equal_family_mean(losses: torch.Tensor, families: torch.Tensor) -> torch.Tensor:
    """두 계열의 평균 오류에 각각 절반의 비중을 준다."""
    if losses.ndim != 1 or families.shape != losses.shape:
        raise ValueError("손실과 계열은 같은 길이의 벡터여야 한다")
    if not bool(((families == 0) | (families == 1)).all()):
        raise ValueError("계열 번호는 0 또는 1이어야 한다")
    if not bool((families == 0).any()) or not bool((families == 1).any()):
        raise ValueError("각 배치에 두 계열이 모두 있어야 한다")
    return 0.5 * losses[families == 0].mean() + 0.5 * losses[families == 1].mean()
