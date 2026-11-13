"""기존 출력층을 정규화 서술자 대신 스펙트럼 이득의 입력으로 쓴다."""

import torch
from torch import nn
from torch.nn import functional as F


class SpectralGainReadout(nn.Module):
    """새 층 없이 기존128차원 출력층의 정규화 전 값을 반환한다."""

    def __init__(
        self, reference: nn.Module, initial: torch.Tensor, head_only: bool = False
    ) -> None:
        super().__init__()
        self.net = reference
        self.net.gnn.use_residual_gate = False
        if not isinstance(self.net.gnn.output_proj, nn.Linear):
            raise TypeError("128차원 선형 출력층이 필요하다")
        with torch.no_grad():
            self.net.gnn.output_proj.weight.zero_()
            self.net.gnn.output_proj.bias.copy_(initial)
        self._logits = None
        self._handle = self.net.gnn.output_proj.register_forward_hook(self._capture)
        for name, param in self.net.named_parameters():
            learn = name.startswith("gnn.output_proj.")
            learn |= name.startswith("gnn.convs.") and not head_only
            param.requires_grad_(learn)

    def _capture(self, module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        self._logits = output

    def forward(self, graph) -> torch.Tensor:
        """원 특징 단위 벡터와 정규화 전128개 출력을 함께 반환한다."""
        value = self.net(graph)
        return torch.cat((value[:, :288], self._logits), -1)


class GlobalGainReadout(nn.Module):
    """동일 시작점의128개 공통 이득만 학습하는 대조다."""

    def __init__(self, initial: torch.Tensor) -> None:
        super().__init__()
        self.value = nn.Parameter(initial.clone())

    def forward(self, graph) -> torch.Tensor:
        """스캔 입력과 무관한 출력을 같은416차원 인터페이스로 반환한다."""
        raw = torch.zeros(len(graph.x), 288, device=self.value.device)
        raw[:, 0] = 1
        return torch.cat((raw, self.value[None].expand(len(graph.x), -1)), -1)
