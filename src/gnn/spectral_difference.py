"""라이다 방향을 맞춘 복소 스펙트럼의 시간 차이를 전달한다."""

from typing import Optional

import torch


def transport_difference(
    spectrum: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor,
    phase: torch.Tensor,
    strengths: torch.Tensor,
    attentions: Optional[torch.Tensor] = None,
    transport: bool = True,
) -> torch.Tensor:
    """두 층의 이웃 차이를 더한다. 주의 머리 네 개는 주파수 두 개씩 맡는다.

    spectrum은 (N,16,8), indices/mask는 (N,11), phase는 (N,8),
    strengths는 (2,N,1), attentions는 (2,N,11,4)다.
    자기 자신 칸의 mask는 0이며, 이웃이 없으면 원본을 보존한다.
    """
    if strengths.shape != (2, len(spectrum), 1):
        raise ValueError("두 층의 노드별 세기 (2,N,1)가 필요하다")
    norm = spectrum.abs().flatten(1).norm(dim=1).clamp_min(1e-12)
    world = spectrum / norm[:, None, None]
    if transport:
        world = world * phase.conj()[:, None]
    if attentions is None:
        weights = mask / mask.sum(1, keepdim=True).clamp_min(1)
        weights = weights[None, :, :, None].expand(2, -1, -1, 4)
    else:
        weights = attentions
    for layer in range(2):
        update = torch.zeros_like(world)
        for slot in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10):
            a = weights[layer, :, slot].repeat_interleave(2, dim=-1)
            a = a * mask[:, slot, None]
            delta = world[indices[:, slot]] - world
            update = update + a[:, None] * delta
        world = world + strengths[layer, :, :, None] * update
    return world * phase[:, None] if transport else world
