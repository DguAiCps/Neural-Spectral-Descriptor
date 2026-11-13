#!/usr/bin/env python3
"""공식 RING#의 고유 각도 CNN 계산만 작은 묶음으로 나눈다."""
from __future__ import annotations

import numpy as np
import torch
import _ringsharp_official as official


@torch.no_grad()
def translation_branch(base, trans_cnn, bts: torch.Tensor, qfeat: torch.Tensor,
                       cand: np.ndarray, angles: np.ndarray,
                       angle_batch: int = 8) -> np.ndarray:
    """회전·후보·FFT 점수는 유지하며 rotate→CNN 작업 크기만 제한한다."""
    assert angle_batch >= 1
    a1 = torch.tensor(angles, dtype=torch.float)
    a2 = torch.tensor(angles - np.pi, dtype=torch.float)
    uniq, inv = torch.unique(torch.cat([a1, a2]), return_inverse=True)
    qmaps = torch.cat([trans_cnn(base.rotate_bev_batch(qfeat, uniq[k:k + angle_batch]))
                      for k in range(0, len(uniq), angle_batch)])
    inv1, inv2 = inv[:len(a1)], inv[len(a1):]
    errors = []
    # 후보별 FFT 상관의128배치와 계산 순서는 공식 경로와 같다.
    for first in range(0, len(cand), official.BATCH):
        last = first + official.BATCH
        candidate = bts[cand[first:last]]
        q1 = qmaps[inv1[first:last].to(qmaps.device)]
        q2 = qmaps[inv2[first:last].to(qmaps.device)]
        e1 = official.trans_errors_exact(q1, candidate)
        e2 = official.trans_errors_exact(q2, candidate)
        errors.append(np.minimum(e1.detach().numpy().reshape(-1),
                                 e2.detach().numpy().reshape(-1)))
    return np.concatenate(errors)
