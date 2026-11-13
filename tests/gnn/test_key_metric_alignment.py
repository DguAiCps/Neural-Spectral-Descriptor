"""거리 정합이 원 특징 출력만 바꾸고 시간 의존 범위를 지키는지 검사한다."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]
import _pdv2_key_metric_alignment as run


def test_raw_metric_only_and_frozen_scale():
    """같은 문맥·간선에서 원 특징 재계량만 바꾸고 유한 기울기를 확인한다."""
    control, aligned = run.model("control", 1).eval(), run.model("aligned", 1).eval()
    rng = np.random.default_rng(81)
    x = rng.uniform(0.1, 2, (40, 288)).astype(np.float32)
    pose = np.repeat(np.eye(4)[None], 40, axis=0)
    pose[:, 0, 3] = np.arange(40)
    graph = run.BASE.make_graph(x, pose, np.zeros(40, np.int64))
    a, b = control(graph), aligned(graph)
    assert a.shape == b.shape == (40, 416)
    assert a.dtype == b.dtype == torch.float32
    torch.testing.assert_close(a[:, 288:], b[:, 288:], atol=0, rtol=0)
    torch.testing.assert_close(
        b[:, :288], F.normalize(graph.x * aligned.gnn.key_scale, dim=1)
    )
    assert not torch.allclose(a[:, :288], b[:, :288])
    run.InfoNCELoss(0.1)(b[:10], b[10:20], b[20:30]).backward()
    assert aligned.gnn.key_scale.grad is None
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in aligned.parameters()
    )
    assert aligned.gnn.output_proj.weight.grad.abs().sum() > 0
    short = run.BASE.make_graph(x[:31], pose[:31], np.zeros(31, np.int64))
    torch.testing.assert_close(aligned(short)[:16], b[:16], atol=4e-6, rtol=4e-5)
