"""문맥 개입의 연결·전달 범위와 정답 독립성을 검사한다."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _pdv2_context_repair import choose, message_gates


def test_edge_count_threshold_and_prefix():
    """각 질의의 하한·개수·동점 순서는 나중 질의와 무관하다."""
    q = np.repeat([30, 31, 32], [2, 4, 3])
    p = np.array([.1, .1, .8, .2, .9, .4, .2, .3, .6], np.float32)
    for arm, cap in [("C_top1", 1), ("C_top3", 3), ("C_cut", 10), ("C_message", 10)]:
        ids = choose(q, p, arm, .5)
        assert np.bincount(q[ids], minlength=33).max() <= cap
        short = choose(q[:6], p[:6], arm, .5)
        np.testing.assert_array_equal(short, ids[ids < 6])
        if arm == "C_cut":
            assert (p[ids] >= .5).all() and not (q[ids] == 30).any()
    assert choose(q, p, "C_top1", 0)[0] == 0


def test_message_gate_preserves_temporal_edges():
    """낮은 확률은 유사 메시지만 줄이며 시간 메시지는 보존한다."""
    kind = torch.tensor([0, 1, 1, 1])
    attr = torch.zeros(4, 5)
    attr[:, 4] = torch.tensor([.05, 0., .25, 1.])
    gates = message_gates(kind, attr)
    torch.testing.assert_close(gates, torch.tensor([[1.], [0.], [.25], [1.]]))
    assert gates.shape == (4, 1) and gates.dtype == torch.float32
    assert torch.isfinite(gates).all()
