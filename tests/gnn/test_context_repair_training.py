"""학습용 간선 제거의 시간 보존·질의 단위·재현성을 검사한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _pdv2_context_repair_train import node_dropout_mask


def test_node_dropout_is_grouped_and_preserves_time():
    """노드가 선택되면 유사 이웃 전부를 함께 제거한다."""
    kind = np.array([0, 1, 1, 0, 1, 1, 0, 1])
    target = np.array([1, 1, 1, 2, 2, 2, 3, 3])
    for p in (0., .5, 1.):
        mask = node_dropout_mask(kind, target, 4, p, 7)
        np.testing.assert_array_equal(mask, node_dropout_mask(kind, target, 4, p, 7))
        assert mask.dtype == bool and mask.shape == kind.shape
        assert mask[kind == 0].all()
        assert mask[1] == mask[2] and mask[4] == mask[5]
        if p == 0:
            assert mask.all()
        if p == 1:
            np.testing.assert_array_equal(mask, kind == 0)
