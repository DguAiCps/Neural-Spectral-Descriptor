"""연결 거절의 동점·질의 독립성과 상위10 근거 범위를 확인한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _pdv2_context_abstention import POLICIES, choose


def test_gate_is_query_local_and_preserves_ties():
    """뒤 질의를 붙여도 앞 질의의 선택은 같고 동점은 원래 순서다."""
    q = np.repeat([30, 31, 32], [2, 2, 2])
    prob = np.array([.9, .9, .2, .1, .8, .7], np.float32)
    align = np.array([.2, .3, .9, .8, .8, .9], np.float32)
    threshold = {"classifier": .5, "alignment": .6}
    np.testing.assert_array_equal(choose(q, prob, align, 33, "classifier_gate", threshold), [0, 1, 4, 5])
    np.testing.assert_array_equal(choose(q, prob, align, 33, "alignment_gate", threshold), [2, 3, 4, 5])
    for policy in POLICIES:
        full = choose(q, prob, align, 33, policy, threshold)
        short = choose(q[:4], prob[:4], align[:4], 32, policy, threshold)
        np.testing.assert_array_equal(short, full[full < 4])
    assert choose(q, prob, align, 33, "top1", threshold)[0] == 0


def test_alignment_evidence_is_from_selected_ten():
    """선택 밖 높은 점수는 연결을 허용하지 않으며 1개 조건도 같은 근거다."""
    q = np.repeat([30, 31], 11)
    prob = np.tile(np.arange(11, 0, -1) / 12, 2)
    align = np.zeros(22)
    align[10], align[12] = 1., 1.
    threshold = {"classifier": 0., "alignment": .5}
    ids = choose(q, prob, align, 32, "alignment_gate", threshold)
    np.testing.assert_array_equal(ids, np.arange(11, 21))
    np.testing.assert_array_equal(choose(q, prob, align, 32, "alignment_gate_top1", threshold), [11])
    assert ids.dtype == np.int64 and ids.ndim == 1
