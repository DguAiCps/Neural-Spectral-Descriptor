"""정답 없는 합성 입력으로 간선 개수·선택과 점수 속성의 분리를 검증한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _pdv2_edge_reliability_final as exp


def test_selection_and_attributes_are_separate():
    """같은 C 연결에서 속성만 교체하며 직접 선택은 원시값을 속성에 넣지 않는다."""
    c = {"query": np.repeat([60, 61], 30), "neighbor": np.tile(np.arange(30), 2)}
    scores = {"A": np.linspace(0, 1, 60, dtype=np.float32), "B": np.linspace(1, 0, 60, dtype=np.float32)}
    for op in exp.OPS:
        scores["C_" + op] = scores["B"]
        scores["extra_" + op] = np.stack([scores["B"] * 20, np.ones(60)], axis=1).astype(np.float32)
        a, ca = exp.choose(c, scores, op, "C_selection")
        b, cb = exp.choose(c, scores, op, "C")
        np.testing.assert_array_equal(ca, cb)
        assert a is scores["A"] and b is scores["C_" + op]
        direct, ids = exp.choose(c, scores, op, "alignment")
        assert direct is scores["A"]
        np.testing.assert_array_equal(ids, ca)
        _, tied = exp.choose(c, scores, op, "column")
        np.testing.assert_array_equal(tied, np.r_[np.arange(10), np.arange(30, 40)])
        for arm in exp.ARMS:
            p, selected = exp.choose(c, scores, op, arm)
            assert p.dtype == np.float32 and np.isfinite(p).all()
            np.testing.assert_array_equal(np.bincount(c["query"][selected], minlength=62)[60:], [10, 10])


def test_early_queries_and_prefix_keep_counts():
    """초기 가변 후보와 입력 끝 노드에서도 최대10개 규칙을 지킨다."""
    c = {"query": np.r_[30, 31, 31, np.repeat(80, 30)], "neighbor": np.r_[0, 0, 1, np.arange(30)]}
    p = np.linspace(0, 1, 33, dtype=np.float32)
    scores = {"A": p, "B": p, **{"C_" + op: p for op in exp.OPS},
              **{"extra_" + op: np.stack([p, p], axis=1) for op in exp.OPS}}
    for arm in exp.ARMS:
        _, ids = exp.choose(c, scores, exp.OPS[0], arm)
        assert len(ids) == 13 and ids.dtype == np.int64
        short = {k: v[:3] for k, v in c.items()}
        _, si = exp.choose(short, {k: v[:3] for k, v in scores.items()}, exp.OPS[0], arm)
        np.testing.assert_array_equal(ids[c["query"][ids] <= 31], si)
    assert len(exp.choose(c, scores, exp.OPS[0], "temporal")[1]) == 0
