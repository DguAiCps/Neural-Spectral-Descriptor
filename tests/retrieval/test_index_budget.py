"""색인 대조에서 동점 처리·실제 후보 수·공통 후보 규칙을 검증한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _pdv2_contrast_index_budget as study


def test_equal_indexes_match_actual_union_size():
    scores = np.arange(350, dtype=np.float32)
    groups = study.candidates(scores, scores)
    assert len(groups["key_200"]) == len(groups["aux_200"]) == 200
    assert len(groups["dual_100_100"]) == 100
    np.testing.assert_array_equal(groups["key_matched"], groups["dual_100_100"])
    np.testing.assert_array_equal(groups["aux_matched"], groups["dual_100_100"])


def test_small_history_and_distinct_indexes():
    for size in (1, 73, 100, 199, 420):
        x = np.arange(size, dtype=np.float32)
        groups = study.candidates(x, -x)
        assert len(groups["key_200"]) == min(size, 200)
        assert len(groups["key_matched"]) == len(groups["aux_matched"]) == len(groups["dual_100_100"])
        for ids in groups.values():
            assert ids.dtype == np.int64
            assert (ids >= 0).all() and (ids < size).all()
            assert (np.diff(ids) > 0).all()


def test_ties_reproduce_original_prefix_topk():
    rng = np.random.default_rng(1729)
    for repeated in (False, True):
        bank = rng.integers(-2, 3, size=(351, 8)).astype(np.float32)
        if repeated:
            bank[:] = 1
        for count in (1, 100, 133, 200, 400):
            expected = study.run.base.prefix_topk(bank, 350, 320, count)
            actual = study.topk(bank[:320] @ bank[350], count)
            np.testing.assert_array_equal(actual, expected)
