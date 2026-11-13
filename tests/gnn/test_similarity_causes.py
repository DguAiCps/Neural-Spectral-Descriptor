"""유사 간선 원인진단의 개입 범위와 개수 통제를 합성 입력으로 검사한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _pdv2_similarity_causes as exp


def sample():
    """정답 수가 서로 다른 세 질의와 크기가 다른 원래 키를 만든다."""
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(70, 288)).astype(np.float32) * 3
    c = {"query": np.repeat(np.arange(60, 63, dtype=np.int64), 30),
         "neighbor": np.tile(np.arange(30, dtype=np.int64), 3),
         "features": rng.normal(size=(90, 14)).astype(np.float32),
         "l2": rng.random(90).astype(np.float32),
         "prob_xyz": rng.random(90).astype(np.float32),
         "selected_xyz": np.r_[np.arange(10), np.arange(30, 40), np.arange(60, 70)].astype(np.int64),
         "distance_3d": np.full(90, 100.)}
    c["distance_3d"][[0, 3, 5, 31]] = 5.
    return c, raw


def test_quality_and_count_controls():
    """정답/비정답 분할과 노드별 임의 개수 일치를 확인한다."""
    c, raw = sample()
    ids = {arm: exp.intervention(c, raw, arm)[2] for arm in ("xyz", "temporal", "gt_true_only", "gt_nontrue_only", "gt_random_matched")}
    np.testing.assert_array_equal(np.sort(np.r_[ids["gt_true_only"], ids["gt_nontrue_only"]]), np.sort(ids["xyz"]))
    assert len(ids["temporal"]) == 0 and len(ids["gt_true_only"]) == 4
    np.testing.assert_array_equal(np.bincount(c["query"][ids["gt_true_only"]], minlength=70), np.bincount(c["query"][ids["gt_random_matched"]], minlength=70))
    np.testing.assert_array_equal(ids["gt_random_matched"], exp.intervention(c, raw, "gt_random_matched")[2])
    assert np.isin(ids["gt_random_matched"], ids["xyz"]).all()


def test_attributes_and_selection_separate():
    """속성 변경은 연결을 유지하고 선택 변경은 분류기 점수를 유지한다."""
    c, raw = sample()
    cand, prob, ids = exp.intervention(c, raw, "probability_one")
    assert cand is c and np.all(prob == 1)
    np.testing.assert_array_equal(ids, c["selected_xyz"])
    cand, prob, ids = exp.intervention(c, raw, "raw_key_attributes")
    q, j = c["query"][ids], c["neighbor"][ids]
    np.testing.assert_allclose(cand["l2"][ids], np.linalg.norm(raw[q] - raw[j], axis=1))
    np.testing.assert_array_equal(cand["features"][:, 1:], c["features"][:, 1:])
    np.testing.assert_array_equal(prob, c["prob_xyz"])
    assert np.isfinite(cand["features"]).all() and cand["features"].dtype == np.float32
    assert (cand["l2"][ids] > 20).all()  # 정규화된 키의 L2로 잘못 대체하면 실패한다.
    cand, prob, ids = exp.intervention(c, raw, "cosine_top10")
    assert cand is c and prob is c["prob_xyz"]
    for query in (60, 61, 62):
        candidates = np.flatnonzero(c["query"] == query)
        selected = ids[c["query"][ids] == query]
        assert len(selected) == 10
        assert c["features"][selected, 0].min() >= np.sort(c["features"][candidates, 0])[-10]
