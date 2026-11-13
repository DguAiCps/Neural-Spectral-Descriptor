"""불균형·동점·후보 부재의 지표 분모를 합성 입력으로 검증한다."""
import numpy as np
from retrieval.edge_reliability import (
    candidate_matrices, query_metrics, summarize_queries, retention_threshold, confidence_bins,
)


def test_within_query_and_no_candidate_truth():
    q = np.repeat(np.arange(3), 3)
    d = np.array([1, 30, 40, 2, 35, 15, 30, 40, 50.])
    s = np.array([.9, .3, .1, .5, .5, .8, .9, .8, .7])
    dm, _ = candidate_matrices(q, d, 3)
    sm, _ = candidate_matrices(q, s, 3)
    r = query_metrics(sm, dm, np.ones(3, bool))
    np.testing.assert_allclose(r["within_query_AUC"][:2], [1, .5])
    assert np.isnan(r["within_query_AUC"][2])
    np.testing.assert_array_equal(r["top1_correct"], [True, False, False])
    a = summarize_queries(r, np.ones(3, bool))
    assert a["queries"] == 3 and a["mixed_queries"] == 2
    assert a["hit10_given_available"] == 1 and a["hit10_all_queries"] == 2 / 3
    assert a["within_query_AUC"] == .75


def test_padding_ties_threshold_and_confidence():
    q = np.array([0, 1, 1, 1])
    d, count = candidate_matrices(q, np.array([35., 1, 30, 40]), 3)
    s, _ = candidate_matrices(q, np.array([.2, .6, .6, .6]), 3)
    r = query_metrics(s, d, np.array([True, True, False]))
    assert r["within_query_AUC"][1] == .5
    np.testing.assert_array_equal(count, [1, 3, 0])
    t = retention_threshold(np.arange(100.), .95)
    assert t == 5 and (np.arange(100.) >= t).mean() == .95
    bins = confidence_bins(np.array([0., .95, 1.]), np.array([2., 10., 30.]))
    assert bins[0]["correct"] == 1 and bins[9]["ambiguous"] == 1 and bins[9]["far"] == 1
