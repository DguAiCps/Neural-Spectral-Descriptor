"""합성 입력으로 인과 후보, 동점, 실패 분해와 분모를 확인한다."""
import numpy as np
import pytest

from _collision_diagnostics_lib import (
    matched_threshold, miss_categories, rank_prefix, transition_counts,
)


def test_prefix_excludes_future_and_recent_frames():
    bank = np.array([[0., 1.], [1., 0.], [1., 0.], [1., 0.]], np.float32)
    positions = np.array([[0., 0., 0.], [100., 0., 0.],
                          [0., 0., 0.], [0., 0., 0.]])
    result = rank_prefix(bank, 2, positions, gap=2)
    assert result["top1_idx"] == 0
    assert result["best_pos_rank"] == 1
    assert np.isclose(result["best_pos_dist"], np.sqrt(2))
    with pytest.raises(ValueError):
        rank_prefix(bank, 1, positions, gap=2)


def test_tied_scores_use_lowest_index_and_no_revisit_is_explicit():
    bank = np.ones((4, 1), dtype=np.float32)
    positions = np.array([[100., 0., 0.], [0., 0., 0.],
                          [0., 0., 0.], [1000., 0., 0.]])
    result = rank_prefix(bank, 2, positions, gap=1)
    assert result["top1_idx"] == 0
    assert result["best_pos_idx"] == 1
    assert result["best_pos_rank"] == 2
    absent = rank_prefix(bank, 3, positions, gap=1)
    assert absent["best_pos_rank"] == -1
    assert np.isnan(absent["best_pos_dist"])


def test_categories_have_explicit_boundary_and_spread_precedence():
    result = miss_categories(np.array([5., 4., 4., 4., 26.]),
                             np.array([5., 100., 25., 24.9, 26.]),
                             np.array([.2, .10001, .1, .1, np.nan]), .1)
    np.testing.assert_array_equal(result, [0, 1, 2, 3, 4])
    assert result.dtype == np.int8


def test_matched_threshold_uses_weights_and_keeps_ties():
    values = np.array([.1, .2, .2, .4])
    weights = np.array([.1, .1, .3, .5])
    threshold, achieved = matched_threshold(values, weights, .4)
    assert threshold == .2
    assert achieved == .5
    threshold, achieved = matched_threshold(values, weights, 0.)
    assert threshold < .1 and achieved == 0.


def test_recovery_also_counts_regression_and_ignores_nonrevisits():
    result = transition_counts([False, True, True, False, False],
                               [True, False, True, False, True],
                               [True, True, True, True, False])
    assert result == {"n_positive": 4, "recovered": 1, "regressed": 1,
                      "both_correct": 1, "both_wrong": 1}
