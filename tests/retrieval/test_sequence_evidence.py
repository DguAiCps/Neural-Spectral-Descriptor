"""시퀀스 대응의 관측 범위·기준 재현·중복 가중치를 검사한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _xs_eval_lib import variant_inputs
from _pdv2_arc_evidence import scores as original_scores
from retrieval.sequence_evidence import (
    distance_indices, neighbor_plans, path_means,
    repeated_candidate_weights, sequence_scores)


def test_distance_plans_handle_speed_change_and_available_prefix():
    path = np.r_[np.arange(50)*.5, 25+np.arange(50)*2.]
    np.testing.assert_array_equal(distance_indices(path, np.array([20]), np.array([3.])), [26])
    q, j = np.array([30, 60, 80]), np.array([0, 20, 42])
    for name, (qn, jn, valid) in neighbor_plans(path, q, j).items():
        assert ((qn <= q[None, :, None]+5) | ~valid).all(), name
        assert ((jn <= q[None, :, None]-20) | ~valid).all(), name
        assert ((np.abs(jn-j[None, :, None]) <= 10) | ~valid).all(), name


def test_repeated_candidate_has_one_total_vote_and_empty_neighbors_are_finite():
    ids = np.array([[[2, 2, 4, 9]]])
    valid = np.array([[[True, True, True, False]]])
    weights = repeated_candidate_weights(ids, valid)
    np.testing.assert_array_equal(weights, [[[.5, .5, 1., 0.]]])
    peak = np.array([[[.2, .2, .8, 1.]]])
    actual = path_means(np.array([.5]), peak, valid, weights)
    np.testing.assert_allclose(actual, [[.5]])
    np.testing.assert_array_equal(path_means(np.array([.5]), peak, np.zeros_like(valid)), [[.5]])


def test_sparse_baseline_and_real_prefix_reproduce_both_channels():
    rng = np.random.default_rng(17)
    sk = (rng.normal(size=(100,16,8))+1j*rng.normal(size=(100,16,8))).astype(np.complex64)
    bank, path = variant_inputs(sk, 8), np.cumsum(rng.uniform(.1, 2., 100))
    q, j = np.array([30, 60, 80]), np.array([0, 20, 42])
    for channel in ("c_spec", "c_col"):
        all_scores, cost = sequence_scores(bank, path, q, j, channel)
        old, _ = original_scores(bank, path, q, j, channel, "arc_match10")
        np.testing.assert_array_equal(all_scores["arc_sparse_b25"], .75*all_scores["raw"]+.25*old)
        assert len(all_scores) == 21
        assert all(v.shape == q.shape and np.isfinite(v).all() for v in all_scores.values())
        assert (cost["arc_sparse"]["pair_count"] <= 5).all()
        for b in range(len(q)):
            end = q[b]+16
            small, _ = sequence_scores(variant_inputs(sk[:end],8), path[:end], q[b:b+1], j[b:b+1], channel)
            for key in all_scores:
                np.testing.assert_array_equal(small[key], all_scores[key][b:b+1])
