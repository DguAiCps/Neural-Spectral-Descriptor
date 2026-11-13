"""과거 평균 보정의 직접 계산·관측 범위·상수 문맥을 검사한다."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from _pdv2_context_relative_lib import relative_features, unit64, weighted_pairs


def example() -> tuple:
    rng = np.random.default_rng(913)
    key = rng.normal(size=(180, 416)).astype(np.float32)
    path = np.cumsum(rng.uniform(.05, 4., 180))
    row = {"query_idx": np.array([30, 61, 111]), "j_star": np.array([0, 10, 37]),
           "direction": np.array([0, 1, 0])}
    return key, path, row


def test_relative_features_equal_explicit_past_mean_and_pair_loop():
    key, path, row = example()
    output = relative_features(key, path, row, chunk_size=2)
    qa, ja, mass = weighted_pairs(path, row["query_idx"], row["j_star"], row["direction"])
    for name, sl in (("d", slice(0, 288)), ("context", slice(288, 416))):
        vectors = unit64(key[:, sl])
        for i, q in enumerate(row["query_idx"]):
            mean = vectors[:q - 29].mean(0)
            expected = {k: 0. for k in ("excess", "centered_dot", "centered_cos")}
            for a, b, w in zip(qa[i], ja[i], mass[i]):
                left, right = vectors[a], vectors[b]
                expected["excess"] += w * (left @ right - np.mean(vectors[:q - 29] @ left))
                u, v = left - mean, right - mean
                expected["centered_dot"] += w * (u @ v)
                expected["centered_cos"] += w * (u @ v) / (max(np.linalg.norm(u), 1e-6) * max(np.linalg.norm(v), 1e-6))
            for kind, value in expected.items():
                np.testing.assert_allclose(output[name + "_" + kind][i], value, atol=2e-12)


def test_observation_prefix_and_batch_order_do_not_change_scores():
    key, path, row = example()
    expected = relative_features(key, path, row)
    for i, q in enumerate(row["query_idx"]):
        one = {name: value[i:i + 1] for name, value in row.items()}
        actual = relative_features(key[:q + 16], path[:q + 16], one)
        poisoned = key.copy()
        poisoned[q + 16:] = 1000.
        poisoned_path = path.copy()
        poisoned_path[q + 16:] += 1e6
        changed = relative_features(poisoned, poisoned_path, one)
        for field in expected:
            assert actual[field].dtype == np.float64
            np.testing.assert_allclose(actual[field], expected[field][i:i + 1], atol=1e-12)
            np.testing.assert_allclose(changed[field], actual[field], atol=1e-12)


def test_constant_context_has_zero_evidence_and_stays_finite():
    key, path, row = example()
    key[:, 288:] = 1.
    output = relative_features(key, path, row)
    for kind in ("excess", "centered_dot", "centered_cos"):
        assert np.isfinite(output["context_" + kind]).all()
        np.testing.assert_allclose(output["context_" + kind], 0., atol=1e-12)
    np.testing.assert_allclose(output["context_mean_norm"], 1., atol=1e-12)


def test_excess_respects_same_query_candidate_ranking():
    key, path, row = example()
    vectors = unit64(key[:, 288:])
    q = 111
    candidates = np.arange(q - 29)
    scores = vectors[candidates] @ vectors[q]
    excess = scores - vectors[:q - 29].mean(0) @ vectors[q]
    np.testing.assert_array_equal(np.argsort(scores), np.argsort(excess))
