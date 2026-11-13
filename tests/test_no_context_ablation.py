"""문맥 제거가 나머지 매칭 규칙과 과거 전용 제약을 보존하는지 확인한다."""
import numpy as np

from _causal_pr_dump import dump_op
from _pdv2_no_context_ablation import evaluate_keys, no_context_key
from _xs_eval_lib import normalize, variant_inputs


def fixture_arrays():
    rng = np.random.default_rng(41)
    raw = rng.uniform(.1, 3, size=(76, 288)).astype(np.float32)
    weights = rng.uniform(.1, 2, size=288).astype(np.float32)
    spec = (rng.normal(size=(76, 16, 8))
            + 1j * rng.normal(size=(76, 16, 8))).astype(np.complex64)
    pos = rng.normal(size=(76, 3)) * 10
    minimum = np.full(76, np.nan)
    for q in range(30, 76):
        minimum[q] = np.linalg.norm(pos[:q - 29] - pos[q], axis=1).min()
    return raw, weights, spec, pos, minimum


def test_only_key_block_remains():
    raw, weights, _, _, _ = fixture_arrays()
    key = no_context_key(raw, weights)
    assert key.shape == (76, 416) and key.dtype == np.float32
    assert np.isfinite(key).all() and not key[:, 288:].any()
    np.testing.assert_array_equal(key[:, :288], normalize(raw * weights))
    np.testing.assert_allclose(np.linalg.norm(key, axis=1), 1, atol=2e-7)


def test_matches_canonical_full_pipeline_for_each_weight():
    raw, weights, spec, pos, minimum = fixture_arrays()
    key = no_context_key(raw, weights)
    bank = variant_inputs(spec, 8)
    settings = {1: (1., 1., 2., 2.), 2: (.5, 1., 2., 2.), 3: (.5, 1., 2., 1.)}
    result = evaluate_keys(key, bank, pos, minimum, settings)
    for seed, w in settings.items():
        expected = dump_op(key, *bank, pos, minimum, w, 100)
        for field, arr in expected.items():
            np.testing.assert_allclose(result[seed][field], arr, atol=1e-12, rtol=0)
        assert np.all(result[seed]["j_star"] <= result[seed]["query_idx"] - 30)


def test_future_change_does_not_change_prefix_decisions():
    raw, weights, spec, pos, minimum = fixture_arrays()
    setting = {1: (1., 1., 2., 2.)}
    before = evaluate_keys(no_context_key(raw, weights), variant_inputs(spec, 8),
                           pos, minimum, setting)[1]
    raw[60:] = raw[60:, ::-1]
    spec[60:] *= 1 + 3j
    after = evaluate_keys(no_context_key(raw, weights), variant_inputs(spec, 8),
                          pos, minimum, setting)[1]
    for field in before:
        np.testing.assert_array_equal(before[field][:30], after[field][:30])
