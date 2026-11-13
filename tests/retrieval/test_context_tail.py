"""높은 오답 점수 손실의 미분과 장소 제외를 합성 자료로 검사한다."""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import approx_fprime

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from _pdv2_context_tail_lib import fit_tail, objective, training_arrays


def sample(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0., 1., (120, 2))
    correct = x[:, 0] + .3 * x[:, 1] > .7
    return {"baseline": x[:, 0], "context_arc": x[:, 1], "query_idx": np.arange(120),
            "min_past_gt_dist": np.where(correct, 1., 30.),
            "top1_geo_dist": np.where(correct, 1., 35.)}


def test_tail_gradient_and_output_contract():
    data = {"a/1": sample(1), "b/1": sample(2)}
    arrays = training_arrays(data, (), .01)
    theta = np.array([.7, .3, -.2])
    numeric = approx_fprime(theta, lambda x: objective(x, arrays, 4.)[0], 1e-6)
    np.testing.assert_allclose(objective(theta, arrays, 4.)[1], numeric, atol=2e-6)
    model = fit_tail(data, (), .01, 4.)
    assert model["coefficients"][0] > 0 and model["coefficients"][1] >= 0
    assert np.isfinite(model["loss"]) and model["projected_gradient_max"] < 2e-5


def test_heldout_values_never_enter_fit_or_mining():
    data = {"a/1": sample(1), "b/1": sample(2), "c/1": sample(3)}
    first = fit_tail(data, ("c/1",), 0., 1.)
    data["c/1"] = {key: np.full_like(value, np.nan, dtype=float)
                   for key, value in data["c/1"].items()}
    second = fit_tail(data, ("c/1",), 0., 1.)
    assert first == second
    assert all(p["sequence"] != "c/1" for p in first["mining"])


def test_no_positive_sequence_remains_in_bce():
    data = {"a/1": sample(1), "b/1": sample(2)}
    data["b/1"]["min_past_gt_dist"][:] = 30.
    data["b/1"]["top1_geo_dist"][:] = 35.
    arrays = training_arrays(data, (), .01)
    assert len(arrays["y"]) == 240 and len(arrays["pairs"]) == 1
    assert arrays["mining"][1]["positive_count"] == 0
    assert len(arrays["mining"][1]["hard_query_idx"]) == 2
