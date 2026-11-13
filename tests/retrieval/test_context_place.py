"""점수 적합의 장소 분리와 보수적 선택 계약을 검사한다."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from _pdv2_context_place_lib import (
    PLACES, choose, false_rate_threshold, fit_fold, robust_pass, score,
)


def sample(offset: float = 0.) -> dict:
    values = np.linspace(0, 1, 100)
    return {"baseline": values, "context_arc": values**2+offset,
            "min_past_gt_dist": np.where(values > .6, 1., 30.),
            "top1_geo_dist": np.where(values > .6, 1., 30.)}


def test_place_groups_and_heldout_values_never_enter_fit():
    assert set(PLACES["helipr_bridge"]) == {f"helipr/Bridge0{i}" for i in (1,2,3,4)}
    assert set(PLACES["kitti_urban"]) == {"kitti/05", "kitti/06"}
    data = {"kitti/02": sample(), "helipr/Bridge01": sample(.4)}
    a = fit_fold(data, ("helipr/Bridge01",), "context_arc")
    data["helipr/Bridge01"] = {k: np.full_like(v, np.nan) for k,v in sample().items()}
    b = fit_fold(data, ("helipr/Bridge01",), "context_arc")
    assert a == b and a["training"] == ["kitti/02"]
    np.testing.assert_array_equal(np.argsort(score(sample(), a, 0)), np.arange(100))


def test_false_rate_threshold_handles_ties():
    x = np.r_[np.zeros(95), np.ones(5)]
    tau = false_rate_threshold(x, .01)
    assert np.isfinite(tau) and (x >= tau).mean() <= .01


def test_robust_selection_requires_high_precision_and_negative_place():
    bad = [{"n_pos": 100, "AP": .02, "R@100P": -.02}]
    assert not robust_pass(bad, .02)
    bad = [{"n_pos": 0, "incorrect_accept_rate": .1}]
    assert not robust_pass(bad, .02)
    result = choose([{"factor": 0., "AP": .8, "pass": False},
                     {"factor": .5, "AP": .9, "pass": False}])
    assert result["robust_factor"] == 0 and result["mean_ap_factor"] == .5
