"""최종 수락 실험의 정답 정의·동점·고정 점수 계약을 검사한다."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from _pdv2_context_accept_lib import auc, fit_score, labels, predict_score


def test_labels_keep_wrong_positive_and_drop_ambiguous():
    ledger = {"min_past_gt_dist": np.array([1., 1., 15., 30.]),
              "top1_geo_dist": np.array([1., 10., 15., 30.])}
    eligible, y = labels(ledger)
    np.testing.assert_array_equal(eligible, [True, True, False, True])
    np.testing.assert_array_equal(y, [1., 0., 0., 0.])


def test_auc_ties_and_constant():
    assert auc(np.array([0., 1., 1., 2.]), np.array([0, 0, 1, 1])) == .875
    assert auc(np.zeros(4), np.array([0, 0, 1, 1])) == .5
    assert auc(np.zeros(4), np.ones(4)) is None


def test_calibration_monotone_and_external_fixed():
    x = np.linspace(-2, 2, 100)[:, None]
    y = (x[:, 0] > .4).astype(float)
    fitted = fit_score(x, y, np.ones(len(x)))
    score = predict_score(x, fitted)
    assert np.all(np.diff(score) > 0) and np.isfinite(score).all()
    # 검증 표본의 구성이나 순서로 정규화·계수가 달라지면 안 된다.
    np.testing.assert_allclose(predict_score(x[[80, 20]], fitted), score[[80, 20]])
    assert fitted["coefficients"][0] > 0
