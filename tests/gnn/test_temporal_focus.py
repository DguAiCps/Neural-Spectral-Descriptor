"""방향 쌍 교체와 최종 AP 회계의 결정적 합성 검사다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_pair_replacement_preserves_budget_and_unselected_rows():
    from _pdv2_temporal_focus import opposite_replace
    original = np.tile(np.array([[0, 1, 5], [2, 3, 5]], dtype=np.int64), (8, 1))
    pools = [np.array([2]), np.array([], dtype=np.int64), np.array([0])]
    negatives = np.array([5, 5, 5])
    a, rows = opposite_replace(original, pools, negatives, 42, .25)
    b, rows2 = opposite_replace(original, pools, negatives, 42, .25)
    assert a.shape == original.shape and a.dtype == original.dtype
    assert len(rows) == 4
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(rows, rows2)
    keep = np.ones(len(a), bool)
    keep[rows] = False
    np.testing.assert_array_equal(a[keep], original[keep])
    assert all(p in pools[q] and n == negatives[q] for q, p, n in a[rows])


def test_ap_accounting_and_ambiguous_queries():
    from _pdv2_temporal_focus_diagnose import ap_parts, CATEGORIES
    a = dict(query_idx=np.arange(6), min_past_gt_dist=np.array([0., 0., 0., 0., 40., 15.]),
             top1_geo_dist=np.array([30., 0., 0., 30., 40., 15.]), score=np.array([.3, .8, .9, .2, .7, .99]))
    b = {**a, "top1_geo_dist": np.array([0., 30., 0., 30., 40., 15.]), "score": np.array([.8, .7, .9, .1, .1, .01])}
    r = ap_parts(a, b, "score")
    assert abs(sum(r[k] for k in CATEGORIES) - (r["after"] - r["before"])) < 1e-12
    z = ap_parts(a, a, "score")
    assert all(z[k] == 0 for k in CATEGORIES)
    assert np.isfinite(list(r.values())).all()
