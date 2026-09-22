"""Unit tests for scripts/_pr_eval_lib.py (torch-free; verification item 4 of
docs/CAUSAL_PR_PREREGISTRATION.md — query 3-way classification, FP verdict,
fixed recall denominator, AP formula).

Run: PYTHONPATH=src:scripts python3 -m pytest -q tests/test_pr_eval_lib.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import _pr_eval_lib as pr  # noqa: E402


# ------------------------------------------------- §1.2 query classification

def test_classification_boundaries():
    # Frozen boundaries: d <= 5 positive; 5 < d <= 25 ambiguous; d > 25 negative.
    d = np.array([0.0, 5.0, 5.0 + 1e-9, 25.0, 25.0 + 1e-9, 100.0, np.inf])
    c = pr.classify_queries(d)
    assert c["positive"].tolist() == [True, True, False, False, False, False, False]
    assert c["ambiguous"].tolist() == [False, False, True, True, False, False, False]
    assert c["negative"].tolist() == [False, False, False, False, True, True, True]
    assert (c["n_pos"], c["n_ambiguous"], c["n_neg"], c["n_total"]) == (2, 2, 3, 7)
    # Masks partition the query set.
    assert np.all(c["positive"] ^ c["ambiguous"] ^ c["negative"])


def test_classification_rejects_nan():
    with pytest.raises(ValueError):
        pr.classify_queries(np.array([1.0, np.nan]))


# --------------------------------------------- §1.3 anti-optimism FP verdict

def test_bad_match_on_positive_query_is_fp():
    # Two positive queries. q1 is accepted but its returned top-1 is 10 m away:
    # it must count FP (never excused by the buffer zone) and is NOT a TP.
    qc = pr.classify_queries(np.array([1.0, 2.0]))
    scores = np.array([0.9, 0.8])
    top1 = np.array([1.0, 10.0])
    op = pr.operating_point(scores, top1, qc, tau=0.8)
    assert (op["TP"], op["FP"], op["FN"], op["TN"]) == (1, 1, 0, 0)
    assert op["precision"] == 0.5   # a lenient (match-buffered) impl says 1.0
    assert op["recall"] == 0.5      # q1 is simultaneously not a TP
    # The sweep row at tau=0.8 agrees.
    sw = pr.pr_sweep(scores, top1, qc)
    k = int(np.flatnonzero(sw["thresholds"] == 0.8)[0])
    assert sw["precision"][k] == 0.5
    assert sw["recall"][k] == 0.5


def test_fixed_recall_denominator():
    # A good@0.9, B accepted-with-bad-match@0.8, C good-but-rejected@0.1.
    # At tau=0.8: TP=1 (A), FP=1 (B), FN=1 (C). A naive implementation that
    # drops B from FN and uses TP/(TP+FN) would report recall 1/2;
    # the frozen spec fixes the denominator at n_pos=3 -> recall 1/3.
    qc = pr.classify_queries(np.array([1.0, 1.0, 1.0]))
    scores = np.array([0.9, 0.8, 0.1])
    top1 = np.array([1.0, 10.0, 1.0])
    sw = pr.pr_sweep(scores, top1, qc)
    k = int(np.flatnonzero(sw["thresholds"] == 0.8)[0])
    assert sw["recall"][k] == pytest.approx(1.0 / 3.0)
    op = pr.operating_point(scores, top1, qc, tau=0.8)
    assert (op["TP"], op["FP"], op["FN"], op["TN"]) == (1, 1, 1, 0)
    assert op["recall"] == pytest.approx(1.0 / 3.0)
    # Full acceptance: recall reaches 2/3, never 1 (B is FP, not TP).
    assert sw["recall"][-1] == pytest.approx(2.0 / 3.0)


# ------------------------------------- §1.4 full sweep on a 6-query toy case
#
# d_min  = [ 1,   50,  2,   3,   60,  10 ]  -> classes [P, N, P, P, N, A]
# scores = [.95, .90, .85, .60, .20, .99]   (ambiguous q5 excluded entirely,
# top1   = [ 1,   48,  2,   3,   55,  .5]    its 0.99 is not a sweep point)
#
# n_pos=3, n_neg=2, n_ambiguous=1. Sweep (tau desc):
#   tau   accepted        TP FP FN TN   P     R
#   inf   {}               0  0  3  2   1     0
#   .95   {q0}             1  0  2  2   1     1/3
#   .90   {q0,q1}          1  1  2  1   1/2   1/3
#   .85   {q0,q1,q2}       2  1  1  1   2/3   2/3
#   .60   {..,q3}          3  1  0  1   3/4   1
#   .20   {..,q4}          3  2  0  0   3/5   1
# P_interp (reverse cummax of P): [1, 1, 3/4, 3/4, 3/4, 3/5]
# AP = (1/3-0)*1 + 0*3/4 + (2/3-1/3)*3/4 + (1-2/3)*3/4 + 0*3/5
#    = 1/3 + 1/4 + 1/4 = 5/6
# F1 = [0, 1/2, 2/5, 2/3, 6/7, 3/4] -> max F1 = 6/7 at tau = 0.60
# R@100P = R@99P = R@95P = 1/3 (only the P=1 points beyond the sentinel)

TOY_DMIN = np.array([1.0, 50.0, 2.0, 3.0, 60.0, 10.0])
TOY_SCORES = np.array([0.95, 0.90, 0.85, 0.60, 0.20, 0.99])
TOY_TOP1 = np.array([1.0, 48.0, 2.0, 3.0, 55.0, 0.5])


def _toy_sweep():
    return pr.pr_sweep(TOY_SCORES, TOY_TOP1, pr.classify_queries(TOY_DMIN))


def test_toy_sweep_curve_and_metrics():
    sw = _toy_sweep()
    assert sw["thresholds"].tolist() == [np.inf, 0.95, 0.90, 0.85, 0.60, 0.20]
    assert sw["precision"].tolist() == pytest.approx(
        [1.0, 1.0, 1 / 2, 2 / 3, 3 / 4, 3 / 5])
    assert sw["recall"].tolist() == pytest.approx(
        [0.0, 1 / 3, 1 / 3, 2 / 3, 1.0, 1.0])
    assert sw["AP"] == pytest.approx(5.0 / 6.0)
    assert sw["max_F1"] == pytest.approx(6.0 / 7.0)
    assert sw["max_F1_tau"] == pytest.approx(0.60)
    assert sw["R@100P"] == pytest.approx(1 / 3)
    assert sw["R@99P"] == pytest.approx(1 / 3)
    assert sw["R@95P"] == pytest.approx(1 / 3)
    assert sw["counts"] == {"n_pos": 3, "n_neg": 2,
                            "n_ambiguous": 1, "n_total": 6}


def test_toy_tn_accounting():
    qc = pr.classify_queries(TOY_DMIN)
    # tau=0.90: negative q1 accepted (FP), negative q4 rejected (TN).
    op = pr.operating_point(TOY_SCORES, TOY_TOP1, qc, tau=0.90)
    assert (op["TP"], op["FP"], op["FN"], op["TN"]) == (1, 1, 2, 1)
    # tau=0.20: both negatives accepted -> no TN left.
    op = pr.operating_point(TOY_SCORES, TOY_TOP1, qc, tau=0.20)
    assert (op["TP"], op["FP"], op["FN"], op["TN"]) == (3, 2, 0, 0)


def test_operating_point_matches_sweep_rows():
    sw = _toy_sweep()
    qc = pr.classify_queries(TOY_DMIN)
    for k, tau in enumerate(sw["thresholds"]):
        if not np.isfinite(tau):
            continue
        op = pr.operating_point(TOY_SCORES, TOY_TOP1, qc, tau=tau)
        assert op["precision"] == pytest.approx(sw["precision"][k]), tau
        assert op["recall"] == pytest.approx(sw["recall"][k]), tau


def test_ambiguous_query_fully_excluded():
    # q5 (ambiguous) has the HIGHEST score and a 0.5 m top-1: were it leaked
    # into the eligible set it would add a sweep point at 0.99 and a TP.
    sw = _toy_sweep()
    assert 0.99 not in sw["thresholds"].tolist()
    assert sw["recall"][-1] == pytest.approx(1.0)  # 3 TP / 3 positives only


# ------------------------------------------------------- R@100P collapse

def test_r_at_100p_collapse():
    # One FP at the very top of the ranking: precision < 1 at every non-empty
    # acceptance -> R@100P collapses to 0, while at full acceptance
    # P = 19/20 = 0.95 and R = 1 -> R@95P = 1.
    n_pos = 19
    d_min = np.r_[60.0, np.full(n_pos, 1.0)]
    scores = np.r_[0.99, np.linspace(0.9, 0.5, n_pos)]
    top1 = np.r_[55.0, np.full(n_pos, 1.0)]
    sw = pr.pr_sweep(scores, top1, pr.classify_queries(d_min))
    assert sw["R@100P"] == 0.0
    assert sw["R@99P"] == 0.0
    assert sw["R@95P"] == pytest.approx(1.0)


# ------------------------------------------------------------ NaN handling

def test_nan_score_never_accepted():
    # q1: empty candidate set (NaN score, NaN top-1). It is a positive query
    # -> permanent FN, still in the recall denominator at every tau.
    qc = pr.classify_queries(np.array([1.0, 1.0]))
    scores = np.array([0.9, np.nan])
    top1 = np.array([1.0, np.nan])
    sw = pr.pr_sweep(scores, top1, qc)
    assert sw["thresholds"].tolist() == [np.inf, 0.9]  # NaN is not a sweep point
    assert np.all(sw["recall"] <= 0.5)                 # recall can never reach 1
    assert sw["AP"] == pytest.approx(0.5)              # (R=1/2, P=1) only
    op = pr.operating_point(scores, top1, qc, tau=-1e9)
    assert (op["TP"], op["FP"], op["FN"], op["TN"]) == (1, 0, 1, 0)


# ------------------------------------------------- consistency / convention

def test_negative_query_close_top1_raises():
    # A negative query (d_min > 25 m) cannot have a returned top-1 within 5 m.
    qc = pr.classify_queries(np.array([50.0]))
    with pytest.raises(ValueError):
        pr.pr_sweep(np.array([0.5]), np.array([2.0]), qc)
    with pytest.raises(ValueError):
        pr.operating_point(np.array([0.5]), np.array([2.0]), qc, tau=0.0)


def test_lower_is_better_rejected():
    qc = pr.classify_queries(np.array([1.0]))
    with pytest.raises(ValueError):
        pr.pr_sweep(np.array([0.5]), np.array([1.0]), qc,
                    higher_is_better=False)


# ------------------------------------------------------- §1.6 aggregation

def test_aggregate_groups_macro_of_macros():
    def res(ap):
        return {"AP": ap, "max_F1": ap, "R@100P": ap, "R@99P": ap, "R@95P": ap}
    per_seq = {"a1": res(0.8), "a2": res(0.6), "b1": res(0.4)}
    agg = pr.aggregate_groups(per_seq, {"A": ["a1", "a2"], "B": ["b1"]})
    assert agg["groups"]["A"]["AP"] == pytest.approx(0.7)
    assert agg["groups"]["B"]["AP"] == pytest.approx(0.4)
    # Macro of group means = (0.7+0.4)/2 = 0.55, NOT the pooled 3-seq mean 0.6.
    assert agg["overall"]["AP"] == pytest.approx(0.55)


# ------------------------------------------------------------------ output

def test_result_payload_roundtrip(tmp_path, monkeypatch):
    sw = _toy_sweep()
    op = pr.operating_point(TOY_SCORES, TOY_TOP1,
                            pr.classify_queries(TOY_DMIN), tau=0.60)
    payload = pr.result_payload("nsd", "kitti_00", sw, seed=1, op=op)
    assert payload["schema"] == "pr-eval-v1"
    assert payload["curve"]["thresholds"][0] is None  # +inf sentinel -> null
    json.dumps(payload, allow_nan=False)              # strict-JSON clean
    monkeypatch.setattr(pr, "OUT_DIR", tmp_path)
    p = pr.write_result(payload, "toy")
    back = json.loads(p.read_text())
    assert back["metrics"]["AP"] == pytest.approx(sw["AP"])
    assert back["op"]["F1"] == pytest.approx(6.0 / 7.0)
    assert back["counts"]["n_ambiguous"] == 1
