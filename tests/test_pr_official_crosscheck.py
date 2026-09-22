"""Difference-detection cross-check: our frozen PR accounting vs the official
RING PR implementation (pre-registered verification item §4.3 of
docs/CAUSAL_PR_PREREGISTRATION.md).

Accounting-difference summary (cited in the paper appendix)
-----------------------------------------------------------
The official RING PR accounting (``compute_PR_pairs`` in
``_external/RING/evaluation/plot_PR_curve.py``) and our frozen PR library
(``scripts/_pr_eval_lib.py``) produce identical confusion cells on a shared
candidate set — both count ANY accepted match whose returned top-1 lies beyond
the 5 m revisit radius as FP (also on positive queries), and both count a
rejected positive query as FN — so precision agrees at every threshold. The
sole substantive difference is the recall denominator: the official code
computes recall = TP/(TP+FN), and because a positive query that accepts a
wrong match sits in the FP cell (not FN), that denominator SHRINKS exactly
when wrong matches are accepted on positive queries, inflating recall (in the
extreme, recall reads 1.0 while positive queries are being answered with wrong
matches). Our library fixes the denominator at the total positive-query count
(recall = TP/n_pos), which is never larger. The two accountings coincide
wherever no positive query accepts a wrong match — in particular on the entire
100%-precision segment (P=1 iff FP=0 iff no wrong acceptances), so R@100P is
invariant to the difference, while recall at P<1 (hence AP and max F1) is
inflated by the official accounting. Two further cosmetic differences: the
official adds 1e-10 to both denominators, and at zero acceptances defines
precision as 0 (0/1e-10; the RINGSharp copy of the same loop instead pins it
to 1 via ``precisions[num_hits == 0] = 1.0``) where our sweep's +inf sentinel
defines it as 1; the comparisons below therefore use atol=1e-9 for the
"equal" assertions and skip the zero-acceptance sentinel point.

Run:  PYTHONPATH=src:scripts python3 -m pytest -q --noconftest \
          tests/test_pr_official_crosscheck.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import _pr_eval_lib as prlib  # noqa: E402  (sys.path insertion above)

# --------------------------------------------------------------------------
# Official implementation: provenance + drift guard.
#
# Canonical source (the file the pre-registration §1.3/§4.3 cites):
#   _external/RING/evaluation/plot_PR_curve.py
#     lines 13-15  calculate_dist
#     lines 18-55  compute_PR_pairs (accounting loop 27-49, P/R at 51-54)
# That module imports matplotlib AND sklearn at module top (sklearn is not a
# dependency of this repo's PR evaluation and is absent on the eval host), so
# it cannot be imported here; instead the accounting is replicated VERBATIM
# below and test_official_source_unchanged() hashes the exact source segment
# (lines 13-56, "def calculate_dist" up to the "# plot PR curve" marker) so
# any upstream drift fails this file loudly.
# Replication deltas (all accounting-neutral, listed exhaustively):
#   * sklearn.neighbors.KDTree -> _KDTree (brute-force query_radius with the
#     same closed-ball semantics; no synthetic query sits near the 5 m
#     boundary, so open/closed cannot matter here);
#   * the matplotlib plotting block (lines 56-72) and the save_path argument
#     are dropped;
#   * the four count arrays are returned alongside precisions/recalls/f1s
#     (extra return values only — no accounting line is altered).
# The RINGSharp copy (_external/RINGSharp/tools/plot_PR_curve.py, seq_num=1)
# is the identical loop except the extra ``precisions[num_hits == 0] = 1.0``
# anchor; the RING file is used as the canonical reference.
# --------------------------------------------------------------------------

OFFICIAL_PATH = REPO / "_external" / "RING" / "evaluation" / "plot_PR_curve.py"
OFFICIAL_SEGMENT_SHA256 = (
    "f955262dd08044bfb57a90b4d988d7c0a8e7d6a4563c8e3ec10e8a3fcd598b75"
)


def test_official_source_unchanged():
    """Drift guard: the replicated loop below mirrors this exact source."""
    if not OFFICIAL_PATH.exists():
        pytest.skip(f"official file not in this checkout: {OFFICIAL_PATH}")
    src = OFFICIAL_PATH.read_text()
    start = src.index("def calculate_dist")
    end = src.index("    # plot PR curve")
    digest = hashlib.sha256(src[start:end].encode()).hexdigest()
    assert digest == OFFICIAL_SEGMENT_SHA256, (
        "official plot_PR_curve.py accounting segment changed — re-verify the "
        "verbatim replication in this test against the new source")


class _KDTree:
    """Stand-in for sklearn.neighbors.KDTree (query_radius, closed ball)."""

    def __init__(self, X):
        self._X = np.asarray(X, dtype=np.float64)

    def query_radius(self, X, r):
        X = np.asarray(X, dtype=np.float64)
        return [np.flatnonzero(np.linalg.norm(self._X - x, axis=1) <= r)
                for x in X]


def calculate_dist(pose1, pose2):
    dist = np.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)
    return dist


def compute_PR_pairs(pair_dists, query_positions, map_positions, thresholds,
                     revisit_threshold: float = 10.0):
    tree = _KDTree(map_positions)
    num_thresholds = len(thresholds)
    num_hits = np.zeros(num_thresholds)
    num_false_alarms = np.zeros(num_thresholds)
    num_correct_rejections = np.zeros(num_thresholds)
    num_misses = np.zeros(num_thresholds)

    for i in range(pair_dists.shape[0]):
        dist = pair_dists[i]
        query_position = query_positions[i]
        indices = tree.query_radius(query_position.reshape(1, -1),
                                    revisit_threshold)[0]
        for j, thre in enumerate(thresholds):
            if(min(dist) < thre):
                # if under the theshold, it is considered seen.
                # and then check the correctness
                real_dist = calculate_dist(query_position,
                                           map_positions[np.argmin(dist)])
                if real_dist < revisit_threshold:
                    # TP: Hit
                    num_hits[j] = num_hits[j] + 1
                else:
                    # FP: False Alarm
                    num_false_alarms[j] = num_false_alarms[j] + 1

            else:
                if len(indices) == 0:
                    # TN: Correct rejection
                    num_correct_rejections[j] = num_correct_rejections[j] + 1
                else:
                    # FN: MISS
                    num_misses[j] = num_misses[j] + 1

    precisions = num_hits / (num_hits + num_false_alarms + 1e-10)
    recalls = num_hits / (num_hits + num_misses + 1e-10)
    precisions[np.isnan(precisions)] = 1.0
    recalls[np.isnan(recalls)] = 0.0
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    return (precisions, recalls, f1s,
            num_hits, num_false_alarms, num_correct_rejections, num_misses)


# --------------------------------------------------------------------------
# Synthetic world: 300 frames on a 1-D line (embedded in 2-D), out-and-back.
#
#   t =   0..149  outbound, x = 2t (0..298 m)      -> the map / past DB
#   t = 150..179  turnaround loiter at x = 298     -> 30-frame exclusion
#                 buffer (never a candidate, never a query)
#   t = 180..269  return pass, x = 296 - 3(t-180)  -> 90 POSITIVE queries
#                 (nearest map frame at 0 or 1 m)
#   t = 270..299  novel territory, x = -26-5(t-270)-> 30 NEGATIVE queries
#                 (nearest map frame >= 26 m)
#
# Candidate set for BOTH accountings = the 150 map frames. Every query index
# t >= 180 and every map index <= 149 <= t-31, so the shared candidate set is
# past-only with the 30-frame recent exclusion (§1.1/§1.2), realised as the
# official code's own query-session / map-session split. No query falls in
# the (5, 25] m ambiguous band (the official accounting has no such class),
# so query classification is identical on both sides by construction.
#
# Score model (hand-made, deterministic): score-distance
#   D[q, m] = 0.01 * |x_q - x_m| + noise,  noise = seeded U{0..999} * 1e-6
# The noise is quantized to the 1e-6 grid so DISTINCT score values are at
# least ~1e-6 apart by construction (positions are integers, so D is exactly
# (10000*geo + k) * 1e-6) — required by the EPS_THRE accept-boundary shim.
# Lower = better for the official loop; our library takes the negated top-1
# (higher = better) and the geometric distance of the returned top-1.
# --------------------------------------------------------------------------

MAP_N = 150
REVISIT_M = 5.0          # official revisit_threshold == our MATCH_RADIUS_M
EPS_THRE = 1e-9          # accept-boundary shim: ours "score >= tau" is
#                          non-strict, official "min(dist) < thre" is strict,
#                          so thre = -tau + EPS_THRE gives identical accept
#                          sets provided unique top-1 scores are > 1e-6 apart
#                          (asserted below).


def _build_world():
    x_map = 2.0 * np.arange(MAP_N)
    x_pos = 296.0 - 3.0 * np.arange(90)      # queries t = 180..269
    x_neg = -26.0 - 5.0 * np.arange(30)      # queries t = 270..299
    x_q = np.r_[x_pos, x_neg]
    geo = np.abs(x_q[:, None] - x_map[None, :])          # (120, 150)
    map_positions = np.c_[x_map, np.zeros(MAP_N)]        # embed in 2-D
    query_positions = np.c_[x_q, np.zeros(len(x_q))]
    rng = np.random.default_rng(20260811)
    noise = rng.integers(0, 1000, size=geo.shape).astype(np.float64)
    D1 = (10000.0 * geo + noise) * 1e-6
    return geo, D1, query_positions, map_positions


def _wrong_rows():
    """10 positive queries forced to accept a wrong match in case 2."""
    return np.arange(0, 90, 9)               # rows 0, 9, ..., 81 (positives)


def _case2_matrix(geo, D1):
    """Raise every within-5m candidate of 10 positive queries so their top-1
    becomes the nearest frame BEYOND 5 m (6 or 7 m away, score-dist ~0.06-
    0.072): accepted at mid thresholds, geometrically wrong."""
    D2 = D1.copy()
    for r in _wrong_rows():
        D2[r, geo[r] <= REVISIT_M] += 1.0
    return D2


def _our_side(D, geo):
    """Per-query top-1 inputs for the library + classification masks."""
    j_star = D.argmin(axis=1)
    rows = np.arange(D.shape[0])
    scores = -D[rows, j_star]                # higher = better (lib convention)
    top1_geo = geo[rows, j_star]
    qc = prlib.classify_queries(geo.min(axis=1))
    return scores, top1_geo, qc


def _run_both(D, geo, query_positions, map_positions):
    """Sweep our library and the official loop over the SAME accept sets."""
    scores, top1_geo, qc = _our_side(D, geo)
    sweep = prlib.pr_sweep(scores, top1_geo, qc)
    taus = sweep["thresholds"][1:]           # drop the +inf sentinel (zero-
    #                                          acceptance anchor: ours P=1 by
    #                                          definition, official P=0 — a
    #                                          documented cosmetic difference)
    gaps = np.diff(np.unique(-scores))
    assert gaps.size == 0 or gaps.min() > 1e-7, "EPS_THRE shim precondition"
    thre = -taus + EPS_THRE                  # identical acceptance sets
    p_off, r_off, _f1, hits, fas, crs, misses = compute_PR_pairs(
        D, query_positions, map_positions, thre, revisit_threshold=REVISIT_M)
    ours = np.array([[op["TP"], op["FP"], op["FN"], op["TN"]]
                     for op in (prlib.operating_point(scores, top1_geo, qc, t)
                                for t in taus)], dtype=np.float64)
    return scores, top1_geo, qc, sweep, taus, p_off, r_off, \
        np.c_[hits, fas, misses, crs], ours


def test_agreement_when_no_positive_wrong_accepts():
    """§4.3 case 1: with no positive query accepting a wrong match, the two
    accountings MUST agree — identical cells, precision and recall equal at
    every (finite) threshold, and the official recall denominator TP+FN stays
    fixed at n_pos throughout."""
    geo, D1, qpos, mpos = _build_world()
    scores, top1_geo, qc, sweep, taus, p_off, r_off, cells_off, cells_ours = \
        _run_both(D1, geo, qpos, mpos)

    # World premises: 90/30/0 query split, and every positive query's top-1
    # is geometrically correct (so no FP can come from a positive query).
    assert (qc["n_pos"], qc["n_neg"], qc["n_ambiguous"]) == (90, 30, 0)
    assert np.all(top1_geo[qc["positive"]] < REVISIT_M)

    # cells: [TP, FP, FN, TN] per threshold, identical accounting.
    np.testing.assert_array_equal(cells_off, cells_ours)
    tp, _fp, fn = cells_ours[:, 0], cells_ours[:, 1], cells_ours[:, 2]
    np.testing.assert_array_equal(tp + fn, np.full(len(taus), qc["n_pos"]))

    # Curves: equal up to the official's +1e-10 denominators.
    np.testing.assert_allclose(sweep["precision"][1:], p_off,
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(sweep["recall"][1:], r_off, rtol=0, atol=1e-9)


def test_divergence_is_exactly_the_recall_denominator():
    """§4.3 case 2: with positive-query wrong-match acceptances, the ONLY
    divergence is the documented one — both sides still produce identical
    TP/FP/FN/TN cells (the wrong-accepted positive is FP, not FN, in BOTH),
    but the official recall divides by the variable TP+FN (which drops those
    queries) while ours divides by the fixed n_pos. The official output is
    reconstructed exactly from our counts as TP/(TP+FN+1e-10)."""
    geo, D1, qpos, mpos = _build_world()
    D2 = _case2_matrix(geo, D1)
    scores, top1_geo, qc, sweep, taus, p_off, r_off, cells_off, cells_ours = \
        _run_both(D2, geo, qpos, mpos)

    # Premise: exactly the 10 planted positives return a wrong (>5 m) top-1.
    wrong = np.zeros(len(scores), dtype=bool)
    wrong[_wrong_rows()] = True
    assert np.array_equal(qc["positive"] & (top1_geo > REVISIT_M), wrong)

    # Identical confusion cells even here (wrong-accept positive -> FP cell
    # on BOTH sides; official adds it to FP but NOT to FN, exactly like us).
    np.testing.assert_array_equal(cells_off, cells_ours)
    tp, fp, fn = cells_ours[:, 0], cells_ours[:, 1], cells_ours[:, 2]

    # Wrong-acceptance count per threshold == the recall-denominator deficit.
    n_wrong_acc = np.array([(wrong & (scores >= t)).sum() for t in taus])
    np.testing.assert_array_equal(qc["n_pos"] - (tp + fn), n_wrong_acc)
    assert n_wrong_acc.max() == 10 and (n_wrong_acc > 0).any()

    # Precision still agrees everywhere.
    np.testing.assert_allclose(sweep["precision"][1:], p_off,
                               rtol=0, atol=1e-9)

    # Official recall reconstructed from OUR counts: TP/(TP+FN_official),
    # FN_official = our FN (the wrong-accepted queries are excluded from the
    # denominator); ours stays TP/n_pos.
    np.testing.assert_allclose(r_off, tp / (tp + fn + 1e-10),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(sweep["recall"][1:], tp / qc["n_pos"],
                               rtol=0, atol=1e-12)

    # Divergence appears exactly where wrong acceptances exist (TP>0 there),
    # always inflating the official recall; elsewhere the curves agree.
    div = n_wrong_acc > 0
    assert np.all(tp[div] > 0)
    assert np.all(r_off[div] > sweep["recall"][1:][div] + 1e-6)
    np.testing.assert_allclose(r_off[~div], sweep["recall"][1:][~div],
                               rtol=0, atol=1e-9)

    # Headline for the appendix: once all 90 positives are accepted (10 of
    # them wrongly), the official accounting reads recall ~ 1.0 while ours
    # reads 80/90 — the 10 wrong-matched positive queries vanish from the
    # official denominator.
    full = n_wrong_acc == 10
    assert full.any()
    i = np.flatnonzero(full & (fn == 0))
    assert i.size > 0
    assert np.all(r_off[i] > 0.999)
    np.testing.assert_allclose(sweep["recall"][1:][i], 80.0 / 90.0,
                               rtol=0, atol=1e-12)


def test_r_at_100p_relationship():
    """§4.3 case 3: our fixed-denominator R@100P can never exceed the
    official's — and in fact they are EQUAL by construction: precision = 1
    iff FP = 0 iff no accepted match is wrong, and with zero wrong
    acceptances TP+FN = n_pos, so both recall definitions coincide on the
    entire 100%-precision segment. The accounting difference therefore
    cannot move R@100P at all; it inflates only the P<1 part of the official
    curve (hence AP / max F1). Asserted as ours <= official plus exact
    equality within the 1e-10-denominator tolerance."""
    geo, D1, qpos, mpos = _build_world()
    D2 = _case2_matrix(geo, D1)
    _s, _d, _qc, sweep, _taus, p_off, r_off, _co, _cu = \
        _run_both(D2, geo, qpos, mpos)

    at_full_p = p_off >= 1.0 - 1e-9          # tolerance for TP/(TP+1e-10)
    r100_official = float(r_off[at_full_p].max()) if at_full_p.any() else 0.0
    r100_ours = sweep["R@100P"]

    assert r100_ours > 0.0                   # non-degenerate scenario
    np.testing.assert_allclose(r100_ours, 80.0 / 90.0, rtol=0, atol=1e-12)
    assert r100_ours <= r100_official + 1e-9
    np.testing.assert_allclose(r100_ours, r100_official, rtol=0, atol=1e-9)
