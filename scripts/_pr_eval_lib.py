#!/usr/bin/env python3
"""Torch-free PR-evaluation library for the pre-registered causal
loop-closure evaluation (docs/CAUSAL_PR_PREREGISTRATION.md, rev.2 —
frozen accounting; this module implements §1.2, §1.3, §1.4, §1.6 verbatim
and is the "PR lib" of track D / verification item 4).

Frozen rules implemented here
  * Query 3-way classification (§1.2), from the distance between query t and
    its NEAREST past-DB frame (past = index <= t - 30; the caller precomputes
    this and only passes frames with t >= 30):
      positive   d_min <= 5 m
      negative   d_min > 25 m
      ambiguous  5 m < d_min <= 25 m  -> excluded from PR, counted + reported
  * Buffer zone classifies QUERIES only, never accepted matches (§1.3). The
    single match-correctness criterion is: returned top-1 within 5 m of the
    query. An accepted match beyond 5 m is FP regardless of query class
    (anti-optimism rule — a mis-returned 10 m match on a positive query is FP,
    and that query is NOT simultaneously TP or FN).
      TP: positive query, accepted, top-1 <= 5 m
      FN: positive query, not accepted
      TN: negative query, not accepted
      FP: any accepted match with top-1 > 5 m (positive OR negative query)
    Precision = TP/(TP+FP).
    Recall = TP / n_pos with n_pos = TOTAL positive queries — the denominator
    is threshold-independent (never TP/(TP+FN); the RING plot_PR_curve.py
    variable-denominator accounting is explicitly rejected, §1.3/§4.3).
  * Score convention: SINGLE convention, higher score = more confident accept
    (accepted = score >= tau). Callers with distance-like scores must negate
    them before calling; `higher_is_better=False` raises rather than silently
    flipping. NaN score = empty candidate set = never accepted at any tau
    (a positive query with NaN score is a permanent FN).
  * Metrics (§1.4): full sweep over all unique finite scores of ELIGIBLE
    (positive|negative) queries plus a +inf sentinel (0 acceptances; the
    precision of an empty acceptance set is defined as 1.0, the standard
    P(R=0) anchor). AP is the step-wise right-continuous interpolated AP:
        AP = sum_i (R_i - R_{i-1}) * P_interp(R_i),
        P_interp(R) = max precision over sweep points with recall >= R.
    R@100P / R@99P / R@95P = max recall among sweep points with precision >=
    1.0 / 0.99 / 0.95 (0.0 if only the sentinel qualifies). These are test
    sweep statistics; the deployable operating point is `operating_point`
    at a val-selected fixed tau*.
  * Aggregation (§1.6): per-sequence sweep -> dataset-group macro mean ->
    overall = macro mean of the group means (no global single-tau pooling).

Result payloads use schema "pr-eval-v1" and land in results/pd2026_pr/,
mirroring the scripts/_xs_eval_lib.py house style.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results" / "pd2026_pr"

SCHEMA = "pr-eval-v1"
POSITIVE_MAX_M = 5.0    # query class: d_min <= 5 m -> positive
NEGATIVE_MIN_M = 25.0   # query class: d_min > 25 m -> negative
MATCH_RADIUS_M = 5.0    # single match-correctness criterion (never buffered)
MIN_INDEX_GAP = 30      # past DB = index <= t - 30 (caller-enforced, documented)
AGG_METRICS = ("AP", "max_F1", "R@100P", "R@99P", "R@95P")


# ----------------------------------------------------- query classification

def classify_queries(min_past_gt_dist: np.ndarray) -> dict:
    """Frozen 3-way query classification (§1.2).

    min_past_gt_dist[t] = distance from query t to the NEAREST past-DB frame
    (past = index <= t - 30). The caller precomputes this and passes only
    frames with t >= 30; a query with no frame within range naturally carries
    a large / +inf distance and classifies negative.

    Returns boolean masks partitioning the queries — positive (d <= 5),
    negative (d > 25), ambiguous (5 < d <= 25; excluded from PR downstream,
    counted here so it is always reported) — plus counts.
    """
    d = np.asarray(min_past_gt_dist, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError(f"min_past_gt_dist must be 1-D, got shape {d.shape}")
    if np.isnan(d).any():
        raise ValueError("min_past_gt_dist contains NaN (would misclassify)")
    positive = d <= POSITIVE_MAX_M
    negative = d > NEGATIVE_MIN_M
    ambiguous = ~positive & ~negative
    return {
        "positive": positive,
        "negative": negative,
        "ambiguous": ambiguous,
        "n_pos": int(positive.sum()),
        "n_neg": int(negative.sum()),
        "n_ambiguous": int(ambiguous.sum()),
        "n_total": int(len(d)),
    }


# ------------------------------------------------------------- accounting

def _prepare(scores, top1_dists, query_class):
    """Shared validation for pr_sweep / operating_point.

    Enforces the §1.3 geometric-consistency invariant: a negative query's
    nearest past frame is > 25 m away, so its returned top-1 can never be
    within 5 m — if it is, poses/scores are inconsistent and we raise instead
    of silently accounting. NaN top-1 distance (empty candidate set, paired
    with a NaN score) is exempt: no match was returned.
    """
    s = np.asarray(scores, dtype=np.float64)
    d = np.asarray(top1_dists, dtype=np.float64)
    pos = np.asarray(query_class["positive"], dtype=bool)
    neg = np.asarray(query_class["negative"], dtype=bool)
    amb = np.asarray(query_class["ambiguous"], dtype=bool)
    n = len(s)
    if not (len(d) == len(pos) == len(neg) == len(amb) == n):
        raise ValueError("scores / top1_dists / query_class length mismatch")
    if (np.any(pos & neg) or np.any(pos & amb) or np.any(neg & amb)
            or not np.all(pos | neg | amb)):
        raise ValueError("query_class masks must partition the query set")
    bad = neg & np.isfinite(d) & (d <= MATCH_RADIUS_M)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} negative query(ies) with returned top-1 within "
            f"{MATCH_RADIUS_M} m — impossible if d_min > {NEGATIVE_MIN_M} m; "
            "query classification and top1_dists are inconsistent")
    correct = np.zeros(n, dtype=bool)
    fin_d = np.isfinite(d)
    correct[fin_d] = d[fin_d] <= MATCH_RADIUS_M
    return s, pos, neg, amb, correct


def pr_sweep(scores, top1_dists, query_class, higher_is_better=True) -> dict:
    """Full-threshold PR sweep with the frozen §1.3 accounting.

    scores      : per-query top-1 decision score; np.nan = empty candidate
                  set = never accepted at any threshold.
    top1_dists  : geometric distance of the RETURNED top-1 match to the query
                  (np.nan only when no match was returned).
    query_class : dict of masks from classify_queries().

    Single score convention: higher = more confident; accepted = score >= tau.
    Callers with distance-like scores negate them first (the returned
    thresholds are then negated distances); higher_is_better=False raises.

    Thresholds are all unique finite scores of eligible (positive|negative)
    queries, descending, preceded by a +inf sentinel (0 acceptances,
    precision defined 1.0). Ambiguous queries are excluded entirely — they
    contribute neither confusion counts nor sweep points.

    Returns arrays (thresholds, precision, recall — aligned, tau-descending /
    recall-non-decreasing) and scalars AP, max_F1 (+ max_F1_tau; ties broken
    toward the highest tau), R@100P, R@99P, R@95P, and counts.
    """
    if not higher_is_better:
        raise ValueError(
            "single-convention library: negate the scores yourself and pass "
            "higher_is_better=True (accepted = score >= tau)")
    s, pos, neg, amb, correct = _prepare(scores, top1_dists, query_class)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())

    # Sort eligible finite-score queries by descending score; NaN-score
    # queries never enter `order` (never accepted: positives stay FN at every
    # tau, negatives stay TN).
    fin = (pos | neg) & np.isfinite(s)
    order = np.flatnonzero(fin)[np.argsort(-s[fin], kind="stable")]
    s_sorted = s[order]
    tp_cum = np.cumsum((pos & correct)[order])
    fp_cum = np.cumsum((neg | (pos & ~correct))[order])
    pos_acc_cum = np.cumsum(pos[order])
    neg_acc_cum = np.cumsum(neg[order])

    # One sweep point per unique score value: at tau = v, accepted = the full
    # prefix up to the LAST occurrence of v. +inf sentinel first (R=0 anchor).
    if len(order):
        last = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])
    else:
        last = np.empty(0, dtype=np.int64)
    thresholds = np.r_[np.inf, s_sorted[last]]
    tp = np.r_[0, tp_cum[last]].astype(np.int64)
    fp = np.r_[0, fp_cum[last]].astype(np.int64)
    fn = n_pos - np.r_[0, pos_acc_cum[last]].astype(np.int64)
    tn = n_neg - np.r_[0, neg_acc_cum[last]].astype(np.int64)

    # Accounting identity: every eligible query lands in exactly one cell
    # (TP / FP / FN / TN) at every threshold.
    assert np.all(tp + fp + fn + tn == n_pos + n_neg), "confusion cells leak"
    # FROZEN INVARIANT (§1.3): the recall denominator is the total positive
    # query count at EVERY threshold. TP+FN *does* vary with tau (a positive
    # query accepted with a bad top-1 is FP, neither TP nor FN) — recall is
    # NEVER computed as TP/(TP+FN).
    recall_denom = np.full(thresholds.shape, n_pos, dtype=np.int64)
    assert np.unique(recall_denom).size <= 1, "recall denominator varies"
    assert np.all(tp + fn <= recall_denom), "TP+FN exceeds n_pos"

    recall = tp / max(n_pos, 1)
    acc = tp + fp
    precision = np.where(acc > 0, tp / np.maximum(acc, 1), 1.0)

    # AP = sum_i (R_i - R_{i-1}) * P_interp(R_i), P_interp(R) = max precision
    # at recall >= R (right-continuous step-wise interpolation). Points are
    # recall-non-decreasing, so P_interp is the reverse running max.
    p_interp = np.maximum.accumulate(precision[::-1])[::-1]
    ap = float(np.sum(np.diff(recall) * p_interp[1:]))

    pr_sum = precision + recall
    f1 = np.where(pr_sum > 0,
                  2.0 * precision * recall / np.maximum(pr_sum, 1e-300), 0.0)
    i_best = int(np.argmax(f1))  # first max in tau-descending order

    def _max_recall_at(p_req: float) -> float:
        m = precision >= p_req
        return float(recall[m].max()) if m.any() else 0.0

    return {
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "AP": ap,
        "max_F1": float(f1[i_best]),
        "max_F1_tau": float(thresholds[i_best]),
        "R@100P": _max_recall_at(1.0),
        "R@99P": _max_recall_at(0.99),
        "R@95P": _max_recall_at(0.95),
        "counts": {"n_pos": n_pos, "n_neg": n_neg,
                   "n_ambiguous": int(amb.sum()), "n_total": int(len(s))},
    }


def operating_point(scores, top1_dists, query_class, tau) -> dict:
    """Confusion counts + P/R/F1 at a FIXED externally-chosen tau (the
    val-selected tau* row, §1.4). Same frozen accounting as pr_sweep;
    accepted = finite score >= tau, ambiguous queries excluded entirely."""
    s, pos, neg, amb, correct = _prepare(scores, top1_dists, query_class)
    accepted = (pos | neg) & np.isfinite(s) & (s >= float(tau))
    tp = int((accepted & pos & correct).sum())
    fp = int((accepted & (neg | (pos & ~correct))).sum())
    fn = int((pos & ~accepted).sum())
    tn = int((neg & ~accepted).sum())
    n_pos = int(pos.sum())
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / n_pos if n_pos else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) \
        if (precision + recall) else 0.0
    return {"tau": float(tau), "precision": precision, "recall": recall,
            "F1": f1, "TP": tp, "FP": fp, "FN": fn, "TN": tn}


# ------------------------------------------------------------- aggregation

def aggregate_groups(per_sequence_results: dict, groups: dict) -> dict:
    """§1.6 macro aggregation: per-sequence -> dataset-group mean -> overall
    = unweighted mean of the group means, per metric in AGG_METRICS.
    No global single-tau pooling (score scales are heterogeneous, disclosed).

    per_sequence_results: {seq_key: dict containing the AGG_METRICS scalars}
    (pr_sweep results work directly). groups: {group_name: [seq_keys]}.
    Missing seq keys raise KeyError (fail loudly).
    """
    group_means = {}
    for gname, keys in groups.items():
        if not keys:
            raise ValueError(f"group {gname!r} has no sequences")
        group_means[gname] = {
            m: float(np.mean([per_sequence_results[k][m] for k in keys]))
            for m in AGG_METRICS}
    overall = {m: float(np.mean([group_means[g][m] for g in groups]))
               for m in AGG_METRICS}
    return {"groups": group_means, "overall": overall}


# ------------------------------------------------------------------ output

def _finite_list(arr) -> list:
    """JSON-strict list: non-finite floats (the +inf sentinel) become None."""
    return [float(v) if np.isfinite(v) else None
            for v in np.asarray(arr, dtype=np.float64)]


def result_payload(method: str, sequence: str, sweep: dict, *,
                   variant=None, seed=None, op=None, extra=None) -> dict:
    """Schema pr-eval-v1 payload (house style of _xs_eval_lib.result_payload).
    `op` is the operating_point dict for the val-selected tau* row."""
    payload = {
        "schema": SCHEMA,
        "method": method,
        "sequence": sequence,
        "protocol": {
            "positive_max_m": POSITIVE_MAX_M,
            "negative_min_m": NEGATIVE_MIN_M,
            "match_radius_m": MATCH_RADIUS_M,
            "min_index_gap": MIN_INDEX_GAP,
            "recall_denominator": "n_pos (fixed, threshold-independent)",
            "score_convention": "higher_is_better",
            "ap": "step-wise right-continuous interpolated",
        },
        "counts": sweep["counts"],
        "metrics": {
            "AP": sweep["AP"],
            "max_F1": sweep["max_F1"],
            "max_F1_tau": (float(sweep["max_F1_tau"])
                           if np.isfinite(sweep["max_F1_tau"]) else None),
            "R@100P": sweep["R@100P"],
            "R@99P": sweep["R@99P"],
            "R@95P": sweep["R@95P"],
        },
        "curve": {
            "thresholds": _finite_list(sweep["thresholds"]),
            "precision": _finite_list(sweep["precision"]),
            "recall": _finite_list(sweep["recall"]),
        },
    }
    if variant is not None:
        payload["variant"] = variant
    if seed is not None:
        payload["seed"] = seed
    if op is not None:
        payload["op"] = op
    if extra:
        payload.update(extra)
    return payload


def write_result(payload: dict, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2))
    return p
