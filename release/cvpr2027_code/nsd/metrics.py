"""Evaluation protocol (Sec. 4.1): query classes, per-sequence metrics and macro averages over cities.

Ground truth comes from the dataset reference poses (poses_gt) and is used only here.

* Query classes from the distance to the nearest eligible past keyframe (j <= q - 30):
  positive  <= 5 m, negative  > 25 m, otherwise ambiguous and excluded from the precision-recall sweep.
* A returned place is correct when it lies within 5 m of the query. An accepted pair is a true positive only
  for a positive query with a correct returned place; every other accepted pair is a false positive.
  Recall is TP / (number of positive queries) at every threshold.
* AP: step-wise interpolated average precision over all unique scores (accept if score >= tau);
  max F1 over the same sweep; R@100P = largest recall with no false positive;
  revisit R@1 = fraction of positive queries whose returned place is correct (ignores the score).
* Macro: sequence metrics -> mean over the sequences of a city that have positives -> mean over cities.
"""
from __future__ import annotations

import numpy as np

POSITIVE_M = 5.0
NEGATIVE_M = 25.0
CORRECT_M = 5.0
MIN_GAP = 30
METRICS = ("AP", "max_F1", "R@100P", "R@1")


def min_past_distance(positions: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Distance (3-D, reference poses) from each query to its nearest keyframe j <= q - 30."""
    return np.array([np.linalg.norm(positions[:q - MIN_GAP + 1] - positions[q], axis=1).min() for q in query])


def returned_distance(positions: np.ndarray, query: np.ndarray, returned: np.ndarray) -> np.ndarray:
    """Distance (reference poses) between each query and its returned place."""
    return np.linalg.norm(positions[query] - positions[returned], axis=1)


def sequence_metrics(min_past: np.ndarray, top1: np.ndarray, score: np.ndarray) -> dict | None:
    """AP, max F1, R@100P and revisit R@1 of one sequence; None if it has no positive query."""
    positive = min_past <= POSITIVE_M
    negative = min_past > NEGATIVE_M
    n_pos = int(positive.sum())
    if not n_pos:
        return None
    correct = top1 <= CORRECT_M
    eligible = np.flatnonzero((positive | negative) & np.isfinite(score))
    order = eligible[np.argsort(-score[eligible], kind="stable")]
    s = score[order]
    last = np.flatnonzero(np.r_[s[1:] != s[:-1], True])             # one sweep point per unique score
    tp = np.r_[0, np.cumsum((positive & correct)[order])[last]]
    fp = np.r_[0, np.cumsum((negative | (positive & ~correct))[order])[last]]
    recall = tp / n_pos
    accepted = tp + fp
    precision = np.where(accepted > 0, tp / np.maximum(accepted, 1), 1.0)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    pr = precision + recall
    f1 = np.where(pr > 0, 2.0 * precision * recall / np.maximum(pr, 1e-300), 0.0)
    return {"AP": float(np.sum(np.diff(recall) * envelope[1:])),
            "max_F1": float(np.max(f1)),
            "R@100P": float(recall[precision >= 1.0].max()),
            "R@1": float(np.mean(top1[positive] <= CORRECT_M)),
            "n_pos": n_pos}


def macro(per_sequence: dict, cities: dict, keys=METRICS) -> dict:
    """{sequence: metrics or None} -> per-city means (sequences with positives) and their unweighted mean."""
    groups = {}
    for city, sequences in cities.items():
        rows = [per_sequence[s] for s in sequences if per_sequence.get(s) is not None]
        if rows:
            groups[city] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    return {"groups": groups, "macro": {k: float(np.mean([g[k] for g in groups.values()])) for k in keys}}


def seed_mean(macros: list, keys=METRICS) -> dict:
    """Mean and sample standard deviation over training seeds of the macro metrics."""
    out = {}
    for k in keys:
        v = np.asarray([m["macro"][k] for m in macros], float)
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None}
    return out
