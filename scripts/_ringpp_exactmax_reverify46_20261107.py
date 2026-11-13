#!/usr/bin/env python3
"""정확max RING++ 46시험 원장을 실제 파일에서 다시 집계한다(학습·재평가 없음)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_pilot import CITY, TEST  # noqa: E402

SOURCE = REPO / "results/pdv2_ringpp_maxfix_20261105"
COMMON = REPO / "results/pdv2_symmetric3_20261013/s1"
TABLE2 = REPO / "results/pdv2_baseline_seeds/summary.json"
OUT = REPO / "results/pdv2_baseline_revision_20261107/ringpp/test_reverify"
METRICS = ("AP", "max_F1", "R@100P", "R@1")
CLAIMED = {"AP": 0.445363022, "max_F1": 0.493928360, "R@100P": 0.120762904, "R@1": 0.767860312}


def sha(path: Path) -> str:
    """파일 sha256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(score: np.ndarray, top1: np.ndarray, nearest: np.ndarray) -> dict:
    """5m 양성·25m 초과 음성, 점수 동점 묶음 단위의 PR에서 네 지표를 낸다."""
    positive = nearest <= 5
    used = (positive | (nearest > 25)) & np.isfinite(score)
    hit = positive & (top1 <= 5)
    order = np.argsort(-score[used], kind="stable")
    s, h = score[used][order], hit[used][order]
    last = np.r_[s[1:] != s[:-1], True]
    tp = np.cumsum(h)[last].astype(np.float64)
    accepted = np.flatnonzero(last).astype(np.float64) + 1
    precision = np.r_[1.0, tp / accepted]
    recall = np.r_[0.0, tp / positive.sum()]
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-300), 0)
    return {"AP": float(np.sum(np.diff(recall) * envelope[1:])), "max_F1": float(f1.max()),
            "R@100P": float(recall[precision >= 1].max()), "R@1": float(hit[positive].mean())}


def main() -> None:
    """46원장 해시·공통질의·과거후보를 확인하고 6도시 macro를 다시 만든다."""
    OUT.mkdir(parents=True, exist_ok=True)
    claimed = json.loads((SOURCE / "summary_verified.json").read_text())
    partial = json.loads((SOURCE / "partial_verified.json").read_text())["per_sequence"]
    per, ledgers, queries, positives = {}, {}, 0, 0
    for sensor, seq in TEST:
        name = f"{sensor}/{seq}"
        path = SOURCE / "odom" / f"dump_ringpp_{sensor}_{seq}.npz"
        digest = sha(path)
        assert digest == partial[name]["ledger_sha256"], name
        ledgers[str(path.relative_to(REPO))] = digest
        with np.load(path) as z, np.load(COMMON / f"test_selected_symmetric_pbev128_main_{sensor}_{seq}.npz") as ref:
            meta = json.loads(str(z["meta"]))
            assert "np.fmax.at" in meta["pipeline"], name
            index = np.searchsorted(z["query_idx"], ref["query_idx"])
            assert np.array_equal(z["query_idx"][index], ref["query_idx"]), name
            assert np.array_equal(z["min_past_gt_dist"][index], ref["min_past_gt_dist"]), name
            assert np.all(z["j_star"][index] <= ref["query_idx"] - 30), name
            score, top1, nearest = (z[k][index] for k in ("score_corr", "top1_geo_dist", "min_past_gt_dist"))
        queries += len(index)
        n_pos = int((nearest <= 5).sum())
        positives += n_pos
        row = {"n_queries": int(len(index)), "n_positive": n_pos}
        if n_pos:
            row.update(metrics(score, top1, nearest))
            for k in METRICS:
                assert abs(row[k] - claimed["ringpp_maxfix"]["per_sequence"][name][k]) < 1e-12, (name, k)
        per[name] = row
    cities = {city: {k: float(np.mean([per[s][k] for s in names if per[s]["n_positive"]])) for k in METRICS}
              for city, names in CITY.items()}
    macro = {k: float(np.mean([cities[c][k] for c in cities])) for k in METRICS}
    table2 = json.loads(TABLE2.read_text())["methods"]["ringpp"]["mean_std"]
    result = {
        "passed": True, "sequences": len(per), "common_queries": queries, "positives": positives,
        "excluded_no_positive": [s for s, v in per.items() if not v["n_positive"]],
        "macro": macro, "cities": cities,
        "claimed_macro": {k: claimed["ringpp_maxfix"]["macro"][k] for k in METRICS},
        "max_abs_diff_vs_summary_verified": max(abs(macro[k] - claimed["ringpp_maxfix"]["macro"][k]) for k in METRICS),
        "matches_instruction_values_9dp": all(abs(macro[k] - CLAIMED[k]) < 5e-10 for k in METRICS),
        "table2_current_oldkernel": {k: table2[k]["mean"] for k in METRICS},
        "delta_exactmax_minus_oldkernel_pp": {k: 100 * (macro[k] - table2[k]["mean"]) for k in METRICS},
        "per_sequence": per, "ledger_sha256": ledgers,
        "summary_verified_sha256": sha(SOURCE / "summary_verified.json"),
        "method": "독립 PR 계산(동점 묶음, 보간 정밀도 AP), 5m 양성·25m 초과 음성, 45양성시퀀스→6도시 동일가중",
        "source_sha256": sha(Path(__file__)),
    }
    assert result["sequences"] == 46 and queries == 195726 and positives == 61682
    assert result["max_abs_diff_vs_summary_verified"] < 1e-12 and result["matches_instruction_values_9dp"]
    (OUT / "verification.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("passed", "common_queries", "macro", "table2_current_oldkernel",
                                             "delta_exactmax_minus_oldkernel_pp")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
