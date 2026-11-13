#!/usr/bin/env python3
"""R@100P를 제한하는 최고 점수 오답이 연산 변경으로 달라지는지 확인한다."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_no_context_ablation import sha256, write_json


def records(folder: Path, epoch: int) -> dict:
    """선택 모델의 모든 검증 원장에서 고정 정답 규칙을 적용한다."""
    rows = {}
    for path in sorted((folder / f"epoch{epoch}").glob("*.npz")):
        if path.name.startswith(("keys_", "coverage_")):
            continue
        with np.load(path) as data:
            positive = data["min_past_gt_dist"] <= 5
            valid = positive | (data["min_past_gt_dist"] > 25)
            correct = positive & (data["top1_geo_dist"] <= 5)
            false = valid & ~correct
            score = data["arc_unique_b100"]
            cutoff = float(score[false].max()) if false.any() else None
            strict_correct = int((correct & (score > cutoff)).sum()) if cutoff is not None else int(correct.sum())
            tied = np.flatnonzero(false & (score == cutoff)) if cutoff is not None else np.array([], dtype=int)
            measured = strict_correct / int(positive.sum()) if positive.any() else None
            expected = json.loads(path.with_suffix(".json").read_text())["metrics"]["arc_unique_b100"]["R@100P"]
            if measured is None:
                assert expected in (None, 0)
                measured = expected  # 양성 없는 시퀀스는 평균에서 제외한다.
            else:
                np.testing.assert_allclose(measured, expected, atol=1e-12, rtol=0)
            rows[path.stem] = {"positive": int(positive.sum()), "highest_false_score": cutoff,
                "strict_correct_above_false": strict_correct, "R@100P": measured,
                "highest_false_cases": [{"query": int(data["query_idx"][i]),
                    "winner": int(data["j_star"][i]), "distance_m": float(data["top1_geo_dist"][i]),
                    "has_true_loop": bool(positive[i])} for i in tied], "ledger_sha256": sha256(path)}
    return rows


def main() -> None:
    """완료된 실험만 진단하며 최고 점수 오답의 유지 여부를 기록한다."""
    for name in ("value_relation", "grouped_relation"):
        root = REPO / "results" / ("pdv2_" + name + "_20261021")
        if not (root / "summary.json").exists():
            continue
        summary = json.loads((root / "summary.json").read_text())
        models = {arm: records(root / "s1" / arm, value["selected"]["epoch"])
                  for arm, value in summary["per_seed"]["1"].items()}
        baseline = models["diff_linear"]
        comparisons = {}
        for arm, values in models.items():
            comparisons[arm] = {}
            for seq, row in values.items():
                reference = baseline[seq]
                old = {(r["query"], r["winner"]) for r in reference["highest_false_cases"]}
                new = {(r["query"], r["winner"]) for r in row["highest_false_cases"]}
                comparisons[arm][seq] = {
                    "same_highest_false_score": row["highest_false_score"] == reference["highest_false_score"],
                    "shared_highest_false_pairs": len(old & new),
                    "old_highest_false_pairs": len(old), "new_highest_false_pairs": len(new),
                    "strict_correct_change": row["strict_correct_above_false"] - reference["strict_correct_above_false"],
                }
        write_json(root / "precision_diagnosis.json", {"models": models, "versus_diff_linear": comparisons,
            "scope": "선택 모델 고정. 회색 질의 제외 후 최고 점수 오답보다 점수가 높은 정답 수로 R@100P 독립 재현. 인코더 학습이나 새 설정 선택이 아니다.",
            "runtime_sha256": sha256(Path(__file__)), "test_used": False})
        print("PRECISION_DIAGNOSED", name, flush=True)


if __name__ == "__main__":
    main()
