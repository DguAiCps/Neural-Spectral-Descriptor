#!/usr/bin/env python3
"""혼합 값의 후보 확보·정답 선택·고정밀 오류를 같은 질의에서 진단한다."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_mixed_value as run
import _pdv2_value_relation_diagnose as operator
from _pdv2_relation_precision import records as precision_records
from _pdv2_no_context_ablation import sha256, write_json


def state(row: dict, coverage: dict) -> np.ndarray:
    """양성 질의를 후보 누락0·후보 있지만 오답1·최종 정답2로 분류한다."""
    correct = row["top1_geo_dist"] <= 5
    assert not np.any(correct & ~coverage["union_hit"])
    return np.where(correct, 2, np.where(coverage["union_hit"], 1, 0))


def matrix(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> list:
    """행은 기준, 열은 혼합 모델인 3×3 정답 상태 전이 수를 계산한다."""
    result = np.bincount(3 * before[mask] + after[mask], minlength=9).reshape(3, 3)
    assert int(result.sum()) == int(mask.sum())
    return result.tolist()


def main() -> None:
    """선택 가중치를 고정하고 재학습 없이 원장과 활성값을 진단한다."""
    spec = run.protocol(True)
    assert run.read(run.OUT / "status.json")["audit_passed"]
    summary = run.read(run.OUT / "summary.json")
    sources = {str(p.relative_to(REPO)): sha256(p) for p in (
        Path(__file__), REPO / "scripts/_pdv2_relation_precision.py",
        REPO / "scripts/_pdv2_value_relation_diagnose.py", run.OUT / "summary.json")}
    rows, coverage, precision = {}, {}, {}
    for arm, saved in summary["per_seed"]["1"].items():
        epoch = saved["selected"]["epoch"]
        folder = run.base.root(1, arm)
        precision[arm] = precision_records(folder, epoch)
        rows[arm], coverage[arm] = {}, {}
        for seq in spec["val"]:
            name = seq.replace("/", "_") + ".npz"
            p, c = folder / f"epoch{epoch}" / name, folder / f"epoch{epoch}" / ("coverage_" + name)
            rows[arm][seq], coverage[arm][seq] = run.base.exp.arrays(p), run.base.exp.arrays(c)
            sources[str(p.relative_to(REPO))] = sha256(p)
            sources[str(c.relative_to(REPO))] = sha256(c)
    comparisons = {}
    for ref in ("diff_linear", "diff_tanh", "standard_mixed"):
        sequences = {}
        for seq in spec["val"]:
            before, after = rows[ref][seq], rows["diff_mixed"][seq]
            old, new = coverage[ref][seq], coverage["diff_mixed"][seq]
            for field in ("query_idx", "min_past_gt_dist", "group"):
                np.testing.assert_array_equal(before[field], after[field])
            np.testing.assert_array_equal(old["aux_hit"], new["aux_hit"])
            positive = before["min_past_gt_dist"] <= 5
            old_state, new_state = state(before, old), state(after, new)
            name = seq.replace("/", "_")
            p_old, p_new = precision[ref][name], precision["diff_mixed"][name]
            pairs_old = {(p["query"], p["winner"]) for p in p_old["highest_false_cases"]}
            pairs_new = {(p["query"], p["winner"]) for p in p_new["highest_false_cases"]}
            sequences[seq] = {
                "positive": int(positive.sum()), "transition": matrix(old_state, new_state, positive),
                "direction_transitions": {str(g): matrix(old_state, new_state,
                    positive & (before["group"] == g)) for g in (0, 1, 2)},
                "key_top1_correct_before": int((positive & (old["key_top1_geo"] <= 5)).sum()),
                "key_top1_correct_after": int((positive & (new["key_top1_geo"] <= 5)).sum()),
                "key_pool_hit_before": int((positive & old["key_hit"]).sum()),
                "key_pool_hit_after": int((positive & new["key_hit"]).sum()),
                "aux_pool_hit": int((positive & old["aux_hit"]).sum()),
                "highest_false_same_pairs_and_score": pairs_old == pairs_new and
                    p_old["highest_false_score"] == p_new["highest_false_score"],
                "strict_correct_above_false_delta": p_new["strict_correct_above_false"] -
                    p_old["strict_correct_above_false"],
            }
        families = {}
        for family in ("kitti", "helipr", "mulran"):
            selected = [v for seq, v in sequences.items() if seq.startswith(family)]
            families[family] = {
                "transition": np.sum([v["transition"] for v in selected], axis=0).tolist(),
                **{key: sum(v[key] for v in selected) for key in (
                    "positive", "key_top1_correct_before", "key_top1_correct_after",
                    "key_pool_hit_before", "key_pool_hit_after", "aux_pool_hit")},
            }
        comparisons[ref] = {"sequences": sequences, "families": families,
            "total_transition": np.sum([v["transition"] for v in sequences.values()], axis=0).tolist()}
    write_json(run.OUT / "candidate_diagnosis.json", {
        "comparisons": comparisons, "precision": precision, "sources": sources,
        "state_order": ["all_true_candidates_absent", "true_in_pool_but_wrong", "correct"],
        "scope": "시드1 선택모델 고정. 상태전이는 양성 질의의 합계이며 계열 평균 AP와 다르다. 학습된 모델 간 비교라 tanh 포화가 원인이라고 확정하지 않는다.",
        "test_used": False,
    })
    print("CANDIDATES_DIAGNOSED", flush=True)
    # 기존 관측 코드를 재사용하되 이번에 학습한 두 모델만 측정한다.
    operator.run = run
    original_arms = run.ARMS
    run.ARMS = tuple(run.CFG["train_arms"])
    try:
        operator.main()
    finally:
        run.ARMS = original_arms


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
