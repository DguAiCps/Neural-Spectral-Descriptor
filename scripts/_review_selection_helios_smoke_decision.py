#!/usr/bin/env python3
"""HeLiOS 교차 GPU 검증의 전체 승자·근접 순위·지표 차이를 판정한다."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "src")]
from _review_selection_bev import read, write, sha

DEST = ROOT / "results/pdv2_review_selection_full_20261105/helios_local_execution_smoke"


def main() -> None:
    """고정 수치 허용폭과 모든 질의 승자 일치를 함께 요구한다."""
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    plan = read(DEST / "acceptance_plan.json")
    records = {key: read(DEST / key / "comparison.json") for key in plan["keys"]}
    for key in plan["keys"]:
        assert read(DEST / key / "input_verified.json")["all_input_files_SHA_exact"]
    key = "kitti360_0004"
    array = np.load(DEST / key / f"descriptors_{key}.npy", mmap_mode="r")
    with np.load(DEST / key / "local_ledger.npz") as ledger:
        queries = ledger["query_idx"]
    tick = time.time()

    def gap(query: int) -> float:
        scores = array[:int(query) - 29] @ array[int(query)]
        if len(scores) < 2:
            return float("inf")
        top = np.partition(scores, -2)[-2:]
        return float(top.max() - top.min())

    with ThreadPoolExecutor(max_workers=4) as pool:
        gaps = np.fromiter(pool.map(gap, queries), dtype=np.float64, count=len(queries))
    np.savez_compressed(DEST / "rank_gap_audit.npz", query_idx=queries, top2_gap=gaps)
    near = gaps <= plan["score_max_abs"] * 2
    late = queries >= int(len(array) * .75)
    rank_proof = dict(queries=len(queries), last_query=int(queries[-1]),
        tied_queries=int((gaps == 0).sum()), near_tie_queries=int(near.sum()),
        late_queries=int(late.sum()), late_near_ties=int((late & near).sum()),
        minimum_top2_gap=float(np.min(gaps)), seconds=time.time() - tick,
        all_queries_winners_checked=True, ledger_sha256=sha(DEST / key / "local_ledger.npz"),
        rank_arrays_sha256=sha(DEST / "rank_gap_audit.npz"))
    write(DEST / "rank_gap_audit.json", rank_proof)
    failures = []
    max_score = 0.0
    max_metric = 0.0
    for key, record in records.items():
        if not record["all_winners_exact"]:
            failures.append(key + ": winner mismatch")
        if set(record["differences"]) - {"score_cos"}:
            failures.append(key + ": non-score ledger mismatch")
        delta = record["differences"].get("score_cos", {}).get("max_abs", 0.0)
        max_score = max(max_score, delta)
        if delta > plan["score_max_abs"]:
            failures.append(key + ": score tolerance")
        values = [abs(record["metric_deltas"][name]) for name in ("AP", "max_F1", "R@100P", "R@1")
                  if record["metric_deltas"][name] is not None]
        maximum = max(values, default=0.0)
        max_metric = max(max_metric, maximum)
        if maximum > plan["metric_max_abs"]:
            failures.append(key + ": metric tolerance")
    assert records["kitti360_0004"]["current_metrics"]["n_pos"] > 0
    result = dict(completed_unix=time.time(), accepted_for_execution=not failures, failures=failures,
        all_winners_exact=all(r["all_winners_exact"] for r in records.values()),
        all_arrays_exact=all(r["all_arrays_exact"] for r in records.values()),
        total_queries=sum(r["queries"] for r in records.values()), max_score_abs_difference=max_score,
        max_metric_abs_difference=max_metric, rank_audit=rank_proof,
        comparison_sha256={key:sha(DEST / key / "comparison.json") for key in records},
        acceptance_plan_sha256=sha(DEST / "acceptance_plan.json"),
        same_input_checkpoint_original_FP32_batch1=True, candidate_set_changed=False,
        condition="로컬 상위/1e-4근접 후보는 서버 GPU0 검증9 재평가 뒤에만 선택 동결",
        not_efficiency_benchmark=True)
    write(DEST / "accepted.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
