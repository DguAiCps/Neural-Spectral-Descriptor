#!/usr/bin/env python3
"""BLAS1의 원래 질의 계산을 유지하고 서로 다른 질의만 CPU4에 나눈다."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "src")]
from _review_selection_bev import read, write, sha
from _review_selection_full_inputs import OUT, cache
from _causal_pr_dump import prefix_topk


def ledger_parallel(desc: np.ndarray, positions: np.ndarray, workers: int = 4,
                    queries: np.ndarray | None = None) -> dict:
    """질의별 동일 GEMV·top1·기하 연산을 독립 실행하고 원래 순서로 모은다."""
    assert workers in (1, 4) and len(desc) == len(positions)
    qs = np.arange(30, len(desc), dtype=np.int64) if queries is None else np.asarray(queries, dtype=np.int64)
    assert np.all((qs >= 30) & (qs < len(desc)))

    def block(subset: np.ndarray) -> list:
        rows = []
        for value in subset:
            q = int(value)
            prefix = q - 29
            winner = int(prefix_topk(desc, q, prefix, 1)[0])
            score = float(desc[winner] @ desc[q])
            distance = float(np.linalg.norm(positions[winner] - positions[q]))
            minimum = np.linalg.norm(positions[:prefix] - positions[q], axis=1).min()
            rows.append((winner, score, distance, minimum, prefix))
        return rows

    with threadpool_limits(limits=1, user_api="blas"):
        assert all(item["num_threads"] == 1 for item in threadpool_info() if item["user_api"] == "blas")
        if workers == 1:
            rows = block(qs)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = []
                for result in pool.map(block, [qs[i:i + 64] for i in range(0, len(qs), 64)]):
                    rows.extend(result)
                    if len(rows) % 1024 == 0:
                        print("PARALLEL_QUERY", len(rows), len(qs), flush=True)
    values = np.asarray(rows)
    return dict(query_idx=qs.copy(), j_star=values[:, 0].astype(np.int64), score_cos=values[:, 1],
        top1_geo_dist=values[:, 2], min_past_gt_dist=values[:, 3], n_candidates=values[:, 4].astype(np.int64))


def main() -> None:
    """합성 동률과 실제 긴 검증 전체 원장에서 원래 계산과 바이트를 대조한다."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    home = OUT / "helios_query_parallel"
    home.mkdir(exist_ok=True)
    manifest = read(OUT / "cosine_acceleration_audit/manifest.json")
    case = next(r for r in manifest["cases"] if r["label"].startswith("helios"))
    source = ROOT / case["descriptors"]
    assert sha(source) == case["sha256"]
    vectors = np.load(source, mmap_mode="r")
    positions = np.load(ROOT / case["positions"])
    digest = hashlib.sha256()
    digest.update(json.dumps([list(vectors.shape), str(vectors.dtype), sha(cache("mulran", "Sejong03")),
                             sha(ROOT / "scripts/_causal_bevplace_pr.py")]).encode())
    digest.update(np.ascontiguousarray(vectors).tobytes())
    references = [(path, read(path)) for path in (OUT / "retained/val/helios").glob("*/*/mulran_Sejong03.json")]
    references = [(path, row) for path, row in references
                  if row.get("audit", {}).get("descriptor_sha256") == digest.hexdigest()]
    assert references
    reference_path, reference = references[0]
    assert sha(ROOT / reference["ledger"]) == reference["ledger_sha256"]
    plan = dict(created_unix=time.time(), source=str(source.relative_to(ROOT)), source_sha256=sha(source),
        reference_json=str(reference_path.relative_to(ROOT)), reference_json_sha256=sha(reference_path),
        reference_ledger_sha256=reference["ledger_sha256"], BLAS_threads=1, query_workers=4,
        fixed_model_and_scoring=True, GPU_used=False, changes_reduction_order=False)
    write(home / "plan.json", plan)
    rng = np.random.default_rng(50310)
    synthetics = []
    n, dim = 8192, vectors.shape[1]
    qs = np.asarray([30, 127, 255, 511, 1023, 4095, n - 16], dtype=np.int64)
    for tag in ("random", "all_equal", "duplicate", "near_tie", "zero"):
        bank = rng.normal(size=(n, dim)).astype(np.float32)
        bank /= np.linalg.norm(bank, axis=1, keepdims=True)
        if tag in ("all_equal", "near_tie"):
            bank[:] = bank[0]
        if tag == "duplicate":
            bank[::2] = bank[0]
        if tag == "near_tie":
            bank[::3, 0] = np.nextafter(bank[::3, 0], np.float32(np.inf))
        if tag == "zero":
            bank[:] = 0
        pos = rng.normal(size=(n, 3))
        # 7질의만 한 블록에 들어가면 실제 병렬 실행 검산이 아니다.
        many = np.tile(qs, 48)
        expected = ledger_parallel(bank, pos, workers=1, queries=many)
        actual = ledger_parallel(bank, pos, workers=4, queries=many)
        for field in expected:
            np.testing.assert_array_equal(actual[field], expected[field])
        synthetics.append(dict(kind=tag, exact=True, queries=len(many), late_query=int(qs[-1])))
        print("SYNTHETIC_EXACT", tag, flush=True)
    del bank
    late = np.arange(len(vectors) - 1040, len(vectors) - 16, dtype=np.int64)
    current_times = {}
    for workers in (1, 4):
        tick = time.time()
        current = ledger_parallel(vectors, positions, workers=workers, queries=late)
        current_times[workers] = time.time() - tick
        if workers == 1:
            expected = current
        else:
            for field in expected:
                np.testing.assert_array_equal(current[field], expected[field])
    write(home / "progress.json", dict(synthetics=synthetics, late1024_seconds=current_times,
        note="같은 공유부하에서 차례로 측정한 실행 관측이며 논문 효율 benchmark가 아님"))
    tick = time.time()
    actual = ledger_parallel(vectors, positions)
    elapsed = time.time() - tick
    with np.load(ROOT / reference["ledger"]) as old:
        take = np.searchsorted(actual["query_idx"], old["query_idx"])
        fields = [k for k in old.files if k != "meta"]
        differences = {k: int(np.count_nonzero(actual[k][take] != old[k]))
                       for k in fields if not np.array_equal(actual[k][take], old[k])}
        count = len(old["query_idx"])
    result = dict(completed_unix=time.time(), all_array_fields_exact=not differences,
        differences=differences, n_queries=count, fields=fields, synthetic_checks=synthetics,
        BLAS_threads=1, query_workers=4, late1024_seconds=current_times, full_current_seconds=elapsed,
        historical_original_seconds=reference["timing"]["search_s"], plan_sha256=sha(home / "plan.json"),
        implementation_sha256=sha(Path(__file__)), original_sha256=sha(ROOT / "scripts/_causal_bevplace_pr.py"),
        original_prefix_sha256=sha(ROOT / "scripts/_causal_pr_dump.py"), source_descriptor_sha256=case["sha256"])
    write(home / "verified.json", result)
    assert not differences, differences
    print("FULL_EXACT", count, elapsed, current_times, flush=True)


if __name__ == "__main__":
    main()
