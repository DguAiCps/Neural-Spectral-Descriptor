#!/usr/bin/env python3
"""GPU 후보와 보수적 반올림 여유로 공식 CPU L2 결과를 보존한다.

단위 길이 FP32 벡터의 누적 반올림 오차보다 큰 여유 안의 후보를 모두
공식 CPU IndexFlatL2로 다시 계산한다. 여유 밖 후보를 배제할 수 없으면
전체 CPU 과거 색인으로 복귀한다. 별도 프로세스에서 TF32를 끈다.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import faiss
import numpy as np


def ledger_exact(descriptors: np.ndarray, positions: np.ndarray) -> tuple:
    """과거 접두의 공식 CPU 첫 후보·점수를 계산한다."""
    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"
    faiss.omp_set_num_threads(1)
    vectors = np.ascontiguousarray(descriptors, dtype=np.float32)
    count, dim = vectors.shape
    assert dim == 8192 and len(positions) == count
    norm_sq = np.einsum("ij,ij->i", vectors, vectors, dtype=np.float64)
    assert np.isfinite(vectors).all() and norm_sq.max() < 1.001
    # FP32 단위 반올림 u. 양쪽 거리와 두 후보의 오차를 함께 넉넉히 포함.
    # CPU 차이 제곱 합과 GPU 내적식 각각의 오차는 대략 4*gamma_d*R^2
    # 이하이며, 두 거리 차이까지 포함한 16배보다 큰 32배를 사용한다.
    unit = float(np.finfo(np.float32).eps) / 2
    gamma = (dim + 8) * unit / (1 - (dim + 8) * unit)
    margin = 32 * gamma * float(norm_sq.max()) + 1e-10
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(256 * 1024 * 1024)
    config = faiss.GpuIndexFlatConfig()
    config.device = 0
    config.useFloat16 = False
    gpu = faiss.GpuIndexFlatL2(resources, dim, config)
    cpu = faiss.IndexFlatL2(dim)
    queries = np.arange(30, count, dtype=np.int64)
    result = {
        "query_idx": queries,
        "j_star": np.empty(len(queries), np.int64),
        "score_negl2": np.empty(len(queries), np.float64),
        "score_cos": np.empty(len(queries), np.float64),
        "top1_geo_dist": np.empty(len(queries), np.float64),
        "min_past_gt_dist": np.empty(len(queries), np.float64),
        "n_candidates": queries - 29,
    }
    sizes, fallback, margins = [], 0, []
    for row, query in enumerate(queries):
        end = int(query - 29)
        cpu.add(vectors[cpu.ntotal:end])
        gpu.add(vectors[gpu.ntotal:end])
        assert cpu.ntotal == gpu.ntotal == end
        k = min(32, end)
        while True:
            distances, candidates = gpu.search(vectors[query:query + 1], k)
            gap = float(distances[0, -1] - distances[0, 0])
            if k == end or gap > margin:
                selected = np.sort(candidates[0])
                assert (selected >= 0).all() and (selected < end).all()
                exact = faiss.IndexFlatL2(dim)
                exact.add(vectors[selected])
                exact_distance, local = exact.search(vectors[query:query + 1], 1)
                winner = int(selected[local[0, 0]])
                if k < end:
                    margins.append(gap - margin)
                break
            if k >= 2048:
                exact_distance, local = cpu.search(vectors[query:query + 1], 1)
                winner = int(local[0, 0])
                fallback += 1
                k = end
                break
            k = min(2 * k, 2048, end)
        result["j_star"][row] = winner
        result["score_negl2"][row] = -float(np.sqrt(max(float(exact_distance[0, 0]), 0.)))
        result["score_cos"][row] = float(vectors[winner] @ vectors[query])
        result["top1_geo_dist"][row] = float(np.linalg.norm(positions[winner] - positions[query]))
        result["min_past_gt_dist"][row] = np.linalg.norm(positions[:end] - positions[query], axis=1).min()
        sizes.append(k)
        if row and row % 4000 == 0:
            print("PREFIX", row, len(queries), flush=True)
    audit = {
        "roundoff_margin_squared_l2": margin,
        "max_norm_squared": float(norm_sq.max()),
        "shortlist_mean": float(np.mean(sizes)), "shortlist_max": max(sizes),
        "full_cpu_fallback_queries": fallback,
        "minimum_exclusion_slack": min(margins) if margins else None,
        "all_candidates_past_only": True, "tf32": False, "float16": False,
        "final_distance": "공식 CPU FAISS IndexFlatL2, 후보는 원래 번호 순서",
    }
    return result, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    arrays, checks = ledger_exact(np.load(args.descriptors), np.load(args.positions))
    np.savez_compressed(args.out, **arrays)
    checks["runtime_s"] = time.perf_counter() - started
    args.out.with_suffix(".json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    print("EXACT_PREFIX", checks, flush=True)
