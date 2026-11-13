#!/usr/bin/env python3
"""공통 과거 접두만 묶어 조회하고 공식 CPU L2로 최종 후보를 정한다."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import faiss
import numpy as np


def ledger_exact(vectors: np.ndarray, positions: np.ndarray) -> tuple:
    """각 질의의 과거 후보만 읽는 묶음 제안과 정확 CPU 재계산을 한다."""
    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"
    faiss.omp_set_num_threads(1)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    count, dim = vectors.shape
    assert dim == 8192 and len(positions) == count
    norm_sq = np.einsum("ij,ij->i", vectors, vectors, dtype=np.float64)
    assert np.isfinite(vectors).all() and norm_sq.max() < 1.001
    unit = float(np.finfo(np.float32).eps) / 2
    gamma = (dim + 8) * unit / (1 - (dim + 8) * unit)
    margin = 32 * gamma * float(norm_sq.max()) + 1e-10
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(256 * 1024 * 1024)
    config = faiss.GpuIndexFlatConfig()
    config.device, config.useFloat16 = 0, False
    gpu = faiss.GpuIndexFlatL2(resources, dim, config)
    cpu = faiss.IndexFlatL2(dim)
    queries = np.arange(30, count, dtype=np.int64)
    arrays = {"query_idx": queries, "n_candidates": queries - 29}
    for key in ("score_negl2", "score_cos", "top1_geo_dist", "min_past_gt_dist"):
        arrays[key] = np.empty(len(queries), np.float64)
    arrays["j_star"] = np.empty(len(queries), np.int64)
    sizes, slacks, fallback = [], [], 0
    for start in range(0, len(queries), 128):
        block = queries[start:start + 128]
        # 묶음 첫 질의의 과거 접두: 묶음의 모든 질의에도 허용되는 과거다.
        common_end = int(block[0] - 29)
        gpu.add(vectors[gpu.ntotal:common_end])
        cpu.add(vectors[cpu.ntotal:common_end])
        proposals, pending = {}, np.arange(len(block))
        k = min(32, common_end)
        while len(pending):
            distance, candidates = gpu.search(vectors[block[pending]], k)
            remaining = []
            for slot, row in enumerate(pending):
                gap = float(distance[slot, -1] - distance[slot, 0])
                if k == common_end or gap > margin:
                    proposals[int(row)] = candidates[slot].copy()
                    if k < common_end:
                        slacks.append(gap - margin)
                elif k >= 2048:
                    proposals[int(row)] = None
                else:
                    remaining.append(row)
            pending = np.asarray(remaining, dtype=np.int64)
            k = min(2 * k, 2048, common_end)
        for row, q in enumerate(block):
            end = int(q - 29)
            recent = np.arange(common_end, end, dtype=np.int64)
            proposal = proposals[row]
            if proposal is None:
                # 드문 불확실한 경우 전체 허용 접두로 복귀한다.
                selected = np.arange(end, dtype=np.int64)
                fallback += 1
            else:
                selected = np.sort(np.concatenate((proposal, recent)))
            assert len(selected) and selected.min() >= 0 and selected.max() < end
            index = faiss.IndexFlatL2(dim)
            index.add(vectors[selected])
            distance, winner = index.search(vectors[q:q + 1], 1)
            j = int(selected[winner[0, 0]])
            offset = start + row
            arrays["j_star"][offset] = j
            arrays["score_negl2"][offset] = -float(np.sqrt(max(float(distance[0, 0]), 0.0)))
            arrays["score_cos"][offset] = float(vectors[j] @ vectors[q])
            arrays["top1_geo_dist"][offset] = float(np.linalg.norm(positions[j] - positions[q]))
            arrays["min_past_gt_dist"][offset] = np.linalg.norm(positions[:end] - positions[q], axis=1).min()
            sizes.append(len(selected))
        if start and start % 4096 == 0:
            print("PREFIX_BATCH", start, len(queries), flush=True)
    return arrays, dict(roundoff_margin_squared_l2=margin, shortlist_mean=float(np.mean(sizes)),
        shortlist_max=max(sizes), full_cpu_fallback_queries=fallback,
        minimum_exclusion_slack=min(slacks) if slacks else None, all_candidates_past_only=True,
        batch=128, tf32=False, float16=False, final_distance="공식 CPU FAISS IndexFlatL2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    arrays, audit = ledger_exact(np.load(args.descriptors), np.load(args.positions))
    np.savez_compressed(args.out, **arrays)
    audit["runtime_s"] = time.perf_counter() - start
    args.out.with_suffix(".json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print("EXACT_PREFIX_BATCH", audit, flush=True)
