#!/usr/bin/env python3
"""공식 RING++의 비결정적 BEV를 보존하며 동일 실행의 검색을 검증한다."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import platform
import resource
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_name] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import numpy as np
import torch
import _pdv2_table3_benchmark as bench
import _pdv2_table3_ring as original
from _causal_ring_pr import correlate_prefix_fast

OUT = bench.OUT
AUDIT = OUT / "native_trace/ringpp_n1000_audit.json"
PROTOCOL = OUT / "ringpp_v2_protocol.json"


def settings() -> dict:
    """native 경합 진단과 동일 실행 검증으로 바꾼 계측 규약을 동결한다."""
    base = original.settings("ringpp")
    audit = bench.read(AUDIT)
    assert audit["every_increment_matches_360"]
    assert not audit["cuda_api_errors"]
    assert audit["voxelfeat_live_requested_at_end"] == 360 * audit["native_feature_calls"]
    assert base["compiled_extensions"] == audit["extensions"]
    sources = [Path(__file__), Path(original.__file__), AUDIT,
               OUT / "native_trace/ringpp_n1000_trace.jsonl",
               bench.REPO / "scripts/_pdv2_table3_native_trace.cpp"]
    return {
        "schema": "pdv2-table3-ringpp-protocol-v2", "revision": 2,
        "parent_protocol_sha256": bench.sha(OUT / "protocol.json"),
        "base_protocol": base,
        "sources": {str(p.relative_to(bench.REPO)): bench.sha(p) for p in sources},
        "reason": "official CUDA feature-cell maximum uses non-atomic writes; independent forwards differ. Preserve the official encoder and report repeat variation instead of asserting cross-forward equality.",
        "unchanged": "Table2 method/selection, official raw preprocessing and CUDA kernels, batch1 encoding, exact hoisted exhaustive correlation, same initial map and 96 arrivals x 3 repeats",
        "verification": "first12 raw encodes finite/occupancy guards and actual fresh TI-RING vs official score; after all timing/memory snapshots, all96 queries in each repeat are checked against official exhaustive batch_circorr using that repeat's actual query and candidate TI-RING",
        "score_tolerance": 1e-4, "winner_rule": "exact official-vs-hoisted first argmax within each execution",
        "audit_fixture": "actual new TI-RING and full candidate score arrays copied/stored after timed query; excluded from method resident buffers and explicitly counted; official reference verification runs after all measured memory/peak snapshots",
        "native_memory": "separately traced 360 requested CUDA bytes remain per valid feature encoding; use this process's actual counter including warmup; physical CUDA rounding and temporary native peaks remain distinct",
        "native_host_metadata": "official Cython wrappers also retain unreachable C++ manager objects without __dealloc__; manager.hh fields are pointers/scalars with no owned persistent host numerical arrays. These small metadata leaks are excluded from numerical resident buffers and remain visible in process RSS.",
        "repeat_statistics": "winner changes, maximum final confidence spread, TI-RING absolute and relative differences are recorded, not rejected merely for native nondeterminism",
    }


def freeze(config: dict) -> None:
    """공통·기존 RING 규약을 변경하지 않고 별도 개정 파일을 쓴다."""
    if PROTOCOL.exists():
        assert bench.read(PROTOCOL) == config, "RING++ 개정 계측 소스 변경"
    else:
        bench.write(PROTOCOL, config)


def cache(n_map: int) -> None:
    """공식 초기 지도 생성은 기존 어댑터를 그대로 사용한다."""
    original.configure()
    freeze(settings())
    original.cache("ringpp", n_map)


class Runtime(original.Runtime):
    """기존 실행을 유지하고 추적된 native 요청 payload를 메모리에 더한다."""

    def query_with_audit(self, q: int) -> tuple:
        """동일 검색의 후보 점수 배열을 검증 fixture로 반환한다."""
        count = q - 29
        scores = correlate_prefix_fast(self.bank, self.on_gpu, q, count,
                                       original.CHUNK, "cuda")
        winner = int(np.argmax(scores))
        return winner, float(scores[winner] - 1.), count, scores

    def memory(self) -> dict:
        """수치 버퍼와 실제 미해제 native 요청 용량의 합을 반환한다."""
        info = super().memory()
        native = self.feature_calls * 360
        info.update({"torch_cuda_allocated": info["gpu"],
                     "external_requested_gpu_bytes": native,
                     "gpu": info["gpu"] + native,
                     "total": info["cpu"] + info["gpu"] + native,
                     "publish_ready": True,
                     "memory_basis": "CPU unique numerical buffers + torch allocated + traced live native requested capacity",
                     "native_audit_sha256": bench.sha(AUDIT),
                     "limitation": "native component is requested capacity, not physical CUDA allocator rounding; transient native allocation peaks excluded from torch peak. Leaked C++ manager pointer/scalar metadata is excluded from numerical buffers and reflected in RSS; no persistent host numerical arrays are owned by those objects."})
        return info


def warmup(runtime: Runtime, work: bench.Workload) -> dict:
    """새로 만든 실제 특징으로 공식 점수 일치를 확인하고 계측을 예열한다."""
    checks = []
    with torch.no_grad():
        for i, points in enumerate(work.points[:bench.WARMUP]):
            t = work.initial + i
            value = runtime.encode(points, check=True)
            actual = value.cpu().numpy()
            assert np.isfinite(actual).all()
            runtime.append(value)
            candidates = np.unique(np.linspace(0, t - 30, 16, dtype=np.int64))
            reference = torch.from_numpy(np.array(runtime.raw_source[candidates], copy=True)).cuda()
            distances, _ = runtime.core.batch_circorr(value[None].repeat(len(candidates), 1, 1, 1), reference)
            direct = 1. - np.asarray(distances, np.float64).reshape(-1)
            fast = correlate_prefix_fast(runtime.bank, runtime.on_gpu, t, t - 29,
                                          original.CHUNK, "cuda")[candidates]
            error = float(abs(direct - fast).max())
            assert error < 1e-4
            assert int(np.argmax(direct)) == int(np.argmax(fast))
            runtime.query_with_audit(t - 15)
            old = np.asarray(runtime.raw_source[t])
            checks.append({"arrival": t, "direct_score_max_error": error,
                           "fresh_vs_cached_tiring_max_abs": float(abs(actual - old).max()),
                           "fresh_vs_cached_tiring_relative_l2": float(np.linalg.norm(actual - old) / max(np.linalg.norm(old), 1e-12))})
            del value, reference
    runtime.reset()
    return {"status": "passed", "raw_regeneration": len(checks),
            "fresh_feature_direct_score_checks": checks,
            "scope": "CPU occupancy and feature-support guards passed; raw differences vs a separate cached forward are diagnostics of native nondeterminism"}


def verify_all(runtime: Runtime, work: bench.Workload, actual: np.ndarray,
               score_fixture: np.ndarray, rows: list) -> dict:
    """매 반복의 실제 특징으로 모든 질의의 공식 전체 후보 점수를 재계산한다."""
    from _ring_official_pr import official_scores

    checks = []
    for repeat in range(bench.REPEATS):
        reference = np.array(runtime.raw_source[:work.end], copy=True)
        reference[work.initial:] = actual[repeat]
        bank = torch.from_numpy(reference)
        for i in range(bench.SAMPLES):
            row = rows[repeat * bench.SAMPLES + i]
            q, count = row["query"], row["n_candidates"]
            direct = official_scores(runtime.core, bank, q, count)
            saved = score_fixture[repeat, i, :count]
            error = float(abs(direct - saved).max())
            j = int(np.argmax(direct))
            assert error < 1e-4, (repeat, i, error)
            assert j == row["winner"] == int(np.argmax(saved)), (repeat, i, j, row["winner"])
            assert abs(row["confidence"] - (direct[j] - 1.)) < 1e-4
            checks.append({"repeat": repeat, "query": q, "n_candidates": count,
                           "winner": j, "all_candidate_score_max_error": error})
        del bank, reference
        print("RINGPP_OFFICIAL_VERIFY", repeat + 1, bench.REPEATS, flush=True)
    return {"status": "passed", "queries": len(checks), "checks": checks,
            "score_tolerance": 1e-4, "winners_exact_within_each_execution": True,
            "scope": "actual repeat-specific query/candidate TI-RING; all reference work after measured memory snapshots"}


def repeat_variation(actual: np.ndarray, rows: list) -> dict:
    """별도 공식 forward들 사이의 변동을 성능 검증과 분리해 요약한다."""
    per_query = []
    for i in range(bench.SAMPLES):
        same = [rows[k * bench.SAMPLES + i] for k in range(bench.REPEATS)]
        confidence = [r["confidence"] for r in same]
        winners = [r["winner"] for r in same]
        per_query.append({"query": same[0]["query"], "winners": winners,
                          "unique_winners": len(set(winners)),
                          "confidence_spread": max(confidence) - min(confidence)})
    reference = actual[0]
    raw_changes = []
    for repeat in range(1, bench.REPEATS):
        difference = actual[repeat] - reference
        flat_diff = difference.reshape(bench.SAMPLES, -1)
        flat_base = reference.reshape(bench.SAMPLES, -1)
        relative = np.linalg.norm(flat_diff, axis=1) / np.maximum(np.linalg.norm(flat_base, axis=1), 1e-12)
        raw_changes.append({"repeat": repeat, "max_abs_vs_repeat0": float(abs(difference).max()),
                            "max_relative_l2_vs_repeat0": float(relative.max()),
                            "mean_relative_l2_vs_repeat0": float(relative.mean()),
                            "changed_frames": int(np.count_nonzero(np.any(flat_diff != 0, axis=1)))})
    return {"winner_changed_queries": sum(r["unique_winners"] > 1 for r in per_query),
            "queries": bench.SAMPLES,
            "max_confidence_spread": max(r["confidence_spread"] for r in per_query),
            "raw_feature_changes": raw_changes, "per_query": per_query,
            "interpretation": "official non-atomic native maximum is preserved; this is repeatability of runtime output, not a new Table2 performance evaluation"}


def measure(n_map: int) -> None:
    """동일 실행 검증을 통과한 RING++ 비용과 반복 변동을 함께 게시한다."""
    original.configure()
    config = settings()
    freeze(config)
    target = OUT / "methods" / f"ringpp_n{n_map}.json"
    if target.exists():
        saved = bench.read(target)
        assert saved["status"] == "passed" and saved.get("protocol_revision") == 2
        return
    spec = bench.read(OUT / "protocol.json")
    work = bench.Workload(spec, n_map)
    tick = time.perf_counter()
    runtime = Runtime(config["base_protocol"], work)
    initial_s = time.perf_counter() - tick
    checked = warmup(runtime, work)
    # 검증 fixture는 방법의 resident state가 아니다. 시간 구간 밖에서 복사한다.
    actual = np.empty((bench.REPEATS, bench.SAMPLES, 6, 120, 120), np.float32)
    score_fixture = np.zeros((bench.REPEATS, bench.SAMPLES, n_map + bench.SAMPLES - 1), np.float64)
    rows, memories = [], []
    for repeat in range(bench.REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = bench.rss()
        for i, points in enumerate(work.points):
            t = work.initial + i
            encoded, encode_ms = bench.timed(lambda: runtime.encode(points))
            _, state_ms = bench.timed(lambda: runtime.append(encoded))
            result, query_ms = bench.timed(lambda: runtime.query_with_audit(t - 15))
            rows.append({"repeat": repeat, "arrival": t, "query": t - 15,
                         "n_candidates": t - 44, "encode": encode_ms,
                         "state_update": state_ms, "update": encode_ms + state_ms,
                         "query_ms": query_ms, "processing": encode_ms + state_ms + query_ms,
                         "winner": result[0], "confidence": result[1], "rerank_candidates": result[2]})
            actual[repeat, i] = encoded.cpu().numpy()
            score_fixture[repeat, i, :result[2]] = result[3]
        info = runtime.memory()
        info.update({"process_rss_before": before, "process_rss_after": bench.rss(),
                     "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     "cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved(),
                     "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
                     "cuda_peak_reserved": torch.cuda.max_memory_reserved()})
        memories.append(info)
        print("RINGPP_MEASURE", repeat + 1, bench.REPEATS, flush=True)
    del encoded
    artifact = OUT / "ringpp_v2_audit" / f"n{n_map}"
    artifact.mkdir(parents=True, exist_ok=True)
    np.save(artifact / "actual_tiring.npy", actual)
    np.save(artifact / "candidate_scores.npy", score_fixture)
    bench.write(artifact / "timed_rows.json", {"rows": rows, "memory_repeats": memories})
    checked["all_queries_same_execution"] = verify_all(runtime, work, actual, score_fixture, rows)
    variation = repeat_variation(actual, rows)
    checked["official_native_repeat_variation"] = variation
    checked["native_requested_bytes_per_feature_call"] = 360
    checked["native_audit_sha256"] = bench.sha(AUDIT)
    bench.write(OUT / "verification" / f"ringpp_n{n_map}.json", checked)
    files = {str((artifact / name).relative_to(bench.REPO)): bench.sha(artifact / name)
             for name in ("actual_tiring.npy", "candidate_scores.npy", "timed_rows.json")}
    result = {
        "schema": "pdv2-table3-measurement-v1", "method": "ringpp", "n_map": n_map,
        "status": "passed", "publish_ready": True, "memory_publish_ready": True,
        "protocol_revision": 2, "protocol_sha256": bench.sha(OUT / "protocol.json"),
        "method_protocol_sha256": bench.sha(PROTOCOL), "source_sha256": bench.sha(Path(__file__)),
        "sequence": bench.SEQUENCE, "seed": None,
        "candidate_range": [n_map, n_map + bench.SAMPLES - 1],
        "mean_candidates": n_map + (bench.SAMPLES - 1) / 2,
        "samples": bench.SAMPLES, "repeats": bench.REPEATS, "warmup": bench.WARMUP,
        "timings_ms": {key: bench.stats([r[field] for r in rows]) for key, field in (
            ("encode", "encode"), ("state_update", "state_update"), ("update", "update"),
            ("query", "query_ms"), ("processing", "processing"))},
        "resident_bytes": memories[-1], "memory_repeats": memories, "verification": checked,
        "raw_input_fixture_bytes": work.input_bytes, "raw_input_sha256": work.input_hash,
        "reference_fixture_bytes": int(runtime.source.nbytes + runtime.raw_source.nbytes),
        "post_timing_audit_fixture_bytes": int(actual.nbytes + score_fixture.nbytes),
        "audit_artifacts": files, "official_native_repeat_variation": variation,
        "initial_cached_state_build_s": initial_s,
        "reference_batch1_cache": bench.read(original.directory("ringpp", n_map) / "complete.json"),
        "raw_samples": rows,
        "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "platform": platform.platform(), "threads": 1},
        "scope": spec["scope"] + " Official RING++ native non-atomic feature max preserved; repeated outputs may differ; score correctness checked within each actual execution. Memory includes measured-call-count native requested live capacity, not physical CUDA rounding.",
        "completed_unix": time.time(),
    }
    bench.write(target, result)
    bench.summarize()
    print("COMPLETE", "ringpp", n_map, result["timings_ms"], variation["winner_changed_queries"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cache", "measure"))
    parser.add_argument("--n-map", type=int, choices=(1000, 5000, 10000), default=1000)
    args = parser.parse_args()
    globals()[args.action](args.n_map)
