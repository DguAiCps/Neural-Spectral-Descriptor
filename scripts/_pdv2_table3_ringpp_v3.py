#!/usr/bin/env python3
"""공식 RING++의 동일 수식 검색을 128후보 청크로 제한해 전체 비용을 잰다."""
from __future__ import annotations

import argparse
import gc
import json
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
import _pdv2_table3_ringpp_v2 as previous
from _causal_ring_pr import correlate_prefix_fast

OUT = bench.OUT
PROTOCOL = OUT / "ringpp_v3_protocol.json"
CHUNK = 128
DIAGNOSTIC = OUT / "ring_query_workspace_diagnostic/n1000"
EVIDENCE = ("chunk128_growing_pad0_cached", "chunk256_growing_pad0_cached",
            "chunk512_growing_pad0_cached", "chunk1024_growing_pad0_cached",
            "chunk1024_fixed_pad0_cached", "chunk1024_growing_pad1_cached")


def settings() -> dict:
    """공식 특징·수식·검증을 보존한 청크 제한과 진단 근거를 동결한다."""
    base = original.settings("ringpp")
    audit = bench.read(previous.AUDIT)
    assert audit["every_increment_matches_360"] and not audit["cuda_api_errors"]
    assert audit["voxelfeat_live_requested_at_end"] == 360 * audit["native_feature_calls"]
    assert base["compiled_extensions"] == audit["extensions"]
    paths = [Path(__file__), Path(previous.__file__), Path(original.__file__),
             bench.REPO / "scripts/_pdv2_table3_ring_query_diag.py", previous.AUDIT,
             OUT / "native_trace/ringpp_n1000_trace.jsonl",
             bench.REPO / "scripts/_pdv2_table3_native_trace.cpp"]
    for name in EVIDENCE:
        path = DIAGNOSTIC / f"{name}.json"
        evidence = bench.read(path)
        assert evidence["status"] == "passed" and evidence["all_official_winners_equal"]
        paths.append(path)
    return {
        "schema": "pdv2-table3-ringpp-protocol-v3", "revision": 3,
        "parent_protocol_sha256": bench.sha(OUT / "protocol.json"),
        "base_cache_protocol": base,
        "sources": {str(p.relative_to(bench.REPO)): bench.sha(p) for p in paths},
        "pairs_per_chunk": CHUNK, "tail_padding": False,
        "reason": "A growing large tail under chunk1024 retained 19.705 GB in the PyTorch allocator cache although active tensors were much smaller. Fixed-prefix and padded-tail controls remove the pressure. Chunk128 preserves official scores/winners with bounded temporary allocations.",
        "unchanged": "official raw preprocessing and CUDA features; same fixed initial TI-RING cache; InstanceNorm/FFT values; official exhaustive circular-correlation math; first argmax; Table2 selection; batch1, CPU1, TF32off; 96 arrivals x 3 repeats",
        "override_scope": "base_cache_protocol describes the immutable initial cache; only retrieval chunk is overridden to128 for warmup and measured queries at every map size",
        "verification": "first12 fresh official features and actual direct score checks; after all timing/memory snapshots, all96 queries x3 are verified against official batch16 exhaustive correlation using each repeat's actual query/candidate features",
        "score_tolerance": 1e-4, "winner_rule": "exact first argmax within the same execution",
        "audit_fixture": "actual newly encoded TI-RING and candidate scores copied after timed query; fixtures excluded from method resident buffers; official verification after all timing/memory snapshots",
        "native_memory": "same compiled CUDA feature code retains360 requested bytes per valid encode; actual process count includes warmup; physical allocation rounding and temporary native peaks remain distinct",
        "process_lifetime": "fresh measurement process initializes map from cache and performs warmup12+measured288 encodes; prior cache-process native allocations are not inherited; uninterrupted online native leakage continues with valid calls",
        "native_host_metadata": "Cython wrappers retain unreachable C++ manager pointer/scalar objects without persistent host numerical arrays; metadata excluded from numerical resident state, reflected in RSS",
        "repeat_variation": "official native non-atomic maximum remains nondeterministic; report actual feature/confidence/winner variation separately from same-execution score equivalence",
    }


def freeze(config: dict) -> None:
    """기존 규약을 변경하지 않고 개정 3 원천만 저장한다."""
    if PROTOCOL.exists():
        assert bench.read(PROTOCOL) == config, "RING++ v3 동결 원천 변경"
    else:
        bench.write(PROTOCOL, config)


class Runtime(previous.Runtime):
    """공식 FFT 뱅크를 사용하며 검색 작업 묶음만128후보로 제한한다."""

    def query_with_audit(self, q: int) -> tuple:
        """원본 전수 상관의 모든 점수와 첫 승자를 반환한다."""
        count = q - 29
        scores = correlate_prefix_fast(self.bank, self.on_gpu, q, count, CHUNK, "cuda")
        winner = int(np.argmax(scores))
        return winner, float(scores[winner] - 1.), count, scores


def warmup(runtime: Runtime, work: bench.Workload) -> dict:
    """실제 새 특징으로 공식 점수와128청크 경로를 대조한다."""
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
            fast = correlate_prefix_fast(runtime.bank, runtime.on_gpu, t, t - 29, CHUNK, "cuda")[candidates]
            error = float(abs(direct - fast).max())
            assert error < 1e-4 and int(np.argmax(direct)) == int(np.argmax(fast))
            runtime.query_with_audit(t - 15)
            old = np.asarray(runtime.raw_source[t])
            checks.append({"arrival": t, "direct_score_max_error": error,
                           "fresh_vs_cached_tiring_max_abs": float(abs(actual - old).max()),
                           "fresh_vs_cached_tiring_relative_l2": float(np.linalg.norm(actual - old) / max(np.linalg.norm(old), 1e-12))})
            del value, reference
    runtime.reset()
    return {"status": "passed", "raw_regeneration": len(checks),
            "fresh_feature_direct_score_checks": checks,
            "scope": "official occupancy/feature-support guards; independent-forward differences retained as native nondeterminism diagnostics"}


def cache(n_map: int) -> None:
    """기존 공식 배치1 초기 캐시를 그대로 생성하거나 확인한다."""
    original.configure()
    freeze(settings())
    original.cache("ringpp", n_map)


def atomic_result(path: Path, result: dict) -> None:
    """모든 검증이 끝난 결과만 한 번에 공개한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def measure(n_map: int) -> None:
    """실제 새 점군 인코딩부터 최종 검색까지 개정3 비용을 측정한다."""
    original.configure()
    config = settings()
    freeze(config)
    target = OUT / "methods" / f"ringpp_n{n_map}.json"
    if target.exists():
        saved = bench.read(target)
        assert saved["status"] == "passed" and saved.get("protocol_revision") == 3
        assert saved["method_protocol_sha256"] == bench.sha(PROTOCOL)
        return
    spec = bench.read(OUT / "protocol.json")
    work = bench.Workload(spec, n_map)
    started = time.perf_counter()
    runtime = Runtime(config["base_cache_protocol"], work)
    initial_s = time.perf_counter() - started
    checked = warmup(runtime, work)
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
                     "cuda_peak_reserved": torch.cuda.max_memory_reserved(),
                     "cuda_device_total_bytes": torch.cuda.get_device_properties(0).total_memory})
        assert info["cuda_peak_reserved"] < info["cuda_device_total_bytes"], "예약 메모리가 물리 장치 용량을 넘음"
        memories.append(info)
        print("RINGPP_V3_MEASURE", repeat + 1, bench.REPEATS, flush=True)
    del encoded
    artifact = OUT / "ringpp_v3_audit" / f"n{n_map}"
    artifact.mkdir(parents=True, exist_ok=True)
    np.save(artifact / "actual_tiring.npy", actual)
    np.save(artifact / "candidate_scores.npy", score_fixture)
    bench.write(artifact / "timed_rows.json", {"rows": rows, "memory_repeats": memories})
    checked["all_queries_same_execution"] = previous.verify_all(runtime, work, actual, score_fixture, rows)
    variation = previous.repeat_variation(actual, rows)
    checked["official_native_repeat_variation"] = variation
    checked["native_requested_bytes_per_feature_call"] = 360
    checked["native_audit_sha256"] = bench.sha(previous.AUDIT)
    bench.write(OUT / "verification" / f"ringpp_n{n_map}.json", checked)
    files = {str((artifact / name).relative_to(bench.REPO)): bench.sha(artifact / name)
             for name in ("actual_tiring.npy", "candidate_scores.npy", "timed_rows.json")}
    result = {
        "schema": "pdv2-table3-measurement-v1", "method": "ringpp", "n_map": n_map,
        "status": "passed", "publish_ready": True, "memory_publish_ready": True,
        "protocol_revision": 3, "protocol_sha256": bench.sha(OUT / "protocol.json"),
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
        "scope": spec["scope"] + " RING++ official math with128-pair bounded chunks; native nondeterminism preserved; all queries checked within each actual execution. Native memory is requested live capacity after300 encodes in a fresh cached-map process, not physical CUDA rounding or a long-run bound.",
        "completed_unix": time.time(),
    }
    atomic_result(target, result)
    bench.summarize()
    print("COMPLETE_RINGPP_V3", n_map, result["timings_ms"], variation["winner_changed_queries"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cache", "measure"))
    parser.add_argument("--n-map", type=int, choices=(1000, 5000, 10000), default=1000)
    args = parser.parse_args()
    globals()[args.action](args.n_map)
