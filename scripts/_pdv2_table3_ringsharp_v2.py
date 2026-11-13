#!/usr/bin/env python3
"""공식 RING#-L의 각도 CNN만16개씩 처리하며 전체 온라인 비용을 계측한다."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import time

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import numpy as np
import torch
import _pdv2_table3_benchmark as bench
import _pdv2_table3_ringsharp as original
import _pdv2_table3_ringsharp_bounded as bounded
from _pdv2_table3_model_memory import audit_numeric_state

OUT = bench.OUT
PROTOCOL = OUT / "ringsharp_v2_protocol.json"
DIAGNOSTIC = OUT / "ringsharp_workspace_diagnostic/n5000/summary.json"
MODEL_AUDIT = OUT / "model_memory/ringsharp.json"
ANGLE_BATCH = 16


def configure() -> None:
    """기존 CPU1·TF32 비활성 설정을 유지한다."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def settings() -> dict:
    """선택 가중치·캐시·수식과16각도 작업 크기를 고정한다."""
    diagnostic = bench.read(DIAGNOSTIC)
    assert diagnostic["status"] == "passed"
    case = next(c for c in diagnostic["cases"] if c["angle_batch"] == ANGLE_BATCH)
    assert case["all_winners_equal"] and case["max_score_error"] < 1e-5
    sources = [Path(__file__), Path(original.__file__), Path(bounded.__file__),
               Path(original.official.__file__), bench.REPO / "scripts/_ringsharp_eval.py",
               bench.REPO / "scripts/_pdv2_table3_model_memory.py", DIAGNOSTIC, MODEL_AUDIT]
    return {"schema": "pdv2-table3-ringsharp-protocol-v2", "revision": 2,
            "base_cache_config": original.settings(), "angle_cnn_batch": ANGLE_BATCH,
            "parent_protocol_sha256": bench.sha(OUT / "protocol.json"),
            "sources": {str(p.relative_to(bench.REPO)): bench.sha(p) for p in sources},
            "unchanged": "selected model/weight; raw160x160x20 BEV and128-channel query features; exhaustive rotation at128pairs; top1000 translation; unique angles; rotate then trans_cnn; candidate FFT errors at128pairs; final first argmin; batch1 arrivals, CPU1, TF32off",
            "change": "Only the unique-angle rotate→trans_cnn loop uses16angles instead of128. Same value inputs and candidate FFT operations; no parameter, model, candidate, precision or CUDA allocator policy changes.",
            "verification": "first12 actual batch1 encodes match official cache; warmup first/last query compares all1000errors against official per-candidate solve_translation; after all timed memory snapshots all96queries x3 compare actual query/candidate features against original128-angle path, all1000errors, exact winner and FP32 margin",
            "tolerance": 1e-5,
            "fixture": "actual new spec/bt/packed and each query top/angles/errors stored after timed calls, excluded from method buffers; original reference verification after measurement",
            "size_policy": "same16-angle bound for N1000,5000,10000"}


def freeze(config: dict) -> None:
    """기존 코드·규약을 유지하고 개정2 원천만 동결한다."""
    if PROTOCOL.exists():
        assert bench.read(PROTOCOL) == config
    else:
        bench.write(PROTOCOL, config)


def query_details(runtime, q: int, official_path: bool = False,
                  official_loop: bool = False) -> dict:
    """같은 회전 순위·후보1000개에서 번역 오차와 최종 승자를 반환한다."""
    candidates = np.arange(q - 29)
    distance, angles = original.official.rotation_branch(runtime.base, runtime.spec,
                                                        runtime.spec[q:q + 1], candidates)
    top = np.argsort(distance)[:original.official._depth(len(candidates))]
    feature = original.official.query_features(runtime.base, runtime.model,
                                               runtime.packed[q:q + 1], "cuda")
    if official_path:
        error = original.official.translation_branch(runtime.base, runtime.trans, runtime.bt,
                                                     feature, top, angles[top], official_loop,
                                                     dedup=not official_loop)
    else:
        error = bounded.translation_branch(runtime.base, runtime.trans, runtime.bt,
                                            feature, top, angles[top], ANGLE_BATCH)
    best = int(np.argmin(error))
    return {"winner": int(top[best]), "confidence": -float(error[best]),
            "top": top, "angles": angles[top], "errors": error}


def compare(actual: dict, reference: dict) -> dict:
    """후보 전체 점수·최종 승자·승자 여유를 확인한다."""
    np.testing.assert_array_equal(actual["top"], reference["top"])
    np.testing.assert_array_equal(actual["angles"], reference["angles"])
    difference = float(abs(actual["errors"] - reference["errors"]).max())
    errors = reference["errors"]
    margin = float(np.partition(errors, 1)[1] - errors.min())
    assert difference < 1e-5 and actual["winner"] == reference["winner"], (difference, margin)
    return {"all_candidate_score_max_error": difference, "reference_margin": margin,
            "error_over_margin": difference / max(margin, 1e-30),
            "winner": actual["winner"], "candidates": len(errors)}


class Runtime(original.Runtime):
    """공식 저장 상태를 유지하며 미등록 CPU 수치 상태도 함께 센다."""

    def __init__(self, work) -> None:
        super().__init__(work)
        self.numeric = audit_numeric_state(self.model, self.trans)
        expected = bench.read(MODEL_AUDIT)["audit"]
        assert self.numeric["status"] == "passed"
        assert self.numeric["cpu_unique_bytes"] == expected["cpu_unique_bytes"]

    def memory(self) -> dict:
        info = super().memory()
        delta = self.numeric["extra_cpu_bytes"]
        info["cpu"] += delta
        info["total"] += delta
        info["breakdown"]["model_extra_cpu_numeric"] = delta
        info["model_numeric_audit"] = {"cpu_unique_bytes": self.numeric["cpu_unique_bytes"],
                                       "registered_cpu_bytes": self.numeric["registered_cpu_bytes"],
                                       "extra_cpu_bytes": delta,
                                       "source_path": str(MODEL_AUDIT.relative_to(bench.REPO))}
        return info


def warmup(runtime: Runtime, work) -> dict:
    """실제 첫12스캔 특징과 첫·마지막의 공식1000후보 오차를 확인한다."""
    checks = []
    for i in range(bench.WARMUP):
        t = work.initial + i
        encoded = runtime.encode(work.points[i])
        for key, value in zip(("spec", "bt", "packed"), encoded):
            actual = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
            np.testing.assert_allclose(actual, runtime.sources[key][t], atol=1e-6, rtol=1e-6)
        runtime.append(encoded)
        actual = query_details(runtime, t - 15)
        if i in (0, bench.WARMUP - 1):
            reference = query_details(runtime, t - 15, True, True)
            checks.append({"query": t - 15, **compare(actual, reference)})
    return {"status": "passed", "raw_batch1_feature_checks": bench.WARMUP,
            "official_loop_all_candidates": checks,
            "checkpoint_sha256": runtime.config["weight_sha256"]}


def verify_all(runtime: Runtime, work, actual: dict, outputs: list) -> dict:
    """실측의 실제 새 특징을 복원해 원본128각도 경로와288질의를 대조한다."""
    checks = []
    for repeat in range(bench.REPEATS):
        runtime.reset()
        torch.cuda.empty_cache()
        for i in range(bench.SAMPLES):
            encoded = (torch.from_numpy(actual["spec"][repeat, i]).cuda(),
                       torch.from_numpy(actual["bt"][repeat, i]).cuda(),
                       actual["packed"][repeat, i])
            runtime.append(encoded)
            q = work.initial + i - 15
            reference = query_details(runtime, q, True)
            checked = compare(outputs[repeat * bench.SAMPLES + i], reference)
            checks.append({"repeat": repeat, "query": q, **checked})
            # 참조는 계측 이후이며 큰 가변 임시 버퍼를 다음 질의에 남기지 않는다.
            del encoded, reference
            torch.cuda.empty_cache()
        print("RINGSHARP_V2_VERIFY", repeat + 1, bench.REPEATS, flush=True)
    return {"status": "passed", "queries": len(checks), "checks": checks,
            "scope": "actual repeat-specific spec/bt/occupancy; complete original top1000 translation errors; reference work after all timed memory snapshots"}


def atomic_result(target: Path, value: dict) -> None:
    """통과한 개정 결과만 게시하고 기존 측정은 먼저 보존한다."""
    if target.exists():
        previous = OUT / "before_ringsharp_workspace_audit" / target.name
        previous.parent.mkdir(parents=True, exist_ok=True)
        if previous.exists():
            assert previous.read_bytes() == target.read_bytes()
        else:
            previous.write_bytes(target.read_bytes())
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, target)


def measure(n_map: int) -> None:
    """고정16각도로96실제도착×3반복을 계측하고 전후보 검증 후 게시한다."""
    configure()
    config = settings()
    freeze(config)
    target = OUT / "methods" / f"ringsharp_n{n_map}.json"
    if target.exists() and bench.read(target).get("protocol_revision") == 2:
        saved = bench.read(target)
        assert saved["status"] == "passed" and saved["method_protocol_sha256"] == bench.sha(PROTOCOL)
        return
    work = bench.Workload(bench.read(OUT / "protocol.json"), n_map)
    runtime = Runtime(work)
    checked = warmup(runtime, work)
    actual = {key: np.empty((bench.REPEATS, bench.SAMPLES, *source.shape[1:]), source.dtype)
              for key, source in runtime.sources.items()}
    rows, memories, outputs = [], [], []
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
            answer, query_ms = bench.timed(lambda: query_details(runtime, t - 15))
            rows.append({"repeat": repeat, "arrival": t, "query": t - 15,
                         "n_candidates": t - 44, "encode": encode_ms, "state_update": state_ms,
                         "update": encode_ms + state_ms, "query_ms": query_ms,
                         "processing": encode_ms + state_ms + query_ms,
                         "winner": answer["winner"], "confidence": answer["confidence"],
                         "rerank_candidates": len(answer["top"])})
            outputs.append(answer)
            for key, value in zip(("spec", "bt", "packed"), encoded):
                actual[key][repeat, i] = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
        info = runtime.memory()
        info.update({"process_rss_before": before, "process_rss_after": bench.rss(),
                     "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     "cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved(),
                     "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
                     "cuda_peak_reserved": torch.cuda.max_memory_reserved(),
                     "cuda_device_total_bytes": torch.cuda.get_device_properties(0).total_memory})
        assert info["cuda_peak_reserved"] < info["cuda_device_total_bytes"], "예약량이 물리 GPU 용량을 넘음"
        memories.append(info)
        print("RINGSHARP_V2_MEASURE", n_map, repeat + 1, bench.REPEATS, flush=True)
    del encoded
    for i in range(bench.SAMPLES):
        values = [rows[k * bench.SAMPLES + i] for k in range(bench.REPEATS)]
        assert len({v["winner"] for v in values}) == 1
        assert np.ptp([v["confidence"] for v in values]) < 1e-6
    artifact = OUT / "ringsharp_v2_audit" / f"n{n_map}"
    artifact.mkdir(parents=True, exist_ok=True)
    np.savez(artifact / "actual_features.npz", **actual)
    np.savez(artifact / "query_outputs.npz", **{key: np.stack([v[key] for v in outputs])
                                             for key in ("top", "angles", "errors")})
    bench.write(artifact / "timed_rows.json", {"rows": rows, "memory_repeats": memories})
    checked["all_queries_same_execution"] = verify_all(runtime, work, actual, outputs)
    bench.write(artifact / "verification.json", checked)
    result = {"schema": "pdv2-table3-measurement-v1", "method": "ringsharp", "n_map": n_map,
              "status": "passed", "publish_ready": True, "memory_publish_ready": True, "protocol_revision": 2,
              "protocol_sha256": bench.sha(OUT / "protocol.json"), "method_protocol_sha256": bench.sha(PROTOCOL),
              "source_sha256": bench.sha(Path(__file__)), "method_config": config,
              "sequence": bench.SEQUENCE, "seed": 1, "candidate_range": [n_map, n_map + bench.SAMPLES - 1],
              "mean_candidates": n_map + (bench.SAMPLES - 1) / 2, "samples": bench.SAMPLES,
              "repeats": bench.REPEATS, "warmup": bench.WARMUP,
              "timings_ms": {key: bench.stats([r[field] for r in rows]) for key, field in (
                  ("encode", "encode"), ("state_update", "state_update"), ("update", "update"),
                  ("query", "query_ms"), ("processing", "processing"))},
              "resident_bytes": memories[-1], "memory_repeats": memories,
              "verification": checked, "raw_samples": rows,
              "audit_artifacts": {str((artifact / name).relative_to(bench.REPO)): bench.sha(artifact / name)
                                  for name in ("actual_features.npz", "query_outputs.npz", "timed_rows.json", "verification.json")},
              "raw_input_fixture_bytes": work.input_bytes, "raw_input_sha256": work.input_hash,
              "post_timing_audit_fixture_bytes": sum(x.nbytes for x in actual.values()) +
                  sum(v[key].nbytes for v in outputs for key in ("top", "angles", "errors")),
              "model_numeric_memory_correction": {"audit_path": str(MODEL_AUDIT.relative_to(bench.REPO)),
                  "audit_sha256": bench.sha(MODEL_AUDIT), "cpu_delta_bytes": runtime.numeric["extra_cpu_bytes"],
                  "registered_cpu_bytes": runtime.numeric["registered_cpu_bytes"],
                  "full_cpu_model_bytes": runtime.numeric["cpu_unique_bytes"],
                  "scope": "Fresh model audited at initialization; verified against prior constant-state audit and included in each original measurement snapshot, not added later."},
              "completed_unix": time.time(),
              "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "threads": 1, "tf32": False},
              "scope": "Official batch1 BEV and chosen checkpoint; all-prefix rotation and top1000 translation unchanged; only unique-angle rotate→CNN uses16angles, candidate FFT batch128 unchanged. Actual live features verified for all288queries after timing; model CPU numeric state audited."}
    atomic_result(target, result)
    bench.summarize()
    print("RINGSHARP_V2_COMPLETE", n_map, result["timings_ms"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-map", type=int, choices=(1000, 5000, 10000), default=1000)
    args = parser.parse_args()
    measure(args.n_map)
