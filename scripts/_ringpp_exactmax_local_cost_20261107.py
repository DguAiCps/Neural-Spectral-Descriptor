#!/usr/bin/env python3
"""정확max RING++의 표2 비용을 로컬 RTX5080에서 재시작 가능하게 잰다.

시간 측정 경로는 `_pdv2_table3_ringpp_v3.py`(공식 특징·라돈·상관, 청크128)와 같고
격자 max만 `_review_native_ring_maxfix.install_maxfix`(np.fmax.at)로 바꾼다.
이 파일의 추가 변경은 시간 구간 밖의 저장 위치·메모리 위생·검증 방식뿐이다.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import mmap
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
import _pdv2_table3_ringpp_v3 as measured
import _review_native_ring_maxfix as fix
from _causal_ring_pr import correlate_prefix_fast

REPO = bench.REPO
WORK = REPO / "results/pdv2_baseline_revision_20261107/ringpp"
OUT = WORK / "cost"
# 대형 파일은 D:(컨테이너 /workspace/data)에 둔다. C:의 WSL 디스크 증가를 피한다.
LARGE = Path(os.environ.get("RINGPP_LARGE_ROOT", "/workspace/data/pdv2_ringpp_exactmax_cost_20261107"))
CACHE = LARGE / "ringpp_batch1/n10000"
AUDIT = LARGE / "audit/n10000"
N_MAP = 10000
BLOCK = 128
MIN_AVAILABLE = 8 << 30
EXIT_LOW_MEMORY = 75
TRACER = REPO / "results/pdv2_table3_20261029/native_trace/libnsd_cuda_trace.so"
OLD_LOCAL = REPO / "results/pdv2_table3_20261029/methods/ringpp_n10000.json"
OLD_EXACTMAX_PROTOCOL = REPO / "results/pdv2_seqot_ring_repair_20261106/ringpp_cost/ringpp_exactmax_protocol.json"
SERVER_ROWS = REPO / "results/pdv2_seqot_ring_repair_20261106/ringpp_server_cost_v2/timed_rows.json"


class LowMemory(RuntimeError):
    """host MemAvailable이 기준 아래로 내려가 자신의 계산만 멈춘다."""


def available() -> int:
    """WSL VM 전체의 MemAvailable 바이트."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable 없음")


def guard(where: str) -> None:
    """여유 메모리가 8GiB 아래면 결과를 쓰지 않고 중단한다."""
    value = available()
    if value < MIN_AVAILABLE:
        raise LowMemory(f"{where}: MemAvailable {value} < {MIN_AVAILABLE}")


def release(array: np.memmap) -> None:
    """읽기 전용 매핑의 상주 페이지를 프로세스에서 반환한다."""
    array._mmap.madvise(mmap.MADV_DONTNEED)


def status(stage: str, **extra) -> None:
    """계산 프로세스 자신의 상태를 남긴다."""
    bench.write(OUT / "compute_status.json", {"stage": stage, "pid": os.getpid(), "updated_unix": time.time(),
                                              "rss_bytes": bench.rss(), "mem_available": available(), **extra})


def configure() -> dict:
    """기존 정확max 비용 규약과 같은 설정을 만들고 캐시 위치만 D:로 둔다."""
    original.configure()
    assert fix.native.rol.import_ring_core() is original.reference.import_ring_core()
    config = measured.settings()
    config["schema"] = "pdv2-table3-ringpp-exactmax-v1"
    config["base_cache_protocol"]["encoder"] += "; grid maximum replaced by verified np.fmax.at"
    # 이 파일의 해시는 각 산출물에 따로 남긴다(검증 코드 수정이 캐시 규약을 바꾸지 않게).
    config["base_cache_protocol"]["sources"].update({
        str(Path(fix.__file__).relative_to(REPO)): bench.sha(Path(fix.__file__)),
    })
    config["reason"] = "확인된 비원자 max 오류를 고친 동일46시험 구현의 실제 비용. 성능에 따른 설정 선택 없음."
    config["unchanged"] = "공통Sejong01·초기1만 후보·96도착×3·CPU1·RTX5080·실제점군 인코딩·같은점특징/라돈/상관/청크128"
    config["verification"] = "새 특징의 CPU max 정확성·모든288질의 공식상관 대조·3반복 특징 및 승자 동일성"
    config["repeat_variation"] = "수정된 격자 집계는 결정적이어야 함. 반복의 특징/승자/점수 변동 검사"
    config["external_memory_basis"] = "GPUFeatureExtractor는 그대로여서 그 경로의360바이트/호출 미해제 요청을 기존 계측대로 합산"
    config["matching_test_evidence"] = "results/pdv2_ringpp_maxfix_20261105/summary_verified.json"
    config["local_run"] = {
        "cache_location": str(CACHE), "audit_location": str(AUDIT),
        "untimed_changes": ["초기 캐시를 128행 단위 순차 기록·이어받기", "reset 뒤 읽기 매핑 페이지 반환",
                            "검증의 원시 TI-RING 참조를 CPU 복사 대신 GPU에 한 벌만 올림(시간·메모리 스냅샷 이후)",
                            "MemAvailable 8GiB 미만이면 결과 없이 중단"],
        "timed_path": "_pdv2_table3_ringpp_v3 측정 루프와 같은 encode/append/query_with_audit",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    common = OUT / "common_protocol.json"
    spec = bench.read(bench.OUT / "protocol.json")
    if common.exists():
        assert bench.read(common) == spec
    else:
        bench.write(common, spec)
    path = OUT / "ringpp_exactmax_protocol.json"
    if path.exists():
        assert bench.read(path) == config, "동결한 정확max 비용 규약 변경"
    else:
        bench.write(path, config)
    original.directory = lambda variant, n_map: (CACHE if (variant, n_map) == ("ringpp", N_MAP) else None)
    fix.install_maxfix()
    return config


def write_header(handle, dtype, shape) -> int:
    """전체 크기의 npy 머리를 쓰고 자료 시작 위치를 돌려준다."""
    np.lib.format.write_array_header_1_0(handle, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
                                                  "fortran_order": False, "shape": shape})
    return handle.tell()


def block_sha(value: np.ndarray) -> str:
    """구간 수치 바이트의 해시."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def cache() -> None:
    """초기 지도 TI-RING·FFT를 128행씩 순차 기록하고 중단 지점에서 잇는다."""
    config = configure()["base_cache_protocol"]
    CACHE.mkdir(parents=True, exist_ok=True)
    done = CACHE / "complete.json"
    if done.exists():
        meta = bench.read(done)
        assert meta["config"] == config
        print("CACHE_ALREADY_COMPLETE", flush=True)
        return
    work = bench.Workload(bench.read(bench.OUT / "protocol.json"), N_MAP)
    old = bench.OUT / "ringpp_batch1/n10000"
    np.testing.assert_array_equal(np.load(old / "scan_ids.npy"), work.ids)
    shape = (work.end, 6, 120, 120)
    files = {"raw": (CACHE / "raw.npy", np.float32), "fft": (CACHE / "fft.npy", np.complex64)}
    cursor = CACHE / "resume_cursor.json"
    state = bench.read(cursor) if cursor.exists() else {"config": config, "completed": 0, "blocks": [], "point_counts": []}
    assert state["config"] == config
    handles, offsets = {}, {}
    for key, (path, dtype) in files.items():
        row = int(np.prod(shape[1:])) * np.dtype(dtype).itemsize
        if state["completed"] == 0:
            handle = path.open("wb")
            offsets[key] = write_header(handle, dtype, shape)
        else:
            handle = path.open("r+b")
            offsets[key] = write_header(handle, dtype, shape)
            handle.truncate(offsets[key] + state["completed"] * row)
            handle.seek(0, os.SEEK_END)
            last = state["blocks"][-1]
            handle.seek(offsets[key] + last["begin"] * row)
            check = np.frombuffer(handle.read((last["end"] - last["begin"]) * row), dtype).reshape(-1, *shape[1:])
            assert block_sha(check) == last[f"{key}_sha256"], ("이어받기 꼬리 불일치", key)
            handle.seek(0, os.SEEK_END)
        handles[key] = (handle, row)
    loader = original.reference.make_loader("mulran", "Sejong01")
    core = original.reference.import_ring_core()
    normalizer = torch.nn.InstanceNorm2d(1, affine=False, track_running_stats=False)
    counts = list(state["point_counts"])
    start, tick = state["completed"], time.time()
    print("CACHE_RESUME_FROM", start, work.end, flush=True)
    with torch.no_grad():
        for lo in range(start, work.end, BLOCK):
            hi = min(lo + BLOCK, work.end)
            raw = np.empty((hi - lo, *shape[1:]), np.float32)
            spectral = np.empty((hi - lo, *shape[1:]), np.complex64)
            block_counts = []
            for i in range(lo, hi):
                guard(f"cache row {i}")
                value, count = original.encode(core, "ringpp", loader[int(work.ids[i])]["points"], check=True)
                raw[i - lo] = value.cpu().numpy()
                spectral[i - lo] = torch.fft.fft2(normalizer(value[None]), dim=-2, norm="ortho")[0].cpu().numpy()
                block_counts.append(int(count))
            for key, value in (("raw", raw), ("fft", spectral)):
                handle, _ = handles[key]
                handle.write(value.tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            counts.extend(block_counts)
            state["blocks"].append({"begin": lo, "end": hi, "raw_sha256": block_sha(raw), "fft_sha256": block_sha(spectral)})
            state["completed"] = hi
            state["point_counts"] = counts
            bench.write(cursor, state)
            status("cache", completed=hi, total=work.end, seconds=round(time.time() - tick, 1))
            print("CACHE_BLOCK", hi, work.end, round(time.time() - tick, 1), "rss_mb", round(bench.rss() / 1e6), flush=True)
            del raw, spectral
    for handle, _ in handles.values():
        handle.close()
    old_counts = np.load(old / "point_counts.npy")
    np.testing.assert_array_equal(np.asarray(counts, np.int64), old_counts)
    np.save(CACHE / "scan_ids.npy", work.ids)
    np.save(CACHE / "point_counts.npy", np.asarray(counts, np.int64))
    # 기록한 파일을 다시 읽어 모든 구간 해시와 FFT 표본을 대조한다.
    raw_map = np.load(CACHE / "raw.npy", mmap_mode="r")
    fft_map = np.load(CACHE / "fft.npy", mmap_mode="r")
    assert raw_map.shape == shape and fft_map.shape == shape
    for block in state["blocks"]:
        assert block_sha(raw_map[block["begin"]:block["end"]]) == block["raw_sha256"]
        assert block_sha(fft_map[block["begin"]:block["end"]]) == block["fft_sha256"]
        release(raw_map)
        release(fft_map)
    fft_error = 0.0
    with torch.no_grad():
        for i in np.unique(np.linspace(0, work.end - 1, 32, dtype=np.int64)):
            value = torch.from_numpy(np.array(raw_map[i])).cuda()
            again = torch.fft.fft2(normalizer(value[None]), dim=-2, norm="ortho")[0].cpu().numpy()
            fft_error = max(fft_error, float(abs(again - fft_map[i]).max()))
    assert fft_error < 3e-5, fft_error
    del raw_map, fft_map
    bench.write(done, {"config": config, "frames": work.end, "batch": 1,
                       "raw_input_sha256": work.input_hash, "seconds": time.time() - tick,
                       "resumed_from_row": start, "cache_source_sha256": bench.sha(Path(__file__)), "blocks_verified_after_write": len(state["blocks"]),
                       "fft_recompute_probe_max_abs": fft_error,
                       "point_counts_equal_oldkernel_cache": True,
                       "native_allocation_audit": original.native_audit(),
                       "files": {name: bench.sha(CACHE / name) for name in (
                           "raw.npy", "fft.npy", "scan_ids.npy", "point_counts.npy")}})
    print("CACHE_COMPLETE", flush=True)


class Runtime(measured.Runtime):
    """v3 실행과 같고 초기화 뒤 읽기 매핑 페이지만 반환한다."""

    def reset(self) -> None:
        """동일한 초기 지도를 복원하고 매핑 페이지를 프로세스에서 뗀다."""
        super().reset()
        release(self.source)


def official_verify(runtime: Runtime, work: bench.Workload, actual: np.ndarray,
                    score_fixture: np.ndarray, rows: list) -> dict:
    """v2 verify_all과 같은 공식 batch_circorr 검산을 GPU 참조 한 벌로 한다."""
    from _ring_official_pr import official_scores

    reference = torch.empty((work.end, 6, 120, 120), dtype=torch.float32, device="cuda")
    for i in range(0, work.initial, 512):
        end = min(i + 512, work.initial)
        reference[i:end].copy_(torch.from_numpy(np.array(runtime.raw_source[i:end], copy=True)))
    release(runtime.raw_source)
    checks = []
    for repeat in range(bench.REPEATS):
        reference[work.initial:].copy_(torch.from_numpy(actual[repeat]))
        for i in range(bench.SAMPLES):
            row = rows[repeat * bench.SAMPLES + i]
            q, count = row["query"], row["n_candidates"]
            direct = official_scores(runtime.core, reference, q, count)
            saved = score_fixture[repeat, i, :count]
            error = float(abs(direct - saved).max())
            j = int(np.argmax(direct))
            assert error < 1e-4, (repeat, i, error)
            assert j == row["winner"] == int(np.argmax(saved)), (repeat, i, j, row["winner"])
            assert abs(row["confidence"] - (direct[j] - 1.)) < 1e-4
            checks.append({"repeat": repeat, "query": q, "n_candidates": count,
                           "winner": j, "all_candidate_score_max_error": error})
        status("verify", repeat=repeat + 1)
        print("RINGPP_OFFICIAL_VERIFY", repeat + 1, bench.REPEATS, flush=True)
    del reference
    return {"status": "passed", "queries": len(checks), "checks": checks,
            "score_tolerance": 1e-4, "winners_exact_within_each_execution": True,
            "reference_on": "GPU single copy (initial prefix from cache raw.npy + each repeat's actual arrivals)",
            "scope": "actual repeat-specific query/candidate TI-RING; all reference work after measured memory snapshots"}


def measure() -> None:
    """실제 새 점군 인코딩부터 최종 검색까지 96도착×3을 잰다."""
    config = configure()
    target = OUT / f"methods/ringpp_n{N_MAP}.json"
    if target.exists():
        assert bench.read(target)["status"] == "passed"
        print("MEASURE_ALREADY_COMPLETE", flush=True)
        return
    guard("measure start")
    spec = bench.read(bench.OUT / "protocol.json")
    work = bench.Workload(spec, N_MAP)
    torch.cuda.init()
    device_free_start, device_total = torch.cuda.mem_get_info()
    started = time.perf_counter()
    runtime = Runtime(config["base_cache_protocol"], work)
    initial_s = time.perf_counter() - started
    release(runtime.source)
    status("warmup")
    checked = measured.warmup(runtime, work)
    actual = np.empty((bench.REPEATS, bench.SAMPLES, 6, 120, 120), np.float32)
    score_fixture = np.zeros((bench.REPEATS, bench.SAMPLES, N_MAP + bench.SAMPLES - 1), np.float64)
    rows, memories, device_used = [], [], []
    for repeat in range(bench.REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        guard(f"repeat {repeat}")
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
            free, _ = torch.cuda.mem_get_info()
            device_used.append(device_total - free)
            guard(f"repeat {repeat} arrival {i}")
        info = runtime.memory()
        info.update({"process_rss_before": before, "process_rss_after": bench.rss(),
                     "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     "cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved(),
                     "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
                     "cuda_peak_reserved": torch.cuda.max_memory_reserved(),
                     "cuda_device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                     "device_used_max_after_arrival_in_repeat": int(max(device_used[-bench.SAMPLES:]))})
        assert info["cuda_peak_reserved"] < info["cuda_device_total_bytes"]
        memories.append(info)
        status("measure", repeat=repeat + 1)
        print("RINGPP_EXACTMAX_MEASURE", repeat + 1, bench.REPEATS, flush=True)
    del encoded
    AUDIT.mkdir(parents=True, exist_ok=True)
    np.save(AUDIT / "actual_tiring.npy", actual)
    np.save(AUDIT / "candidate_scores.npy", score_fixture)
    bench.write(OUT / f"timed_rows_n{N_MAP}.json", {"rows": rows, "memory_repeats": memories})
    # 시간·메모리 스냅샷 뒤 검증만 한다. GPU 뱅크는 이후 쓰지 않으므로 먼저 놓는다.
    runtime.bank = None
    gc.collect()
    torch.cuda.empty_cache()
    checked["all_queries_same_execution"] = official_verify(runtime, work, actual, score_fixture, rows)
    variation = previous.repeat_variation(actual, rows)
    assert variation["winner_changed_queries"] == 0
    assert variation["max_confidence_spread"] == 0
    assert all(v["max_abs_vs_repeat0"] == 0 for v in variation["raw_feature_changes"])
    checked["official_native_repeat_variation"] = variation
    checked["native_requested_bytes_per_feature_call"] = 360
    checked["native_audit_sha256"] = bench.sha(previous.AUDIT)
    bench.write(OUT / f"verification_measure_n{N_MAP}.json", checked)
    files = {str(AUDIT / name): bench.sha(AUDIT / name) for name in ("actual_tiring.npy", "candidate_scores.npy")}
    files[str((OUT / f"timed_rows_n{N_MAP}.json").relative_to(REPO))] = bench.sha(OUT / f"timed_rows_n{N_MAP}.json")
    result = {
        "schema": "pdv2-table3-measurement-v1", "method": "ringpp", "n_map": N_MAP,
        "status": "passed", "publish_ready": False, "memory_publish_ready": True,
        "implementation": "released point features + verified exact grid max",
        "implementation_source_sha256": bench.sha(Path(fix.__file__)),
        "wrapper_source_sha256": bench.sha(Path(__file__)),
        "same_test_implementation": True,
        "protocol_revision": 3, "protocol_sha256": bench.sha(OUT / "common_protocol.json"),
        "method_protocol_sha256": bench.sha(OUT / "ringpp_exactmax_protocol.json"),
        "source_sha256": bench.sha(Path(__file__)), "v3_source_sha256": bench.sha(Path(measured.__file__)),
        "sequence": bench.SEQUENCE, "seed": None,
        "candidate_range": [N_MAP, N_MAP + bench.SAMPLES - 1],
        "mean_candidates": N_MAP + (bench.SAMPLES - 1) / 2,
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
        "device_memory": {"total": device_total, "used_before_runtime": device_total - device_free_start,
                          "used_max_after_arrival": int(max(device_used)),
                          "scope": "device-wide cudaMemGetInfo after each arrival; includes other processes and CUDA context"},
        "reference_batch1_cache": bench.read(CACHE / "complete.json"),
        "raw_samples": rows,
        "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "platform": platform.platform(), "threads": 1,
                        "torch_threads": torch.get_num_threads()},
        "scope": spec["scope"] + " RING++ official point features with verified exact grid max (np.fmax.at); official Radon/correlation with128-pair bounded chunks; all queries checked within each actual execution. Native memory is requested live capacity after300 encodes in a fresh cached-map process, not physical CUDA rounding or a long-run bound.",
        "completed_unix": time.time(),
    }
    measured.atomic_result(target, result)
    status("measure_done")
    print("COMPLETE_RINGPP_EXACTMAX", json.dumps(result["timings_ms"]), result["resident_bytes"]["total"], flush=True)


def memtrace() -> None:
    """LD_PRELOAD 추적기 아래에서 같은 실행을 한 번 더 돌려 외부 CUDA 할당을 센다(시간 제외)."""
    import ctypes

    assert os.environ.get("LD_PRELOAD", "").endswith("libnsd_cuda_trace.so")
    config = configure()
    target = OUT / "memtrace/summary.json"
    if target.exists():
        print("MEMTRACE_ALREADY_COMPLETE", flush=True)
        return
    trace = Path(os.environ["NSD_CUDA_TRACE_PATH"])
    snap = ctypes.CDLL(None).nsd_native_trace_snapshot
    snap.argtypes, snap.restype = [ctypes.c_char_p], None
    labels = []

    def mark(label: str) -> None:
        torch.cuda.synchronize()
        snap(label.encode())
        free, total = torch.cuda.mem_get_info()
        labels.append({"label": label, "device_used": total - free, "torch_allocated": torch.cuda.memory_allocated(),
                       "torch_reserved": torch.cuda.memory_reserved(), "feature_calls": runtime.feature_calls if runtime else 0})

    runtime = None
    guard("memtrace start")
    work = bench.Workload(bench.read(bench.OUT / "protocol.json"), N_MAP)
    torch.cuda.init()
    mark("context")
    runtime = Runtime(config["base_cache_protocol"], work)
    release(runtime.source)
    mark("map_loaded")
    with torch.no_grad():
        for i, points in enumerate(work.points[:bench.WARMUP]):
            runtime.append(runtime.encode(points))
            runtime.query_with_audit(work.initial + i - 15)
    runtime.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mark("measure_start")
    for i, points in enumerate(work.points):
        t = work.initial + i
        encoded = runtime.encode(points)
        mark(f"encoded_{i}")
        runtime.append(encoded)
        runtime.query_with_audit(t - 15)
        mark(f"queried_{i}")
        guard("memtrace")
    peak_alloc, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    mark("measure_end")
    events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    live, owner_of, current, peak, peak_event, peak_owner = {}, {}, 0, 0, None, {}
    window = [e["event_id"] for e in events if e.get("type") == "snapshot" and e["label"] in ("measure_start", "measure_end")]
    by_owner = {}
    for e in events:
        if e.get("type") == "allocate" and e.get("error") == 0:
            live[e["pointer"]] = e["requested_bytes"]
            owner_of[e["pointer"]] = e["owner"]
            current += e["requested_bytes"]
            by_owner[e["owner"]] = by_owner.get(e["owner"], 0) + e["requested_bytes"]
        elif e.get("type") == "free" and e["pointer"] in live:
            size, owner = live.pop(e["pointer"]), owner_of.pop(e["pointer"])
            current -= size
            by_owner[owner] -= size
        if window[0] <= e["event_id"] <= window[1] and current > peak:
            peak, peak_event, peak_owner = current, e["event_id"], dict(by_owner)
    snaps = {e["label"]: e for e in events if e.get("type") == "snapshot"}
    voxelfeat = lambda label: sum(v for k, v in snaps[label]["by_owner"].items() if "voxelfeat" in k)
    increments = [voxelfeat(f"encoded_{i}") - (voxelfeat(f"queried_{i-1}") if i else voxelfeat("measure_start"))
                  for i in range(bench.SAMPLES)]
    errors = [e for e in events if e.get("error", 0) != 0]
    summary = {
        "status": "passed" if not errors else "cuda_errors", "trace": str(trace.relative_to(REPO)),
        "trace_sha256": bench.sha(trace), "tracer_sha256": bench.sha(TRACER),
        "window": "measure_start..measure_end (96 arrivals, one repeat, after warmup12 and reset)",
        "peak_live_requested_bytes_all_cuda_apis": peak, "peak_event_id": peak_event,
        "peak_by_owner": {k: v for k, v in peak_owner.items() if v},
        "torch_peak_allocated": peak_alloc, "torch_peak_reserved": peak_reserved,
        "external_at_peak_bytes": int(peak - sum(v for k, v in peak_owner.items() if "libc10_cuda" in k)),
        "voxelfeat_increment_per_encode": sorted(set(increments)),
        "voxelfeat_live_at_end": voxelfeat("measure_end"), "feature_calls_at_end": runtime.feature_calls,
        "end_live_by_owner": snaps["measure_end"]["by_owner"], "cuda_api_errors": errors[:5],
        "device_snapshots": labels[:3] + labels[-3:],
        "device_used_max": max(x["device_used"] for x in labels),
        "scope": "requested bytes of intercepted cudaMalloc/cudaMallocArray/cudaMalloc3DArray incl. torch caching-allocator segments; not physical rounding nor CUDA context/driver/cuFFT-internal non-intercepted memory. device_used is device-wide.",
    }
    bench.write(target, summary)
    print("MEMTRACE_DONE", json.dumps({k: summary[k] for k in ("peak_live_requested_bytes_all_cuda_apis", "torch_peak_reserved", "external_at_peak_bytes", "voxelfeat_increment_per_encode")}), flush=True)


def equivalence() -> None:
    """비용 경로 인코더로 시험 mulran/DCC01 앞128키프레임을 다시 만들고 46시험 원장과 대조한다."""
    configure()
    target = WORK / "equivalence/verification.json"
    if target.exists():
        print("EQUIV_ALREADY_COMPLETE", flush=True)
        return
    source = REPO / "results/pdv2_ringpp_maxfix_20261105"
    protocol = bench.read(source / "protocol.json")
    record = next(r for r in protocol["sequences"] if (r["sensor"], r["sequence"]) == ("mulran", "DCC01"))
    rol = original.reference
    cache_path = rol.split_cache("odom", "test", "mulran", "DCC01")
    assert bench.sha(cache_path) == record["cache_sha256"]
    ids, _ = rol.load_keyframes(cache_path)
    assert hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest() == record["scan_ids_sha256"]
    frames = 128
    loader = rol.make_loader("mulran", "DCC01")
    core = rol.import_ring_core()
    normalizer = torch.nn.InstanceNorm2d(1, affine=False, track_running_stats=False)
    packed = np.load(source / "points/mulran_DCC01/shard_000000.npz")
    assert np.array_equal(packed["scan_ids"][:frames], ids[:frames])
    bank = torch.empty((frames, 6, 120, 120), dtype=torch.complex64, device="cuda")
    normalized_equal = 0
    with torch.no_grad():
        for k in range(frames):
            points = loader[int(ids[k])]["points"]
            _, pcn = rol.official_points("mulran", points)
            normalized_equal += hashlib.sha256(np.ascontiguousarray(pcn, dtype=np.float32).tobytes()).hexdigest() == str(packed["normalized_sha256"][k])
            value, _ = original.encode(core, "ringpp", points, check=True)
            bank[k] = torch.fft.fft2(normalizer(value[None]), dim=-2, norm="ortho")[0]
    with np.load(source / "odom/dump_ringpp_mulran_DCC01.npz") as z:
        ledger = {k: z[k] for k in ("query_idx", "j_star", "score_corr")}
    rows = []
    for t in range(30, frames):
        scores = correlate_prefix_fast(bank, True, t, t - 29, measured.CHUNK, "cuda")
        j = int(np.argmax(scores))
        index = int(np.searchsorted(ledger["query_idx"], t))
        assert ledger["query_idx"][index] == t
        rows.append({"query": t, "winner": j, "ledger_winner": int(ledger["j_star"][index]),
                     "score": float(scores[j] - 1.), "ledger_score": float(ledger["score_corr"][index])})
    diff = [abs(r["score"] - r["ledger_score"]) for r in rows]
    result = {"sequence": "mulran/DCC01", "frames": frames, "queries": len(rows),
              "normalized_points_sha_equal": int(normalized_equal),
              "winner_equal": sum(r["winner"] == r["ledger_winner"] for r in rows),
              "score_max_abs_diff": float(max(diff)), "score_mean_abs_diff": float(np.mean(diff)),
              "encoder_path": "cost-path original.encode with install_maxfix (raw loader → official_points → GPUFeatureExtractor → np.fmax.at → generate_RING)",
              "ledger_hardware": "RTX PRO 6000 MIG (results/pdv2_ringpp_maxfix_20261105/bev/mulran_DCC01/status.json)",
              "local_gpu": torch.cuda.get_device_name(),
              "ledger_sha256": bench.sha(source / "odom/dump_ringpp_mulran_DCC01.npz"),
              "rows": rows, "source_sha256": bench.sha(Path(__file__))}
    result["passed"] = result["normalized_points_sha_equal"] == frames and result["winner_equal"] == len(rows) and result["score_max_abs_diff"] < 1e-4
    bench.write(target, result)
    print("EQUIV", json.dumps({k: result[k] for k in ("passed", "queries", "winner_equal", "normalized_points_sha_equal", "score_max_abs_diff")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("equiv", "cache", "measure", "memtrace"))
    args = parser.parse_args()
    try:
        {"equiv": equivalence, "cache": cache, "measure": measure, "memtrace": memtrace}[args.stage]()
    except LowMemory as error:
        status("paused_low_memory", reason=str(error))
        print("LOW_MEMORY_PAUSE", error, flush=True)
        raise SystemExit(EXIT_LOW_MEMORY)
