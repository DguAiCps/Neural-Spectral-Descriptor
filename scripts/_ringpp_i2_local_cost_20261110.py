#!/usr/bin/env python3
"""I2 정규화 RING++의 표2·표S17 비용을 로컬 RTX5080에서 잰다(정확max 비용 경로의 opt-in 사본).

docker exec -w /workspace/Neural-Spectral-Codec -e PYTHONDONTWRITEBYTECODE=1 ring_baseline \
  python3 scripts/_ringpp_i2_local_cost_20261110.py {equiv,cache,measure,memtrace} [--n-map N]

시간 경로는 `_ringpp_exactmax_local_cost_20261107.py`(= `_pdv2_table3_ringpp_v3.py` 측정 루프)와 같다.
- encode: 공식 전처리·점 특징 + 정확max(np.fmax.at) + 공식 generate_RING (바꾸지 않음)
- append: 프레임마다 한 번 I2 정규화(float64 평균·모분산, 정확 상수 채널 0, eps 없음) + 열 FFT + 뱅크 추가
- query: `_causal_ring_pr.correlate_prefix_fast` 청크128 전수 순환 상관(바꾸지 않음)
I2 식은 시험46 동결 `i2_norm.py`(sha256 051a6179…)를 해시 그대로 복사해 쓴다. 공식 I1 FFT 캐시(fft.npy)는 읽지 않는다.
초기 지도는 기존 정확max raw TI-RING 캐시(raw.npy)에서 I2 FFT를 D에 블록별로 새로 만든다.
시간 구간 밖 변경: reset·검증 참조 적재 때 512행마다 읽기 매핑 페이지 반환(PSS 6GiB 제한), 검증은 I2 opt-in 공식 상관.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import _ringpp_exactmax_local_cost_20261107 as cost  # 스레드·TF32 환경 변수를 먼저 설정한다

import numpy as np
import torch

bench, original, previous, measured, fix = cost.bench, cost.original, cost.previous, cost.measured, cost.fix
from _causal_ring_pr import correlate_prefix_fast  # noqa: E402

REPO = bench.REPO
LARGE = Path(os.environ.get("RINGPP_I2_ROOT", "/workspace/data/pdv2_ringpp_i2_cost_20261110"))
OUT = LARGE / "cost"
RAW_CACHE = cost.LARGE / "ringpp_batch1/n10000"  # 기존 정확max 캐시(읽기만). raw.npy만 쓴다.
FFT_CACHE = LARGE / "i2_fft/n10000"
I2_SOURCE = LARGE / "code/i2_norm.py"
I2_SHA256 = "051a6179a018dba565f6abfd7195c8609c5f336b4c279243a2ec872e21d4130c"
TEST46_PROTOCOL_SHA256 = "5e72d07d1eef45b12e5f188b99b89ebdfded976adea23f89f78b2442d81c85fe"
I2_LEDGER = LARGE / "inputs/dump_ringpp_i2_mulran_DCC01.npz"
CHUNK = measured.CHUNK
BLOCK = cost.BLOCK
FRAMES_10000 = 10000 + 44 + bench.SAMPLES
SCORE_TOL = 1e-4
TIE_TOL = 1e-5
FFT_TOL = 3e-5
MAX_PSS = 6 << 30
EXIT_LOW_MEMORY = cost.EXIT_LOW_MEMORY


def load_i2():
    """해시를 확인한 동결 I2 모듈을 불러온다."""
    assert bench.sha(I2_SOURCE) == I2_SHA256, "동결 i2_norm.py 해시 불일치"
    spec = importlib.util.spec_from_file_location("i2_norm", I2_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["i2_norm"] = module
    spec.loader.exec_module(module)
    return module


I = load_i2()


def pss() -> int:
    """자기 프로세스의 PSS 바이트."""
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        if line.startswith("Pss:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Pss 없음")


def guard(where: str, check_pss: bool = False) -> None:
    """MemAvailable 8GiB, 필요하면 자기 PSS 6GiB 기준을 넘으면 결과 없이 멈춘다."""
    cost.guard(where)
    if check_pss and pss() > MAX_PSS:
        raise cost.LowMemory(f"{where}: PSS {pss()} > {MAX_PSS}")


def status(stage: str, **extra) -> None:
    """계산 프로세스 자신의 상태를 D에 남긴다."""
    bench.write(OUT / "compute_status.json", {"stage": stage, "pid": os.getpid(), "updated_unix": time.time(),
                                              "rss_bytes": bench.rss(), "pss_bytes": pss(),
                                              "mem_available": cost.available(), **extra})


def settings() -> dict:
    """정확max 비용 규약(동결 파일 대조)에 I2 정규화와 검증 경계만 더해 고정한다."""
    exactmax = cost.configure()  # 기존 정확max 규약 파일과 같아야 통과하고, 격자 max를 설치한다
    raw_meta = bench.read(RAW_CACHE / "complete.json")
    return {
        "schema": "pdv2-table3-ringpp-i2-v1",
        "exactmax_protocol_sha256": bench.sha(cost.OUT / "ringpp_exactmax_protocol.json"),
        "exactmax_protocol_matches": True,
        "common_protocol_sha256": bench.sha(cost.OUT / "common_protocol.json"),
        "base_cache_protocol": exactmax["base_cache_protocol"],
        "normalization": {
            "name": "I2", "source": str(I2_SOURCE), "sha256": I2_SHA256,
            "function": "i2_norm.unit_variance_torch",
            "definition": "(sample, channel)별 H×W float64 평균·모분산, max==min이면 0, 아니면 (x−μ)/√σ² → float32, eps 없음",
            "test46_protocol_sha256": TEST46_PROTOCOL_SHA256,
            "relation_to_official": "공식 circorr2 InstanceNorm2d(eps 1e-5)가 아닌 논문 '평균0·분산1' 정의 기반 고정 변형. 각도 G1·채널·검색·정답·점수 규약 불변",
        },
        "timed_path": {"encode": "original.encode(ringpp) with install_maxfix (unchanged)",
                       "append": "fft2(I2(tiring[None]), dim=-2, ortho)[0] -> bank row (I2 statistics per frame once)",
                       "query": f"correlate_prefix_fast chunk{CHUNK} exhaustive, first argmax, score=corr-1 (unchanged)",
                       "loop": "_ringpp_exactmax_local_cost_20261107.measure와 같은 timed encode/append/query, 96도착×3, 워밍업12"},
        "initial_cache": {"raw_source": str(RAW_CACHE / "raw.npy"),
                          "raw_sha256": raw_meta["files"]["raw.npy"],
                          "raw_complete_sha256": bench.sha(RAW_CACHE / "complete.json"),
                          "i2_fft": str(FFT_CACHE / "fft_i2.npy"),
                          "i2_fft_build": "raw.npy 128행 블록 -> 프레임별 append와 같은 I2+FFT(배치1) -> D 순차 기록·블록 해시·이어받기",
                          "map_sizes": "N=1000·5000은 같은 1만 파일의 앞 N+140행(장면 id 접두 일치 확인, 각 행은 그 스캔만으로 결정)",
                          "official_i1_fft_reuse": False},
        "untimed_changes": ["reset·검증 참조 적재 때 512행마다 읽기 매핑 페이지 반환(PSS 6GiB 제한)",
                            "검증의 공식 상관은 i2_norm.batch_circorr_i2(공식 circorr2 클래스, 정규화 모듈만 I2)",
                            "자기 PSS·MemAvailable 검사(반복 경계·검증 단계)"],
        "tolerances": {"official_i2_score_abs": SCORE_TOL, "winner": "exact first argmax; 다르면 공식 I2 상위 두 점수 차 <= 1e-5만 near-tie로 기록",
                       "near_tie_abs": TIE_TOL, "fft_cache_abs": FFT_TOL,
                       "repeat_determinism": "세 반복 특징 차 0·승자 변화 0·점수 범위 0",
                       "equivalence": "DCC01 앞128키프레임 98질의: 시험 I2 원장과 승자 같음(또는 공식 I2 near-tie <= 1e-5), 점수 차 < 1e-4"},
        "memory_definition": "previous.Runtime.memory 그대로: CPU 고유 수치 버퍼 + torch 현재 할당 + native 미해제 요청 360B×호출 수",
        "stage_order": ["equiv", "cache", "measure 10000", "measure 1000", "measure 5000", "memtrace 1000"],
        "no_selection": "시험셋·비용 결과로 설정을 고르거나 다시 재지 않는다. 계측 오류가 있을 때만 수정 전후를 보존한다.",
    }


def freeze() -> dict:
    """첫 측정 전 규약을 고정하고 이후 같은지 확인한다."""
    config = settings()
    path = OUT / "ringpp_i2_protocol.json"
    OUT.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert bench.read(path) == config, "동결한 I2 비용 규약 변경"
    else:
        bench.write(path, config)
    return config


@torch.no_grad()
def i2_fft(tiring: torch.Tensor) -> torch.Tensor:
    """append와 같은 프레임 하나의 I2 정규화·열 FFT."""
    return torch.fft.fft2(I.unit_variance_torch(tiring[None]), dim=-2, norm="ortho")[0]


def i2_scores(reference: torch.Tensor, q: int, count: int) -> np.ndarray:
    """I2 opt-in 공식 circorr2로 질의 q와 앞 count 후보의 점수(1 − dist)."""
    bs = original.reference.import_ring()["cfg"].batch_size
    out = np.empty(count, np.float64)
    query = reference[q].unsqueeze(0)
    with torch.no_grad():
        for j0 in range(0, count, bs):
            j1 = min(j0 + bs, count)
            dist, _ = I.batch_circorr_i2(query.repeat(j1 - j0, 1, 1, 1), reference[j0:j1])
            out[j0:j1] = 1. - np.asarray(dist, np.float64).reshape(-1)
    return out


def winner_check(direct: np.ndarray, winner: int) -> dict:
    """공식 첫 argmax와 같은지, 다르면 near-tie인지 판정한다."""
    j = int(np.argmax(direct))
    gap = float(direct[j] - direct[winner])
    return {"official_argmax": j, "same": j == winner, "official_gap": gap, "near_tie": j != winner and gap <= TIE_TOL}


def cache() -> None:
    """기존 raw TI-RING에서 초기 지도 I2 FFT를 128행씩 D에 기록하고 중단 지점에서 잇는다."""
    config = freeze()
    FFT_CACHE.mkdir(parents=True, exist_ok=True)
    done = FFT_CACHE / "complete.json"
    if done.exists():
        assert bench.read(done)["protocol_sha256"] == bench.sha(OUT / "ringpp_i2_protocol.json")
        print("CACHE_ALREADY_COMPLETE", flush=True)
        return
    guard("cache start", True)
    tick = time.time()
    raw_meta = bench.read(RAW_CACHE / "complete.json")
    raw_sha = bench.sha(RAW_CACHE / "raw.npy")
    assert raw_sha == raw_meta["files"]["raw.npy"] == config["initial_cache"]["raw_sha256"]
    raw = np.load(RAW_CACHE / "raw.npy", mmap_mode="r")
    shape = raw.shape
    assert shape == (FRAMES_10000, 6, 120, 120), shape
    path, cursor = FFT_CACHE / "fft_i2.npy", FFT_CACHE / "resume_cursor.json"
    row = int(np.prod(shape[1:])) * np.dtype(np.complex64).itemsize
    state = bench.read(cursor) if cursor.exists() else {"completed": 0, "blocks": []}
    handle = path.open("wb" if state["completed"] == 0 else "r+b")
    offset = cost.write_header(handle, np.complex64, shape)
    if state["completed"]:
        handle.truncate(offset + state["completed"] * row)
        last = state["blocks"][-1]
        handle.seek(offset + last["begin"] * row)
        tail = np.frombuffer(handle.read((last["end"] - last["begin"]) * row), np.complex64)
        assert cost.block_sha(tail) == last["fft_sha256"], "이어받기 꼬리 불일치"
    handle.seek(0, os.SEEK_END)
    start = state["completed"]
    raw_blocks = {(b["begin"], b["end"]): b["raw_sha256"] for b in bench.read(RAW_CACHE / "resume_cursor.json")["blocks"]}
    const_channels, min_var = 0, float("inf")
    print("CACHE_RESUME_FROM", start, shape[0], flush=True)
    for lo in range(start, shape[0], BLOCK):
        hi = min(lo + BLOCK, shape[0])
        guard(f"cache row {lo}", True)
        block_raw = np.array(raw[lo:hi], copy=True)
        cost.release(raw)
        assert cost.block_sha(block_raw) == raw_blocks[(lo, hi)], ("raw 블록 해시 불일치", lo)
        values = torch.from_numpy(block_raw).cuda()
        spectral = np.empty((hi - lo, *shape[1:]), np.complex64)
        for k in range(hi - lo):
            spectral[k] = i2_fft(values[k]).cpu().numpy()
        mu, var, const = I.channel_stats_torch(values)
        const_channels += int(const.sum())
        nonconst = var[~const]
        if nonconst.numel():
            min_var = min(min_var, float(nonconst.min()))
        handle.write(spectral.tobytes())
        handle.flush()
        os.fsync(handle.fileno())
        state["blocks"].append({"begin": lo, "end": hi, "fft_sha256": cost.block_sha(spectral)})
        state["completed"] = hi
        bench.write(cursor, state)
        status("cache", completed=hi, total=shape[0], seconds=round(time.time() - tick, 1))
        del values, spectral, block_raw
    handle.close()
    fft_map = np.load(path, mmap_mode="r")
    assert fft_map.shape == shape and fft_map.dtype == np.complex64
    for block in state["blocks"]:
        assert cost.block_sha(fft_map[block["begin"]:block["end"]]) == block["fft_sha256"]
        cost.release(fft_map)
    probes = np.unique(np.linspace(0, shape[0] - 1, 32, dtype=np.int64))
    append_error, batch_error = 0.0, 0.0
    with torch.no_grad():
        values = torch.from_numpy(np.array(raw[probes], copy=True)).cuda()
        batch, _ = I.precompute_fft_bank_i2(values, "cuda")  # 시험46 경로(512 묶음)의 같은 행
        for k, i in enumerate(probes):
            again = i2_fft(values[k]).cpu().numpy()
            append_error = max(append_error, float(abs(again - fft_map[i]).max()))
            batch_error = max(batch_error, float(abs(batch[k].cpu().numpy() - fft_map[i]).max()))
    assert append_error < FFT_TOL and batch_error < FFT_TOL, (append_error, batch_error)
    cost.release(fft_map)
    cost.release(raw)
    del fft_map, raw
    bench.write(done, {"protocol_sha256": bench.sha(OUT / "ringpp_i2_protocol.json"), "frames": shape[0],
                       "raw_sha256": raw_sha, "seconds": time.time() - tick, "resumed_from_row": start,
                       "blocks_verified_after_write": len(state["blocks"]),
                       "probe_rows": probes.tolist(), "probe_append_path_max_abs": append_error,
                       "probe_test46_batch_path_max_abs": batch_error,
                       "exact_constant_channels": const_channels if start == 0 else None,
                       "min_nonconstant_float64_var": min_var if start == 0 else None,
                       "cache_source_sha256": bench.sha(Path(__file__)),
                       "files": {"fft_i2.npy": bench.sha(path),
                                 "scan_ids.npy(raw cache)": bench.sha(RAW_CACHE / "scan_ids.npy")}})
    print("CACHE_COMPLETE", round(time.time() - tick, 1), append_error, batch_error, flush=True)


class Runtime(measured.Runtime):
    """v3 실행과 같고 초기 지도 FFT가 I2이며 append가 I2 정규화를 쓴다."""

    def __init__(self, work: bench.Workload) -> None:
        self.work = work
        self.config = {"method": "ringpp", "channels": 6}
        self.variant = "ringpp"
        self.core = original.reference.import_ring_core()
        meta = bench.read(FFT_CACHE / "complete.json")
        assert meta["protocol_sha256"] == bench.sha(OUT / "ringpp_i2_protocol.json")
        ids = np.load(RAW_CACHE / "scan_ids.npy")
        np.testing.assert_array_equal(ids[:work.end], work.ids)
        self.source = np.load(FFT_CACHE / "fft_i2.npy", mmap_mode="r")[:work.end]
        self.raw_source = np.load(RAW_CACHE / "raw.npy", mmap_mode="r")[:work.end]
        shape = (work.end, 6, 120, 120)
        self.on_gpu = np.prod(shape) * 8 <= original.GPU_LIMIT
        assert self.on_gpu
        self.bank = torch.empty(shape, dtype=torch.complex64, device="cuda")
        self.feature_calls = 0
        self.reset()

    def reset(self) -> None:
        """동일한 초기 지도를 복원하고 512행마다 매핑 페이지를 뗀다."""
        self.n = self.work.initial
        for i in range(0, self.n, 512):
            end = min(i + 512, self.n)
            self.bank[i:end].copy_(torch.from_numpy(np.array(self.source[i:end], copy=True)))
            release_view(self.source)
        bench.sync()

    @torch.no_grad()
    def append(self, tiring: torch.Tensor) -> None:
        """새 프레임의 I2 정규화·FFT를 실행하고 실제 색인 상태에 추가한다."""
        value = torch.fft.fft2(I.unit_variance_torch(tiring[None]), dim=-2, norm="ortho")[0]
        self.bank[self.n].copy_(value)
        self.n += 1


def release_view(array: np.memmap) -> None:
    """앞부분 view가 공유하는 읽기 매핑의 상주 페이지를 프로세스에서 반환한다."""
    assert isinstance(array, np.memmap) and array._mmap is not None
    cost.release(array)


def warmup(runtime: Runtime, work: bench.Workload) -> dict:
    """처음 12도착의 새 특징·I2 FFT·공식 I2 점수를 대조하고 계측을 예열한다."""
    checks = []
    with torch.no_grad():
        for i, points in enumerate(work.points[:bench.WARMUP]):
            t = work.initial + i
            value = runtime.encode(points, check=True)
            actual = value.cpu().numpy()
            assert np.isfinite(actual).all()
            runtime.append(value)
            fft_error = float(abs(runtime.bank[t].cpu().numpy() - runtime.source[t]).max())
            assert fft_error < FFT_TOL, fft_error
            candidates = np.unique(np.linspace(0, t - 30, 16, dtype=np.int64))
            reference = torch.from_numpy(np.array(runtime.raw_source[candidates], copy=True)).cuda()
            distances, _ = I.batch_circorr_i2(value[None].repeat(len(candidates), 1, 1, 1), reference)
            direct = 1. - np.asarray(distances, np.float64).reshape(-1)
            fast = correlate_prefix_fast(runtime.bank, runtime.on_gpu, t, t - 29, CHUNK, "cuda")[candidates]
            error = float(abs(direct - fast).max())
            assert error < SCORE_TOL and int(np.argmax(direct)) == int(np.argmax(fast)), (t, error)
            runtime.query_with_audit(t - 15)
            old = np.asarray(runtime.raw_source[t])
            checks.append({"arrival": t, "direct_i2_score_max_error": error, "fresh_vs_cached_i2_fft_max_abs": fft_error,
                           "fresh_vs_cached_tiring_max_abs": float(abs(actual - old).max())})
            del value, reference
    release_view(runtime.raw_source)
    runtime.reset()
    return {"status": "passed", "raw_regeneration": len(checks), "fresh_feature_direct_score_checks": checks,
            "official_correlation": "i2_norm.batch_circorr_i2 (공식 circorr2, 정규화 모듈만 I2)"}


def official_verify(runtime: Runtime, work: bench.Workload, actual: np.ndarray,
                    score_fixture: np.ndarray, rows: list) -> dict:
    """모든 288질의를 각 반복의 실제 특징으로 I2 opt-in 공식 상관과 전 후보 대조한다."""
    reference = torch.empty((work.end, 6, 120, 120), dtype=torch.float32, device="cuda")
    for i in range(0, work.initial, 512):
        end = min(i + 512, work.initial)
        reference[i:end].copy_(torch.from_numpy(np.array(runtime.raw_source[i:end], copy=True)))
        release_view(runtime.raw_source)
    checks, near_ties = [], []
    for repeat in range(bench.REPEATS):
        reference[work.initial:].copy_(torch.from_numpy(actual[repeat]))
        for i in range(bench.SAMPLES):
            row = rows[repeat * bench.SAMPLES + i]
            q, count = row["query"], row["n_candidates"]
            direct = i2_scores(reference, q, count)
            saved = score_fixture[repeat, i, :count]
            error = float(abs(direct - saved).max())
            assert error < SCORE_TOL, (repeat, i, error)
            assert row["winner"] == int(np.argmax(saved))
            w = winner_check(direct, row["winner"])
            assert w["same"] or w["near_tie"], (repeat, i, w)
            if not w["same"]:
                near_ties.append({"repeat": repeat, "query": q, **w})
            assert abs(row["confidence"] - (direct[row["winner"]] - 1.)) < SCORE_TOL
            checks.append({"repeat": repeat, "query": q, "n_candidates": count, "winner": row["winner"],
                           "official_argmax": w["official_argmax"], "all_candidate_score_max_error": error})
        guard(f"verify {repeat}", True)
        status("verify", repeat=repeat + 1)
        print("RINGPP_I2_OFFICIAL_VERIFY", repeat + 1, bench.REPEATS, flush=True)
    del reference
    return {"status": "passed", "queries": len(checks), "checks": checks, "score_tolerance": SCORE_TOL,
            "winners_exact": len(checks) - len(near_ties), "near_ties": near_ties, "near_tie_tolerance": TIE_TOL,
            "reference_on": "GPU single copy (initial prefix from raw.npy + each repeat's actual arrivals)",
            "official_correlation": "i2_norm.batch_circorr_i2 in cfg.batch_size batches",
            "scope": "actual repeat-specific query/candidate TI-RING; all reference work after measured memory snapshots"}


def measure(n_map: int) -> None:
    """실제 새 점군 인코딩부터 최종 검색까지 96도착×3을 잰다(정확max 측정 루프와 같다)."""
    freeze()
    target = OUT / f"methods/ringpp_i2_n{n_map}.json"
    if target.exists():
        assert bench.read(target)["status"] == "passed"
        print("MEASURE_ALREADY_COMPLETE", n_map, flush=True)
        return
    guard("measure start", True)
    spec = bench.read(bench.OUT / "protocol.json")
    work = bench.Workload(spec, n_map)
    torch.cuda.init()
    device_free_start, device_total = torch.cuda.mem_get_info()
    started = time.perf_counter()
    runtime = Runtime(work)
    initial_s = time.perf_counter() - started
    status("warmup", n_map=n_map)
    checked = warmup(runtime, work)
    actual = np.empty((bench.REPEATS, bench.SAMPLES, 6, 120, 120), np.float32)
    score_fixture = np.zeros((bench.REPEATS, bench.SAMPLES, n_map + bench.SAMPLES - 1), np.float64)
    rows, memories, device_used, pss_repeats = [], [], [], []
    for repeat in range(bench.REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        guard(f"repeat {repeat}", True)
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
            cost.guard(f"repeat {repeat} arrival {i}")
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
        pss_repeats.append(pss())
        status("measure", n_map=n_map, repeat=repeat + 1)
        print("RINGPP_I2_MEASURE", n_map, repeat + 1, bench.REPEATS, flush=True)
    del encoded
    audit = LARGE / f"audit/n{n_map}"
    audit.mkdir(parents=True, exist_ok=True)
    np.save(audit / "actual_tiring.npy", actual)
    np.save(audit / "candidate_scores.npy", score_fixture)
    bench.write(OUT / f"timed_rows_n{n_map}.json", {"rows": rows, "memory_repeats": memories})
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
    bench.write(OUT / f"verification_measure_n{n_map}.json", checked)
    files = {str(audit / name): bench.sha(audit / name) for name in ("actual_tiring.npy", "candidate_scores.npy")}
    files[str(OUT / f"timed_rows_n{n_map}.json")] = bench.sha(OUT / f"timed_rows_n{n_map}.json")
    result = {
        "schema": "pdv2-table3-measurement-v1", "method": "ringpp", "n_map": n_map,
        "status": "passed", "publish_ready": False, "memory_publish_ready": True,
        "implementation": "released point features + verified exact grid max + I2 normalization (zero mean, unit variance, no eps)",
        "normalization": "I2", "normalization_source_sha256": I2_SHA256,
        "implementation_source_sha256": bench.sha(Path(fix.__file__)),
        "wrapper_source_sha256": bench.sha(Path(__file__)),
        "exactmax_wrapper_sha256": bench.sha(Path(cost.__file__)),
        "same_test_implementation": True,
        "protocol_revision": 3, "protocol_sha256": bench.sha(cost.OUT / "common_protocol.json"),
        "method_protocol_sha256": bench.sha(OUT / "ringpp_i2_protocol.json"),
        "source_sha256": bench.sha(Path(__file__)), "v3_source_sha256": bench.sha(Path(measured.__file__)),
        "sequence": bench.SEQUENCE, "seed": None,
        "candidate_range": [n_map, n_map + bench.SAMPLES - 1],
        "mean_candidates": n_map + (bench.SAMPLES - 1) / 2,
        "samples": bench.SAMPLES, "repeats": bench.REPEATS, "warmup": bench.WARMUP,
        "timings_ms": {key: bench.stats([r[field] for r in rows]) for key, field in (
            ("encode", "encode"), ("state_update", "state_update"), ("update", "update"),
            ("query", "query_ms"), ("processing", "processing"))},
        "resident_bytes": memories[-1], "memory_repeats": memories, "verification": checked,
        "process_pss_after_repeat": pss_repeats,
        "raw_input_fixture_bytes": work.input_bytes, "raw_input_sha256": work.input_hash,
        "reference_fixture_bytes": int(runtime.source.nbytes + runtime.raw_source.nbytes),
        "post_timing_audit_fixture_bytes": int(actual.nbytes + score_fixture.nbytes),
        "audit_artifacts": files, "official_native_repeat_variation": variation,
        "initial_cached_state_build_s": initial_s,
        "device_memory": {"total": device_total, "used_before_runtime": device_total - device_free_start,
                          "used_max_after_arrival": int(max(device_used)),
                          "scope": "device-wide cudaMemGetInfo after each arrival; includes other processes and CUDA context"},
        "reference_i2_fft_cache": bench.read(FFT_CACHE / "complete.json"),
        "raw_samples": rows,
        "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "platform": platform.platform(), "threads": 1,
                        "torch_threads": torch.get_num_threads()},
        "scope": spec["scope"] + " RING++ official point features with verified exact grid max (np.fmax.at); I2 normalization (float64 mean/population variance, exact-constant channel 0, no eps) once per frame before the column FFT; official Radon and exhaustive circular correlation with128-pair bounded chunks; all queries checked within each actual execution against the I2 opt-in official circorr2. Native memory is requested live capacity after300 encodes in a fresh cached-map process.",
        "completed_unix": time.time(),
    }
    measured.atomic_result(target, result)
    status("measure_done", n_map=n_map)
    print("COMPLETE_RINGPP_I2", n_map, json.dumps(result["timings_ms"]), result["resident_bytes"]["total"], flush=True)


def equivalence() -> None:
    """비용 경로 인코더·I2 append로 시험 mulran/DCC01 앞128키프레임을 원시 스캔부터 만들어 시험 I2 원장과 대조한다."""
    freeze()
    target = OUT / "equivalence/verification.json"
    if target.exists():
        print("EQUIV_ALREADY_COMPLETE", flush=True)
        return
    guard("equiv start", True)
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
    packed = np.load(source / "points/mulran_DCC01/shard_000000.npz")
    assert np.array_equal(packed["scan_ids"][:frames], ids[:frames])
    raw = torch.empty((frames, 6, 120, 120), dtype=torch.float32, device="cuda")
    bank = torch.empty((frames, 6, 120, 120), dtype=torch.complex64, device="cuda")
    normalized_equal = 0
    with torch.no_grad():
        for k in range(frames):
            points = loader[int(ids[k])]["points"]
            _, pcn = rol.official_points("mulran", points)
            normalized_equal += hashlib.sha256(np.ascontiguousarray(pcn, dtype=np.float32).tobytes()).hexdigest() == str(packed["normalized_sha256"][k])
            value, _ = original.encode(core, "ringpp", points, check=True)
            raw[k] = value
            bank[k] = i2_fft(value)
    with np.load(I2_LEDGER) as z:
        ledger = {k: z[k] for k in ("query_idx", "j_star", "score_corr")}
    rows, near = [], []
    for t in range(30, frames):
        scores = correlate_prefix_fast(bank, True, t, t - 29, CHUNK, "cuda")
        j = int(np.argmax(scores))
        direct = i2_scores(raw, t, t - 29)
        index = int(np.searchsorted(ledger["query_idx"], t))
        assert ledger["query_idx"][index] == t
        lj = int(ledger["j_star"][index])
        w = winner_check(direct, lj)
        rows.append({"query": t, "winner": j, "ledger_winner": lj, "official_i2_argmax": w["official_argmax"],
                     "score": float(scores[j] - 1.), "ledger_score": float(ledger["score_corr"][index]),
                     "fast_vs_official_i2_max_abs": float(abs(scores - direct).max())})
        if j != lj:
            near.append({"query": t, "cost_winner": j, "ledger_winner": lj,
                         "official_gap_cost_vs_ledger": float(abs(direct[j] - direct[lj]))})
    diff = [abs(r["score"] - r["ledger_score"]) for r in rows]
    result = {"sequence": "mulran/DCC01", "frames": frames, "queries": len(rows),
              "normalized_points_sha_equal": int(normalized_equal),
              "winner_equal": sum(r["winner"] == r["ledger_winner"] for r in rows),
              "winner_differs": near,
              "score_max_abs_diff": float(max(diff)), "score_mean_abs_diff": float(np.mean(diff)),
              "fast_vs_official_i2_max_abs": float(max(r["fast_vs_official_i2_max_abs"] for r in rows)),
              "encoder_path": "cost-path original.encode with install_maxfix (raw loader → official_points → GPUFeatureExtractor → np.fmax.at → generate_RING) → I2 append FFT",
              "ledger": str(I2_LEDGER), "ledger_sha256": bench.sha(I2_LEDGER),
              "ledger_origin": "ringpp_i2_test46_20261109/result/ledgers/dump_ringpp_i2_mulran_DCC01.npz (root verified, 서버 RTX PRO 6000 BEV 캐시)",
              "local_gpu": torch.cuda.get_device_name(), "rows": rows, "source_sha256": bench.sha(Path(__file__))}
    result["passed"] = (result["normalized_points_sha_equal"] == frames and result["score_max_abs_diff"] < SCORE_TOL
                        and result["fast_vs_official_i2_max_abs"] < SCORE_TOL
                        and all(x["official_gap_cost_vs_ledger"] <= TIE_TOL for x in near))
    bench.write(target, result)
    print("EQUIV", json.dumps({k: result[k] for k in ("passed", "queries", "winner_equal", "normalized_points_sha_equal",
                                                       "score_max_abs_diff", "fast_vs_official_i2_max_abs")}), flush=True)
    assert result["passed"]


def memtrace(n_map: int) -> None:
    """LD_PRELOAD 추적기 아래 같은 I2 실행을 한 번 더 돌려 native 360B/호출 보존을 센다(시간 제외)."""
    import ctypes

    assert os.environ.get("LD_PRELOAD", "").endswith("libnsd_cuda_trace.so")
    freeze()
    folder = LARGE / f"memtrace/n{n_map}"
    target = folder / "summary.json"
    if target.exists():
        print("MEMTRACE_ALREADY_COMPLETE", flush=True)
        return
    trace = Path(os.environ["NSD_CUDA_TRACE_PATH"])
    assert trace.parent == folder
    snap = ctypes.CDLL(None).nsd_native_trace_snapshot
    snap.argtypes, snap.restype = [ctypes.c_char_p], None
    labels = []
    runtime = None

    def mark(label: str) -> None:
        torch.cuda.synchronize()
        snap(label.encode())
        free, total = torch.cuda.mem_get_info()
        labels.append({"label": label, "device_used": total - free, "torch_allocated": torch.cuda.memory_allocated(),
                       "torch_reserved": torch.cuda.memory_reserved(), "feature_calls": runtime.feature_calls if runtime else 0})

    guard("memtrace start", True)
    work = bench.Workload(bench.read(bench.OUT / "protocol.json"), n_map)
    torch.cuda.init()
    mark("context")
    runtime = Runtime(work)
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
        cost.guard("memtrace")
    peak_alloc, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    mark("measure_end")
    events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    live, owner_of, current, peak, peak_event, peak_owner, by_owner = {}, {}, 0, 0, None, {}, {}
    window = [e["event_id"] for e in events if e.get("type") == "snapshot" and e["label"] in ("measure_start", "measure_end")]
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
    voxelfeat = lambda label: sum(v for k, v in snaps[label]["by_owner"].items() if "voxelfeat" in k)  # noqa: E731
    increments = [voxelfeat(f"encoded_{i}") - (voxelfeat(f"queried_{i-1}") if i else voxelfeat("measure_start"))
                  for i in range(bench.SAMPLES)]
    errors = [e for e in events if e.get("error", 0) != 0]
    summary = {
        "status": "passed" if not errors and increments and set(increments) == {360} else "check_failed",
        "n_map": n_map, "trace": str(trace), "trace_sha256": bench.sha(trace), "tracer_sha256": bench.sha(cost.TRACER),
        "window": "measure_start..measure_end (96 arrivals, one repeat, after warmup12 and reset)",
        "peak_live_requested_bytes_all_cuda_apis": peak, "peak_event_id": peak_event,
        "peak_by_owner": {k: v for k, v in peak_owner.items() if v},
        "torch_peak_allocated": peak_alloc, "torch_peak_reserved": peak_reserved,
        "external_at_peak_bytes": int(peak - sum(v for k, v in peak_owner.items() if "libc10_cuda" in k)),
        "voxelfeat_increment_per_encode": sorted(set(increments)),
        "voxelfeat_live_at_end": voxelfeat("measure_end"), "feature_calls_at_end": runtime.feature_calls,
        "end_live_by_owner": snaps["measure_end"]["by_owner"], "cuda_api_errors": errors[:5],
        "device_snapshots": labels[:3] + labels[-3:], "device_used_max": max(x["device_used"] for x in labels),
        "comparison": "정확max I1 추적(results/pdv2_baseline_revision_20261107/ringpp/cost/memtrace/summary.json, N=10000)의 360B/인코딩과 같은지 확인",
        "scope": "requested bytes of intercepted cudaMalloc/cudaMallocArray/cudaMalloc3DArray incl. torch caching-allocator segments; not physical rounding nor CUDA context/driver/cuFFT-internal non-intercepted memory. device_used is device-wide.",
    }
    bench.write(target, summary)
    print("MEMTRACE_DONE", json.dumps({k: summary[k] for k in ("status", "peak_live_requested_bytes_all_cuda_apis", "torch_peak_reserved", "voxelfeat_increment_per_encode")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("equiv", "cache", "measure", "memtrace"))
    parser.add_argument("--n-map", type=int, choices=(1000, 5000, 10000), default=10000)
    args = parser.parse_args()
    try:
        if args.stage == "equiv":
            equivalence()
        elif args.stage == "cache":
            cache()
        elif args.stage == "measure":
            measure(args.n_map)
        else:
            memtrace(args.n_map)
    except cost.LowMemory as error:
        status("paused_low_memory", reason=str(error))
        print("LOW_MEMORY_PAUSE", error, flush=True)
        raise SystemExit(EXIT_LOW_MEMORY)
