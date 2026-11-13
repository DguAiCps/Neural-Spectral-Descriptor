#!/usr/bin/env python3
"""공식 RING/RING++의 실제 도착과 접두 전체 검색 비용을 계측한다."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
from pathlib import Path
import platform
import resource
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_name] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import numpy as np
import torch
import _pdv2_table3_benchmark as bench
import _ring_official_lib as reference
from _causal_ring_pr import correlate_prefix_fast, precompute_fft_bank

OUT = bench.OUT
CHUNK = 1024
GPU_LIMIT = 9 << 30


def configure() -> None:
    """CPU 한 스레드와 공식 RING·공통 로더의 이름공간을 설정한다."""
    import joblib

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert joblib.effective_n_jobs(-1) == 1
    reference.import_ring()
    name = "utils.cyclic_shift_distance"
    if name not in sys.modules:
        source = bench.REPO / "src/utils/cyclic_shift_distance.py"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    required = 81920 * 1024
    if soft != resource.RLIM_INFINITY and soft < required:
        assert hard == resource.RLIM_INFINITY or hard >= required
        resource.setrlimit(resource.RLIMIT_STACK, (required, hard))


def settings(variant: str) -> dict:
    """표 2의 공식 구현과 FFT 재사용을 그대로 동결한다."""
    assert variant in ("ring", "ringpp")
    root = reference.RING_DIR
    sources = [Path(__file__), Path(bench.__file__), Path(reference.__file__),
               bench.REPO / "scripts/_ring_official_pr.py",
               bench.REPO / "scripts/_causal_ring_pr.py",
               bench.REPO / "src/utils/cyclic_shift_distance.py"]
    sources += [root / name for name in (
        "utils/core.py", "utils/circorr2.py", "utils/config.py",
        "utils/extract_local_descriptor.py", "datasets/MulRanDataset.py",
        "utils/generate_bev_occ_cython/wrapper.pyx",
        "utils/generate_bev_occ_cython/src/manager.cu",
        "utils/generate_bev_occ_cython/src/kernel.cu",
        "utils/generate_bev_pointfeat_cython/wrapper.pyx",
        "utils/generate_bev_pointfeat_cython/src/manager.cu",
        "utils/generate_bev_pointfeat_cython/src/kernel.cu")]
    extensions = {}
    for name in ("voxelocc", "voxelfeat"):
        spec = importlib.util.find_spec(name)
        assert spec is not None
        path = Path(spec.origin)
        extensions[name] = {"path": str(path), "sha256": bench.sha(path)}
    return {
        "schema": "pdv2-table3-ring-protocol-v1", "revision": 1,
        "method": variant, "parent_protocol_sha256": bench.sha(OUT / "protocol.json"),
        "official_repository": "lus6-Jenny/RING", "official_commit": reference.RING_COMMIT,
        "dataset_table": reference.DATASET_TABLE["mulran"],
        "sources": {str(p.relative_to(bench.REPO)): bench.sha(p) for p in sources},
        "compiled_extensions": extensions,
        "encoder": "official read_pc/normalize_pc; generate_bev_occ" +
                   ("" if variant == "ring" else "/generate_bev_feats") +
                   "; generate_RING; batch1 float32 TI-RING",
        "channels": 1 if variant == "ring" else 6,
        "retrieval": "unchanged _causal_ring_pr.correlate_prefix_fast; exhaustive official normalized circular correlation; first argmax, score=corr-1",
        "update": "new TI-RING only; per-frame InstanceNorm and column FFT reused exactly as the Table2 ledger scorer",
        "fft_bank_gpu_limit_bytes": GPU_LIMIT, "pairs_per_chunk": CHUNK,
        "threads": 1, "joblib_n_jobs_minus_one_effective": 1,
        "batch": 1, "tf32_matmul": False, "tf32_cudnn": False,
        "external_memory": "torch allocator excludes temporary native voxel/radon allocations; RING++ native source additionally leaves three 30-float CUDA allocations per nonempty feature encoding; its full resident memory is unpublished until separately traced",
    }


def freeze(config: dict) -> None:
    """공통 동결 파일을 유지하고 방법별 원천을 고정한다."""
    path = OUT / f"{config['method']}_protocol.json"
    if path.exists():
        assert bench.read(path) == config, "동결한 RING 계측 원천 변경"
    else:
        bench.write(path, config)


def directory(variant: str, n_map: int) -> Path:
    """지도 크기별 배치 1 참조 캐시 위치다."""
    return OUT / f"{variant}_batch1" / f"n{n_map}"


def native_audit() -> dict:
    """외부 CUDA 할당의 해제 경로와 확인되지 않은 물리 크기를 기록한다."""
    root = reference.RING_DIR
    return {
        "occupancy": {"source": str((root / "utils/generate_bev_occ_cython/src/manager.cu").relative_to(bench.REPO)),
                      "allocate_lines": [32, 33, 34, 35], "free_lines": [69, 70, 71, 72],
                      "persistent_device_payload_after_retrieve": 0},
        "feature": {"source": str((root / "utils/generate_bev_pointfeat_cython/src/manager.cu").relative_to(bench.REPO)),
                    "unreleased_allocations": ["d_neighborhoodZ", "d_diffZ", "d_varZ"],
                    "allocate_lines": [144, 145, 146],
                    "get_features_free_lines": [172, 173, 174, 175],
                    "requested_bytes_per_feature_call": 3 * 30 * 4,
                    "physical_allocator_bytes": None,
                    "status": "source-confirmed missing cudaFree; compiled runtime live allocations need external tracing"},
        "host_wrappers": "official Cython wrappers allocate C++ manager objects with new and expose no __dealloc__; unreachable host object storage is reflected by process RSS but is not a live numerical tensor buffer",
        "radon": "generate_RING creates local ParallelBeam; its native texture allocation is temporary and absent from torch peak allocated",
        "peak_limit": "CUDA peaks below are PyTorch allocator peaks, not an external-CUDA-aware peak",
    }


@torch.no_grad()
def encode(core, variant: str, points: np.ndarray, check: bool = False) -> tuple:
    """원시 점군에서 공식 TI-RING을 만들고 유효 점 수를 돌려준다."""
    _, normalized = reference.official_points("mulran", points)
    count = len(normalized)
    if variant == "ring":
        bev = core.generate_bev_occ(normalized).float() if count else torch.zeros(
            (1, 120, 120), dtype=torch.float32, device="cuda")
        if check:
            reference.check_bev(bev.cpu().numpy(), normalized)
    else:
        bev = core.generate_bev_feats(normalized).float() if count >= 30 else torch.zeros(
            (6, 120, 120), dtype=torch.float32, device="cuda")
        if check:
            occupancy = core.generate_bev_occ(normalized).cpu().numpy() if count else np.zeros((1, 120, 120))
            reference.check_bev(occupancy, normalized, bev.cpu().numpy() if count >= 30 else None)
    _, tiring = core.generate_RING(bev)
    assert tiring.dtype == torch.float32
    return tiring, count


def cache(variant: str, n_map: int) -> None:
    """공식 전처리·BEV·TI-RING의 초기 지도 참조를 배치 1로 만든다."""
    configure()
    config = settings(variant)
    freeze(config)
    target = directory(variant, n_map)
    done = target / "complete.json"
    if done.exists():
        meta = bench.read(done)
        assert meta["config"] == config
        for name, digest in meta["files"].items():
            assert bench.sha(target / name) == digest
        return
    target.mkdir(parents=True, exist_ok=True)
    work = bench.Workload(bench.read(OUT / "protocol.json"), n_map)
    loader = reference.make_loader("mulran", "Sejong01")
    core = reference.import_ring_core()
    shape = (work.end, config["channels"], 120, 120)
    raw = np.lib.format.open_memmap(target / "raw.npy", mode="w+", dtype=np.float32, shape=shape)
    spectral = np.lib.format.open_memmap(target / "fft.npy", mode="w+", dtype=np.complex64, shape=shape)
    normalizer = torch.nn.InstanceNorm2d(1, affine=False, track_running_stats=False)
    count = np.zeros(work.end, np.int64)
    started = time.perf_counter()
    with torch.no_grad():
        for k, scan_id in enumerate(work.ids):
            value, count[k] = encode(core, variant, loader[int(scan_id)]["points"], check=True)
            raw[k] = value.cpu().numpy()
            spectral[k] = torch.fft.fft2(normalizer(value[None]), dim=-2, norm="ortho")[0].cpu().numpy()
            if k % 100 == 0:
                print("RING_CACHE", variant, k, work.end, round(time.perf_counter() - started, 1), flush=True)
    raw.flush()
    spectral.flush()
    np.save(target / "scan_ids.npy", work.ids)
    np.save(target / "point_counts.npy", count)
    bench.write(done, {"config": config, "frames": work.end, "batch": 1,
                      "raw_input_sha256": work.input_hash, "seconds": time.perf_counter() - started,
                      "native_allocation_audit": native_audit(),
                      "files": {name: bench.sha(target / name) for name in (
                          "raw.npy", "fft.npy", "scan_ids.npy", "point_counts.npy")}})


class Runtime:
    """공식 상관 점수를 바꾸지 않고 프레임별 FFT를 한 번만 계산한다."""

    def __init__(self, config: dict, work: bench.Workload) -> None:
        self.work, self.config = work, config
        self.variant = config["method"]
        self.core = reference.import_ring_core()
        target = directory(self.variant, work.n_map)
        meta = bench.read(target / "complete.json")
        assert meta["config"] == config
        for name, digest in meta["files"].items():
            assert bench.sha(target / name) == digest
        np.testing.assert_array_equal(np.load(target / "scan_ids.npy"), work.ids)
        self.source = np.load(target / "fft.npy", mmap_mode="r")
        self.raw_source = np.load(target / "raw.npy", mmap_mode="r")
        shape = (work.end, config["channels"], 120, 120)
        self.on_gpu = np.prod(shape) * 8 <= GPU_LIMIT
        self.bank = torch.empty(shape, dtype=torch.complex64,
                                device="cuda" if self.on_gpu else "cpu")
        self.normalizer = torch.nn.InstanceNorm2d(1, affine=False, track_running_stats=False)
        self.feature_calls = 0
        self.reset()

    def reset(self) -> None:
        """동일한 초기 지도와 지원 프레임을 복원한다."""
        self.n = self.work.initial
        for i in range(0, self.n, 512):
            end = min(i + 512, self.n)
            self.bank[i:end].copy_(torch.from_numpy(np.array(self.source[i:end], copy=True)))
        bench.sync()

    def encode(self, points: np.ndarray, check: bool = False) -> torch.Tensor:
        """실제 새 점군 하나를 공식 전처리·BEV·TI-RING으로 바꾼다."""
        value, count = encode(self.core, self.variant, points, check)
        if self.variant == "ringpp" and count >= 30:
            self.feature_calls += 1
        return value

    @torch.no_grad()
    def append(self, tiring: torch.Tensor) -> None:
        """새 프레임의 정규화·FFT를 실행하고 실제 색인 상태에 추가한다."""
        value = torch.fft.fft2(self.normalizer(tiring[None]), dim=-2, norm="ortho")[0]
        self.bank[self.n].copy_(value)
        self.n += 1

    def query(self, q: int) -> tuple:
        """모든 허용 과거 후보의 상관을 계산하여 공식 최종 점수를 반환한다."""
        n_prefix = q - 29
        scores = correlate_prefix_fast(self.bank, self.on_gpu, q, n_prefix, CHUNK, "cuda")
        j = int(np.argmax(scores))
        return j, float(scores[j] - 1.), n_prefix

    def memory(self) -> dict:
        """실제 FFT 뱅크를 세고 미검증 외부 할당은 따로 공개한다."""
        cpu = self.bank.untyped_storage().nbytes() if not self.on_gpu else 0
        gpu = torch.cuda.memory_allocated()
        complete = self.variant == "ring"
        return {"cpu": cpu, "gpu": gpu, "total": cpu + gpu if complete else None,
                "known_numerical_buffer_total": cpu + gpu, "publish_ready": complete,
                "model_gpu": 0, "fft_bank_on_gpu": bool(self.on_gpu),
                "breakdown": {"cpu_fft_bank": cpu, "gpu_fft_bank": self.bank.untyped_storage().nbytes() if self.on_gpu else 0},
                "logical_feature_payload_bytes_per_frame": self.config["channels"] * 120 * 120 * 4,
                "logical_feature_payload_bytes": self.n * self.config["channels"] * 120 * 120 * 4,
                "external_source_expected_live_requested_bytes": self.feature_calls * 360,
                "external_feature_encode_calls": self.feature_calls,
                "external_physical_allocation_bytes": None if not complete else 0,
                "external_native_audit": native_audit(),
                "timepoint": "after last arrival; FFT bank preallocated to N+140 observed/support capacity",
                "limitation": "RING++ full memory unpublished until external CUDA allocations are traced" if not complete else "temporary native CUDA peaks excluded from torch allocator peaks"}


def verify(runtime: Runtime, work: bench.Workload) -> dict:
    """첫 12도착의 특징·증분 상태·전수 검색을 공식 함수와 대조한다."""
    from _ring_official_pr import official_scores

    checks = []
    for i, points in enumerate(work.points[:bench.WARMUP]):
        t = work.initial + i
        tiring = runtime.encode(points, check=True)
        actual = tiring.cpu().numpy()
        expected = np.asarray(runtime.raw_source[t])
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-6)
        runtime.append(tiring)
        actual_fft = runtime.bank[t].cpu().numpy()
        np.testing.assert_allclose(actual_fft, runtime.source[t], atol=3e-5, rtol=3e-6)
        q = t - 15
        bank = torch.from_numpy(np.array(runtime.raw_source[:t + 1], copy=True))
        official = official_scores(runtime.core, bank, q, q - 29)
        fast = correlate_prefix_fast(runtime.bank, runtime.on_gpu, q, q - 29, CHUNK, "cuda")
        score_error = float(abs(official - fast).max())
        j = int(np.argmax(official))
        assert score_error < 1e-4
        assert j == int(np.argmax(fast))
        live = runtime.query(q)
        assert live[0] == j and abs(live[1] - (official[j] - 1)) < 1e-4
        checks.append({"arrival": t, "winner": j, "score_max_abs_error": score_error,
                       "tiring_max_abs_error": float(abs(actual - expected).max())})
        del bank, tiring
    runtime.reset()
    return {"status": "passed", "raw_regeneration": len(checks),
            "official_prefix_checks": checks,
            "cuda_occupancy_guard": "first12 exact CPU occupancy transcription; RING++ finite/occupied-cell support guard",
            "tolerance": "TI-RING atol2e-5 rtol2e-6; FFT atol3e-5 rtol3e-6; official-vs-hoisted scores abs1e-4; winner exact",
            "native_allocation_audit": native_audit()}


def measure(variant: str, n_map: int) -> None:
    """검증한 방법의 공통 96도착×3반복 비용을 독립 프로세스에서 잰다."""
    configure()
    config = settings(variant)
    freeze(config)
    target = OUT / "methods" / f"{variant}_n{n_map}.json"
    if target.exists():
        assert bench.read(target)["status"] == "passed"
        return
    spec = bench.read(OUT / "protocol.json")
    work = bench.Workload(spec, n_map)
    tick = time.perf_counter()
    runtime = Runtime(config, work)
    initial_s = time.perf_counter() - tick
    checked = verify(runtime, work)
    bench.write(OUT / "verification" / f"{variant}_n{n_map}.json", checked)
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
            result, query_ms = bench.timed(lambda: runtime.query(t - 15))
            rows.append({"repeat": repeat, "arrival": t, "query": t - 15,
                         "n_candidates": t - 44, "encode": encode_ms,
                         "state_update": state_ms, "update": encode_ms + state_ms,
                         "query_ms": query_ms, "processing": encode_ms + state_ms + query_ms,
                         "winner": result[0], "confidence": result[1], "rerank_candidates": result[2]})
        info = runtime.memory()
        info.update({"process_rss_before": before, "process_rss_after": bench.rss(),
                     "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     "cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved(),
                     "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
                     "cuda_peak_reserved": torch.cuda.max_memory_reserved()})
        memories.append(info)
    for i in range(bench.SAMPLES):
        values = [rows[k * bench.SAMPLES + i] for k in range(bench.REPEATS)]
        assert len({v["winner"] for v in values}) == 1
        assert max(v["confidence"] for v in values) - min(v["confidence"] for v in values) < 1e-7
    result = {
        "schema": "pdv2-table3-measurement-v1", "method": variant, "n_map": n_map,
        "status": "passed", "publish_ready": True, "memory_publish_ready": variant == "ring",
        "protocol_sha256": bench.sha(OUT / "protocol.json"),
        "method_protocol_sha256": bench.sha(OUT / f"{variant}_protocol.json"),
        "source_sha256": bench.sha(Path(__file__)), "sequence": bench.SEQUENCE, "seed": None,
        "candidate_range": [n_map, n_map + bench.SAMPLES - 1],
        "mean_candidates": n_map + (bench.SAMPLES - 1) / 2,
        "samples": bench.SAMPLES, "repeats": bench.REPEATS, "warmup": bench.WARMUP,
        "timings_ms": {key: bench.stats([row[field] for row in rows]) for key, field in (
            ("encode", "encode"), ("state_update", "state_update"), ("update", "update"),
            ("query", "query_ms"), ("processing", "processing"))},
        "resident_bytes": memories[-1], "memory_repeats": memories, "verification": checked,
        "raw_input_fixture_bytes": work.input_bytes, "raw_input_sha256": work.input_hash,
        "reference_fixture_bytes": int(runtime.source.nbytes + runtime.raw_source.nbytes),
        "initial_cached_state_build_s": initial_s,
        "reference_batch1_cache": bench.read(directory(variant, n_map) / "complete.json"),
        "raw_samples": rows,
        "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "platform": platform.platform(), "threads": 1},
        "scope": spec["scope"], "completed_unix": time.time(),
    }
    bench.write(target, result)
    bench.summarize()
    print("COMPLETE", variant, n_map, result["timings_ms"], flush=True)


def cpu_check() -> None:
    """GPU 없이 이름공간·공식 점수와 FFT 재사용의 수치 일치를 확인한다."""
    configure()
    from utils.circorr2 import circorr2
    from _pd_build_test_caches import _make_loader
    import joblib

    loader = _make_loader("mulran", "Sejong01")
    points = loader[0]["points"]
    _, normalized = reference.official_points("mulran", points)
    assert reference.occupancy_numpy(normalized).sum() > 0
    checks = []
    for channels in (1, 6):
        generator = torch.Generator().manual_seed(17 + channels)
        raw = torch.rand((17, channels, 120, 120), generator=generator)
        bank, on_gpu = precompute_fft_bank(raw, "cpu")
        fast = correlate_prefix_fast(bank, on_gpu, 16, 12, 7, "cpu")
        scorer = circorr2(is_circular=True, zero_mean_normalize=True)
        with torch.no_grad():
            _, expected, _ = scorer(raw[16:17].repeat(12, 1, 1, 1), raw[:12])
        difference = float(abs(fast - expected.numpy()).max())
        assert difference < 1e-5
        assert int(np.argmax(fast)) == int(expected.argmax())
        checks.append({"channels": channels, "score_max_error": difference})
    report = {"status": "passed", "gpu_work": False, "checks": checks,
              "effective_joblib_n_jobs_minus_one": joblib.effective_n_jobs(-1),
              "common_loader_namespace": "passed", "native_allocation_audit": native_audit(),
              "source_sha256": bench.sha(Path(__file__))}
    bench.write(OUT / "ring_cpu_check.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cache", "measure", "cpu_check"))
    parser.add_argument("--method", choices=("ring", "ringpp"), default="ring")
    parser.add_argument("--n-map", type=int, choices=(1000, 5000, 10000), default=1000)
    args = parser.parse_args()
    if args.action == "cpu_check":
        cpu_check()
    else:
        globals()[args.action](args.method, args.n_map)
