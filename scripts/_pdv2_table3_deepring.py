#!/usr/bin/env python3
"""표2 DeepRING 재구현의 실제 단일 도착·접두 전체 검색 비용을 잰다."""
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

os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import numpy as np
import torch
import _pdv2_table3_benchmark as bench
import _deepring_reimpl as reference
from _deepring_reimpl_model import SpecBank, corr_max_direct, ring_fft, UNET_FILE

OUT = bench.OUT


def configure() -> None:
    """공통 단일 스레드·정밀도 규약을 적용한다."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    reference.scorer_fp32_guard("cuda")
    # 저장소의 src/utils보다 공식 RING의 namespace가 먼저 등록되어야 한다.
    reference.rol.import_ring()
    # 공통 로더의 사용하지 않는 SC helper import도 같은 프로세스에서 해석한다.
    name = "utils.cyclic_shift_distance"
    if name not in sys.modules:
        source = bench.REPO / "src/utils/cyclic_shift_distance.py"
        module_spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)


def settings() -> dict:
    """표2 시드1의 선택된 모델과 변경 없는 재구현 경로를 동결한다."""
    selection_path = bench.REPO / "results/pdv2_deepring_reimpl/train_selection.json"
    selection = bench.read(selection_path)
    checkpoint = bench.REPO / "_external/deepring_reimpl_pdv2/runs/s1/epoch_01.pth"
    assert selection["finished"] and selection["best"]["epoch"] == 1
    expected = "7c8c60e43c6e573af87646bf645e55b97973ddbdcae49c98a848ce6c1384571b"
    assert selection["best"]["sha256"] == expected == bench.sha(checkpoint)
    sources = [Path(__file__), Path(reference.__file__),
               bench.REPO / "scripts/_deepring_reimpl_model.py",
               bench.REPO / "scripts/_ring_official_lib.py",
               bench.REPO / "scripts/_causal_ring_pr.py",
               bench.REPO / "src/utils/cyclic_shift_distance.py",
               bench.REPO / "_external/RING/utils/core.py",
               bench.REPO / "_external/RING/utils/circorr2.py",
               bench.REPO / "_external/RING/utils/config.py", UNET_FILE]
    return {"schema": "pdv2-table3-deepring-protocol-v1",
            "parent_protocol_sha256": bench.sha(OUT / "protocol.json"),
            "checkpoint": str(checkpoint.relative_to(bench.REPO)),
            "checkpoint_sha256": expected, "selected_epoch": 1,
            "selection_sha256": bench.sha(selection_path), "seed": 1,
            "implementation": "Table2 paper-based DeepRING reimplementation; not official DeepRING code",
            "sources": {str(path.relative_to(bench.REPO)): bench.sha(path) for path in sources},
            "dataset_table": reference.rol.DATASET_TABLE["mulran"],
            "encoder": "official RING points/occupancy/sinogram; unchanged selected DeepRING UNet and row DFT; batch1",
            "update": "actual new complex64 SpecBank row from unchanged ring_fft; allocated bank includes observed support frames",
            "retrieval": "unchanged SpecBank.corr and BlockScorer(block=1), exhaustive eligible prefix, W*C+b then first argmax and ledger minus-one offset",
            "encoding_adapter": "Binary occupancy stays on GPU between unchanged generate_bev_occ and Sino; reference pack/unpack identity is verified outside timing; CPU transcription guard is verification only",
            "threads": 1, "tf32_matmul": False, "tf32_cudnn": False,
            "pairs_per_correlation_chunk": 4096}


def directory(n_map: int) -> Path:
    """참조 표현과 메타데이터 저장 위치다."""
    return OUT / "deepring_batch1" / f"n{n_map}"


def freeze(config: dict) -> None:
    """기존 공통 프로토콜과 별도로 방법 소스를 고정한다."""
    target = OUT / "deepring_protocol.json"
    if target.exists():
        assert bench.read(target) == config, "동결한 DeepRING 계측 소스가 변경됐다"
    else:
        bench.write(target, config)


def cache(n_map: int) -> None:
    """동일 원시 입력을 기존 드라이버의 캐시 경로로 batch1 재현한다."""
    configure()
    config = settings()
    spec = bench.read(OUT / "protocol.json")
    target = directory(n_map)
    done = target / "complete.json"
    if done.exists():
        meta = bench.read(done)
        assert meta["config"] == config
        for name, digest in meta["files"].items():
            assert bench.sha(target / name) == digest
        return
    target.mkdir(parents=True, exist_ok=True)
    from _pd_build_test_caches import _make_loader

    work = bench.Workload(spec, n_map)
    loader = _make_loader("mulran", "Sejong01")
    model = reference.load_model(bench.REPO / config["checkpoint"], "cuda")
    sino = reference.Sino("gpu")
    packed = np.lib.format.open_memmap(target / "packed.npy", mode="w+", dtype=np.uint8,
                                     shape=(work.end, 120, 15))
    raw = np.lib.format.open_memmap(target / "raw.npy", mode="w+", dtype=np.float32,
                                  shape=(work.end, 1, 120, 120))
    spectral = np.lib.format.open_memmap(target / "spec.npy", mode="w+", dtype=np.complex64,
                                       shape=(work.end, 1, 120, 120))
    started = time.perf_counter()
    with torch.no_grad():
        for i, scan_id in enumerate(work.ids):
            occupancy, _ = reference.occupancy_bev(sino.core, "mulran", loader[int(scan_id)]["points"], "gpu")
            packed[i] = np.packbits(occupancy, axis=-1)
            embedding = model.embed(sino(reference.unpack(packed[i:i+1]).cuda()))
            raw[i] = embedding[0].cpu().numpy()
            spectral[i] = ring_fft(embedding)[0].cpu().numpy()
            if i % 100 == 0:
                print("DEEPRING_CACHE", i, work.end, round(time.perf_counter()-started, 1), flush=True)
    for array in (packed, raw, spectral):
        array.flush()
    np.save(target / "scan_ids.npy", work.ids)
    bench.write(done, {"config": config, "frames": work.end, "batch": 1,
                      "raw_input_sha256": work.input_hash,
                      "seconds": time.perf_counter()-started,
                      "W_b": list(model.wb()), "files": {name: bench.sha(target / name) for name in
                      ("packed.npy", "raw.npy", "spec.npy", "scan_ids.npy")}})


class Runtime:
    """한 도착씩 특징을 추가하고 고정 분류 점수로 과거 후보를 전수 검색한다."""

    def __init__(self, config: dict, work: bench.Workload):
        self.work = work
        self.model = reference.load_model(bench.REPO / config["checkpoint"], "cuda")
        self.sino = reference.Sino("gpu")
        self.source = np.load(directory(work.n_map) / "spec.npy", mmap_mode="r")
        self.bank = SpecBank(work.end, "cuda", pairs=config["pairs_per_correlation_chunk"])
        self.W, self.b = self.model.wb()
        self.scorer = reference.BlockScorer(self.bank, self.W, self.b, work.end, 1)
        self.n = work.initial
        self.reset()

    def reset(self) -> None:
        """계측 밖에서 같은 초기 표현으로 복원한다."""
        self.n = self.work.initial
        self.bank.bank[:self.n].copy_(torch.from_numpy(np.array(self.source[:self.n], copy=True)).cuda())
        self.scorer.cache.clear()
        bench.sync()

    @torch.no_grad()
    def occupancy(self, points: np.ndarray) -> torch.Tensor:
        """공식 전처리와 CUDA 점유 영상을 만든다."""
        _, normalized = reference.rol.official_points("mulran", points)
        if len(normalized) == 0:
            return torch.zeros((1, 1, 120, 120), dtype=torch.float32, device="cuda")
        return self.sino.core.generate_bev_occ(normalized).reshape(1, 1, 120, 120).float()

    @torch.no_grad()
    def encode(self, points: np.ndarray) -> torch.Tensor:
        """점군부터 기존 UNet·row DFT의 실제 한 프레임 표현까지다."""
        return self.model.embed(self.sino(self.occupancy(points)))

    @torch.no_grad()
    def append(self, embedding: torch.Tensor) -> None:
        """실제 새 행의 정규화·FFT를 계산해서 뱅크에 삽입한다."""
        self.bank.put(self.n, embedding)
        self.n += 1

    @torch.no_grad()
    def query(self, query: int) -> tuple:
        """기존 분류 점수와 first-max tie 규약을 유지한다."""
        scores = self.scorer(query, query-29)
        winner = int(np.argmax(scores))
        return winner, float(scores[winner])-1., query-29

    def memory(self) -> dict:
        """라이브 모델·복소 뱅크·작업공간을 CUDA 총량에서 한 번씩 센다."""
        model = bench.tensor_bytes(self.model)
        gpu = torch.cuda.memory_allocated()
        bank = self.bank.bank.untyped_storage().nbytes()
        return {"cpu": model["cpu"], "gpu": gpu, "total": model["cpu"]+gpu,
                "model_gpu": model["gpu"], "bank_gpu": bank,
                "cuda_workspace_and_allocator_overhead": gpu-model["gpu"]-bank,
                "breakdown": {"cpu_model": model["cpu"], "gpu_model": model["gpu"],
                              "gpu_complex_bank": bank, "gpu_live_workspace": gpu-model["gpu"]-bank},
                "logical_feature_payload_bytes_per_frame": 120*120*4,
                "logical_feature_payload_bytes": self.n*120*120*4,
                "allocated_bank_capacity": self.work.end,
                "timepoint": "after last measured arrival",
                "capacity_note": "complex64 search bank uses fixed final observed capacity, including 45 support frames; raw descriptor/audit cache and common input fixture excluded; model/bank already included in allocated GPU total"}


@torch.no_grad()
def verify(runtime: Runtime, work: bench.Workload) -> tuple:
    """기존 캐시·분류 검색·직접 상관 연산과 실제 도착 경로를 대조한다."""
    target = directory(work.n_map)
    raw = np.load(target / "raw.npy", mmap_mode="r")
    packed = np.load(target / "packed.npy", mmap_mode="r")
    reference_bank = SpecBank(work.end, "cuda")
    reference_bank.put(0, torch.from_numpy(np.array(raw, copy=True)).cuda())
    # 단일 도착끼리 exact 비교한다. FFT의 배치 크기가 커질 때의 별도 오차도 기록한다.
    max_batched_fft_error = float(abs(reference_bank.bank.cpu().numpy()-runtime.source).max())
    reference_bank.bank.copy_(torch.from_numpy(np.array(runtime.source, copy=True)).cuda())
    reference_scorer = reference.BlockScorer(reference_bank, runtime.W, runtime.b, work.end, 1)
    expected = []
    for i in range(bench.SAMPLES):
        query = work.initial+i-15
        scores = reference_scorer(query, query-29)
        winner = int(np.argmax(scores))
        expected.append((winner, float(scores[winner])-1.))
    checks = []
    for i in range(bench.WARMUP):
        arrival = work.initial+i
        occupancy = runtime.occupancy(work.points[i])[0, 0].cpu().numpy()
        np.testing.assert_array_equal(occupancy, np.unpackbits(packed[arrival], axis=-1, count=120))
        embedding = runtime.encode(work.points[i])
        error = float(abs(embedding[0].cpu().numpy()-raw[arrival]).max())
        np.testing.assert_allclose(embedding[0].cpu().numpy(), raw[arrival], atol=1e-7, rtol=1e-6)
        runtime.append(embedding)
        fft_error = float(abs(runtime.bank.bank[arrival].cpu().numpy()-runtime.source[arrival]).max())
        np.testing.assert_allclose(runtime.bank.bank[arrival].cpu().numpy(), runtime.source[arrival], atol=2e-5, rtol=1e-5)
        actual = runtime.query(arrival-15)
        assert actual[:2] == expected[i]
        checks.append({"arrival": arrival, "descriptor_max_error": error,
                       "fft_max_error": fft_error, "winner": actual[0], "score_error": 0.})
    # 기존 재구현이 쓰는 공식 RING double 직접 상관과 128후보를 독립 대조한다.
    query = work.initial-15
    raw_query = torch.from_numpy(np.array(raw[query:query+1], copy=True)).double()
    raw_gallery = torch.from_numpy(np.array(raw[:128], copy=True)).double()
    direct = corr_max_direct(raw_query, raw_gallery)[0].numpy()
    fast = reference_bank.corr([query], 128)[0].double().cpu().numpy()
    relative = float(np.max(abs(direct-fast))/max(np.max(abs(direct)), 1e-30))
    assert relative < 1e-4
    assert int(np.argmax(fast)) == int(np.argmax(direct))
    del reference_bank, reference_scorer, raw_query, raw_gallery, embedding
    runtime.reset()
    gc.collect()
    torch.cuda.empty_cache()
    return {"status": "passed", "raw_regeneration": len(checks), "fullprefix_checks": checks,
            "reference_fullprefix_queries": len(expected), "batch1_vs_batched_fft_max_error": max_batched_fft_error,
            "official_direct_correlation": {"candidates": 128, "relative_max_error": relative,
                                            "winner_exact": True},
            "tolerance": "occupancy exact; batch1 descriptor atol1e-7 rtol1e-6; winner and final logit exact; independent official double correlation rel<1e-4"}, expected


def measure(n_map: int) -> None:
    """독립 프로세스에서 검증 뒤 실제 96도착을 세 번 계측한다."""
    configure()
    config = settings()
    freeze(config)
    meta = bench.read(directory(n_map) / "complete.json")
    assert meta["config"] == config
    for name, digest in meta["files"].items():
        assert bench.sha(directory(n_map) / name) == digest
    spec = bench.read(OUT / "protocol.json")
    work = bench.Workload(spec, n_map)
    assert work.input_hash == meta["raw_input_sha256"]
    tick = time.perf_counter()
    runtime = Runtime(config, work)
    initial_build_s = time.perf_counter()-tick
    verification, expected = verify(runtime, work)
    rows, memory = [], []
    for repeat in range(bench.REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rss_before = bench.rss()
        for i, points in enumerate(work.points):
            arrival = work.initial+i
            embedding, encode_ms = bench.timed(lambda: runtime.encode(points))
            _, state_ms = bench.timed(lambda: runtime.append(embedding))
            result, query_ms = bench.timed(lambda: runtime.query(arrival-15))
            assert result[:2] == expected[i]
            rows.append({"repeat": repeat, "arrival": arrival, "query": arrival-15,
                         "n_candidates": arrival-44, "encode": encode_ms, "state_update": state_ms,
                         "update": encode_ms+state_ms, "query_ms": query_ms,
                         "processing": encode_ms+state_ms+query_ms,
                         "winner": result[0], "confidence": result[1], "rerank_candidates": result[2]})
        info = runtime.memory()
        info.update({"process_rss_before": rss_before, "process_rss_after": bench.rss(),
                     "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     "cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved(),
                     "cuda_peak_allocated": torch.cuda.max_memory_allocated(), "cuda_peak_reserved": torch.cuda.max_memory_reserved()})
        memory.append(info)
    verification["measured_queries_checked"] = len(rows)
    verification["W_b"] = [runtime.W, runtime.b]
    verification["three_repeats_exact"] = True
    result = {"schema": "pdv2-table3-measurement-v1", "method": "deepring", "n_map": n_map,
              "status": "passed", "publish_ready": True, "protocol_sha256": bench.sha(OUT / "protocol.json"),
              "source_sha256": bench.sha(Path(__file__)), "method_protocol": config,
              "method_protocol_sha256": bench.sha(OUT / "deepring_protocol.json"),
              "sequence": bench.SEQUENCE, "seed": 1,
              "candidate_range": [n_map, n_map+bench.SAMPLES-1],
              "mean_candidates": n_map+(bench.SAMPLES-1)/2,
              "samples": bench.SAMPLES, "repeats": bench.REPEATS, "warmup": bench.WARMUP,
              "timings_ms": {key: bench.stats([row[field] for row in rows]) for key, field in
                             (("encode", "encode"), ("state_update", "state_update"), ("update", "update"),
                              ("query", "query_ms"), ("processing", "processing"))},
              "resident_bytes": memory[-1], "memory_repeats": memory, "verification": verification,
              "raw_input_fixture_bytes": work.input_bytes, "raw_input_sha256": work.input_hash,
              "initial_cached_state_build_s": initial_build_s, "raw_samples": rows,
              "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "platform": platform.platform(), "threads": 1},
              "scope": spec["scope"], "completed_unix": time.time()}
    bench.write(OUT / "methods" / f"deepring_n{n_map}.json", result)
    print("COMPLETE", "deepring", n_map, result["timings_ms"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cache", "measure"))
    parser.add_argument("--n-map", type=int, choices=(1000, 5000, 10000), default=1000)
    args = parser.parse_args()
    globals()[args.action](args.n_map)
