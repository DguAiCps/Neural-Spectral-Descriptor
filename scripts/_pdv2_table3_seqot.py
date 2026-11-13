#!/usr/bin/env python3
"""공식 centered SeqOT의 배치 1 증분 비용을 표 3 공통 규약으로 잰다.

``cache``는 선택된 두 가중치로 원시 연속 스캔의 참조 캐시를 만든다.
``measure``는 새 원시 스캔마다 투영과 FE를 한 번 실행하고 최근 3개
거리 영상·20개 부분 특징을 재사용한다. GeM은 완성된 키프레임마다
한 번만 실행한다. GPU 작업은 ring_baseline 안에서 직렬 실행한다.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import os
from pathlib import Path
import platform
import resource
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["SEQOT_EVAL_WINDOW"] = "centered"
os.environ["SEED"] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "_external/table3_faiss_cpu"), str(REPO),
               str(REPO / "src"), str(REPO / "scripts")]

import numpy as np
import torch

import _pdv2_table3_benchmark as common
import _seqot_official_train as official
from _seqot_official_prep import range_image

OUT = common.OUT
PROTOCOL = OUT / "seqot_protocol.json"
DIM = 256
RANGE_SHAPE = (32, 900)


def configure() -> None:
    """공통 단일 CPU 스레드·FP32 규약을 적용한다."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def prepare() -> dict:
    """선택된 centered 가중치와 어댑터 원천을 별도로 동결한다."""
    spec = common.read(OUT / "protocol.json")
    assert spec["sequence"] == "mulran/Sejong01"
    assert spec["samples"] == common.SAMPLES
    assert spec["repeats"] == common.REPEATS
    stage1, c1 = official._chosen("stage1")
    stage2, c2 = official._chosen("stage2")
    sources = [Path(__file__), Path(common.__file__),
               REPO / "scripts/_seqot_official_train.py",
               REPO / "scripts/_seqot_official_prep.py",
               REPO / "_external/SeqOT/modules/seqTransformerCat.py",
               REPO / "_external/SeqOT/modules/netvlad.py",
               REPO / "_external/SeqOT/modules/gem.py",
               REPO / "_external/SeqOT/tools/utils/utils.py",
               official.SELECTION, stage1, stage2]
    addon = {
        "schema": "pdv2-table3-seqot-protocol-v1", "revision": 1,
        "common_protocol_sha256": common.sha(OUT / "protocol.json"),
        "stage1": c1, "stage2": c2, "seed": 1,
        "input": "continuous raw scans; official range_projection; 32x900",
        "window": "FE [s-1,s+1]; GeM [s-10,s+9]; official boundary fallbacks and final zero subdescriptor row retained",
        "incremental": "project each new raw scan once; FE once per newly complete raw center; GeM once per completed keyframe; recent 3 range images and 20 subdescriptors reused",
        "retrieval": "official unnormalized 256D GeM; exact optimized CPU FAISS IndexFlatL2; j<=q-30; score=-sqrt(squared L2)",
        "batch": 1, "threads": 1, "tf32": False,
        "initialization": "separate batch1 raw-scan reference cache; copy only already available descriptors and recent streaming buffers into the initial state",
        "memory": "CPU FAISS allocated capacity + range/subdescriptor/keyframe rings + pending raw/keyframe ids + unique CPU model tensors; CUDA allocated includes both models and live buffers. Cached reference/input fixtures excluded and reported separately.",
        "sources": {str(p.relative_to(REPO)): common.sha(p) for p in sources},
    }
    if PROTOCOL.exists():
        assert common.read(PROTOCOL) == addon, "SeqOT 계측 원천 변경: 새 개정 필요"
    else:
        common.write(PROTOCOL, addon)
    return spec


def cache_dir(n_map: int) -> Path:
    """지도 크기별 독립 참조 캐시 위치다."""
    return OUT / "seqot_batch1" / f"n{n_map}"


def keyframe_ids(spec: dict, n_map: int) -> np.ndarray:
    """정답 없이 KISS-ICP 키프레임 원시 인덱스만 읽는다."""
    end = n_map + 44 + common.SAMPLES
    with np.load(REPO / spec["input_paths"]["cache"]) as saved:
        ids = saved["scan_ids"][:end].astype(np.int64)
    assert len(ids) == end and np.all(np.diff(ids) > 0)
    return ids


def gem_window(raw: int, n_raw: int) -> np.ndarray:
    """공식 N+1 부분 특징 행과 GeM 경계 대체 규칙을 쓴다."""
    return official.gem_windows([raw], lambda r: 0 <= r <= n_raw,
                                official.GEM_SEQLEN, "centered")[0]


def required_raw(raw: int, n_raw: int) -> int:
    """키프레임 특징을 확정하는 데 필요한 마지막 실제 원시 스캔이다."""
    if 11 <= raw < n_raw - 10:
        return raw + 10
    sub_ids = gem_window(raw, n_raw)
    sub_ids = sub_ids[sub_ids < n_raw]
    windows = official.subdesc_windows(
        sub_ids, lambda r: 0 <= r < n_raw, official.SEQL, "centered")
    return int(windows.max())


def cache(n_map: int) -> None:
    """원시 투영·FE·GeM을 배치 1로 실행한 참조 입력과 특징을 만든다."""
    from _pd_build_test_caches import _make_loader

    configure()
    spec = prepare()
    target = cache_dir(n_map)
    done = target / "complete.json"
    if done.exists():
        meta = common.read(done)
        assert meta["seqot_protocol_sha256"] == common.sha(PROTOCOL)
        for name, digest in meta["files"].items():
            assert common.sha(target / name) == digest
        return
    target.mkdir(parents=True, exist_ok=True)
    loader = _make_loader("mulran", "Sejong01")
    n_raw = len(loader)
    ids = keyframe_ids(spec, n_map)
    windows = np.stack([gem_window(int(s), n_raw) for s in ids])
    valid_sub = windows[windows < n_raw]
    last_sub = int(valid_sub.max())
    last_raw = max(required_raw(int(s), n_raw) for s in ids)
    started = time.perf_counter()
    ranges = np.lib.format.open_memmap(
        target / "range.npy", mode="w+", dtype=np.float32,
        shape=(last_raw + 1, *RANGE_SHAPE))
    for r in range(last_raw + 1):
        ranges[r] = range_image("mulran", loader[r]["points"])
        if r % 250 == 0:
            print("SEQOT_RANGE", n_map, r, last_raw + 1, flush=True)
    ranges.flush()
    del loader
    projection_s = time.perf_counter() - started
    fe = official._load_fe(official._chosen("stage1")[0], "cuda")
    gem = official._load_gem(official._chosen("stage2")[0], "cuda")
    sub = np.lib.format.open_memmap(target / "sub.npy", mode="w+",
                                  dtype=np.float32, shape=(last_sub + 1, DIM))
    common.sync()
    tick = time.perf_counter()
    with torch.no_grad():
        for r in range(last_sub + 1):
            w = official.subdesc_windows(
                [r], lambda x: 0 <= x < n_raw, 3, "centered")[0]
            sub[r] = fe(torch.from_numpy(np.asarray(ranges[w])[None]).cuda())[0].cpu().numpy()
            if r % 250 == 0:
                print("SEQOT_FE_BATCH1", n_map, r, last_sub + 1, flush=True)
    common.sync()
    sub_s = time.perf_counter() - tick
    sub.flush()
    final = np.lib.format.open_memmap(target / "final.npy", mode="w+",
                                    dtype=np.float32, shape=(len(ids), DIM))
    tick = time.perf_counter()
    with torch.no_grad():
        for k, w in enumerate(windows):
            rows = np.zeros((20, DIM), np.float32)
            valid = w < n_raw
            rows[valid] = sub[w[valid]]
            final[k] = gem(torch.from_numpy(rows[None]).cuda())[0, 0].cpu().numpy()
    common.sync()
    gem_s = time.perf_counter() - tick
    final.flush()
    np.save(target / "scan_ids.npy", ids)
    common.write(done, {
        "status": "passed", "seqot_protocol_sha256": common.sha(PROTOCOL),
        "n_map": n_map, "n_raw": n_raw, "keyframes": len(ids),
        "range_scans": last_raw + 1, "fe_centers": last_sub + 1,
        "batch": 1, "projection_and_disk_read_s": projection_s,
        "fe_s": sub_s, "gem_s": gem_s,
        "seconds": time.perf_counter() - started,
        "files": {name: common.sha(target / name)
                  for name in ("range.npy", "sub.npy", "final.npy", "scan_ids.npy")},
        "scope": "reference fixture for initial state plus measured arrivals; projection preparation includes disk IO; excluded from incremental timings",
    })


class SeqOTWorkload:
    """공통 도착 사이의 원시 스캔을 미리 읽어 디스크 I/O를 제외한다."""

    def __init__(self, spec: dict, n_map: int) -> None:
        from _pd_build_test_caches import _make_loader

        self.n_map = n_map
        self.initial = n_map + 44
        self.end = self.initial + common.SAMPLES
        self.ids = keyframe_ids(spec, n_map)
        target = cache_dir(n_map)
        self.meta = common.read(target / "complete.json")
        assert self.meta["seqot_protocol_sha256"] == common.sha(PROTOCOL)
        for name, digest in self.meta["files"].items():
            assert common.sha(target / name) == digest
        np.testing.assert_array_equal(np.load(target / "scan_ids.npy"), self.ids)
        self.n_raw = int(self.meta["n_raw"])
        self.ranges = np.load(target / "range.npy", mmap_mode="r")
        self.sub = np.load(target / "sub.npy", mmap_mode="r")
        self.final = np.load(target / "final.npy", mmap_mode="r")
        self.raw_batches = []
        loader = _make_loader("mulran", "Sejong01")
        self.input_bytes = 0
        all_hash, keyframe_hash = hashlib.sha256(), hashlib.sha256()
        for t in range(self.initial, self.end):
            batch = []
            for r in range(int(self.ids[t - 1]) + 1, int(self.ids[t]) + 1):
                points = np.ascontiguousarray(loader[r]["points"])
                batch.append((r, points))
                self.input_bytes += points.nbytes
                all_hash.update(np.int64(r).tobytes())
                all_hash.update(points.tobytes())
            keyframe_hash.update(batch[-1][1].tobytes())
            self.raw_batches.append(batch)
        del loader
        self.input_hash = all_hash.hexdigest()
        self.keyframe_input_hash = keyframe_hash.hexdigest()
        self.reference_fixture_bytes = int(self.ids.nbytes + self.ranges.nbytes
                                           + self.sub.nbytes + self.final.nbytes)


class SeqOT:
    """세 거리 영상·20 부분 특징을 재사용하는 공식 centered 추론이다."""

    def __init__(self, work: SeqOTWorkload) -> None:
        import faiss

        assert "AVX512" in faiss.get_compile_options()
        faiss.omp_set_num_threads(1)
        self.faiss, self.work = faiss, work
        self.fe = official._load_fe(official._chosen("stage1")[0], "cuda")
        self.gem = official._load_gem(official._chosen("stage2")[0], "cuda")
        self.models = torch.nn.ModuleList([self.fe, self.gem])
        self.range_ring = np.zeros((3, *RANGE_SHAPE), np.float32)
        self.sub_ring = np.zeros((20, DIM), np.float32)
        self.recent = np.zeros((46, DIM), np.float32)
        self.recent_ids = np.full(46, -1, np.int64)
        self.pending = np.full((11, 2), -1, np.int64)
        self.capture = False
        self.checks = []
        self.reset()

    def reset(self) -> None:
        """미래 원시 스캔을 사용하지 않는 초기 지도 상태를 복원한다."""
        self.n = self.work.initial
        self.last_raw = int(self.work.ids[self.n - 1])
        self.index = self.faiss.IndexFlatL2(DIM)
        self.index.add(np.zeros((self.work.n_map + common.SAMPLES - 1, DIM), np.float32))
        self.index_pointer = int(self.index.get_xb())
        self.index_capacity = int(self.index.codes.size())
        self.index.reset()
        self.index.add(np.asarray(self.work.final[:self.work.n_map - 1]))
        for r in range(self.last_raw - 2, self.last_raw + 1):
            self.range_ring[r % 3] = self.work.ranges[r]
        for r in range(self.last_raw - 20, self.last_raw):
            self.sub_ring[r % 20] = self.work.sub[r]
        self.recent.fill(0)
        self.recent_ids.fill(-1)
        self.pending.fill(-1)
        self.pending_n = 0
        for k in range(self.n - 45, self.n):
            s = int(self.work.ids[k])
            if required_raw(s, self.work.n_raw) <= self.last_raw:
                self.recent[k % 46] = self.work.final[k]
                self.recent_ids[k % 46] = k
            else:
                self._enqueue(k, s)
        self.counters = {"raw_projection": 0, "fe": 0, "gem": 0}
        self.checks = []

    def _enqueue(self, keyframe: int, raw: int) -> None:
        assert self.pending_n < len(self.pending)
        self.pending[self.pending_n] = (keyframe, raw)
        self.pending_n += 1

    def _complete(self, raw_arrival: int) -> None:
        while self.pending_n:
            k, s = map(int, self.pending[0])
            if required_raw(s, self.work.n_raw) > raw_arrival:
                break
            w = gem_window(s, self.work.n_raw)
            rows = self.sub_ring[w % 20].copy()
            rows[w == self.work.n_raw] = 0.
            with torch.no_grad():
                final = self.gem(torch.from_numpy(rows[None]).cuda())[0, 0].cpu().numpy()
            self.recent[k % 46] = final
            self.recent_ids[k % 46] = k
            self.counters["gem"] += 1
            if self.capture:
                np.testing.assert_allclose(final, self.work.final[k], atol=1e-7, rtol=1e-6)
                self.checks.append({"kind": "gem", "keyframe": k,
                                    "max_error": float(abs(final - self.work.final[k]).max())})
            self.pending[:self.pending_n - 1] = self.pending[1:self.pending_n]
            self.pending_n -= 1
            self.pending[self.pending_n] = -1

    def encode(self, batch: list) -> None:
        """신규 원시 스캔만 투영·FE 처리하고 완성 키프레임을 GeM한다."""
        for r, points in batch:
            assert r == self.last_raw + 1
            image = range_image("mulran", points)
            self.range_ring[r % 3] = image
            rows = self.range_ring[np.arange(r - 2, r + 1) % 3]
            with torch.no_grad():
                descriptor = self.fe(torch.from_numpy(rows[None]).cuda())[0].cpu().numpy()
            self.sub_ring[(r - 1) % 20] = descriptor
            self.counters["raw_projection"] += 1
            self.counters["fe"] += 1
            if self.capture:
                np.testing.assert_array_equal(image, self.work.ranges[r])
                np.testing.assert_allclose(descriptor, self.work.sub[r - 1], atol=1e-7, rtol=1e-6)
                self.checks.append({"kind": "fe", "raw": r,
                                    "max_error": float(abs(descriptor - self.work.sub[r - 1]).max())})
            self.last_raw = r
            self._complete(r)
        assert self.last_raw == int(self.work.ids[self.n])
        self._enqueue(self.n, self.last_raw)
        self._complete(self.last_raw)

    def append(self, unused=None) -> None:
        """새로 허용된 후보 하나를 기존 FAISS 색인에 추가한다."""
        k = self.n - 45
        assert self.recent_ids[k % 46] == k
        self.index.add(self.recent[k % 46:k % 46 + 1])
        assert int(self.index.get_xb()) == self.index_pointer
        self.n += 1

    def query(self, q: int) -> tuple:
        """공식 L2 승자와 음의 L2 최종 판정 점수를 반환한다."""
        assert self.index.ntotal == q - 29
        assert self.recent_ids[q % 46] == q
        distances, indices = self.index.search(self.recent[q % 46:q % 46 + 1], 1)
        return (int(indices[0, 0]),
                -float(np.sqrt(max(float(distances[0, 0]), 0.))), self.index.ntotal)

    def memory(self) -> dict:
        """FAISS 용량·재사용 버퍼·두 모델의 실제 저장소를 센다."""
        model = common.tensor_bytes(self.models)
        arrays = {"recent_range3": self.range_ring, "recent_sub20": self.sub_ring,
                  "recent_keyframes46": self.recent, "recent_keyframe_ids": self.recent_ids,
                  "pending_keyframe_raw_ids": self.pending}
        parts = {name: int(value.nbytes) for name, value in arrays.items()}
        parts.update(faiss_codes=self.index_capacity, model_cpu=model["cpu"])
        cpu = sum(parts.values())
        gpu = torch.cuda.memory_allocated()
        return {"cpu": cpu, "gpu": gpu, "total": cpu + gpu,
                "model_gpu": model["gpu"],
                "cuda_workspace_and_allocator_overhead": gpu - model["gpu"],
                "breakdown": parts, "index_size_bytes": int(self.index.codes.size()),
                "index_capacity_bytes": self.index_capacity,
                "capacity_note": "preallocated final live size; get_xb pointer remains unchanged",
                "logical_feature_payload_bytes_per_frame": DIM * 4,
                "logical_feature_payload_bytes": self.n * DIM * 4,
                "timepoint": "after last measured arrival; temporal raw buffers included"}


def verify(runtime: SeqOT, work: SeqOTWorkload) -> dict:
    """증분 투영·FE·GeM과 독립 배치 1 참조·접두 검색을 대조한다."""
    runtime.capture = True
    queries = []
    for i, batch in enumerate(work.raw_batches[:common.WARMUP]):
        t = work.initial + i
        runtime.encode(batch)
        runtime.append()
        value = runtime.query(t - 15)
        index = runtime.faiss.IndexFlatL2(DIM)
        index.add(np.asarray(work.final[:t - 44]))
        distances, indices = index.search(np.asarray(work.final[t - 15:t - 14]), 1)
        assert value[0] == int(indices[0, 0])
        score_error = abs(value[1] + float(np.sqrt(max(float(distances[0, 0]), 0.))))
        assert score_error < 2e-5
        queries.append({"arrival": t, "winner": value[0], "score_error": score_error})
    n_raw = sum(map(len, work.raw_batches[:common.WARMUP]))
    assert runtime.counters["raw_projection"] == runtime.counters["fe"] == n_raw
    result = {"status": "passed", "official_prefix_checks": queries,
              "streaming_reference_checks": runtime.checks.copy(),
              "new_raw_scans": n_raw, "counters": runtime.counters.copy(),
              "tolerance": "range exact; batch1 FE/GeM atol1e-7 rtol1e-6; winner exact; L2 score abs2e-5",
              "reference_cache": str(cache_dir(work.n_map).relative_to(REPO)),
              "reference_cache_manifest_sha256": common.sha(cache_dir(work.n_map) / "complete.json")}
    runtime.capture = False
    runtime.reset()
    return result


def measure(n_map: int) -> None:
    """공통 96도착×3반복을 수행하고 검증된 표 3 행을 저장한다."""
    configure()
    spec = prepare()
    target = OUT / "methods" / f"seqot_n{n_map}.json"
    if target.exists():
        assert common.read(target)["status"] == "passed"
        return
    work = SeqOTWorkload(spec, n_map)
    tick = time.perf_counter()
    runtime = SeqOT(work)
    common.sync()
    initial_s = time.perf_counter() - tick
    checked = verify(runtime, work)
    common.write(OUT / "verification" / f"seqot_n{n_map}.json", checked)
    rows, memory = [], []
    for repeat in range(common.REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rss_before = common.rss()
        for i, batch in enumerate(work.raw_batches):
            t = work.initial + i
            before = runtime.counters.copy()
            _, encode_ms = common.timed(lambda: runtime.encode(batch))
            _, state_ms = common.timed(runtime.append)
            result, query_ms = common.timed(lambda: runtime.query(t - 15))
            rows.append({"repeat": repeat, "arrival": t, "query": t - 15,
                         "n_candidates": t - 44, "new_raw_scans": len(batch),
                         "fe_evaluations": runtime.counters["fe"] - before["fe"],
                         "gem_evaluations": runtime.counters["gem"] - before["gem"],
                         "encode": encode_ms, "state_update": state_ms,
                         "update": encode_ms + state_ms, "query_ms": query_ms,
                         "processing": encode_ms + state_ms + query_ms,
                         "winner": result[0], "confidence": result[1],
                         "rerank_candidates": result[2]})
        info = runtime.memory()
        info.update({"process_rss_before": rss_before, "process_rss_after": common.rss(),
                     "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     "cuda_allocated": torch.cuda.memory_allocated(),
                     "cuda_reserved": torch.cuda.memory_reserved(),
                     "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
                     "cuda_peak_reserved": torch.cuda.max_memory_reserved()})
        memory.append(info)
    for i in range(common.SAMPLES):
        values = [rows[k * common.SAMPLES + i] for k in range(common.REPEATS)]
        assert len({v["winner"] for v in values}) == 1
        assert max(v["confidence"] for v in values) - min(v["confidence"] for v in values) < 1e-7
        assert all(v["new_raw_scans"] == v["fe_evaluations"] for v in values)
    result = {
        "schema": "pdv2-table3-measurement-v1", "method": "seqot", "n_map": n_map,
        "status": "passed", "publish_ready": True,
        "protocol_sha256": common.sha(OUT / "protocol.json"),
        "method_protocol_sha256": common.sha(PROTOCOL),
        "source_sha256": common.sha(Path(__file__)),
        "sequence": common.SEQUENCE, "seed": 1,
        "candidate_range": [n_map, n_map + common.SAMPLES - 1],
        "mean_candidates": n_map + (common.SAMPLES - 1) / 2,
        "samples": common.SAMPLES, "repeats": common.REPEATS, "warmup": common.WARMUP,
        "timings_ms": {key: common.stats([r[field] for r in rows]) for key, field in (
            ("encode", "encode"), ("state_update", "state_update"), ("update", "update"),
            ("query", "query_ms"), ("processing", "processing"))},
        "resident_bytes": memory[-1], "memory_repeats": memory, "verification": checked,
        "raw_input_fixture_bytes": work.input_bytes, "raw_input_sha256": work.input_hash,
        "shared_keyframe_input_sha256": work.keyframe_input_hash,
        "reference_fixture_bytes": work.reference_fixture_bytes,
        "initial_cached_state_build_s": initial_s,
        "reference_batch1_cache": work.meta,
        "raw_samples": rows,
        "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "platform": platform.platform(), "threads": 1},
        "scope": spec["scope"] + " SeqOT additionally consumes the continuous raw scans between the same keyframe arrivals; their disk IO is also excluded, and each new scan is projected/FE-encoded once.",
        "completed_unix": time.time(),
    }
    common.write(target, result)
    common.summarize()
    print("COMPLETE", "seqot", n_map, result["timings_ms"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "cache", "measure"))
    parser.add_argument("--n-map", type=int, choices=(1000, 5000, 10000), default=1000)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        globals()[args.action](args.n_map)
