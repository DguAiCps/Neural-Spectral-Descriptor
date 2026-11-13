#!/usr/bin/env python3
"""현재 주모델의 실제 키프레임 추가와 최종 질의 비용을 측정한다."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "_external/table3_faiss_cpu"), str(REPO), str(REPO / "src"), str(REPO / "scripts")]
OUT = REPO / "results/pdv2_table3_20261029"
SEQUENCE = "mulran/Sejong01"
SAMPLES = 96
REPEATS = 3
WARMUP = 12


def read(path: Path) -> dict:
    """JSON 원천을 읽는다."""
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    """부분 파일을 노출하지 않고 JSON을 갱신한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path: Path) -> str:
    """파일의 SHA256을 반환한다."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(values: list) -> dict:
    """밀리초 표본의 평균과 백분위수를 반환한다."""
    a = np.asarray(values, np.float64)
    return {"mean": float(a.mean()), "p50": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "n": len(a)}


def sync() -> None:
    """GPU 완료까지 기다린다."""
    torch.cuda.synchronize()


def timed(fn):
    """동기화된 wall time과 반환값을 구한다."""
    sync()
    started = time.perf_counter_ns()
    value = fn()
    sync()
    return value, (time.perf_counter_ns() - started) / 1e6


def rss() -> int:
    """현재 프로세스 실제 RSS 바이트다."""
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def tensor_bytes(module) -> dict:
    """가중치·등록 버퍼의 고유 저장소를 한 번씩 센다."""
    seen, result = set(), {"cpu": 0, "gpu": 0}
    for tensor in list(module.parameters()) + list(module.buffers()):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            result["gpu" if tensor.is_cuda else "cpu"] += storage.nbytes()
    return result


def prepare() -> dict:
    """정답을 보지 않고 공통 workload와 방법을 동결한다."""
    import _pdv2_spectral_keys as keys
    import faiss
    assert "AVX512" in faiss.get_compile_options()
    source = read(REPO / "results/pdv2_spectral_whitening_20261026/fixed_test/freeze.json")
    _, _, paths, _ = keys.base.exp.old.base.inputs("val", SEQUENCE)
    checkpoint = REPO / "_external/bevplace_official_pdv2/runs/Oct12_05-12-15/model_best.pth.tar"
    assert sha(checkpoint) == "b0dd3a16c329bd8fc847e4f44ad5b848d98e821772cd6e39097f41940f90a96f"
    spec = {"schema": "pdv2-table3-protocol-v1", "sequence": SEQUENCE,
            "map_sizes": [1000, 5000, 10000], "samples": SAMPLES,
            "repeats": REPEATS, "warmup": WARMUP, "seed": 1,
            "map_rule": "N is the first eligible gallery size; the 96 sequential arrivals use N..N+95 candidates. q=t-15, j<=q-30. The initial prefix contains N+44 observed frames.",
            "scope": "batch1; same raw incoming scans and KISS poses; disk I/O and shared KISS-ICP excluded for every method; no retraining, tuning, or accuracy selection.",
            "update": "raw scan encoding, actual state append, affected temporal-key recomputation, bank/index update; initial cached map construction excluded and reported separately",
            "query": "eligible-prefix retrieval, reranking and final frozen confidence",
            "memory": "CPU unique resident numerical buffers including actual allocated array capacity, graph/pose/context/index/model/W plus measured CUDA allocated bytes including live workspace; logical payload, process RSS and CUDA allocator counters reported separately. Shared preloaded raw-input fixture excluded from buffer total and explicitly reported. CUDA allocator reserved not added to tensor bytes.",
            "threads": 1, "cuda_sync": True, "tf32_matmul": False, "tf32_cudnn": False, "revision": 2, "faiss": {"version": faiss.__version__, "compile_options": faiss.get_compile_options(), "algorithm": "official IndexFlatL2 unchanged; separately installed same-version optimized CPU wheel"},
            "nsd_models": source["models"], "nsd_confidence": source["confidence"],
            "nsd_fraction": 0.25, "bev_checkpoint": str(checkpoint.relative_to(REPO)),
            "input_paths": {k: str(v.relative_to(REPO)) for k, v in paths.items()},
            "sources": {str(p.relative_to(REPO)): sha(p) for p in [
                Path(__file__), REPO / "_external/table3_faiss_cpu/faiss/_swigfaiss_avx512.cpython-312-x86_64-linux-gnu.so", paths["cache"], paths["spec"], checkpoint,
                REPO / "results/pdv2_spectral_whitening_20261026/fixed_test/freeze.json",
                REPO / "results/pdv2_spectral_whitening_20261026/statistics.npz",
                REPO / "scripts/_pdv2_spectral_keys.py", REPO / "scripts/_pdv2_input_alignment_lib.py",
                REPO / "scripts/_pdv2_sequence_fixed_test.py", REPO / "scripts/_xs_eval_lib.py",
                REPO / "src/gnn/model.py", REPO / "src/gnn/geometry_interaction.py",
                REPO / "src/retrieval/sequence_evidence.py", REPO / "src/retrieval/spectral_evidence.py"]}}
    path = OUT / "protocol.json"
    if path.exists():
        assert read(path) == spec, "계측 소스 변경: 새 protocol/revision을 명시해야 한다"
    else:
        write(path, spec)
    return spec


class Workload:
    """초기 상태와 이후 실제 도착 점군을 준비한다. 정답 자세는 읽지 않는다."""
    def __init__(self, spec: dict, n_map: int):
        from _pd_build_test_caches import _make_loader
        self.n_map = n_map
        self.initial = n_map + 44
        self.end = self.initial + SAMPLES
        with np.load(REPO / spec["input_paths"]["cache"]) as z:
            self.ids = z["scan_ids"][:self.end].copy()
            self.raw = z["descriptors"][:self.end].astype(np.float32)
            self.poses = z["poses"][:self.end].copy()
        with np.load(REPO / spec["input_paths"]["spec"]) as z:
            np.testing.assert_array_equal(z["scan_ids"][:self.end], self.ids)
            self.coeff = (z["sk_re"][:self.end, :, :8] + 1j * z["sk_im"][:self.end, :, :8]).astype(np.complex64)
        loader = _make_loader(*SEQUENCE.split("/"))
        self.points = [np.ascontiguousarray(loader[int(i)]["points"]) for i in self.ids[self.initial:]]
        self.input_bytes = sum(x.nbytes for x in self.points)
        self.input_hash = hashlib.sha256(b"".join(x.tobytes() for x in self.points)).hexdigest()
        del loader


def flat_bank(bank: tuple) -> list:
    """중첩 뱅크를 배열 다섯 개로 펼친다."""
    return [bank[0], bank[1], bank[2], bank[3][0], bank[3][1]]


def bank_tuple(arrays: list) -> tuple:
    """배열 다섯 개를 참조 평가의 뱅크로 묶는다."""
    return arrays[0], arrays[1], arrays[2], (arrays[3], arrays[4])


class NSD:
    """고정 시간 모델의 증분 추론. 최종 점수는 참조 함수를 그대로 쓴다."""
    def __init__(self, spec: dict, work: Workload, seed: int = 1):
        import _pdv2_spectral_keys as keys
        import _pdv2_spectral_whitening as white
        from _pd_build_test_caches import _effective_config, _make_desc_encoder
        self.keys_module, self.white = keys, white
        self.net = keys.factory(288).cuda().eval()
        ck = spec["nsd_models"][str(seed)]
        assert sha(REPO / ck["checkpoint"]) == ck["sha256"]
        self.net.load_state_dict(torch.load(REPO / ck["checkpoint"], map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
        self.net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(keys.OUT / "remetric_octave288.npy")).float().cuda(), requires_grad=False)
        self.encoder = _make_desc_encoder(_effective_config(), "mulran", "cuda").eval()
        self.matrix = np.load(white.OUT / "statistics.npz")["full"]
        self.conf = spec["nsd_confidence"][str(seed)]
        self.work = work
        self.capacity = work.end
        self.n = work.initial
        self.keys = np.empty((self.capacity, 416), np.float32)
        self.coeff = np.empty((self.capacity, 16, 8), np.complex64)
        self.poses = np.empty((self.capacity, 4, 4), work.poses.dtype)
        self.travel = np.empty(self.capacity, np.float64)
        self.recent = np.empty((31, 288), np.float32)
        self.original = self.allocate_bank()
        self.corrected = self.allocate_bank()
        self.reset()

    def allocate_bank(self) -> list:
        """실배열 용량을 고정하고 실제 메모리에 포함한다."""
        sample = flat_bank(self.white.variant_inputs(np.zeros((1, 16, 8), np.complex64), 8))
        return [np.empty((self.capacity, *a.shape[1:]), a.dtype) for a in sample]

    def reset(self) -> None:
        """측정 밖에서 동일한 초기 지도 상태를 복원한다."""
        n = self.work.initial
        self.n = n
        self.keys[:n] = self.keys_module.make_keys(self.net, self.work.raw[:n], self.work.poses[:n])
        self.coeff[:n] = self.work.coeff[:n]
        self.poses[:n] = self.work.poses[:n]
        self.travel[:n] = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.poses[:n, :3, 3], axis=0), axis=1))]
        self.recent[:] = self.work.raw[n - 31:n]
        for target, values in ((self.original, self.coeff[:n]), (self.corrected, self.white.transform(self.coeff[:n], self.matrix))):
            for dst, src in zip(target, flat_bank(self.white.variant_inputs(values, 8))):
                dst[:n] = src

    def encode(self, points: np.ndarray) -> tuple:
        """현재288키와0절단 BEV 보존 계수를 원시 점군에서 만든다."""
        from _pbev_branch_fusion import BP
        from encoding.phase_features import _phase_sketch, _pool_rows
        with torch.no_grad():
            raw = self.encoder.encode_points(points).float().cpu().numpy()
        bev, _ = BP.project(points, keep_intensity=False)
        coeff = _phase_sketch(_pool_rows(np.maximum(bev, 0.), 16), 8)
        return raw, coeff

    def append(self, raw: np.ndarray, coeff: np.ndarray, pose: np.ndarray) -> None:
        """새 도착 하나를 추가하고 영향받는 마지막11개 키를 재계산한다."""
        t = self.n
        self.poses[t] = pose
        self.travel[t] = self.travel[t-1] + np.linalg.norm(pose[:3, 3] - self.poses[t-1, :3, 3])
        self.coeff[t] = coeff
        self.recent[:-1] = self.recent[1:]
        self.recent[-1] = raw
        local = self.keys_module.make_keys(self.net, self.recent, self.poses[t-30:t+1])
        self.keys[t-10:t+1] = local[-11:]
        for target, values in ((self.original, coeff[None]), (self.corrected, self.white.transform(coeff[None], self.matrix))):
            for dst, src in zip(target, flat_bank(self.white.variant_inputs(values, 8))):
                dst[t] = src[0]
        self.n += 1

    def query(self, q: int) -> tuple:
        """고정100+100검색·네점수재정렬·75:25최종시간판정이다."""
        from _causal_pr_dump import prefix_topk
        from _xs_eval_lib import align_scores, column_align_scores, minmax
        from _pdv2_sequence_fixed_test import frozen_scores, METHOD
        from retrieval.spectral_evidence import correlation_profiles
        n_prefix = q - 29
        bank = bank_tuple([a[:self.n] for a in self.original])
        whitebank = bank_tuple([a[:self.n] for a in self.corrected])
        candidates = np.unique(np.r_[prefix_topk(self.keys, q, n_prefix, 100), prefix_topk(bank[0], q, n_prefix, 100)])
        raw_a = align_scores(bank[1], bank[2], q, candidates)
        raw_c = column_align_scores(bank[3], q, candidates)
        sn, sh = minmax(self.keys[candidates] @ self.keys[q]), minmax(bank[0][candidates] @ bank[0][q])
        w = self.conf["tuple"]
        fused = w[0]*sn + w[1]*sh + w[2]*minmax(raw_a) + w[3]*minmax(raw_c)
        best = int(np.argsort(-fused, kind="stable")[0])
        j = int(candidates[best])
        channel = self.conf["decision"]
        raw = np.array([raw_c[best] if channel == "c_col" else raw_a[best]], np.float64)
        qs, js = np.array([q]), np.array([j])
        original = frozen_scores(bank, self.travel[:self.n], qs, js, channel, raw)[METHOD][0]
        raw_white = correlation_profiles(whitebank, qs, js, channel).max(1)
        corrected = frozen_scores(whitebank, self.travel[:self.n], qs, js, channel, raw_white)[METHOD][0]
        return j, float(.75*original + .25*corrected), len(candidates)

    def memory(self) -> dict:
        """실제 생성한 배열과 모델·버퍼를 중복 없이 센다."""
        arrays = {"keys_capacity": self.keys, "retained_bev_capacity": self.coeff,
                  "poses_capacity": self.poses, "travel_capacity": self.travel,
                  "recent_raw31": self.recent, "shared_whitening": self.matrix}
        for tag, bank in (("original", self.original), ("whitened", self.corrected)):
            for name, value in zip(("aux_key", "spectrum", "norm", "column_fft", "mask_fft"), bank):
                arrays[tag + "_" + name] = value
        cpu = {k: a.nbytes for k, a in arrays.items()}
        a, b = tensor_bytes(self.net), tensor_bytes(self.encoder)
        cpu["model_encoder_buffers"] = a["cpu"] + b["cpu"]
        model_gpu = a["gpu"] + b["gpu"]
        gpu = torch.cuda.memory_allocated()
        return {"cpu": sum(cpu.values()), "gpu": gpu, "total": sum(cpu.values())+gpu,
                "breakdown": cpu, "model_gpu": model_gpu, "cuda_workspace_and_allocator_overhead": gpu-model_gpu,
                "logical_feature_payload_bytes_per_frame": 2688,
                "logical_feature_payload_bytes": self.n*2688,
                "graph": "local31-node graph rebuilt per update; only transient graph storage, included in measured CUDA peak",
                "timepoint": "after last measured arrival; CPU arrays preallocated to final N+140 context capacity"}


class BEVPlace:
    """공식 BEV 영상·가중치·8192D L2검색을 실제 증분으로 실행한다."""
    def __init__(self, spec: dict, work: Workload):
        import faiss
        import _bevplace_official_bev as bev
        from _bevplace_official_dump import _official_model
        faiss.omp_set_num_threads(1)
        self.faiss, self.bev = faiss, bev
        _, _, self.model = _official_model(REPO / spec["bev_checkpoint"], 1, 1, "cuda")
        self.model.eval()
        self.get_bev = bev.official_getBEV()
        self.work = work
        self.source = np.load(OUT / "bev_batch1.npy", mmap_mode="r")
        self.recent = np.empty((46, 8192), np.float32)
        self.reset()

    def reset(self) -> None:
        """같은 최초 접두 색인을 복원한다."""
        self.n = self.work.initial
        self.index = self.faiss.IndexFlatL2(8192)
        self.index.add(np.zeros((self.work.n_map+SAMPLES-1, 8192), np.float32))
        self.index_pointer = int(self.index.get_xb())
        self.index_capacity = int(self.index.codes.size())
        self.index.reset()
        self.index.add(np.asarray(self.source[:self.work.n_map-1]))
        self.recent[:45] = self.source[self.n-45:self.n]

    def encode(self, points: np.ndarray) -> np.ndarray:
        """공식 getBEV와 OpenCV PNG 변환·/256·REIN 추론이다."""
        import cv2
        image = self.bev.bev_image(points, self.get_bev)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        value = (image.astype(np.float32)/256)[None].repeat(3, axis=0)
        with torch.no_grad():
            _, _, descriptor = self.model(torch.from_numpy(value[None]).cuda())
        return descriptor[0].float().cpu().numpy()

    def append(self, descriptor: np.ndarray, *unused) -> None:
        """45프레임 지원 버퍼와 후보 색인에 실제 새 항목을 추가한다."""
        self.recent[45] = descriptor
        self.index.add(self.recent[:1])
        assert int(self.index.get_xb()) == self.index_pointer
        self.n += 1

    def query(self, q: int) -> tuple:
        """공식 L2 승자·음의 L2 판정 점수를 반환한다."""
        assert self.index.ntotal == q-29
        distances, indices = self.index.search(self.recent[30:31], 1)
        return int(indices[0, 0]), -float(np.sqrt(max(float(distances[0, 0]), 0.))), self.index.ntotal

    def advance(self) -> None:
        """측정 단계 말미에 지원 버퍼를 한 칸 이동한다."""
        self.recent[:45] = self.recent[1:46]

    def memory(self) -> dict:
        """FAISS 실제 색인 크기와 지원 버퍼·공식모델을 센다."""
        model = tensor_bytes(self.model)
        index_size = int(self.index.codes.size())
        capacity = self.index_capacity
        counted = capacity if capacity is not None else index_size
        cpu = counted+self.recent.nbytes+model["cpu"]
        gpu = torch.cuda.memory_allocated()
        return {"cpu": cpu, "gpu": gpu, "total": cpu+gpu, "model_gpu": model["gpu"],
                "cuda_workspace_and_allocator_overhead": gpu-model["gpu"],
                "breakdown": {"faiss_codes": counted, "recent46": self.recent.nbytes, "model_cpu": model["cpu"]},
                "index_size_bytes": index_size, "index_capacity_bytes": capacity,
                "capacity_note": "FAISS capacity unavailable: live vector size is a lower bound; process RSS separately measured" if capacity is None else "preallocated once at final live size; get_xb pointer unchanged throughout incremental adds",
                "logical_feature_payload_bytes_per_frame": 32768,
                "logical_feature_payload_bytes": self.n*32768,
                "timepoint": "after last measured arrival"}


def verify_nsd(runtime: NSD, work: Workload) -> dict:
    """실제증분키와전체접두키·캐시특징·최종승자를 대조한다."""
    checks = []
    for i in range(min(WARMUP, len(work.points))):
        raw, coeff = runtime.encode(work.points[i])
        t = work.initial+i
        np.testing.assert_allclose(raw, work.raw[t], atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(coeff, work.coeff[t])
        runtime.append(raw, coeff, work.poses[t])
        full = runtime.keys_module.make_keys(runtime.net, work.raw[:t+1], work.poses[:t+1])
        err = float(np.max(abs(full-runtime.keys[:t+1])))
        np.testing.assert_allclose(runtime.keys[:t+1], full, atol=3e-6, rtol=3e-6)
        live = runtime.query(t-15)
        saved = runtime.keys[:t+1].copy()
        runtime.keys[:t+1] = full
        reference = runtime.query(t-15)
        runtime.keys[:t+1] = saved
        assert live[0] == reference[0]
        assert abs(live[1]-reference[1]) < 1e-7
        checks.append({"t": t, "key_max_error": err, "score_error": abs(live[1]-reference[1]), "winner": live[0]})
    runtime.reset()
    return {"status": "passed", "raw_regeneration": len(checks), "incremental_fullprefix_checks": checks,
            "tolerance": "keys atol=rtol=3e-6; winner exact; final confidence abs<1e-7; retained coefficients exact"}


def verify_bev(runtime: BEVPlace, work: Workload) -> dict:
    """원시입력추론과저장공식특징·단독FAISS접두검색을 대조한다."""
    checks = []
    for i in range(min(WARMUP, len(work.points))):
        t = work.initial+i
        descriptor = runtime.encode(work.points[i])
        error = float(abs(descriptor-runtime.source[t]).max())
        np.testing.assert_allclose(descriptor, runtime.source[t], atol=1e-7, rtol=1e-6)
        runtime.append(descriptor)
        value = runtime.query(t-15)
        ref = runtime.faiss.IndexFlatL2(8192)
        ref.add(np.asarray(runtime.source[:t-44]))
        distance, winner = ref.search(np.asarray(runtime.source[t-15:t-14]), 1)
        assert value[0] == int(winner[0, 0])
        assert abs(value[1]+float(np.sqrt(distance[0, 0]))) < 2e-5
        checks.append({"t": t, "descriptor_max_error": error, "winner": value[0]})
        runtime.advance()
    runtime.reset()
    return {"status": "passed", "raw_regeneration": len(checks), "official_prefix_checks": checks,
            "tolerance": "batch1 raw vs official PNG batch1 descriptors atol1e-7 rtol1e-6; winner exact; score abs2e-5"}


def bev_cache() -> None:
    """공식 PNG와 공식 추론을 batch1로 재현해 초기 지도를 준비한다."""
    from _bevplace_official_dump import _official_model
    from _bevplace_official_bev import BEV
    spec = prepare()
    target = OUT / "bev_batch1.npy"
    if target.exists():
        meta = read(target.with_suffix(".json"))
        assert meta["sha256"] == sha(target)
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    _, dataset_module, model = _official_model(REPO / spec["bev_checkpoint"], 1, 1, "cuda")
    model.eval()
    dataset = dataset_module.InferDataset(seq="mulran_Sejong01", dataset_path=str(BEV / "val_odom")+"/")
    n = 10000+44+SAMPLES
    values = np.empty((n, 8192), np.float32)
    started = time.time()
    with torch.no_grad():
        for i in range(n):
            image, _ = dataset[i]
            values[i] = model(torch.from_numpy(image[None]).cuda())[2][0].float().cpu().numpy()
            if i % 1000 == 0:
                print("BEV_BATCH1", i, n, round(time.time()-started, 1), flush=True)
                write(OUT / "status.json", {"status": "official_batch1_initial_cache", "done": i, "total": n})
    np.save(target, values)
    previous = np.load(REPO / "results/pdv2_bevplace_val_odom_20261018/descriptors_mulran_Sejong01.npy", mmap_mode="r")
    write(target.with_suffix(".json"), {"sha256": sha(target), "frames": n, "batch": 1,
        "checkpoint_sha256": sha(REPO / spec["bev_checkpoint"]), "tf32_cudnn": False,
        "input": "official InferDataset PNG /256; no labels; cached map initialization excluded from incremental timing",
        "vs_previous_batch32_max_abs": float(abs(values-previous[:n]).max()), "seconds": time.time()-started})


def measure(method: str, n_map: int) -> None:
    """검증된방법하나의지도크기하나를독립프로세스에서계측한다."""
    spec = prepare()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    target = OUT / "methods" / f"{method}_n{n_map}.json"
    if target.exists():
        assert read(target)["status"] == "passed"
        return
    write(OUT / "status.json", {"status": "preparing", "method": method, "n_map": n_map, "unix": time.time()})
    work = Workload(spec, n_map)
    tick = time.perf_counter()
    runtime = NSD(spec, work) if method == "nsd" else BEVPlace(spec, work)
    initial_build_s = time.perf_counter()-tick
    verify = verify_nsd(runtime, work) if method == "nsd" else verify_bev(runtime, work)
    write(OUT / "verification" / f"{method}_n{n_map}.json", verify)
    rows, memory = [], []
    gc.collect()
    torch.cuda.empty_cache()
    write(OUT / "status.json", {"status": "measuring", "method": method, "n_map": n_map, "unix": time.time()})
    for repeat in range(REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rss_before = rss()
        for i, points in enumerate(work.points):
            t = work.initial+i
            encoded, encode_ms = timed(lambda: runtime.encode(points))
            def update():
                if method == "nsd":
                    runtime.append(*encoded, work.poses[t])
                else:
                    runtime.append(encoded)
            _, state_ms = timed(update)
            result, query_ms = timed(lambda: runtime.query(t-15))
            if method == "bevplace":
                _, shift_ms = timed(runtime.advance)
                state_ms += shift_ms
            rows.append({"repeat": repeat, "arrival": t, "query": t-15, "n_candidates": t-44,
                         "encode": encode_ms, "state_update": state_ms, "update": encode_ms+state_ms,
                         "query_ms": query_ms, "processing": encode_ms+state_ms+query_ms,
                         "winner": result[0], "confidence": result[1], "rerank_candidates": result[2]})
        info = runtime.memory()
        info.update({"process_rss_before": rss_before, "process_rss_after": rss(),
                     "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     "cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved(),
                     "cuda_peak_allocated": torch.cuda.max_memory_allocated(), "cuda_peak_reserved": torch.cuda.max_memory_reserved()})
        memory.append(info)
    for i in range(SAMPLES):
        values = [rows[k*SAMPLES+i] for k in range(REPEATS)]
        assert len({v["winner"] for v in values}) == 1
        assert max(v["confidence"] for v in values)-min(v["confidence"] for v in values) < 1e-7
    if method == "nsd":
        path = REPO / "results/pdv2_spectral_whitening_20261026/blend/quarter/s1/mulran_Sejong01.npz"
        with np.load(path) as ledger:
            indices = np.searchsorted(ledger["query_idx"], [r["query"] for r in rows[:SAMPLES]])
            np.testing.assert_array_equal(ledger["j_star"][indices], [r["winner"] for r in rows[:SAMPLES]])
            np.testing.assert_array_equal(ledger["score"][indices], [r["confidence"] for r in rows[:SAMPLES]])
        verify["frozen_ledger"] = {"path": str(path.relative_to(REPO)), "sha256": sha(path),
                                    "queries": SAMPLES, "winners_exact": True, "scores_bit_exact": True}
    result = {"schema": "pdv2-table3-measurement-v1", "method": method, "n_map": n_map,
              "status": "passed", "publish_ready": True, "protocol_sha256": sha(OUT / "protocol.json"),
              "source_sha256": sha(Path(__file__)), "sequence": SEQUENCE, "seed": 1,
              "candidate_range": [n_map, n_map+SAMPLES-1], "mean_candidates": n_map+(SAMPLES-1)/2,
              "samples": SAMPLES, "repeats": REPEATS, "warmup": WARMUP,
              "timings_ms": {key: stats([r[field] for r in rows]) for key, field in (
                  ("encode", "encode"), ("state_update", "state_update"), ("update", "update"),
                  ("query", "query_ms"), ("processing", "processing"))},
              "resident_bytes": memory[-1], "memory_repeats": memory, "verification": verify,
              "raw_input_fixture_bytes": work.input_bytes, "raw_input_sha256": work.input_hash,
              "initial_cached_state_build_s": initial_build_s, "raw_samples": rows,
              "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "platform": platform.platform(), "threads": 1},
              "scope": spec["scope"], "completed_unix": time.time()}
    write(target, result)
    summarize()
    print("COMPLETE", method, n_map, result["timings_ms"], flush=True)


def summarize() -> None:
    """완료되고 검증된 행만 공통그림·표용 JSON으로 모은다."""
    spec = read(OUT / "protocol.json")
    records = []
    for path in sorted((OUT / "methods").glob("*.json")):
        row = read(path)
        assert row["status"] == "passed" and row["verification"]["status"] == "passed"
        assert row["protocol_sha256"] == sha(OUT / "protocol.json")
        records.append({k: v for k, v in row.items() if k not in ("raw_samples", "memory_repeats")})
    write(OUT / "summary.json", {"schema": "pdv2-table3-v1", "protocol": spec,
                                "protocol_sha256": sha(OUT / "protocol.json"), "measurements": records})
    write(OUT / "status.json", {"status": "stage_complete", "completed": [{"method": r["method"], "n_map": r["n_map"]} for r in records], "unix": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "bev_cache", "measure", "summarize"))
    parser.add_argument("--method", choices=("nsd", "bevplace"))
    parser.add_argument("--n-map", type=int, choices=(1000, 5000, 10000), default=1000)
    args = parser.parse_args()
    if args.action == "measure":
        measure(args.method, args.n_map)
    else:
        globals()[args.action]()
