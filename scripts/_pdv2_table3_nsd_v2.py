#!/usr/bin/env python3
"""NSD 개정 2: 31노드 GNN을 CPU에서 돌리고 파생 뱅크를 저장하지 않는다.

점수 함수와 승자·최종 점수 검증은 개정 1(`_pdv2_table3_benchmark.py`)과 같다.
결과는 nsd_v2/에만 쓰며 methods/와 summary.json은 바꾸지 않는다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  python3 scripts/_pdv2_table3_nsd_v2.py measure --n-map 1000
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import platform
import resource
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch
import _pdv2_table3_benchmark as bench
from _pdv2_table3_model_memory import audit_numeric_state

REPO = bench.REPO
OUT = bench.OUT / "nsd_v2"
PROTOCOL = bench.OUT / "nsd_v2_protocol.json"
SIZES = (1000, 5000, 10000)
LEDGER = REPO / "results/pdv2_spectral_whitening_20261026/blend/quarter/s1/mulran_Sejong01.npz"


def settings() -> dict:
    """개정 1에서 바꾸는 두 가지와 그대로 두는 것을 동결한다."""
    paths = [Path(__file__), Path(bench.__file__), REPO / "scripts/_pdv2_table3_model_memory.py",
             REPO / "scripts/_pdv2_spectral_keys.py", REPO / "scripts/_pdv2_spectral_whitening.py",
             REPO / "scripts/_xs_eval_lib.py", REPO / "scripts/_causal_pr_dump.py",
             REPO / "scripts/_pdv2_sequence_fixed_test.py", REPO / "src/retrieval/spectral_evidence.py",
             REPO / "src/retrieval/sequence_evidence.py", REPO / "src/gnn/model.py",
             REPO / "src/gnn/geometry_interaction.py", LEDGER]
    return {
        "schema": "pdv2-table3-nsd-protocol-v2", "revision": 2,
        "parent_protocol_sha256": bench.sha(bench.OUT / "protocol.json"),
        "parent_measurements": {f"n{n}": bench.sha(bench.OUT / "methods" / f"nsd_n{n}.json") for n in SIZES},
        "sources": {str(p.relative_to(REPO)): bench.sha(p) for p in paths},
        "changes": [
            "temporal GNN (31-node incremental graph and initial full-prefix keys) runs on CPU with one thread; same checkpoint and key_scale",
            "spectrum, column FFT and mask FFT banks (original and whitened) are not stored; each query computes them only for the needed rows with the reference variant_inputs and whitening transform",
        ],
        "resident_state": "keys, retained 16x8 coefficients, 128-d BEV index key, poses, travel, recent raw31, shared whitening matrix, model buffers, CUDA allocated",
        "unchanged": "raw encoding (range image on GPU, BEV on CPU), candidate rule, reference scoring functions, fusion weights, decision channel, 75:25 final confidence; MulRan Sejong01; batch1, CPU1, TF32off; 96 arrivals x 3 repeats, warmup12",
        "cuda_workspace": "the GPU reference GNN is used only for verification; its cuBLAS workspace is released and allocator cache emptied before warmup and timing",
        "verification": "derived rows computed one by one equal the full-batch banks bitwise (original and whitened) for every frame; first12 raw regeneration (raw atol2e-6, coefficients exact); incremental CPU keys vs GPU full-prefix keys atol=rtol=3e-6 with winner exact and confidence abs<1e-7; all 96 winners x3 exact and confidence bitwise equal to the frozen ledger",
        "publication": "results stay in nsd_v2/ until the author decides; methods/ and summary.json are unchanged",
    }


def freeze() -> dict:
    """측정 전에 개정 2 원천을 고정한다."""
    config = settings()
    if PROTOCOL.exists():
        assert bench.read(PROTOCOL) == config, "NSD v2 동결 원천 변경"
    else:
        bench.write(PROTOCOL, config)
    return config


def index_key(values: np.ndarray) -> np.ndarray:
    """variant_inputs의 hkey 식을 그대로 쓴 128차원 BEV 색인 키다."""
    from _xs_eval_lib import l2
    skf = values[:, :, :8]
    return l2(np.abs(skf).reshape(len(skf), -1).astype(np.float32))


class Rows:
    """필요한 행만 계산한 파생 배열을 절대 프레임 번호로 읽는다."""
    def __init__(self, rows: np.ndarray, values: np.ndarray):
        self.rows, self.values, self.shape = rows, values, values.shape

    def __getitem__(self, index):
        position = np.searchsorted(self.rows, index)
        assert np.array_equal(self.rows[position], index)
        return self.values[position]


class Runtime(bench.NSD):
    """개정 1과 같은 계산을 CPU GNN과 질의별 파생 행으로 수행한다."""
    def __init__(self, spec: dict, work: bench.Workload, seed: int = 1):
        import _pdv2_spectral_keys as keys
        import _pdv2_spectral_whitening as white
        from _pd_build_test_caches import _effective_config, _make_desc_encoder
        self.keys_module, self.white = keys, white
        ck = spec["nsd_models"][str(seed)]
        assert bench.sha(REPO / ck["checkpoint"]) == ck["sha256"]
        self.net = keys.factory(288).eval()
        self.net.load_state_dict(torch.load(REPO / ck["checkpoint"], map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
        self.net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(keys.OUT / "remetric_octave288.npy")).float(), requires_grad=False)
        self.encoder = _make_desc_encoder(_effective_config(), "mulran", "cuda").eval()
        self.matrix = np.load(white.OUT / "statistics.npz")["full"]
        self.conf = spec["nsd_confidence"][str(seed)]
        self.work = work
        self.capacity = work.end
        self.n = work.initial
        self.keys = np.empty((self.capacity, 416), np.float32)
        self.coeff = np.empty((self.capacity, 16, 8), np.complex64)
        self.hkey = np.empty((self.capacity, 128), np.float32)
        self.poses = np.empty((self.capacity, 4, 4), work.poses.dtype)
        self.travel = np.empty(self.capacity, np.float64)
        self.recent = np.empty((31, 288), np.float32)
        self.transient = 0
        self.reset()

    def make_keys(self, features: np.ndarray, poses: np.ndarray) -> np.ndarray:
        """_pdv2_spectral_keys.make_keys와 같은 그래프·정규화를 CPU에서 계산한다."""
        base = self.keys_module.base
        graph = base.make_graph(features, poses, np.zeros(len(features), np.int64))
        with torch.no_grad():
            key = base.normalize(self.net(graph).float().numpy())
        assert key.shape == (len(features), features.shape[1] + 128) and np.isfinite(key).all()
        return key

    def reset(self) -> None:
        """측정 밖에서 동일한 초기 지도 상태를 복원한다."""
        n = self.work.initial
        self.n = n
        self.keys[:n] = self.make_keys(self.work.raw[:n], self.work.poses[:n])
        self.coeff[:n] = self.work.coeff[:n]
        self.hkey[:n] = index_key(self.coeff[:n])
        self.poses[:n] = self.work.poses[:n]
        self.travel[:n] = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.poses[:n, :3, 3], axis=0), axis=1))]
        self.recent[:] = self.work.raw[n - 31:n]

    def append(self, raw: np.ndarray, coeff: np.ndarray, pose: np.ndarray) -> None:
        """새 도착 하나를 추가하고 영향받는 마지막11개 키를 재계산한다."""
        t = self.n
        self.poses[t] = pose
        self.travel[t] = self.travel[t-1] + np.linalg.norm(pose[:3, 3] - self.poses[t-1, :3, 3])
        self.coeff[t] = coeff
        self.recent[:-1] = self.recent[1:]
        self.recent[-1] = raw
        local = self.make_keys(self.recent, self.poses[t-30:t+1])
        self.keys[t-10:t+1] = local[-11:]
        self.hkey[t] = index_key(coeff[None])[0]
        self.n += 1

    def bank(self, values: np.ndarray, rows: np.ndarray) -> tuple:
        """주어진 행의 계수로 참조 variant_inputs 뱅크를 만든다."""
        derived = bench.flat_bank(self.white.variant_inputs(values, 8))
        self.transient += sum(a.nbytes for a in derived)
        return bench.bank_tuple([Rows(rows, a) for a in derived])

    def query(self, q: int) -> tuple:
        """고정100+100검색·네점수재정렬·75:25최종시간판정이다."""
        from _causal_pr_dump import prefix_topk
        from _xs_eval_lib import align_scores, column_align_scores, minmax
        from _pdv2_sequence_fixed_test import frozen_scores, METHOD
        from retrieval.spectral_evidence import correlation_profiles
        self.transient = 0
        n_prefix = q - 29
        candidates = np.unique(np.r_[prefix_topk(self.keys, q, n_prefix, 100), prefix_topk(self.hkey, q, n_prefix, 100)])
        rows = np.union1d(candidates, [q])
        bank = self.bank(self.coeff[rows], rows)
        raw_a = align_scores(bank[1], bank[2], q, candidates)
        raw_c = column_align_scores(bank[3], q, candidates)
        sn, sh = minmax(self.keys[candidates] @ self.keys[q]), minmax(self.hkey[candidates] @ self.hkey[q])
        w = self.conf["tuple"]
        fused = w[0]*sn + w[1]*sh + w[2]*minmax(raw_a) + w[3]*minmax(raw_c)
        best = int(np.argsort(-fused, kind="stable")[0])
        j = int(candidates[best])
        channel = self.conf["decision"]
        raw = np.array([raw_c[best] if channel == "c_col" else raw_a[best]], np.float64)
        qs, js = np.array([q]), np.array([j])
        travel = self.travel[:self.n]
        # 시간 판정의 이웃은 질의 q±5와 distance_indices가 자르는 후보 j±10 창 안에 있다.
        span = np.r_[np.arange(q-5, q+6), np.arange(j-10, j+11)]
        rows = np.unique(span[(span >= 0) & (span < self.n)])
        values = self.coeff[rows]
        original = frozen_scores(self.bank(values, rows), travel, qs, js, channel, raw)[METHOD][0]
        white = self.bank(self.white.transform(values, self.matrix), rows)
        raw_white = correlation_profiles(white, qs, js, channel).max(1)
        corrected = frozen_scores(white, travel, qs, js, channel, raw_white)[METHOD][0]
        return j, float(.75*original + .25*corrected), len(candidates)

    def memory(self) -> dict:
        """실제 상주 배열과 CPU 모델·GPU 할당을 중복 없이 센다."""
        arrays = {"keys_capacity": self.keys, "retained_bev_capacity": self.coeff,
                  "index_bev_key_capacity": self.hkey, "poses_capacity": self.poses,
                  "travel_capacity": self.travel, "recent_raw31": self.recent, "shared_whitening": self.matrix}
        cpu = {k: a.nbytes for k, a in arrays.items()}
        a, b = bench.tensor_bytes(self.net), bench.tensor_bytes(self.encoder)
        audit = audit_numeric_state(self.net, self.encoder)
        assert audit["status"] == "passed" and not audit["unknown_capacity_paths"]
        assert audit["registered_cpu_bytes"] == a["cpu"] + b["cpu"]
        cpu["model_encoder_buffers"] = a["cpu"] + b["cpu"]
        cpu["model_extra_cpu_numeric"] = audit["extra_cpu_bytes"]
        model_gpu = a["gpu"] + b["gpu"]
        gpu = torch.cuda.memory_allocated()
        return {"cpu": sum(cpu.values()), "gpu": gpu, "total": sum(cpu.values())+gpu,
                "breakdown": cpu, "model_gpu": model_gpu, "cuda_workspace_and_allocator_overhead": gpu-model_gpu,
                "model_numeric_audit": {k: audit[k] for k in ("cpu_unique_bytes", "registered_cpu_bytes", "extra_cpu_bytes")},
                "logical_feature_payload_bytes_per_frame": 2688,
                "logical_feature_payload_bytes": self.n*2688,
                "graph": "local31-node graph rebuilt per update on CPU; transient",
                "derived_banks": "computed per query for needed rows only; transient, not resident; per-query bytes in raw_samples",
                "timepoint": "after last measured arrival; CPU arrays preallocated to final N+140 context capacity"}


CHUNKS = (177, 23, 7, 64, 2, 301)


def check_rows(runtime: Runtime, work: bench.Workload) -> dict:
    """한 행씩, 그리고 묶음 크기를 바꿔 계산한 파생 배열·색인 키가 전체 일괄 계산과 비트 단위로 같은지 본다."""
    coeff = work.coeff[:work.end]
    white = runtime.white
    full = {"original": bench.flat_bank(white.variant_inputs(coeff, 8)),
            "whitened": bench.flat_bank(white.variant_inputs(white.transform(coeff, runtime.matrix), 8))}
    np.testing.assert_array_equal(index_key(coeff), full["original"][0])
    for sizes in ((1,), CHUNKS):
        start, k = 0, 0
        while start < len(coeff):
            stop = min(start + sizes[k % len(sizes)], len(coeff))
            part = coeff[start:stop]
            np.testing.assert_array_equal(index_key(part), full["original"][0][start:stop])
            for tag, values in (("original", part), ("whitened", white.transform(part, runtime.matrix))):
                for a, b in zip(bench.flat_bank(white.variant_inputs(values, 8)), full[tag]):
                    np.testing.assert_array_equal(a, b[start:stop])
            start, k = stop, k + 1
    return {"derived_rows_bitwise_equal": {"frames": len(coeff), "chunk_sizes": [1, *CHUNKS]}}


def verify(runtime: Runtime, work: bench.Workload, reference) -> dict:
    """원시 재생성, 증분 CPU 키와 GPU 전체 접두 키, 실제 질의를 대조한다."""
    checks = []
    for i in range(min(bench.WARMUP, len(work.points))):
        raw, coeff = runtime.encode(work.points[i])
        t = work.initial+i
        np.testing.assert_allclose(raw, work.raw[t], atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(coeff, work.coeff[t])
        runtime.append(raw, coeff, work.poses[t])
        full = runtime.keys_module.make_keys(reference, work.raw[:t+1], work.poses[:t+1])
        err = float(np.max(abs(full-runtime.keys[:t+1])))
        np.testing.assert_allclose(runtime.keys[:t+1], full, atol=3e-6, rtol=3e-6)
        live = runtime.query(t-15)
        saved = runtime.keys[:t+1].copy()
        runtime.keys[:t+1] = full
        ref = runtime.query(t-15)
        runtime.keys[:t+1] = saved
        assert live[0] == ref[0]
        assert abs(live[1]-ref[1]) < 1e-7
        checks.append({"t": t, "key_max_error_vs_gpu_fullprefix": err, "score_error": abs(live[1]-ref[1]), "winner": live[0]})
    runtime.reset()
    return {"status": "passed", "raw_regeneration": len(checks), "incremental_fullprefix_checks": checks,
            **check_rows(runtime, work),
            "tolerance": "keys atol=rtol=3e-6 vs GPU full prefix; winner exact; final confidence abs<1e-7; retained coefficients exact; derived rows bitwise"}


def reference_net(spec: dict, seed: int = 1):
    """검증 전용 GPU 참조 GNN(개정 1과 같은 적재)이다."""
    import _pdv2_spectral_keys as keys
    ck = spec["nsd_models"][str(seed)]
    net = keys.factory(288).cuda().eval()
    net.load_state_dict(torch.load(REPO / ck["checkpoint"], map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(keys.OUT / "remetric_octave288.npy")).float().cuda(), requires_grad=False)
    return net


def measure(n_map: int) -> None:
    """한 지도 크기를 독립 프로세스에서 검증하고 계측한다."""
    spec = bench.prepare()
    config = freeze()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    target = OUT / f"nsd_n{n_map}.json"
    if target.exists():
        assert bench.read(target)["status"] == "passed"
        return
    work = bench.Workload(spec, n_map)
    tick = time.perf_counter()
    runtime = Runtime(spec, work)
    initial_build_s = time.perf_counter()-tick
    reference = reference_net(spec)
    checked = verify(runtime, work, reference)
    del reference
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    for i in range(bench.WARMUP):
        t = work.initial+i
        runtime.append(*runtime.encode(work.points[i]), work.poses[t])
        runtime.query(t-15)
    bench.write(OUT / f"verification_n{n_map}.json", checked)
    rows, memory = [], []
    for repeat in range(bench.REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rss_before = bench.rss()
        for i, points in enumerate(work.points):
            t = work.initial+i
            encoded, encode_ms = bench.timed(lambda: runtime.encode(points))
            _, state_ms = bench.timed(lambda: runtime.append(*encoded, work.poses[t]))
            result, query_ms = bench.timed(lambda: runtime.query(t-15))
            rows.append({"repeat": repeat, "arrival": t, "query": t-15, "n_candidates": t-44,
                         "encode": encode_ms, "state_update": state_ms, "update": encode_ms+state_ms,
                         "query_ms": query_ms, "processing": encode_ms+state_ms+query_ms,
                         "winner": result[0], "confidence": result[1], "rerank_candidates": result[2],
                         "transient_derived_bytes": runtime.transient})
        info = runtime.memory()
        info.update({"process_rss_before": rss_before, "process_rss_after": bench.rss(),
                     "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     "cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved(),
                     "cuda_peak_allocated": torch.cuda.max_memory_allocated(), "cuda_peak_reserved": torch.cuda.max_memory_reserved()})
        memory.append(info)
    with np.load(LEDGER) as ledger:
        indices = np.searchsorted(ledger["query_idx"], [r["query"] for r in rows])
        np.testing.assert_array_equal(ledger["j_star"][indices], [r["winner"] for r in rows])
        np.testing.assert_array_equal(ledger["score"][indices], [r["confidence"] for r in rows])
    checked["frozen_ledger"] = {"path": str(LEDGER.relative_to(REPO)), "sha256": bench.sha(LEDGER),
                                "queries": len(rows), "winners_exact": True, "scores_bit_exact": True}
    bench.write(OUT / f"verification_n{n_map}.json", checked)
    parent = bench.read(bench.OUT / "methods" / f"nsd_n{n_map}.json")
    result = {"schema": "pdv2-table3-measurement-v1", "method": "nsd", "n_map": n_map,
              "status": "passed", "publish_ready": False, "protocol_revision": 2,
              "protocol_sha256": bench.sha(bench.OUT / "protocol.json"), "method_protocol_sha256": bench.sha(PROTOCOL),
              "source_sha256": bench.sha(Path(__file__)), "sequence": bench.SEQUENCE, "seed": 1,
              "candidate_range": [n_map, n_map+bench.SAMPLES-1], "mean_candidates": n_map+(bench.SAMPLES-1)/2,
              "samples": bench.SAMPLES, "repeats": bench.REPEATS, "warmup": bench.WARMUP,
              "timings_ms": {key: bench.stats([r[field] for r in rows]) for key, field in (
                  ("encode", "encode"), ("state_update", "state_update"), ("update", "update"),
                  ("query", "query_ms"), ("processing", "processing"))},
              "resident_bytes": memory[-1], "memory_repeats": memory, "verification": checked,
              "transient_derived_bytes_max": max(r["transient_derived_bytes"] for r in rows),
              "raw_input_fixture_bytes": work.input_bytes, "raw_input_sha256": work.input_hash,
              "initial_cached_state_build_s": initial_build_s, "raw_samples": rows,
              "parent_revision1": {"timings_ms": parent["timings_ms"], "resident_total_bytes": parent["resident_bytes"]["total"]},
              "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "platform": platform.platform(), "threads": 1},
              "scope": spec["scope"], "changes": config["changes"], "completed_unix": time.time()}
    bench.write(target, result)
    print("COMPLETE nsd_v2", n_map, result["timings_ms"]["processing"], result["resident_bytes"]["total"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("measure",))
    parser.add_argument("--n-map", type=int, choices=SIZES, default=1000)
    args = parser.parse_args()
    measure(args.n_map)
