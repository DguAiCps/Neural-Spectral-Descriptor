#!/usr/bin/env python3
"""NSD 개정 3: 개정 2에 질의 정리 세 가지를 더한다(파이썬만).

① 시간 판정은 쓰는 점수(arc_unique_b100)만 계산한다. 원본·보정 뱅크가 같은 이웃 계획을 쓰고
   중심·이웃 상관을 뱅크마다 한 번에 계산한다. frozen_scores와 비트 단위로 같다.
② 정렬 점수는 16행 합을 역FFT 앞에서 한다(선형성). 식은 같고 반올림만 다르다.
③ 원본·보정 열 정렬 뱅크를 저장한다. 스펙트럼 정렬은 저장한 16×8 계수로 바로 계산한다.

갱신(인코딩, CPU GNN)은 개정 2와 같다. 결과는 nsd_v3/에만 쓰며 methods/와 summary.json은 바꾸지 않는다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  python3 scripts/_pdv2_table3_nsd_v3.py measure --n-map 1000
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import platform
import resource
import time

import _pdv2_table3_nsd_v2 as base
import numpy as np
import torch

bench = base.bench
REPO = bench.REPO
OUT = bench.OUT / "nsd_v3"
PROTOCOL = bench.OUT / "nsd_v3_protocol.json"
SIZES = base.SIZES
LEDGER = base.LEDGER
DENSE = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
BETA = (2./3.)*(11./10.)
SCORE_TOLERANCE = 1e-6


def settings() -> dict:
    """개정 2에서 바꾸는 세 가지와 검증 기준을 동결한다."""
    paths = [Path(__file__), Path(base.__file__), Path(bench.__file__), REPO / "scripts/_pdv2_table3_model_memory.py",
             REPO / "scripts/_pdv2_spectral_keys.py", REPO / "scripts/_pdv2_spectral_whitening.py",
             REPO / "scripts/_xs_eval_lib.py", REPO / "scripts/_causal_pr_dump.py",
             REPO / "scripts/_pdv2_sequence_fixed_test.py", REPO / "src/retrieval/spectral_evidence.py",
             REPO / "src/retrieval/sequence_evidence.py", REPO / "src/gnn/model.py",
             REPO / "src/gnn/geometry_interaction.py", LEDGER]
    return {
        "schema": "pdv2-table3-nsd-protocol-v3", "revision": 3,
        "parent_protocol_sha256": bench.sha(bench.OUT / "protocol.json"),
        "parent_revision2_protocol_sha256": bench.sha(base.PROTOCOL),
        "parent_measurements": {f"n{n}": bench.sha(base.OUT / f"nsd_n{n}.json") for n in SIZES},
        "sources": {str(p.relative_to(REPO)): bench.sha(p) for p in paths},
        "changes": [
            "as revision 2: temporal GNN runs on CPU with one thread; same checkpoint and key_scale",
            "temporal decision computes only the published arc_unique_b100 score: one arc_dense neighbor plan shared by the original and whitened banks, center and neighbor correlations in one call per bank; the unused arc_sparse score and the other neighbor styles are not computed",
            "spectral and column alignment sum the 16 radial rows before the inverse FFT (linearity); spectral alignment uses the retained 16x8 coefficients directly",
            "original and whitened column-alignment banks (column FFT, mask FFT) are stored; spectrum FFT and norm banks are not stored",
        ],
        "resident_state": "keys, retained 16x8 coefficients, 128-d BEV index key, original and whitened column banks, poses, travel, recent raw31, shared whitening matrix, model buffers, CUDA allocated",
        "unchanged": "raw encoding (range image on GPU, BEV on CPU), candidate rule, fusion weights, decision channel, 75:25 final confidence; MulRan Sejong01; batch1, CPU1, TF32off; 96 arrivals x 3 repeats, warmup12",
        "cuda_workspace": "the GPU reference GNN is used only for verification; its cuBLAS workspace is released and allocator cache emptied before warmup and timing",
        "verification": "temporal decision bitwise equal to frozen_scores for every ledger query in the workload range (original and whitened banks); stored column rows computed one by one and in chunks equal the reference full-batch banks bitwise; revision 2 derived-row checks; first12 raw regeneration, incremental CPU keys vs GPU full-prefix keys atol=rtol=3e-6, live vs revision-2 reference query winner exact and confidence abs<=1e-6; all 96 winners x3 exact, confidence abs<=1e-6 and tau* acceptance unchanged against the frozen ledger",
        "publication": "results stay in nsd_v3/ until the author decides; methods/ and summary.json are unchanged",
    }


def freeze() -> dict:
    """측정 전에 개정 3 원천을 고정한다."""
    config = settings()
    if PROTOCOL.exists():
        assert bench.read(PROTOCOL) == config, "NSD v3 동결 원천 변경"
    else:
        bench.write(PROTOCOL, config)
    return config


def column_bank(values: np.ndarray) -> tuple:
    """variant_inputs의 열 정렬 뱅크 줄만 그대로 계산한다."""
    from _xs_eval_lib import column_align_bank
    hfft = np.zeros((len(values), 16, 31), np.complex64)
    hfft[:, :, 1:9] = values[:, :, :8]
    return column_align_bank(np.fft.irfft(hfft, n=60, axis=-1).astype(np.float32))


def spectral_scores(coeff: np.ndarray, q: int, candidates: np.ndarray) -> np.ndarray:
    """align_scores와 같은 식이다. 16행 합을 역FFT 앞에서 한다."""
    product = np.zeros((len(candidates), 31), np.complex64)
    product[:, 1:9] = (coeff[candidates] * np.conj(coeff[q])).sum(1)
    rows = coeff[np.r_[candidates, q]]
    norm = np.clip(np.linalg.norm(np.abs(rows).reshape(len(rows), -1), axis=1), 1e-8, None)
    return np.fft.irfft(product, n=60, axis=-1).max(-1) / (norm[:-1] * norm[-1])


def column_scores(bank: tuple, q: int, candidates: np.ndarray) -> np.ndarray:
    """column_align_scores와 같은 식이다. 16행 합을 역FFT 앞에서 한다."""
    fft, mask = bank
    corr = np.fft.irfft((fft[candidates] * np.conj(fft[q])).sum(1), n=60, axis=-1)
    count = np.fft.irfft(mask[candidates] * np.conj(mask[q]), n=60, axis=-1)
    return np.where(count > .5, corr / np.maximum(count, 1.), -1.).max(-1)


def temporal(columns: dict, travel: np.ndarray, q: int, j: int, raw: np.ndarray) -> list:
    """frozen_scores의 arc_unique_b100을 원본·보정 뱅크에 대해 계산한다."""
    from retrieval.sequence_evidence import distance_indices, path_means, repeated_candidate_weights
    from retrieval.spectral_evidence import correlation_profiles
    n = len(travel)
    qn = q + DENSE
    qsafe = np.clip(qn, 0, n-1)
    jn = np.stack([distance_indices(travel, np.array([j]), d*(travel[qsafe]-travel[q])) for d in (1, -1)])[:, None]
    qn = np.broadcast_to(qn, jn.shape)
    valid = (qn >= 0) & (qn < n) & (jn >= 0) & (jn < n) & (jn != j)
    unique, inverse = np.unique((qn*n+jn)[valid], return_inverse=True)
    weights = repeated_candidate_weights(jn, valid)
    qs, js = np.r_[q, unique//n], np.r_[j, unique % n]
    scores = []
    for tag, center_raw in (("original", raw), ("whitened", None)):
        peaks = correlation_profiles((None, None, None, columns[tag]), qs, js, "c_col").max(1)
        values = np.zeros(qn.shape, dtype=peaks.dtype)
        values[valid] = peaks[1:][inverse]
        start = peaks[:1] if center_raw is None else center_raw
        scores.append(((1-BETA)*start + BETA*path_means(peaks[:1], values, valid, weights).max(0))[0])
    return scores


class Runtime(base.Runtime):
    """개정 2의 갱신에 저장 열 뱅크와 정리한 질의를 더한다."""
    def __init__(self, spec: dict, work: bench.Workload, seed: int = 1):
        assert spec["nsd_confidence"][str(seed)]["decision"] == "c_col"
        fft, mask = column_bank(np.zeros((1, 16, 8), np.complex64))
        self.columns = {tag: (np.empty((work.end, *fft.shape[1:]), fft.dtype), np.empty((work.end, *mask.shape[1:]), mask.dtype))
                        for tag in ("original", "whitened")}
        super().__init__(spec, work, seed)

    def reset(self) -> None:
        """측정 밖에서 동일한 초기 지도 상태를 복원한다."""
        super().reset()
        n = self.n
        for tag, values in (("original", self.coeff[:n]), ("whitened", self.white.transform(self.coeff[:n], self.matrix))):
            for dst, src in zip(self.columns[tag], column_bank(values)):
                dst[:n] = src

    def append(self, raw: np.ndarray, coeff: np.ndarray, pose: np.ndarray) -> None:
        """개정 2의 갱신 뒤 새 프레임의 열 뱅크 두 벌을 저장한다."""
        super().append(raw, coeff, pose)
        t = self.n - 1
        for tag, values in (("original", coeff[None]), ("whitened", self.white.transform(coeff[None], self.matrix))):
            for dst, src in zip(self.columns[tag], column_bank(values)):
                dst[t] = src[0]

    def query(self, q: int) -> tuple:
        """고정100+100검색·네점수재정렬·75:25최종시간판정이다."""
        from _causal_pr_dump import prefix_topk
        from _xs_eval_lib import minmax
        n_prefix = q - 29
        candidates = np.unique(np.r_[prefix_topk(self.keys, q, n_prefix, 100), prefix_topk(self.hkey, q, n_prefix, 100)])
        raw_a = spectral_scores(self.coeff, q, candidates)
        raw_c = column_scores(self.columns["original"], q, candidates)
        sn, sh = minmax(self.keys[candidates] @ self.keys[q]), minmax(self.hkey[candidates] @ self.hkey[q])
        w = self.conf["tuple"]
        fused = w[0]*sn + w[1]*sh + w[2]*minmax(raw_a) + w[3]*minmax(raw_c)
        best = int(np.argsort(-fused, kind="stable")[0])
        j = int(candidates[best])
        raw = np.array([raw_c[best]], np.float64)
        original, corrected = temporal(self.columns, self.travel[:self.n], q, j, raw)
        return j, float(.75*original + .25*corrected), len(candidates)

    def memory(self) -> dict:
        """개정 2의 상주 상태에 저장 열 뱅크를 더한다."""
        info = super().memory()
        for tag, (fft, mask) in self.columns.items():
            info["breakdown"][tag + "_column_fft_capacity"] = fft.nbytes
            info["breakdown"][tag + "_mask_fft_capacity"] = mask.nbytes
        info["cpu"] = sum(info["breakdown"].values())
        info["total"] = info["cpu"] + info["gpu"]
        info["derived_banks"] = "original and whitened column banks stored; spectral alignment uses retained coefficients; nothing else derived is stored"
        return info


def check_columns(runtime: Runtime, work: bench.Workload) -> dict:
    """한 행씩, 그리고 묶음 크기를 바꿔 저장한 열 뱅크가 참조 전체 일괄 계산과 비트 단위로 같은지 본다."""
    coeff = work.coeff[:work.end]
    white = runtime.white
    full = {"original": white.variant_inputs(coeff, 8)[3],
            "whitened": white.variant_inputs(white.transform(coeff, runtime.matrix), 8)[3]}
    for sizes in ((1,), base.CHUNKS):
        start, k = 0, 0
        while start < len(coeff):
            stop = min(start + sizes[k % len(sizes)], len(coeff))
            part = coeff[start:stop]
            for tag, values in (("original", part), ("whitened", white.transform(part, runtime.matrix))):
                for a, b in zip(column_bank(values), full[tag]):
                    np.testing.assert_array_equal(a, b[start:stop])
            start, k = stop, k + 1
    return {"stored_column_rows_bitwise_equal": {"frames": len(coeff), "chunk_sizes": [1, *base.CHUNKS]}}


def check_temporal(runtime: Runtime, work: bench.Workload) -> dict:
    """정리한 시간 판정이 frozen_scores와 비트 단위로 같은지 작업 범위의 모든 원장 질의에서 본다."""
    from _pdv2_sequence_fixed_test import frozen_scores, METHOD
    from retrieval.spectral_evidence import correlation_profiles
    coeff = work.coeff[:work.end]
    white = runtime.white
    banks = {"original": bench.bank_tuple(bench.flat_bank(white.variant_inputs(coeff, 8))),
             "whitened": bench.bank_tuple(bench.flat_bank(white.variant_inputs(white.transform(coeff, runtime.matrix), 8)))}
    columns = {tag: bank[3] for tag, bank in banks.items()}
    poses = work.poses[:work.end]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]
    with np.load(LEDGER) as ledger:
        keep = ledger["query_idx"] < work.end
        pairs = list(zip(ledger["query_idx"][keep].tolist(), ledger["j_star"][keep].tolist()))
    for q, j in pairs:
        qs, js = np.array([q]), np.array([j])
        raw = correlation_profiles(banks["original"], qs, js, "c_col").max(1).astype(np.float64)
        expected = [frozen_scores(banks["original"], travel, qs, js, "c_col", raw)[METHOD][0]]
        white_raw = correlation_profiles(banks["whitened"], qs, js, "c_col").max(1)
        expected.append(frozen_scores(banks["whitened"], travel, qs, js, "c_col", white_raw)[METHOD][0])
        np.testing.assert_array_equal(temporal(columns, travel, q, j, raw), expected)
    return {"temporal_bitwise_equal": {"ledger_queries": len(pairs), "banks": ["original", "whitened"]}}


def verify(runtime: Runtime, work: bench.Workload, reference) -> dict:
    """원시 재생성, 증분 CPU 키, 개정 2 참조 질의와의 차이를 대조한다."""
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
        previous = base.Runtime.query(runtime, t-15)
        saved = runtime.keys[:t+1].copy()
        runtime.keys[:t+1] = full
        keyed = runtime.query(t-15)
        runtime.keys[:t+1] = saved
        assert live[0] == previous[0] == keyed[0]
        assert abs(live[1]-previous[1]) <= SCORE_TOLERANCE and abs(live[1]-keyed[1]) < 1e-7
        checks.append({"t": t, "key_max_error_vs_gpu_fullprefix": err, "winner": live[0],
                       "score_error_vs_revision2_query": abs(live[1]-previous[1]), "score_error_vs_fullprefix_keys": abs(live[1]-keyed[1])})
    runtime.reset()
    return {"status": "passed", "raw_regeneration": len(checks), "incremental_checks": checks,
            **check_temporal(runtime, work), **check_columns(runtime, work), **base.check_rows(runtime, work),
            "tolerance": "temporal decision and stored column rows bitwise; keys atol=rtol=3e-6 vs GPU full prefix; winner exact; confidence abs<=1e-6 vs revision 2 reference query and ledger; tau* acceptance unchanged"}


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
    reference = base.reference_net(spec)
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
                         "winner": result[0], "confidence": result[1], "rerank_candidates": result[2]})
        info = runtime.memory()
        info.update({"process_rss_before": rss_before, "process_rss_after": bench.rss(),
                     "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     "cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved(),
                     "cuda_peak_allocated": torch.cuda.max_memory_allocated(), "cuda_peak_reserved": torch.cuda.max_memory_reserved()})
        memory.append(info)
    tau = spec["nsd_confidence"]["1"]["tau_star"]
    with np.load(LEDGER) as ledger:
        indices = np.searchsorted(ledger["query_idx"], [r["query"] for r in rows])
        np.testing.assert_array_equal(ledger["j_star"][indices], [r["winner"] for r in rows])
        expected, actual = ledger["score"][indices], np.array([r["confidence"] for r in rows])
    error = float(np.abs(actual-expected).max())
    assert error <= SCORE_TOLERANCE
    np.testing.assert_array_equal(actual >= tau, expected >= tau)
    checked["frozen_ledger"] = {"path": str(LEDGER.relative_to(REPO)), "sha256": bench.sha(LEDGER),
                                "queries": len(rows), "winners_exact": True, "score_max_abs_error": error,
                                "scores_bit_exact": bool(np.array_equal(actual, expected)),
                                "tau_star": tau, "acceptance_unchanged": True,
                                "min_ledger_distance_to_tau": float(np.abs(expected-tau).min())}
    bench.write(OUT / f"verification_n{n_map}.json", checked)
    previous = bench.read(base.OUT / f"nsd_n{n_map}.json")
    first = bench.read(bench.OUT / "methods" / f"nsd_n{n_map}.json")
    result = {"schema": "pdv2-table3-measurement-v1", "method": "nsd", "n_map": n_map,
              "status": "passed", "publish_ready": False, "protocol_revision": 3,
              "protocol_sha256": bench.sha(bench.OUT / "protocol.json"), "method_protocol_sha256": bench.sha(PROTOCOL),
              "source_sha256": bench.sha(Path(__file__)), "sequence": bench.SEQUENCE, "seed": 1,
              "candidate_range": [n_map, n_map+bench.SAMPLES-1], "mean_candidates": n_map+(bench.SAMPLES-1)/2,
              "samples": bench.SAMPLES, "repeats": bench.REPEATS, "warmup": bench.WARMUP,
              "timings_ms": {key: bench.stats([r[field] for r in rows]) for key, field in (
                  ("encode", "encode"), ("state_update", "state_update"), ("update", "update"),
                  ("query", "query_ms"), ("processing", "processing"))},
              "resident_bytes": memory[-1], "memory_repeats": memory, "verification": checked,
              "raw_input_fixture_bytes": work.input_bytes, "raw_input_sha256": work.input_hash,
              "initial_cached_state_build_s": initial_build_s, "raw_samples": rows,
              "parent_revision2": {"timings_ms": previous["timings_ms"], "resident_total_bytes": previous["resident_bytes"]["total"]},
              "parent_revision1": {"timings_ms": first["timings_ms"], "resident_total_bytes": first["resident_bytes"]["total"]},
              "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "platform": platform.platform(), "threads": 1},
              "scope": spec["scope"], "changes": config["changes"], "completed_unix": time.time()}
    bench.write(target, result)
    print("COMPLETE nsd_v3", n_map, result["timings_ms"]["processing"], result["resident_bytes"]["total"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("measure",))
    parser.add_argument("--n-map", type=int, choices=SIZES, default=1000)
    args = parser.parse_args()
    measure(args.n_map)
