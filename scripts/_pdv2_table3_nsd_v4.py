#!/usr/bin/env python3
"""NSD 개정 4: 개정 2의 갱신·상주 상태에 개정 3의 질의 정리 ①②만 쓴다(파이썬만).

열 정렬 뱅크는 저장하지 않고 질의 때 후보·질의 행(원본)과 시간 판정 이웃 행(원본·보정)만 계산한다.
결과는 nsd_v4/에만 쓰며 methods/와 summary.json은 바꾸지 않는다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  python3 scripts/_pdv2_table3_nsd_v4.py measure --n-map 10000
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import platform
import resource
import time

import _pdv2_table3_nsd_v3 as v3
import numpy as np
import torch

base, bench = v3.base, v3.bench
REPO = bench.REPO
OUT = bench.OUT / "nsd_v4"
PROTOCOL = bench.OUT / "nsd_v4_protocol.json"
SIZES = base.SIZES
LEDGER = base.LEDGER


def settings() -> dict:
    """개정 2·3에서 가져오는 것과 검증 기준을 동결한다."""
    paths = [Path(__file__), Path(v3.__file__), Path(base.__file__), Path(bench.__file__),
             REPO / "scripts/_pdv2_table3_model_memory.py", REPO / "scripts/_pdv2_spectral_keys.py",
             REPO / "scripts/_pdv2_spectral_whitening.py", REPO / "scripts/_xs_eval_lib.py",
             REPO / "scripts/_causal_pr_dump.py", REPO / "scripts/_pdv2_sequence_fixed_test.py",
             REPO / "src/retrieval/spectral_evidence.py", REPO / "src/retrieval/sequence_evidence.py",
             REPO / "src/gnn/model.py", REPO / "src/gnn/geometry_interaction.py", LEDGER]
    return {
        "schema": "pdv2-table3-nsd-protocol-v4", "revision": 4,
        "parent_protocol_sha256": bench.sha(bench.OUT / "protocol.json"),
        "parent_revision2_protocol_sha256": bench.sha(base.PROTOCOL),
        "parent_revision3_protocol_sha256": bench.sha(v3.PROTOCOL),
        "parent_measurements": {f"n{n}": bench.sha(v3.OUT / f"nsd_n{n}.json") for n in SIZES},
        "sources": {str(p.relative_to(REPO)): bench.sha(p) for p in paths},
        "changes": [
            "as revision 2: temporal GNN runs on CPU with one thread; same checkpoint and key_scale",
            "as revision 2: no derived bank is stored; column-alignment banks are computed per query only for the candidate and query rows (original) and the temporal-decision neighbor rows q-5..q+5 and j-10..j+10 (original and whitened)",
            "as revision 3: temporal decision computes only the published arc_unique_b100 score with one shared arc_dense neighbor plan and one correlation call per bank",
            "as revision 3: spectral and column alignment sum the 16 radial rows before the inverse FFT (linearity); spectral alignment uses the retained 16x8 coefficients directly",
        ],
        "resident_state": "keys, retained 16x8 coefficients, 128-d BEV index key, poses, travel, recent raw31, shared whitening matrix, model buffers, CUDA allocated",
        "unchanged": "raw encoding (range image on GPU, BEV on CPU), candidate rule, fusion weights, decision channel, 75:25 final confidence; MulRan Sejong01; batch1, CPU1, TF32off; 96 arrivals x 3 repeats, warmup12",
        "cuda_workspace": "the GPU reference GNN is used only for verification; its cuBLAS workspace is released and allocator cache emptied before warmup and timing",
        "verification": "temporal decision bitwise equal to frozen_scores for every ledger query in the workload range; column rows computed one by one and in chunks equal the reference full-batch banks bitwise; revision 2 derived-row checks; first12 raw regeneration, incremental CPU keys vs GPU full-prefix keys atol=rtol=3e-6, live vs revision-2 reference query winner exact and confidence abs<=1e-6; all 96 winners x3 exact, confidence abs<=1e-6 and tau* acceptance unchanged against the frozen ledger; winners and confidences bitwise equal to revision 3",
        "publication": "results stay in nsd_v4/ until moved into methods/ by the author's decision",
    }


def freeze() -> dict:
    """측정 전에 개정 4 원천을 고정한다."""
    config = settings()
    if PROTOCOL.exists():
        assert bench.read(PROTOCOL) == config, "NSD v4 동결 원천 변경"
    else:
        bench.write(PROTOCOL, config)
    return config


class Runtime(base.Runtime):
    """개정 2의 상태에서 개정 3의 정리한 질의를 필요한 열 뱅크 행만으로 계산한다."""
    def __init__(self, spec: dict, work: bench.Workload, seed: int = 1):
        assert spec["nsd_confidence"][str(seed)]["decision"] == "c_col"
        super().__init__(spec, work, seed)

    def columns(self, rows: np.ndarray, tags: tuple) -> dict:
        """주어진 행의 열 뱅크를 절대 프레임 번호로 읽게 만든다."""
        out = {}
        for tag in tags:
            values = self.coeff[rows] if tag == "original" else self.white.transform(self.coeff[rows], self.matrix)
            bank = v3.column_bank(values)
            self.transient += sum(a.nbytes for a in bank)
            out[tag] = tuple(base.Rows(rows, a) for a in bank)
        return out

    def query(self, q: int) -> tuple:
        """고정100+100검색·네점수재정렬·75:25최종시간판정이다."""
        from _causal_pr_dump import prefix_topk
        from _xs_eval_lib import minmax
        self.transient = 0
        n_prefix = q - 29
        candidates = np.unique(np.r_[prefix_topk(self.keys, q, n_prefix, 100), prefix_topk(self.hkey, q, n_prefix, 100)])
        raw_a = v3.spectral_scores(self.coeff, q, candidates)
        raw_c = v3.column_scores(self.columns(np.union1d(candidates, [q]), ("original",))["original"], q, candidates)
        sn, sh = minmax(self.keys[candidates] @ self.keys[q]), minmax(self.hkey[candidates] @ self.hkey[q])
        w = self.conf["tuple"]
        fused = w[0]*sn + w[1]*sh + w[2]*minmax(raw_a) + w[3]*minmax(raw_c)
        best = int(np.argsort(-fused, kind="stable")[0])
        j = int(candidates[best])
        raw = np.array([raw_c[best]], np.float64)
        # 시간 판정의 이웃은 질의 q±5와 distance_indices가 자르는 후보 j±10 창 안에 있다.
        span = np.r_[np.arange(q-5, q+6), np.arange(j-10, j+11)]
        rows = np.unique(span[(span >= 0) & (span < self.n)])
        original, corrected = v3.temporal(self.columns(rows, ("original", "whitened")), self.travel[:self.n], q, j, raw)
        return j, float(.75*original + .25*corrected), len(candidates)


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
        assert abs(live[1]-previous[1]) <= v3.SCORE_TOLERANCE and abs(live[1]-keyed[1]) < 1e-7
        checks.append({"t": t, "key_max_error_vs_gpu_fullprefix": err, "winner": live[0],
                       "score_error_vs_revision2_query": abs(live[1]-previous[1]), "score_error_vs_fullprefix_keys": abs(live[1]-keyed[1])})
    runtime.reset()
    return {"status": "passed", "raw_regeneration": len(checks), "incremental_checks": checks,
            **v3.check_temporal(runtime, work), **v3.check_columns(runtime, work), **base.check_rows(runtime, work),
            "tolerance": "temporal decision and column rows bitwise; keys atol=rtol=3e-6 vs GPU full prefix; winner exact; confidence abs<=1e-6 vs revision 2 reference query and ledger; tau* acceptance unchanged; bitwise equal to revision 3"}


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
                         "winner": result[0], "confidence": result[1], "rerank_candidates": result[2],
                         "transient_derived_bytes": runtime.transient})
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
    assert error <= v3.SCORE_TOLERANCE
    np.testing.assert_array_equal(actual >= tau, expected >= tau)
    third = bench.read(v3.OUT / f"nsd_n{n_map}.json")
    assert [(r["query"], r["winner"], r["confidence"]) for r in rows] == \
           [(r["query"], r["winner"], r["confidence"]) for r in third["raw_samples"]]
    checked["frozen_ledger"] = {"path": str(LEDGER.relative_to(REPO)), "sha256": bench.sha(LEDGER),
                                "queries": len(rows), "winners_exact": True, "score_max_abs_error": error,
                                "scores_bit_exact": bool(np.array_equal(actual, expected)),
                                "tau_star": tau, "acceptance_unchanged": True,
                                "min_ledger_distance_to_tau": float(np.abs(expected-tau).min())}
    checked["revision3_identity"] = {"path": str((v3.OUT / f"nsd_n{n_map}.json").relative_to(REPO)),
                                     "winners_and_confidences_bitwise_equal": True}
    bench.write(OUT / f"verification_n{n_map}.json", checked)
    previous = bench.read(base.OUT / f"nsd_n{n_map}.json")
    first = bench.read(bench.OUT / "methods" / f"nsd_n{n_map}.json")
    result = {"schema": "pdv2-table3-measurement-v1", "method": "nsd", "n_map": n_map,
              "status": "passed", "publish_ready": False, "protocol_revision": 4,
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
              "parent_revision2": {"timings_ms": previous["timings_ms"], "resident_total_bytes": previous["resident_bytes"]["total"]},
              "parent_revision1": {"timings_ms": first["timings_ms"], "resident_total_bytes": first["resident_bytes"]["total"]},
              "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "platform": platform.platform(), "threads": 1},
              "scope": spec["scope"], "changes": config["changes"], "completed_unix": time.time()}
    bench.write(target, result)
    print("COMPLETE nsd_v4", n_map, result["timings_ms"]["processing"], result["resident_bytes"]["total"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("measure",))
    parser.add_argument("--n-map", type=int, choices=SIZES, default=10000)
    args = parser.parse_args()
    measure(args.n_map)
