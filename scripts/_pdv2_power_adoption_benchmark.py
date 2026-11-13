#!/usr/bin/env python3
"""채택된 크기 압축 NSD를 기존 표3의 12워밍업·96×3 규약으로 잰다."""
from __future__ import annotations

import argparse
import gc
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

import _pdv2_spectral_power_timing as pilot

study, current, bench = pilot.study, pilot.current, pilot.bench
OUT = study.OUT / "adoption/table3"


def measure(n_map: int) -> None:
    """독립 프로세스에서 초기 지도 하나를 측정하고 원장과 검산한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"nsd_n{n_map}.json"
    assert not target.exists(), target
    frozen = study.read(study.OUT / "test_freeze.json")
    selected = frozen["selected"]
    assert selected == "p050__aux_half__equal"
    sources = [Path(__file__), Path(pilot.__file__), Path(current.__file__),
               study.OUT / "test_freeze.json", study.OUT / "table2_summary.json"]
    protocol = {"selected": selected, "warmup": 12, "samples": 96, "repeats": 3,
                "scope": "표3와 같은 개별 워밍업12 이후 매 반복96개 전체 계측. 선택/재학습 없음. 초기상태 구축 제외.",
                "sources": {str(p.relative_to(study.REPO)): study.sha256(p) for p in sources}}
    pp = OUT / "protocol.json"
    if pp.exists():
        assert study.read(pp) == protocol
    else:
        study.write_json(pp, protocol)
    spec = bench.prepare()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    work = bench.Workload(spec, n_map)
    tick = time.perf_counter()
    runtime = pilot.Runtime(spec, work)
    build_s = time.perf_counter() - tick
    ref = current.base.reference_net(spec)
    checked = current.verify(runtime, work, ref)
    del ref
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    for i in range(bench.WARMUP):
        t = work.initial + i
        runtime.append(*runtime.encode(work.points[i]), work.poses[t])
        runtime.query_power(t - 15, selected)
    rows, memory = [], []
    for repeat in range(bench.REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rss_before = bench.rss()
        for i, points in enumerate(work.points):
            t = work.initial + i
            encoded, encode_ms = bench.timed(lambda: runtime.encode(points))
            _, state_ms = bench.timed(lambda: runtime.append(*encoded, work.poses[t]))
            result, query_ms = bench.timed(lambda: runtime.query_power(t - 15, selected))
            rows.append({"repeat": repeat, "arrival": t, "query": t - 15,
                         "encode": encode_ms, "state_update": state_ms, "update": encode_ms + state_ms,
                         "query_ms": query_ms, "processing": encode_ms + state_ms + query_ms,
                         "winner": result[0], "confidence": result[1], "rerank_candidates": result[2],
                         "transient_derived_bytes": runtime.transient})
        info = runtime.memory()
        info.update(process_rss_before=rss_before, process_rss_after=bench.rss(),
                    process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    cuda_allocated=torch.cuda.memory_allocated(), cuda_reserved=torch.cuda.memory_reserved(),
                    cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved())
        memory.append(info)
    path = pilot.cal.OUT / "val/s1/mulran_Sejong01.npz"
    values_path = study.OUT / "val/s1/mulran_Sejong01.npz"
    src, vals = study.arrays(path), study.arrays(values_path)
    fi = pilot.cal.FUSIONS.index("aux_half")
    indices = np.searchsorted(src["query_idx"], [r["query"] for r in rows])
    np.testing.assert_array_equal(src["query_idx"][indices], [r["query"] for r in rows])
    np.testing.assert_array_equal(src["candidates"][indices, fi], [r["winner"] for r in rows])
    expected = .5 * (30 * vals["p050_c_spec"][indices, fi] + vals["p050_c_col"][indices, fi])
    actual = np.array([r["confidence"] for r in rows])
    error = float(abs(expected - actual).max())
    assert len(rows) == 288 and error <= 1e-6
    checked["new_final_confidence"] = {"winners_exact": True, "queries": len(rows), "score_max_abs_error": error,
        "sources": {str(p.relative_to(study.REPO)): study.sha256(p) for p in (path, values_path)},
        "threshold": "고정 임계값 미보정. 기존 tau를 새 점수에 사용하지 않는다."}
    result = {"schema": "pdv2-table3-measurement-v1", "method": "nsd", "n_map": n_map,
              "status": "passed", "publish_ready": True, "selected": selected,
              "protocol_sha256": bench.sha(bench.OUT / "protocol.json"), "method_protocol_sha256": study.sha256(pp),
              "source_sha256": study.sha256(Path(__file__)), "sequence": bench.SEQUENCE, "seed": 1,
              "candidate_range": [n_map, n_map + 95], "mean_candidates": n_map + 47.5,
              "samples": 96, "repeats": 3, "warmup": 12,
              "timings_ms": {k: bench.stats([r[f] for r in rows]) for k, f in
                  (("encode", "encode"), ("state_update", "state_update"), ("update", "update"),
                   ("query", "query_ms"), ("processing", "processing"))},
              "resident_bytes": memory[-1], "memory_repeats": memory, "verification": checked,
              "transient_derived_bytes_max": max(r["transient_derived_bytes"] for r in rows),
              "raw_input_fixture_bytes": work.input_bytes, "raw_input_sha256": work.input_hash,
              "initial_cached_state_build_s": build_s, "raw_samples": rows,
              "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "platform": platform.platform(), "threads": 1},
              "scope": spec["scope"], "completed_unix": time.time()}
    study.write_json(target, result)
    print("COMPLETE", n_map, result["timings_ms"], result["resident_bytes"]["total"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-map", type=int, choices=(1000, 5000, 10000), required=True)
    measure(parser.parse_args().n_map)
