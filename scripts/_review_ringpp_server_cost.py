#!/usr/bin/env python3
"""WSL 중단 뒤 서버에서 RING++ 정확max를 검산하고 별도 비용을 잰다."""
from __future__ import annotations

import argparse
import gc
import platform
from pathlib import Path
import time

import numpy as np
import torch
import _pdv2_table3_benchmark as bench
import _pdv2_table3_ring as original
import _pdv2_table3_ringpp_v3 as measured
import _pdv2_table3_ringpp_v2 as checks
import _review_native_ring_maxfix as fix

OUT = bench.REPO / "results/pdv2_seqot_ring_repair_20261106/ringpp_server_cost_v2"


def install_input_mapping() -> None:
    """복사한 원시 파일의 원래 스캔 번호를 보존하고 공식 로더로 읽는다."""
    import _pd_build_test_caches as loaders
    from data.mulran_loader import MulRanLoader

    root = bench.REPO / "data/table3_server_input/mulran/Sejong01"
    manifest = bench.read(root / "scan_mapping.json")

    class MappedInput:
        def __init__(self):
            self.scan_files = {int(k): root / name for k, name in manifest["scan_files"].items()}

        def __getitem__(self, index):
            return {"points": MulRanLoader._load_point_cloud(self, index)}

    def load(sensor, sequence):
        assert (sensor, sequence) == ("mulran", "Sejong01")
        return MappedInput()

    loaders._make_loader = load
    original.reference.make_loader = load


def configure() -> dict:
    """입력·수식은 유지하고 서버 측정을 로컬 표3과 분리한다."""
    original.configure()
    install_input_mapping()
    original.OUT = OUT
    original.CHUNK = 128
    config = original.settings("ringpp")
    config.update({
        "schema": "ringpp-exactmax-server-cost-v1", "pairs_per_chunk": 128,
        "encoder": config["encoder"] + "; verified exact np.fmax.at grid maximum",
        "scope": "서버 RTX PRO 6000 측정. 로컬 RTX5080 표3을 대체하지 않음. 외부 CUDA 할당의 서버 추적 없음.",
        "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "platform": platform.platform()},
    })
    for path in (Path(__file__), Path(fix.__file__), Path(measured.__file__), Path(checks.__file__)):
        config["sources"][str(path.relative_to(bench.REPO))] = bench.sha(path)
    mapping = bench.REPO / "data/table3_server_input/mulran/Sejong01/scan_mapping.json"
    config["sources"][str(mapping.relative_to(bench.REPO))] = bench.sha(mapping)
    original.settings = lambda variant: config
    fix.install_maxfix()
    original.freeze(config)
    return config


def measure(config: dict) -> None:
    """실제 도착96개를3회 처리하고288질의를 공식 전수 상관과 대조한다."""
    workload_protocol = bench.read(OUT / "protocol.json")
    work = bench.Workload(workload_protocol, 10000)
    assert work.input_hash == workload_protocol["expected_raw_input_sha256"]
    runtime = measured.Runtime(config, work)
    verification = measured.warmup(runtime, work)
    actual = np.empty((bench.REPEATS, bench.SAMPLES, 6, 120, 120), np.float32)
    scores = np.zeros((bench.REPEATS, bench.SAMPLES, 10095), np.float64)
    rows, memory = [], []
    with torch.no_grad():
        for repeat in range(bench.REPEATS):
            runtime.reset()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for i, points in enumerate(work.points):
                t = work.initial + i
                value, encode_ms = bench.timed(lambda: runtime.encode(points))
                _, append_ms = bench.timed(lambda: runtime.append(value))
                found, query_ms = bench.timed(lambda: runtime.query_with_audit(t - 15))
                rows.append({"repeat": repeat, "arrival": t, "query": t - 15,
                             "n_candidates": t - 44, "encode": encode_ms,
                             "state_update": append_ms, "update": encode_ms + append_ms,
                             "query_ms": query_ms, "processing": encode_ms + append_ms + query_ms,
                             "winner": found[0], "confidence": found[1], "rerank_candidates": found[2]})
                actual[repeat, i] = value.cpu().numpy()
                scores[repeat, i, :found[2]] = found[3]
            info = original.Runtime.memory(runtime)
            info.update({"process_rss": bench.rss(), "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
                         "cuda_peak_reserved": torch.cuda.max_memory_reserved()})
            memory.append(info)
            print("SERVER_MEASURE", repeat + 1, flush=True)
    del value
    np.save(OUT / "actual_tiring.npy", actual)
    np.save(OUT / "candidate_scores.npy", scores)
    bench.write(OUT / "timed_rows.json", {"rows": rows, "memory": memory})
    verification["all_queries"] = checks.verify_all(runtime, work, actual, scores, rows)
    variation = checks.repeat_variation(actual, rows)
    assert variation["winner_changed_queries"] == 0
    assert variation["max_confidence_spread"] == 0
    assert all(v["max_abs_vs_repeat0"] == 0 for v in variation["raw_feature_changes"])
    test = bench.read(bench.REPO / "results/pdv2_ringpp_maxfix_20261105/protocol.json")
    fix_path = str(Path(fix.__file__).relative_to(bench.REPO))
    assert config["sources"][fix_path] == test["source_sha256"][fix_path]
    result = {"status": "passed", "publish_ready": False, "hardware": config["environment"],
              "not_table3_hardware": True, "queries": len(rows), "initial_candidates": 10000,
              "timings_ms": {key: bench.stats([row[field] for row in rows]) for key, field in
                             (("encode", "encode"), ("update", "update"), ("query", "query_ms"), ("processing", "processing"))},
              "memory_repeats": memory, "verification": verification, "repeat_variation": variation,
              "raw_input_sha256": work.input_hash, "same_test_fix_source": True,
              "protocol_sha256": bench.sha(OUT / "ringpp_protocol.json"),
              "artifacts_sha256": {p.name: bench.sha(p) for p in (OUT / "actual_tiring.npy", OUT / "candidate_scores.npy", OUT / "timed_rows.json")},
              "scope": config["scope"], "completed_unix": time.time()}
    bench.write(OUT / "measurement.json", result)
    print("SERVER_COMPLETE", result["timings_ms"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cache", "measure"))
    args = parser.parse_args()
    cfg = configure()
    if args.stage == "cache":
        original.cache("ringpp", 10000)
        complete = bench.read(original.directory("ringpp", 10000) / "complete.json")
        assert complete["raw_input_sha256"] == bench.read(OUT / "protocol.json")["expected_raw_input_sha256"]
    else:
        measure(cfg)
