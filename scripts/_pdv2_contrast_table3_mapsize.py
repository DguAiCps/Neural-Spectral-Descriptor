#!/usr/bin/env python3
"""현재 NSD(window 구현)의 증분 비용을 표2와 같은 규약으로 지도 크기 N=1,000·5,000·10,000에서 잰다.

표2 비용 열의 원천(`_pdv2_contrast_table3_opt.py`)은 해시로 고정되어 있어 고치지 않고, 그 모듈의
`Runtime`·검사·상수를 그대로 가져와 초기 지도 크기만 바꾼다. 비교 방법의 지도 크기별 값은
`results/pdv2_table3_20261029/summary.json`(같은 Sejong01·96도착×3·워밍업12 규약)을 쓴다.
"""
import argparse
import gc
import platform
import resource
import time
from pathlib import Path

import _pdv2_contrast_table3_opt as O   # 스레드 환경변수를 먼저 고정하는 모듈 순서를 그대로 따른다

import numpy as np
import torch

T, bench, current = O.T, O.bench, O.current
ROOT = O.ROOT
OUT = ROOT / "results/pdv2_contrast_table3_mapsize_20261106"


def measure(n_map: int) -> None:
    """같은 96도착을 세 번 실행하고 고정 검증 원장과 일치하는지 검사한다."""
    current.e.configure()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"nsd_n{n_map}_window.json"
    assert not target.exists(), target
    before = O.environment_check()
    sources = [Path(__file__), Path(O.__file__), Path(T.__file__), T.MAIN / "protocol.json", T.LEDGER,
               ROOT / "src/gnn/contrast_spectral_difference.py", ROOT / "src/gnn/spectral_difference.py",
               ROOT / "src/gnn/learned_spectral_difference.py", ROOT / "src/encoding/spectral_encoder.py",
               ROOT / "src/encoding/range_image.py", ROOT / "src/encoding/bev_image.py",
               ROOT / "results/pdv2_contrast_table3_opt_20261106/protocol_window.json"]
    protocol = dict(method="diff_s1/matching_only", variant="window", warmup=12, samples=96, repeats=3, threads=1,
                    map_sizes=[1000, 5000, 10000], changes=O.CHANGES,
                    sources={str(p.relative_to(ROOT)): bench.sha(p) for p in sources},
                    checkpoint=bench.read(T.MAIN / "protocol.json")["cases"]["diff_s1"],
                    workload_protocol_sha256=bench.sha(bench.OUT / "protocol.json"),
                    scope="표2 비용 열과 같은 구현·규약에서 초기 지도 크기만 바꾼다. 초기구축·I/O·공통KISS·CUDA graph 캡처 제외. 시험 선택 없음.")
    path = OUT / "protocol_window.json"
    if path.exists():
        assert bench.read(path) == protocol
    else:
        bench.write(path, protocol)
    spec = bench.read(bench.OUT / "protocol.json")
    assert n_map in spec["map_sizes"]
    work = bench.Workload(spec, n_map)
    tick = time.perf_counter()
    runtime = O.Runtime(spec, work, nodewise=False)
    build_s = time.perf_counter() - tick
    print("BUILT", n_map, build_s, runtime.graph_pool, flush=True)
    identity = O.identity_checks(runtime, work)
    checked = T.verify(runtime, work)
    print("VERIFIED", n_map, checked["status"], identity["score_difference_max"], flush=True)
    current.previous.source.COEFFICIENT_CACHE[:] = [None, None]
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    for i in range(bench.WARMUP):
        t = work.initial + i
        runtime.append(*runtime.encode(work.points[i]), work.poses[t])
        runtime.query(t - 15)
    rows, memory = [], []
    for repeat in range(bench.REPEATS):
        runtime.reset()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for i, points in enumerate(work.points):
            t = work.initial + i
            encoded, encode_ms = bench.timed(lambda: runtime.encode(points))
            _, state_ms = bench.timed(lambda: runtime.append(*encoded, work.poses[t]))
            result, query_ms = bench.timed(lambda: runtime.query(t - 15))
            rows.append(dict(repeat=repeat, arrival=t, query=t - 15, encode=encode_ms, state_update=state_ms,
                             update=encode_ms + state_ms, query_ms=query_ms, processing=encode_ms + state_ms + query_ms,
                             winner=result[0], confidence=result[1], rerank_candidates=result[2],
                             transient_derived_bytes=runtime.transient))
        info = runtime.memory()
        info.update(process_rss=bench.rss(), process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    cuda_reserved=torch.cuda.memory_reserved(), cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved=torch.cuda.max_memory_reserved())
        memory.append(info)
    after = O.environment_check()
    ledger = dict(np.load(T.LEDGER))
    ids = np.searchsorted(ledger["query_idx"], [r["query"] for r in rows])
    np.testing.assert_array_equal(ledger["query_idx"][ids], [r["query"] for r in rows])
    np.testing.assert_array_equal(ledger["j_star"][ids], [r["winner"] for r in rows])
    error = float(abs(ledger["score"][ids] - [r["confidence"] for r in rows]).max())
    assert error < 8e-6 and len(rows) == 288, error
    checked["final_ledger"] = dict(queries=len(rows), winners_exact=True, score_max_error=error, sha256=bench.sha(T.LEDGER))
    checked["identity_vs_original_implementation"] = identity
    result = dict(schema="pdv2-table3-measurement-v1", method="nsd", selected="diff/matching_only", variant="window",
                  n_map=n_map, status="passed", seed=1, sequence=bench.SEQUENCE,
                  samples=96, repeats=3, warmup=12, candidate_range=[n_map, n_map + 95], mean_candidates=n_map + 47.5,
                  method_protocol_sha256=bench.sha(path), protocol_sha256=bench.sha(bench.OUT / "protocol.json"),
                  timings_ms={k: bench.stats([r[f] for r in rows]) for k, f in
                              (("encode", "encode"), ("state_update", "state_update"), ("update", "update"),
                               ("query", "query_ms"), ("processing", "processing"))},
                  resident_bytes=memory[-1], memory_repeats=memory, verification=checked,
                  initial_cached_state_build_s=build_s, raw_input_sha256=work.input_hash, raw_input_fixture_bytes=work.input_bytes,
                  raw_samples=rows, environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      cuda=torch.version.cuda, platform=platform.platform(), threads=1, before=before, after=after),
                  completed_unix=time.time())
    bench.write(target, result)
    print("COMPLETE", n_map, {k: round(v["mean"], 4) for k, v in result["timings_ms"].items()},
          result["resident_bytes"]["total"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-map", type=int, choices=(1000, 5000, 10000), required=True)
    measure(parser.parse_args().n_map)
