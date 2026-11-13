#!/usr/bin/env python3
"""정확max RING++ 부록 지도크기(1천·5천) 비용: 1만 캐시의 앞부분을 그대로 쓰고 같은 측정을 한다."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _ringpp_exactmax_local_cost_20261107 as cost

bench = cost.bench
_configure = cost.configure


def configure() -> dict:
    """동결 규약(1만 위치 기록)과 비교할 때만 1만 경로를 쓰고 이후엔 크기별 경로로 돌린다."""
    saved = cost.CACHE, cost.AUDIT
    cost.CACHE, cost.AUDIT = cost.LARGE / "ringpp_batch1/n10000", cost.LARGE / "audit/n10000"
    try:
        return _configure()
    finally:
        cost.CACHE, cost.AUDIT = saved


cost.configure = configure


def derive(n_map: int) -> None:
    """프레임별로 결정적인 1만 캐시의 앞 n+140행을 새 파일로 옮기고 해시를 남긴다."""
    config = configure()["base_cache_protocol"]
    source = cost.LARGE / "ringpp_batch1/n10000"
    parent = bench.read(source / "complete.json")
    assert parent["config"] == config
    target = cost.CACHE
    if (target / "complete.json").exists():
        assert bench.read(target / "complete.json")["config"] == config
        print("DERIVED_ALREADY_COMPLETE", n_map, flush=True)
        return
    target.mkdir(parents=True, exist_ok=True)
    work = bench.Workload(bench.read(bench.OUT / "protocol.json"), n_map)
    ids = np.load(source / "scan_ids.npy")
    np.testing.assert_array_equal(ids[:work.end], work.ids)
    for name, dtype in (("raw.npy", np.float32), ("fft.npy", np.complex64)):
        parent_map = np.load(source / name, mmap_mode="r")
        shape = (work.end, *parent_map.shape[1:])
        with (target / name).open("wb") as handle:
            cost.write_header(handle, dtype, shape)
            for lo in range(0, work.end, cost.BLOCK):
                handle.write(np.ascontiguousarray(parent_map[lo:min(lo + cost.BLOCK, work.end)]).tobytes())
                cost.release(parent_map)
        child = np.load(target / name, mmap_mode="r")
        for lo in range(0, work.end, cost.BLOCK):
            hi = min(lo + cost.BLOCK, work.end)
            assert cost.block_sha(child[lo:hi]) == cost.block_sha(parent_map[lo:hi])
            cost.release(child)
            cost.release(parent_map)
        del child, parent_map
    np.save(target / "scan_ids.npy", work.ids)
    np.save(target / "point_counts.npy", np.load(source / "point_counts.npy")[:work.end])
    bench.write(target / "complete.json", {
        "config": config, "frames": work.end, "batch": 1, "raw_input_sha256": work.input_hash,
        "derived_from": str(source), "derived_from_complete_sha256": bench.sha(source / "complete.json"),
        "derivation": "앞 n+140행 복사. 각 행은 그 스캔만으로 결정되는 정확max TI-RING·FFT이므로 지도 크기와 무관",
        "derive_source_sha256": bench.sha(Path(__file__)),
        "native_allocation_audit": cost.original.native_audit(),
        "files": {name: bench.sha(target / name) for name in ("raw.npy", "fft.npy", "scan_ids.npy", "point_counts.npy")}})
    print("DERIVED", n_map, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("derive", "measure"))
    parser.add_argument("--n-map", type=int, choices=(1000, 5000), required=True)
    args = parser.parse_args()
    cost.N_MAP = args.n_map
    cost.CACHE = cost.LARGE / f"ringpp_batch1/n{args.n_map}"
    cost.AUDIT = cost.LARGE / f"audit/n{args.n_map}"
    try:
        derive(args.n_map) if args.stage == "derive" else cost.measure()
    except cost.LowMemory as error:
        cost.status("paused_low_memory", reason=str(error))
        raise SystemExit(cost.EXIT_LOW_MEMORY)
