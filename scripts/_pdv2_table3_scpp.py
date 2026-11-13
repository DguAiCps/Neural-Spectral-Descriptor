#!/usr/bin/env python3
"""공식 MATLAB A-PC의 입력 준비·동결·실제 증분 계측을 연결한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_table3_20261029"
MATLAB_SOURCE = REPO / "scripts/_pdv2_table3_scpp.m"
MEX_SOURCE = REPO / "scripts/_pdv2_table3_scpp_buffers.c"


def read(path: Path) -> dict:
    """JSON 기록을 읽는다."""
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    """완성된 JSON만 노출한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path: Path) -> str:
    """파일 SHA256을 구한다."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def folder(n_map: int) -> Path:
    """지도 크기별 원시 입력과 MATLAB 작업 경로다."""
    return OUT / f"scpp_n{n_map}"


def official_sources() -> dict:
    """표2와 같은 수정하지 않은 공식 소스를 기록한다."""
    root = REPO / "_external/scancontext_tro/tro2022/scancontext"
    files = sorted((root / "desc").glob("*.m")) + sorted((root / "alg").glob("*.m"))
    files += [REPO / "scripts/scpp_official/scpp_official_describe.m",
              REPO / "scripts/scpp_official/scpp_official_match.m"]
    return {str(path.relative_to(REPO)): sha(path) for path in files}


def prepare(n_map: int) -> None:
    """컨테이너 공통 로더의 첫 N+140 점군을 손실 없이 내보낸다."""
    sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
    from _pd_build_test_caches import _make_loader

    spec = read(OUT / "protocol.json")
    initial, end = n_map + 44, n_map + 44 + spec["samples"]
    work = folder(n_map)
    work.mkdir(parents=True, exist_ok=True)
    os.chmod(work, 0o777)  # 컨테이너 root 생성물을 호스트 MATLAB 계정도 갱신한다.
    meta_path = work / "input.json"
    if meta_path.exists():
        meta = read(meta_path)
        assert meta["parent_protocol_sha256"] == sha(OUT / "protocol.json")
        for name, digest in meta["files_sha256"].items():
            assert sha(work / name) == digest
        return
    with np.load(REPO / spec["input_paths"]["cache"]) as data:
        ids = data["scan_ids"][:end].copy()
    assert len(ids) == end
    loader = _make_loader("mulran", "Sejong01")
    raw_hash = hashlib.sha256()
    raw_bytes = 0
    counts = np.empty(end, np.int64)
    started = time.time()
    temporary = work / "chunk_000000.xyz.part"
    with temporary.open("wb") as handle:
        for index, scan_id in enumerate(ids):
            points = np.ascontiguousarray(loader[int(scan_id)]["points"])
            if index >= initial:
                raw_hash.update(points.tobytes())
                raw_bytes += points.nbytes
            xyz = np.ascontiguousarray(points[:, :3], np.float32)
            assert np.isfinite(xyz).all()
            counts[index] = len(xyz)
            handle.write(xyz.tobytes())
            if index % 100 == 0:
                print("SC++_RAW", index, end, round(time.time() - started, 1), flush=True)
    temporary.replace(work / "chunk_000000.xyz")
    counts.tofile(work / "chunk_000000.cnt")
    np.save(work / "scan_ids.npy", ids)
    meta = {"schema": "pdv2-table3-scpp-input-v1", "sequence": spec["sequence"],
            "n_map": n_map, "initial": initial, "end": end,
            "parent_protocol_sha256": sha(OUT / "protocol.json"),
            "raw_input_sha256": raw_hash.hexdigest(), "raw_input_fixture_bytes": raw_bytes,
            "xyz_input_fixture_bytes": int(counts[initial:].sum()) * 3 * 4,
            "files_sha256": {name: sha(work / name) for name in
                             ("chunk_000000.xyz", "chunk_000000.cnt", "scan_ids.npy")},
            "point_counts": counts.tolist(), "seconds": time.time() - started}
    write(meta_path, meta)
    for path in work.iterdir():
        if path.is_file():
            os.chmod(path, 0o666)


def settings(n_map: int) -> dict:
    """공통 프로토콜을 변경하지 않고 공식 SC++ 경로를 동결한다."""
    ledger = REPO / "results/pdv2_scpp_official_hsens/apc/odom/dump_scpp_mulran_DCC01.npz"
    with np.load(ledger) as data:
        reference = json.loads(str(data["meta"]))
    assert reference["sc_params"]["lidar_height"] == 1.693
    assert reference["sc_params"]["k_candidates"] == 1
    commit = subprocess.check_output(["git", "-C", str(REPO / "_external/scancontext_tro"),
                                      "rev-parse", "HEAD"], text=True).strip()
    assert commit == "b463bb7159a883ae8c36b5825547299c5bbded26"
    return {"schema": "pdv2-table3-scpp-protocol-v1", "n_map": n_map,
            "parent_protocol_sha256": sha(OUT / "protocol.json"),
            "official_commit": "b463bb7159a883ae8c36b5825547299c5bbded26",
            "variant": "apc", "lidar_height": 1.693, "search_ratio": 0,
            "n_aug": 3, "num_ring": 20, "num_sector": 60, "max_range": 80,
            "threads": 1, "retrieval": "official createns exact eligible prefix; knnsearch K=1; official aligning-key shift and SC distance; NaN and ties unchanged",
            "encoding": "double conversion, sensor-height addition, pointCloud, unchanged ptcloud2polarcontextAug with 0.5m grid; 3 original/lateral augmented descriptors",
            "memory": "unique mxGetData numerical buffers in allocated D/RK/SK and recursively expanded KDTreeSearcher; duplicate aliases counted once; object/cell metadata and interpreter tracked only in RSS",
            "table2_reference": str(ledger.relative_to(REPO)), "table2_reference_sha256": sha(ledger),
            "initial_cache_files": {name: sha(folder(n_map) / name) for name in
                                    ("input.json", "desc_000000.mat", "official_reference.csv",
                                     "official_reference.csv.json", "scpp_buffers_mex.mexa64")},
            "official_sources": official_sources(),
            "adapter_sources": {str(path.relative_to(REPO)): sha(path) for path in
                                (Path(__file__), MATLAB_SOURCE, MEX_SOURCE)}}


def matlab(action: str, n_map: int) -> None:
    """호스트 공식 MATLAB에서 준비 또는 승인된 계측을 실행한다."""
    work = folder(n_map)
    assert (work / "input.json").exists(), "먼저 컨테이너에서 prepare를 실행한다"
    (work / "pdv2_table3_scpp.m").write_bytes(MATLAB_SOURCE.read_bytes())
    if action == "measure":
        config = settings(n_map)
        target = work / "method_protocol.json"
        if target.exists():
            assert read(target) == config, "동결한 SC++ 계측 소스가 변경됐다"
        else:
            write(target, config)
    quoted_work = str(work).replace("'", "''")
    quoted_repo = str(REPO).replace("'", "''")
    command = f"addpath('{quoted_work}'); pdv2_table3_scpp('{action}', {n_map}, '{quoted_repo}');"
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    subprocess.run(["/usr/local/bin/matlab", "-singleCompThread", "-batch", command],
                   check=True, cwd=REPO, env=env)
    if action == "measure":
        collect(n_map)


def statistics(values: list) -> dict:
    """공통 Python 방식의 표본 통계를 계산한다."""
    array = np.asarray(values, np.float64)
    return {"mean": float(array.mean()), "p50": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "n": len(array)}


def collect(n_map: int) -> None:
    """공식 참조 대조를 통과한 계측만 표3 공통 스키마로 만든다."""
    work = folder(n_map)
    config = settings(n_map)
    assert read(work / "method_protocol.json") == config
    value = read(work / "measurement.json")
    assert value["verification"]["status"] == "passed"
    rows = value["raw_samples"]
    spec, inputs = read(OUT / "protocol.json"), read(work / "input.json")
    assert len(rows) == spec["samples"] * spec["repeats"]
    for i in range(spec["samples"]):
        repeated = [rows[k * spec["samples"] + i] for k in range(spec["repeats"])]
        assert len({row["winner"] for row in repeated}) == 1
        assert len({row["confidence"] for row in repeated}) == 1
    result = {"schema": "pdv2-table3-measurement-v1", "method": "scpp", "n_map": n_map,
              "status": "passed", "publish_ready": True,
              "protocol_sha256": sha(OUT / "protocol.json"),
              "source_sha256": sha(Path(__file__)), "method_protocol": config,
              "method_protocol_sha256": sha(work / "method_protocol.json"),
              "sequence": spec["sequence"], "seed": None,
              "candidate_range": [n_map, n_map + spec["samples"] - 1],
              "mean_candidates": n_map + (spec["samples"] - 1) / 2,
              "samples": spec["samples"], "repeats": spec["repeats"], "warmup": spec["warmup"],
              "timings_ms": {key: statistics([row[field] for row in rows]) for key, field in
                             (("encode", "encode"), ("state_update", "state_update"),
                              ("update", "update"), ("query", "query_ms"), ("processing", "processing"))},
              "resident_bytes": value["memory_repeats"][-1], "memory_repeats": value["memory_repeats"],
              "verification": value["verification"], "raw_samples": rows,
              "raw_input_fixture_bytes": inputs["raw_input_fixture_bytes"],
              "actual_preloaded_xyz_fixture_bytes": inputs["xyz_input_fixture_bytes"],
              "raw_input_sha256": inputs["raw_input_sha256"],
              "initial_cached_state_build_s": value["initial_cached_state_build_s"],
              "environment": value["environment"], "scope": spec["scope"], "completed_unix": time.time()}
    write(OUT / "methods" / f"scpp_n{n_map}.json", result)
    print("COMPLETE", "scpp", n_map, result["timings_ms"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "cache", "measure", "collect"))
    parser.add_argument("--n-map", type=int, choices=(1000, 5000, 10000), default=1000)
    args = parser.parse_args()
    if args.action in ("cache", "measure"):
        matlab(args.action, args.n_map)
    else:
        globals()[args.action](args.n_map)
