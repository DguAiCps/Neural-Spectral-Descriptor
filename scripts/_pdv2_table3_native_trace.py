#!/usr/bin/env python3
"""표 3과 분리한 RING++ CUDA 요청 누수·반복 추론 진단이다."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_name] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_table3_20261029/native_trace"
SOURCE = Path(__file__).with_suffix(".cpp")
LIBRARY = OUT / "libnsd_cuda_trace.so"


def write(path: Path, obj: dict) -> None:
    """진단 JSON을 원자적으로 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def build() -> None:
    """GPU 실행 없이 CUDA ABI 추적기를 컴파일한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["g++", "-std=c++17", "-O2", "-fPIC", "-shared",
                    "-I/usr/local/cuda/include", str(SOURCE), "-o", str(LIBRARY),
                    "-ldl", "-pthread"], check=True)


def events(path: Path) -> list:
    """flush된 JSONL 추적 기록을 읽는다."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def snapshots(path: Path) -> dict:
    """호출자가 지정한 단계별 살아 있는 요청 집합을 반환한다."""
    return {row["label"]: row for row in events(path) if row["type"] == "snapshot"}


def selftest() -> None:
    """실제 CUDA를 부르지 않는 모의 라이브러리로 할당·해제 추적을 검사한다."""
    build()
    root = OUT / "cpu_mock"
    root.mkdir(parents=True, exist_ok=True)
    runtime = root / "runtime.cpp"
    runtime.write_text(textwrap.dedent(r'''
        #include <cuda_runtime_api.h>
        #include <cstdlib>
        extern "C" cudaError_t cudaMalloc(void** p, size_t n) {
            *p = malloc(n); return cudaSuccess;
        }
        extern "C" cudaError_t cudaFree(void* p) { free(p); return cudaSuccess; }
        extern "C" cudaError_t cudaMalloc3DArray(cudaArray_t* p,
                const cudaChannelFormatDesc*, cudaExtent e, unsigned int) {
            *p = reinterpret_cast<cudaArray_t>(malloc(e.width*e.height*e.depth*4));
            return cudaSuccess;
        }
        extern "C" cudaError_t cudaFreeArray(cudaArray_t p) { free(p); return cudaSuccess; }
    '''))
    feature = root / "feature.cpp"
    feature.write_text(textwrap.dedent(r'''
        #include <cuda_runtime_api.h>
        extern "C" void feature() {
            void *a, *b, *c, *temporary;
            cudaMalloc(&a,120); cudaMalloc(&b,120); cudaMalloc(&c,120);
            cudaMalloc(&temporary,4096); cudaFree(temporary);
            cudaChannelFormatDesc channel{32,0,0,0,cudaChannelFormatKindFloat};
            cudaExtent extent{2,3,4}; cudaArray_t array;
            cudaMalloc3DArray(&array,&channel,extent,0); cudaFreeArray(array);
        }
    '''))
    main = root / "main.cpp"
    main.write_text(textwrap.dedent(r'''
        #include <dlfcn.h>
        extern "C" void feature();
        int main() {
            auto snap = reinterpret_cast<void (*)(const char*)>(dlsym(RTLD_DEFAULT,"nsd_native_trace_snapshot"));
            snap("before"); feature(); snap("one"); feature(); snap("two");
        }
    '''))
    for source, target, extra in (
        (runtime, root / "libmock_cuda.so", []),
        (feature, root / "libmock_voxelfeat.so", ["-L" + str(root), "-lmock_cuda"])):
        subprocess.run(["g++", "-std=c++17", "-fPIC", "-shared", "-I/usr/local/cuda/include",
                        str(source), "-o", str(target), *extra], check=True)
    exe = root / "check"
    subprocess.run(["g++", str(main), "-o", str(exe), "-L" + str(root),
                    "-lmock_voxelfeat", "-lmock_cuda", "-ldl"], check=True)
    trace = root / "trace.jsonl"
    trace.unlink(missing_ok=True)
    env = dict(os.environ, LD_PRELOAD=str(LIBRARY), NSD_CUDA_TRACE_PATH=str(trace),
               LD_LIBRARY_PATH=str(root), CUDA_VISIBLE_DEVICES="")
    subprocess.run([str(exe)], env=env, check=True)
    saved = snapshots(trace)
    assert [saved[name]["requested_live_bytes"] for name in ("before", "one", "two")] == [0, 360, 720]
    assert all(row["requested_bytes"] == 120 for row in saved["two"]["live"])
    assert all("mock_voxelfeat" in row["owner"] for row in saved["two"]["live"])
    array_calls = [r for r in events(trace) if r.get("api") == "cudaMalloc3DArray"]
    assert len(array_calls) == 2 and all(r["requested_bytes"] == 96 for r in array_calls)
    write(OUT / "cpu_selftest.json", {"status": "passed", "gpu_work": False,
          "requested_live_bytes": {name: saved[name]["requested_live_bytes"] for name in saved},
          "array_payload_per_allocation": 96, "array_freed": True,
          "owner_attribution": "mock_voxelfeat", "scope": "requested payload, not physical GPU allocation"})
    print("NATIVE_TRACE_CPU_SELFTEST passed", flush=True)


def snapshot(label: str) -> dict:
    """추적 라이브러리에 동기화된 현재 단계 표시를 남긴다."""
    import torch

    torch.cuda.synchronize()
    function = ctypes.CDLL(None).nsd_native_trace_snapshot
    function.argtypes = [ctypes.c_char_p]
    function.restype = None
    function(label.encode())
    free, total = torch.cuda.mem_get_info()
    return {"label": label, "device_free_bytes": free, "device_total_bytes": total,
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved()}


def worker(n_map: int, query_count: int, repeats: int, gallery_count: int) -> None:
    """실제 공식 RING++ 반복 출력과 native 요청 누적을 별도 프로세스에서 잰다."""
    import numpy as np
    import torch
    import _pdv2_table3_ring as adapter
    import _pdv2_table3_benchmark as bench

    adapter.configure()
    spec = bench.read(bench.OUT / "protocol.json")
    with np.load(bench.REPO / spec["input_paths"]["cache"]) as saved:
        scan_ids = saved["scan_ids"]
    first_arrival = n_map + 44
    query_ids = np.arange(first_arrival - 15, first_arrival - 15 + query_count)
    gallery_ids = np.unique(np.linspace(0, n_map - 1, gallery_count, dtype=np.int64))
    loader = adapter.reference.make_loader("mulran", "Sejong01")
    query_points = [loader[int(scan_ids[q])]["points"] for q in query_ids]
    gallery_points = [loader[int(scan_ids[j])]["points"] for j in gallery_ids]
    core = adapter.reference.import_ring_core()
    driver = [snapshot("loaded")]
    native_calls = 0
    for i in range(3):
        value, count = adapter.encode(core, "ringpp", query_points[0], check=True)
        native_calls += int(count >= 30)
        del value
    driver.append(snapshot("warmup_done"))
    gallery = []
    for points in gallery_points:
        value, count = adapter.encode(core, "ringpp", points, check=True)
        native_calls += int(count >= 30)
        gallery.append(value)
    gallery = torch.stack(gallery)
    driver.append(snapshot("gallery_done"))
    comparisons = []
    calls_after_gallery = native_calls
    for i, points in enumerate(query_points):
        values, scores, winners = [], [], []
        for repeat in range(repeats):
            label = f"query{i}_repeat{repeat}"
            value, count = adapter.encode(core, "ringpp", points, check=True)
            native_calls += int(count >= 30)
            distances, _ = core.batch_circorr(value[None].repeat(len(gallery), 1, 1, 1), gallery)
            score = -np.asarray(distances, np.float64).reshape(-1)
            values.append(value.cpu().numpy())
            scores.append(score)
            winners.append(int(gallery_ids[int(np.argmax(score))]))
            del value
            driver.append(snapshot(label))
        reference_value, reference_score = values[0], scores[0]
        comparisons.append({"query_keyframe": int(query_ids[i]),
            "raw_scan": int(scan_ids[query_ids[i]]), "winner_keyframes": winners,
            "unique_winners": len(set(winners)),
            "feature_max_abs_vs_first": [float(abs(v - reference_value).max()) for v in values],
            "feature_relative_l2_vs_first": [float(np.linalg.norm(v - reference_value) / max(np.linalg.norm(reference_value), 1e-12)) for v in values],
            "score_max_abs_vs_first": [float(abs(s - reference_score).max()) for s in scores],
            "max_score_each_repeat": [float(s.max()) for s in scores]})
    driver.append(snapshot("diagnosis_done"))
    trace = Path(os.environ["NSD_CUDA_TRACE_PATH"])
    traced = snapshots(trace)
    def native_bytes(row: dict) -> int:
        return sum(value for owner, value in row["by_owner"].items() if "voxelfeat" in owner)
    labels = [f"query{i}_repeat{r}" for i in range(query_count) for r in range(repeats)]
    baseline = native_bytes(traced["gallery_done"])
    increments, previous = [], baseline
    for label in labels:
        now = native_bytes(traced[label])
        increments.append({"label": label, "live_requested_bytes": now,
                           "increment_bytes": now - previous})
        previous = now
    all_events = events(trace)
    report = {
        "status": "completed", "measurement_type": "separate instrumented diagnosis; exclude from Table3 timings",
        "sequence": bench.SEQUENCE, "n_map": n_map, "query_count": query_count,
        "repeats": repeats, "gallery_keyframes": gallery_ids.tolist(),
        "gallery_scope": "fixed uniformly spaced subset of eligible past candidates; winner changes are diagnostic, not a full Table2 result",
        "native_feature_calls": native_calls, "native_calls_before_queries": calls_after_gallery,
        "source_predicted_requested_increment": 360, "observed_increments": increments,
        "every_increment_matches_360": bool(increments and all(r["increment_bytes"] == 360 for r in increments)),
        "voxelfeat_live_requested_at_loaded": native_bytes(traced["loaded"]),
        "voxelfeat_live_requested_at_end": native_bytes(traced["diagnosis_done"]),
        "native_owner_live_at_end": traced["diagnosis_done"]["by_owner"],
        "live_native_allocations": [r for r in traced["diagnosis_done"]["live"] if "voxelfeat" in r["owner"] or "voxelocc" in r["owner"] or "radon" in r["owner"]],
        "cuda_api_errors": [r for r in all_events if r.get("error", 0) != 0],
        "driver_snapshots": driver, "repeat_output_comparisons": comparisons,
        "physical_memory": "requested allocation payload is exact for intercepted APIs; CUDA rounding, context and physical committed pages are not inferred. mem_get_info is device-wide and recorded only as a diagnostic.",
        "array_memory": "CUDA array trace reports channel payload extent, not pitched/physical array allocation",
        "sources": {str(p.relative_to(REPO)): bench.sha(p) for p in (SOURCE, Path(__file__), Path(adapter.__file__))},
        "library_sha256": bench.sha(LIBRARY),
        "extensions": adapter.settings("ringpp")["compiled_extensions"],
        "native_source_audit": adapter.native_audit(),
    }
    write(OUT / f"ringpp_n{n_map}_audit.json", report)
    print("NATIVE_AUDIT", {"increment_matches_360": report["every_increment_matches_360"],
          "increments": [r["increment_bytes"] for r in increments],
          "max_feature_error": max(max(r["feature_max_abs_vs_first"]) for r in comparisons),
          "winner_changes": sum(r["unique_winners"] > 1 for r in comparisons)}, flush=True)


def run(n_map: int, queries: int, repeats: int, gallery: int) -> None:
    """추적 오버헤드가 별도 프로세스에만 들어가도록 워커를 시작한다."""
    if not LIBRARY.exists():
        build()
    trace = OUT / f"ringpp_n{n_map}_trace.jsonl"
    trace.unlink(missing_ok=True)
    env = dict(os.environ, LD_PRELOAD=str(LIBRARY), NSD_CUDA_TRACE_PATH=str(trace))
    subprocess.run([sys.executable, str(Path(__file__)), "worker", "--n-map", str(n_map),
                    "--queries", str(queries), "--repeats", str(repeats),
                    "--gallery", str(gallery)], env=env, check=True)


def derive(n_map: int) -> None:
    """완료된 본 계측의 실제 호출 수로 native 요청 payload만 보충한다."""
    audit_path = OUT / "ringpp_n1000_audit.json"
    audit = json.loads(audit_path.read_text())
    assert audit["every_increment_matches_360"] and not audit["cuda_api_errors"]
    assert audit["voxelfeat_live_requested_at_end"] == audit["native_feature_calls"] * 360
    measurement_path = OUT.parent / "methods" / f"ringpp_n{n_map}.json"
    measurement = json.loads(measurement_path.read_text())
    assert measurement["status"] == "passed"
    compiled = measurement["reference_batch1_cache"]["config"]["compiled_extensions"]
    assert compiled["voxelfeat"]["sha256"] == audit["extensions"]["voxelfeat"]["sha256"]
    memory = measurement["resident_bytes"]
    calls = int(memory["external_feature_encode_calls"])
    native = calls * 360
    known = int(memory["cpu"]) + int(memory.get("torch_cuda_allocated", memory["gpu"]))
    sidecar = {
        "status": "passed", "method": "ringpp", "n_map": n_map,
        "measurement_sha256": hashlib.sha256(measurement_path.read_bytes()).hexdigest(),
        "trace_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "compiled_voxelfeat_sha256": compiled["voxelfeat"]["sha256"],
        "actual_feature_encode_calls_in_measurement_process": calls,
        "traced_requested_bytes_per_call": 360,
        "native_live_requested_payload_bytes": native,
        "known_cpu_and_torch_cuda_bytes": known,
        "combined_numerical_buffer_payload_bytes": known + native,
        "physical_gpu_bytes": None,
        "scope": "completed measurement call counter times separately traced native allocation requests; no instrumentation overhead enters original timings; includes verification calls because their leaked requests remain live",
        "limitation": "combined value counts native requested payload, not physical CUDA rounding/committed pages; process RSS and driver free memory remain separate diagnostics",
    }
    path = OUT.parent / "memory_audit" / f"ringpp_n{n_map}_native.json"
    write(path, sidecar)
    print(path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "selftest", "run", "worker", "derive"))
    parser.add_argument("--n-map", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gallery", type=int, default=32)
    args = parser.parse_args()
    if args.action == "derive":
        derive(args.n_map)
    elif args.action in ("build", "selftest"):
        globals()[args.action]()
    else:
        globals()[args.action](args.n_map, args.queries, args.repeats, args.gallery)
