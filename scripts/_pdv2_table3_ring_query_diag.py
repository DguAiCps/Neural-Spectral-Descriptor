#!/usr/bin/env python3
"""공식 RING++ 점수를 유지하며 검색 청크·FFT 모양·할당자 상태를 진단한다."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_name] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import numpy as np
import torch
import _pdv2_table3_benchmark as bench
import _pdv2_table3_ring as original
from _causal_ring_pr import correlate_prefix_fast, precompute_fft_bank

OUT = bench.OUT / "ring_query_workspace_diagnostic"


def write(path: Path, value: dict) -> None:
    """별도 진단 원천을 JSON으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def padded_scores(bank: torch.Tensor, q: int, count: int, chunk: int) -> np.ndarray:
    """마지막 후보 묶음만 영으로 채우고 공식 수식의 유효 점수를 반환한다."""
    query = bank[q]
    channels, height, width = query.shape
    denom = .15 * channels * height * width
    result = np.empty(count, np.float64)
    with torch.no_grad():
        for first in range(0, count, chunk):
            last = min(first + chunk, count)
            candidate = bank[first:last]
            if last - first < chunk:
                padded = torch.zeros((chunk, channels, height, width),
                                     dtype=bank.dtype, device=bank.device)
                padded[:last - first].copy_(candidate)
                candidate = padded
            corr = torch.fft.ifft2(query.unsqueeze(0) * candidate.conj(),
                                   dim=-2, norm="ortho")
            magnitude = torch.sqrt(corr.real ** 2 + corr.imag ** 2 + 1e-15)
            scores = magnitude.sum(dim=-3).sum(dim=-1).max(dim=-1)[0] / denom
            result[first:last] = scores[:last - first].double().cpu().numpy()
    return result


def snapshot() -> dict:
    """동기화된 시점의 PyTorch 할당자·FFT 계획 캐시·드라이버 값을 읽는다."""
    bench.sync()
    stats = torch.cuda.memory_stats()
    free, total = torch.cuda.mem_get_info()
    keys = ("allocated_bytes.all.current", "allocated_bytes.all.peak",
            "reserved_bytes.all.current", "reserved_bytes.all.peak",
            "active_bytes.all.current", "inactive_split_bytes.all.current",
            "requested_bytes.all.current", "num_alloc_retries", "num_ooms")
    return {"allocator": {key: stats.get(key) for key in keys},
            "driver_free_bytes": free, "driver_total_bytes": total,
            "cufft_plan_cache_size": torch.backends.cuda.cufft_plan_cache.size,
            "cufft_plan_cache_max_size": torch.backends.cuda.cufft_plan_cache.max_size,
            "process_rss": bench.rss()}


def inputs(n_map: int) -> tuple:
    """고정한 공식 TI-RING·FFT 캐시를 읽고 해시로 확인한다."""
    target = original.directory("ringpp", n_map)
    meta = bench.read(target / "complete.json")
    for filename, expected in meta["files"].items():
        assert bench.sha(target / filename) == expected
    return (np.load(target / "raw.npy", mmap_mode="r"),
            np.load(target / "fft.npy", mmap_mode="r"), meta)


def source_config(n_map: int, meta: dict) -> dict:
    """기존 측정을 수정하지 않는 진단 원천과 특징을 고정한다."""
    paths = [Path(__file__), Path(original.__file__),
             bench.REPO / "scripts/_causal_ring_pr.py",
             bench.REPO / "scripts/_ring_official_pr.py",
             bench.REPO / "_external/RING/utils/circorr2.py"]
    return {"schema": "ringpp-query-workspace-diagnostic-v1", "n_map": n_map,
            "source_sha256": {str(p.relative_to(bench.REPO)): bench.sha(p) for p in paths},
            "reference_cache": meta, "threads": 1, "tf32": False,
            "native_encoding": False,
            "scope": "same fixed official TI-RING and FFT features; query only; no method or Table2 selection change; diagnostics are not Table3 timings"}


def reference(args: argparse.Namespace) -> None:
    """진단 전체에 공유할 공식 전수 상관 점수를 직접 계산한다."""
    original.configure()
    raw, _, meta = inputs(args.n_map)
    config = source_config(args.n_map, meta)
    target = OUT / f"n{args.n_map}"
    info = target / "reference.json"
    if info.exists():
        saved = bench.read(info)
        assert saved["config"] == config
        assert saved["samples"] == args.samples
        assert bench.sha(target / "official_scores.npy") == saved["score_sha256"]
        return
    from _ring_official_pr import official_scores
    core = original.reference.import_ring_core()
    # 참조 계산은 작은 공식 배치 16을 쓰고 전체 FFT GPU 뱅크를 만들지 않는다.
    raw_bank = torch.from_numpy(np.array(raw, copy=True))
    scores = np.full((args.samples, args.n_map + args.samples - 1), np.nan, np.float64)
    start = time.perf_counter()
    for i in range(args.samples):
        count = args.n_map + i
        scores[i, :count] = official_scores(core, raw_bank, args.n_map + 29 + i, count)
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "official_scores.npy", scores)
    write(info, {"config": config, "samples": args.samples,
                 "score_sha256": bench.sha(target / "official_scores.npy"),
                 "seconds": time.perf_counter() - start,
                 "scope": "fixed-prefix scores are slices of the same official per-pair scores"})
    print("REFERENCE_COMPLETE", args.n_map, flush=True)


def case_name(args: argparse.Namespace) -> str:
    """실험 조건을 파일 이름으로 나타낸다."""
    return f"chunk{args.chunk}_{args.mode}_pad{int(args.pad_tail)}_{args.plan_mode}"


def case(args: argparse.Namespace) -> None:
    """새 프로세스에서 원본 검색과 마지막 청크 패딩의 상태를 기록한다."""
    original.configure()
    _, spectral, meta = inputs(args.n_map)
    config = source_config(args.n_map, meta)
    target = OUT / f"n{args.n_map}"
    saved_reference = bench.read(target / "reference.json")
    assert saved_reference["config"] == config
    assert saved_reference["samples"] == args.samples
    expected = np.load(target / "official_scores.npy")
    bank = torch.empty(spectral.shape, dtype=torch.complex64, device="cuda")
    for first in range(0, len(spectral), 512):
        last = min(first + 512, len(spectral))
        bank[first:last].copy_(torch.from_numpy(np.array(spectral[first:last], copy=True)))
    bench.sync()

    def query(index: int) -> np.ndarray:
        count = args.n_map + index if args.mode == "growing" else args.n_map
        q = args.n_map + 29 + index
        if args.pad_tail:
            return padded_scores(bank, q, count, args.chunk)
        return correlate_prefix_fast(bank, True, q, count, args.chunk, "cuda")

    phases = {"loaded": snapshot()}
    for i in range(min(args.warmup, args.samples)):
        query(i)
    phases["warmup"] = snapshot()
    rows, resets = [], []
    for repeat in range(args.repeats):
        gc.collect()
        torch.cuda.empty_cache()
        if args.plan_mode == "clear_after_warmup":
            torch.backends.cuda.cufft_plan_cache.clear()
        torch.cuda.reset_peak_memory_stats()
        resets.append({"repeat": repeat, "state": snapshot()})
        for i in range(args.samples):
            if args.plan_mode == "clear_each_query":
                torch.backends.cuda.cufft_plan_cache.clear()
            scores, elapsed = bench.timed(lambda: query(i))
            count = len(scores)
            error = float(abs(scores - expected[i, :count]).max())
            winner = int(np.argmax(scores))
            assert error < 1e-4, (repeat, i, error)
            assert winner == int(np.argmax(expected[i, :count])), (repeat, i, winner)
            rows.append({"repeat": repeat, "query": args.n_map + 29 + i,
                         "count": count, "query_ms": elapsed, "winner": winner,
                         "score_max_error": error, "memory": snapshot()})
        print("CASE_REPEAT", case_name(args), repeat + 1,
              round(np.mean([r["query_ms"] for r in rows[-args.samples:]]), 3), flush=True)
    phases["finished"] = snapshot()
    torch.cuda.empty_cache()
    phases["empty_cache_only"] = snapshot()
    torch.backends.cuda.cufft_plan_cache.clear()
    torch.cuda.empty_cache()
    phases["clear_plans_then_empty_cache"] = snapshot()
    result = {"status": "passed", "config": config,
              "case": {k: getattr(args, k) for k in ("chunk", "mode", "pad_tail", "plan_mode", "samples", "repeats", "warmup")},
              "timings_ms": bench.stats([r["query_ms"] for r in rows]),
              "max_score_error": max(r["score_max_error"] for r in rows),
              "all_official_winners_equal": True, "phases": phases,
              "reset_states": resets, "rows": rows,
              "allocator_environment": {key: os.environ.get(key) for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")},
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "completed_unix": time.time()}
    write(target / f"{case_name(args)}.json", result)
    print("CASE_COMPLETE", case_name(args), result["timings_ms"], flush=True)


def suite(args: argparse.Namespace) -> None:
    """조건마다 독립 프로세스로 실행하여 앞선 FFT 계획의 영향을 없앤다."""
    base = [sys.executable, str(Path(__file__).resolve())]
    common = ["--n-map", str(args.n_map), "--samples", str(args.samples),
              "--repeats", str(args.repeats), "--warmup", str(args.warmup)]
    subprocess.run(base + ["reference"] + common, check=True)
    for chunk in args.chunks:
        for mode in ("growing", "fixed"):
            for padded in (False, True):
                command = base + ["case"] + common + ["--chunk", str(chunk), "--mode", mode]
                if padded:
                    command.append("--pad-tail")
                subprocess.run(command, check=True)
    for policy in ("clear_after_warmup", "clear_each_query"):
        subprocess.run(base + ["case"] + common + ["--chunk", "1024", "--mode", "growing", "--plan-mode", policy], check=True)


def cpu_check() -> None:
    """합성 특징으로 청크 패딩이 유효 후보 점수를 보존하는지 확인한다."""
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(721)
    raw = torch.rand((24, 6, 12, 12), generator=generator)
    bank, _ = precompute_fft_bank(raw, "cpu")
    errors = []
    for count in (1, 7, 16, 17):
        direct = correlate_prefix_fast(bank, False, 23, count, 5, "cpu")
        for chunk in (4, 8, 16, 32):
            padded = padded_scores(bank, 23, count, chunk)
            error = float(abs(padded - direct).max())
            assert error < 1e-5
            assert int(np.argmax(padded)) == int(np.argmax(direct))
            errors.append(error)
    print(json.dumps({"status": "passed", "gpu_work": False,
                      "checks": len(errors), "max_padding_score_error": max(errors)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("reference", "case", "suite", "cpu_check"))
    parser.add_argument("--n-map", type=int, choices=(1000, 5000, 10000), default=1000)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--chunks", type=int, nargs="+", default=[1024, 128, 256, 512, 64])
    parser.add_argument("--mode", choices=("growing", "fixed"), default="growing")
    parser.add_argument("--pad-tail", action="store_true")
    parser.add_argument("--plan-mode", choices=("cached", "clear_after_warmup", "clear_each_query"), default="cached")
    args = parser.parse_args()
    assert 1 <= args.samples <= 96 and args.repeats >= 1 and args.chunk >= 1
    if args.action == "cpu_check":
        cpu_check()
    else:
        globals()[args.action](args)
