#!/usr/bin/env python3
"""RING# 원본128각도 CNN과 작은 각도 묶음의 전후보 점수·메모리를 대조한다."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import numpy as np
import torch
import _pdv2_table3_benchmark as bench
import _pdv2_table3_ringsharp as original
import _pdv2_table3_ringsharp_bounded as bounded

OUT = bench.OUT / "ringsharp_workspace_diagnostic"


def configure() -> None:
    """기존 CPU1·FP32 정책을 유지한다."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def memory() -> dict:
    """현재 할당·예약·최대값과 장치 여유를 읽는다."""
    bench.sync()
    free, total = torch.cuda.mem_get_info()
    return {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(), "driver_free": free,
            "device_total": total, "fft_plans": torch.backends.cuda.cufft_plan_cache.size}


def parts(runtime, q: int) -> tuple:
    """공식 전체 회전 검색과 raw128채널 질의 특징을 만든다."""
    cand = np.arange(q - 29)
    dist, angle = original.official.rotation_branch(runtime.base, runtime.spec,
                                                   runtime.spec[q:q + 1], cand)
    top = np.argsort(dist)[:original.official._depth(len(cand))]
    feature = original.official.query_features(runtime.base, runtime.model,
                                               runtime.packed[q:q + 1], "cuda")
    return top, angle[top], feature


def run(n_map: int, samples: int) -> None:
    """같은 캐시 특징96질의의 전체1000후보 점수와 승자를 대조한다."""
    configure()
    work = bench.Workload(bench.read(bench.OUT / "protocol.json"), n_map)
    runtime = original.Runtime(work)
    for start in range(0, work.end, 128):
        end = min(start + 128, work.end)
        runtime.spec[start:end].copy_(torch.from_numpy(np.array(runtime.sources["spec"][start:end], copy=True)))
        runtime.bt[start:end].copy_(torch.from_numpy(np.array(runtime.sources["bt"][start:end], copy=True)))
    runtime.packed[:] = runtime.sources["packed"]
    reference = []
    cases = []
    for angle_batch in (128, 8, 16):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for i in range(samples):
            q = n_map + 29 + i
            bench.sync()
            started = time.perf_counter()
            top, angles, feature = parts(runtime, q)
            before_translation = memory()
            if angle_batch == 128:
                error = original.official.translation_branch(runtime.base, runtime.trans, runtime.bt,
                                                             feature, top, angles, False, dedup=True)
            else:
                error = bounded.translation_branch(runtime.base, runtime.trans, runtime.bt,
                                                    feature, top, angles, angle_batch)
            bench.sync()
            elapsed = (time.perf_counter() - started) * 1000
            if angle_batch == 128:
                reference.append({"top": top.copy(), "angles": angles.copy(), "error": error.copy()})
            ref = reference[i]
            np.testing.assert_array_equal(top, ref["top"])
            np.testing.assert_array_equal(angles, ref["angles"])
            difference = float(abs(error - ref["error"]).max())
            winner = int(np.argmin(error))
            expected = int(np.argmin(ref["error"]))
            margin = float(np.partition(ref["error"], 1)[1] - ref["error"].min())
            assert difference < 1e-5 and winner == expected, (angle_batch, i, difference, margin)
            unique = np.unique(np.concatenate([angles, angles - np.pi]))
            rows.append({"query": q, "query_ms_diagnostic": elapsed, "top_count": len(top),
                         "unique_angles": len(unique), "all_translation_score_max_error": difference,
                         "winner": int(top[winner]), "reference_margin": margin,
                         "error_over_margin": difference / max(margin, 1e-30),
                         "before_translation": before_translation, "after_translation": memory()})
            del feature
            if i % 24 == 0:
                print("RINGSHARP_DIAG", n_map, angle_batch, i, round(elapsed, 2), flush=True)
        final = memory()
        torch.cuda.empty_cache()
        cases.append({"angle_batch": angle_batch, "rows": rows, "memory": final,
                      "after_empty_cache": memory(),
                      "timing_ms_diagnostic": bench.stats([r["query_ms_diagnostic"] for r in rows]),
                      "max_score_error": max(r["all_translation_score_max_error"] for r in rows),
                      "all_winners_equal": True})
    loop_checks = []
    for i in (0, samples - 1):
        q = n_map + 29 + i
        top, angles, feature = parts(runtime, q)
        ref = original.official.translation_branch(runtime.base, runtime.trans, runtime.bt,
                                                   feature, top, angles, True, dedup=False)
        difference = float(abs(ref - reference[i]["error"]).max())
        assert difference < 1e-5 and int(np.argmin(ref)) == int(np.argmin(reference[i]["error"]))
        loop_checks.append({"query": q, "all_candidates": len(top), "max_score_error": difference,
                            "winner": int(top[int(np.argmin(ref))])})
        del feature
        torch.cuda.empty_cache()
    source = [Path(__file__), Path(bounded.__file__), Path(original.__file__),
              Path(original.official.__file__), bench.REPO / "scripts/_ringsharp_eval.py"]
    target = OUT / f"n{n_map}"
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / "official_reference.npz",
             top=np.stack([r["top"] for r in reference]),
             angles=np.stack([r["angles"] for r in reference]),
             errors=np.stack([r["error"] for r in reference]))
    report = {"status": "passed", "n_map": n_map, "samples": samples, "cases": cases,
              "official_full_candidate_loop_checks": loop_checks,
              "source_sha256": {str(p.relative_to(bench.REPO)): bench.sha(p) for p in source},
              "method_config": original.settings(), "reference_sha256": bench.sha(target / "official_reference.npz"),
              "scope": "same fixed official cached features; only unique-angle rotate→trans_cnn batch changes; rotation/top1000/candidate FFT batch128 unchanged; diagnostic timings not Table3 timings"}
    bench.write(target / "summary.json", report)
    print("RINGSHARP_DIAG_COMPLETE", n_map, [(r["angle_batch"], r["max_score_error"], r["memory"]["peak_reserved"]) for r in cases], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-map", type=int, default=5000)
    parser.add_argument("--samples", type=int, default=96)
    args = parser.parse_args()
    run(args.n_map, args.samples)
