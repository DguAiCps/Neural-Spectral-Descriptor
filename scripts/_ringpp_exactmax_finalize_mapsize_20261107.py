#!/usr/bin/env python3
"""정확max RING++ 1천·5천·1만 비용을 검산하고 부록 지도크기 표 행을 만든다(호스트 CPU)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import _ringpp_exactmax_finalize_20261107 as main

REPO, WORK, COST, LARGE = main.REPO, main.WORK, main.COST, main.LARGE


def check(n: int) -> tuple:
    """한 지도 크기 결과의 산식·결정성·공식 검산·해시를 확인한다."""
    result = main.read(COST / f"methods/ringpp_n{n}.json")
    rows, memory = result["raw_samples"], result["resident_bytes"]
    variation = result["official_native_repeat_variation"]
    official = result["verification"]["all_queries_same_execution"]
    cache = result["reference_batch1_cache"]
    checks = {
        "status": result["status"] == "passed" and len(rows) == 288,
        "timings": all(abs(main.stats([r[f] for r in rows])["mean"] - result["timings_ms"][k]["mean"]) < 1e-9
                       for k, f in (("update", "update"), ("query", "query_ms"), ("processing", "processing"))),
        "env": "5080" in result["environment"]["gpu"] and result["environment"]["torch_threads"] == 1,
        "memory_formula": memory["total"] == memory["cpu"] + memory["torch_cuda_allocated"] + memory["external_requested_gpu_bytes"],
        "native_300": memory["external_feature_encode_calls"] == 300,
        "deterministic": variation["winner_changed_queries"] == 0 and variation["max_confidence_spread"] == 0
                         and all(v["max_abs_vs_repeat0"] == 0 for v in variation["raw_feature_changes"]),
        "official_288": official["queries"] == 288 and official["status"] == "passed",
        "artifacts": all(main.sha(main.host(p)) == d for p, d in result["audit_artifacts"].items()),
        "cache": all(main.sha(LARGE / f"ringpp_batch1/n{n}" / name) == d for name, d in cache["files"].items()),
        "fresh_equals_cache": all(c["fresh_vs_cached_tiring_max_abs"] == 0 for c in result["verification"]["fresh_feature_direct_score_checks"]),
    }
    if n != 10000:
        parent = LARGE / "ringpp_batch1/n10000"
        child = LARGE / f"ringpp_batch1/n{n}"
        checks["prefix_of_10000"] = cache["derived_from_complete_sha256"] == main.sha(parent / "complete.json")
        for name in ("raw.npy", "fft.npy"):
            a, b = np.load(parent / name, mmap_mode="r"), np.load(child / name, mmap_mode="r")
            checks[f"prefix_equal_{name}"] = all(np.array_equal(a[i:min(i + 128, len(b))], b[i:i + 128]) for i in range(0, len(b), 128))
    return result, checks


def main_() -> None:
    results, checks = {}, {}
    for n in (1000, 5000, 10000):
        results[n], checks[n] = check(n)
    old = {m["n_map"]: m for m in main.read(main.TABLE3 / "summary.json")["measurements"] if m["method"] == "ringpp"}
    update = [results[n]["timings_ms"]["update"]["mean"] for n in (1000, 5000, 10000)]
    row = {"update_ms_mean_min_max": [sum(update) / 3, min(update), max(update)],
           "query_ms": {n: results[n]["timings_ms"]["query"]["mean"] for n in (1000, 5000, 10000)},
           "resident_mb": {n: results[n]["resident_bytes"]["total"] / 1e6 for n in (1000, 5000, 10000)},
           "processing_ms": {n: results[n]["timings_ms"]["processing"]["mean"] for n in (1000, 5000, 10000)}}
    old_update = [old[n]["timings_ms"]["update"]["mean"] for n in (1000, 5000, 10000)]
    old_row = {"update_ms_mean_min_max": [sum(old_update) / 3, min(old_update), max(old_update)],
               "query_ms": {n: old[n]["timings_ms"]["query"]["mean"] for n in (1000, 5000, 10000)},
               "resident_mb": {n: old[n]["resident_bytes"]["total"] / 1e6 for n in (1000, 5000, 10000)}}
    cells = (f"{row['update_ms_mean_min_max'][0]:.2f} [{row['update_ms_mean_min_max'][1]:.2f}, {row['update_ms_mean_min_max'][2]:.2f}]",
             *[f"{row['query_ms'][n]:.2f}" for n in (1000, 5000, 10000)],
             *[f"{row['resident_mb'][n]:.2f}" for n in (1000, 5000, 10000)])
    out = {"passed": all(all(c.values()) for c in checks.values()), "checks": {str(k): v for k, v in checks.items()},
           "exactmax_s17_row": row, "oldkernel_s17_row": old_row,
           "s17_latex_cells": "RING++ & " + " & ".join(cells) + r" \\",
           "sources": {str(n): str((COST / f"methods/ringpp_n{n}.json").relative_to(REPO)) for n in (1000, 5000, 10000)},
           "source_sha256": main.sha(Path(__file__))}
    (WORK / "costs_mapsize.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": out["passed"], "row": out["s17_latex_cells"]}, ensure_ascii=False))
    raise SystemExit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main_()
