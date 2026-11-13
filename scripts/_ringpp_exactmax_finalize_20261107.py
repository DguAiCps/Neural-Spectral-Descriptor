#!/usr/bin/env python3
"""정확max RING++ 로컬 비용·46시험 결과를 검산하고 주표 교체용 수치 패키지를 만든다(호스트 CPU)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "results/pdv2_baseline_revision_20261107/ringpp"
COST = WORK / "cost"
LARGE = Path("/mnt/d/NSD_datasets/pdv2_ringpp_exactmax_cost_20261107")
CONTAINER_LARGE = "/workspace/data/pdv2_ringpp_exactmax_cost_20261107"
TABLE3 = REPO / "results/pdv2_table3_20261029"
SERVER = REPO / "results/pdv2_seqot_ring_repair_20261106/ringpp_server_cost_v2"
OLD_EXACTMAX_PROTOCOL = REPO / "results/pdv2_seqot_ring_repair_20261106/ringpp_cost/ringpp_exactmax_protocol.json"
FRAMES_FOR_KB = 10095
METRICS = ("AP", "max_F1", "R@100P", "R@1")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def host(path: str) -> Path:
    """컨테이너 경로를 호스트 경로로 바꾼다."""
    if path.startswith(CONTAINER_LARGE):
        return LARGE / path[len(CONTAINER_LARGE) + 1:]
    return REPO / path


def stats(values) -> dict:
    a = np.asarray(values, np.float64)
    return {"mean": float(a.mean()), "p50": float(np.median(a)), "p95": float(np.percentile(a, 95)), "n": len(a)}


def diff(a, b, path="") -> list:
    """두 JSON 값의 다른 경로 목록."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}/{key} ({'only_new' if key in a else 'only_old'})")
            else:
                out += diff(a[key], b[key], f"{path}/{key}")
        return out
    return [] if a == b else [path]


def cells(m: dict) -> dict:
    """표2 비용 열과 같은 산식(상주 KB/프레임=총바이트/10095/1000)."""
    return {"resident_kb_per_frame": m["resident_bytes"]["total"] / (FRAMES_FOR_KB * 1000),
            "update_ms": m["timings_ms"]["update"]["mean"], "query_ms": m["timings_ms"]["query"]["mean"],
            "processing_ms": m["timings_ms"]["processing"]["mean"],
            "resident_mb": m["resident_bytes"]["total"] / 1e6}


def main() -> None:
    checks = {}
    result = read(COST / "methods/ringpp_n10000.json")
    rows = result["raw_samples"]
    checks["status_passed"] = result["status"] == "passed"
    checks["rows_288"] = len(rows) == 288 and sorted({r["repeat"] for r in rows}) == [0, 1, 2]
    for key, field in (("encode", "encode"), ("state_update", "state_update"), ("update", "update"),
                       ("query", "query_ms"), ("processing", "processing")):
        again = stats([r[field] for r in rows])
        checks[f"timing_{key}_recomputed"] = all(abs(again[s] - result["timings_ms"][key][s]) < 1e-9 for s in ("mean", "p50", "p95"))
    checks["processing_is_update_plus_query"] = all(abs(r["processing"] - r["update"] - r["query_ms"]) < 1e-9 for r in rows)
    checks["environment_rtx5080_1thread"] = "5080" in result["environment"]["gpu"] and result["environment"]["threads"] == 1 \
        and result["environment"].get("torch_threads") == 1
    memory = result["resident_bytes"]
    checks["memory_total_formula"] = memory["total"] == memory["cpu"] + memory["torch_cuda_allocated"] + memory["external_requested_gpu_bytes"]
    checks["native_calls_300_x_360"] = memory["external_feature_encode_calls"] == 300 and memory["external_requested_gpu_bytes"] == 108000
    checks["fft_bank_on_gpu"] = memory["fft_bank_on_gpu"] and memory["breakdown"]["gpu_fft_bank"] == 10140 * 6 * 120 * 120 * 8
    variation = result["official_native_repeat_variation"]
    checks["repeat_deterministic"] = variation["winner_changed_queries"] == 0 and variation["max_confidence_spread"] == 0 \
        and all(v["max_abs_vs_repeat0"] == 0 and v["changed_frames"] == 0 for v in variation["raw_feature_changes"])
    official = result["verification"]["all_queries_same_execution"]
    checks["official_288_queries"] = official["status"] == "passed" and official["queries"] == 288 \
        and max(c["all_candidate_score_max_error"] for c in official["checks"]) < 1e-4
    checks["warmup12_direct_checks"] = result["verification"]["raw_regeneration"] == 12
    for name, digest in result["audit_artifacts"].items():
        checks[f"artifact_sha:{Path(name).name}"] = sha(host(name)) == digest
    cache = result["reference_batch1_cache"]
    for name, digest in cache["files"].items():
        checks[f"cache_sha:{name}"] = sha(LARGE / "ringpp_batch1/n10000" / name) == digest
    checks["cache_point_counts_equal_oldkernel"] = cache["point_counts_equal_oldkernel_cache"] and np.array_equal(
        np.load(LARGE / "ringpp_batch1/n10000/point_counts.npy"), np.load(TABLE3 / "ringpp_batch1/n10000/point_counts.npy"))
    checks["input_hash_equal_oldkernel_run"] = result["raw_input_sha256"] == read(TABLE3 / "methods/ringpp_n10000.json")["raw_input_sha256"]

    protocol = read(COST / "ringpp_exactmax_protocol.json")
    checks["protocol_sha_in_result"] = sha(COST / "ringpp_exactmax_protocol.json") == result["method_protocol_sha256"]
    checks["common_protocol_equal_table3"] = read(COST / "common_protocol.json") == read(TABLE3 / "protocol.json")
    vs_old_exactmax = diff(protocol, read(OLD_EXACTMAX_PROTOCOL))
    checks["protocol_vs_earlier_exactmax_only_local_fields"] = set(vs_old_exactmax) == {
        "/local_run (only_new)", "/base_cache_protocol/sources/scripts/_review_ringpp_maxfix_cost.py (only_old)"}
    v3 = read(TABLE3 / "ringpp_v3_protocol.json")
    vs_v3 = diff(protocol["base_cache_protocol"], v3["base_cache_protocol"])
    checks["base_protocol_vs_oldkernel_v3_only_maxfix"] = set(vs_v3) == {
        "/encoder", "/sources/scripts/_review_native_ring_maxfix.py (only_new)"}
    for path, digest in protocol["base_cache_protocol"]["sources"].items():
        checks[f"source_unchanged:{path}"] = sha(REPO / path) == digest

    # 같은 구현의 서버(RTX PRO 6000) 실행과 승자·점수 대조(하드웨어가 달라 비용은 쓰지 않는다).
    server = read(SERVER / "timed_rows.json")["rows"]
    same_winner = sum(a["winner"] == b["winner"] and a["query"] == b["query"] for a, b in zip(rows, server))
    conf_gap = max(abs(a["confidence"] - b["confidence"]) for a, b in zip(rows, server))
    checks["server_same_winners_288"] = same_winner == 288
    checks["server_confidence_gap_lt_1e-4"] = conf_gap < 1e-4

    equivalence = read(WORK / "equivalence/verification.json")
    checks["test_ledger_equivalence_dcc01"] = equivalence["passed"]
    reverify = read(WORK / "test_reverify/verification.json")
    checks["test46_reverified"] = reverify["passed"] and reverify["common_queries"] == 195726 and reverify["matches_instruction_values_9dp"]
    memtrace = read(COST / "memtrace/summary.json")
    checks["memtrace_no_cuda_errors"] = memtrace["status"] == "passed"
    checks["memtrace_360_per_encode"] = memtrace["voxelfeat_increment_per_encode"] == [360]
    checks["memtrace_trace_sha"] = sha(REPO / memtrace["trace"]) == memtrace["trace_sha256"]

    old = read(TABLE3 / "methods/ringpp_n10000.json")
    new_cells, old_cells = cells(result), cells(old)
    costs = {
        "exactmax_local_rtx5080_n10000": {
            "source": str((COST / "methods/ringpp_n10000.json").relative_to(REPO)),
            "source_sha256": sha(COST / "methods/ringpp_n10000.json"),
            "timings_ms": result["timings_ms"], "table2_cells": new_cells,
            "resident_bytes_total": memory["total"],
            "resident_breakdown": {"gpu_fft_bank": memory["breakdown"]["gpu_fft_bank"],
                                   "torch_allocated": memory["torch_cuda_allocated"],
                                   "native_requested_live": memory["external_requested_gpu_bytes"], "cpu": memory["cpu"]},
            "cuda_peak_allocated": memory["cuda_peak_allocated"], "cuda_peak_reserved": memory["cuda_peak_reserved"],
            "process_peak_rss": memory["process_peak_rss"], "device_memory": result["device_memory"],
            "initial_cached_state_build_s": result["initial_cached_state_build_s"],
            "cache_build_s": cache["seconds"],
        },
        "oldkernel_local_rtx5080_n10000_current_table2": {
            "source": str((TABLE3 / "methods/ringpp_n10000.json").relative_to(REPO)),
            "timings_ms": old["timings_ms"], "table2_cells": old_cells, "resident_bytes_total": old["resident_bytes"]["total"]},
        "delta_exactmax_minus_oldkernel": {k: new_cells[k] - old_cells[k] for k in new_cells},
        "external_cuda_accounting": {
            "source": str((COST / "memtrace/summary.json").relative_to(REPO)),
            "peak_live_requested_bytes_all_cuda_apis": memtrace["peak_live_requested_bytes_all_cuda_apis"],
            "peak_by_owner": memtrace["peak_by_owner"], "external_at_peak_bytes": memtrace["external_at_peak_bytes"],
            "torch_peak_reserved_traced_run": memtrace["torch_peak_reserved"],
            "voxelfeat_increment_per_encode": memtrace["voxelfeat_increment_per_encode"],
            "device_used_max": memtrace["device_used_max"], "scope": memtrace["scope"]},
        "excluded_server_measurement": {"source": str((SERVER / "measurement.json").relative_to(REPO)),
                                        "processing_ms": read(SERVER / "measurement.json")["timings_ms"]["processing"]["mean"],
                                        "reason": "RTX PRO 6000; 원고 표2 하드웨어(RTX5080)와 달라 비용으로 쓰지 않음. 승자·점수 대조에만 사용",
                                        "same_winners": same_winner, "max_confidence_gap": conf_gap},
    }
    performance = {
        "exactmax_46": reverify["macro"], "oldkernel_table2": reverify["table2_current_oldkernel"],
        "delta_pp": reverify["delta_exactmax_minus_oldkernel_pp"], "cities": reverify["cities"],
        "common_queries": reverify["common_queries"], "positives": reverify["positives"],
        "source": "results/pdv2_ringpp_maxfix_20261105/summary_verified.json",
        "reverification": str((WORK / "test_reverify/verification.json").relative_to(REPO)),
        "table2_cells_percent": {k: round(100 * reverify["macro"][k], 2) for k in METRICS},
    }
    verification = {"passed": all(checks.values()), "checks": checks,
                    "failed": [k for k, v in checks.items() if not v],
                    "protocol_diff_vs_earlier_exactmax": vs_old_exactmax, "base_protocol_diff_vs_oldkernel_v3": vs_v3,
                    "source_sha256": sha(Path(__file__))}
    for name, value in (("verification.json", verification), ("costs.json", costs), ("performance.json", performance)):
        (WORK / name).write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": verification["passed"], "failed": verification["failed"],
                      "new_cells": new_cells, "old_cells": old_cells}, ensure_ascii=False))
    raise SystemExit(0 if verification["passed"] else 1)


if __name__ == "__main__":
    main()
