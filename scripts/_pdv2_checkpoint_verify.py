#!/usr/bin/env python3
"""체크포인트 선택 실험의 원장·동결 순서·최종 집계를 검사한다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_checkpoint_select import (
    BASE, OUT, PRIMARY, aggregate_rows, protocol, read_json,
)
from _pdv2_lag15_report import read as read_ledger, sequence_metrics
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """전체 결과를 다시 계산해 저장 지표와 대조하고 검사 기록을 남긴다."""
    spec = protocol()
    frozen = read_json(OUT / "freeze.json")
    validation = read_json(OUT / "validation_summary.json")
    summary = read_json(OUT / "summary.json")
    assert read_json(OUT / "status.json")["status"] == "complete"
    assert frozen["protocol_sha256"] == sha256(OUT / "protocol.json")
    assert frozen["val_summary_sha256"] == sha256(OUT / "validation_summary.json")
    assert frozen["selected"] == validation["selected"] == summary["selected"]
    for name, expected in validation["inputs_sha256"].items():
        assert sha256(REPO / name) == expected, name
    jobs = []
    for seed, entries in spec["candidates"].items():
        for entry in entries:
            jobs.append((seed, entry["id"], "val"))
        jobs.append((seed, frozen["selected"][seed]["id"], "test"))
    counts = {"val": 0, "test": 0}
    ledger_count = 0
    totals = {}
    observed_test_files = set()
    largest_error = 0.0
    for seed, cid, role in jobs:
        selections = read_json(BASE / f"s{seed}/confidence_selection.json")["ops"]
        rows = {}
        for sequence in spec["roster"][role]:
            sensor, seq = sequence.split("/")
            path = OUT / f"s{seed}/{cid}/{role}_{sensor}_{seq}.json"
            row = read_json(path)
            assert row["protocol_sha256"] == sha256(OUT / "protocol.json")
            assert row["seed"] == int(seed) and row["checkpoint"]["id"] == cid
            assert row["sequence"] == sequence and row["role"] == role
            if role == "test":
                assert path.stat().st_mtime >= frozen["created_unix"]
                observed_test_files.add(path)
            for op, record in row["ops"].items():
                ledger = REPO / record["ledger"]
                assert sha256(ledger) == record["sha256"], ledger
                arrays, _ = read_ledger(ledger)
                reference, _ = read_ledger(
                    BASE / f"s{seed}/{role}_selected_symmetric_{op}_{sensor}_{seq}.npz"
                )
                for field in ("query_idx", "min_past_gt_dist", "group"):
                    np.testing.assert_array_equal(arrays[field], reference[field])
                assert np.all(arrays["j_star"] <= arrays["query_idx"] - 30)
                assert np.all(arrays["j_star"] >= 0)
                selection = selections[op]
                metrics = sequence_metrics(arrays, selection["decision"], selection["tau_star"])
                for metric, actual in metrics.items():
                    expected = record["metrics"][metric]
                    if actual is None:
                        assert expected is None
                    else:
                        error = abs(actual - expected)
                        largest_error = max(largest_error, error)
                        assert error < 1e-12, (path, op, metric)
                ledger_count += 1
            rows[sequence] = row
            counts[role] += 1
        op = next(iter(selections))
        total = {field: sum(r["ops"][op]["metrics"][field] for r in rows.values())
                 for field in ("n_queries", "n_positive")}
        assert total == ({"n_queries": 99236, "n_positive": 10036} if role == "val"
                         else {"n_queries": 195726, "n_positive": 61682})
        totals[f"s{seed}/{cid}/{role}"] = total
        for op in selections:
            recomputed = aggregate_rows(rows, role, op)
            if role == "val":
                stored = next(r for r in validation["per_seed"][seed] if r["id"] == cid)["ops"][op]
            else:
                stored = summary["ops"][op]["per_seed"][seed]["selected"]
            for metric in PRIMARY:
                assert abs(recomputed["macro"][metric] - stored["macro"][metric]) < 1e-12
        print("VERIFIED", seed, cid, role, flush=True)
    assert counts == {"val": 153, "test": 138}
    assert set(OUT.glob("s*/*/test_*.json")) == observed_test_files
    for seed, entries in validation["per_seed"].items():
        maximum = max(row["joint_val_AP"] for row in entries)
        tied = [row for row in entries if maximum - row["joint_val_AP"] <= 1e-12]
        chosen = min(tied, key=lambda row: (not row["reference"], row["epoch"], row["id"]))
        assert chosen["id"] == frozen["selected"][seed]["id"]
    for op, result in summary["ops"].items():
        for condition, metrics in result["mean_std"].items():
            for metric, stored in metrics.items():
                values = [row[condition]["macro"][metric] if condition == "selected"
                          else row[condition][metric] for row in result["per_seed"].values()]
                assert abs(float(np.mean(values)) - stored["mean"]) < 1e-12
                assert abs(float(np.std(values, ddof=1)) - stored["std"]) < 1e-12
    write_json(OUT / "completion_validation.json", {
        "status": "passed", "completed_jobs": counts, "checked_ledgers": ledger_count,
        "metric_max_error": largest_error, "query_counts": totals,
        "selected_only_test": True, "test_results_after_freeze": True,
        "source_hashes_unchanged": True,
        "sha256": {name: sha256(OUT / name) for name in
                   ("protocol.json", "preflight_amendment.json", "freeze.json",
                    "validation_summary.json", "summary.json")},
    })
    print("PASSED", counts, ledger_count, flush=True)


if __name__ == "__main__":
    main()
