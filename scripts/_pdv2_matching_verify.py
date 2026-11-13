#!/usr/bin/env python3
"""교차 비교의 학습량·표본 공유·최종 원장과 선택 결과를 검사한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_matching_eval import (
    OUT, BASE, ARMS, EPOCHS, VAL, TEST, OPS, aggregate, folder, protocol, read_json,
)
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_lag15_report import read as read_ledger, sequence_metrics


def main() -> None:
    """완료된 모든 조건에서 원장을 재계산하고 검증 기록을 쓴다."""
    spec = protocol()
    status = read_json(OUT / "status.json")
    assert status["status"] == "complete"
    pilot = read_json(OUT / "pilot_summary.json")
    assert status["test_used"] == pilot["expand"]
    seeds = (1, 2, 3) if pilot["expand"] else (1,)
    files = 0
    jobs = 0
    query_counts = {}
    for name, digest in spec["prepared_data_sha256"].items():
        assert sha256(REPO / name) == digest
    for seed in seeds:
        selections = read_json(BASE / f"s{seed}/confidence_selection.json")["ops"]
        selected = read_json(OUT / f"s{seed}/validation_summary.json")["arms"]
        sample_dir = OUT / f"s{seed}/samples"
        sample_audit = read_json(sample_dir / "audit.json")
        for name, digest in sample_audit["sample_sha256"].items():
            assert sha256(sample_dir / name) == digest
        for epoch in (1, 6, 11, 16):
            with np.load(sample_dir / f"original_epoch{epoch}.npz") as a, \
                    np.load(sample_dir / f"matching_epoch{epoch}.npz") as b:
                np.testing.assert_array_equal(a["triplets"][:, :2], b["triplets"][:, :2])
                assert a["triplets"].shape == b["triplets"].shape == (183249, 3)
        for model in ("gnn", "point"):
            a, b = [OUT / f"s{seed}/{model}_{policy}" for policy in ("original", "matching")]
            assert read_json(a / "initial_model.json") == read_json(b / "initial_model.json")
            assert read_json(a / "training_graph.json") == read_json(b / "training_graph.json")
        for arm in ARMS:
            progress = read_json(OUT / f"s{seed}/{arm}/training_progress.json")
            assert progress["epoch"] == 20 and progress["optimizer_updates"] == 900
            for epoch in EPOCHS:
                recomputed = aggregate(seed, arm, epoch, "val")
                stored = next(r for r in selected[arm]["candidates"] if r["epoch"] == epoch)
                assert recomputed == stored["ops"]
            best = max(r["joint_AP"] for r in selected[arm]["candidates"])
            assert abs(best - selected[arm]["joint_AP"]) < 1e-12
            work = [(epoch, "val", VAL) for epoch in EPOCHS]
            if pilot["expand"]:
                work.append((selected[arm]["epoch"], "test", TEST))
            for epoch, role, roster in work:
                totals = {"n_queries": 0, "n_positive": 0}
                for sensor, seq in roster:
                    path = folder(seed, arm, epoch) / f"{role}_{sensor}_{seq}.json"
                    row = read_json(path)
                    assert row["protocol_sha256"] == sha256(OUT / "protocol.json")
                    if role == "test":
                        assert path.stat().st_mtime >= read_json(OUT / "freeze_test.json")["created_unix"]
                    for op in OPS:
                        record = row["ops"][op]
                        ledger = REPO / record["ledger"]
                        assert sha256(ledger) == record["sha256"]
                        a, _ = read_ledger(ledger)
                        ref, _ = read_ledger(BASE / f"s{seed}/{role}_selected_symmetric_{op}_{sensor}_{seq}.npz")
                        for field in ("query_idx", "min_past_gt_dist", "group"):
                            np.testing.assert_array_equal(a[field], ref[field])
                        assert np.all(a["j_star"] <= a["query_idx"] - 30)
                        assert np.all(a["j_star"] >= 0)
                        sel = selections[op]
                        metrics = sequence_metrics(a, sel["decision"], sel["tau_star"])
                        assert metrics == record["metrics"]
                        files += 1
                    for field in totals:
                        totals[field] += row["ops"][OPS[0]]["metrics"][field]
                    jobs += 1
                assert totals == ({"n_queries": 99236, "n_positive": 10036} if role == "val"
                                  else {"n_queries": 195726, "n_positive": 61682})
                query_counts[f"s{seed}/{arm}/{epoch}/{role}"] = totals
            print("VERIFIED", seed, arm, flush=True)
    if not pilot["expand"]:
        assert not list(OUT.glob("s*/*/epoch*/test_*.npz"))
        assert not (OUT / "freeze_test.json").exists()
    write_json(OUT / "completion_validation.json", {"status": "passed", "seeds": list(seeds),
               "checked_jobs": jobs, "checked_ledgers": files, "query_counts": query_counts,
               "paired_positives_identical": True, "paired_initial_weights_identical": True,
               "paired_graph_identical": True, "optimizer_updates_per_arm": 900,
               "final_metrics_exactly_reproduced": True, "test_used": status["test_used"],
               "protocol_sha256": sha256(OUT / "protocol.json"),
               "pilot_summary_sha256": sha256(OUT / "pilot_summary.json")})
    print("PASSED", jobs, files, flush=True)


if __name__ == "__main__":
    main()
