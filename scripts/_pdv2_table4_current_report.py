#!/usr/bin/env python3
"""현재 표4 원장을 독립 재집계하고 원고 연결용 완료 증거를 쓴다."""
import argparse
import time
from pathlib import Path

import numpy as np

import _pdv2_table4_current as run
from _pdv2_no_context_ablation import sha256, write_json


def check_main() -> dict:
    """주모델 표2의 모든 저장 원장과 집계 수치를 다시 확인한다."""
    data = run.read(run.MAIN / "table2_summary.json")
    spec = run.read(run.MAIN / "freeze.json")
    rows, count = {}, 0
    for seed in run.SEEDS:
        rows[str(seed)] = {}
        total = positives = 0
        for sequence in spec["test"]:
            p = run.MAIN / f"s{seed}" / (sequence.replace("/", "_") + ".npz")
            record = run.read(p.with_suffix(".json"))
            assert sha256(p) == record["sha256"]
            row = run.source.base.exp.arrays(p)
            metric = run.independent_metrics(row, row["score"])
            assert metric == data["per_sequence"][str(seed)][sequence]
            np.testing.assert_array_equal(row["query_idx"], run.source.base.exp.arrays(run.reference_paths(seed, "test", sequence)[0])["query_idx"])
            rows[str(seed)][sequence] = metric
            total += len(row["query_idx"])
            positives += metric["n_pos"]
            count += 1
        assert total == 195726 and positives == 61682
        aggregate = run.macro(rows[str(seed)], spec["cities"])
        assert aggregate["macro"] == data["per_seed"][str(seed)]["macro"]
    result = {"status": "passed", "ledgers": count, "queries_per_seed": 195726,
              "positive_queries_per_seed": 61682, "all_final_metrics_equal": True,
              "metrics": data["mean_std"]["whitening25"],
              "summary_sha256": sha256(run.MAIN / "table2_summary.json"),
              "audit_script_sha256": sha256(Path(__file__)), "created_unix": time.time()}
    target = run.OUT / "main_reproduction.json"
    if target.exists():
        previous = run.read(target)
        assert previous["summary_sha256"] == result["summary_sha256"]
        assert previous["metrics"] == result["metrics"] and previous["ledgers"] == result["ledgers"]
        return previous
    write_json(target, result)
    return result


def complete(group: str) -> None:
    """새 3시드 행의 원천·질의·선택·지표를 완료 표지에 연결한다."""
    main = check_main()
    summary_path = run.OUT / f"summary_{group}.json"
    data = run.read(summary_path)
    frozen = run.freeze(group)
    spec = run.protocol(True)
    main_sources = run.read(run.MAIN / "freeze.json")["sources"]
    original_sources = run.read(run.source.OUT / "table2_summary.json")["sources"]
    assert data["checks_passed"]
    assert data["freeze_sha256"] == sha256(run.OUT / f"freeze_{group}.json")
    for name, digest in data["sources"].items():
        assert sha256(run.REPO / name) == digest, name
    counts, delta = {}, {}
    for arm, value in data["arms"].items():
        assert value["checks_passed"] and value["final_metrics"]
        counts[arm] = 0
        for seed in run.SEEDS:
            field = "score_" + frozen["models"][str(seed)]["no_alignment_channel"] if arm == "no_alignment" else "score"
            rows, queries, positives = {}, 0, 0
            for sequence in spec["test"]:
                path = run.root(seed, arm) / "test" / (sequence.replace("/", "_") + ".npz")
                info = run.read(path.with_suffix(".json"))
                arrays = run.source.base.exp.arrays(path)
                expected_checkpoint = (spec["main_models"][str(seed)]["sha256"] if group == "retrieval"
                                       else frozen["models"][str(seed)]["sha256"])
                assert info["checkpoint_sha256"] == expected_checkpoint
                assert sha256(path) == info["sha256"]
                reference_path = run.reference_paths(seed, "test", sequence)[0]
                reference_info = run.read(reference_path.with_suffix(".json"))
                assert reference_info["checkpoint_sha256"] == spec["main_models"][str(seed)]["sha256"]
                for source_path, digest in info["inputs"].items():
                    expected = main_sources.get(source_path, original_sources.get(source_path))
                    if expected is not None:
                        assert digest == expected, source_path
                reference = run.source.base.exp.arrays(reference_path)
                for name in ("query_idx", "min_past_gt_dist", "group"):
                    np.testing.assert_array_equal(arrays[name], reference[name])
                assert (arrays["j_star"] <= arrays["query_idx"] - 30).all()
                rows[sequence] = run.independent_metrics(arrays, arrays[field])
                assert rows[sequence] == value["per_sequence"][str(seed)][sequence]
                queries += len(arrays["query_idx"])
                positives += rows[sequence]["n_pos"]
                counts[arm] += 1
            assert queries == 195726 and positives == 61682
            assert run.macro(rows, spec["cities"]) == value["per_seed"][str(seed)]
        for metric in run.METRICS:
            values = [value["per_seed"][str(s)]["macro"][metric] for s in run.SEEDS]
            np.testing.assert_allclose(np.mean(values), value["mean_std"][metric]["mean"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.std(values, ddof=1), value["mean_std"][metric]["std"], atol=1e-12, rtol=0)
        delta[arm] = {m: 100 * (main["metrics"][m]["mean"] - value["mean_std"][m]["mean"]) for m in run.METRICS}
    result = {"status": "passed", "group": group, "arms": list(data["arms"]),
              "summary_sha256": sha256(summary_path),
              "protocol_sha256": sha256(run.OUT / "protocol.json"),
              "freeze_sha256": sha256(run.OUT / f"freeze_{group}.json"),
              "audit_script_sha256": sha256(Path(__file__)),
              "main_reproduction_sha256": sha256(run.OUT / "main_reproduction.json"),
              "ledger_counts": counts, "same_common_queries": True, "final_scores_only": True,
              "test_used_for_selection": False, "main_minus_ablation_pp": delta,
              "created_unix": time.time()}
    write_json(run.OUT / f"completion_{group}.json", result)
    print("AUDITED", group, delta, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=("main", "retrieval", "graph_free", "standard"))
    args = parser.parse_args()
    if args.group == "main":
        check_main()
    else:
        complete(args.group)
