#!/usr/bin/env python3
"""다섯 대조의 완료 증거와 시드·도시별 차이를 한 보고서로 묶는다."""
import csv
import time
from pathlib import Path

import numpy as np

import _pdv2_table4_current as run
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """시험 선택 없이 완료한 모든 행을 고정 순서로 저장한다."""
    spec = run.protocol(True)
    state = run.read(run.OUT / "status.json")
    assert state["state"] == "complete"
    main_path = run.MAIN / "table2_summary.json"
    main_data = run.read(main_path)
    assert run.read(run.OUT / "main_reproduction.json")["all_final_metrics_equal"]
    controls, sources = {}, {str(main_path.relative_to(run.REPO)): sha256(main_path)}
    for group in ("retrieval", "graph_free", "standard"):
        summary = run.OUT / f"summary_{group}.json"
        completed = run.OUT / f"completion_{group}.json"
        proof, data = run.read(completed), run.read(summary)
        assert proof["status"] == "passed" and proof["summary_sha256"] == sha256(summary)
        assert proof["protocol_sha256"] == sha256(run.OUT / "protocol.json")
        controls.update(data["arms"])
        for p in (summary, completed, run.OUT / f"freeze_{group}.json"):
            sources[str(p.relative_to(run.REPO))] = sha256(p)
    assert set(controls) == set(run.ARMS)
    seeds, cities, means = {}, {}, {}
    for arm, row in controls.items():
        seeds[arm] = {str(s): {m: 100 * (main_data["per_seed"][str(s)]["macro"][m]
                              - row["per_seed"][str(s)]["macro"][m]) for m in run.METRICS}
                      for s in run.SEEDS}
        cities[arm] = {city: {m: 100 * float(np.mean([
            main_data["per_seed"][str(s)]["cities"][city][m]
            - row["per_seed"][str(s)]["groups"][city][m] for s in run.SEEDS]))
            for m in run.METRICS} for city in spec["cities"]}
        means[arm] = {m: float(np.mean([v[m] for v in seeds[arm].values()])) for m in run.METRICS}
    rows = {
        "NSD": main_data["mean_std"]["whitening25"],
        "without_correlation_correction": main_data["mean_std"]["octave288"],
        **{arm: controls[arm]["mean_std"] for arm in ("graph_free", "standard", *run.RETRIEVAL)},
    }
    result = {
        "created_unix": time.time(), "configuration": "octave288_temporal_diff_geometry_interaction_arc_unique_whitening25",
        "mean_std": rows, "main_minus_control_pp": means,
        "per_seed_main_minus_control_pp": seeds, "per_city_main_minus_control_pp": cities,
        "controls": controls, "sources": sources, "checks_passed": True,
        "test_used_for_selection": False, "runtime_seconds": state["elapsed_s"],
        "scope": "최종 수락 점수·최종 승자의 표2 규약. 모든 5대조 보고; 시험 결과로 설정 변경/주모델 교체 없음. "
                 "그래프 없는 투영은 신경망의 이웃 입력만 제거하고 거리 대응 시간 판정은 유지. "
                 "표준 attention은 전체 attention 블록 대조. 짝지은 불확실성 검정 미수행.",
    }
    write_json(run.OUT / "summary.json", result)
    with (run.OUT / "table4.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", *[f"{m}_{field}" for m in run.METRICS for field in ("mean", "sd")]])
        for label, row in rows.items():
            writer.writerow([label, *[row[m][field] for m in run.METRICS for field in ("mean", "std")]])
    records, queries, prefixes = {"val": 0, "test": 0}, {"val": 0, "test": 0}, 0
    for arm in run.ARMS:
        for seed in run.SEEDS:
            for role in ("val", "test"):
                folders = ("epoch10", "epoch20") if role == "val" and arm in run.LEARNED else (role,)
                for folder in folders:
                    for p in (run.root(seed, arm) / folder).glob("*.npz"):
                        info = run.read(p.with_suffix(".json"))
                        assert sha256(p) == info["sha256"]
                        with np.load(p) as data:
                            queries[role] += len(data["query_idx"])
                        records[role] += 1
                        prefixes += len(info["prefixes"])
                        assert all(r["winner_equal"] and r["score_error"] == 0 for r in info["prefixes"])
    assert records == {"val": 189, "test": 690}
    assert queries == {"val": 2083956, "test": 2935890}
    assert prefixes == 1758
    write_json(run.OUT / "completion_validation.json", {
        "status": "passed", "summary_sha256": sha256(run.OUT / "summary.json"),
        "csv_sha256": sha256(run.OUT / "table4.csv"), "source_groups": sources,
        "records": records, "queries": queries, "prefix_checks": prefixes,
        "synthetic_tests": 4, "main_reproduction_ledgers": 138,
        "finalize_script_sha256": sha256(Path(__file__)),
        "paper_verification": "원고 수치와 게시 검증은 publication_validation.json에 별도로 기록한다.",
    })
    print("ALL_CONTROLS_COMPLETE", rows, means, flush=True)


if __name__ == "__main__":
    main()
