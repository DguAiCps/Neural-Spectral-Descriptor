"""사용자가 채택한 크기 압축 NSD의 검산된 결과만 원고에 연결한다."""
import hashlib
import statistics

from _paper_current_nsd import REPO, METRICS, _cells, _close, _read

ROOT = "results/pdv2_spectral_power_20261101"


def apply_power_main(tables: dict, manifest: dict) -> None:
    """표2와 같은 설정의 표4를 생성하고 원천 해시를 기록한다."""
    data = _read(ROOT + "/table2_summary.json", manifest)
    freeze = _read(ROOT + "/test_freeze.json", manifest)
    proof = _read(ROOT + "/test_completion_validation.json", manifest)
    adoption = _read(ROOT + "/adoption_decision.json", manifest)
    selected = data["selected"]
    assert selected == freeze["selected"] == adoption["selected"] == "p050__aux_half__equal"
    assert adoption["decision"] == "adopt_main" and data["checks_passed"] and not data["test_used_for_selection"]
    assert proof["status"] == "passed" and proof["summary_sha256"] == hashlib.sha256((REPO / ROOT / "table2_summary.json").read_bytes()).hexdigest()
    assert data["freeze_sha256"] == hashlib.sha256((REPO / ROOT / "test_freeze.json").read_bytes()).hexdigest()
    assert data["query_counts"] == {str(s): 195726 for s in (1, 2, 3)}
    stats = data["mean_std"][selected]
    for seed in ("1", "2", "3"):
        per = data["per_sequence"][selected][seed]
        assert set(per) == set(freeze["test"]) and sum(v["n_pos"] for v in per.values()) == 61682
        for metric in METRICS:
            cities = [statistics.mean(per[k][metric] for k in roster if per[k]["n_pos"])
                      for roster in freeze["cities"].values()]
            _close(statistics.mean(cities), data["per_seed"][selected][seed]["macro"][metric])
    for m in METRICS:
        values = [data["per_seed"][selected][s]["macro"][m] for s in ("1", "2", "3")]
        _close(statistics.mean(values), stats[m]["mean"])
        _close(statistics.stdev(values), stats[m]["std"])
    text = tables["table1_online.tex"].splitlines()
    tables["table1_online.tex"] = "\n".join(
        r"\textbf{NSD} & " + _cells(stats) + r" \\" if line.startswith(r"\textbf{NSD} &") else line for line in text) + "\n"
    tables["table1_online.tex"] = tables["table1_online.tex"].replace(
        "results/pdv2_spectral_whitening_20261026/fixed_test/table2_summary.json", ROOT + "/table2_summary.json")
    abpath = ROOT + "/adoption/table4/summary.json"
    ab = _read(abpath, manifest)
    completed = _read(ROOT + "/adoption/table4/completion_validation.json", manifest)
    prefix = _read(ROOT + "/adoption/table4/prefix_audit/completion.json", manifest)
    assert ab["checks_passed"] and not ab["test_used_for_selection"]
    assert completed["status"] == "passed" and completed["main_table2_exact"]
    assert prefix["status"] == "passed" and prefix["checks"] == 2208
    assert completed["summary_sha256"] == hashlib.sha256((REPO / abpath).read_bytes()).hexdigest()
    assert ab["mean_std"]["NSD"] == stats
    labels = (("NSD", r"\textbf{NSD}"), ("without_correlation_correction", "w/o correlation correction"),
              ("without_amplitude_compression", "w/o amplitude compression"),
              ("graph_free", "Graph-free learned projection"), ("standard", "Standard temporal attention"),
              ("key_only", "Key index only"), ("aux_only", "Auxiliary index only"))
    text = tables["table_online_components.tex"].splitlines()
    first = next(i for i, s in enumerate(text) if s.startswith(r"\textbf{NSD} &"))
    last = text.index(r"\bottomrule")
    text[first:last] = [label + " & " + _cells(ab["mean_std"][arm], False) + r" \\" for arm, label in labels]
    tables["table_online_components.tex"] = "\n".join(text) + "\n"
    tables["table_online_components.tex"] = tables["table_online_components.tex"].replace(
        "results/pdv2_spectral_whitening_20261026/fixed_test/table2_summary.json", abpath)
    manifest["main_configuration"] = "octave288_temporal_diff_geometry_interaction_arc_unique_whitening25_power05_auxhalf_equal"
    manifest["main_decision_statistic"] = {"fraction": .25, "power": .5, "channels": "(30*c_spec+c_col)/2",
        "temporal_score": "arc_unique_b100", "fixed_threshold_calibrated_for_mixture": False,
        "retrieval_coefficients": "original", "fusion": {"1": [1, .5, 2, 1], "2": [2, .5, 2, 2], "3": [2, .5, 2, 2]}}
    for name, sources in (("table1_online.tex", [ROOT + "/table2_summary.json", ROOT + "/test_freeze.json"]),
                          ("table_online_components.tex", [abpath, ROOT + "/adoption/table4/protocol.json"])):
        manifest["tables"][name].update(source_paths=sources, json_field="mean_std", filled="새 주모델과 같은 최종 판정 규약의 검산된 수치")
        manifest["output_sha256"][name] = hashlib.sha256(tables[name].encode()).hexdigest()
    manifest["tables"]["table1_online.tex"]["json_field"] = "mean_std." + selected
    manifest["tables"]["table_online_components.tex"].update(
        completed_ablation_rows=[a for a, _ in labels if a != "NSD"],
        selection="검증으로 고정한 체크포인트 유지. 새 주모델의 공통 융합·판정에서 해당 성분만 대조. 시험 선택 없음.")
