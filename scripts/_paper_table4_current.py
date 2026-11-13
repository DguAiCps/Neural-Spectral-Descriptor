"""현재 구성의 검산 완료 표4 행만 원고 생성기에 제공한다."""
import hashlib
import math
import statistics
from pathlib import Path

ROOT = "results/pdv2_table4_current_20261030"
METRICS = ("AP", "max_F1", "R@100P", "R@1")


def verified_rows(repo: Path, read, manifest: dict) -> tuple:
    """완료 표지·질의·집계·원장 해시를 검증한 행과 출처를 반환한다."""
    rows, paths = {}, []
    for group in ("retrieval", "graph_free", "standard"):
        completion_path = ROOT + f"/completion_{group}.json"
        if not (repo / completion_path).exists():
            continue
        summary_path = ROOT + f"/summary_{group}.json"
        freeze_path = ROOT + f"/freeze_{group}.json"
        completed = read(completion_path, manifest)
        summary = read(summary_path, manifest)
        frozen = read(freeze_path, manifest)
        spec = read(ROOT + "/protocol.json", manifest)
        paths.extend((completion_path, summary_path, freeze_path, ROOT + "/protocol.json"))
        for key, path in (("summary_sha256", summary_path), ("freeze_sha256", freeze_path),
                          ("protocol_sha256", ROOT + "/protocol.json")):
            if hashlib.sha256((repo / path).read_bytes()).hexdigest() != completed[key]:
                raise ValueError("표4 완료 기록 해시 불일치: " + path)
        if (completed["status"] != "passed" or not completed["same_common_queries"]
                or not completed["final_scores_only"] or completed["test_used_for_selection"]
                or not summary["checks_passed"] or frozen["test_used_for_selection"]):
            raise ValueError("표4 완료·선택 조건 불일치")
        if summary["protocol_sha256"] != completed["protocol_sha256"]:
            raise ValueError("표4 다른 프로토콜의 원천")
        for path, digest in summary["sources"].items():
            if hashlib.sha256((repo / path).read_bytes()).hexdigest() != digest:
                raise ValueError("표4 원장 변경: " + path)
        for arm, value in summary["arms"].items():
            if (not value["checks_passed"] or not value["final_metrics"]
                    or value["query_counts"] != {str(s): 195726 for s in (1, 2, 3)}
                    or completed["ledger_counts"][arm] != 138):
                raise ValueError("표4 불완전한 3시드 행: " + arm)
            for seed in ("1", "2", "3"):
                seq = value["per_sequence"][seed]
                if set(seq) != set(spec["test"]) or len(seq) != 46:
                    raise ValueError("표4 시험 목록 불일치")
                if sum(r["n_pos"] for r in seq.values()) != 61682:
                    raise ValueError("표4 양성 분모 불일치")
                for metric in METRICS:
                    cities = [statistics.mean(seq[s][metric] for s in members if seq[s]["n_pos"])
                              for members in spec["cities"].values()]
                    if not math.isclose(statistics.mean(cities), value["per_seed"][seed]["macro"][metric], abs_tol=1e-12, rel_tol=0):
                        raise ValueError("표4 도시 평균 불일치")
            for metric in METRICS:
                values = [value["per_seed"][s]["macro"][metric] for s in ("1", "2", "3")]
                for actual, key in ((statistics.mean(values), "mean"), (statistics.stdev(values), "std")):
                    if not math.isclose(actual, value["mean_std"][metric][key], abs_tol=1e-12, rel_tol=0):
                        raise ValueError("표4 시드 집계 불일치")
            rows[arm] = value["mean_std"]
    return rows, sorted(set(paths))
