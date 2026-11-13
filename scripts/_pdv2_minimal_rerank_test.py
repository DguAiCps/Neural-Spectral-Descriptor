#!/usr/bin/env python3
"""검증 기준을 통과한 후보 하나만 표2의 최종 시험 규약으로 평가한다."""
from __future__ import annotations

import concurrent.futures
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import _pdv2_minimal_rerank as run

if "--local" in sys.argv:
    import _pdv2_local_refine as run

OUT, REPO = run.OUT, run.REPO


def freeze() -> dict:
    """검증으로 선택한 하나의 설정과 실행 소스를 시험 전에 고정한다."""
    target = OUT / "test_freeze.json"
    if target.exists():
        spec = run.read(target)
        for p, h in spec["sources"].items():
            assert run.sha256(REPO / p) == h, p
        return spec
    run.prepare()
    summary = run.read(OUT / "summary.json")
    selected = summary["selected"]
    assert selected and summary["decisions"][selected]["eligible"], "검증 승격 조건 미달"
    if hasattr(run, "parent"):
        first = run.read(run.parent.OUT / "summary.json")
        assert summary["stats"][selected]["AP"]["mean"] >= first["stats"][first["selected"]]["AP"]["mean"] - 1e-12
    assert run.read(OUT / "completion_validation.json")["status"] == "passed"
    old = run.read(run.white.OUT / "fixed_test/freeze.json")
    spec = run.read(OUT / "protocol.json")
    paths = {Path(__file__), Path(run.__file__), OUT / "protocol.json", OUT / "summary.json",
             OUT / "completion_validation.json", run.white.OUT / "fixed_test/freeze.json",
             run.white.OUT / "fixed_test/table2_summary.json",
             REPO / "results/pdv2_baseline_seeds/summary.json"}
    for seed in spec["seeds"]:
        for seq in old["test"]:
            name = seq.replace("/", "_") + ".npz"
            paths.update((run.reference_path(seed, "test", name),
                          run.source_folder(seed, "test", spec) / ("keys_" + name)))
    result = {"created_unix": time.time(), "selected": selected, "test": old["test"],
              "cities": old["cities"], "seeds": spec["seeds"],
              "scope": "검증에서선택한한조건과기존NSD만최종평가. 시험설정선택없음. 시험성적을본뒤조건을다시고르지않음. 기존195726질의,양성없는KITTI07제외45시퀀스→6도시→3시드. 원고자동수정없음.",
              "sources": {str(p.relative_to(REPO)): run.sha256(p) for p in sorted(paths)}}
    run.write_json(target, result)
    return result


def finish() -> None:
    """기존 표2 재현과 새 결과의 도시 평균을 검사하고 비교한다."""
    spec = freeze()
    selected = spec["selected"]
    stats, per_seed, per_sequence = {}, {}, {}
    records, count = [], {}
    for arm in ("baseline", selected):
        per_seed[arm], per_sequence[arm] = {}, {}
        for seed in spec["seeds"]:
            seqs, n = {}, 0
            for seq in spec["test"]:
                info = run.read(OUT / "test" / f"s{seed}" / (seq.replace("/", "_") + ".json"))
                item = info["arms"][arm]
                path = REPO / item["path"]
                assert run.sha256(path) == item["sha256"]
                row = run.arrays(path)
                metric = run.independent_metrics(row, row["score"])
                assert metric == item["metrics"]
                seqs[seq] = metric
                n += len(row["query_idx"])
                records.append({"path": item["path"], "sha256": item["sha256"]})
            assert n == 195726
            count[str(seed)] = n
            cities = {}
            for city, members in spec["cities"].items():
                valid = [seqs[s] for s in members if seqs[s]["n_pos"]]
                assert valid
                cities[city] = {m: float(np.mean([r[m] for r in valid])) for m in run.METRICS}
            per_sequence[arm][str(seed)] = seqs
            per_seed[arm][str(seed)] = {"cities": cities,
                "macro": {m: float(np.mean([r[m] for r in cities.values()])) for m in run.METRICS}}
        stats[arm] = {m: {"mean": float(np.mean([r["macro"][m] for r in per_seed[arm].values()])),
                         "std": float(np.std([r["macro"][m] for r in per_seed[arm].values()], ddof=1))}
                      for m in run.METRICS}
    old = run.read(run.white.OUT / "fixed_test/table2_summary.json")
    for m in run.METRICS:
        for stat in ("mean", "std"):
            np.testing.assert_allclose(stats["baseline"][m][stat], old["mean_std"]["whitening25"][m][stat], atol=1e-10, rtol=0)
    bev = run.read(REPO / "results/pdv2_baseline_seeds/summary.json")["methods"]["bevplace"]["mean_std"]
    delta = {m: 100 * (stats[selected][m]["mean"] - stats["baseline"][m]["mean"]) for m in run.METRICS}
    delta_bev = {m: 100 * (stats[selected][m]["mean"] - bev[m]["mean"]) for m in run.METRICS}
    city_delta = {c: {m: 100 * float(np.mean([per_seed[selected][str(s)]["cities"][c][m] - per_seed["baseline"][str(s)]["cities"][c][m] for s in spec["seeds"]])) for m in run.METRICS} for c in spec["cities"]}
    report = {"selected": selected, "mean_std": {**stats, "bevplace": bev},
              "per_seed": per_seed, "per_sequence": per_sequence, "delta_pp": delta,
              "delta_vs_bevplace_pp": delta_bev, "city_delta_pp": city_delta,
              "query_counts": count, "test_used_for_selection": False, "paper_changed": False,
              "checks_passed": True, "freeze_sha256": run.sha256(OUT / "test_freeze.json")}
    run.write_json(OUT / "table2_summary.json", report)
    run.write_json(OUT / "test_completion_validation.json", {"status": "passed", "records": records,
                  "ledgers": len(records), "query_rows": 2 * sum(count.values()),
                  "original_table2_reproduced": True,
                  "summary_sha256": run.sha256(OUT / "table2_summary.json")})
    with (OUT / "table2_comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *run.METRICS])
        for name, record in (("current_NSD", stats["baseline"]), (selected, stats[selected]), ("BEVPlace++", bev)):
            writer.writerow([name, *[100 * record[m]["mean"] for m in run.METRICS]])
    print("FINAL_TABLE2", stats, "DELTA_PP", delta, "VS_BEVPLACE_PP", delta_bev, flush=True)


def main() -> None:
    """설정을 고정한 뒤 같은 시험 규약으로 실행한다."""
    spec = freeze()
    logs = run.LOG / "test"
    logs.mkdir(parents=True, exist_ok=True)
    def invoke(item: tuple) -> None:
        seed, seq = item
        with (logs / f"s{seed}_{seq.replace('/', '_')}.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(run.__file__)), "worker", "--seed", str(seed),
                            "--sequence", seq, "--role", "test"], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    jobs = [(s, q) for s in spec["seeds"] for q in spec["test"]]
    tick = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(invoke, item): item for item in jobs}
        for n, future in enumerate(concurrent.futures.as_completed(pending), 1):
            future.result()
            run.write_json(OUT / "test_status.json", {"status": "running", "done": n,
                "total": len(jobs), "elapsed_s": time.time() - tick, "selected": spec["selected"]})
            print("TEST", n, len(jobs), pending[future], flush=True)
    finish()
    run.write_json(OUT / "test_status.json", {"status": "complete", "elapsed_s": time.time() - tick})


if __name__ == "__main__":
    main()
