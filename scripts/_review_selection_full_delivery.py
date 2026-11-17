#!/usr/bin/env python3
"""완료·독립 검산을 통과한 선택 대조를 원고 통합용 작은 JSON으로 전달한다."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _review_selection_bev import METRICS, read, write, sha

OUT = ROOT / "results/pdv2_review_selection_full_20261105"


def summarize(method: str) -> None:
    """선택 효과·규약 효과·실행 경로 차이를 구분해 저장한다."""
    source = OUT / "retained" / f"report_{method}.json"
    report = read(source)
    proof_path = OUT / f"independent_verification_{method}.json"
    proof = read(proof_path)
    assert proof["status"] == "complete" and proof["report_sha256"] == sha(source)
    frozen = report["selection"]
    samples = {
        "AP_selection": {s: r["AP"]["macro"] for s, r in report["test"].items()},
        "R1_selection": {s: r["R@1"]["macro"] for s, r in report["test"].items()},
        "original_same_runtime": {s: r["macro"] for s, r in report["same_runtime_original_selection_test"].items()},
        "original_historical": {s: r["macro"] for s, r in report["original_selection_test"].items()},
        "selection_objective_effect_AP_minus_R1": {
            s: {m: r["AP"]["macro"][m] - r["R@1"]["macro"][m] for m in METRICS}
            for s, r in report["test"].items()},
        "protocol_effect_common_R1_minus_original": report["common_R1_minus_original_protocol_selection_test"],
        "runtime_drift_original_current_minus_historical": report["current_runtime_original_minus_historical_original"],
    }
    rows = {name: {metric: dict(mean=100 * statistics.mean(v[metric] for v in seeds.values()),
                              sample_sd=100 * statistics.stdev(v[metric] for v in seeds.values()),
                              per_seed={s: 100 * v[metric] for s, v in seeds.items()})
                   for metric in METRICS} for name, seeds in samples.items()}
    limits = ["동일 시험셋이 이미 알려진 사후 선택 민감도 분석이다.",
              "시험 값으로 모델·입력·점수식·선택 목적을 고르지 않았다."]
    if method != "disco":
        limits.append("보존되지 않은 중간 에포크의 가중치까지 복원한 결과는 아니다.")
    if method == "seqot":
        limits.append("원래 시드별 stage1을 고정하고 보존된 stage2 99개씩만 비교한다. stage1 전수 선택은 미완료다.")
    if method == "imlpr":
        limits.append("시드별 원래 best와 final 두 텐서만 남아 있다. 100에포크 전수 AP 최적 선택을 뜻하지 않는다.")
    if method == "helios":
        limits.append("원래80에포크 중 주기저장·best·final의 고유26후보를 비교한다. 비보존 에포크의 전수 선택은 미완료다.")
    if method == "disco":
        limits.append("원래 20에포크씩의 가중치가 모두 보존되어 시드당 20개를 비교한다. 추가 학습은 없다.")
    write(OUT / f"final_delivery_{method}.json", dict(
        status="complete_for_retained_candidate_scope", completed_unix=time.time(), method=method,
        coverage=proof["counts"], selection_epochs={s: {o: r["epoch"] for o, r in choices.items()}
                                                  for s, choices in frozen["selected"].items()},
        original_epochs={s: r["candidate"]["epoch"] for s, r in frozen["original_choices"].items()},
        selected_checkpoints=frozen["selected"],
        original_control_sources={s: ("current_same_runtime" if "Original_runtime" in report["test"][s] else "historical_ledger_no_changed_choice") for s in report["test"]}, units="% for rows; percentage points for effects",
        rows=rows, report=str(source.relative_to(ROOT)), report_sha256=sha(source),
        selection_sha256=proof["selection_sha256"], independent_verification_sha256=sha(proof_path),
        independent_metric_max_error=proof["independent_metric_max_error"],
        limitations=limits, main_table_changed=False, current_NSD_reselected=False))


def watch() -> None:
    """로컬 두 방법의 완료를 기다려 GPU 없이 독립 검산·전달을 잇는다."""
    lock = (OUT / "delivery_watch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = OUT / "delivery_watch_state.json"
    completed = []
    for method in ("seqot", "imlpr"):
        target = OUT / f"final_delivery_{method}.json"
        if target.exists():
            completed.append(method)
            continue
        while not (OUT / "retained" / f"report_{method}.json").exists():
            write(state_path, dict(status="waiting_report", active=method, completed=completed,
                                   pid=__import__("os").getpid(), updated_unix=time.time()))
            time.sleep(15)
        write(state_path, dict(status="verifying", active=method, completed=completed,
                               pid=__import__("os").getpid(), updated_unix=time.time()))
        log = OUT / "logs" / f"independent_verification_{method}.log"
        with log.open("a") as handle:
            result = subprocess.run([sys.executable, "-u", str(ROOT / "scripts/_review_selection_full_verify.py"),
                                     "--method", method], stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            write(state_path, dict(status="verification_failed", active=method, completed=completed,
                                   returncode=result.returncode, updated_unix=time.time()))
            raise RuntimeError(f"independent verification failed: {method}; {log}")
        summarize(method)
        completed.append(method)
        paths = [str(OUT / f"independent_verification_{method}.json"), str(target)]
        subprocess.run(["rsync", "-aO", "--no-owner", "--no-group", "--no-perms", *paths,
                        "joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/" + str(OUT.relative_to(ROOT)) + "/"], check=True)
    write(state_path, dict(status="complete", completed=completed, updated_unix=time.time()))


if __name__ == "__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    watch() if args.watch else summarize(args.method)
