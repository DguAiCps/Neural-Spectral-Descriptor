#!/usr/bin/env python3
"""BEVPlace++와 비교할 성능을 유지하면서 기존 문맥 모듈 학습을 확장한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import _pdv2_nonrevisit_learning as run

PRIOR = run.OUT
OUT = ROOT / "results/pdv2_diffattn_comparable_20261104"
ARMS = ("unit_weight", "half_weight")
EPOCHS = (4, 8, 16)
ORIGINAL_CONTEXT = run.context_score
ORIGINAL_LOSS = run.acceptance_loss
BEV = ROOT / "results/pdv2_review_response_20261104/bev"


def prepare() -> dict:
    """학습량·판정 비중·비교 기준을 새 평가 전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = run.READ(target)
        for name, digest in spec["sources"].items():
            assert run.HASH(ROOT / name) == digest, name
        return spec
    spec = run.READ(PRIOR / "protocol.json")
    balanced = ROOT / "results/pdv2_nonrevisit_balanced_20261104"
    paths = [ROOT / p for p in spec["sources"]]
    paths += [Path(__file__), PRIOR / "protocol.json", PRIOR / "calibration.json",
              PRIOR / "runtime_amendments.json", balanced / "protocol.json",
              BEV / "val_summary.json", ROOT / "scripts/_review_response_bev.py"]
    for seq in spec["val"]:
        for folder in (PRIOR / "initial/epoch0/val", BEV / "val/s1"):
            for suffix in (".json", ".npz"):
                paths.append(folder / (seq.replace("/", "_")+suffix))
    spec.update(created_unix=time.time(), arms=ARMS, epochs=EPOCHS,
        acceptance_gain=run.READ(balanced / "protocol.json")["acceptance_gain"],
        training_context_coefficients={"unit_weight": 1., "half_weight": .5},
        objective="현재 DiffAttnConv 구조·차이 메시지·주변 투영·게이트를 함께 추가 학습. 기존 InfoNCE + 재방문 수락 BCE. BCE의 문맥 계수1 또는.5, 학습량256/512/1024업데이트를 대조. half_weight의 학습·추론 비중 일치 효과는 동일 추론.5에서 따로 보고. 구조·검색·정렬·온라인 지연 변경 없음.",
        selection="검증9·초기시드1. 모든 체크포인트와 학습 전 모델에 같은5추론비중0/.125/.25/.5/1 제공. 원고 성능 초과나 모든 지표 보존을 요구하지 않음. BEVPlace++ 시드1 동일 시간 판정 대비 AP/F1 각각2%p·R100/R1 각각3%p 이내 열세를 후속 시험 작업용 기준으로 고정. 학습 전 최적AP보다+.1%p, 같은비중 학습전보다+.1%p인 후보 중 AP 최대, 동률 작은비중→이른epoch→arm이름. 이 허용폭은 통계적 동등성 선언이 아니며 데이터셋별 격차도 보고. 통과하면 모델·비중을 해시로 고정하고 시험46에 한 번 적용. 시험으로 재선택하지 않음.",
        limitations="단일 초기시드와 반복 사용 검증의 탐색. 보유 시험분할도 과거 개발 노출이 있어 독립 미노출 확증 아님. BEVPlace++와 구조·학습 입력량·선택 목적까지 동일하지 않음. 원고 표·그림 자동 변경 없음.",
        sources={str(p.relative_to(ROOT)): run.HASH(p) for p in paths})
    run.WRITE(target, spec)
    (OUT / "pools").symlink_to(PRIOR / "pools", target_is_directory=True)
    fit = run.READ(PRIOR / "calibration.json")
    fit.update(protocol_sha256=run.HASH(target), reused_from=str(PRIOR.relative_to(ROOT)))
    run.WRITE(OUT / "calibration.json", fit)
    return spec


def activate(training_arm: str = None) -> dict:
    """기존 구현에 학습 계수만 공급하며 추론 문맥은 원래 계산을 유지한다."""
    spec = prepare()
    run.OUT = OUT
    run.prepare = prepare
    run.EPOCHS = EPOCHS
    if training_arm is not None:
        fit = run.calibration()
        coefficient = spec["training_context_coefficients"][training_arm]
        mean = fit["context_mean"]
        run.context_score = lambda *a, **kw: mean + coefficient * (ORIGINAL_CONTEXT(*a, **kw)-mean)
        run.acceptance_loss = lambda *a, **kw: spec["acceptance_gain"] * ORIGINAL_LOSS(*a, **kw)
    return spec


def reference() -> dict:
    """같은 질의·거리 라벨의 BEVPlace++ 기본·시간 판정 평균을 재계산한다."""
    spec = prepare()
    results = {}
    for label in ("native", "temporal"):
        per = {}
        for seq in spec["val"]:
            name = seq.replace("/", "_")
            meta = run.READ(BEV / "val/s1" / (name+".json"))
            path = BEV / "val/s1" / (name+".npz")
            assert run.HASH(path) == meta["sha256"]
            row = run.ARR(path)
            nsd = run.ARR(PRIOR / "initial/epoch0/val" / (name+".npz"))
            for field in ("query_idx", "min_past_gt_dist"):
                np.testing.assert_array_equal(row[field], nsd[field])
            score = row["score_negl2"] if label == "native" else row["score"]
            per[seq] = run.base.independent_metrics(row, score)
        results[label] = dict(**run.base.macro(per), sequences=per)
        expected = run.READ(BEV / "val_summary.json")["per_seed"]["1"][label]
        for metric in run.base.METRICS:
            np.testing.assert_allclose(results[label]["macro"][metric], expected[metric], atol=1e-12, rtol=0)
    run.WRITE(OUT / "bev_validation.json", results)
    return results


def finish() -> None:
    """최종 검증만으로 후보를 선택하고 시험 전에 학습 유무 설정을 고정한다."""
    spec = activate()
    bev = reference()
    all_results, metric_rows = {}, 0
    for arm in ("frozen", *ARMS):
        for epoch in ((0,) if arm == "frozen" else EPOCHS):
            folder = PRIOR / "initial/epoch0/val" if arm == "frozen" else OUT / arm / f"epoch{epoch}/val"
            per = {lam: {} for lam in run.LAMBDAS}
            for seq in spec["val"]:
                path = folder / (seq.replace("/", "_")+".json")
                record, row = run.READ(path), run.ARR(path.with_suffix(".npz"))
                assert record["status"] == "passed" and len(record["prefix_checks"]) == 2
                assert run.HASH(path.with_suffix(".npz")) == record["ledger_sha256"]
                if arm != "frozen":
                    assert record["checkpoint_sha256"] == run.HASH(OUT / arm / f"epoch{epoch}.pth")
                for lam in run.LAMBDAS:
                    label = f"lambda_{lam:g}"
                    metric = run.base.independent_metrics(row, row[label])
                    assert metric == record["metrics"][label]
                    per[lam][seq] = metric
                    metric_rows += 1
            for lam in run.LAMBDAS:
                all_results[f"{arm}_{epoch}_{lam:g}"] = dict(arm=arm, epoch=epoch, coefficient=lam,
                    **run.base.macro(per[lam]), sequences=per[lam])
    order = lambda r: (-round(r["macro"]["AP"], 12), r["coefficient"], r["epoch"], r["arm"])
    frozen = sorted([r for r in all_results.values() if r["arm"] == "frozen"], key=order)[0]
    checks, eligible = {}, []
    tolerances = {"AP": .02, "max_F1": .02, "R@100P": .03, "R@1": .03}
    for name, row in all_results.items():
        if row["arm"] == "frozen":
            continue
        paired = all_results[f"frozen_0_{row['coefficient']:g}"]
        tests = {m:row["macro"][m] >= bev["temporal"]["macro"][m]-tol for m,tol in tolerances.items()}
        tests.update(learned_over_retuned_frozen=row["macro"]["AP"] >= frozen["macro"]["AP"]+.001,
                     learned_over_same_mixture=row["macro"]["AP"] >= paired["macro"]["AP"]+.001)
        checks[name] = tests
        if all(tests.values()):
            eligible.append(row)
    selected = sorted(eligible, key=order)[0] if eligible else None
    train = {a:run.READ(OUT / a / "training_complete.json") for a in ARMS}
    assert {v["updates"] for v in train.values()} == {1024}
    orders = [[v["batch_hashes"] for v in t["records"]] for t in train.values()]
    assert orders[0] == orders[1]
    summary = dict(all_results=all_results, selected=selected, frozen_best=frozen,
        bev=bev, advancement_checks=checks, test_used=False,
        protocol_sha256=run.HASH(OUT / "protocol.json"))
    run.WRITE(OUT / "validation_summary.json", summary)
    run.WRITE(OUT / "validation_checks.json", dict(status="passed", new_ledgers=54,
        metric_rows=metric_rows, new_prefix_checks=108, equal_training_samples=True, test_used=False))
    if selected is not None:
        checkpoint = OUT / selected["arm"] / f"epoch{selected['epoch']}.pth"
        freeze = dict(created_unix=time.time(), seed=1, selected=selected,
            selected_checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=run.HASH(checkpoint),
            frozen_best_coefficient=frozen["coefficient"],
            protocol_sha256=run.HASH(OUT / "protocol.json"),
            validation_sha256=run.HASH(OUT / "validation_summary.json"),
            test=["/".join(s) for s in run.current.TEST],
            test_policy="선택 모델과 학습 전 모델만 같은 시험46에 고정 적용. 학습 전은 검증 최적비중과 선택모델과 같은비중을 모두 보고. 시험 결과로 모델·비중·학습을 다시 선택하지 않음.",
            test_used_for_selection=False)
        target = OUT / "test_freeze.json"
        if target.exists():
            old = run.READ(target)
            assert old["checkpoint_sha256"] == freeze["checkpoint_sha256"]
            assert old["selected"]["coefficient"] == selected["coefficient"]
        else:
            run.WRITE(target, freeze)
    run.WRITE(OUT / "status.json", dict(status="validation_complete", selected=None if selected is None else
        {k:selected[k] for k in ("arm", "epoch", "coefficient", "macro")}, test_used=False))
    print("VALIDATION_RESULT", None if selected is None else {k:selected[k] for k in ("arm", "epoch", "coefficient", "macro")}, flush=True)


def campaign() -> None:
    """학습과 검증을 완료하며 시험은 고정 파일을 만든 뒤 별도로 실행한다."""
    spec = prepare()
    reference()
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    def call(args: list, name: str) -> None:
        with (logs / (name+".log")).open("w") as log:
            subprocess.run([sys.executable, __file__, *args], stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        for arm in ARMS:
            run.WRITE(OUT / "status.json", dict(status="running", phase="training", arm=arm))
            call(["train", "--arm", arm], "train_"+arm)
        jobs = [(a,e,s) for a in ARMS for e in EPOCHS for s in spec["val"]]
        def evaluate(job: tuple) -> None:
            arm, epoch, seq = job
            call(["val", "--arm", arm, "--epoch", str(epoch), "--sequence", seq],
                 f"val_{arm}_{epoch}_{seq.replace('/', '_')}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for i, future in enumerate(concurrent.futures.as_completed([pool.submit(evaluate,j) for j in jobs]), 1):
                future.result()
                run.WRITE(OUT / "status.json", dict(status="running", phase="validation", done=i, total=len(jobs)))
                print("VALIDATION_PROGRESS", i, len(jobs), flush=True)
        finish()
    except Exception as exc:
        run.WRITE(OUT / "status.json", dict(status="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "val", "finish", "run"))
    parser.add_argument("--arm", choices=ARMS, default="unit_weight")
    parser.add_argument("--epoch", type=int, default=16)
    parser.add_argument("--sequence", default="kitti/09")
    args = parser.parse_args()
    run.configure()
    if args.action == "prepare":
        prepare()
    elif args.action == "train":
        activate(args.arm)
        run.train(args.arm)
    elif args.action == "val":
        activate()
        run.validate(args.arm, args.epoch, args.sequence)
    elif args.action == "finish":
        finish()
    else:
        campaign()
