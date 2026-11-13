#!/usr/bin/env python3
"""주변 투영을 고정한 DiffAttnConv 두 층의 큰 갱신 폭을 대조한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]
import _pdv2_diffattn_margin as experiment

run = experiment.run
PRIOR = experiment.OUT
OUT = ROOT / "results/pdv2_diffattn_step_20261104"
ARMS = ("conv_only", "conv_guard")


def prepare() -> dict:
    """낮은 학습률 결과를 본 후속임을 밝히고 새 평가 전에 설정을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = run.READ(target)
        for p,h in spec["sources"].items():
            assert run.HASH(ROOT / p) == h, p
        return spec
    spec = run.READ(PRIOR / "protocol.json")
    paths = [ROOT / p for p in spec["sources"]]
    paths += [Path(__file__), PRIOR / "protocol.json", PRIOR / "runtime_amendments.json"]
    spec.update(created_unix=time.time(), arms=ARMS, learning_rate=1e-4,
        objective="기존 DiffAttnConv 두 층의 Q/K/V·기하 bias MLP 14개 텐서만 추가 학습. 나머지 전부 고정. conv_only와 conv_guard의 손실은 앞 실험과 같고 학습률만1e-5에서1e-4로 높인다. 같은 초기 체크포인트·표본·256회·epoch4·판정 혼합.5. 보호계수10 유지.",
        continuation="이전1e-5 conv_only의 검증 AP 이득+.018%p가 작아 확장한 학습률 탐색. 이전 전체 문맥 모듈 학습과 학습률·갱신범위가 모두 다름을 구분. 더 큰 학습률을 독립 확증처럼 서술하지 않음.",
        sources={str(p.relative_to(ROOT)):run.HASH(p) for p in paths})
    run.WRITE(target, spec)
    (OUT / "pools").symlink_to(PRIOR / "pools", target_is_directory=True)
    fit = run.READ(PRIOR / "calibration.json")
    fit.update(protocol_sha256=run.HASH(target), reused_from=str(PRIOR.relative_to(ROOT)))
    run.WRITE(OUT / "calibration.json", fit)
    return spec


def activate() -> dict:
    """독립 실행에서 앞 실험의 동일 학습·평가 구현을 재사용한다."""
    experiment.OUT = OUT
    experiment.prepare = prepare
    return experiment.activate()


def finish() -> None:
    """원장을 검산하고 동일 판정의 기존 모델·작은 갱신과 대조한다."""
    spec = activate()
    prior = run.READ(PRIOR / "summary.json")
    results = {"frozen":prior["results"]["frozen"],
               "full_control":prior["results"]["full_control"]}
    for arm in ARMS:
        per = {}
        for seq in spec["val"]:
            p = OUT / arm / "epoch4/val" / (seq.replace("/", "_")+".json")
            record, row = run.READ(p), run.ARR(p.with_suffix(".npz"))
            assert record["status"] == "passed" and len(record["prefix_checks"]) == 2
            assert record["ledger_sha256"] == run.HASH(p.with_suffix(".npz"))
            assert record["checkpoint_sha256"] == run.HASH(OUT / arm / "epoch4.pth")
            for label in ("lambda_0", "lambda_0.5"):
                assert record["metrics"][label] == run.base.independent_metrics(row, row[label])
            per[seq] = record["metrics"]["lambda_0.5"]
        results[arm] = dict(**run.base.macro(per), sequences=per)
        t = run.READ(OUT / arm / "training_complete.json")
        ref = run.READ(PRIOR / arm / "training_complete.json")
        assert t["updates"] == ref["updates"] == 256
        assert [v["batch_hashes"] for v in t["records"]] == [v["batch_hashes"] for v in ref["records"]]
        assert all(p.startswith("gnn.convs.") for p in t["changed_parameters"])
    deltas = {a:{ref:{m:100*(results[a]["macro"][m]-results[ref]["macro"][m]) for m in run.base.METRICS}
                 for ref in ("frozen", "full_control")} for a in ARMS}
    checks = {a:deltas[a]["frozen"]["AP"]>=.1 and all(deltas[a]["frozen"][m]>=0 for m in ("max_F1", "R@100P", "R@1")) for a in ARMS}
    selected = max((a for a in ARMS if checks[a]), key=lambda a:results[a]["macro"]["AP"], default=None)
    run.WRITE(OUT / "summary.json", dict(results=results, delta_pp=deltas,
        low_learning_rate={a:prior["results"][a] for a in ARMS}, selected=selected,
        advancement_checks=checks, fixed_coefficient=.5, test_used=False,
        protocol_sha256=run.HASH(OUT / "protocol.json")))
    run.WRITE(OUT / "completion_validation.json", dict(status="passed", ledgers=18,
        metric_rows=36, prefix_checks=36, checkpoints=2,
        frozen_parameter_audit=True, equal_samples_and_updates=True, test_used=False))
    run.WRITE(OUT / "status.json", dict(status="complete", selected=selected, test_used=False))
    print("RESULT", {a:r["macro"] for a,r in results.items()}, "SELECTED",selected,flush=True)


def campaign() -> None:
    """두 조건의 추가 학습과 최종 검증을 실행한다."""
    spec = prepare()
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    def call(args: list, name: str) -> None:
        with (logs / (name+".log")).open("w") as log:
            subprocess.run([sys.executable, __file__, *args], stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        for arm in ARMS:
            run.WRITE(OUT / "status.json", dict(status="running", phase="training", arm=arm))
            call(["train", "--arm", arm], "train_"+arm)
        def evaluate(job: tuple) -> None:
            arm, seq = job
            call(["val", "--arm", arm, "--sequence",seq], "val_"+arm+"_"+seq.replace("/", "_"))
        jobs = [(a,s) for a in ARMS for s in spec["val"]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for i,f in enumerate(concurrent.futures.as_completed([pool.submit(evaluate,j) for j in jobs]),1):
                f.result()
                run.WRITE(OUT / "status.json",dict(status="running",phase="validation",done=i,total=len(jobs)))
                print("VAL_PROGRESS",i,len(jobs),flush=True)
        finish()
    except Exception as exc:
        run.WRITE(OUT / "status.json",dict(status="failed",error=str(exc)))
        raise


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("prepare","train","val","finish","run"))
    p.add_argument("--arm",choices=ARMS,default="conv_guard")
    p.add_argument("--sequence",default="kitti/09")
    args=p.parse_args()
    run.configure()
    if args.action == "prepare": prepare()
    elif args.action == "train":
        activate(); experiment.train(args.arm)
    elif args.action == "val":
        activate(); run.validate(args.arm,4,args.sequence)
    elif args.action == "finish": finish()
    else: campaign()
