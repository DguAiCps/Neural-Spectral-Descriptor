#!/usr/bin/env python3
"""학습 자료의 초기 기울기 크기로 수락 손실의 비중을 고정한 후속 대조."""
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
import _pdv2_nonrevisit_learning as base

PRIOR = base.OUT
OUT = ROOT / "results/pdv2_nonrevisit_balanced_20261104"
ARMS = ("positive_only", "all_queries")
ORIGINAL_LOSS = base.acceptance_loss


def prepare() -> dict:
    """같은 배치·예산과 학습 전용 계수를 원래 실험과 분리해 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = base.READ(target)
        for p, h in spec["sources"].items():
            assert base.HASH(ROOT / p) == h, p
        return spec
    spec = base.READ(PRIOR / "protocol.json")
    diagnostic = base.READ(PRIOR / "gradient_diagnostic.json")
    ratios = [v["gradient_norm"]["retrieval"]/v["gradient_norm"]["all"]
              for v in diagnostic["sequences"].values()]
    gain = float(np.median(ratios))
    sources = {p: base.HASH(ROOT / p) for p in spec["sources"]}
    for p in (Path(__file__), PRIOR / "protocol.json", PRIOR / "gradient_diagnostic.json",
              PRIOR / "calibration.json", PRIOR / "runtime_amendments.json"):
        sources[str(p.relative_to(ROOT))] = base.HASH(p)
    spec.update(created_unix=time.time(), arms=ARMS, epochs=[4], sources=sources,
        acceptance_gain=gain, gradient_ratios=ratios,
        continuation="기본 실험의 초기 학습 배치에서 판정 기울기가 작음을 확인한 후속. 새 전체 질의 검증 결과로 계수를 고르지 않음. 처음 선택한 같은 checkpoint에서 새로256회. 검색 손실 그대로, 양성전용·전체 수락BCE 모두 같은 중앙값 계수 적용. 두 조건 모두 epoch4만 평가.",
        selection="epoch4의 동일5혼합계수 중 검증 최종AP 최대, 동률 작은계수. 원래256업데이트 대조와 함께 보고. 기본2/4epoch 실험과 선택 기회 차이를 구분. 추가 시험·원고 변경 없음.")
    base.WRITE(target, spec)
    (OUT / "pools").symlink_to(PRIOR / "pools", target_is_directory=True)
    fit = base.READ(PRIOR / "calibration.json")
    fit.update(protocol_sha256=base.HASH(target), reused_from=str((PRIOR / "calibration.json").relative_to(ROOT)))
    base.WRITE(OUT / "calibration.json", fit)
    print("BALANCED_FROZEN", gain, ratios, flush=True)
    return spec


def activate() -> dict:
    """기존 학습 구현에 고정한 스칼라 손실 비중만 공급한다."""
    spec = prepare()
    base.OUT = OUT
    base.prepare = prepare
    base.EPOCHS = (4,)
    base.acceptance_loss = lambda *args, **kwargs: spec["acceptance_gain"] * ORIGINAL_LOSS(*args, **kwargs)
    return spec


def finish() -> None:
    """같은 체크포인트 예산의 두 조건을 최종 지표로 비교한다."""
    spec = prepare()
    selected, all_results = {}, {}
    for arm in ARMS:
        options = []
        for lam in base.LAMBDAS:
            per = {}
            for seq in spec["val"]:
                p = OUT / arm / "epoch4/val" / (seq.replace("/", "_")+".json")
                r = base.READ(p)
                assert r["status"] == "passed" and base.HASH(p.with_suffix(".npz")) == r["ledger_sha256"]
                per[seq] = r["metrics"][f"lambda_{lam:g}"]
            result = dict(arm=arm, epoch=4, coefficient=lam, **base.base.macro(per), sequences=per)
            all_results[f"{arm}__epoch4__lambda{lam:g}"] = result
            options.append(result)
        selected[arm] = sorted(options, key=lambda r: (-round(r["macro"]["AP"],12), r["coefficient"]))[0]
        own = base.READ(OUT / arm / "training_complete.json")
        previous = base.READ(PRIOR / arm / "training_complete.json")
        assert own["updates"] == previous["updates"] == 256
        assert [r["batch_hashes"] for r in own["records"]] == [r["batch_hashes"] for r in previous["records"]]
    base.WRITE(OUT / "summary.json", dict(selected=selected, all_results=all_results,
        acceptance_gain=spec["acceptance_gain"], equal_updates_and_samples=True, test_used=False,
        protocol_sha256=base.HASH(OUT / "protocol.json")))
    base.WRITE(OUT / "status.json", dict(status="complete", training_conditions=2, validation_jobs=18, test_used=False))
    print("BALANCED_SUMMARY", {a:(v["coefficient"],v["macro"]) for a,v in selected.items()}, flush=True)


def run() -> None:
    """두 손실 조건을 학습하고 원래 실험과 겹치지 않는 원장을 쓴다."""
    spec = prepare()
    log = OUT / "logs"
    log.mkdir(exist_ok=True)
    def call(args: list, label: str) -> None:
        with (log / (label+".log")).open("w") as f:
            subprocess.run([sys.executable, __file__, *args], stdout=f, stderr=subprocess.STDOUT, check=True)
    try:
        for arm in ARMS:
            base.WRITE(OUT / "status.json", dict(status="running", phase="training", arm=arm))
            call(["train", "--arm", arm], "train_"+arm)
        # 원 실험 두 작업자와 함께 돌아 GPU 추론·CPU 검색 경합을 제한한다.
        jobs = [(a,s) for a in ARMS for s in spec["val"]]
        def evaluate(job: tuple) -> None:
            a,s = job
            call(["val", "--arm", a, "--sequence", s], "val_"+a+"_"+s.replace("/", "_"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for i, f in enumerate(concurrent.futures.as_completed([pool.submit(evaluate,j) for j in jobs]),1):
                f.result()
                base.WRITE(OUT / "status.json", dict(status="running", phase="validation", done=i, total=18))
                print("BALANCED_PROGRESS", i, 18, flush=True)
        finish()
    except Exception as exc:
        base.WRITE(OUT / "status.json", dict(status="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "train", "val", "run", "finish"))
    p.add_argument("--arm", choices=ARMS, default="all_queries")
    p.add_argument("--sequence", default="kitti/09")
    a = p.parse_args()
    base.configure()
    if a.action == "prepare": prepare()
    elif a.action == "run": run()
    elif a.action == "finish": finish()
    else:
        activate()
        if a.action == "train": base.train(a.arm)
        else: base.validate(a.arm, 4, a.sequence)
