#!/usr/bin/env python3
"""스펙트럼 조절에서 차이의 부호와 크기 전달을 동일 용량으로 비교한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import torch

import _pdv2_diffattn_gain_readout as base
import _pdv2_diffattn_gain_readout_checked as checked
from gnn.moment_difference import MomentDifferenceConv

PARENT = base.OUT
OUT = base.ROOT / "results/pdv2_diffattn_gain_moments_20261104"
REFERENCE_MODEL = base.model
KINDS = ("signed", "square", "odd_even")
ARMS = {
    f"{k}_{tag}": (k, lr) for k in KINDS for tag, lr in (("lr4", 1e-4), ("lr3", 1e-3))
}


def prepare() -> dict:
    """새 값 변환과 고정 대조의 확장 판단을 먼저 기록한다."""
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        spec = base.run.READ(path)
        for p, h in spec["sources"].items():
            assert base.run.HASH(base.ROOT / p) == h, p
    else:
        spec = dict(base.run.READ(PARENT / "protocol.json"))
        spec["sources"] = dict(spec["sources"])
        paths = [
            Path(__file__),
            base.ROOT / "src/gnn/moment_difference.py",
            base.ROOT / "tests/gnn/test_moment_difference.py",
            PARENT / "summary.json",
            PARENT / "validation_runtime_amendment.json",
            Path(checked.__file__),
        ]
        spec.update(
            created_unix=time.time(),
            arms=ARMS,
            method="차이K-tanh·기하상호작용·Q/K/V/MLP매개변수·두층·기존출력층을유지. signed=기존tanh(V차이),square=모든채널제곱,odd_even=head별절반부호/절반제곱. signed재실행은코드재현대조. 새층이나학습매개변수추가없음.",
            rationale="스펙트럼성분안정성을조절하는새과제에서평균차이와변화량의차이를비교. 기존서술자학습에서odd_even을이미시험했으므로완전한신규연산이라고주장하지않음. 상쇄가현재실패원인이라고단정하지않음.",
            selection="square/odd_even최적AP가동일signed최적및앞실험의standard/graph_free/head_only/global최적보다+.1%p이고F1손실<=.3%p,R100손실<=1%p이면매칭연구후보. 앞원키검색의고정R1은94.145%로BEV보다3.065%p낮아어떤출력학습도기존R1허용3%p조건을충족할수없다. 그기준실패를숨기지않고검색복구와시험비교를완료하기전목표달성이나comparable확증을주장하지않음. 후보통과후3시드고정확인. 시험없음.",
            verification="앞실험의검증검사수정정책을처음부터고정. 실제절단/같은크기미래제거의최종점수8e-6. 원키허용오차유지·실제이득차이<=.002·정규화전출력차이는기록. 원장val_checked사용.",
        )
        spec["sources"].update(
            {str(p.relative_to(base.ROOT)): base.run.HASH(p) for p in paths}
        )
        base.run.WRITE(path, spec)
        base.run.WRITE(
            OUT / "validation_runtime_amendment.json",
            dict(
                scope="앞실험에서검증한정책의새실험시작시점고정. 이번실험의사후수정이아님.",
                parent_sha256=base.run.HASH(
                    PARENT / "validation_runtime_amendment.json"
                ),
            ),
        )
    pools = OUT / "pools"
    if not pools.exists():
        pools.symlink_to("../" + PARENT.name + "/pools", target_is_directory=True)
    assert pools.resolve() == (PARENT / "pools").resolve()
    base.run.OUT = OUT
    return spec


def model(kind: str, checkpoint: Path = None) -> torch.nn.Module:
    """기존 체크포인트 가중치를 유지한 채 값의 부호/크기 처리만 교체한다."""
    if kind == "global":
        return REFERENCE_MODEL("global", checkpoint)
    net = REFERENCE_MODEL("diff", checkpoint)
    for i, conv in enumerate(net.net.gnn.convs):
        net.net.gnn.convs[i] = MomentDifferenceConv(conv, kind)
    return net.cuda().eval()


def finish() -> None:
    """기존 구조 대조까지 포함한 같은 최종 검증 지표를 집계한다."""
    spec = prepare()
    rec = {
        s: base.run.READ(OUT / "val_checked" / (s.replace("/", "_") + ".json"))
        for s in spec["val"]
    }
    tags = next(iter(rec.values()))["metrics"]
    results = {
        k: base.run.base.macro({s: r["metrics"][k] for s, r in rec.items()})
        for k in tags
    }
    best = {
        kind: max(
            [
                f"{a}_{ep}"
                for a, (k, _) in ARMS.items()
                if k == kind
                for ep in base.EPOCHS
            ],
            key=lambda t: results[t]["macro"]["AP"],
        )
        for kind in KINDS
    }
    old = base.run.READ(PARENT / "summary.json")
    controls = {
        k: old["results"][old["best"][k]]["macro"]
        for k in ("standard", "graph_free", "head_only", "global")
    }
    controls["signed"] = results[best["signed"]]["macro"]
    candidate = max(
        (best[k] for k in ("square", "odd_even")),
        key=lambda t: results[t]["macro"]["AP"],
    )
    m = results[candidate]["macro"]
    tests = {
        f"{metric}_{c}": m[metric] >= v[metric] + delta
        for c, v in controls.items()
        for metric, delta in (("AP", 0.001), ("max_F1", -0.003), ("R@100P", -0.01))
    }
    selected = candidate if all(tests.values()) else None
    base.run.WRITE(
        OUT / "summary.json",
        dict(
            results=results,
            best=best,
            checks=tests,
            selected_matching_candidate=selected,
            comparable_not_confirmed=True,
            test_used=False,
            ledger_root="val_checked",
        ),
    )
    print(
        "FINISHED",
        selected,
        {k: results[v]["macro"] for k, v in best.items()},
        flush=True,
    )


def campaign() -> None:
    """여섯 동일 예산 학습과 실제 최종 판정을 완료한다."""
    spec = prepare()
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)

    def call(args, tag):
        with (logs / (tag + ".log")).open("w") as stream:
            subprocess.run(
                [sys.executable, __file__, *args],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    for i, a in enumerate(ARMS, 1):
        call(["train", "--arm", a], "train_" + a)
        base.run.WRITE(
            OUT / "status.json", dict(phase="training", done=i, total=len(ARMS))
        )
        print("TRAINED", a, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(call, ["val", "--sequence", s], "val_" + s.replace("/", "_"))
            for s in spec["val"]
        ]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            base.run.WRITE(
                OUT / "status.json",
                dict(phase="validation", done=i, total=len(futures)),
            )
            print("VALIDATED", i, len(futures), flush=True)
    finish()
    base.run.WRITE(OUT / "status.json", dict(phase="complete", test_used=False))


base.OUT = OUT
base.ARMS = ARMS
base.KINDS = KINDS
base.prepare = prepare
base.model = model
checked.prepare = prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "val", "finish", "run"))
    parser.add_argument("--arm", default="square_lr4")
    parser.add_argument("--sequence", default="kitti/09")
    args = parser.parse_args()
    base.run.configure()
    if args.action == "train":
        base.train(args.arm)
    elif args.action == "val":
        checked.evaluate(args.sequence)
    elif args.action == "run":
        campaign()
    else:
        globals()[args.action]()
