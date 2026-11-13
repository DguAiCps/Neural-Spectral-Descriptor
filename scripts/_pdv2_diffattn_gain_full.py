#!/usr/bin/env python3
"""입력 투영까지 새 스펙트럼 조절 과제에 맞추는 같은 예산 구조 대조."""
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

PARENT = base.OUT
OUT = base.ROOT / "results/pdv2_diffattn_gain_full_20261104"
REFERENCE_MODEL = base.model
KINDS = ("diff", "standard", "graph_free")
ARMS = {
    f"{k}_{tag}": (k, lr) for k in KINDS for tag, lr in (("lr5", 1e-5), ("lr4", 1e-4))
}


def prepare() -> dict:
    """고정 입력 표현 제약을 푸는 새 대조의 범위를 고정한다."""
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        spec = base.run.READ(path)
        for p, h in spec["sources"].items():
            assert base.run.HASH(base.ROOT / p) == h, p
    else:
        spec = dict(base.run.READ(PARENT / "protocol.json"))
        spec["sources"] = dict(spec["sources"])
        spec.update(
            created_unix=time.time(),
            arms=ARMS,
            architecture="기존Diff/표준/그래프없는모델의스펙트럼이득출력유지. 출력weight0·공통이득bias동일. 새모듈없음. 원키검색/시간판정고정.",
            training="사용하는전체기존모듈학습:입력투영·정규화학습계수·간선부호기·두층·출력층. BN통계는추론모드고정,미사용게이트와키재계량고정. 동일학습8승자·양음64·512갱신·4/8epoch·lr1e-5/1e-4·같은배치. 세구조모두동일범위로해제.",
            rationale="두층+출력층과제에서Diff가표준보다낮고값제곱/혼합도회복하지못함. 현재과제와다른서술자학습의입력투영을고정한제약을분리. 고정투영이원인이라고확정한실험은아님.",
            selection="전체Diff최적AP가전체standard/graph_free각최적및기존부분학습Diff/head_only/global최적보다+.1%p,F1손실<=.3%p,R100<=1%p이면매칭연구후보. 고정원키R1이BEV보다3.065%p낮아기존허용3%p에못미친다는한계유지. 검색복구·3시드·고정후시험전까지최종목표달성주장없음. 반복검증탐색.",
            verification="앞실험의원키·실제이득·최종판정접두검사정책을그대로고정. val_checked원장.",
        )
        paths = [
            Path(__file__),
            Path(checked.__file__),
            PARENT / "summary.json",
            base.ROOT / "results/pdv2_diffattn_gain_moments_20261104/summary.json",
            base.ROOT
            / "results/pdv2_diffattn_gain_attention_probe_20261104/summary.json",
        ]
        spec["sources"].update(
            {str(p.relative_to(base.ROOT)): base.run.HASH(p) for p in paths}
        )
        base.run.WRITE(path, spec)
        base.run.WRITE(
            OUT / "validation_runtime_amendment.json",
            dict(
                scope="이미검증한접두검사정책을이번실험시작부터고정",
                parent_sha256=base.run.HASH(
                    PARENT / "validation_runtime_amendment.json"
                ),
            ),
        )
    pools = OUT / "pools"
    if not pools.exists():
        pools.symlink_to("../" + PARENT.name + "/pools", target_is_directory=True)
    base.run.OUT = OUT
    return spec


def model(kind: str, checkpoint: Path = None) -> torch.nn.Module:
    """사용하는 기존 신경망 전체를 갱신하고 BN 통계는 고정한다."""
    net = REFERENCE_MODEL(kind, checkpoint)
    if kind != "global":
        for name, p in net.named_parameters():
            p.requires_grad_(
                not (name.startswith("net.gnn.gate.") or name.endswith("key_scale"))
            )
    return net.eval()


def finish() -> None:
    """부분학습·표준·그래프없는대조와 최종 성능을 비교한다."""
    spec = prepare()
    rec = {
        s: base.run.READ(OUT / "val_checked" / (s.replace("/", "_") + ".json"))
        for s in spec["val"]
    }
    results = {
        k: base.run.base.macro({s: r["metrics"][k] for s, r in rec.items()})
        for k in next(iter(rec.values()))["metrics"]
    }
    best = {
        kind: max(
            [
                f"{a}_{ep}"
                for a, (k, _) in ARMS.items()
                if k == kind
                for ep in base.EPOCHS
            ],
            key=lambda tag: results[tag]["macro"]["AP"],
        )
        for kind in KINDS
    }
    old = base.run.READ(PARENT / "summary.json")
    controls = {
        "partial_" + k: old["results"][old["best"][k]]["macro"]
        for k in ("diff", "head_only", "global")
    }
    controls.update({k: results[best[k]]["macro"] for k in ("standard", "graph_free")})
    candidate = best["diff"]
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
    """동일 출발점의 전체 학습과 최종 검증을 완료한다."""
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
    parser.add_argument("--arm", default="diff_lr4")
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
