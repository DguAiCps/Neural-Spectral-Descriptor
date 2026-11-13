#!/usr/bin/env python3
"""스펙트럼 조절 모델에서 attention·기하·메시지의 최종 효과를 분리한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import torch

import _pdv2_diffattn_gain_readout as e
import _pdv2_diffattn_gain_readout_checked as checked

PARENT = e.OUT
OUT = e.ROOT / "results/pdv2_diffattn_gain_attention_probe_20261104"
REFERENCE = e.model
MODES = (
    "learned",
    "uniform_attention",
    "content_only",
    "geometry_only",
    "zero_messages",
)


def prepare() -> dict:
    """재학습이 아닌 고정 모델 개입의 범위와 출처를 고정한다."""
    OUT.mkdir(exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = e.run.READ(target)
        for p, h in spec["sources"].items():
            assert e.run.HASH(e.ROOT / p) == h, p
    else:
        spec = dict(e.run.READ(PARENT / "protocol.json"))
        spec["sources"] = dict(spec["sources"])
        previous = e.run.READ(PARENT / "summary.json")
        tag = previous["best"]["diff"]
        arm, ep = tag.rsplit("_", 1)
        cp = PARENT / arm / f"epoch{ep}.pth"
        spec.update(
            created_unix=time.time(),
            selected_checkpoint=str(cp.relative_to(e.ROOT)),
            modes=MODES,
            scope="최고AP Diff출력층모델의고정검증개입. 균등attention=Q0·기하끔,내용만=기하끔,기하만=Q0,메시지0=각conv출력0. 나머지학습가중치·BN통계·출력층·고정후보·최종스펙트럼판정동일. 재학습대조나배포모델로선택하지않음. 마지막조건의BN상수출력은남아있어완전한그래프없는재학습과다름.",
            motivation="새출력과제에서도DiffAP80.522가표준81.427보다낮고제곱/혼합값도이를회복하지못함. 추가값변환을택하기전가중치선택과전달값의영향을최종점수에서분리.",
            test_used=False,
        )
        for p in (Path(__file__), Path(checked.__file__), cp, PARENT / "summary.json"):
            spec["sources"][str(p.relative_to(e.ROOT))] = e.run.HASH(p)
        e.run.WRITE(target, spec)
        e.run.WRITE(
            OUT / "validation_runtime_amendment.json",
            dict(
                scope="부모의확인된접두검사정책을시작전에고정",
                parent_sha256=e.run.HASH(PARENT / "validation_runtime_amendment.json"),
            ),
        )
    pools = OUT / "pools"
    if not pools.exists():
        pools.symlink_to("../" + PARENT.name + "/pools", target_is_directory=True)
    for mode in MODES:
        folder = OUT / mode
        folder.mkdir(exist_ok=True)
        cp = folder / "epoch8.pth"
        if not cp.exists():
            cp.symlink_to(e.ROOT / spec["selected_checkpoint"])
    e.run.OUT = OUT
    return spec


def model(kind: str, checkpoint: Path = None) -> torch.nn.Module:
    """체크포인트는 보존하고 메모리의 평가용 복사본만 개입한다."""
    if kind == "global":
        return REFERENCE("global", checkpoint)
    net = REFERENCE("diff", checkpoint)
    with torch.no_grad():
        for conv in net.net.gnn.convs:
            if kind in ("uniform_attention", "geometry_only"):
                conv.W_q.weight.zero_()
            if kind in ("uniform_attention", "content_only"):
                conv.geometry_enabled = False
            if kind == "zero_messages":
                conv.register_forward_hook(
                    lambda module, inputs, output: torch.zeros_like(output)
                )
    return net.eval()


def finish() -> None:
    """같은 승자의 최종 판정 차이를 구조별로 집계한다."""
    spec = prepare()
    rec = {
        s: e.run.READ(OUT / "val_checked" / (s.replace("/", "_") + ".json"))
        for s in spec["val"]
    }
    results = {
        k: e.run.base.macro({s: r["metrics"][k] for s, r in rec.items()})
        for k in next(iter(rec.values()))["metrics"]
    }
    e.run.WRITE(
        OUT / "summary.json",
        dict(
            results=results,
            scope="고정가중치개입·재학습/선택/시험없음",
            test_used=False,
        ),
    )
    print({k: v["macro"] for k, v in results.items()}, flush=True)


def campaign() -> None:
    """독립 시퀀스 개입 진단을 완료한다."""
    spec = prepare()
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)

    def call(s):
        with (logs / (s.replace("/", "_") + ".log")).open("w") as stream:
            subprocess.run(
                [sys.executable, __file__, "val", "--sequence", s],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for f in concurrent.futures.as_completed(
            [pool.submit(call, s) for s in spec["val"]]
        ):
            f.result()
    finish()


e.OUT = OUT
e.ARMS = {k: (k, 0) for k in MODES}
e.EPOCHS = (8,)
e.prepare = prepare
e.model = model
checked.prepare = prepare

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("val", "finish", "run"))
    p.add_argument("--sequence", default="kitti/09")
    args = p.parse_args()
    e.run.configure()
    if args.action == "val":
        checked.evaluate(args.sequence)
    elif args.action == "run":
        campaign()
    else:
        finish()
