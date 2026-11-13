#!/usr/bin/env python3
"""방향 보정 이웃 공통 성분의 감산을 고정 조건으로 진단한다."""
import argparse
import time
from pathlib import Path

import _pdv2_diffattn_complex_difference as e

PARENT = e.OUT
OUT = e.ROOT / "results/pdv2_diffattn_complex_contrast_20261104"
CASES = [("identity", 0.0, True)] + [
    (f"{name}_{eta:g}", eta, transport)
    for name, transport in (("transport", True), ("plain", False))
    for eta in (-0.0625, -0.125, -0.25, -0.5)
]


def prepare() -> dict:
    """양의 전달 손실을 본 뒤 설계한 감산 대조를 먼저 고정한다."""
    OUT.mkdir(exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = e.run.READ(target)
        for p, h in spec["sources"].items():
            assert e.run.HASH(e.ROOT / p) == h, p
    else:
        spec = dict(e.run.READ(PARENT / "protocol.json"))
        spec.update(
            created_unix=time.time(),
            cases=CASES,
            rationale="양의평균전달에서오답점수가정답보다더오르는경향. 이웃과공통인성분을빼는음의세기로중심차이를강조하는후속탐색. 신경망/학습/시험없음.",
            representation="단위복소계수의방향보정차이2층. 세기0/-.0625/-.125/-.25/-.5. 무보정동일세기대조포함.",
        )
        spec["sources"] = dict(spec["sources"])
        spec["sources"].update(
            {
                str(p.relative_to(e.ROOT)): e.run.HASH(p)
                for p in (
                    Path(__file__),
                    PARENT / "summary.json",
                    PARENT / "integrated_validation.json",
                )
            }
        )
        e.run.WRITE(target, spec)
    e.run.OUT = OUT
    return spec


e.OUT = OUT
e.CASES = CASES
e.prepare = prepare

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("run", "val", "finish", "prepare"))
    p.add_argument("--sequence", default="kitti/09")
    a = p.parse_args()
    e.run.configure()
    if a.action == "run":
        # 원 실행기의 자식 프로세스도 이 대조를 진입점으로 쓴다.
        e.__file__ = __file__
        e.campaign()
    elif a.action == "val":
        e.evaluate(a.sequence)
    elif a.action == "finish":
        e.finish()
    else:
        prepare()
