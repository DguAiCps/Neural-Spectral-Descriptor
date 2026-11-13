#!/usr/bin/env python3
"""새 감산 출력과 균등 학습 표본에 맞춘 손실 척도만 교체한다."""
import argparse
import time
from pathlib import Path

import torch

import _pdv2_diffattn_contrast_learn as previous

e = previous.e
PARENT = previous.OUT
OUT = e.ROOT / "results/pdv2_diffattn_contrast_recalibrated_20261104"
CALIBRATION = PARENT / "calibration_audit/summary.json"
ORIGINAL_SCORE = e.prior.score


def prepare() -> dict:
    """학습용 양의 선형 변환을 학습8에서 고정하고 대조 예산을 유지한다."""
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        spec = e.run.READ(path)
        for p, h in spec["sources"].items():
            assert e.run.HASH(e.ROOT / p) == h, p
    else:
        spec = dict(e.run.READ(PARENT / "protocol.json"))
        spec.update(
            created_unix=time.time(),
            rationale="고정감산이학습AP82.144→83.745%개선해도기존척도의균등표본BCE.28144→.36993악화. 같은초기실제배치의eta미분평균-.24773. 새출력·동일표본비율의척도는BCE.233999·eta미분+.04965. 점수분포와표본비율을맞춘손실의대조.",
            calibration="학습8만사용,시퀀스동일가중·각시퀀스정답/오답균등(없는클래스제외)으로초기균등감산-.5의점수에양의로지스틱선형변환적합. ridge.001, Newton잔차1e-7이하검증. 척도고정뒤신경망학습. 검증/시험로척도선택없음.",
            inference="새양의선형변환은학습BCE에서만사용. 검증의원점수·최종판정·승자·모든기준원장은그대로유지. 양의공통변환은순위와AP/F1/R100에영향없는척도이며배포모듈추가없음.",
        )
        spec["sources"] = dict(spec["sources"])
        spec["sources"].update(
            {
                str(p.relative_to(e.ROOT)): e.run.HASH(p)
                for p in (
                    Path(__file__),
                    CALIBRATION,
                    e.ROOT / "scripts/_pdv2_contrast_calibration.py",
                    PARENT / "summary.json",
                    PARENT / "integrated_validation.json",
                )
            }
        )
        spec["loss_fit"] = e.run.READ(CALIBRATION)["fits"]["contrast_balanced"]
        e.run.WRITE(path, spec)
    link = OUT / "train_inputs"
    if not link.exists():
        link.symlink_to("../" + PARENT.name + "/train_inputs", target_is_directory=True)
    e.run.OUT = OUT
    return spec


def score(z: torch.Tensor, plan: dict) -> torch.Tensor:
    """추론 판정은 유지하고 미분하는 학습 손실에서만 척도를 바꾼다."""
    value = ORIGINAL_SCORE(z, plan)
    if torch.is_grad_enabled():
        f = e.run.READ(OUT / "protocol.json")["loss_fit"]
        value = (value - f["mean"][0]) * f["coefficients"][0] / f["scale"][0] + f[
            "intercept"
        ]
    return value


previous.OUT = e.OUT = OUT
previous.prepare = e.prepare = prepare
e.prior.score = score

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "action", choices=("prepare", "preflight", "train", "val", "finish", "run")
    )
    p.add_argument("--arm", default="diff_lr4")
    p.add_argument("--sequence", default="kitti/09")
    p.add_argument("--repeat", action="store_true")
    a = p.parse_args()
    e.configure()
    if a.action == "run":
        e.__file__ = __file__
        e.campaign()
    elif a.action == "train":
        e.train(a.arm, a.repeat)
    elif a.action == "val":
        e.evaluate(a.sequence)
    elif a.action == "finish":
        e.finish()
    elif a.action == "preflight":
        e.preflight()
    else:
        prepare()
