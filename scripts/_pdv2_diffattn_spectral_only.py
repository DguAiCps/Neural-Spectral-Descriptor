#!/usr/bin/env python3
"""가산 문맥 점수 없이 DiffAttnConv를 스펙트럼 조절에 학습한다."""
import argparse
import time
from pathlib import Path

import _pdv2_diffattn_spectral_gain as base

ORIGIN = base.OUT
OUT = base.ROOT / "results/pdv2_diffattn_spectral_only_20261104"


def prepare() -> dict:
    """두 역할의 간섭을 분리하는 후속 대조를 평가 전에 고정한다."""
    OUT.mkdir(exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = base.run.READ(target)
        for p, h in spec["sources"].items():
            assert base.run.HASH(base.ROOT / p) == h, p
    else:
        spec = dict(base.run.READ(ORIGIN / "protocol.json"))
        spec.update(
            created_unix=time.time(),
            method="기존128차원게이트문맥u를g=exp(s*tanh(sqrt(128)*u))의16x8양의이득으로사용. 가산문맥항완전제거. 기존검색키는유지·학습후재검색. 위상보존·원본/보정제곱근계수·두채널·75/25·60각도·시간판정·q+15동일.",
            train_scope="초기추가학습전동일모델. DiffAttnConv두층14텐서만학습. 스펙트럼최종점수만학습8기존척도로BCE. 문맥내적을최종수락에직접더하지않음. 동일512갱신·양음균등64표본·두학습률. s0은학습그래디언트0이므로학습중s=.5고정.",
            motivation="앞대조의2에포크AP이득과R100손실관찰후설계. 초기학습8미분분해에서가산문맥경로가스펙트럼경로보다큼. 전체결과가나오기전역할분리대조를고정. 사전등록이나독립검증으로표현하지않음.",
        )
        spec["sources"] = dict(spec["sources"])
        spec["sources"][str(Path(__file__).relative_to(base.ROOT))] = base.run.HASH(
            Path(__file__)
        )
        spec["sources"][
            str((ORIGIN / "gradient_diagnostic.json").relative_to(base.ROOT))
        ] = base.run.HASH(ORIGIN / "gradient_diagnostic.json")
        base.run.WRITE(target, spec)
    base.run.OUT = OUT
    return spec


def total_score(key, z, plan, strength):
    """가산 문맥을 제외한 스펙트럼 판정과 미분 경로를 반환한다."""
    fit = base.run.READ(base.PARENT / "calibration_recenter.json")["spectral"]
    score = base.gain_score(key, z, plan, strength)
    slope = fit["coefficients"][0] / fit["scale"][0]
    return slope * (score - fit["mean"][0]) + fit["intercept"], score


base.OUT = OUT
base.prepare = prepare
base.total_score = total_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "preflight", "train", "val", "finish")
    )
    parser.add_argument("--arm", default="conv_lr4")
    parser.add_argument("--epoch", type=int, default=4)
    parser.add_argument("--sequence", default="kitti360/0004")
    args = parser.parse_args()
    base.run.configure()
    if args.action == "train":
        base.train(args.arm)
    elif args.action == "val":
        base.validate(args.arm, args.epoch, args.sequence)
    else:
        getattr(base, args.action)()
