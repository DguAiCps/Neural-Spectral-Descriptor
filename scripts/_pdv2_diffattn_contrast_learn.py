#!/usr/bin/env python3
"""복소 이웃 공통 성분 제거에서 DiffAttnConv의 학습 기여를 분리한다."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import _pdv2_diffattn_complex_learn as e
from gnn.contrast_spectral_difference import (
    ContrastSpectralDifference,
    GlobalContrastDifference,
)

PARENT = e.OUT
OUT = e.ROOT / "results/pdv2_diffattn_contrast_learn_20261104"
FIXED = e.ROOT / "results/pdv2_diffattn_complex_contrast_range_20261104"
KINDS = (*e.KINDS, "global")
ARMS = {
    f"{kind}_{tag}": (kind, lr)
    for kind in KINDS
    for tag, lr in (("lr5", 1e-5), ("lr4", 1e-4))
}
ORIGINAL_MODEL = e.model


def prepare() -> dict:
    """음의 공통 성분 제거와 같은 출발점·예산 대조를 고정한다."""
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
            arms=ARMS,
            architecture="기존두층attention과게이트, 복소차이전달구조유지. 세기=-sigmoid로-1~0. 초기-.5. global은두층별공통세기2개만학습. 새신경망층없음.",
            rationale="양의전달최적학습도전달없음보다낮음. 음의균등전달-.5가AP/F1/R100동시개선. 세기-1/-2/-4확장격자에서-.5가최고AP이고강한제거는MulRan손실. 이결과를본후초기-.5/범위-1~0을고정한탐색.",
            initialization="앞양의전달과같은pretrained가중치·Q0·기하최종0·게이트최종0. 모든조건초기균등-.5. global도-.5. 사전학습의Diff유리가능성을공시.",
            selection="Diff최적AP가모든학습대조최적및보정/무보정고정감산격자각조건보다+.1%p,F1손실<=.3%p,R100<=1%p면매칭후보. 신경망학습효과와고정시간차이효과를분리. 원키R1은BEV보다3.065%p낮아검색복구·3시드·고정시험전까지목표완료주장없음.",
        )
        spec["sources"] = dict(spec["sources"])
        spec["sources"].update(
            {
                str(p.relative_to(e.ROOT)): e.run.HASH(p)
                for p in (
                    Path(__file__),
                    e.ROOT / "src/gnn/contrast_spectral_difference.py",
                    PARENT / "summary.json",
                    PARENT / "integrated_validation.json",
                    FIXED / "summary.json",
                    FIXED / "protocol.json",
                )
            }
        )
        e.run.WRITE(target, spec)
    path = OUT / "train_inputs"
    if not path.exists():
        path.symlink_to("../" + PARENT.name + "/train_inputs", target_is_directory=True)
    e.run.OUT = OUT
    return spec


def model(kind: str, checkpoint: Path = None) -> torch.nn.Module:
    """동일 기존 가중치의 게이트 읽기 부호와 범위만 바꾼다."""
    if kind == "global":
        net = GlobalContrastDifference().cuda().eval()
    else:
        net = ORIGINAL_MODEL(kind)
        # 보존된 모듈·가중치·학습가능 범위를 그대로 쓰는 forward 대체다.
        net.__class__ = ContrastSpectralDifference
    if checkpoint is not None:
        net.load_state_dict(
            torch.load(checkpoint, map_location="cuda", weights_only=False)[
                "model_state_dict"
            ],
            strict=True,
        )
    return net


def preflight() -> None:
    """초기 음의 전달 일치·미분·같은 입력의 결정성을 확인한다."""
    spec = prepare()
    name = spec["train"][0].replace("/", "_")
    graph = torch.load(
        OUT / "train_inputs" / (name + ".pt"), map_location="cuda", weights_only=False
    )
    data = e.run.ARR(e.prior.PARENT / "pools/train" / name / "pool.npz")
    pp = {
        k: torch.from_numpy(data["plan_" + k][:32]).cuda()
        for k in ("query", "candidate", "weight")
    }
    expected = e.prior.filtered(graph, -0.5)
    checks = {}
    for kind in KINDS:
        net = model(kind)
        z = net(graph)
        error = float((z - expected).detach().abs().max())
        assert error < 4e-6
        repeat = net(graph)
        assert torch.equal(z, repeat)
        value = e.prior.score(z, pp)
        loss = F.binary_cross_entropy_with_logits(
            value, torch.from_numpy(data["correct"][:32]).cuda().float()
        )
        loss.backward()
        gradients = {
            n: float(p.grad.norm())
            for n, p in net.named_parameters()
            if p.grad is not None
        }
        assert sum(gradients.values()) > 0 and all(
            np.isfinite(v) for v in gradients.values()
        )
        if kind in ("diff", "abs_k", "abs_kv"):
            assert sum(v for n, v in gradients.items() if "W_q" in n) > 0
        checks[kind] = dict(
            initial_error=error,
            gradients=gradients,
            trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),
        )
        del net, z, repeat, value, loss
    e.run.WRITE(
        OUT / "preflight.json", dict(status="passed", checks=checks, deterministic=True)
    )
    print("PREFLIGHT PASSED", flush=True)


e.OUT, e.KINDS, e.ARMS = OUT, KINDS, ARMS
e.prepare, e.model, e.preflight = prepare, model, preflight
e.prior.OUT = FIXED
e.prior.CASES = [("identity", 0.0, True)] + [
    (f"{name}_{eta:g}", eta, transport)
    for name, transport in (("transport", True), ("plain", False))
    for eta in (-0.5, -1.0, -2.0, -4.0)
]

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
    else:
        globals()[a.action]()
