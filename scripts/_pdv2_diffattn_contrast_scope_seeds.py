#!/usr/bin/env python3
"""DiffAttnConv 갱신 범위 분리를 고정 설정의 다른 두 시드에서 확인한다."""
import argparse
import functools
import os
import time
from pathlib import Path

import torch

import _pdv2_diffattn_contrast_scope as scope
from gnn.contrast_spectral_difference import ContrastSpectralDifference

e = scope.e
SOURCE = scope.OUT
SEED = int(os.environ.get("NSD_CONTRAST_SEED", "2"))
assert SEED in (2, 3)
PARENT = e.ROOT / f"results/pdv2_diffattn_contrast_seeds_20261104/s{SEED}"
OUT = SOURCE / f"s{SEED}"
RAW_COEFFICIENTS = e.prior.coefficients
CACHE = [None, None]


def coefficients(z: torch.Tensor) -> torch.Tensor:
    """이미 동등성을 확인한 추론 변환만 캐시한다."""
    if torch.is_grad_enabled():
        return RAW_COEFFICIENTS(z)
    if CACHE[0] is not z:
        CACHE[:] = [z, RAW_COEFFICIENTS(z)]
    return CACHE[1]


def prepare() -> dict:
    """같은 갱신 범위와512회 예산을 시드별로 고정한다."""
    OUT.mkdir(exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = e.run.READ(target)
        for p, h in spec["sources"].items():
            assert e.run.HASH(e.ROOT / p) == h, p
    else:
        spec = dict(e.run.READ(SOURCE / "protocol.json"))
        spec.update(
            created_unix=time.time(),
            seed=SEED,
            rationale="첫시드convs_only고정감산대비AP+.163%p,R100+.829%p를본뒤같은설정을다른사전학습시드2/3에고정적용. 표본순서같음,재선택없음.",
        )
        spec["sources"] = dict(spec["sources"])
        paths = [
            Path(__file__),
            SOURCE / "summary.json",
            PARENT / "summary.json",
            PARENT / "preflight.json",
            e.ROOT
            / f"results/pdv2_spectral_keys_20261025/s{SEED}/octave288/checkpoints/checkpoint_epoch_20.pth",
        ]
        spec["sources"].update(
            {str(p.relative_to(e.ROOT)): e.run.HASH(p) for p in paths}
        )
        e.run.WRITE(target, spec)
    link = OUT / "train_inputs"
    if not link.exists():
        link.symlink_to("../train_inputs", target_is_directory=True)
    e.run.OUT = OUT
    return spec


def model(kind: str, checkpoint: Path = None) -> torch.nn.Module:
    """시드별 기존 모델에서 같은 두 가지 갱신 범위를 적용한다."""
    torch.manual_seed(SEED)
    ref = e.run.source.factory(288)
    path = (
        e.ROOT
        / f"results/pdv2_spectral_keys_20261025/s{SEED}/octave288/checkpoints/checkpoint_epoch_20.pth"
    )
    ref.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"],
        strict=True,
    )
    net = ContrastSpectralDifference(ref, "diff").cuda().eval()
    for name, p in net.named_parameters():
        conv = name.startswith("convs.")
        p.requires_grad_(conv if kind == "convs_only" else not conv)
    if checkpoint is not None:
        net.load_state_dict(
            torch.load(checkpoint, map_location="cuda", weights_only=False)[
                "model_state_dict"
            ],
            strict=True,
        )
    return net


scope.OUT = e.OUT = scope.previous.OUT = scope.previous.previous.OUT = OUT
scope.PARENT = PARENT
scope.prepare = e.prepare = prepare
scope.model = e.model = model
e.run.inputs = functools.lru_cache(maxsize=1)(e.run.inputs)
e.prior.coefficients = coefficients

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("run", "prepare", "train", "val", "finish"))
    p.add_argument("--arm", default="convs_only")
    p.add_argument("--sequence", default="kitti/09")
    a = p.parse_args()
    e.configure()
    if a.action == "run":
        scope.__file__ = __file__
        scope.campaign()
    elif a.action == "train":
        e.train(a.arm)
    elif a.action == "val":
        e.evaluate(a.sequence)
    elif a.action == "finish":
        scope.finish()
    else:
        prepare()
