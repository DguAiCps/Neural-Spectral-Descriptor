#!/usr/bin/env python3
"""첫 시드에서 고정한 차이 전달 학습을 다른 초기 학습 시드에 적용한다."""
import argparse
import functools
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import _pdv2_diffattn_contrast_recalibrated as previous
from gnn.contrast_spectral_difference import (
    ContrastSpectralDifference,
    GlobalContrastDifference,
)

e = previous.e
SOURCE = previous.OUT
CAMPAIGN = e.ROOT / "results/pdv2_diffattn_contrast_seeds_20261104"
SEED = int(os.environ.get("NSD_CONTRAST_SEED", "2"))
assert SEED in (2, 3)
OUT = CAMPAIGN / f"s{SEED}"
ARMS = {f"{k}_lr4": (k, 1e-4) for k in e.KINDS}
ARMS.update({f"{k}_lr5": (k, 1e-5) for k in ("point_gate", "gate_only")})
RAW_INPUTS = e.run.inputs
RAW_COEFFICIENTS = e.prior.coefficients
COEFFICIENT_CACHE = [None, None]


def coefficients(z: torch.Tensor) -> torch.Tensor:
    """추론에서 같은 입력의 고정 변환만 재사용하며 학습 미분은 보존한다."""
    if torch.is_grad_enabled():
        return RAW_COEFFICIENTS(z)
    if COEFFICIENT_CACHE[0] is not z:
        COEFFICIENT_CACHE[:] = [z, RAW_COEFFICIENTS(z)]
    return COEFFICIENT_CACHE[1]


def freeze() -> dict:
    """다른 시드를 보기 전에 첫 시드 선택과 확인 기준을 고정한다."""
    CAMPAIGN.mkdir(exist_ok=True)
    path = CAMPAIGN / "freeze.json"
    if path.exists():
        spec = e.run.READ(path)
        for p, h in spec["sources"].items():
            assert e.run.HASH(e.ROOT / p) == h, p
        return spec
    source = e.run.READ(SOURCE / "summary.json")
    sources = [
        Path(__file__),
        SOURCE / "protocol.json",
        SOURCE / "summary.json",
        SOURCE / "learning_report.json",
        SOURCE / "integrated_validation.json",
    ]
    for seed in (2, 3):
        sources += [
            e.ROOT
            / f"results/pdv2_spectral_keys_20261025/s{seed}/octave288/checkpoints/checkpoint_epoch_20.pth",
            e.ROOT
            / f"results/pdv2_table4_current_20261030/s{seed}/graph_free/checkpoints/checkpoint_epoch_20.pth",
        ]
    spec = dict(
        created_unix=time.time(),
        seeds=[1, 2, 3],
        selected_tags=source["best"],
        fixed_difference_tags={k: f"{k}_lr4_8" for k in e.KINDS},
        interpretation="첫시드의설정선택을고정하고시드2/3에서다시고르지않는다. 원래900회사전학습시드가다른시작점이며추가학습의표본순서는시드간에도동일하게유지해시작모델차이를확인한다. 세추가학습표본시드전체를독립적으로바꾼실험은아니다.",
        main_criteria="3시드평균Diff최종AP가고정감산과각학습대조(abs_k제외)보다+.1%p,F1손실<=.3%p,R100손실<=1%p,Diff학습이득AP양수최소2/3시드. K차이는독립대조로보고우위필수조건에넣지않음. 원첫시드모든대조초과조건은미통과였음을유지.",
        value_criteria="같은lr1e-4/epoch8에서abs_k-abs_kv평균AP양수·R100양수이고최소2/3시드AP양수인지별도확인.",
        controls="point/gate는첫시드선택lr1e-5/epoch4와동일예산lr1e-4/epoch8모두보고. global은같은초기0과같은배치라시드별동일가중치일것이며검산. 모든구조512회학습기회를유지.",
        scope="검증9만사용. 원키검색승자R1한계가남아목표완료또는최종BEV비교가능성확정이아님. 설정은노출된시험에서선택하지않음. 원고/그림유지.",
        optimization="입력읽기1시퀀스캐시,추론에서같은복소텐서의고정변환캐시. 값·수식·미분불변이며첫시드선택모델의원장과별도로재현검사.",
        sources={str(p.relative_to(e.ROOT)): e.run.HASH(p) for p in sources},
    )
    e.run.WRITE(path, spec)
    link = CAMPAIGN / "s1"
    if not link.exists():
        link.symlink_to("../" + SOURCE.name, target_is_directory=True)
    return spec


def prepare() -> dict:
    """시드별 모델의 출처와 같은 학습 조건을 동결한다."""
    frozen = freeze()
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        spec = e.run.READ(path)
        for p, h in spec["sources"].items():
            assert e.run.HASH(e.ROOT / p) == h, p
    else:
        spec = dict(e.run.READ(SOURCE / "protocol.json"))
        spec.update(
            created_unix=time.time(),
            seed=SEED,
            arms=ARMS,
            selected_tags=frozen["selected_tags"],
            checkpoint_selection="첫 시드의 구조별 설정을 고정하며 새 시드에서 재선택하지 않는다.",
        )
        spec["sources"] = dict(spec["sources"])
        spec["sources"].update(frozen["sources"])
        spec["sources"][str((CAMPAIGN / "freeze.json").relative_to(e.ROOT))] = (
            e.run.HASH(CAMPAIGN / "freeze.json")
        )
        e.run.WRITE(path, spec)
    link = OUT / "train_inputs"
    if not link.exists():
        link.symlink_to(
            "../../" + SOURCE.name + "/train_inputs", target_is_directory=True
        )
    e.run.OUT = OUT
    return spec


def model(kind: str, checkpoint: Path = None) -> torch.nn.Module:
    """시드별 사전학습 가중치를 쓰되 최종 초기 감산은 모두 -.5로 맞춘다."""
    torch.manual_seed(SEED)
    if kind == "global":
        net = GlobalContrastDifference()
    else:
        if kind == "point_gate":
            ref = e.prior.base.table4.model("graph_free", SEED)
            path = (
                e.ROOT
                / f"results/pdv2_table4_current_20261030/s{SEED}/graph_free/checkpoints/checkpoint_epoch_20.pth"
            )
        else:
            ref = e.run.source.factory(288)
            path = (
                e.ROOT
                / f"results/pdv2_spectral_keys_20261025/s{SEED}/octave288/checkpoints/checkpoint_epoch_20.pth"
            )
        ref.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)[
                "model_state_dict"
            ],
            strict=True,
        )
        net = ContrastSpectralDifference(ref, kind)
    net = net.cuda().eval()
    if checkpoint is not None:
        net.load_state_dict(
            torch.load(checkpoint, map_location="cuda", weights_only=False)[
                "model_state_dict"
            ],
            strict=True,
        )
    return net


def preflight() -> None:
    """실제 입력·초기점수·기울기와 추론 캐시의 항등성을 확인한다."""
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
    with torch.no_grad():
        direct = RAW_COEFFICIENTS(expected)
        torch.testing.assert_close(coefficients(expected), direct, atol=0, rtol=0)
        assert coefficients(expected) is coefficients(expected)
    for kind in e.KINDS:
        net = model(kind)
        z = net(graph)
        error = float((z - expected).detach().abs().max())
        assert error < 4e-6
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
        checks[kind] = dict(initial_error=error, gradients=gradients)
        del net, z, value, loss
    e.run.WRITE(
        OUT / "preflight.json",
        dict(status="passed", checks=checks, cache_identical=True),
    )
    print("PREFLIGHT", SEED, flush=True)


def finish() -> None:
    """다른 시드의 결과를 첫 시드 선택으로만 요약한다."""
    spec = prepare()
    rows = {
        s: e.run.READ(OUT / "val" / (s.replace("/", "_") + ".json"))
        for s in spec["val"]
    }
    results = {
        k: e.run.base.macro({s: r["metrics"][k] for s, r in rows.items()})
        for k in next(iter(rows.values()))["metrics"]
    }
    e.run.WRITE(
        OUT / "summary.json",
        dict(
            results=results,
            best=spec["selected_tags"],
            selection_fixed_by_seed1=True,
            seed=SEED,
            test_used=False,
        ),
    )
    print(
        "FINISHED_SEED",
        SEED,
        {k: results[v]["macro"] for k, v in spec["selected_tags"].items()},
        flush=True,
    )


e.OUT = previous.OUT = previous.previous.OUT = OUT
e.ARMS = ARMS
e.prepare, e.model, e.preflight, e.finish = prepare, model, preflight, finish
e.run.inputs = functools.lru_cache(maxsize=1)(RAW_INPUTS)
e.prior.coefficients = coefficients

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
        finish()
    elif a.action == "preflight":
        preflight()
    else:
        prepare()
