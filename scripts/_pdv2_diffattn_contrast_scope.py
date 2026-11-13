#!/usr/bin/env python3
"""개선된 감산 학습에서 차이 두 층과 나머지 모듈의 갱신을 분리한다."""
import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import torch

import _pdv2_diffattn_contrast_recalibrated as previous

e = previous.e
PARENT = previous.OUT
OUT = e.ROOT / "results/pdv2_diffattn_contrast_scope_20261104"
MODEL = e.model
ARMS = {"convs_only": ("convs_only", 1e-4), "other_only": ("other_only", 1e-4)}


def prepare() -> dict:
    """같은 초기 모델·손실·512회 학습에서 갱신 범위만 고정한다."""
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
            epochs=[8],
            scope="convs_only는convs.*두층만,other_only는두층을동결하고입력투영/입력정규화/BN학습계수/간선부호기/게이트를학습. BN통계모두고정. 동일초기-.5·교정손실·배치·lr1e-4·512회·8epoch. 미사용파라미터의0기울기도기록.",
            rationale="전체사용모듈의개선을DiffAttnConv가중치자체학습으로해석하지않기위한기여분리. 시드1양성,시드2에서V절대대조AP우위까지본뒤고정한탐색.",
            selection="별도주모델선택없음. 같은8epoch full-frozen,convs_only-frozen,other_only-frozen,full-other_only와full-convs_only를최종세지표로보고. 시험안씀.",
        )
        spec["sources"] = dict(spec["sources"])
        spec["sources"].update(
            {
                str(p.relative_to(e.ROOT)): e.run.HASH(p)
                for p in (
                    Path(__file__),
                    PARENT / "summary.json",
                    PARENT / "learning_report.json",
                    e.ROOT
                    / "results/pdv2_diffattn_contrast_seeds_20261104/s2/summary.json",
                )
            }
        )
        e.run.WRITE(target, spec)
    link = OUT / "train_inputs"
    if not link.exists():
        link.symlink_to("../" + PARENT.name + "/train_inputs", target_is_directory=True)
    e.run.OUT = OUT
    return spec


def model(kind: str, checkpoint: Path = None) -> torch.nn.Module:
    """첫 시드 전체 모델과 같은 시작점에서 갱신 범위를 제한한다."""
    net = MODEL("diff")
    for name, param in net.named_parameters():
        conv = name.startswith("convs.")
        param.requires_grad_(conv if kind == "convs_only" else not conv)
    if checkpoint is not None:
        net.load_state_dict(
            torch.load(checkpoint, map_location="cuda", weights_only=False)[
                "model_state_dict"
            ],
            strict=True,
        )
    return net


def finish() -> None:
    """같은 학습 예산의 전체·부분·동결 효과를 분리한다."""
    spec = prepare()
    rows = {
        s: e.run.READ(OUT / "val" / (s.replace("/", "_") + ".json"))
        for s in spec["val"]
    }
    results = {
        k: e.run.base.macro({s: r["metrics"][k] for s, r in rows.items()})
        for k in next(iter(rows.values()))["metrics"]
    }
    original = e.run.READ(PARENT / "summary.json")
    results["full_8"] = original["results"]["diff_lr4_8"]
    contrasts = {}
    for a, b in (
        ("full_8", "transport_-0.5"),
        ("convs_only_8", "transport_-0.5"),
        ("other_only_8", "transport_-0.5"),
        ("full_8", "other_only_8"),
        ("full_8", "convs_only_8"),
    ):
        contrasts[a + "-" + b] = {
            k: 100 * (results[a]["macro"][k] - results[b]["macro"][k])
            for k in e.run.base.METRICS
        }
    e.run.WRITE(
        OUT / "summary.json",
        dict(results=results, contrasts_pp=contrasts, test_used=False),
    )
    print(
        "SCOPE",
        {k: results[k]["macro"] for k in ("full_8", "convs_only_8", "other_only_8")},
        contrasts,
        flush=True,
    )


def campaign() -> None:
    """두 범위의 학습과9시퀀스 평가를 완료한다."""
    spec = prepare()
    # 원 모델의 입력/미분 검사를 통과했으며 이 실행은 requires_grad만 바꾼다.
    e.run.WRITE(
        OUT / "preflight.json",
        dict(
            status="passed",
            scope="same forward, parameter training mask only",
            parent_sha256=e.run.HASH(PARENT / "preflight.json"),
        ),
    )
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

    for a in ARMS:
        call(["train", "--arm", a], "train_" + a)
        print("TRAINED", a, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(call, ["val", "--sequence", s], "val_" + s.replace("/", "_"))
            for s in spec["val"]
        ]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            e.run.WRITE(
                OUT / "status.json",
                dict(phase="validation", done=i, total=len(futures)),
            )
    finish()
    e.run.WRITE(OUT / "status.json", dict(phase="complete", test_used=False))


e.OUT = previous.OUT = previous.previous.OUT = OUT
e.prepare, e.model = prepare, model
e.ARMS, e.KINDS, e.EPOCHS = ARMS, tuple(ARMS), (8,)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("run", "prepare", "train", "val", "finish"))
    p.add_argument("--arm", default="convs_only")
    p.add_argument("--sequence", default="kitti/09")
    a = p.parse_args()
    e.configure()
    if a.action == "run":
        campaign()
    elif a.action == "train":
        e.train(a.arm)
    elif a.action == "val":
        e.evaluate(a.sequence)
    else:
        globals()[a.action]()
