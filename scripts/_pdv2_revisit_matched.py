#!/usr/bin/env python3
"""학습 초기 기울기 강도를 맞춘 문맥 재방문 제약을 검증한다."""
import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_real_revisit as engine
from _pdv2_no_context_ablation import sha256, write_json

PARENT = engine.OUT
OUT = REPO / "results/pdv2_revisit_matched_20261025"
CFG = copy.deepcopy(engine.CFG)
CFG["coefficient_clip"] = [0.0, 100.0]
CFG["new_training_arms"] = ["pair_context"]
CFG["reused_arms"] = ["clean_rank", "pair_key"]
engine.OUT, engine.CFG = OUT, CFG
base, factor, prior = engine.base, engine.factor, engine.prior
METHOD, SUPPORT = engine.METHOD, engine.SUPPORT
protocol, model, initial_model = engine.protocol, engine.model, engine.initial_model
pool_path, load_pool, terms_for = engine.pool_path, engine.load_pool, engine.terms_for
auxiliary_batches, local_auxiliary, pair_terms = (
    engine.auxiliary_batches,
    engine.local_auxiliary,
    engine.pair_terms,
)


def prepare() -> None:
    """이미 완료한 대조는 재사용하고 문맥 계수의 하한만 제거한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    parent = base.read(PARENT / "protocol.json")
    audit = base.read(PARENT / "integrated_validation.json")
    assert audit["passed"] and not audit["test_used"]
    for path, digest in audit["sources"].items():
        assert sha256(REPO / path) == digest, path
    for name in (
        "clean_rank",
        "pair_key",
        "auxiliary_pool.npz",
        "auxiliary_graph.pt",
        "pool_audit.json",
        "calibration_gradients.npz",
    ):
        dest = OUT / name
        assert not dest.exists()
        dest.symlink_to(
            os.path.relpath(PARENT / name, OUT),
            target_is_directory=name in CFG["reused_arms"],
        )
    calibration = base.read(PARENT / "calibration.json")
    gradients = base.exp.arrays(PARENT / "calibration_gradients.npz")
    coefficients = dict(parent["coefficients"])
    coefficients["pair_context"] = float(
        0.25
        * np.linalg.norm(gradients["rank"])
        / np.linalg.norm(gradients["pair_context"])
    )
    assert abs(coefficients["pair_context"] - 0.005132463032957791) < 1e-14
    write_json(
        OUT / "calibration.json",
        {
            **calibration,
            "coefficients": coefficients,
            "source_calibration": str((PARENT / "calibration.json").relative_to(REPO)),
            "rule": "문맥 계수의 .01 하한만 제거. 동일 학습 초기 기울기의 크기 비. 검증 계수 탐색 없음.",
        },
    )
    runtime, inputs = dict(parent["runtime"]), dict(parent["inputs"])
    for name in (
        "_pdv2_revisit_matched.py",
        "_pdv2_revisit_matched_run.py",
        "_pdv2_revisit_matched_finish.py",
        "_pdv2_revisit_matched_report.py",
        "_pdv2_revisit_matched_audit.py",
    ):
        runtime["scripts/" + name] = sha256(REPO / "scripts" / name)
    for path in (
        PARENT / "integrated_validation.json",
        PARENT / "summary.json",
        PARENT / "selection.json",
        OUT / "calibration.json",
    ):
        inputs[str(path.relative_to(REPO))] = sha256(path)
    write_json(
        OUT / "protocol.json",
        {
            **parent,
            "runtime": runtime,
            "inputs": inputs,
            "config": CFG,
            "coefficients": coefficients,
            "reused_arms": CFG["reused_arms"],
            "new_training_updates": 252,
            "scope": "문맥 초기 보조 기울기 비를 .48709557에서 .25로 맞춘 한 조건의 추가 대조. "
            "정상과 전체키 대조는 체크포인트·표본·코드 해시 확인 후 그대로 재사용. 새 업데이트252회. "
            "시험·원고·추론 모듈 변경 없음. 추가2/4에포크 중 기존 최종 검증AP 규칙 선택.",
        },
    )
    protocol(True)
    print("MATCHED_PREPARED", coefficients, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "evaluate", "select"))
    parser.add_argument("--arm", choices=CFG["arms"])
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--role", choices=("train", "val"))
    parser.add_argument("--sequence")
    args = parser.parse_args()
    torch.set_num_threads(4 if args.action in ("prepare", "train") else 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == "prepare":
        prepare()
    elif args.action == "train":
        assert args.arm == "pair_context"
        engine.train(args.arm)
    else:
        engine.bind_evaluator()
        if args.action == "select":
            engine.evaluator.select()
        else:
            engine.evaluator.evaluate(args.arm, args.epoch, args.role, args.sequence)
