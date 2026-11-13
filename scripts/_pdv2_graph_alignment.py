#!/usr/bin/env python3
"""유사 간선의 학습·추론 정합 실험을 고정하고 최종 판정을 집계한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_pdv2_graph_alignment_run.py
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_attention_neighbors as attention
from _pdv2_matching_mine import cache_path
from _pdv2_no_context_ablation import sha256, write_json

CONFIG = REPO / "configs/pdv2_graph_alignment.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
OUT = REPO / "results/pdv2_graph_alignment_20261015"
CONTROL = REPO / CFG["control_root"]
SAMPLES = REPO / CFG["sample_source"]
ARMS = tuple(CFG["arms"])
EPOCHS = tuple(CFG["checkpoint_epochs"])
NEW = "diff_online"
base = attention.base


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def arm_root(arm: str) -> Path:
    """기존 대조군은 원래 결과를 읽기 전용으로 참조한다."""
    assert arm in ARMS
    return (OUT if arm == NEW else CONTROL) / "s1" / arm


def checkpoint_dir(seed: int, arm: str) -> Path:
    """주모델과 기존 대조군 체크포인트를 덮어쓰지 않는다."""
    assert seed == 1 and arm in ARMS
    tag = "graph_alignment" if arm == NEW else "attention"
    return REPO / f"checkpoints/pdv2_{tag}_s1_{arm}"


def folder(seed: int, arm: str, epoch: int) -> Path:
    """조건별 평가 경로를 반환한다."""
    assert seed == 1 and epoch in EPOCHS
    p = arm_root(arm) / f"epoch{epoch}"
    if arm == NEW:
        p.mkdir(parents=True, exist_ok=True)
    return p


def training_config(arm: str) -> Path:
    """새 조건의 고정 설정 경로다."""
    assert arm == NEW
    return OUT / "configs/diff_online.yaml"


def layout_path(sensor: str, seq: str) -> Path:
    """기존 학습 키프레임과 정렬된 원통 특징 경로다."""
    p = REPO / f"data/preprocessed_v2/layouts_{sensor}_{seq}_layout60.npz"
    return p if p.exists() else REPO / f"data/preprocessed_cross_sensor_operating/{sensor}_operating_{seq}_layout60_stride1.npz"


def protocol(data: bool = False) -> dict:
    """고정된 실행 코드와 입력을 검사한다."""
    p = read(OUT / "protocol.json")
    for name, digest in p["runtime_sha256"].items():
        assert sha256(REPO / name) == digest, name
    if data:
        for name, digest in p["input_sha256"].items():
            assert sha256(REPO / name) == digest, name
    return p


def prepare() -> None:
    """새 학습 전에 예산·대조군·선택 규칙을 고정한다."""
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    old = read(CONTROL / "protocol.json")
    for name, digest in {**old["runtime_sha256"], **old["input_sha256"]}.items():
        assert sha256(REPO / name) == digest, name
    cfg = yaml.safe_load((REPO / CFG["base_training_config"]).read_text())
    assert cfg["data"]["datasets"]["test"] == []
    assert cfg["gnn"]["diffattn_value_source"] == "diff"
    path = training_config(NEW)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    runtime = {REPO / p for p in old["runtime_sha256"]}
    runtime.update(REPO / p for p in (
        "configs/pdv2_graph_alignment.yaml", "scripts/_pdv2_graph_alignment.py",
        "scripts/_train_graph_alignment.py", "scripts/_pdv2_graph_alignment_run.py",
        "scripts/_pdv2_graph_alignment_verify.py", "scripts/_pd_build_edge_classifier.py",
        "scripts/evaluate_kitti_checkpoint.py", "src/encoding/phase_alignment.py",
        "tests/gnn/test_graph_alignment.py"))
    runtime.add(path)
    inputs = {REPO / p for p in old["input_sha256"]}
    inputs.update((CONTROL / "protocol.json", CONTROL / "completion_validation.json",
                   REPO / "checkpoints/pdv2_lag15_s1/best_model.pth"))
    # 실제 입력 설정 경로는 기준 로드 함수의 상수에서 가져온다.
    from _pdv2_lag15_campaign import CONFIG as inference_config
    inputs.add(inference_config)
    for sensor, seq in base.TRAIN:
        inputs.update((cache_path(sensor + "/" + seq), layout_path(sensor, seq)))
    for arm in ("diff_full", "diff_temporal"):
        inputs.update(p for p in arm_root(arm).rglob("*") if p.is_file())
        inputs.update(checkpoint_dir(1, arm) / f"checkpoint_epoch_{e}.pth" for e in EPOCHS)
    value = {
        "created_unix": time.time(), "config": CFG, "seed": 1, "arms": ARMS,
        "train_roster": old["train_roster"], "val_roster": old["val_roster"], "test_used": False,
        "control_reuse": "완료한 attention 실험의 diff_full/diff_temporal 원장·가중치를 해시로 고정하고 읽기 전용 재사용한다.",
        "fixed": "같은 초기43텐서·167561파라미터·삼중쌍·InfoNCE·20에포크900업데이트·Adam·AMP·BN/dropout 학습 규칙·시간 간선·기준 자세·키프레임. 손실에 들어가는 원래 키의 재계량 설정도 유지한다.",
        "intervention": "diff_online은 매 optimizer 업데이트 직전에 현재 가중치의 eval/FP32 첫 통과와 현재 온라인 분류기로 유사 간선을 새로 만든다. 첫 통과 .993, j<=i-6, 최대10; 둘째 f0 코사인 후보30, j<=i-30, 분류기 상위10과 동일5D속성. 원래 평가 함수를 재사용한다. 후보·간선 속성은 미분하지 않고 둘째 통과를 원래 손실로 학습한다.",
        "probe": "첫 통과의 키 블록에만 고정 RVEC를 적용해 온라인 f0와 맞춘다. f0는 두 블록을 이어 붙인 원래 출력이며, 후보 코사인 계산은 정규화하지만 간선 L2는 원래 f0에서 계산한다. 프로브는 BN통계·dropout·난수를 변경하지 않는다. 간선 분류기 재학습 없음.",
        "remaining_differences": "유사 간선 불일치를 분리한다. 모든 조건 학습의 기준 자세/키프레임과 검증의 KISS-ICP 차이는 유지한다. 학습 둘째 통과의 BN/dropout/AMP와 추론 eval/FP32 차이도 유지한다. 전체 학습·배포 완전 동일성 실험은 아니다.",
        "evaluation": "동일 검증9, q+15, j<=q-30, 재계량·분류기·100+100·정렬·융합·수락 점수/임계값 고정. 10/20에포크 중 시퀀스→3계열→NSD/H 평균 최종AP 최대,1e-12동률이면 이른 시점. 같은 에포크 비교도 병기한다.",
        "interpretation": "시드1 검증 진단이다. 정합 조건이 현재 조건보다 높으면 불일치 영향 근거, 시간 이웃 조건까지 높으면 유사 간선 기여 근거다. 단일시드로 유의성을 주장하지 않고 시험/주표 교체도 하지 않는다.",
        "runtime_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(runtime)},
        "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(inputs)},
    }
    write_json(OUT / "protocol.json", value)
    print("PREPARED", len(runtime), len(inputs), flush=True)


def report() -> None:
    """최종 판정 지표의 선택 결과와 같은 에포크 차이를 저장한다."""
    protocol()
    base.select(1)
    selected = read(OUT / "s1/validation_summary.json")["arms"]
    effects = {}
    for label, epoch in (("selected", None), ("epoch10", 10), ("epoch20", 20)):
        cells = {a: selected[a]["ops"] if epoch is None else base.aggregate(1, a, epoch, "val") for a in ARMS}
        effects[label] = {op: {metric: {
            "online_minus_current": cells[NEW][op]["macro"][metric] - cells["diff_full"][op]["macro"][metric],
            "online_minus_temporal": cells[NEW][op]["macro"][metric] - cells["diff_temporal"][op]["macro"][metric],
        } for metric in base.PRIMARY} for op in base.OPS}
    write_json(OUT / "summary.json", {"arms": selected, "effects": effects, "test_used": False,
               "scope": "시드1 검증9의 최종 판정. 기존 대조군 재사용.",
               "protocol_sha256": sha256(OUT / "protocol.json")})
    with (OUT / "comparison.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["arm", "epoch", "op", "AP", "max_F1", "R@100P", "R@1"])
        for arm, row in selected.items():
            for op, data in row["ops"].items():
                w.writerow([arm, row["epoch"], op, *[data["macro"][k] for k in ("AP", "max_F1", "R@100P", "R@1")]])
    print("SUMMARY", {a: {o: r["ops"][o]["macro"] for o in base.OPS} for a, r in selected.items()}, flush=True)


# 완료한 공통 학습·평가 도구를 이 프로세스 안에서만 새 실험 경로에 연결한다.
attention.OUT, attention.ARMS, attention.EPOCHS = OUT, ARMS, EPOCHS
attention.protocol, attention.checkpoint_dir, attention.training_config = protocol, checkpoint_dir, training_config
base.OUT, base.ARMS, base.EPOCHS = OUT, ARMS, EPOCHS
base.protocol, base.checkpoint_dir, base.folder = protocol, checkpoint_dir, folder

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "keys", "score", "report"))
    p.add_argument("--epoch", type=int, choices=EPOCHS)
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "prepare":
        prepare()
    elif a.action == "keys":
        attention.keys(NEW, a.epoch, a.sequence)
    elif a.action == "score":
        base.score(1, NEW, a.epoch, "val", a.sequence)
    else:
        report()
