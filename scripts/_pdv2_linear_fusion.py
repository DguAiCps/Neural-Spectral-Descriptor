#!/usr/bin/env python3
"""동결된 후보 학습 실행기를 재사용해 네 원시 점수 가중합을 학습한다."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_candidate_learning as base
from _pdv2_no_context_ablation import sha256, write_json
from retrieval.linear_fusion import FEATURE_NAMES, LinearFusionScorer, fusion_features

PREVIOUS = REPO / "results/pdv2_candidate_learning_20261014"
OUT = REPO / "results/pdv2_linear_fusion_20261015"
CONFIG = REPO / "configs/pdv2_linear_fusion.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
NORMALIZATION = OUT / "normalization.json"
ORIGINAL_SNAPSHOT = base.snapshot
ORIGINAL_BUILD_POOL = base.build_pool
ACTIVE_REFERENCE = None


class Scorer(LinearFusionScorer):
    """공통 학습셋 정규화를 사용하고 기존 실행기 생성 규약을 따른다."""

    def __init__(self, hidden: tuple = ()) -> None:
        assert not hidden, "숨은 층을 추가하지 않는다"
        stats = base.read(NORMALIZATION)
        super().__init__(stats["mean"], stats["std"])


def fit_normalization() -> None:
    """학습의 두 초기 표현·두 운영점 후보만으로 공통 고정 통계를 구한다."""
    if NORMALIZATION.exists():
        return
    total, sums, squares = 0, np.zeros(4), np.zeros(4)
    sources = {}
    for kind in ("gnn", "point"):
        for sensor, seq in base.TRAIN:
            folder = PREVIOUS / "pools" / f"{kind}_initial/train/{sensor}_{seq}"
            with np.load(folder / "keys.npz") as z:
                embedding = torch.from_numpy(z["embedding"]).cuda()
            for op in base.OPS:
                with np.load(folder / (op + ".npz")) as z:
                    data = {k: z[k] for k in ("queries", "candidates", "static")}
                with torch.no_grad():
                    for start in range(0, len(data["queries"]), 128):
                        sl = slice(start, start + 128)
                        q, c, static = [torch.from_numpy(data[k][sl]).cuda() for k in ("queries", "candidates", "static")]
                        valid = c >= 0
                        features = fusion_features(embedding, q, c, static.float(), valid)[valid].cpu().numpy().astype(np.float64)
                        total += len(features)
                        sums += features.sum(0)
                        squares += np.square(features).sum(0)
            sources[str((folder / "complete.json").relative_to(REPO))] = sha256(folder / "complete.json")
    mean = sums / total
    std = np.sqrt(np.maximum(squares / total - mean ** 2, 0))
    std = np.maximum(std, CFG["normalization_std_floor"])
    write_json(NORMALIZATION, {"scope": "학습8의 두 초기 모델·두 운영점 실제 후보만. 검증·시험 없음.",
               "features": list(FEATURE_NAMES), "n_candidate_pairs": total,
               "mean": mean.tolist(), "std": std.tolist(), "sources": sources})
    print("NORMALIZATION", total, mean.tolist(), std.tolist(), flush=True)


def prepare() -> None:
    """불변 초기 후보를 읽기 전용으로 공유하고 학습 정규화만 계산한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pools").mkdir(exist_ok=True)
    for kind in ("gnn", "point"):
        source = PREVIOUS / "pools" / (kind + "_initial")
        target = OUT / "pools" / (kind + "_initial")
        if not target.exists():
            target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)
        assert target.resolve() == source.resolve()
    fit_normalization()


def protocol(create: bool = False) -> dict:
    """새 실험의 변경점·예산·공유 캐시와 모든 실행 코드를 학습 전에 고정한다."""
    path = OUT / "protocol.json"
    if path.exists():
        result = base.read(path)
        for name, digest in result["code_sha256"].items():
            assert sha256(REPO / name) == digest, name
        return result
    assert create and NORMALIZATION.exists()
    previous = base.read(PREVIOUS / "protocol.json")
    for name, digest in previous["code_sha256"].items():
        assert sha256(REPO / name) == digest, name
    files = [CONFIG, Path(__file__), REPO / "scripts/_pdv2_linear_run.py",
             REPO / "scripts/_pdv2_linear_verify.py", REPO / "src/retrieval/linear_fusion.py",
             REPO / "tests/test_linear_fusion.py"]
    files.extend(REPO / name for name in (
        "scripts/_pdv2_lag15_campaign.py", "scripts/_pdv2_pilot.py",
        "scripts/_pdv2_pair_evaluate.py", "scripts/_pdv2_c3.py",
        "scripts/_pdv2_no_context_ablation.py", "scripts/_causal_pr_dump.py",
        "scripts/_causal_branch_fusion.py", "configs/training_place_disjoint_v2_lag15.yaml"))
    code = dict(previous["code_sha256"])
    code.update({str(p.relative_to(REPO)): sha256(p) for p in files})
    inputs = dict(previous["input_sha256"])
    inputs[str(NORMALIZATION.relative_to(REPO))] = sha256(NORMALIZATION)
    inputs[str((PREVIOUS / "summary.json").relative_to(REPO))] = sha256(PREVIOUS / "summary.json")
    for kind in ("gnn", "point"):
        for marker in sorted((PREVIOUS / "pools" / (kind + "_initial")).glob("*/*/complete.json")):
            metadata = base.read(marker)
            inputs[str(marker.relative_to(REPO))] = sha256(marker)
            for filename, digest in metadata["files"].items():
                source = marker.parent / filename
                assert sha256(source) == digest, source
                inputs[str(source.relative_to(REPO))] = digest
    result = {"created_unix": time.time(), "config": CFG, "code_sha256": code,
              "input_sha256": inputs, "scope": "시드1 학습8·검증9 전용. 시험 실행·원고 교체 없음.",
              "previous_protocol_sha256": sha256(PREVIOUS / "protocol.json"),
              "train_sequences": previous["train_sequences"], "val_sequences": previous["val_sequences"],
              "score": "기존 key_cos·aux_cos·c_spec·c_col 원점수 4개. 양의 가중치4+절편1, 숨은층 없음. 같은 최대 점수로 선택·거절. 가중치 양수화는 학습용이며 추론에는 상수5개만 필요.",
              "normalization": "학습8·두 초기 표현·두 운영점의 실제 후보 쌍에서 공통 평균/표준편차를 한번 추정. 질의별 minmax 없음. 정규화를 원점수 가중치와 절편에 흡수해 학습/평가/내보내기 경로 동일.",
              "controls": "앞선 MLP 실험과 같은 초기 표현·질의·손실·12에포크·612업데이트·encoder LR 5e-5. 작은 선형식의 LR은 1e-2로 사전 고정(MLP 1e-3). 네 현재 조건에는 동일하게 적용. 10→4특징·정규화·양의가중치·LR도 바뀌므로 MLP와의 차이를 층수만의 효과로 해석하지 않음.",
              "online": previous["online"], "refresh": previous["refresh"], "loss": previous["loss"],
              "selection": previous["selection"],
              "reuse": "초기 후보/키/학습 그래프를 그대로 공유. 갱신 모델은 키와100+100후보를 다시 검색한다. 같은(q,j)의 보조·정렬 원점수만 초기 후보 캐시에서 재사용하며 새 쌍은 기존 함수로 계산. 각 후보 생성의 첫3질의는 새 계산과 대조.",
              "interpretation": "GNN과 이웃없는 투영의 표현고정/공동학습 네 조건. GNN 연산 변경 없음, 파라미터수 대조 아님. 기존·MLP 결과는 검증 참조로 병기. AP/F1/R100와 공통 임계값·방향을 함께 보고하며 유리한 지표만으로 시험 확장하지 않음."}
    write_json(path, result)
    return result


def cached_build_pool(embedding, bank, queries, positions) -> dict:
    """키와 후보는 재계산하고 모델과 무관한 동일 쌍 정렬만 재사용한다."""
    if ACTIVE_REFERENCE is None:
        return ORIGINAL_BUILD_POOL(embedding, bank, queries, positions)
    hkey, hfft, hnorm, column = bank
    op = {128: "pbev128_main", 240: "hspec12"}[hkey.shape[1]]
    with np.load(ACTIVE_REFERENCE / (op + ".npz")) as z:
        old = {k: z[k] for k in ("queries", "candidates", "static")}
    old_rows = np.searchsorted(old["queries"], queries)
    np.testing.assert_array_equal(old["queries"][old_rows], queries)
    candidates = np.full((len(queries), 200), -1, np.int64)
    static = np.zeros((len(queries), 200, 3), np.float64)
    distances = np.full((len(queries), 200), np.inf)
    reused, fresh_count = 0, 0
    for i, q in enumerate(queries):
        a = base.prefix_topk(embedding, int(q), int(q) - 29, 100)
        b = base.prefix_topk(hkey, int(q), int(q) - 29, 100)
        union = np.union1d(a, b)
        assert len(union) <= 200 and union.max() <= q - 30
        row = old_rows[i]
        valid = old["candidates"][row] >= 0
        previous = old["candidates"][row, valid]
        ix = np.searchsorted(previous, union)
        found = (ix < len(previous)) & (previous[ix.clip(max=len(previous) - 1)] == union)
        values = np.empty((len(union), 3), np.float64)
        values[found] = old["static"][row, ix[found]]
        missing = union[~found]
        if len(missing):
            values[~found] = np.stack([hkey[missing] @ hkey[q],
                base.align_scores(hfft, hnorm, int(q), missing),
                base.column_align_scores(column, int(q), missing)], axis=1)
        if i < 3:
            recomputed = np.stack([hkey[union] @ hkey[q], base.align_scores(hfft, hnorm, int(q), union),
                                   base.column_align_scores(column, int(q), union)], axis=1)
            np.testing.assert_allclose(values, recomputed, atol=1e-6, rtol=1e-6)
        candidates[i, :len(union)] = union
        static[i, :len(union)] = values
        distances[i, :len(union)] = np.linalg.norm(positions[union] - positions[q], axis=1)
        reused += int(found.sum())
        fresh_count += int((~found).sum())
        if i and i % 4000 == 0:
            print("POOL_QUERIES", i, len(queries), flush=True)
    return {"queries": queries, "candidates": candidates, "static": static,
            "correct": distances <= 5., "distances": distances,
            "reused_pairs": np.asarray(reused), "fresh_pairs": np.asarray(fresh_count)}


def snapshot(kind: str, tag: str, role: str, sequence: str, bundle: tuple) -> Path:
    """현재 체크포인트의 후보를 만들 때만 초기 정렬 재사용 문맥을 연다."""
    global ACTIVE_REFERENCE
    assert role in ("train", "val")
    ACTIVE_REFERENCE = PREVIOUS / "pools" / f"{kind}_initial" / role / sequence.replace("/", "_")
    try:
        return ORIGINAL_SNAPSHOT(kind, tag, role, sequence, bundle)
    finally:
        ACTIVE_REFERENCE = None


def smoke() -> None:
    """실제 학습 그래프 역전파와 정렬 재사용을 결과 선택 전에 확인한다."""
    records = []
    for kind in ("gnn", "point"):
        bundle = base.make_encoder(kind)
        path = PREVIOUS / "pools" / f"{kind}_initial/train/kitti_06"
        graph = torch.load(path / "graph.pt", map_location="cuda", weights_only=False)
        embedding = torch.nn.functional.normalize(bundle[1](graph), dim=1)
        heads = torch.nn.ModuleDict({op: Scorer() for op in base.OPS}).cuda()
        losses = []
        for op in base.OPS:
            pool = base.load_pool(path, op)
            c = pool["candidates"][:64]
            features = fusion_features(embedding, pool["queries"][:64], c, pool["static"][:64], c >= 0)
            losses.append(base.candidate_loss(heads[op](features), pool["correct"][:64], c >= 0))
        loss = torch.stack(losses).mean()
        loss.backward()
        gradients = [p.grad for p in bundle[1].parameters() if p.grad is not None]
        assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
        records.append({"kind": kind, "loss": float(loss), "encoder_gradient_tensors": len(gradients),
                        "scorer_parameters_per_op": sum(p.numel() for p in heads[base.OPS[0]].parameters())})
    write_json(OUT / "preflight_checks.json", {"status": "passed", "actual_graph_backward": records,
               "note": "초기 모델로 역전파만 확인. 이 계산의 가중치는 본 학습에 쓰지 않는다."})
    print("SMOKE_PASSED", records, flush=True)


def report() -> None:
    """동일한 최종 검증 선택 뒤 원점수 가중치와 MLP 참조를 기록한다."""
    base.report()
    result = base.read(OUT / "summary.json")
    result["scope"] = "네 원점수 선형 융합의 시드1 검증9 최종 판정. 시험 실행 없음."
    result["mlp_reference"] = base.read(PREVIOUS / "summary.json")["arms"]
    for arm, row in result["arms"].items():
        state = torch.load(OUT / arm / f"epoch{row['epoch']}.pth", map_location="cuda", weights_only=False)
        heads = torch.nn.ModuleDict({op: Scorer() for op in base.OPS}).cuda()
        heads.load_state_dict(state["heads"], strict=True)
        row["exported_fusion"] = {op: heads[op].export() for op in base.OPS}
    write_json(OUT / "summary.json", result)


# 이미 동결된 MLP 코드 파일은 변경하지 않고 이번 프로세스의 의존성만 주입한다.
base.OUT, base.CFG, base.CONFIG = OUT, CFG, CONFIG
base.protocol, base.snapshot, base.build_pool = protocol, snapshot, cached_build_pool
base.CandidateScorer, base.pair_features = Scorer, fusion_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "smoke", "protocol", "train", "evaluate", "report"))
    parser.add_argument("--arm", choices=CFG["arms"])
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    base.setup()
    if args.action == "prepare":
        prepare()
    elif args.action == "smoke":
        smoke()
    elif args.action == "protocol":
        protocol(True)
    elif args.action == "train":
        base.train(args.arm)
    elif args.action == "evaluate":
        base.evaluate(args.arm, args.epoch, args.sequence)
    else:
        report()
