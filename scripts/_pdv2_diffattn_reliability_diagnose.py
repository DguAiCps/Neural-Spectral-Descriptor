#!/usr/bin/env python3
"""학습 고정 승자에서 신뢰도 학습의 작동과 최종 원장을 검산한다."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import _pdv2_diffattn_reliability as exp


def training_diagnostic() -> None:
    """초기 승자를 고정한 학습 손실과 정오별 신뢰 질량을 측정한다."""
    spec, fit = exp.activate(), exp.calibrate()
    candidates = [("frozen", 0, None)] + [
        (arm, epoch, exp.OUT / arm / f"epoch{epoch}.pth")
        for arm in exp.ARMS
        for epoch in exp.EPOCHS
    ]
    results = {}
    graph_error = 0.0
    for arm, epoch, checkpoint in candidates:
        net = exp.model(checkpoint)
        modes = exp.MODES if arm == "frozen" else (exp.ARMS[arm][0],)
        rows = {mode: {} for mode in modes}
        stats = {mode: {} for mode in modes}
        for seq in spec["train"]:
            name = seq.replace("/", "_")
            arrays = exp.run.ARR(exp.OUT / "pools" / name / "pool.npz")
            graph = torch.load(
                exp.PARENT / "pools" / name / "graph.pt",
                map_location="cuda",
                weights_only=False,
            )
            with torch.no_grad():
                key = torch.nn.functional.normalize(net(graph), dim=-1).cpu().numpy()
            if arm == "frozen":
                cache = exp.run.ARR(exp.run.cache_path(seq))
                check = exp.run.source.make_keys(
                    net, cache["descriptors"], cache["poses"]
                )
                error = float(np.max(np.abs(check - key)))
                np.testing.assert_allclose(check, key, rtol=4e-5, atol=4e-6)
                graph_error = max(graph_error, error)
            data = {
                k: arrays["plan_" + k]
                for k in ("query", "candidate", "weight", "peaks")
            }
            scores = exp.score_all(key, data, modes)
            for mode in modes:
                score = scores[f"{mode}_1"]
                rows[mode][seq] = exp.run.base.independent_metrics(arrays, score)
                logits = exp.run.predict_score(score[:, None], fit[mode])
                loss = np.logaddexp(0.0, logits) - arrays["correct"] * logits
                classes = []
                for label in (False, True):
                    use = arrays["eligible"] & (arrays["correct"] == label)
                    if use.any():
                        classes.append(float(loss[use].mean()))
                stats[mode][seq] = dict(balanced_BCE=float(np.mean(classes)))
            del graph, key
        for mode in modes:
            result = dict(
                **exp.run.base.macro(rows[mode]),
                sequences=rows[mode],
                mean_balanced_BCE=float(
                    np.mean([r["balanced_BCE"] for r in stats[mode].values()])
                ),
                loss_by_sequence=stats[mode],
            )
            results[f"{arm}_{epoch}_{mode}"] = result
            print(
                "TRAIN_DIAG",
                arm,
                epoch,
                mode,
                result["macro"],
                result["mean_balanced_BCE"],
                flush=True,
            )
    exp.run.WRITE(
        exp.OUT / "training_diagnostic.json",
        dict(
            results=results,
            graph_reproduction_max_abs=graph_error,
            source_sha256=exp.run.HASH(Path(__file__)),
            scope="학습8 초기 실제 승자 고정·신뢰도강도1. 학습 후 전체 재검색 또는 새로운 장소의 최종 성능이 아니다.",
            test_used=False,
        ),
    )


def verify() -> None:
    """저장한 원장·체크포인트·접두부·그림의 완료 상태를 확인한다."""
    spec = exp.prepare()
    assert exp.run.READ(exp.OUT / "status.json")["status"] == "validation_complete"
    for name, digest in exp.run.READ(exp.OUT / "input_freeze.json")["sources"].items():
        assert exp.run.HASH(exp.ROOT / name) == digest, name
    count, queries, prefixes, reproduction = 0, 0, 0, 0.0
    for path in sorted(exp.OUT.glob("*/epoch*/val/*.json")):
        record = exp.run.READ(path)
        ledger = path.with_suffix(".npz")
        assert exp.run.HASH(ledger) == record["ledger_sha256"]
        checkpoint = (
            exp.ROOT / spec["initial"]
            if record["arm"] == "frozen"
            else exp.OUT / record["arm"] / f'epoch{record["epoch"]}.pth'
        )
        assert exp.run.HASH(checkpoint) == record["checkpoint_sha256"]
        row = exp.run.ARR(ledger)
        for name, expected in record["metrics"].items():
            actual = exp.run.base.independent_metrics(row, row[name])
            for m in exp.run.base.METRICS:
                if actual[m] is not None:
                    np.testing.assert_allclose(
                        actual[m], expected[m], rtol=0, atol=1e-12
                    )
        count += 1
        queries += len(row["query_idx"])
        prefixes += len(record["prefix_checks"])
        reproduction = max(reproduction, record["zero_reproduction_max_abs"])
    assert (count, queries, prefixes) == (81, 893124, 162)
    for arm in exp.ARMS:
        training = exp.run.READ(exp.OUT / arm / "training_complete.json")
        for name, digest in training["checkpoints"].items():
            assert exp.run.HASH(exp.OUT / arm / name) == digest
    figures = exp.run.READ(
        exp.ROOT / "results/pdv2_learning_plan_20261104/restoration.json"
    )["figure_sha256"]
    for name, digest in figures.items():
        assert exp.run.HASH(exp.ROOT / "docs/paper/body_v2.2" / name) == digest
    exp.run.WRITE(
        exp.OUT / "integrated_validation.json",
        dict(
            status="passed",
            ledgers=count,
            rows=queries,
            prefix_checks=prefixes,
            zero_reproduction_max_abs=reproduction,
            figure_hashes_unchanged=len(figures),
            test_used=False,
            source_sha256=exp.run.HASH(Path(__file__)),
        ),
    )
    print("VERIFIED", count, queries, prefixes, reproduction, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "verify"))
    args = parser.parse_args()
    exp.run.configure()
    if args.action == "train":
        training_diagnostic()
    else:
        verify()
