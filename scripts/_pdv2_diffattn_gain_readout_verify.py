#!/usr/bin/env python3
"""출력층 재학습의 원장·고정 파라미터·검색 접두부를 검산한다."""
from pathlib import Path

import numpy as np
import torch

import _pdv2_diffattn_gain_readout as e


def main() -> None:
    """현재 선택 결과의 전체 자료와 이전 실패 기록을 함께 보존한다."""
    e.run.configure()
    spec = e.prepare()
    orders = []
    checkpoints = 0
    for arm, (kind, _) in e.ARMS.items():
        info = e.run.READ(e.OUT / arm / "training_complete.json")
        assert info["updates"] == 512
        orders.append([r["batch_hashes"] for r in info["records"]])
        net = e.model(kind)
        initial = net.state_dict()
        allowed = {n for n, p in net.named_parameters() if p.requires_grad}
        for name, h in info["checkpoints"].items():
            p = e.OUT / arm / name
            assert e.run.HASH(p) == h
            state = torch.load(p, map_location="cuda", weights_only=False)[
                "model_state_dict"
            ]
            for n, v in state.items():
                assert torch.isfinite(v).all()
                if n not in allowed:
                    torch.testing.assert_close(v, initial[n], rtol=0, atol=0)
            checkpoints += 1
        del net, initial, state
    assert all(o == orders[0] for o in orders)
    queries = metrics = prefixes = retrieval = 0
    max_metric = 0.0
    max_score = 0.0
    max_gain = 0.0
    max_logit = 0.0
    frozen = e.prior.previous.prior.gate.MODEL(
        e.prior.PARENT / "recenter_frozen/epoch0.pth"
    )
    weight = e.run.cal.fusion_weights(spec["confidence"])[
        e.run.cal.FUSIONS.index("aux_half")
    ]
    for s in spec["val"]:
        name = s.replace("/", "_")
        record = e.run.READ(e.OUT / "val_checked" / (name + ".json"))
        path = e.OUT / "val_checked" / (name + ".npz")
        assert e.run.HASH(path) == record["ledger_sha256"]
        row = e.run.ARR(path)
        assert (row["j_star"] <= row["query_idx"] - 30).all()
        queries += len(row["query_idx"])
        for tag, expected in record["metrics"].items():
            actual = e.run.white.prior.BASE.exp.metrics(row[tag], row)
            for m in e.run.base.METRICS:
                if expected[m] is not None:
                    error = abs(actual[m] - expected[m])
                    assert error < 1e-12
                    max_metric = max(max_metric, error)
            metrics += 1
        for r in record["prefix_checks"]:
            max_score = max(max_score, r["score_error"], r["masked_score_error"])
            max_gain = max(max_gain, r["gain_error"], r["masked_gain_error"])
            max_logit = max(max_logit, r["logit_error"])
            prefixes += 1
        cache, pos, _, _, bank, _, _, _ = e.run.inputs("val", s)
        for at in (len(row["query_idx"]) // 3, 2 * len(row["query_idx"]) // 3):
            t = int(row["query_idx"][at])
            end = t + 16
            keys = e.run.source.make_keys(
                frozen, cache["descriptors"], cache["poses"], end
            )
            raw = keys[:, :288].copy()
            raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
            trunc = e.run.current.adoption.previous.source.base.exp.truncate_bank
            winner = e.prior.previous.prior.stages.query_search(
                raw, trunc(bank, end), pos[:end], t, weight, {"full": (60, 200)}
            )[0]["full"]
            assert winner == row["j_star"][at]
            retrieval += 1
    amendment = e.run.READ(e.OUT / "validation_runtime_amendment.json")
    for p, h in amendment["failures"].items():
        assert e.run.HASH(e.ROOT / p) == h
    figures = e.run.READ(
        e.ROOT / "results/pdv2_learning_plan_20261104/restoration.json"
    )["figure_sha256"]
    for p, h in figures.items():
        assert e.run.HASH(e.ROOT / "docs/paper/body_v2.2" / p) == h
    result = dict(
        status="passed",
        training_conditions=len(e.ARMS),
        updates=512 * len(e.ARMS),
        checkpoints=checkpoints,
        ledgers=9,
        queries=queries,
        metric_sets=metrics,
        prefix_checks=prefixes,
        masked_future_checks=prefixes,
        retrieval_prefix_checks=retrieval,
        max_metric_error=max_metric,
        max_score_prefix_error=max_score,
        max_gain_prefix_error=max_gain,
        max_unnormalized_output_error=max_logit,
        equal_batches=True,
        source_sha256=e.run.HASH(Path(__file__)),
        unchanged_figures=len(figures),
        test_used=False,
    )
    e.run.WRITE(e.OUT / "integrated_validation.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
