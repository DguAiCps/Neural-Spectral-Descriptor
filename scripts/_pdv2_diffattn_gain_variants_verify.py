#!/usr/bin/env python3
"""이득 출력의 값 변환·고정 개입·전체 학습 결과를 검산한다."""
import argparse
import importlib
from pathlib import Path

import numpy as np
import torch


def main(kind: str) -> None:
    """원장 재집계·접두 검사·변경 범위·그림 보존을 확인한다."""
    module = {
        "moments": "_pdv2_diffattn_gain_moments",
        "probe": "_pdv2_diffattn_gain_attention_probe",
        "full": "_pdv2_diffattn_gain_full",
    }[kind]
    exp = importlib.import_module(module)
    e = getattr(exp, "base", None) or exp.e
    e.run.configure()
    spec = exp.prepare()
    result = e.run.READ(exp.OUT / "summary.json")
    n_query = n_metric = n_prefix = n_updates = n_checkpoint = 0
    metric_error = score_error = gain_error = 0.0
    orders = []
    for p in sorted((exp.OUT / "val_checked").glob("*.json")):
        rec = e.run.READ(p)
        assert rec["status"] == "passed"
        assert e.run.HASH(p.with_suffix(".npz")) == rec["ledger_sha256"]
        row = e.run.ARR(p.with_suffix(".npz"))
        n_query += len(row["query_idx"])
        assert np.all(row["j_star"] <= row["query_idx"] - 30)
        for tag, expected in rec["metrics"].items():
            actual = e.run.white.prior.BASE.exp.metrics(row[tag], row)
            for m in e.run.base.METRICS:
                if expected[m] is not None:
                    er = abs(actual[m] - expected[m])
                    assert er < 1e-12
                    metric_error = max(metric_error, er)
            n_metric += 1
        for r in rec["prefix_checks"]:
            n_prefix += 1
            score_error = max(score_error, r["score_error"], r["masked_score_error"])
            gain_error = max(gain_error, r["gain_error"], r["masked_gain_error"])
    assert n_query == 99236 and score_error <= 8e-6 and gain_error <= 0.002
    for arm, (architecture, _) in e.ARMS.items():
        path = exp.OUT / arm / "training_complete.json"
        if not path.exists():
            assert kind == "probe"
            continue
        record = e.run.READ(path)
        assert record["updates"] == 512
        n_updates += 512
        orders.append([r["batch_hashes"] for r in record["records"]])
        net = exp.model(architecture)
        initial = net.state_dict()
        allowed = {n for n, v in net.named_parameters() if v.requires_grad}
        for name, h in record["checkpoints"].items():
            cp = exp.OUT / arm / name
            assert e.run.HASH(cp) == h
            state = torch.load(cp, map_location="cuda", weights_only=False)[
                "model_state_dict"
            ]
            for n, v in state.items():
                assert torch.isfinite(v).all()
                if n not in allowed:
                    torch.testing.assert_close(v, initial[n], rtol=0, atol=0)
            n_checkpoint += 1
        del net, state, initial
    assert not orders or all(o == orders[0] for o in orders)
    comparison = {}
    repetition_checks = {}
    if kind == "moments":
        old = e.run.READ(exp.PARENT / "summary.json")
        for arm in ("lr4", "lr3"):
            for epoch in (4, 8):
                a = result["results"][f"signed_{arm}_{epoch}"]["macro"]
                b = old["results"][f"diff_{arm}_{epoch}"]["macro"]
                comparison[f"{arm}_{epoch}"] = {
                    k: 100 * (a[k] - b[k]) for k in e.run.base.METRICS
                }
                repetition_checks[f"{arm}_{epoch}"] = all(
                    abs(a[k] - b[k]) < 1e-4 for k in e.run.base.METRICS
                )
                if arm == "lr4":
                    assert repetition_checks[f"{arm}_{epoch}"]
    figs = e.run.READ(e.ROOT / "results/pdv2_learning_plan_20261104/restoration.json")[
        "figure_sha256"
    ]
    for p, h in figs.items():
        assert e.run.HASH(e.ROOT / "docs/paper/body_v2.2" / p) == h
    report = dict(
        status=(
            "passed_with_training_repetition_warning"
            if repetition_checks and not all(repetition_checks.values())
            else "passed"
        ),
        queries=n_query,
        metric_sets=n_metric,
        prefix_checks=n_prefix,
        masked_future_checks=n_prefix,
        updates=n_updates,
        checkpoints=n_checkpoint,
        maximum_metric_error=metric_error,
        maximum_score_prefix_error=score_error,
        maximum_gain_prefix_error=gain_error,
        equal_batches=True,
        signed_reproduction_delta_pp=comparison,
        training_repetition_checks=repetition_checks,
        repetition_note=(
            "동일수식재실행에서lr1e-4는재현기준통과,lr1e-3는미달. "
            "고학습률0.01%p재현검사의최초실패를보존한다. "
            "해당고학습률의작은차이는연산자효과로해석하지않음. "
            "GPU비결정성의영향을의심하나단독원인은확정하지않음."
            if kind == "moments"
            else None
        ),
        unchanged_figures=len(figs),
        test_used=False,
        source_sha256=e.run.HASH(Path(__file__)),
    )
    e.run.WRITE(exp.OUT / "integrated_validation.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("moments", "probe", "full"))
    main(parser.parse_args().kind)
