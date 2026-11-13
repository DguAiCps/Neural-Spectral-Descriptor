#!/usr/bin/env python3
"""두 문맥 이득 대조와 공통 이득 대조의 저장 결과를 독립 검산한다."""
import json
from pathlib import Path

import numpy as np
import torch

import _pdv2_diffattn_spectral_gain as e


def main() -> None:
    """모든 원장·학습 변경 범위·같은 배치·그림 보존을 확인한다."""
    e.run.configure()
    roots = [
        e.OUT,
        e.ROOT / "results/pdv2_diffattn_spectral_only_20261104",
        e.ROOT / "results/pdv2_diffattn_spectral_global_20261104",
    ]
    ledgers = queries = prefixes = metric_sets = 0
    maximum = 0.0
    orders = []
    frozen = torch.load(
        e.PARENT / "recenter_frozen/epoch0.pth", map_location="cpu", weights_only=False
    )["model_state_dict"]
    for root in roots:
        spec = e.run.READ(root / "protocol.json")
        for p, h in spec.get("sources", {}).items():
            assert e.run.HASH(e.ROOT / p) == h, p
        if "source_sha256" in spec:
            assert (
                e.run.HASH(e.ROOT / "scripts/_pdv2_diffattn_spectral_global.py")
                == spec["source_sha256"]
            )
        paths = (
            list(root.glob("*/epoch*/val/*.json"))
            if root != roots[-1]
            else list((root / "val").glob("*.json"))
        )
        for path in paths:
            rec = e.run.READ(path)
            assert e.run.HASH(path.with_suffix(".npz")) == rec["ledger_sha256"]
            row = e.run.ARR(path.with_suffix(".npz"))
            assert np.all(row["j_star"] <= row["query_idx"] - 30)
            for name, metric in rec["metrics"].items():
                actual = e.run.white.prior.BASE.exp.metrics(row[name], row)
                assert np.isfinite(row[name]).all()
                for m in e.run.base.METRICS:
                    if metric[m] is not None:
                        err = abs(actual[m] - metric[m])
                        maximum = max(maximum, err)
                        assert err < 1e-12
                metric_sets += 1
            ledgers += 1
            queries += len(row["query_idx"])
            prefixes += len(rec.get("prefix_checks", []))
        for arm in e.ARMS:
            record = e.run.READ(root / arm / "training_complete.json")
            assert record["updates"] == 512
            orders.append([r["batch_hashes"] for r in record["records"]])
            for name, h in record["checkpoints"].items():
                path = root / arm / name
                assert e.run.HASH(path) == h
                if path.suffix == ".pth":
                    state = torch.load(path, map_location="cpu", weights_only=False)[
                        "model_state_dict"
                    ]
                    for n, v in state.items():
                        assert torch.isfinite(v).all()
                        if not n.startswith("gnn.convs."):
                            torch.testing.assert_close(v, frozen[n], rtol=0, atol=0)
    assert all(o == orders[0] for o in orders)
    baseline = e.run.READ(
        e.ROOT / "results/pdv2_learning_plan_20261104/restoration.json"
    )["figure_sha256"]
    for p, h in baseline.items():
        assert e.run.HASH(e.ROOT / "docs/paper/body_v2.2" / p) == h, p
    # 항등 강도의 실제 최종 점수가 직전 동결 모델과 같은지 확인한다.
    total_error = 0.0
    for seq in e.run.READ(e.OUT / "protocol.json")["val"]:
        name = seq.replace("/", "_")
        current = e.run.ARR(e.OUT / "frozen/epoch0/val" / (name + ".npz"))
        old = e.run.ARR(
            e.previous.prior.BASELINE / "val/s1" / (name + "_frozen_full.npz")
        )
        np.testing.assert_array_equal(current["j_star"], old["j_star"])
        value = old["score"]
        total_error = max(total_error, float(np.max(abs(current["s0"] - value))))
        np.testing.assert_allclose(current["s0"], value, atol=8e-6, rtol=0)
    result = dict(
        status="passed",
        ledgers=ledgers,
        queries=queries,
        metric_sets=metric_sets,
        prefix_checks=prefixes,
        max_metric_error=maximum,
        equal_batches=True,
        frozen_score_reproduction_error=total_error,
        updates=6 * 512,
        unchanged_figures=len(baseline),
        test_used=False,
        source_sha256=e.run.HASH(Path(__file__)),
    )
    for root in roots:
        e.run.WRITE(root / "integrated_validation.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
