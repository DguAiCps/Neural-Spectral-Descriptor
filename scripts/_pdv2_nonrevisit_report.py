#!/usr/bin/env python3
"""기본·기울기 보정 비재방문 학습 대조를 최종 검증 지표로 통합한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import _pdv2_nonrevisit_learning as base
import _pdv2_nonrevisit_balanced as balanced


def main() -> None:
    """선택 기회·최종 점수·학습 신호를 분리해 결과를 저장한다."""
    spec = base.prepare()
    balanced.prepare()
    prior = base.READ(base.OUT / "summary.json")
    extra = base.READ(balanced.OUT / "summary.json")
    rows = {"current": prior["baseline"], "frozen_context": prior["selected"]["initial"]}
    rows.update({"basic_"+a: v for a,v in prior["selected"].items() if a!="initial"})
    rows.update({"balanced_"+a: v for a,v in extra["selected"].items()})
    comparisons = {}
    now = rows["balanced_all_queries"]
    for name in ("current", "frozen_context", "basic_retrieval_only", "balanced_positive_only"):
        ref = rows[name]
        comparisons[name] = dict(delta_pp={m:100*(now["macro"][m]-ref["macro"][m]) for m in base.base.METRICS},
            family_delta_pp={f:{m:100*(now["families"][f][m]-ref["families"][f][m]) for m in base.base.METRICS}
                             for f in now["families"]})
    checks = {}
    for name, values in comparisons.items():
        d = values["delta_pp"]
        checks[name] = dict(AP=d["AP"]>=.1, F1=d["max_F1"]>=-.1, R100=d["R@100P"]>=-.1,
            family_AP=all(v["AP"]>=-.5 for v in values["family_delta_pp"].values()))
    same_mixture = {}
    for lam in base.LAMBDAS:
        a = extra["all_results"][f"all_queries__epoch4__lambda{lam:g}"]
        p = extra["all_results"][f"positive_only__epoch4__lambda{lam:g}"]
        same_mixture[str(lam)] = {m:100*(a["macro"][m]-p["macro"][m]) for m in base.base.METRICS}
    verified, checkpoints = 0, 0
    for arm in balanced.ARMS:
        record = base.READ(balanced.OUT / arm / "training_complete.json")
        for p,h in record["checkpoints"].items():
            assert base.HASH(balanced.OUT / arm / p) == h
            checkpoints += 1
        for seq in spec["val"]:
            p = balanced.OUT / arm / "epoch4/val" / (seq.replace("/", "_")+".json")
            record = base.READ(p)
            assert record["status"] == "passed" and base.HASH(p.with_suffix(".npz")) == record["ledger_sha256"]
            row = base.ARR(p.with_suffix(".npz"))
            assert (row["j_star"]<=row["query_idx"]-30).all()
            for label, metric in record["metrics"].items():
                assert base.base.independent_metrics(row, row[label]) == metric
                independent = base.white.prior.BASE.exp.metrics(row[label], row)
                for m in base.base.METRICS:
                    if metric[m] is not None:
                        np.testing.assert_allclose(metric[m], independent[m], rtol=0, atol=1e-12)
            verified += 1
    diagnostics = {"basic": base.READ(base.OUT / "training_diagnostic.json"),
                   "balanced": base.READ(balanced.OUT / "training_diagnostic.json")}
    report = dict(status="complete", seed=1, test_used=False, rows=rows,
        balanced_all_queries_comparisons=comparisons, advancement_checks=checks,
        eligible_for_three_seed_confirmation=all(all(v.values()) for v in checks.values()),
        same_mixture_nonrevisit_delta_pp=same_mixture,
        training_diagnostics={name:v["mean_over_available_sequences"] for name,v in diagnostics.items()},
        limitations="학습8/검증9·시드1. 기본은epoch2/4 선택, 기울기보정은epoch4만. 같은5혼합계수. 고정 학습 승자와 GT 학습 입력의 한계. 시험 미사용. 구조 신규성·여러시드 통계 우위 미입증.",
        sources={str(p.relative_to(ROOT)):base.HASH(p) for p in (
            Path(__file__), base.OUT / "summary.json", balanced.OUT / "summary.json",
            base.OUT / "completion_validation.json", base.OUT / "gradient_diagnostic.json")})
    base.WRITE(base.OUT / "integrated_report.json", report)
    base.WRITE(balanced.OUT / "completion_validation.json", dict(status="passed", checkpoints=checkpoints,
        final_ledgers=verified, metric_rows=verified*len(base.LAMBDAS),
        prefix_checks=verified*2, test_used=False, script_sha256=base.HASH(Path(__file__))))
    for label, value in rows.items():
        print(label, value["epoch"], value["coefficient"], {m:round(v*100,5) for m,v in value["macro"].items()}, flush=True)
    print("ELIGIBLE", report["eligible_for_three_seed_confirmation"], flush=True)


if __name__ == "__main__":
    base.configure()
    main()
