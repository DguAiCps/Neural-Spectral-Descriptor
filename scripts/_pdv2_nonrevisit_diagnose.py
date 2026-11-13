#!/usr/bin/env python3
"""비재방문 학습의 학습 적합·검증 손익과 원장 무결성을 확인한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import _pdv2_nonrevisit_learning as run


def training() -> None:
    """초기 승자를 고정한 학습 진단을 최종 재검색 평가와 분리한다."""
    target = run.OUT / "training_diagnostic.json"
    if target.exists():
        return
    spec, fit = run.prepare(), run.calibration()
    result, aggregate = {}, {}
    for arm in ("initial", *run.ARMS):
        net = None if arm == "initial" else run.model(run.OUT / arm / "epoch4.pth")
        per = {}
        for seq in spec["train"]:
            folder = run.OUT / "pools" / seq.replace("/", "_")
            row = run.ARR(folder / "pool.npz")
            if net is None:
                ctx = row["context"]
            else:
                graph = torch.load(folder / "graph.pt", map_location="cuda", weights_only=False)
                with torch.no_grad():
                    key = F.normalize(net(graph), dim=1).cpu().numpy()
                cache = run.ARR(run.cache_path(seq))
                xyz = cache["poses"][:, :3, 3].astype(np.float64)
                path = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
                ctx = run.neural_scores(key, path, row["query_idx"], row["j_star"])
                del graph
            logit = (run.predict_score(row["spectral"][:, None], fit["spectral"]) +
                     (ctx-fit["context_mean"])/fit["context_scale"])
            eligible, y = row["eligible"], row["correct"].astype(float)
            cohorts = {"correct": eligible & row["correct"],
                       "wrong_revisit": eligible & row["positive"] & ~row["correct"],
                       "nonrevisit": eligible & ~row["positive"], "all": eligible}
            per[seq] = {}
            for label, use in cohorts.items():
                per[seq][label] = dict(count=int(use.sum()),
                    BCE=float(np.mean(np.logaddexp(0, logit[use])-y[use]*logit[use])) if use.any() else None,
                    mean_logit=float(logit[use].mean()) if use.any() else None)
            print("TRAIN_DIAG", arm, seq, flush=True)
        result[arm] = per
        aggregate[arm] = {label: {m: float(np.mean([v[label][m] for v in per.values() if v[label][m] is not None]))
                                 for m in ("BCE", "mean_logit")} for label in cohorts}
        if net is not None:
            del net
            torch.cuda.empty_cache()
    run.WRITE(target, dict(status="complete", per_sequence=result, mean_over_available_sequences=aggregate,
        scope="고정 초기 승자·lambda1의 학습 자료 진단. 최종 재검색 검증 성능과 구분.",
        test_used=False, script_sha256=run.HASH(Path(__file__))))


def validation() -> None:
    """선택된 최종 성능의 점수 경로와 실제 승자 변화량을 기록한다."""
    spec = run.prepare()
    summary = run.READ(run.OUT / "summary.json")
    results = {}
    for arm, choice in summary["selected"].items():
        zero = summary["all_results"][f"{arm}__epoch{choice['epoch']}__lambda0"]
        results[arm] = dict(epoch=choice["epoch"], coefficient=choice["coefficient"],
            spectral_only=zero["macro"], final=choice["macro"],
            score_connection_delta_pp={m:100*(choice["macro"][m]-zero["macro"][m]) for m in run.base.METRICS})
    choice = summary["selected"]["all_queries"]
    counts = dict(queries=0, winner_unchanged=0, correct_gained=0, correct_lost=0,
                  same_winner_context_changed=0)
    for seq in spec["val"]:
        name = seq.replace("/", "_")+".npz"
        before = run.ARR(run.OUT / "initial/epoch0/val" / name)
        after = run.ARR(run.OUT / "all_queries" / f"epoch{choice['epoch']}" / "val" / name)
        np.testing.assert_array_equal(before["query_idx"], after["query_idx"])
        np.testing.assert_array_equal(before["min_past_gt_dist"], after["min_past_gt_dist"])
        same = before["j_star"] == after["j_star"]
        old = (before["min_past_gt_dist"]<=5) & (before["top1_geo_dist"]<=5)
        new = (after["min_past_gt_dist"]<=5) & (after["top1_geo_dist"]<=5)
        np.testing.assert_array_equal(before["spectral"][same], after["spectral"][same])
        counts["queries"] += len(same)
        counts["winner_unchanged"] += int(same.sum())
        counts["correct_gained"] += int((new & ~old).sum())
        counts["correct_lost"] += int((old & ~new).sum())
        counts["same_winner_context_changed"] += int((same & (abs(before["context"]-after["context"])>1e-6)).sum())
    run.WRITE(run.OUT / "validation_diagnostic.json", dict(models=results,
        all_queries_vs_initial=counts, test_used=False,
        note="동일 모델의 spectral_only와 final은 후보·승자가 같고 수락 점수만 다르다. 서로 다른 학습 모델은 재검색까지 포함한다."))


def verify() -> None:
    """입력 해시·고정 학습 조건·모든 최종 지표를 다시 검사한다."""
    spec, fit = run.prepare(), run.calibration()
    for p, digest in fit["sources"].items():
        assert run.HASH(ROOT / p) == digest
    pools, ledgers, checkpoints, prefix = 0, 0, 0, 0
    for seq in spec["train"]:
        folder = run.OUT / "pools" / seq.replace("/", "_")
        audit = run.READ(folder / "audit.json")
        for p, h in audit["outputs"].items():
            assert run.HASH(folder / p) == h
        for p, h in audit["inputs"].items():
            assert run.HASH(ROOT / p) == h
        row = run.ARR(folder / "pool.npz")
        assert (row["j_star"]<=row["query_idx"]-30).all()
        eligible = (row["min_past_gt_dist"]<=5) | (row["min_past_gt_dist"]>25)
        np.testing.assert_array_equal(eligible, row["eligible"])
        for col in ("query", "candidate"):
            assert (row["plan_"+col]+10 <= row["query_idx"][eligible, None, None]+15).all()
        pools += 1
    for arm in run.ARMS:
        record = run.READ(run.OUT / arm / "training_complete.json")
        assert record["updates"] == 256 and record["fixed_buffers_verified"]
        for p, h in record["checkpoints"].items():
            assert run.HASH(run.OUT / arm / p) == h
            checkpoints += 1
    original = {}
    for arm in ("initial", *run.ARMS):
        for epoch in ((0,) if arm=="initial" else run.EPOCHS):
            for seq in spec["val"]:
                p = run.OUT / arm / f"epoch{epoch}" / "val" / (seq.replace("/", "_")+".json")
                record = run.READ(p)
                assert record["status"] == "passed" and run.HASH(p.with_suffix(".npz")) == record["ledger_sha256"]
                row = run.ARR(p.with_suffix(".npz"))
                for label, metric in record["metrics"].items():
                    assert run.base.independent_metrics(row, row[label]) == metric
                assert (row["j_star"]<=row["query_idx"]-30).all()
                ledgers += 1
                prefix += len(record["prefix_checks"])
                if arm == "initial":
                    ref = run.ARR(ROOT / "results/paper_fixed_threshold_20261101/nsd_val/s1" / p.with_suffix(".npz").name)
                    original[seq] = run.base.independent_metrics(ref, ref["score"])
    summary = run.READ(run.OUT / "summary.json")
    paper_baseline = run.base.macro(original)
    difference = {m:100*(summary["baseline"]["macro"][m]-paper_baseline["macro"][m]) for m in run.base.METRICS}
    run.WRITE(run.OUT / "completion_validation.json", dict(status="passed", training_pools=pools,
        checkpoints=checkpoints, final_ledgers=ledgers, metric_rows=ledgers*len(run.LAMBDAS),
        validation_prefix_checks=prefix, training_prefix_checks=2*pools,
        original_paper_validation=paper_baseline, regenerated_baseline_delta_pp=difference,
        summary_sha256=run.HASH(run.OUT / "summary.json"), test_used=False,
        script_sha256=run.HASH(Path(__file__))))
    print("VERIFIED", pools, checkpoints, ledgers, difference, flush=True)


if __name__ == "__main__":
    run.configure()
    training()
    validation()
    verify()
