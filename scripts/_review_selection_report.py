#!/usr/bin/env python3
"""동결된 AP/R@1 선택의 검증·시험 지표와 원장 동일성을 검산한다."""
from __future__ import annotations

from pathlib import Path
import csv
import json
import time

import numpy as np

from _review_selection_bev import OUT, ROOT, CITY, TEST, VAL, METRICS, old_val, metrics, read, sha, write


def independent(arrays: dict) -> dict:
    """점수별 TP/FP를 별도 묶어 네 지표를 검산한다."""
    positive = arrays["min_past_gt_dist"] <= 5
    if not positive.any():
        return {key: None for key in METRICS}
    eligible = positive | (arrays["min_past_gt_dist"] > 25)
    correct = arrays["top1_geo_dist"] <= 5
    scores = arrays["score_negl2"][eligible]
    values, inverse = np.unique(scores, return_inverse=True)
    tp_groups = np.bincount(inverse, weights=correct[eligible].astype(np.float64), minlength=len(values))[::-1]
    fp_groups = np.bincount(inverse, weights=(~correct[eligible]).astype(np.float64), minlength=len(values))[::-1]
    tp, fp = np.cumsum(tp_groups), np.cumsum(fp_groups)
    precision, recall = tp / (tp + fp), tp / positive.sum()
    envelope = precision.copy()
    for i in range(len(envelope)-2, -1, -1):
        envelope[i] = max(envelope[i], envelope[i+1])
    ap = float(np.dot(tp_groups / positive.sum(), envelope))
    f1 = 2 * tp / (positive.sum() + tp + fp)
    return dict(AP=ap, max_F1=float(f1.max()),
        **{"R@100P": float(recall[fp == 0].max()) if (fp == 0).any() else 0.0,
           "R@1": float(correct[positive].mean())})


def check_ledger(path: Path, reference: Path, hashes: dict, checkpoint_sha: str) -> tuple:
    """질의·정답·과거 후보·공통 지표를 검산한다."""
    digest = sha(path)
    hashes[str(path.relative_to(ROOT))] = digest
    with np.load(reference) as ref, np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        recorded = meta.get("checkpoint_sha256", meta.get("candidate", {}).get("sha256",
            meta.get("artifacts", {}).get("checkpoint_sha256")))
        assert recorded == checkpoint_sha, (path, recorded, checkpoint_sha)
        assert meta["primary_score"] == "score_negl2"
        ix = np.searchsorted(z["query_idx"], ref["query_idx"])
        arrays = {key: z[key][ix] for key in z.files if key != "meta"}
        for key in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(arrays[key], ref[key])
    assert np.all(arrays["j_star"] <= arrays["query_idx"] - 30)
    assert np.all(arrays["n_candidates"] == arrays["query_idx"] - 29)
    assert np.isfinite(arrays["score_negl2"]).all()
    actual, separate = metrics(arrays), independent(arrays)
    err = max((abs(actual[k]-separate[k]) for k in METRICS if actual[k] is not None), default=0.0)
    assert err < 1e-12
    return actual, err, digest


def main() -> None:
    """선택 확정 후에만 시험 원장을 읽고 결과를 모은다."""
    frozen = read(OUT / "selection_freeze.json")
    spec = read(OUT / "protocol.json")
    assert frozen["protocol_sha256"] == sha(OUT / "protocol.json")
    assert frozen["validation_summary_sha256"] == sha(OUT / "validation_summary.json")
    for name, digest in spec["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in frozen["validation_ledger_sha256"].items():
        assert sha(ROOT / name) == digest, name
    for candidates in spec["candidates"].values():
        for candidate in candidates:
            assert sha(ROOT / candidate["checkpoint"]) == candidate["sha256"]
    images = read(OUT / "input_images_audit.json")
    assert images["all_passed"] and len(images["sequences"]) == 9
    for name in ("kitti_09", "kitti360_0004"):
        replay = read(OUT / "val/s1_best" / f"{name}.json")["replay"]
        assert replay["same_winners"] and replay["max_score_error"] == 0.0
    for row in read(OUT / "preflight/batch_preflight.json").values():
        assert row["all_fields_exact"]
    hashes, maximum_error, validation_rows = {}, 0.0, 0
    for seed in (1, 2, 3):
        for tag in ("best", "final"):
            candidate = next(r for r in spec["candidates"][str(seed)] if r["id"] == tag)
            for sensor, seq in VAL:
                p = old_val(seed, sensor, seq) if tag == "best" else OUT / "val" / f"s{seed}_{tag}" / f"dump_bevplace_{sensor}_{seq}.npz"
                actual, error, _ = check_ledger(p, old_val(1, sensor, seq), hashes, candidate["sha256"])
                maximum_error = max(maximum_error, error)
                validation_rows += actual["n_queries"]
    baseline = read(ROOT / "results/pdv2_baseline_seeds/summary.json")
    test, identities, rows = {}, {}, 0
    for seed in (1, 2, 3):
        choices = frozen["selected"][str(seed)]
        test[str(seed)], identities[str(seed)] = {}, {}
        for objective in ("AP", "R@1"):
            tag = choices[objective]["id"]
            suffix = "" if seed == 1 else f"_s{seed}"
            folder = ROOT / f"results/pdv2_bevplace_official{suffix}/odom" if tag == "best" else OUT / "test" / f"s{seed}_{tag}"
            per, ledger_hashes = {}, {}
            for sensor, seq in TEST:
                p = folder / f"dump_bevplace_{sensor}_{seq}.npz"
                # 현재 NSD matching_only와 동일한 질의인지도 직접 비교한다.
                reference = ROOT / f"results/pdv2_diffattn_contrast_test_20261104/test/{sensor}_{seq}/diff_s1.npz"
                actual, error, digest = check_ledger(p, reference, hashes, choices[objective]["sha256"])
                if tag == "best":
                    assert digest == baseline["input_sha256"][str(p.relative_to(ROOT))]
                per[sensor + "/" + seq] = actual
                ledger_hashes[sensor + "/" + seq] = digest
                maximum_error = max(maximum_error, error)
                rows += actual["n_queries"]
            assert sum(r["n_queries"] for r in per.values()) == 195726
            groups = {g: {k: float(np.mean([per[n][k] for n in names if per[n]["n_pos"]])) for k in METRICS} for g, names in CITY.items()}
            macro = {k: float(np.mean([r[k] for r in groups.values()])) for k in METRICS}
            if tag == "best":
                expected = baseline["methods"]["bevplace"]["per_seed"][str(seed)]["macro"]
                for k in METRICS:
                    assert abs(macro[k] - expected[k]) < 1e-12
            test[str(seed)][objective] = dict(candidate=choices[objective], macro=macro, groups=groups, per_sequence=per)
            identities[str(seed)][objective] = ledger_hashes
    mean = {objective: {k: float(np.mean([test[str(s)][objective]["macro"][k] for s in (1,2,3)])) for k in METRICS} for objective in ("AP", "R@1")}
    nsd = read(ROOT / "results/pdv2_diffattn_contrast_test_20261104/summary.json")["mean"]["diff"]
    same = {s: identities[s]["AP"] == identities[s]["R@1"] for s in identities}
    unique_test_hashes = {digest for objectives in identities.values() for ledgers in objectives.values()
                          for digest in ledgers.values()}
    report = dict(created_unix=time.time(), scope=spec["scope"], selected=frozen["selected"],
        validation=read(OUT / "validation_summary.json"), test=test, mean=mean,
        AP_minus_R1_selection_pp={k:100*(mean["AP"][k]-mean["R@1"][k]) for k in METRICS},
        same_test_ledger_by_seed=same, current_nsd_diff=nsd,
        nsd_minus_AP_selected_bev_pp={k:100*(nsd[k]-mean["AP"][k]) for k in METRICS},
        validation_rows_checked=validation_rows, unique_validation_ledgers=54,
        unique_test_ledgers=len(unique_test_hashes),
        test_rows_checked_including_objective_reuse=rows,
        maximum_independent_metric_error=maximum_error, source_ledger_sha256=hashes,
        reporter_sha256=sha(Path(__file__)),
        input_images_audit_sha256=sha(OUT / "input_images_audit.json"),
        checkpoint_model_audit_sha256=sha(OUT / "checkpoint_model_audit.json"),
        exact_best_replays=2, preflight_queries_exact=11250,
        protocol_sha256=sha(OUT / "protocol.json"), selection_freeze_sha256=sha(OUT / "selection_freeze.json"),
        limitations=spec["limitations"], checks_passed=True)
    write(OUT / "report.json", report)
    with (OUT / "validation_candidates_percent.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "candidate", "epoch_label_zero_based", "completed_epochs",
            "original_GT_bidirectional_pooled_R1", *METRICS, "selected_by_AP", "selected_by_R1"])
        for seed, candidates in report["validation"].items():
            for tag, row in candidates.items():
                candidate = row["candidate"]
                writer.writerow([seed, tag, candidate["epoch_zero_based"], candidate["completed_epochs"],
                    100*candidate["original_rule_r1"], *[100*row["macro"][k] for k in METRICS],
                    tag == frozen["selected"][seed]["AP"]["id"], tag == frozen["selected"][seed]["R@1"]["id"]])
    with (OUT / "test_selection_percent.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["objective", "seed", "candidate", *METRICS])
        for seed, choices in test.items():
            for objective, row in choices.items():
                writer.writerow([objective, seed, row["candidate"]["id"], *[100*row["macro"][k] for k in METRICS]])
        for objective, row in mean.items():
            writer.writerow([objective, "mean", "per_seed_choice", *[100*row[k] for k in METRICS]])
    print("mean", mean, "same", same, "error", maximum_error)


if __name__ == "__main__":
    main()
