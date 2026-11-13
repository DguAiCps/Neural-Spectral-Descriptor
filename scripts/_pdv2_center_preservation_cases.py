#!/usr/bin/env python3
"""중심 보존 재학습이 후보 누락·융합 역전과 기존 손실 사례를 고치는지 확인한다."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_center_preservation as run
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_temporal_test_control_finish import scalar_score

CATEGORIES = ("positive_absent", "present_key_prefers_positive", "present_key_not_prefer_positive")


def main() -> None:
    """표준+tanh의 정답 중 각 조건이 놓친 사례를 같은 규칙으로 재현한다."""
    spec = run.protocol(True)
    summary = run.read(run.OUT/"summary.json")
    audit = run.read(run.OUT/"completion_validation.json")
    for p, expected in audit["sources"].items():
        assert sha256(REPO/p) == expected, p
    epoch_ref = summary["references"]["standard_tanh"]["selected"]["epoch"]
    conf = spec["confidence"]["1"]
    weights = np.asarray(conf["tuple"], dtype=np.float32)
    records = {arm: [] for arm in run.ARMS}
    all_rows = {arm: {} for arm in run.ARMS}
    checks, score_error = 0, 0.
    for sequence in spec["val"]:
        name = sequence.replace("/", "_")+".npz"
        ref = run.base.exp.arrays(run.previous.OUT/f"s1/standard_tanh/epoch{epoch_ref}"/name)
        _, _, paths, _ = run.base.exp.old.base.inputs("val", sequence)
        cache = run.base.exp.arrays(paths["cache"])
        with np.load(paths["spec"]) as z:
            bank = run.base.exp.old.base.bank_for(run.base.exp.OP, z)
        xyz = cache["poses_gt"][:, :3, 3]
        positions = cache["poses"][:, :3, 3]
        travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
        for arm in run.ARMS:
            epoch = summary["per_seed"]["1"][arm]["selected"]["epoch"]
            folder = run.base.root(1, arm)/f"epoch{epoch}"
            row = run.base.exp.arrays(folder/name)
            key = run.base.exp.arrays(folder/("keys_"+name))["embedding"]
            coverage = run.base.exp.arrays(folder/("coverage_"+name))
            all_rows[arm][sequence] = row
            np.testing.assert_array_equal(row["query_idx"], ref["query_idx"])
            lost = (ref["min_past_gt_dist"] <= 5) & (ref["top1_geo_dist"] <= 5) & (row["top1_geo_dist"] > 5)
            for ix in np.flatnonzero(lost):
                q, p, n = int(row["query_idx"][ix]), int(ref["j_star"][ix]), int(row["j_star"][ix])
                a = run.base.prefix_topk(key, q, q-29, 100)
                b = run.base.prefix_topk(bank[0], q, q-29, 100)
                ids = np.unique(np.r_[a, b])
                parts = np.stack([run.base.minmax(key[ids]@key[q]), run.base.minmax(bank[0][ids]@bank[0][q]),
                    run.base.minmax(run.base.align_scores(bank[1], bank[2], q, ids)),
                    run.base.minmax(run.base.column_align_scores(bank[3], q, ids))])
                fused = weights[0]*parts[0]+weights[1]*parts[1]+weights[2]*parts[2]+weights[3]*parts[3]
                wi = int(np.argsort(-fused, kind="stable")[0])
                assert int(ids[wi]) == n
                assert len(ids) == coverage["n_candidates"][ix]
                hit = bool((np.linalg.norm(xyz[ids]-xyz[q], axis=1) <= 5).any())
                assert hit == coverage["union_hit"][ix]
                present = bool(p in ids)
                key_margin = float(key[q]@key[p]-key[q]@key[n])
                if present:
                    pi = int(np.flatnonzero(ids == p)[0])
                    category = "present_key_prefers_positive" if key_margin > 0 else "present_key_not_prefer_positive"
                    contributions = (weights*(parts[:, pi]-parts[:, wi])).tolist()
                    fused_margin = float(fused[pi]-fused[wi])
                    assert fused_margin <= 0
                    assert abs(sum(contributions)-fused_margin) < 1e-6
                else:
                    category, contributions, fused_margin = "positive_absent", [], None
                score = scalar_score(bank, travel, q, n, conf["decision"], float(row["raw"][ix]))
                score_error = max(score_error, abs(score-row[run.METHOD][ix]))
                assert abs(score-row[run.METHOD][ix]) < 1e-12
                records[arm].append({"sequence": sequence, "q": q, "p": p, "n": n,
                    "category": category, "all_positives_absent": not hit, "key_margin": key_margin,
                    "weighted_channel_margins": contributions, "fused_margin": fused_margin})
                checks += 1
    reports = {arm: {"n": len(values), "records": values,
        "counts": {k: sum(x["category"] == k for x in values) for k in CATEGORIES},
        "all_positives_absent": sum(x["all_positives_absent"] for x in values),
        "families": {f: {k: sum(x["category"] == k and x["sequence"].startswith(f) for x in values)
                          for k in CATEGORIES} for f in ("kitti", "helipr", "mulran")}}
               for arm, values in records.items()}
    old_cases = run.read(REPO/"results/pdv2_center_information_20261021/case_pipeline.json")
    assert reports["center_zero"]["counts"] == old_cases["conditions"]["diff_tanh"]["counts"]
    assert reports["center_zero"]["n"] == 91
    recovery = {}
    for arm in run.CFG["train_arms"]:
        groups = {k: {"n": 0, "recovered": 0, "still_wrong": 0} for k in CATEGORIES}
        for case in records["center_zero"]:
            r = all_rows[arm][case["sequence"]]
            ix = np.searchsorted(r["query_idx"], case["q"])
            assert r["query_idx"][ix] == case["q"]
            correct = bool(r["top1_geo_dist"][ix] <= 5)
            group = groups[case["category"]]
            group["n"] += 1
            group["recovered" if correct else "still_wrong"] += 1
        recovered = sum(x["recovered"] for x in groups.values())
        recovery[arm] = {"original_lost": 91, "recovered": recovered, "remaining": 91-recovered,
                        "new_losses_vs_standard_tanh_outside_original91": reports[arm]["n"]-(91-recovered),
                        "by_original_category": groups}
        assert recovery[arm]["new_losses_vs_standard_tanh_outside_original91"] >= 0
    write_json(run.OUT/"case_diagnostics.json", {"conditions": reports, "fixed91_recovery": recovery,
        "winner_replay_checks": checks, "max_final_score_error": score_error, "test_used": False,
        "scope": "각조건vs표준_tanh의최종정답손실과기존91개회복. 다른오류를함께보고해사례선별이득과구분. 전체AP원인비율이아님.",
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in (Path(__file__), run.OUT/"summary.json",
            run.OUT/"completion_validation.json", REPO/"results/pdv2_center_information_20261021/case_pipeline.json")}})
    print("CASES", {a: {k:v for k,v in r.items() if k not in ("records", "families")} for a,r in reports.items()}, flush=True)
    print("RECOVERY", recovery, checks, score_error, flush=True)


if __name__ == "__main__":
    main()
