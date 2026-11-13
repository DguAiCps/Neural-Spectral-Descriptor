#!/usr/bin/env python3
"""SeqOT 표본 복구 대조의 쌍·학습 신호·최종 검증 원장을 독립 검산한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/pdv2_seqot_ring_repair_20261106/seqot"


def read(path: Path) -> dict:
    """기록을 읽는다."""
    return json.loads(path.read_text())


def main() -> None:
    """시험셋 없이 학습 쌍과 전체 검증 지표를 다시 계산한다."""
    import _seqot_official_prep as p
    from _pr_eval_lib import classify_queries, pr_sweep

    plan = read(OUT / "plan.json")
    for path, expected in plan["sources_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    poses, positives = [], {}
    for d, (sensor, seq) in enumerate(plan["train_sequences"]):
        with np.load(p.label_dir(sensor, seq) / "frames.npz") as z:
            poses.append(z["poses"][:, :3, 3])
        i, j, overlap = p.load_explicit(sensor, seq, plan.get("overlap_formula", "code"))
        positives[d] = set(zip(i[overlap > .3].tolist(), j[overlap > .3].tolist()))
        count = len(poses[d])
        pair_keys = i * count + j
        for dd, a, _, neg in plan["repair"]:
            if dd != d:
                continue
            query = (a // 5) * count + np.asarray(neg) // 5
            slots = np.searchsorted(pair_keys, query)
            exists = slots < len(pair_keys)
            exists[exists] &= pair_keys[slots[exists]] == query[exists]
            assert np.all(overlap[slots[exists]] < .3)
    geometry = {}
    for arm in ("control", "repair"):
        distances, gaps = [], []
        for d, a, pos, neg in plan[arm]:
            for b in pos:
                assert (a // 5, b // 5) in positives[d]
                if arm == "repair":
                    assert a != b
                distances.append(float(np.linalg.norm(poses[d][a // 5] - poses[d][b // 5])))
                gaps.append(abs(a - b))
        ds, gs = np.array(distances), np.array(gaps)
        geometry[arm] = {"pairs": len(ds), "distance_quantiles_m": np.quantile(ds, [0, .25, .5, .75, .95, 1]).tolist(),
                         "within5m": float(np.mean(ds <= 5)), "raw_gap_le20": float(np.mean(gs <= 20)),
                         "raw_gap_gt1000": float(np.mean(gs > 1000)), "self_pairs": int(np.sum(gs == 0))}
    assert all(c[:2] == r[:2] and c[3] == r[3] and len(c[2]) == len(r[2])
               for c, r in zip(plan["control"], plan["repair"]))
    (OUT / "plan_verification.json").write_text(json.dumps({"passed": True, "geometry": geometry}, indent=2) + "\n")
    if not all((OUT / arm / "report.json").exists() for arm in ("control", "repair")):
        print(json.dumps({"plan_passed": True, "geometry": geometry, "evaluation": "pending"}), flush=True)
        return
    reports = {arm: read(OUT / arm / "report.json") for arm in ("control", "repair")}
    checked, independent = [], {}
    keys = ("AP", "max_F1", "R@100P", "R@1")
    family = lambda name: "kitti" if name.startswith("kitti") else name.split("/")[0]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for arm in reports:
        actual_rows = {}
        protocol = read(OUT / arm / "protocol.json")
        assert protocol["plan_sha256"] == sha(OUT / "plan.json")
        assert protocol["source_sha256"] == sha(ROOT / "scripts/_review_seqot_sampler_repair.py")
        if "wrapper_source_sha256" in plan:
            assert plan["wrapper_source_sha256"] == sha(ROOT / "scripts/_review_seqot_paper_sampler.py")
        checkpoint_hash = sha(OUT / arm / "model.pth.tar")
        count_queries = 0
        for seq, reference in reports[arm]["sequences"].items():
            path = OUT / arm / ("val_" + seq.replace("/", "_") + ".npz")
            metadata = read(path.with_suffix(".json"))
            assert metadata["ledger_sha256"] == sha(path)
            assert metadata["checkpoint_sha256"] == checkpoint_hash
            with np.load(path) as z:
                q = classify_queries(z["min_past_gt_dist"])
                sweep = pr_sweep(z["score_disco"], z["top1_geo_dist"], q)
                if q["n_pos"]:
                    measured = {k: float(sweep[k]) for k in ("AP", "max_F1", "R@100P")}
                    measured["R@1"] = float((z["top1_geo_dist"][q["positive"]] <= 5).mean())
                    assert all(abs(reference[k] - v) < 1e-12 for k, v in measured.items())
                    actual_rows[seq] = measured
                assert len(z["query_idx"]) == reference["n_queries"]
                count_queries += len(z["query_idx"])
                assert np.all(z["j_star"] <= z["query_idx"] - 30)
            checked.append(str(path.relative_to(ROOT)))
        assert count_queries == 99236
        independent[arm] = {k: float(np.mean([
            np.mean([m[k] for name, m in actual_rows.items() if family(name) == f])
            for f in ("kitti", "helipr", "mulran")])) for k in keys}
        assert all(abs(independent[arm][k] - reports[arm]["macro"][k]) < 1e-12 for k in keys)
    macro = independent
    baseline = read(ROOT / "results/pdv2_baseline_common_diagnosis_20261106/yaw_pilot/verification.json")["original_validation"]
    result = {"passed": True, "ledgers": len(checked), "queries_per_arm": 99236,
              "plan_geometry": geometry, "selected_original": baseline, "macro": macro,
              "negative_labels_checked": True, "aggregate_recomputed": True,
              "ledger_and_checkpoint_hashes_checked": True,
              "repair_minus_control_pp": {k: 100 * (macro["repair"][k] - macro["control"][k]) for k in keys},
              "repair_minus_original_pp": {k: 100 * (macro["repair"][k] - baseline[k]) for k in keys},
              "training_signal": {arm: read(OUT / arm / "train.json") for arm in reports},
              "scope": "단일시드·같은4096기준프레임 대조. 유효양성이 없는 기준프레임은 두 조건에서 제외. 최종검증이지 시험결과가 아님.",
              "test_used": False, "adopted_main": False}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()
    if args.paper:
        OUT = OUT.with_name("seqot_paper")
    main()
