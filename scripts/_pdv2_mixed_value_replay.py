#!/usr/bin/env python3
"""기준 재실행의 한 후보 경계 차이를 보존하고 그 영향 범위를 검산한다."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_mixed_value as run
from _pdv2_no_context_ablation import sha256, write_json


def diagnose() -> dict:
    """관측된 한 예외만 허용하며 다른 차이나 더 큰 영향은 실패시킨다."""
    spec = run.protocol(True)
    summary = run.read(run.OUT / "summary.json")
    old_summary = run.read(run.previous.OUT / "summary.json")
    changes, exact = [], 0
    for arm in (*run.CFG["reused_arms"], "point"):
        for epoch in (10, 20):
            for seq in spec["val"]:
                name = seq.replace("/", "_") + ".npz"
                current = run.base.root(1, arm) / f"epoch{epoch}" / name
                reference = run.previous.OUT / "s1" / arm / f"epoch{epoch}" / name
                a, b = run.base.exp.arrays(current), run.base.exp.arrays(reference)
                changed = {f: np.flatnonzero(a[f] != b[f]).tolist() for f in a
                           if not np.array_equal(a[f], b[f])}
                if not changed:
                    exact += 1
                    continue
                # 이번 실행에서 발견한 한 사례를 명시한다. 허용오차로 승자 차이를 숨기지 않는다.
                assert (arm, epoch, seq) == ("diff_tanh", 20, "helipr/Roundabout03")
                assert set(changed) == {"j_star", "top1_geo_dist", "raw", run.METHOD}
                assert all(indices == [557] for indices in changed.values())
                assert a["query_idx"][557] == b["query_idx"][557] == 587
                assert a["min_past_gt_dist"][557] > 25 and b["min_past_gt_dist"][557] > 25
                assert (a["j_star"][557], b["j_star"][557]) == (448, 451)
                _, _, paths, _ = run.base.exp.old.base.inputs("val", seq)
                with np.load(paths["spec"]) as z:
                    bank = run.base.exp.old.base.bank_for(run.base.exp.OP, z)
                conf = spec["confidence"]["1"]
                observations, keys = {}, {}
                for label, path, ledger in (("current", current, a), ("reference", reference, b)):
                    kp = path.with_name("keys_" + path.name)
                    key = run.base.exp.arrays(kp)["embedding"]
                    keys[label] = key
                    q = 587
                    key_ids = run.base.prefix_topk(key, q, q - 29, 100)
                    scores = key[:q - 29] @ key[q]
                    # 두 후보는 100번째 경계에 있고 점수 차이는 float32 오차 규모다.
                    boundary = float(np.sort(scores)[-100])
                    assert max(abs(float(scores[j]) - boundary) for j in (448, 451)) < 1e-6
                    ids = np.unique(np.r_[key_ids, run.base.prefix_topk(bank[0], q, q - 29, 100)])
                    channels = (key[ids] @ key[q], bank[0][ids] @ bank[0][q],
                        run.base.align_scores(bank[1], bank[2], q, ids),
                        run.base.column_align_scores(bank[3], q, ids))
                    fused = sum(float(w) * run.base.minmax(v) for w, v in zip(conf["tuple"], channels))
                    order = np.argsort(-fused, kind="stable")
                    assert ids[order[0]] == ledger["j_star"][557]
                    observations[label] = {
                        "ledger_sha256": sha256(path), "keys_sha256": sha256(kp),
                        "row": {f: v[557].item() for f, v in ledger.items()},
                        "candidate_ids": ids.tolist(), "key_boundary_score": boundary,
                        "key_scores_448_451": [float(scores[j]) for j in (448, 451)],
                        "key_last_ids": key_ids[-3:].tolist(),
                        "top2_fused": [{"id": int(ids[j]), "score": float(fused[j])} for j in order[:2]],
                    }
                new_ids, old_ids = (set(observations[k]["candidate_ids"]) for k in ("current", "reference"))
                assert new_ids - old_ids == {448} and old_ids - new_ids == {451}
                np.testing.assert_allclose(keys["current"], keys["reference"], atol=4e-6, rtol=4e-5)
                changes.append({"arm": arm, "epoch": epoch, "sequence": seq, "fields": changed,
                    "observations": observations,
                    "max_key_error": float(np.max(abs(keys["current"] - keys["reference"])))})
    assert exact == 89 and len(changes) == 1
    metric_deltas = {}
    for arm in (*run.CFG["reused_arms"], "point"):
        current, reference = summary["per_seed"]["1"][arm], old_summary["per_seed"]["1"][arm]
        assert current["selected"]["epoch"] == reference["selected"]["epoch"]
        if arm != "diff_tanh":
            assert current == reference
        metric_deltas[arm] = {}
        for a, b in zip(current["candidates"], reference["candidates"]):
            assert a["epoch"] == b["epoch"]
            metric_deltas[arm][str(a["epoch"])] = {}
            for method in ("raw", run.METHOD):
                delta = {m: 100 * (a[method]["macro"][m] - b[method]["macro"][m])
                         for m in run.base.exp.METRICS}
                assert max(abs(v) for v in delta.values()) < .0001
                metric_deltas[arm][str(a["epoch"])][method] = delta
    promotion = {}
    selected = summary["per_seed"]["1"]["diff_mixed"]["selected"][run.METHOD]
    for name in run.CFG["promotion"]["comparators"]:
        ref = "standard_mixed" if name == "matched_standard" else name
        source = old_summary if ref in old_summary["per_seed"]["1"] else summary
        other = source["per_seed"]["1"][ref]["selected"][run.METHOD]
        delta = {m: selected["macro"][m] - other["macro"][m] for m in run.base.exp.METRICS}
        rules = run.CFG["promotion"]
        promotion[ref] = {
            "AP": delta["AP"] >= rules["ap_gain_min"],
            "F1": delta["max_F1"] >= -rules["f1_loss_max"],
            "R100": delta["R@100P"] >= -rules["r100_loss_max"],
            "family_AP": all(selected["families"][f]["AP"] >=
                other["families"][f]["AP"] - rules["family_ap_loss_max"] for f in other["families"]),
        }
    assert promotion == summary["promotion_checks"]["diff_mixed"]
    report = {"status": "passed_with_one_documented_candidate_boundary_difference",
        "exact_reference_ledgers": exact, "changed_reference_ledgers": len(changes),
        "changes": changes, "macro_metric_delta_pp": metric_deltas,
        "summary_sha256": sha256(run.OUT / "summary.json"),
        "reference_summary_sha256": sha256(run.previous.OUT / "summary.json"),
        "runtime_sha256": sha256(Path(__file__)), "test_used": False,
        "promotion_identical_with_original_reference_metrics": True,
        "scope": "새 실행의 검산이다. 기존·신규 원장은 바꾸지 않는다. 후보100 경계의 미세차이에 따른 한 건을 명시하며 모든 최종 지표는 재평가 원장으로 계산한다."}
    write_json(run.OUT / "reference_replay_differences.json", report)
    return report


if __name__ == "__main__":
    result = diagnose()
    print(result["status"], result["exact_reference_ledgers"], result["changed_reference_ledgers"])
