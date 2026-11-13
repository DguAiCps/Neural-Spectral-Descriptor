#!/usr/bin/env python3
"""NSD 실제 반환 원장을 baseline의 5m R@1 코드로 독립 검산한다."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.eval_utils import _find_revisit_queries, _score_recalls_from_ranked

OUT = ROOT / "results/pdv2_review_nsd_r1_20261106"


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    """파일 SHA256을 계산한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """원시 정답 자세·반환 장소·baseline 코드부터 선택을 재구성한다."""
    spec, selected = read(OUT / "protocol.json"), read(OUT / "selection.json")
    summary = read(OUT / "validation_summary.json")
    assert selected["protocol_sha256"] == sha(OUT / "protocol.json")
    assert not selected["test_used_for_selection"]
    groups = {tag: {} for tag in spec["tie_order"]}
    rows, count, reproduction, differences = {}, 0, [], {tag: 0 for tag in groups}
    for sequence in spec["validation"]:
        home = OUT / "val" / sequence.replace("/", "_")
        original = read(home / f"{spec['original_AP_choice']}_s1.json")
        cache_path = next(path for path in original["input_sha256"] if "preprocessed_v2_odom/cache" in path)
        assert sha(ROOT / cache_path) == original["input_sha256"][cache_path]
        with np.load(ROOT / cache_path) as cache:
            positions = cache["poses_gt"][:, :3, 3].astype(np.float64)
        queries = np.arange(30, len(positions)-15)
        revisit = [(q, j) for q, j in _find_revisit_queries(positions, 5, 30) if q < len(positions)-15]
        positives = np.isin(queries, [q for q, _ in revisit])
        with np.load(ROOT / original["ledger"]) as z:
            original_winners = z["j_star"].copy()
        for tag in spec["tie_order"]:
            path = home / f"{tag}_s1.json"
            record = read(path)
            assert sha(path) == selected["validation_record_sha256"][str(path.relative_to(ROOT))]
            assert sha(ROOT / record["ledger"]) == record["ledger_sha256"]
            with np.load(ROOT / record["ledger"]) as z:
                np.testing.assert_array_equal(z["query_idx"], queries)
                np.testing.assert_array_equal(z["min_past_gt_dist"] <= 5, positives)
                winners = z["j_star"].copy()
                assert ((winners >= 0) & (winners <= queries-30)).all()
                distance = np.linalg.norm(positions[queries]-positions[winners], axis=1)
                np.testing.assert_allclose(distance, z["top1_geo_dist"], rtol=0, atol=1e-10)
            hits = int(((distance < 5) & positives).sum())
            npositive = int(positives.sum())
            ranked = [np.asarray([winners[q-30]], np.int64) for q, _ in revisit]
            baseline = _score_recalls_from_ranked(revisit, ranked, positions, [1], 5)[1]
            assert baseline == (hits/npositive if npositive else 0)
            assert record["baseline_R1_counts"]["hits"] == hits
            assert record["baseline_R1_counts"]["positives"] == npositive
            sensor = sequence.split("/")[0]
            family = "kitti" if sensor.startswith("kitti") else sensor
            group = groups[tag].setdefault(family, dict(hits=0, positives=0))
            group["hits"] += hits
            group["positives"] += npositive
            differences[tag] += int((winners != original_winners).sum())
            rows[str(path.relative_to(ROOT))] = dict(hits=hits, positives=npositive,
                baseline_R1=baseline, ledger_sha256=record["ledger_sha256"])
            count += len(queries)
            if tag == spec["original_AP_choice"]:
                assert record["original_reproduction"]["winners_exact"]
                reproduction.append(record["original_reproduction"]["max_score_error"])
    exact = {tag: sum((Fraction(g["hits"], g["positives"]) for g in families.values()), Fraction())/3
        for tag, families in groups.items()}
    choice = max(spec["tie_order"], key=lambda tag: exact[tag])
    assert choice == selected["R1_selected"] == spec["original_AP_choice"]
    current = read(ROOT / "results/pdv2_diffattn_contrast_test_20261104/protocol.json")
    weights = {}
    for seed in (1, 2, 3):
        candidate = spec["candidates"][choice][str(seed)]
        old = current["cases"][f"diff_s{seed}"]
        actual = sha(ROOT / candidate["path"])
        assert actual == candidate["sha256"] == old["checkpoint_sha256"]
        weights[str(seed)] = dict(path=candidate["path"], sha256=actual)
    result = dict(status="passed", completed_unix=time.time(), checked_ledgers=len(rows), evaluated_rows=count,
        primary_R1={tag: float(value) for tag, value in exact.items()}, family_counts=groups,
        independent_baseline_function="baselines.eval_utils._find_revisit_queries/_score_recalls_from_ranked",
        exact_fraction={tag: str(value) for tag, value in exact.items()}, rows=rows,
        winner_differences_from_original=differences, selected=choice, same_as_original=True,
        original_model_validation_max_score_error=max(reproduction), same_three_checkpoint_hashes=weights,
        new_test_needed=False, new_training_needed=False, auxiliary_AP_not_used_for_claim=True)
    (OUT / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ["status", "checked_ledgers", "evaluated_rows", "primary_R1", "winner_differences_from_original", "selected"]}))


if __name__ == "__main__":
    main()
