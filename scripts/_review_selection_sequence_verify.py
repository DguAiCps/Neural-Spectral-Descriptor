#!/usr/bin/env python3
"""완성된 시험 한 시퀀스의 모든 동결 선택 모델을 독립 검산한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "src")]
from _review_selection_bev import METRICS, read, sha, write
from _review_selection_full_eval import OUT, SCORES, folder, positions, query_base
from _review_selection_full_finish import evaluation_choices
from _review_selection_full_verify import verify_input_provenance
from _review_selection_report import independent


def verify(method: str, sequence: str) -> None:
    """입력 근거·가중치·공통질의·기하·별도 지표 계산을 모두 확인한다."""
    selection_path = OUT / "retained" / f"selection_{method}.json"
    frozen = read(selection_path)
    assert not frozen["test_used_for_selection"]
    sensor, seq = sequence.split("/")
    paths = []
    for seed in frozen["selected"]:
        choices = evaluation_choices(method, seed, "retained", frozen)
        seen = set()
        for candidate in choices.values():
            if candidate["id"] in seen:
                continue
            seen.add(candidate["id"])
            path = folder(method, int(seed), candidate["id"], "retained", "test") / f"{sensor}_{seq}.json"
            if not path.exists():
                raise SystemExit(76)
            paths.append((path, candidate))
    base = query_base(sensor, seq, "test")
    geometry = positions(sensor, seq, "test")[1]
    reference = ROOT / f"results/pdv2_diffattn_contrast_test_20261104/test/{sensor}_{seq}/diff_s1.npz"
    if not reference.exists():
        manifest = read(OUT / "test_query_reference/manifest.json")
        case = manifest["sequences"][sequence]
        assert case["source"] == str(reference.relative_to(ROOT)) and manifest["all_arrays_exact"]
        reference = ROOT / case["copy"]
        assert sha(reference) == case["copy_sha256"]
    with np.load(reference) as old:
        for field in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(base[field], old[field])
    rows, errors = [], []
    for path, candidate in paths:
        record = read(path)
        assert record["candidate"] == candidate
        assert sha(ROOT / candidate["path"]) == candidate["sha256"]
        ledger = ROOT / record["ledger"]
        assert sha(ledger) == record["ledger_sha256"]
        with np.load(ledger) as old:
            meta = json.loads(str(old["meta"]))
            arrays = {k: old[k] for k in old.files if k != "meta"}
        assert meta["candidate"] == candidate and meta["protocol_sha256"] == frozen["protocol_sha256"]
        for field, value in base.items():
            np.testing.assert_array_equal(arrays[field], value)
        q, j = arrays["query_idx"], arrays["j_star"]
        assert ((j >= 0) & (j <= q - 30)).all()
        np.testing.assert_allclose(arrays["top1_geo_dist"],
            np.linalg.norm(geometry[j] - geometry[q], axis=1), rtol=0, atol=1e-10)
        computed = independent(dict(arrays, score_negl2=arrays[SCORES[method]]))
        for metric in METRICS:
            if computed[metric] is None:
                assert record["metrics"][metric] is None
            else:
                errors.append(abs(computed[metric] - record["metrics"][metric]))
        input_ref = record.get("audit", {}).get("input_prevalidation")
        provenance = None
        if method == "imlpr" and input_ref:
            source = ROOT / input_ref["record"]
            assert sha(source) == input_ref["sha256"]
            provenance = verify_input_provenance(source, sha(selection_path), sensor, seq)
        rows.append(dict(record=str(path.relative_to(ROOT)), record_sha256=sha(path),
            ledger=record["ledger"], ledger_sha256=record["ledger_sha256"],
            input_provenance=provenance, checkpoint_sha256=candidate["sha256"]))
    assert max(errors, default=0) < 1e-12
    result = dict(status="complete_for_this_sequence", completed_unix=time.time(),
        method=method, sequence=sequence, unique_models=len(rows), n_queries=len(base["query_idx"]),
        selection_sha256=sha(selection_path), records=rows,
        independent_metric_max_error=max(errors, default=0), same_current_NSD_queries=True,
        verified_all_winner_geometry=True, full_method_complete=False, test_used_for_selection=False)
    write(OUT / "sequence_verification" / f"{method}_{sensor}_{seq}.json", result)
    print(method, sequence, len(rows), result["independent_metric_max_error"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=SCORES, required=True)
    parser.add_argument("--sequence", required=True)
    args = parser.parse_args()
    verify(args.method, args.sequence)
