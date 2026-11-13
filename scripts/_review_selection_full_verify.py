#!/usr/bin/env python3
"""완료된 선택 대조의 동결·질의·기하·집계·원장 해시를 별도로 검산한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "src")]
from _review_selection_bev import CITY, METRICS, TEST, VAL, read, sha, write
from _review_selection_full_eval import OUT, SCORES, folder, query_base, positions
from _review_selection_report import independent
from _review_selection_full_inputs import cache


def verify_input_provenance(path: Path, selection_sha256: str, sensor: str, seq: str) -> dict:
    """임시 입력 정리 뒤에도 남는 전수 해시·원시 규약·선택동결 근거를 검산한다."""
    proof = read(path)
    assert proof["selection_sha256"] == selection_sha256
    assert proof.get("downloaded_RIV_all_files_SHA_exact", False) or proof.get("generated_RIV_all_files_SHA_exact", False)
    raw = proof["rawpack"]
    assert raw["dtype"] == "float32" and raw["channels"] == 4
    raw_digest = hashlib.sha256((json.dumps(raw, ensure_ascii=False, indent=2) + "\n").encode()).hexdigest()
    assert proof["rawpack_verification"]["all_arrays_exact"]
    assert proof["rawpack_verification"]["done_sha256"] == raw_digest
    hashes = proof["generated_RIV_file_sha256"]
    assert set(hashes) == {f"{i:06d}.npy" for i in range(proof["done"]["n_frames"])}
    digest = hashlib.sha256((json.dumps(hashes, ensure_ascii=False, indent=2) + "\n").encode()).hexdigest()
    assert digest == proof["done"]["generated_hash_manifest_sha256"]
    assert raw["cache_sha256"] == proof["done"]["cache_sha256"] == sha(cache(sensor, seq, "test"))
    original = OUT / "server_riv_smoke/validated.json"
    assert proof["original_RIV_equivalence_proof_sha256"] == sha(original)
    assert read(original)["all_arrays_exact"]
    if proof.get("execution_origin") == "local_CPU_no_network_transfer":
        local = OUT / "local_riv_smoke_verified.json"
        assert proof["local_parallel_smoke_sha256"] == sha(local)
        assert read(local)["all_files_SHA_exact"]
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                frames=proof["done"]["n_frames"], origin=proof.get("execution_origin", "server_CPU_download"))


def verify(method: str, validation_only: bool = False) -> None:
    """시험 보고서가 완성된 방법만 읽어 독립 지표와 근거를 확인한다."""
    start = time.time()
    home = OUT / "retained"
    report_path = home / f"report_{method}.json"
    report = None if validation_only else read(report_path)
    selection_path = home / f"selection_{method}.json"
    frozen = read(selection_path)
    if report is not None:
        assert report["selection"] == frozen
        partition = report.get("test_execution_partition")
        if partition:
            plan_path = ROOT / partition["path"]
            assert sha(plan_path) == partition["sha256"]
            plan = read(plan_path)
            assert plan["selection_sha256"] == sha(selection_path)
            assert plan["same_seed_choices_use_same_device"]
            assert not plan["test_values_read_for_assignment"]
    assert not frozen["test_used_for_selection"]
    assert frozen["protocol_sha256"] == sha(OUT / "retained_protocol.json")
    if "validation_execution_partition_sha256" in frozen:
        assert frozen["validation_execution_partition_sha256"] == sha(OUT / "local_helios_plan.json")
        assert frozen["validation_numerical_guard_sha256"] == sha(OUT / "helios_numerical_margin_verified.json")
    validation_path = home / f"validation_{method}.json"
    assert frozen["validation_sha256"] == sha(validation_path)
    validation = read(validation_path)
    protocol = read(OUT / "retained_protocol.json")
    bases, geometry, checked, errors = {}, {}, {}, []
    input_provenance = {}
    counts = dict(validation_candidates=0, validation_ledgers=0,
                  test_unique_models=0, test_ledgers=0)

    def inspect(path: Path, row: dict, role: str, sensor: str, seq: str) -> dict:
        record = read(path)
        assert record["candidate"] == row
        input_record = record.get("audit", {}).get("input_prevalidation")
        if method == "imlpr" and role == "test" and input_record:
            source = ROOT / input_record["record"]
            assert sha(source) == input_record["sha256"]
            input_provenance[sensor + "/" + seq] = verify_input_provenance(
                source, sha(selection_path), sensor, seq)
        ledger = ROOT / record["ledger"]
        digest = sha(ledger)
        assert digest == record["ledger_sha256"]
        if role == "val":
            assert frozen["validation_ledger_sha256"][record["ledger"]] == digest
        with np.load(ledger) as z:
            meta = json.loads(str(z["meta"]))
            assert meta["candidate"] == row
            assert meta["method"] == method and meta["role"] == role
            assert meta["protocol_sha256"] == frozen["protocol_sha256"]
            arrays = {k: z[k] for k in z.files if k != "meta"}
        key = (role, sensor, seq)
        if key not in bases:
            bases[key] = query_base(sensor, seq, role)
            geometry[key] = positions(sensor, seq, role)[1]
            if role == "test":
                ref = ROOT / ("results/pdv2_diffattn_contrast_test_20261104/test/"
                              f"{sensor}_{seq}/diff_s1.npz")
                if not ref.exists():
                    manifest = read(OUT / "test_query_reference/manifest.json")
                    case = manifest["sequences"][sensor + "/" + seq]
                    assert case["source"] == str(ref.relative_to(ROOT))
                    assert manifest["all_arrays_exact"]
                    ref = ROOT / case["copy"]
                    assert sha(ref) == case["copy_sha256"]
                with np.load(ref) as z:
                    for field in ("query_idx", "min_past_gt_dist"):
                        np.testing.assert_array_equal(bases[key][field], z[field])
        for field, value in bases[key].items():
            np.testing.assert_array_equal(arrays[field], value)
        q, winner = arrays["query_idx"], arrays["j_star"]
        assert ((winner >= 0) & (winner <= q - 30)).all()
        distances = np.linalg.norm(geometry[key][winner] - geometry[key][q], axis=1)
        np.testing.assert_allclose(arrays["top1_geo_dist"], distances, rtol=0, atol=1e-10)
        score = arrays[SCORES[method]]
        assert np.isfinite(score).all()
        computed = independent(dict(arrays, score_negl2=score))
        for metric in METRICS:
            if computed[metric] is None:
                assert record["metrics"][metric] is None
            else:
                errors.append(abs(computed[metric] - record["metrics"][metric]))
        assert max(errors, default=0) < 1e-12
        checked[str(ledger.relative_to(ROOT))] = digest
        return dict(computed, n_queries=len(q), n_pos=int((arrays["min_past_gt_dist"] <= 5).sum()))

    for seed, entry in protocol["methods"][method].items():
        for row in entry["candidates"]:
            assert sha(ROOT / row["path"]) == row["sha256"]
            counts["validation_candidates"] += 1
            per = {}
            for sensor, seq in VAL:
                path = folder(method, int(seed), row["id"], "retained", "val") / f"{sensor}_{seq}.json"
                if "numerical_margin_override_sha256" in frozen:
                    override_path = OUT / f"{method}_numerical_override.json"
                    override = read(override_path)
                    assert sha(override_path) == frozen["numerical_margin_override_sha256"]
                    name = override.get("records", {}).get(seed, {}).get(row["id"], {}).get(sensor + "/" + seq)
                    if name:
                        path = ROOT / name
                per[sensor + "/" + seq] = inspect(path, row, "val", sensor, seq)
                counts["validation_ledgers"] += 1
            assert sum(v["n_queries"] for v in per.values()) == 99236
            for metric in METRICS:
                family = [np.mean([r[metric] for name, r in per.items()
                                   if name.startswith(group) and r["n_pos"]])
                          for group in ("kitti", "helipr", "mulran")]
                errors.append(abs(float(np.mean(family)) - validation[seed][row["id"]]["macro"][metric]))
        ordered = sorted(entry["candidates"], key=lambda r: (r["epoch"], r["id"]))
        for objective in ("AP", "R@1"):
            best = max(ordered, key=lambda r: validation[seed][r["id"]]["macro"][objective])
            assert best == frozen["selected"][seed][objective]

    if validation_only:
        assert max(errors, default=0) < 1e-12
        write(OUT / f"validation_independent_verification_{method}.json", dict(
            status="complete_for_validation", completed_unix=time.time(),
            seconds=time.time() - start, method=method, counts=counts,
            selection_sha256=sha(selection_path), validation_sha256=sha(validation_path),
            independent_metric_max_error=max(errors, default=0),
            verified_top1_geometric_distance=True, verified_tie_rule=True,
            test_read=False, ledger_sha256=checked))
        print(method, counts, "validation maximum_metric_error", max(errors, default=0), flush=True)
        return

    for seed, choices in report["test"].items():
        seen = {}
        for objective, result in choices.items():
            row = result["candidate"]
            if row["id"] in seen:
                assert report["test_ledger_hashes"][seed][objective] == report["test_ledger_hashes"][seed][seen[row["id"]]]
                continue
            seen[row["id"]] = objective
            counts["test_unique_models"] += 1
            per = {}
            for sensor, seq in TEST:
                path = folder(method, int(seed), row["id"], "retained", "test") / f"{sensor}_{seq}.json"
                per[sensor + "/" + seq] = inspect(path, row, "test", sensor, seq)
                record = read(path)
                assert record["ledger_sha256"] == report["test_ledger_hashes"][seed][objective][sensor + "/" + seq]
                counts["test_ledgers"] += 1
            assert sum(v["n_queries"] for v in per.values()) == 195726
            groups = {city: {m: float(np.mean([per[s][m] for s in names if per[s]["n_pos"]]))
                            for m in METRICS} for city, names in CITY.items()}
            for metric in METRICS:
                errors.append(abs(float(np.mean([v[metric] for v in groups.values()])) - result["macro"][metric]))
    assert max(errors, default=0) < 1e-12
    for metric in METRICS:
        for objective in ("AP", "R@1"):
            actual = float(np.mean([r[objective]["macro"][metric] for r in report["test"].values()]))
            assert abs(actual - report["mean"][objective][metric]) < 1e-12
        delta = report["mean"]["AP"][metric] - report["mean"]["R@1"][metric]
        assert abs(delta - report["AP_selection_minus_R1_selection"][metric]) < 1e-12
    write(OUT / f"independent_verification_{method}.json", dict(
        status="complete", completed_unix=time.time(), seconds=time.time() - start,
        method=method, counts=counts, selection_sha256=sha(selection_path),
        report_sha256=sha(report_path), validation_sha256=sha(validation_path),
        independent_metric_max_error=max(errors, default=0),
        query_reference_manifest_sha256=(sha(OUT / "test_query_reference/manifest.json")
                                         if (OUT / "test_query_reference/manifest.json").exists() else None),
        verified_query_identity_with_current_NSD=True,
        verified_top1_geometric_distance=True, verified_tie_rule=True,
        verified_test_input_provenance=input_provenance,
        ledger_sha256=checked))
    print(method, counts, "maximum_metric_error", max(errors, default=0), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    verify(args.method, validation_only=args.validation_only)
