#!/usr/bin/env python3
"""검증에서 고정된 ImLPR 선택만 기존 native 규약으로 대조한다."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle
import statistics
import time

import numpy as np

import _review_native_imlpr as native
from _review_native_learned_verify import independent_pair

BASE = native.OUT
OUT = BASE / "common_selection_native"
DELIVERY = native.REPO / "results/pdv2_review_selection_full_20261105/selection_delivery_imlpr.json"


def freeze() -> None:
    """추론 전에 선택 근거와 실제 모델 텐서를 확인하고 명세를 저장한다."""
    import torch
    from _review_selection_full_catalog import tensor_hash

    torch.set_num_threads(2)
    delivery = json.loads(DELIVERY.read_text())
    assert delivery["test_used_for_selection"] is False
    selection_path = native.REPO / delivery["selection_path"]
    assert native.sha(selection_path) == delivery["selection_sha256"]
    original = native.check()
    candidates = []
    for seed in ("1", "2", "3"):
        entry = delivery["models"][seed]
        old = entry["original"]
        assert old["sha256"] == original["models"][f"v2_s{seed}"]["sha256"]
        groups = {}
        for objective in ("AP", "R@1"):
            chosen = entry["choices"][objective]
            path = native.REPO / chosen["path"]
            assert native.sha(path) == chosen["sha256"]
            state = torch.load(path, map_location="cpu", weights_only=False)
            state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
            digest, _, _ = tensor_hash(state)
            assert digest == chosen["model_tensor_sha256"]
            if digest not in groups:
                groups[digest] = {"id": f"s{seed}_{digest[:12]}", "seed": int(seed),
                    "path": chosen["path"], "sha256": chosen["sha256"],
                    "model_tensor_sha256": digest, "epoch": chosen["epoch"],
                    "objectives": [], "original_alias": f"v2_s{seed}" if digest == old["model_tensor_sha256"] else None}
            groups[digest]["objectives"].append(objective)
        candidates.extend(groups.values())
    manifest = {"selection_sha256": delivery["selection_sha256"], "delivery_sha256": native.sha(DELIVERY),
        "candidates": candidates, "parent_protocol_sha256": native.sha(BASE / "protocol.json"),
        "inference_batch": 1, "ks": [15, 25], "sequences": native.NAMES,
        "policy": "공통검증 AP/R@1 선택만 전달한다. native/공통시험으로 후보를 고르지 않는다. 원래 모델과 텐서가 같으면 검산된 원장을 읽기 전용으로 재사용한다.",
        "scope": "보존 best/final 선택 민감도이며 최적 수렴이나 누락 중간 epoch 검증이 아니다."}
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = copy.deepcopy(original)
    protocol["models"] = {c["id"]: {"path": c["path"], "sha256": c["sha256"]} for c in candidates}
    protocol["source_sha256"].update({str(Path(__file__).relative_to(native.REPO)): native.sha(Path(__file__)),
        "scripts/_review_native_imlpr.py": native.sha(Path(native.__file__)),
        str((BASE / "protocol.json").relative_to(native.REPO)): native.sha(BASE / "protocol.json")})
    for sequence in native.NAMES:
        for name in ["metadata.pickle", "index.npz", "done.json", "transfer_complete.json"]:
            path = BASE / "inputs" / sequence / name
            protocol["source_sha256"][str(path.relative_to(native.REPO))] = native.sha(path)
    protocol["common_selection_manifest"] = manifest
    for name, value in [("selection_manifest.json", manifest), ("protocol.json", protocol)]:
        path = OUT / name
        if path.exists():
            assert json.loads(path.read_text()) == value
        else:
            native.write(path, value)
    links = {OUT / "inputs": "../inputs"}
    links.update({OUT / f"k{k}/data": f"../../k{k}/data" for k in (15, 25)})
    for link, target in links.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            assert not link.exists()
            link.symlink_to(target, target_is_directory=True)
        assert link.resolve().exists()
    print(json.dumps({"frozen": True, "unique": len(candidates),
        "existing_aliases": sum(c["original_alias"] is not None for c in candidates),
        "new_candidates": sum(c["original_alias"] is None for c in candidates)}))


def check() -> dict:
    """동결된 규약과 선택 전달 파일의 해시를 검사한다."""
    native.OUT = OUT
    protocol = native.check()
    manifest = protocol["common_selection_manifest"]
    assert native.sha(DELIVERY) == manifest["delivery_sha256"]
    return manifest


def prediction_root(candidate: dict, k: int) -> Path:
    """원래 모델 alias는 보존 원장, 새 모델은 별도 결과 경로를 쓴다."""
    if candidate["original_alias"]:
        return BASE / f"k{k}" / candidate["original_alias"]
    return OUT / f"k{k}" / candidate["id"]


def verify() -> None:
    """공식 질의함수를 호출하지 않고 새 선택 전체 원장을 재계산한다."""
    manifest = check()
    rows = []
    for candidate in manifest["candidates"]:
        for k in manifest["ks"]:
            for sequence in manifest["sequences"]:
                path = prediction_root(candidate, k) / f"native_batch1_{sequence}.json"
                report = json.loads(path.read_text())
                assert report["inference_batch"] == 1
                metadata = pickle.loads((BASE / "inputs" / sequence / "metadata.pickle").read_bytes())
                with np.load(path.with_suffix(".npz")) as z:
                    evaluated, good, hits, cells, fixed = independent_pair(
                        z["indices"], z["distances"], metadata, metadata, 0, True, 60, True)
                    np.testing.assert_array_equal(cells, z["tp_fp_tn_fn"])
                    assert int(evaluated.sum()) == report["official_evaluated_queries"]
                    assert fixed == report["fixed_positive_denominator"]
                    assert int(hits[0]) == report["hits_at1"]
                    assert hits[0] / evaluated.sum() == report["r1"]
                    assert hits[4] / evaluated.sum() == report["r5"]
                    precision = cells[0] / (cells[0] + cells[1] + 1e-8)
                    recall = cells[0] / (cells[0] + cells[3] + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    assert float(f1.max()) == report["official_max_f1"]
                rows.append({"candidate": candidate, "k": k, "sequence": sequence,
                    "r1": report["r1"], "r5": report["r5"], "max_f1": report["official_max_f1"],
                    "positive": int(evaluated.sum()), "hits_at1": int(hits[0]),
                    "report_path": str(path.relative_to(native.REPO)), "report_sha256": native.sha(path),
                    "prediction_sha256": native.sha(path.with_suffix(".npz")), "passed": True})
    groups = {}
    for objective in ("AP", "R@1"):
        groups[objective] = {}
        for k in manifest["ks"]:
            groups[objective][str(k)] = {}
            for sequence in manifest["sequences"]:
                matching = [r for r in rows if objective in r["candidate"]["objectives"] and r["k"] == k and r["sequence"] == sequence]
                assert len(matching) == 3
                groups[objective][str(k)][sequence] = {
                    metric: {"mean": statistics.mean(r[metric] for r in matching),
                             "sample_sd": statistics.stdev(r[metric] for r in matching),
                             "per_seed": {str(r["candidate"]["seed"]): r[metric] for r in matching}}
                    for metric in ("r1", "r5", "max_f1")}
    native.write(OUT / "independent_verification.json", {"complete": True, "passed": True,
        "unix_time": time.time(), "selection_manifest_sha256": native.sha(OUT / "selection_manifest.json"),
        "rows": rows, "groups": groups, "scope": manifest["scope"]})
    print(json.dumps({"complete": True, "verified": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "infer", "score", "verify"])
    parser.add_argument("--model")
    parser.add_argument("--sequence", choices=native.NAMES)
    parser.add_argument("--k", type=int, choices=[15, 25])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "verify":
        verify()
    else:
        manifest = check()
        candidate = next(c for c in manifest["candidates"] if c["id"] == args.model)
        assert candidate["original_alias"] is None, "보존 원장에 쓰지 않는다."
        if args.stage == "infer":
            native.infer(args.model, args.sequence, args.k, args.smoke)
        else:
            native.score(args.model, args.sequence, args.k)
