#!/usr/bin/env python3
"""저장된 대칭 GNN 체크포인트를 최종 검증 AP로 선택하고 고정 시험한다."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_lag15_campaign import (
    OUT as BASE, CONFIG, RVEC, bank_for, classifier_path, inputs, model,
)
from _pdv2_no_context_ablation import evaluate_keys, no_context_key, sha256, write_json
from _pdv2_lag15_report import read as read_ledger, sequence_metrics, stats
from _pdv2_pilot import CITY, FAMILY, OPS, TEST, VAL

OUT = REPO / "results/pdv2_checkpoint_select_20261014"
SEEDS = (1, 2, 3)
PRIMARY = ("AP", "max_F1", "R@100P", "R@1")


def read_json(path: Path) -> dict:
    """JSON 기록을 읽는다."""
    return json.loads(path.read_text())


def prepare() -> dict:
    """검증 실행 전에 후보 체크포인트와 선택 규칙을 고정한다."""
    import torch
    target = OUT / "protocol.json"
    if target.exists():
        return protocol()
    candidates = {}
    files = [Path(__file__), REPO / "scripts/_pdv2_checkpoint_select_run.py", CONFIG,
             RVEC, BASE / "summary.json", BASE / "freeze.json"]
    files += [REPO / name for name in (
        "scripts/_pdv2_lag15_campaign.py", "scripts/_pdv2_lag15_report.py",
        "scripts/_lag_branch_fusion.py", "scripts/_causal_branch_fusion.py",
        "scripts/_causal_edge_features.py", "scripts/_causal_pr_dump.py",
        "scripts/_pdv2_no_context_ablation.py", "scripts/_pr_eval_lib.py",
        "scripts/_xs_eval_lib.py", "scripts/_pdv2_pilot.py",
        "src/gnn/model.py", "src/keyframe/graph_manager.py")]
    for seed in SEEDS:
        rows = []
        for path in sorted((REPO / f"checkpoints/pdv2_lag15_s{seed}").glob("*.pth")):
            blob = torch.load(path, map_location="cpu", weights_only=False)
            state = blob["model_state_dict"]
            assert all(torch.isfinite(v).all() for v in state.values())
            rows.append({"id": path.stem, "epoch": int(blob["epoch"]) + 1,
                         "path": str(path.relative_to(REPO)), "sha256": sha256(path),
                         "reference": path.name == "best_model.pth"})
            files.append(path)
        assert sum(r["reference"] for r in rows) == 1
        candidates[str(seed)] = sorted(rows, key=lambda r: (r["epoch"], r["id"]))
        files.extend([classifier_path(seed), BASE / f"s{seed}/confidence_selection.json"])
    value = {
        "date": "2026-10-14", "seeds": list(SEEDS), "candidates": candidates,
        "roster": {role: [s + "/" + q for s, q in rows]
                   for role, rows in (("val", VAL), ("test", TEST))},
        "queries": "30 <= q < N-15, candidates j <= q-30, 기존 주표와 같은 질의",
        "fixed": "GNN 체크포인트만 변경. 시드별 기존 간선 분류기, 재계량, 보조 색인, 100+100 후보, 정렬, 융합 가중치, 판정 점수, 임계값을 유지한다.",
        "selection": "시드별 한 체크포인트를 NSD/NSD-H가 공유. 운영점별 검증 시퀀스 AP -> 3센서 계열 균등 평균 -> 두 운영점 평균 최대. 양성 없는 KITTI09는 제외. 기존 최종 AP 집계와 같은 규칙이다.",
        "tie_rule": "1e-12 이내 동률이면 기존 best_model 우선, 그다음 이른 에포크, 파일 이름 순",
        "test_rule": "모든 검증 후보 평가와 세 시드 선택을 동결한 뒤 선택된 체크포인트만 시험. 기존/문맥 제거 원장과 같은 질의에서 비교.",
        "scope": "기존 시험이 개발에 노출된 추가 실험. 간선 분류기/융합을 새 체크포인트에 다시 맞춘 최적 파이프라인 비교는 아니다. 모델 파일과 원고 주표는 교체하지 않는다.",
        "sha256": {str(p.relative_to(REPO)): sha256(p) for p in files},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(target, value)
    return value


def protocol() -> dict:
    """기록한 코드와 모델이 유지되는지 검사한다."""
    value = read_json(OUT / "protocol.json")
    expected_hashes = dict(value["sha256"])
    amendment_path = OUT / "preflight_amendment.json"
    if amendment_path.exists():
        amendment = read_json(amendment_path)
        assert amendment["protocol_sha256"] == sha256(OUT / "protocol.json")
        for name, row in amendment["code_changes"].items():
            assert expected_hashes[name] == row["before"]
            expected_hashes[name] = row["after"]
    for name, expected in expected_hashes.items():
        assert sha256(REPO / name) == expected, name
    return value


def candidate(spec: dict, seed: int, cid: str) -> dict:
    """고정된 후보 목록에서만 모델을 고른다."""
    return next(row for row in spec["candidates"][str(seed)] if row["id"] == cid)


def guard(role: str, seed: int, cid: str, sequence: str) -> dict:
    """검증 목록과 고정된 시험 선택에만 접근한다."""
    spec = protocol()
    assert sequence in spec["roster"][role]
    if role == "test":
        frozen = read_json(OUT / "freeze.json")
        assert frozen["protocol_sha256"] == sha256(OUT / "protocol.json")
        assert frozen["selected"][str(seed)]["id"] == cid
        assert frozen["val_summary_sha256"] == sha256(OUT / "validation_summary.json")
    return spec


def job_dir(seed: int, cid: str) -> Path:
    """체크포인트별 독립 출력 위치를 준비한다."""
    path = OUT / f"s{seed}" / cid
    path.mkdir(parents=True, exist_ok=True)
    return path


def keys(seed: int, cid: str, role: str, sequence: str, audit: bool = False) -> None:
    """같은 추론 경로에 저장된 모델 가중치만 교체해 키를 만든다."""
    import torch
    from _lag_branch_fusion import nsd_embedding_lag
    from _pdv2_pair_evaluate import ArrayCache
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = guard(role, seed, cid, sequence)
    entry = candidate(spec, seed, cid)
    sensor, seq, paths, _ = inputs(role, sequence)
    target = job_dir(seed, cid) / f"keys_{role}_{sensor}_{seq}.npz"
    if target.exists() and not audit:
        return
    if entry["reference"] and not audit:
        return
    started = time.monotonic()
    cfg, net, clf, blob, rvec, _ = model(seed)
    state = torch.load(REPO / entry["path"], map_location="cpu", weights_only=False)["model_state_dict"]
    missing, unexpected = net.load_state_dict(state, strict=False)
    assert missing == ["gnn.key_scale"] and not unexpected, (missing, unexpected)
    actual = net.state_dict()
    assert all(torch.equal(actual[k].cpu(), v) for k, v in state.items())
    np.testing.assert_array_equal(actual["gnn.key_scale"].cpu().numpy(), rvec)
    with np.load(paths["cache"]) as source, np.load(paths["layouts"]) as layouts:
        cache = ArrayCache({k: source[k] for k in source.files})
        layout = layouts["nsd_layouts"].astype(np.float32)
        np.testing.assert_array_equal(cache["scan_ids"], layouts["scan_ids"])
    raw = cache["descriptors"].astype(np.float32)
    emb = nsd_embedding_lag(cache, sensor, raw, layout, cfg, net, clf, blob, rvec, "cuda")
    assert emb.shape == (len(raw), 416) and emb.dtype == np.float32 and np.isfinite(emb).all()
    normalized_raw = emb[:, :288] / np.linalg.norm(emb[:, :288], axis=1, keepdims=True)
    raw_error = float(np.max(np.abs(normalized_raw - no_context_key(raw, rvec)[:, :288])))
    assert raw_error < 3e-7
    meta = {"seed": seed, "checkpoint": entry, "role": role, "sequence": sequence,
            "protocol_sha256": sha256(OUT / "protocol.json"), "raw_block_error": raw_error,
            "inputs_sha256": {str(p.relative_to(REPO)): sha256(p) for p in paths.values()},
            "runtime_s": time.monotonic() - started}
    if audit:
        assert entry["reference"] and role == "val"
        with np.load(BASE / f"s{seed}/keys_val_{sensor}_{seq}.npz") as old:
            error = np.abs(emb - old["symmetric"])
        meta["reference_key_max_error"] = float(error.max())
        meta["reference_key_mean_error"] = float(error.mean())
        meta["reference_key_original_tolerance_passed"] = bool(error.max() < 2e-5 and error.mean() < 1e-6)
        # 국소 간선 선택의 반복 실행 차이를 보존하고 최종 지표도 따로 대조한다.
        assert error.mean() < 1e-6
        target = job_dir(seed, cid) / f"audit_keys_{sensor}_{seq}.npz"
    temp = target.with_suffix(".tmp.npz")
    np.savez_compressed(temp, embedding=emb, scan_ids=cache["scan_ids"], meta=json.dumps(meta))
    temp.replace(target)
    write_json(target.with_suffix(".json"), meta)
    print("KEYS_DONE", seed, cid, role, sequence, round(meta["runtime_s"], 1), flush=True)


def score(seed: int, cid: str, role: str, sequence: str, audit: bool = False) -> None:
    """정렬과 최종 판정을 포함한 동일 원장을 생성하고 집계한다."""
    spec = guard(role, seed, cid, sequence)
    entry = candidate(spec, seed, cid)
    sensor, seq, paths, min_past = inputs(role, sequence)
    folder = job_dir(seed, cid)
    prefix = "audit" if audit else role
    target = folder / f"{prefix}_{sensor}_{seq}.json"
    if target.exists():
        return
    selections = read_json(BASE / f"s{seed}/confidence_selection.json")["ops"]
    started = time.monotonic()
    reuse = entry["reference"] and not audit
    if not reuse:
        key_path = folder / (f"audit_keys_{sensor}_{seq}.npz" if audit else f"keys_{role}_{sensor}_{seq}.npz")
        with np.load(key_path) as keys_file:
            key_meta = json.loads(str(keys_file["meta"]))
            assert key_meta["protocol_sha256"] == sha256(OUT / "protocol.json")
            emb = keys_file["embedding"]
            scan_ids = keys_file["scan_ids"]
        with np.load(paths["cache"]) as cache:
            pos = cache["poses_gt"][:, :3, 3]
            np.testing.assert_array_equal(scan_ids, cache["scan_ids"])
    results = {}
    for op in OPS:
        reference_path = BASE / f"s{seed}/{role}_selected_symmetric_{op}_{sensor}_{seq}.npz"
        ref, _ = read_ledger(reference_path)
        selection = selections[op]
        if reuse:
            arrays = ref
            dest = reference_path
        else:
            with np.load(paths["spec"]) as spectrum:
                np.testing.assert_array_equal(scan_ids, spectrum["scan_ids"])
                bank = bank_for(op, spectrum)
            arrays = evaluate_keys(emb, bank, pos, min_past,
                                   {seed: tuple(float(w) for w in selection["tuple"])})[seed]
            keep = arrays["query_idx"] < len(emb) - 15
            arrays = {k: v[keep] for k, v in arrays.items()}
            arrays["group"] = ref["group"]
            for key in ("query_idx", "min_past_gt_dist"):
                np.testing.assert_array_equal(arrays[key], ref[key])
            assert np.isfinite(arrays["score_abs"]).all()
            dest = folder / f"{prefix}_{op}_{sensor}_{seq}.npz"
            temp = dest.with_suffix(".tmp.npz")
            np.savez_compressed(temp, **arrays, meta=json.dumps({"checkpoint": cid,
                                "protocol_sha256": sha256(OUT / "protocol.json")}))
            temp.replace(dest)
        metrics = sequence_metrics(arrays, selection["decision"], selection["tau_star"])
        results[op] = {"metrics": metrics, "ledger": str(dest.relative_to(REPO)),
                       "sha256": sha256(dest)}
        if audit:
            expected = sequence_metrics(ref, selection["decision"], selection["tau_star"])
            results[op]["reference_candidate_changes"] = int((ref["j_star"] != arrays["j_star"]).sum())
            changes = {m: metrics[m] - expected[m] for m in PRIMARY if metrics[m] is not None}
            assert all(abs(d) < 2e-5 for d in changes.values()), changes
            results[op]["reference_metric_deltas"] = changes
    result = {"seed": seed, "checkpoint": entry, "role": role, "sequence": sequence,
              "reused_reference": reuse, "ops": results,
              "protocol_sha256": sha256(OUT / "protocol.json"),
              "runtime_s": time.monotonic() - started}
    write_json(target, result)
    print("SCORE_DONE", seed, cid, role, sequence, round(result["runtime_s"], 1), flush=True)


def aggregate_rows(rows: dict, role: str, op: str) -> dict:
    """검증 계열 또는 시험 도시를 같은 비중으로 평균한다."""
    groups = CITY if role == "test" else {}
    if role == "val":
        for sensor, seq in VAL:
            groups.setdefault(FAMILY.get(sensor, sensor), []).append(sensor + "/" + seq)
    per = {name: row["ops"][op]["metrics"] for name, row in rows.items()}
    by_group = {g: {m: float(np.mean([per[s][m] for s in names if per[s][m] is not None]))
                    for m in PRIMARY} for g, names in groups.items()}
    return {"macro": {m: float(np.mean([r[m] for r in by_group.values()])) for m in PRIMARY},
            "by_group": by_group, "per_sequence": per}


def select() -> None:
    """검증 AP로만 세 시드의 공통 운영점 체크포인트를 선택하고 동결한다."""
    spec = protocol()
    assert not (OUT / "freeze.json").exists()
    result, selected, hashes = {}, {}, {}
    source = read_json(BASE / "summary.json")
    for seed in SEEDS:
        scored = []
        for entry in spec["candidates"][str(seed)]:
            rows = {}
            for sensor, seq in VAL:
                path = job_dir(seed, entry["id"]) / f"val_{sensor}_{seq}.json"
                row = read_json(path)
                assert row["protocol_sha256"] == sha256(OUT / "protocol.json")
                rows[sensor + "/" + seq] = row
                hashes[str(path.relative_to(REPO))] = sha256(path)
            ops = {op: aggregate_rows(rows, "val", op) for op in OPS}
            joint = float(np.mean([row["macro"]["AP"] for row in ops.values()]))
            scored.append({**entry, "ops": ops, "joint_val_AP": joint})
            if entry["reference"]:
                for op in OPS:
                    ref = source["stages"]["val_selected"]["ops"][op]["per_seed"][str(seed)]["arms"]["symmetric"]["macro"]
                    assert all(abs(ops[op]["macro"][m] - ref[m]) < 1e-12 for m in PRIMARY)
        maximum = max(row["joint_val_AP"] for row in scored)
        tied = [row for row in scored if maximum - row["joint_val_AP"] <= 1e-12]
        chosen = min(tied, key=lambda row: (not row["reference"], row["epoch"], row["id"]))
        result[str(seed)] = scored
        selected[str(seed)] = {k: chosen[k] for k in ("id", "epoch", "path", "sha256", "joint_val_AP")}
        print("SELECTED", seed, selected[str(seed)], flush=True)
    write_json(OUT / "validation_summary.json", {"protocol_sha256": sha256(OUT / "protocol.json"),
               "per_seed": result, "selected": selected, "inputs_sha256": hashes})
    write_json(OUT / "freeze.json", {"protocol_sha256": sha256(OUT / "protocol.json"),
               "selected": selected, "val_summary_sha256": sha256(OUT / "validation_summary.json"),
               "created_unix": time.time(), "selection_rule": spec["selection"]})


def report() -> None:
    """고정 선택의 시험 성능을 기존 전체/문맥 제거와 비교한다."""
    spec = protocol()
    frozen = read_json(OUT / "freeze.json")
    original = read_json(BASE / "summary.json")
    result = {"selected": frozen["selected"], "protocol_sha256": sha256(OUT / "protocol.json"),
              "scope": spec["scope"], "ops": {}}
    for op in OPS:
        seeds = {}
        for seed in SEEDS:
            cid = frozen["selected"][str(seed)]["id"]
            rows = {}
            for sensor, seq in TEST:
                guard("test", seed, cid, sensor + "/" + seq)
                row = read_json(job_dir(seed, cid) / f"test_{sensor}_{seq}.json")
                assert row["protocol_sha256"] == sha256(OUT / "protocol.json")
                ledger = REPO / row["ops"][op]["ledger"]
                assert sha256(ledger) == row["ops"][op]["sha256"]
                rows[sensor + "/" + seq] = row
            new = aggregate_rows(rows, "test", op)
            ref = original["stages"]["test_selected"]["ops"][op]["per_seed"][str(seed)]["arms"]
            seeds[str(seed)] = {"selected": new, "reference": ref["symmetric"]["macro"],
                                "no_context": ref["no_context"]["macro"],
                                "delta_vs_reference": {m: new["macro"][m] - ref["symmetric"]["macro"][m] for m in PRIMARY},
                                "delta_vs_no_context": {m: new["macro"][m] - ref["no_context"]["macro"][m] for m in PRIMARY}}
        means = {condition: {m: stats([row[condition]["macro"][m] if condition == "selected"
                    else row[condition][m] for row in seeds.values()]) for m in PRIMARY}
                 for condition in ("selected", "reference", "no_context", "delta_vs_reference", "delta_vs_no_context")}
        result["ops"][op] = {"per_seed": seeds, "mean_std": means}
    write_json(OUT / "summary.json", result)
    print("REPORT_DONE", {op: r["mean_std"]["selected"] for op, r in result["ops"].items()}, flush=True)


def main() -> None:
    """선택 실험의 한 단계를 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "keys", "score", "select", "report"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--checkpoint")
    parser.add_argument("--role", choices=("val", "test"), default="val")
    parser.add_argument("--sequence")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "select":
        select()
    elif args.action == "report":
        report()
    else:
        assert args.seed is not None and args.checkpoint and args.sequence
        action = keys if args.action == "keys" else score
        action(args.seed, args.checkpoint, args.role, args.sequence, args.audit)


if __name__ == "__main__":
    main()
