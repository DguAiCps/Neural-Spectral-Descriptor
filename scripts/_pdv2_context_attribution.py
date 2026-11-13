#!/usr/bin/env python3
"""대칭 문맥의 기여 경로를 저장 기록과 검증 전용 교차 대조로 진단한다.

시험 기록은 기존 결과의 사후 집계에만 쓴다. 새 점수/후보 조합은 검증셋에만
적용하며 체크포인트, 주평가 설정, 원고 표를 바꾸지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/pdv2_symmetric3_20261013"
OUT = REPO / "results/pdv2_context_attribution_20261014"
OPS = ("pbev128_main", "hspec12")
SEEDS = (1, 2, 3)
GROUPS = ("same_available", "opposite_only", "other")
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]


def digest(path: Path) -> str:
    """입력 파일의 해시를 구한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data: dict) -> None:
    """유한한 JSON을 원자적으로 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n")
    temp.replace(path)


def read(path: Path) -> dict:
    """질의 원장에서 수치 배열만 읽는다."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files if k != "meta"}


def transitions(old: dict, new: dict, mask: np.ndarray) -> dict:
    """키·후보·최종 선택의 회복과 손실을 같은 질의에서 센다."""
    a, b = old["top1_geo_dist"] <= 5, new["top1_geo_dist"] <= 5
    ka, kb = old["key_top1_geo"] <= 5, new["key_top1_geo"] <= 5
    kwin, kloss = ~ka & kb, ka & ~kb
    win, loss = ~a & b, a & ~b
    return {name: int((mask & value).sum()) for name, value in {
        "n": np.ones(len(mask), bool),
        "same_final_candidate": old["j_star"] == new["j_star"],
        "key_recovered": kwin, "key_regressed": kloss,
        "key_recovered_already_final_correct_without_context": kwin & a,
        "key_regressed_still_final_correct_with_context": kloss & b,
        "full_recovered": win, "full_regressed": loss,
        "full_recovered_from_missing_pool": win & ~old["union_hit"],
        "full_regressed_due_to_missing_pool": loss & ~new["union_hit"],
        "pool_recovered": ~old["union_hit"] & new["union_hit"],
        "pool_regressed": old["union_hit"] & ~new["union_hit"],
    }.items()}


def stored() -> None:
    """기존 3시드 기록을 다시 집계하며 설정을 선택하지 않는다."""
    source = BASE / "summary.json"
    summary = json.loads(source.read_text())
    result = {"scope": "기존 대칭 문맥 3시드의 사후 진단. 새 시험 평가·설정 선택 없음.",
              "source_sha256": {str(source.relative_to(REPO)): digest(source)},
              "script_sha256": digest(Path(__file__)), "stages": {},
              "gate_alpha": {role: row["gate_alpha"] for role, row in
                             summary["timing_and_gates"].items()}}
    for stage in ("val_selected", "test_selected"):
        result["stages"][stage] = {}
        for op in OPS:
            reference = summary["stages"][stage]["ops"][op]
            seeds = {}
            for seed in SEEDS:
                ref = reference["per_seed"][str(seed)]
                counts = {}
                for sequence in ref["arms"]["symmetric"]["per_sequence"]:
                    paths = [BASE / f"s{seed}/{stage}_{arm}_{op}_{sequence.replace('/', '_')}.npz"
                             for arm in ("no_context", "symmetric")]
                    old, new = [read(p) for p in paths]
                    for name in ("query_idx", "min_past_gt_dist", "group"):
                        np.testing.assert_array_equal(old[name], new[name])
                    positive = old["min_past_gt_dist"] <= 5
                    for group, mask in [("all", positive)] + [
                            (g, positive & (old["group"] == i))
                            for i, g in enumerate(GROUPS)]:
                        row = transitions(old, new, mask)
                        total = counts.setdefault(group, {k: 0 for k in row})
                        for key, value in row.items():
                            total[key] += value
                    for path in paths:
                        result["source_sha256"][str(path.relative_to(REPO))] = digest(path)
                for name in ("key_recovered", "key_regressed", "full_recovered", "full_regressed"):
                    assert counts["all"][name] == ref["symmetric_paired_vs"]["no_context"]["all"][name]
                seeds[str(seed)] = {"transitions": counts,
                    "direction": {arm: ref["arms"][arm]["direction"]
                                  for arm in ("no_context", "symmetric")},
                    "macro": {arm: ref["arms"][arm]["macro"]
                              for arm in ("no_context", "symmetric")},
                    "group_macro": {arm: ref["arms"][arm]["group_macro"]
                                    for arm in ("no_context", "symmetric")}}
            result["stages"][stage][op] = {"per_seed": seeds,
                                            "mean_std": reference["mean_std"]}
    write(OUT / "stored_diagnosis.json", result)
    print("STORED_DONE", flush=True)


def crossed(seed: int, sequence: str) -> None:
    """검증에서 후보 생성의 문맥과 점수의 문맥을 독립적으로 교차한다."""
    from _pdv2_lag15_campaign import bank_for, inputs
    from _pdv2_no_context_ablation import no_context_key
    from _causal_pr_dump import prefix_topk
    from _xs_eval_lib import align_scores, column_align_scores, minmax, normalize

    started = time.monotonic()
    sensor, seq, paths, _ = inputs("val", sequence)
    dest = OUT / f"val_s{seed}_{sensor}_{seq}.json"
    assert not dest.exists(), "기존 결과를 덮어쓰지 않는다"
    key_path = BASE / f"s{seed}/keys_val_{sensor}_{seq}.npz"
    selection_path = BASE / f"s{seed}/confidence_selection.json"
    selection = json.loads(selection_path.read_text())["ops"]
    rpath = REPO / "artifacts/key_remetrize_r_pdv2.npy"
    with np.load(paths["cache"]) as cache, np.load(key_path) as keys:
        pos = cache["poses_gt"][:, :3, 3]
        raw = no_context_key(cache["descriptors"], np.load(rpath))
        full = keys["symmetric"]
        np.testing.assert_array_equal(cache["scan_ids"], keys["scan_ids"])
    embeddings = {"0": raw, "1": full}
    for beta in (0.25, 0.5):
        emb = full.copy()
        emb[:, 288:] *= beta
        embeddings[str(beta)] = normalize(emb)
    result = {"seed": seed, "sequence": sequence, "role": "val",
              "scope": "동결된 대칭 모델의 후보/점수 교차 및 고정 후보 안 문맥 축소 진단. 재학습·재선택 없음.",
              "script_sha256": digest(Path(__file__)), "ops": {},
              "sha256": {str(p.relative_to(REPO)): digest(p) for p in
                         [key_path, selection_path, rpath, *paths.values()]}}
    for op in OPS:
        references = {name: read(BASE / f"s{seed}/val_selected_{name}_{op}_{sensor}_{seq}.npz")
                      for name in ("no_context", "symmetric")}
        base = references["symmetric"]
        qids = base["query_idx"]
        weights = [float(v) for v in selection[op]["tuple"]]
        fields = ("j_star", "top1_geo_dist", "c_spec", "c_col", "score_abs", "score_margin")
        conditions = {f"pool_{pool}_score_{beta}": {f: np.empty(len(qids),
                      np.int64 if f == "j_star" else np.float64) for f in fields}
                      for pool in ("0", "1") for beta in embeddings}
        with np.load(paths["spec"]) as spectrum:
            hkey, hfft, hnorm, column = bank_for(op, spectrum)
        for qi, q in enumerate(qids):
            q = int(q)
            aux = prefix_topk(hkey, q, q - 29, 100)
            pools = {name: np.unique(np.r_[prefix_topk(embeddings[name], q, q - 29, 100), aux])
                     for name in ("0", "1")}
            union = np.union1d(pools["0"], pools["1"])
            a_all = align_scores(hfft, hnorm, q, union)
            c_all = column_align_scores(column, q, union)
            for pool_name, candidates in pools.items():
                indexes = np.searchsorted(union, candidates)
                a, c = a_all[indexes], c_all[indexes]
                for beta, emb in embeddings.items():
                    # 원래 평가와 같은 덧셈 순서로 재현성을 확인한다.
                    score = (weights[0] * minmax(emb[candidates] @ emb[q]) +
                             weights[1] * minmax(hkey[candidates] @ hkey[q]) +
                             weights[2] * minmax(a) + weights[3] * minmax(c))
                    order = np.argsort(-score, kind="stable")
                    best, j = int(order[0]), int(candidates[order[0]])
                    row = conditions[f"pool_{pool_name}_score_{beta}"]
                    values = {"j_star": j, "top1_geo_dist": np.linalg.norm(pos[j] - pos[q]),
                              "c_spec": a[best], "c_col": c[best], "score_abs": emb[q] @ emb[j],
                              "score_margin": score[best] - score[order[1]] if len(order) > 1 else 0.}
                    for field, value in values.items():
                        row[field][qi] = value
        checks = {}
        for beta, arm in (("0", "no_context"), ("1", "symmetric")):
            arrays = conditions[f"pool_{beta}_score_{beta}"]
            ref = references[arm]
            np.testing.assert_array_equal(arrays["j_star"], ref["j_star"])
            checks[arm] = {}
            for field in fields[1:]:
                error = float(np.max(np.abs(arrays[field] - ref[field])))
                np.testing.assert_allclose(arrays[field], ref[field], rtol=0, atol=2e-6)
                checks[arm][field] = error
        outputs = {}
        from _pr_eval_lib import classify_queries, pr_sweep
        qc = classify_queries(base["min_past_gt_dist"])
        for name, arrays in conditions.items():
            decision = selection[op]["decision"]
            assert decision in ("c_spec", "c_col")
            metrics = None
            if qc["n_pos"]:
                sweep = pr_sweep(arrays[decision], arrays["top1_geo_dist"], qc)
                metrics = {m: float(sweep[m]) for m in ("AP", "max_F1", "R@100P")}
                metrics["R@1"] = float((arrays["top1_geo_dist"][qc["positive"]] <= 5).mean())
            outputs[name] = {"metrics": metrics,
                             "negative_false_accepts": int((qc["negative"] &
                                 (arrays[decision] >= selection[op]["tau_star"])).sum())}
            payload = {**arrays, "query_idx": qids, "group": base["group"],
                       "min_past_gt_dist": base["min_past_gt_dist"]}
            np.savez_compressed(OUT / f"val_s{seed}_{sensor}_{seq}_{op}_{name}.npz", **payload)
        result["ops"][op] = {"conditions": outputs, "reference_reproduction": checks,
                              "selection": selection[op], "n_queries": len(qids),
                              "n_positive": qc["n_pos"]}
    result["runtime_s"] = time.monotonic() - started
    write(dest, result)
    print("CROSSED_DONE", seed, sequence, round(result["runtime_s"], 1), flush=True)


def aggregate() -> None:
    """기존 집계와 같은 순서로 검증 계열 평균을 구한다."""
    from _pdv2_pilot import FAMILY, VAL
    source = json.loads((BASE / "summary.json").read_text())
    result = {"scope": "검증셋 9시퀀스·3시드의 진단. 양성 없는 KITTI09는 PR 평균 제외.",
              "aggregation": "시퀀스 → 센서 계열 → 시드 평균. 표4 시험 도시 평균과 직접 비교하지 않는다.",
              "ops": {}, "sha256": {}}
    for op in OPS:
        per_seed = {}
        for seed in SEEDS:
            rows = {}
            for sensor, seq in VAL:
                path = OUT / f"val_s{seed}_{sensor}_{seq}.json"
                row = json.loads(path.read_text())
                assert row["script_sha256"] == digest(Path(__file__))
                rows[sensor + "/" + seq] = row["ops"][op]
                result["sha256"][str(path.relative_to(REPO))] = digest(path)
            conditions = next(iter(rows.values()))["conditions"]
            macros = {}
            for condition in conditions:
                groups = {}
                for sensor, seq in VAL:
                    metrics = rows[sensor + "/" + seq]["conditions"][condition]["metrics"]
                    if metrics is not None:
                        groups.setdefault(FAMILY.get(sensor, sensor), []).append(metrics)
                macro = {metric: float(np.mean([np.mean([r[metric] for r in group])
                          for group in groups.values()])) for metric in ("AP", "max_F1", "R@100P", "R@1")}
                macros[condition] = macro
            for beta, arm in (("0", "no_context"), ("1", "symmetric")):
                expected = source["stages"]["val_selected"]["ops"][op]["per_seed"][str(seed)]["arms"][arm]["macro"]
                for metric, actual in macros[f"pool_{beta}_score_{beta}"].items():
                    assert abs(actual - expected[metric]) < 1e-12, (op, seed, beta, metric)
            per_seed[str(seed)] = macros
        means = {c: {m: float(np.mean([r[c][m] for r in per_seed.values()]))
                     for m in per_seed["1"][c]} for c in per_seed["1"]}
        result["ops"][op] = {"per_seed": per_seed, "mean": means}
    write(OUT / "crossed_val_summary.json", result)
    print(json.dumps(result["ops"], ensure_ascii=False), flush=True)


def main() -> None:
    """저장 기록 집계 또는 검증 교차 진단을 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stored", "crossed", "aggregate"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == "stored":
        stored()
    elif args.action == "crossed":
        assert args.seed is not None and args.sequence is not None
        crossed(args.seed, args.sequence)
    else:
        aggregate()


if __name__ == "__main__":
    main()
