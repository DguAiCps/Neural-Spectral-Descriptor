#!/usr/bin/env python3
"""대칭 문맥 3시드의 검증 선택·문맥 제거·고정 시험 실행 도구.

GPU 단계는 ring_baseline 안에서 실행한다. --freeze 이전에는 시험 입력을
열지 않는다. 같은 시드의 문맥 유무 비교에는 동일한 검색·판정 설정을 쓴다.
"""
from __future__ import annotations

import argparse
import ast
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_no_context_ablation import (
    OPS, baseline_root, evaluate_keys, ledger_path, local_path, no_context_key,
    sha256, write_json,
)
from _pdv2_pair_evaluate import ArrayCache, direction_groups
from _pdv2_pilot import CITY, FAMILY, TEST, TRAIN, VAL
from _pr_eval_lib import AGG_METRICS, classify_queries, operating_point, pr_sweep
from _xs_eval_lib import hspec_inputs, normalize, variant_inputs
from _xs_eval_lib import align_scores, column_align_scores, minmax
from _causal_pr_dump import prefix_topk

OUT = REPO / "results/pdv2_symmetric3_20261013"
CONFIG = REPO / "configs/training_place_disjoint_v2_lag15.yaml"
RVEC = REPO / "artifacts/key_remetrize_r_pdv2.npy"
LAG = 15
SEEDS = (1, 2, 3)
ARMS = ("no_context", "past", "symmetric")
STATS = {"absolute": "score_abs", "margin": "score_margin", "c_spec": "c_spec", "c_col": "c_col"}


def classifier_path(seed: int, arm: str = "symmetric") -> Path:
    """각 시드에 대응하는 기존 이름 규칙의 분류기를 반환한다."""
    variant = "lag15" if arm == "symmetric" else "causal"
    suffix = "" if seed == 1 else f"_s{seed}"
    return REPO / f"artifacts/edge_classifier_pdv2_{variant}{suffix}.pt"


def checkpoint_path(seed: int, arm: str = "symmetric") -> Path:
    """해당 그래프로 학습한 시드의 체크포인트를 반환한다."""
    variant = "lag15" if arm == "symmetric" else "causal"
    return REPO / f"checkpoints/pdv2_{variant}_s{seed}/best_model.pth"


def seed_dir(seed: int) -> Path:
    """시드별 산출물 폴더를 준비한다."""
    path = OUT / f"s{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def protocol() -> dict:
    """선택·질의·제거 범위를 첫 결과 전에 고정한다."""
    path = OUT / "protocol.json"
    if path.exists():
        value = json.loads(path.read_text())
        expected = effective_hashes(value)["scripts/_pdv2_lag15_campaign.py"]
        assert expected == sha256(Path(__file__))
        return value
    from _height_candidate_fusion import WEIGHTS
    selection = json.loads((baseline_root(1) / "confidence_selection_align.json").read_text())["ops"]
    files = [Path(__file__), CONFIG, RVEC, OUT / "training_protocol.json",
             OUT / "mining_cache_amendment.json", REPO / "train_multi_dataset.py",
             baseline_root(1) / "confidence_selection_align.json"]
    files.extend(REPO / p for p in (
        "scripts/_lag_branch_fusion.py", "scripts/_causal_branch_fusion.py",
        "scripts/_causal_edge_features.py", "scripts/_causal_build_edge_classifier.py",
        "scripts/_pdv2_no_context_ablation.py", "scripts/_causal_val_select.py",
        "scripts/_pr_eval_lib.py", "scripts/_xs_eval_lib.py", "src/keyframe/graph_manager.py",
        "src/gnn/model.py", "scripts/_pdv2_c3.py",
        "scripts/_train_lag15_cached_mining.py", "src/gnn/triplet_miner.py",
        "scripts/_pdv2_pair_evaluate.py", "scripts/_causal_pr_dump.py",
        "scripts/_pdv2_pilot.py", "tests/gnn/test_cached_lag15_mining.py",
        "tests/test_lag15_campaign.py",
    ))
    value = {
        "date": "2026-10-13", "schema": "lag15-three-seed-v1", "seeds": list(SEEDS),
        "models": "대칭 시드1 재사용, 시드2·3 동일 설정 새 학습. 분류기 시드1 재사용, 시드2·3 학습 분할에서 별도 적합.",
        "condition": "현재 odom 키프레임, 자른 BEV, 수정 NCLT 시야각, 평가 CUDA matmul TF32 비활성",
        "queries": "30 <= q < N-15, 후보 j <= q-30; 모든 조건 같은 질의",
        "fixed_ablation": "모든 시드의 세 조건에 기존 과거 시드1 융합과 c_spec 임계값을 공통 적용한다.",
        "fixed_selection": selection,
        "tuple_grid": [list(w) for w in WEIGHTS],
        "tuple_rule": "대칭 모델별로 검증 양성 R@1의 계열 균등 평균 최대. 계열 안에서는 질의 합산, 동률은 격자 첫 항목.",
        "confidence_rule": "absolute, margin, c_spec, c_col의 검증 AP 계열 균등 평균 최대; absolute가 최고값 0.005 안이면 absolute. 임계값은 계열 평균 F1 최대, 동률은 높은 임계값.",
        "empty_positive_rule": "양성이 없는 시퀀스의 AP·재현율은 null, 세 조건 모두 그룹 평균에서 제외하고 비재방문 오수락을 별도 집계한다.",
        "test_ablation": "각 대칭 시드의 검증 선택 설정을 문맥 제거에도 그대로 적용한다. 문맥 제거 전용 선택은 하지 않는다.",
        "freeze_rule": "세 시드의 검증 선택·모델·분류기·코드 해시를 freeze.json에 기록한 뒤에만 시험 입력을 읽는다.",
        "scope": "현재 분할의 추가 실험이며 이미 개발에 노출된 시험셋을 새로운 독립 시험으로 부르지 않는다.",
        "roster": {r: [s + "/" + q for s, q in rows] for r, rows in (("train", TRAIN), ("val", VAL), ("test", TEST))},
        "sha256": {str(p.relative_to(REPO)): sha256(p) for p in files},
    }
    write_json(path, value)
    return value


def effective_hashes(spec: dict) -> dict:
    """사전 기록을 보존하며 시험 접촉 전 수치 검증 보완 기록을 적용한다."""
    expected = dict(spec["sha256"])
    path = OUT / "prefix_verification_amendment.json"
    if path.exists():
        amendment = json.loads(path.read_text())
        assert amendment["protocol_sha256"] == sha256(OUT / "protocol.json")
        for name, row in amendment["code_changes"].items():
            assert expected[name] == row["before"]
            expected[name] = row["after"]
        expected[str(path.relative_to(REPO))] = sha256(path)
    return expected


def prefix_verified_error(row: dict) -> float:
    """GPU 오차를 보존하면서 필요할 때 별도 CPU 참조 검사의 오차를 반환한다."""
    return row.get("cpu_reference_max_error", row["max_error_rows_le_k"])


def frozen_guard(full: bool = False) -> dict:
    """모든 시드의 선택을 고정한 상태에서만 시험 입력 접근을 허용한다."""
    path = OUT / "freeze.json"
    assert path.exists(), "세 시드 검증 선택을 먼저 고정해야 한다"
    frozen = json.loads(path.read_text())
    assert frozen["protocol_sha256"] == sha256(OUT / "protocol.json")
    for name, expected in frozen["sha256"].items():
        path = REPO / name
        stat = path.stat()
        assert [stat.st_size, stat.st_mtime_ns] == frozen["file_identity"][name], name
        if full or name in frozen["runtime_verify"]:
            assert sha256(path) == expected, name
    return frozen


def inputs(role: str, sequence: str) -> tuple:
    """현재 기준 기록의 동일 입력·정답을 읽으며 시험은 동결 검사를 거친다."""
    spec = protocol()
    assert role in ("val", "test") and sequence in spec["roster"][role]
    if role == "test":
        frozen_guard()
    sensor, seq = sequence.split("/")
    path = ledger_path(role, 1, OPS[0], sensor, seq)
    with np.load(path) as archive:
        meta = json.loads(str(archive["meta"]))
        q = archive["query_idx"]
        min_past = np.full(meta["n_frames"], np.nan)
        min_past[q] = archive["min_past_gt_dist"]
    paths = {k: local_path(meta["caches"][k]) for k in ("cache", "layouts", "spec")}
    if sensor == "nclt":
        assert "nclt_fov" in str(paths["cache"])
    assert "bevclip" in str(paths["spec"])
    return sensor, seq, paths, min_past


def model(seed: int, arm: str = "symmetric", device: str = "cuda") -> tuple:
    """필수 인코더·게이트와 동일 재계량으로 모델·분류기를 읽는다."""
    from _causal_branch_fusion import VARIANTS, make_model_causal
    variant = "lag15" if arm == "symmetric" else "causal"
    config = CONFIG if arm == "symmetric" else REPO / "configs/training_place_disjoint_v2_causal.yaml"
    name = f"campaign_{arm}_{seed}"
    VARIANTS[name] = dict(ckpt_dir=f"pdv2_{variant}_s{{seed}}", config=config,
                          rvec=RVEC, cls=classifier_path(seed, arm))
    return make_model_causal(name, seed, device)


def train_classifier(seed: int, only: str | None, train_only: bool) -> None:
    """기존 학습 전용 분류기 절차를 시드별 경로로 실행한다."""
    import torch
    import _pd_build_edge_classifier as pdec
    import _causal_build_edge_classifier as cec
    from _lag_branch_fusion import lag_pass1_graph
    torch.set_num_threads(1)
    assert seed in (2, 3)
    assert only is None or only in protocol()["roster"]["train"]
    pdec.PD_CACHE = REPO / "data/preprocessed_v2"
    train_seqs = {}
    for sensor, seq in TRAIN:
        train_seqs.setdefault(sensor, []).append(seq)
    cec.PD_TRAIN_SEQS = train_seqs
    cec.PD_HOLDOUT = {"kitti/06", "helipr/Bridge01"}
    original_cache = cec._pd_cache
    def materialized(sensor: str, seq: str) -> ArrayCache:
        with original_cache(sensor, seq) as archive:
            return ArrayCache({k: archive[k] for k in archive.files})
    cec._pd_cache = materialized
    cec.causal_pass1_graph = lag_pass1_graph
    name = f"lag15_s{seed}"
    cec.VARIANTS[name] = dict(ckpt_dir="pdv2_lag15_s{seed}", rvec=RVEC, config=CONFIG,
                            out=classifier_path(seed), feat_dir=OUT / f"edge_features_s{seed}")
    sys.argv = [cec.__file__, "--variant", name, "--seed", str(seed)]
    if only:
        sys.argv += ["--only", only]
    if train_only:
        sys.argv += ["--train-only"]
    cec.main()


def keys(seed: int, role: str, sequence: str) -> None:
    """공통 입력의 과거·대칭 키를 생성하고 실제 지연 상한을 검사한다."""
    import torch
    from _causal_branch_fusion import nsd_embedding_causal
    from _lag_branch_fusion import TruncatedCache, nsd_embedding_lag
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sensor, seq, paths, min_past = inputs(role, sequence)
    dest = seed_dir(seed) / f"keys_{role}_{sensor}_{seq}.npz"
    if dest.exists():
        return
    started = time.monotonic()
    with np.load(paths["cache"]) as archive, np.load(paths["layouts"]) as layouts:
        cache = ArrayCache({k: archive[k] for k in archive.files})
        layout = layouts["nsd_layouts"].astype(np.float32)
        np.testing.assert_array_equal(cache["scan_ids"], layouts["scan_ids"])
    raw = cache["descriptors"].astype(np.float32)
    r = np.load(RVEC).astype(np.float32)
    expected = no_context_key(raw, r)[:, :288]
    saved, prefix_rows, artifacts = {}, [], {}
    for arm in (("past", "symmetric") if role == "val" else ("symmetric",)):
        cfg, net, clf, blob, r, ckpt = model(seed, arm)
        embed = nsd_embedding_lag if arm == "symmetric" else nsd_embedding_causal
        emb = embed(cache, sensor, raw, layout, cfg, net, clf, blob, r, "cuda")
        assert emb.shape == (len(raw), 416) and emb.dtype == np.float32
        assert np.isfinite(emb).all()
        assert np.abs(normalize(emb[:, :288]) - expected).max() < 3e-7
        saved[arm] = emb
        artifacts[str(ckpt.relative_to(REPO))] = sha256(ckpt)
        artifacts[str(classifier_path(seed, arm).relative_to(REPO))] = sha256(classifier_path(seed, arm))
        if arm == "symmetric" and role == "val" and sequence in ("kitti/09", "helipr/Roundabout01", "mulran/Sejong01"):
            cpu_model = None
            for k in (40, 300, 900):
                n = k + LAG + 1
                cut = embed(TruncatedCache(cache, n), sensor, raw[:n], layout[:n],
                            cfg, net, clf, blob, r, "cuda")
                error = float(np.abs(cut[:k + 1] - emb[:k + 1]).max())
                row = {"k": k, "max_error_rows_le_k": error,
                       "row_k_error": float(np.abs(cut[k] - emb[k]).max()),
                       "gpu_original_tolerance_passed": error < 2e-6}
                if error >= 2e-6:
                    if cpu_model is None:
                        cpu_cfg, cpu_model, cpu_clf, cpu_blob, cpu_r, _ = model(seed, arm, "cpu")
                        cpu_full = embed(cache, sensor, raw, layout, cpu_cfg, cpu_model,
                                         cpu_clf, cpu_blob, cpu_r, "cpu")
                    cpu_cut = embed(TruncatedCache(cache, n), sensor, raw[:n], layout[:n],
                                    cpu_cfg, cpu_model, cpu_clf, cpu_blob, cpu_r, "cpu")
                    row["cpu_reference_max_error"] = float(np.abs(cpu_cut[:k + 1] - cpu_full[:k + 1]).max())
                    row["verification"] = "GPU 원래 오차 기준 미달, 별도 CPU 참조로 미래 입력 독립성 확인"
                prefix_rows.append(row)
                assert prefix_verified_error(row) < 2e-6, row
            if cpu_model is not None:
                del cpu_model, cpu_clf, cpu_full
        del net, clf
        gc.collect()
        torch.cuda.empty_cache()
    saved["scan_ids"] = cache["scan_ids"]
    saved["groups"] = direction_groups(cache["poses_gt"][:, :3, 3], min_past)
    meta = {"seed": seed, "role": role, "sequence": sequence, "prefix_checks": prefix_rows,
            "runtime_s": time.monotonic() - started, "artifacts_sha256": artifacts,
            "inputs_sha256": {str(p.relative_to(REPO)): sha256(p) for p in paths.values()},
            "protocol_sha256": sha256(OUT / "protocol.json")}
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **saved, meta=json.dumps(meta))
    tmp.replace(dest)
    print("KEYS_DONE", seed, role, sequence, meta["runtime_s"], flush=True)


def bank_for(op: str, spectrum) -> tuple:
    """현재 운영점의 보조 색인과 정렬 입력을 만든다."""
    if op == OPS[0]:
        return variant_inputs(spectrum["sk_re"].astype(np.float32) + 1j * spectrum["sk_im"].astype(np.float32), 8)
    return hspec_inputs(spectrum["raw_height"], 12)


def tuple_grid(positions: np.ndarray, emb: np.ndarray, bank: tuple,
               queries: np.ndarray, weights: list) -> dict:
    """기록 생성과 같은 5m 이하 정답·과거 후보·동률 규칙으로 격자를 센다."""
    hkey, hfft, hnorm, column = bank
    correct = {str(w): 0 for w in weights}
    for q in queries:
        q = int(q)
        a = prefix_topk(emb, q, q - 29, 100)
        b = prefix_topk(hkey, q, q - 29, 100)
        candidates = np.unique(np.r_[a, b])
        assert len(candidates) and candidates.max() <= q - 30
        good = np.linalg.norm(positions[candidates] - positions[q], axis=1) <= 5
        sn = minmax(emb[candidates] @ emb[q])
        sh = minmax(hkey[candidates] @ hkey[q])
        sa = minmax(align_scores(hfft, hnorm, q, candidates))
        sc = minmax(column_align_scores(column, q, candidates))
        for w in weights:
            scores = w[0] * sn + w[1] * sh + w[2] * sa + w[3] * sc
            correct[str(w)] += bool(good[int(np.argmax(scores))])
    return {k: v / max(len(queries), 1) for k, v in correct.items()}


def grid(seed: int, sequence: str) -> None:
    """지연을 감당할 수 있는 검증 양성만으로 기존 융합 격자를 평가한다."""
    spec = protocol()
    sensor, seq, paths, min_past = inputs("val", sequence)
    with np.load(paths["cache"]) as cache, np.load(paths["spec"]) as spectrum, \
            np.load(seed_dir(seed) / f"keys_val_{sensor}_{seq}.npz") as saved:
        pos = cache["poses_gt"][:, :3, 3]
        q = np.flatnonzero(min_past <= 5)
        q = q[q < len(pos) - LAG]
        weights = [tuple(float(v) for v in w) for w in spec["tuple_grid"]]
        for op in OPS:
            path = seed_dir(seed) / f"grid_{op}_{sensor}_{seq}.json"
            if path.exists():
                continue
            values = tuple_grid(pos, saved["symmetric"], bank_for(op, spectrum), q, weights)
            write_json(path, {"seed": seed, "sequence": sequence, "n_queries": len(q),
                             "tuples": values, "protocol_sha256": sha256(OUT / "protocol.json")})
    print("GRID_DONE", seed, sequence, flush=True)


def select_tuple(seed: int) -> None:
    """검증 센서 계열의 질의 합산 R@1 평균으로 융합 가중치를 선택한다."""
    spec = protocol()
    target = seed_dir(seed) / "tuple_selection.json"
    if target.exists():
        return
    groups = {}
    for sensor, seq in VAL:
        groups.setdefault(FAMILY.get(sensor, sensor), []).append(sensor + "/" + seq)
    chosen = {}
    for op in OPS:
        rows = {s + "/" + q: json.loads((seed_dir(seed) / f"grid_{op}_{s}_{q}.json").read_text()) for s, q in VAL}
        weights = [str(tuple(float(v) for v in w)) for w in spec["tuple_grid"]]
        def family_score(w: str, names: list) -> float:
            n = sum(rows[k]["n_queries"] for k in names)
            assert n > 0
            return sum(rows[k]["n_queries"] * rows[k]["tuples"][w] for k in names) / n
        values = {w: statistics.mean(family_score(w, names) for names in groups.values()) for w in weights}
        best = max(weights, key=values.get)
        chosen[op] = {"tuple": list(ast.literal_eval(best)), "family_macro_R1": values[best],
                      "per_family": {f: family_score(best, names) for f, names in groups.items()},
                      "n_positive": sum(r["n_queries"] for r in rows.values()), "grid_scores": values}
    write_json(target, {"ops": chosen, "protocol_sha256": sha256(OUT / "protocol.json")})
    print("TUPLE_SELECTED", seed, {op: row["tuple"] for op, row in chosen.items()}, flush=True)


def score(seed: int, role: str, sequence: str, mode: str) -> None:
    """고정 비교 또는 선택한 대칭 설정에서 후보를 다시 검색한다."""
    spec = protocol()
    assert mode in ("fixed", "selected")
    assert role == "val" or mode == "selected"
    sensor, seq, paths, min_past = inputs(role, sequence)
    selection = (spec["fixed_selection"] if mode == "fixed" else
                 json.loads((seed_dir(seed) / "tuple_selection.json").read_text())["ops"])
    key_path = seed_dir(seed) / f"keys_{role}_{sensor}_{seq}.npz"
    key_hash = sha256(key_path)
    with np.load(paths["cache"]) as cache, np.load(paths["spec"]) as spectrum, np.load(key_path) as saved:
        np.testing.assert_array_equal(cache["scan_ids"], spectrum["scan_ids"])
        np.testing.assert_array_equal(cache["scan_ids"], saved["scan_ids"])
        positions = cache["poses_gt"][:, :3, 3]
        no_context = no_context_key(cache["descriptors"], np.load(RVEC))
        for op in OPS:
            bank = bank_for(op, spectrum)
            arms = ARMS if mode == "fixed" else ("no_context", "symmetric")
            for arm in arms:
                path = seed_dir(seed) / f"{role}_{mode}_{arm}_{op}_{sensor}_{seq}.npz"
                if path.exists():
                    continue
                emb = no_context if arm == "no_context" else saved[arm]
                weights = {seed: tuple(float(v) for v in selection[op]["tuple"])}
                arrays, reused = None, None
                for candidate in sorted(OUT.glob(f"s*/{role}_*_{arm}_{op}_{sensor}_{seq}.npz")):
                    with np.load(candidate) as old:
                        old_meta = json.loads(str(old["meta"]))
                        same_key = arm == "no_context" or old_meta["keys_sha256"] == key_hash
                        if (same_key and old_meta["tuple"] == selection[op]["tuple"]
                                and old_meta["protocol_sha256"] == sha256(OUT / "protocol.json")):
                            arrays = {k: old[k] for k in old.files if k != "meta"}
                            np.testing.assert_array_equal(arrays["query_idx"], np.arange(30, len(emb) - LAG))
                            np.testing.assert_array_equal(arrays["min_past_gt_dist"], min_past[arrays["query_idx"]])
                            np.testing.assert_array_equal(arrays["group"], saved["groups"][arrays["query_idx"]])
                            reused = {"path": str(candidate.relative_to(REPO)), "sha256": sha256(candidate)}
                            break
                if arrays is None:
                    arrays = evaluate_keys(emb, bank, positions, min_past, weights)[seed]
                    keep = arrays["query_idx"] < len(emb) - LAG
                    arrays = {k: v[keep] for k, v in arrays.items()}
                    arrays["group"] = saved["groups"][arrays["query_idx"]]
                meta = {"seed": seed, "role": role, "mode": mode, "arm": arm, "op": op,
                        "sequence": sequence, "tuple": selection[op]["tuple"],
                        "keys_sha256": key_hash, "protocol_sha256": sha256(OUT / "protocol.json"),
                        "reused_identical_settings": reused}
                if role == "test":
                    meta["freeze_sha256"] = sha256(OUT / "freeze.json")
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **arrays, meta=json.dumps(meta))
                tmp.replace(path)
                print("SCORE_DONE", seed, role, mode, sequence, op, arm, flush=True)


def select_confidence(seed: int) -> None:
    """검증 결과만으로 판정 점수와 수락 임계값을 고정한다."""
    from _pdv2_c3 import _family_macro_tau_star
    target = seed_dir(seed) / "confidence_selection.json"
    if target.exists():
        return
    selected = json.loads((seed_dir(seed) / "tuple_selection.json").read_text())["ops"]
    results, excluded = {}, []
    for op in OPS:
        sweeps, groups = {}, {}
        for sensor, seq in VAL:
            name = sensor + "/" + seq
            with np.load(seed_dir(seed) / f"val_selected_symmetric_{op}_{sensor}_{seq}.npz") as z:
                qc = classify_queries(z["min_past_gt_dist"])
                grid_row = json.loads((seed_dir(seed) / f"grid_{op}_{sensor}_{seq}.json").read_text())
                pos = z["min_past_gt_dist"] <= 5
                replay = float((z["top1_geo_dist"][pos] <= 5).mean()) if pos.any() else 0.
                weight_key = str(tuple(float(v) for v in selected[op]["tuple"]))
                assert replay == grid_row["tuples"][weight_key], (seed, name, op, replay)
                assert int(pos.sum()) == grid_row["n_queries"]
                if not qc["n_pos"]:
                    excluded.append(name)
                    continue
                groups.setdefault(FAMILY.get(sensor, sensor), []).append(name)
                sweeps[name] = {stat: pr_sweep(z[column], z["top1_geo_dist"], qc) for stat, column in STATS.items()}
        ap = {stat: statistics.mean(statistics.mean(sweeps[k][stat]["AP"] for k in names)
                                   for names in groups.values()) for stat in STATS}
        best = max(STATS, key=ap.get)
        decision = "absolute" if ap["absolute"] >= ap[best] - .005 else best
        tau, f1, per_f1 = _family_macro_tau_star({k: row[decision] for k, row in sweeps.items()}, groups)
        results[op] = {"tuple": selected[op]["tuple"], "decision": decision, "tau_star": tau,
                       "family_macro_AP": ap, "val_family_macro_F1": f1,
                       "per_sequence_F1_at_tau": per_f1}
    write_json(target, {"ops": results, "excluded_no_positive": sorted(set(excluded)),
                       "protocol_sha256": sha256(OUT / "protocol.json")})
    print("CONFIDENCE_SELECTED", seed, results, flush=True)


def freeze() -> None:
    """세 시드의 전체 선택을 한 번 고정하고 시험 접근을 연다."""
    spec = protocol()
    path = OUT / "freeze.json"
    if path.exists():
        frozen_guard()
        return
    hashes = effective_hashes(spec)
    files = {REPO / p for p in hashes}
    for name, expected in hashes.items():
        assert sha256(REPO / name) == expected, name
    runtime = set(files)
    prefix_checks = []
    for seed in SEEDS:
        files.update((checkpoint_path(seed), classifier_path(seed), seed_dir(seed) / "tuple_selection.json",
                      seed_dir(seed) / "confidence_selection.json"))
        runtime.update((checkpoint_path(seed), classifier_path(seed), seed_dir(seed) / "tuple_selection.json",
                        seed_dir(seed) / "confidence_selection.json"))
        assert len(list(seed_dir(seed).glob("val_fixed_*.npz"))) == len(VAL) * len(OPS) * len(ARMS)
        assert len(list(seed_dir(seed).glob("val_selected_*.npz"))) == len(VAL) * len(OPS) * 2
        for sensor, seq in VAL:
            key_path = seed_dir(seed) / f"keys_val_{sensor}_{seq}.npz"
            files.add(key_path)
            with np.load(key_path) as saved:
                meta = json.loads(str(saved["meta"]))
                assert meta["protocol_sha256"] == sha256(OUT / "protocol.json")
                prefix_checks.extend(meta["prefix_checks"])
                for name, expected in {**meta["inputs_sha256"], **meta["artifacts_sha256"]}.items():
                    assert sha256(REPO / name) == expected, name
                files.update(REPO / name for name in meta["inputs_sha256"])
            for mode, arms in (("fixed", ARMS), ("selected", ("no_context", "symmetric"))):
                for op in OPS:
                    for arm in arms:
                        p = seed_dir(seed) / f"val_{mode}_{arm}_{op}_{sensor}_{seq}.npz"
                        assert p.exists(), p
                        files.add(p)
    assert len(prefix_checks) == 27 and max(prefix_verified_error(p) for p in prefix_checks) < 2e-6
    write_json(path, {"protocol_sha256": sha256(OUT / "protocol.json"),
                     "created_unix": time.time(), "seeds": list(SEEDS),
                     "prefix_checks": prefix_checks,
                     "runtime_verify": sorted(str(p.relative_to(REPO)) for p in runtime),
                     "file_identity": {str(p.relative_to(REPO)): [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(files)},
                     "sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(files)}})
    print("ALL_SEEDS_FROZEN", flush=True)


def main() -> None:
    """선택 단계와 시험 단계를 명시적으로 구분해 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("protocol", "clf", "keys", "grid", "select-tuple", "score", "select-confidence", "freeze"))
    parser.add_argument("--seed", type=int, choices=SEEDS, default=1)
    parser.add_argument("--role", choices=("val", "test"), default="val")
    parser.add_argument("--seq")
    parser.add_argument("--mode", choices=("fixed", "selected"), default="fixed")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if args.action == "protocol": protocol()
    elif args.action == "clf": train_classifier(args.seed, args.seq, args.train_only)
    elif args.action == "keys": keys(args.seed, args.role, args.seq)
    elif args.action == "grid": grid(args.seed, args.seq)
    elif args.action == "select-tuple": select_tuple(args.seed)
    elif args.action == "score": score(args.seed, args.role, args.seq, args.mode)
    elif args.action == "select-confidence": select_confidence(args.seed)
    elif args.action == "freeze": freeze()


if __name__ == "__main__":
    main()
