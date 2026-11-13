#!/usr/bin/env python3
"""현재 NSD에서 문맥만 제거하고 기존 세 시드의 나머지 설정을 고정한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/_pdv2_no_context_ablation.py --run

동일 키프레임·재계량·100+100 합집합·정렬·융합·판정을 사용한다.
검색 키는 [normalize(r*d); zeros(128)]이다. 재학습·검증 재선택은 없다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src"), str(REPO)]
from _causal_pr_dump import dump_op, prefix_topk  # noqa: E402
from _pdv2_pilot import CITY, TEST, VAL  # noqa: E402
from _pr_eval_lib import AGG_METRICS, classify_queries, operating_point, pr_sweep  # noqa: E402
from _xs_eval_lib import (  # noqa: E402
    align_scores, column_align_scores, hspec_inputs, minmax, normalize, variant_inputs,
)

OUT = REPO / "results/pdv2_no_context_20261013"
OPS = ("pbev128_main", "hspec12")
SEEDS = (1, 2, 3)


def sha256(path: Path) -> str:
    """입력·실행 코드 해시를 구한다."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """NaN 없는 JSON을 원자적으로 쓴다."""
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def baseline_root(seed: int) -> Path:
    """현재 원고의 시드별 평가 원천을 반환한다."""
    return REPO / f"results/pdv2_c3_bevclip_s{seed}/odom"


def ledger_path(role: str, seed: int, op: str, sensor: str, seq: str) -> Path:
    """기존 질의 기록의 경로를 반환한다."""
    prefix = "valdump" if role == "val" else "dump"
    return baseline_root(seed) / f"{prefix}_pdv2_causal_{op}_s{seed}_{sensor}_{seq}.npz"


def local_path(value: str) -> Path:
    """기록의 컨테이너 절대 경로를 현재 리포 경로로 바꾼다."""
    return REPO / value.split("/Neural-Spectral-Codec/", 1)[1]


def no_context_key(raw: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """재계량 키를 유지하고 문맥 128개만 0으로 만든다."""
    assert raw.ndim == 2 and raw.shape[1] == 288 and weights.shape == (288,)
    key = np.zeros((len(raw), 416), dtype=np.float32)
    key[:, :288] = normalize(raw.astype(np.float32) * weights.astype(np.float32))
    assert np.isfinite(key).all() and np.all(np.linalg.norm(key, axis=1) > .99999)
    return key


def protocol(out: Path) -> dict:
    """시험 실행 전에 제거 범위·비교 규칙·원래 선택값을 고정한다."""
    files = [Path(__file__), REPO / "scripts/_causal_pr_dump.py",
             REPO / "scripts/_xs_eval_lib.py", REPO / "scripts/_pr_eval_lib.py",
             REPO / "scripts/_pdv2_pilot.py", REPO / "src/gnn/model.py",
             REPO / "artifacts/key_remetrize_r_pdv2.npy"]
    selections = {}
    for seed in SEEDS:
        path = baseline_root(seed) / "confidence_selection_align.json"
        files.extend((path, baseline_root(seed) / "pr_summary_align.json"))
        selections[str(seed)] = json.loads(path.read_text())["ops"]
        assert all(v["decision"] == "c_spec" for v in selections[str(seed)].values())
    result = {"schema": "nsd-no-context-frozen-v1", "date": "2026-10-13",
              "condition": "odom, bevclip, corrected NCLT d", "seeds": list(SEEDS),
              "change": "최종 검색 키의 문맥 128개를 0으로 한다.",
              "key": "[normalize(r*d); zeros(128)], float32, 416D",
              "graph": "두 통과·간선 재선택의 출력은 문맥 외 키 블록에 영향을 주지 않아 계산을 생략한다. 저장된 현재 키 블록과 수치 대조한다.",
              "frozen": "키프레임, d, r, 보조 색인, 과거 j<=q-30, 각 top100 합집합(중복 보충 없음), 네 점수, 시드별 기존 융합과 c_spec 임계값",
              "candidate_note": "제거된 키로 후보를 다시 검색한다. 같은 후보 수 상한·검색 규칙이며 후보 ID는 달라질 수 있다.",
              "selection": "재학습·검증 재선택 없음. 문맥 유무 각각의 최적 구성 비교가 아니다.",
              "scope": "이미 본 시험 자료에 대한 요청된 사후 제거 실험. 기존 pd2026 사전 등록 실험과 별개.",
              "aggregation": "시퀀스 평균→도시 6개 평균, 세 원래 시드 설정의 평균·표본 표준편차. 문맥 없는 키 자체는 시드와 무관.",
              "roster": {"val": [s + "/" + q for s, q in VAL],
                         "test": [s + "/" + q for s, q in TEST]},
              "selection_by_seed": selections,
              "sha256": {str(p.relative_to(REPO)): sha256(p) for p in files}}
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == result, "프로토콜/코드/입력이 달라졌다"
    else:
        write_json(path, result)
    return result


def evaluate_keys(emb: np.ndarray, bank: tuple, positions: np.ndarray,
                  min_past: np.ndarray, weights_by_seed: dict) -> dict:
    """원래 검색·점수 계산으로 문맥 없는 키의 질의 기록을 만든다."""
    hkey, hfft, hnorm, column = bank
    queries = np.arange(30, len(emb), dtype=np.int64)
    n = len(queries)
    common = {"query_idx": queries, "min_past_gt_dist": min_past[queries],
              "n_candidates": np.empty(n, np.int64),
              "key_top1_geo": np.empty(n), "key_hit": np.empty(n, bool),
              "aux_hit": np.empty(n, bool), "union_hit": np.empty(n, bool)}
    per = {seed: {"j_star": np.empty(n, np.int64), "top1_geo_dist": np.empty(n),
                  "c_spec": np.empty(n), "c_col": np.empty(n),
                  "score_abs": np.empty(n), "score_margin": np.empty(n)}
           for seed in weights_by_seed}
    for qi, query in enumerate(queries):
        t = int(query)
        a = prefix_topk(emb, t, t - 29, 100)
        b = prefix_topk(hkey, t, t - 29, 100)
        candidates = np.unique(np.r_[a, b])
        assert candidates.max() <= t - 30 and len(candidates) <= 200
        raw_a = align_scores(hfft, hnorm, t, candidates)
        raw_c = column_align_scores(column, t, candidates)
        sn, sh = minmax(emb[candidates] @ emb[t]), minmax(hkey[candidates] @ hkey[t])
        sa, sc = minmax(raw_a), minmax(raw_c)
        distances = np.linalg.norm(positions[candidates] - positions[t], axis=1)
        common["n_candidates"][qi] = len(candidates)
        common["key_top1_geo"][qi] = np.linalg.norm(positions[a[0]] - positions[t])
        common["key_hit"][qi] = np.any(np.linalg.norm(positions[a] - positions[t], axis=1) <= 5)
        common["aux_hit"][qi] = np.any(np.linalg.norm(positions[b] - positions[t], axis=1) <= 5)
        common["union_hit"][qi] = np.any(distances <= 5)
        for seed, w in weights_by_seed.items():
            assert all(isinstance(x, float) for x in w)
            score = w[0] * sn + w[1] * sh + w[2] * sa + w[3] * sc
            assert score.dtype == np.float32
            order = np.argsort(-score, kind="stable")
            best = int(order[0])
            j = int(candidates[best])
            values = {"j_star": j, "top1_geo_dist": distances[best],
                      "c_spec": raw_a[best], "c_col": raw_c[best],
                      "score_abs": float(emb[t] @ emb[j]),
                      "score_margin": float(score[best] - score[order[1]]) if len(order) > 1 else 0.}
            for name, value in values.items():
                per[seed][name][qi] = value
        if qi and qi % 4000 == 0:
            print(f"  {qi}/{n}", flush=True)
    return {seed: {**common, **arrays} for seed, arrays in per.items()}


def worker(role: str, sequence: str, out: Path, smoke: bool) -> None:
    """한 시퀀스의 같은 후보·정렬 계산을 세 고정 융합 설정에 적용한다."""
    started = time.monotonic()
    spec = protocol(out)
    sensor, seq = sequence.split("/")
    assert sequence in spec["roster"][role]
    original = ledger_path(role, 1, OPS[0], sensor, seq)
    with np.load(original) as archive:
        meta = json.loads(str(archive["meta"]))
        q = archive["query_idx"]
        min_past = np.full(meta["n_frames"], np.nan)
        min_past[q] = archive["min_past_gt_dist"]
    paths = {k: local_path(meta["caches"][k]) for k in ("cache", "spec")}
    with np.load(paths["cache"]) as cache, np.load(paths["spec"]) as spectrum:
        np.testing.assert_array_equal(cache["scan_ids"], spectrum["scan_ids"])
        positions = cache["poses_gt"][:, :3, 3]
        raw = cache["descriptors"].astype(np.float32)
        rpath = local_path(meta["artifacts"]["rvec"])
        emb = no_context_key(raw, np.load(rpath))
        block_difference = None
        if role == "test":
            key_path = REPO / f"results/pdv2_collision_20261013/keys_s1_{sensor}_{seq}.npz"
            with np.load(key_path) as keys:
                np.testing.assert_array_equal(keys["scan_ids"], cache["scan_ids"])
                recovered = normalize(keys["context"][:, :288])
            block_difference = float(np.max(np.abs(recovered - emb[:, :288])))
            np.testing.assert_allclose(recovered, emb[:, :288], atol=2e-7, rtol=2e-6)
            paths["saved_current_keys"] = key_path
        for op in OPS:
            print(role, sequence, op, flush=True)
            bank = (variant_inputs(spectrum["sk_re"].astype(np.float32)
                                   + 1j * spectrum["sk_im"].astype(np.float32), 8)
                    if op == "pbev128_main" else hspec_inputs(spectrum["raw_height"], 12))
            weights = {s: tuple(float(v) for v in spec["selection_by_seed"][str(s)][op]["tuple"])
                       for s in SEEDS}
            per = evaluate_keys(emb, bank, positions, min_past, weights)
            for seed, arrays in per.items():
                source = ledger_path(role, seed, op, sensor, seq)
                with np.load(source) as old:
                    source_meta = json.loads(str(old["meta"]))
                    np.testing.assert_array_equal(old["query_idx"], arrays["query_idx"])
                    np.testing.assert_array_equal(old["min_past_gt_dist"], arrays["min_past_gt_dist"])
                assert source_meta["tuple"] == list(weights[seed])
                assert source_meta["pool_k"] == 100 and source_meta["min_index_gap"] == 30
                assert local_path(source_meta["artifacts"]["rvec"]) == rpath
                for name in ("cache", "spec"):
                    assert local_path(source_meta["caches"][name]) == paths[name]
                if smoke:
                    check = dump_op(emb, *bank, positions, min_past, weights[seed], 100)
                    for name, value in check.items():
                        np.testing.assert_allclose(arrays[name], value, atol=1e-12, rtol=0)
                report = {"sequence": sequence, "role": role, "seed_setting": seed,
                          "op": op, "protocol_sha256": sha256(out / "protocol.json"),
                          "source": str(source.relative_to(REPO)), "source_sha256": sha256(source),
                          "inputs_sha256": {str(p.relative_to(REPO)): sha256(p) for p in paths.values()},
                          "key_block_max_abs_difference": block_difference,
                          "canonical_dump_checked": smoke, "n_frames": len(emb),
                          "runtime_s": time.monotonic() - started}
                path = out / f"{role}_{op}_s{seed}_{sensor}_{seq}.npz"
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, meta=json.dumps(report), **arrays)
                tmp.replace(path)
    write_json(out / f"done_{role}_{sensor}_{seq}.json", {"sequence": sequence,
               "protocol_sha256": sha256(out / "protocol.json"),
               "runtime_s": time.monotonic() - started, "canonical_dump_checked": smoke})
    print(f"DONE {role} {sequence} {time.monotonic() - started:.1f}s", flush=True)


def metrics(arrays: dict, tau: float) -> dict:
    """원래 PR 규칙의 곡선 지표와 고정 임계값 지표를 계산한다."""
    qc = classify_queries(arrays["min_past_gt_dist"])
    sw = pr_sweep(arrays["c_spec"], arrays["top1_geo_dist"], qc)
    opp = operating_point(arrays["c_spec"], arrays["top1_geo_dist"], qc, tau)
    pos = qc["positive"]
    return {**{name: float(sw[name]) for name in AGG_METRICS},
            "R@1_revisit": float((arrays["top1_geo_dist"][pos] <= 5).mean()),
            "P_at_tau": float(opp["precision"]), "R_at_tau": float(opp["recall"])}


def aggregate(out: Path) -> None:
    """같은 질의의 기존 결과와 제거 결과를 도시·시드별로 비교한다."""
    spec = protocol(out)
    result = {"protocol_sha256": sha256(out / "protocol.json"), "ops": {}}
    for op in OPS:
        seed_results = {}
        for seed in SEEDS:
            per, counts = {}, {"positive": 0, "recover": 0, "regress": 0}
            tau = spec["selection_by_seed"][str(seed)][op]["tau_star"]
            published = json.loads((baseline_root(seed) / "pr_summary_align.json").read_text())[op]
            for sensor, seq in TEST:
                path = out / f"test_{op}_s{seed}_{sensor}_{seq}.npz"
                with np.load(path) as archive:
                    new = {k: archive[k] for k in archive.files if k != "meta"}
                    assert json.loads(str(archive["meta"]))["protocol_sha256"] == result["protocol_sha256"]
                source = ledger_path("test", seed, op, sensor, seq)
                with np.load(source) as old, np.load(source.with_name("align_" + source.name)) as align:
                    baseline = {k: old[k] for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist")}
                    baseline["c_spec"] = align["c_spec"]
                    np.testing.assert_array_equal(align["j_star"], baseline["j_star"])
                np.testing.assert_array_equal(new["query_idx"], baseline["query_idx"])
                np.testing.assert_array_equal(new["min_past_gt_dist"], baseline["min_past_gt_dist"])
                name = sensor + "/" + seq
                before, after = metrics(baseline, tau), metrics(new, tau)
                for metric, value in before.items():
                    assert abs(value - published["per_sequence"][name][metric]) < 1e-12
                pos = new["min_past_gt_dist"] <= 5
                old_ok, new_ok = baseline["top1_geo_dist"] <= 5, new["top1_geo_dist"] <= 5
                counts["positive"] += int(pos.sum())
                counts["recover"] += int((pos & ~old_ok & new_ok).sum())
                counts["regress"] += int((pos & old_ok & ~new_ok).sum())
                per[name] = {"with_context": before, "without_context": after,
                             "delta": {m: after[m] - before[m] for m in before},
                             "without_context_candidates": {k: float(new[k][pos].mean())
                                                              for k in ("key_hit", "aux_hit", "union_hit")}}
            cities = {c: {condition: {m: float(np.mean([per[s][condition][m] for s in names]))
                                      for m in per[names[0]][condition]}
                           for condition in ("with_context", "without_context", "delta", "without_context_candidates")}
                      for c, names in CITY.items()}
            macro = {condition: {m: float(np.mean([v[condition][m] for v in cities.values()]))
                                 for m in next(iter(cities.values()))[condition]}
                     for condition in next(iter(cities.values()))}
            seed_results[str(seed)] = {"city_macro6": macro, "city_groups": cities,
                                       "per_sequence": per, "pooled_transitions": counts}
        stats = {condition: {m: {"mean": float(np.mean([s["city_macro6"][condition][m] for s in seed_results.values()])),
                                 "std": float(np.std([s["city_macro6"][condition][m] for s in seed_results.values()], ddof=1))}
                             for m in seed_results["1"]["city_macro6"][condition]}
                 for condition in ("with_context", "without_context", "delta")}
        result["ops"][op] = {"by_seed_setting": seed_results, "across_three_settings": stats}
        print(op, json.dumps(stats, ensure_ascii=False), flush=True)
    write_json(out / "summary.json", result)


def main() -> None:
    """시퀀스마다 독립 프로세스로 고정 제거 실험을 실행한다."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--role", choices=("val", "test"), default="test")
    p.add_argument("--seq")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    protocol(a.out)
    if a.seq:
        worker(a.role, a.seq, a.out, a.smoke)
    elif a.run:
        roster = VAL if a.role == "val" else TEST

        def run_one(item: tuple) -> None:
            sensor, seq = item
            done = a.out / f"done_{a.role}_{sensor}_{seq}.json"
            if done.exists():
                assert json.loads(done.read_text())["protocol_sha256"] == sha256(a.out / "protocol.json")
                print("SKIP", a.role, sensor, seq, flush=True)
                return
            print("START", a.role, sensor, seq, flush=True)
            with (a.out / f"run_{a.role}_{sensor}_{seq}.log").open("w") as log:
                subprocess.run([sys.executable, str(Path(__file__)), "--out", str(a.out),
                                "--role", a.role, "--seq", sensor + "/" + seq],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print("DONE", a.role, sensor, seq, flush=True)

        with ThreadPoolExecutor(max_workers=a.workers) as executor:
            list(executor.map(run_one, roster))
    if a.aggregate:
        aggregate(a.out)


if __name__ == "__main__":
    main()
