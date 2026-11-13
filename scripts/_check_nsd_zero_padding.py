#!/usr/bin/env python3
"""각 도시의 가장 짧은 시험 시퀀스에서 416차원 0 채움과 288차원을 대조한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _pdv2_no_context_ablation import (
    CITY, OPS, OUT, REPO, SEEDS, evaluate_keys, ledger_path, local_path,
    metrics, no_context_key, sha256, write_json,
)
from _xs_eval_lib import hspec_inputs, variant_inputs


def top_indices(scores: np.ndarray) -> np.ndarray:
    """원래 prefix_topk와 같은 부분 선택·안정 정렬을 적용한다."""
    k = min(len(scores), 100)
    top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(k)
    return top[np.argsort(-scores[top], kind="stable")]


def main() -> None:
    """분할·가중치를 바꾸지 않고 저장 형식만 다른 두 경우를 확인한다."""
    protocol = json.loads((OUT / "protocol.json").read_text())
    selected = {}
    for city, sequences in CITY.items():
        lengths = {}
        for sequence in sequences:
            tag = sequence.replace("/", "_")
            with np.load(OUT / f"test_pbev128_main_s1_{tag}.npz") as z:
                lengths[sequence] = len(z["query_idx"])
        selected[city] = min(lengths, key=lambda s: (lengths[s], s))
    result = {"selection": "성능과 무관하게 도시마다 질의 수가 가장 적은 시퀀스 1개",
              "scope": "전체 시험이 아닌 6시퀀스 확인. 수학적 영벡터 불변성과 유한 정밀도 계산을 구분한다.",
              "selected": selected, "per_sequence": {},
              "script_sha256": sha256(Path(__file__)),
              "source_protocol_sha256": sha256(OUT / "protocol.json")}
    for city, sequence in selected.items():
        sensor, seq = sequence.split("/")
        tag = sequence.replace("/", "_")
        source = ledger_path("test", 1, OPS[0], sensor, seq)
        with np.load(source) as z:
            meta = json.loads(str(z["meta"]))
            queries = z["query_idx"]
            min_past = np.full(meta["n_frames"], np.nan)
            min_past[queries] = z["min_past_gt_dist"]
        cache_path = local_path(meta["caches"]["cache"])
        spec_path = local_path(meta["caches"]["spec"])
        with np.load(cache_path) as cache, np.load(spec_path) as spectrum:
            rpath = local_path(meta["artifacts"]["rvec"])
            padded = no_context_key(cache["descriptors"], np.load(rpath))
            compact = np.ascontiguousarray(padded[:, :288])
            positions = cache["poses_gt"][:, :3, 3]
            row = {"city": city, "n_queries": len(queries), "prefix_score_max_abs": 0.,
                   "key_top1_changed": 0, "key_top100_set_changed": 0, "ops": {},
                   "inputs_sha256": {str(p.relative_to(REPO)): sha256(p)
                                      for p in (cache_path, spec_path, rpath)}}
            for q in queries:
                a = padded[:q - 29] @ padded[q]
                b = compact[:q - 29] @ compact[q]
                row["prefix_score_max_abs"] = max(row["prefix_score_max_abs"], float(np.max(np.abs(a - b))))
                ai, bi = top_indices(a), top_indices(b)
                row["key_top1_changed"] += int(ai[0] != bi[0])
                row["key_top100_set_changed"] += int(not np.array_equal(np.sort(ai), np.sort(bi)))
            for op in OPS:
                weights = {s: tuple(float(v) for v in protocol["selection_by_seed"][str(s)][op]["tuple"])
                           for s in SEEDS}
                bank = (variant_inputs(spectrum["sk_re"].astype(np.float32)
                                       + 1j * spectrum["sk_im"].astype(np.float32), 8)
                        if op == "pbev128_main" else hspec_inputs(spectrum["raw_height"], 12))
                compact_results = evaluate_keys(compact, bank, positions, min_past, weights)
                row["ops"][op] = {}
                for seed, new in compact_results.items():
                    padded_path = OUT / f"test_{op}_s{seed}_{tag}.npz"
                    with np.load(padded_path) as old:
                        np.testing.assert_array_equal(old["query_idx"], new["query_idx"])
                        tau = protocol["selection_by_seed"][str(seed)][op]["tau_star"]
                        before, after = metrics(old, tau), metrics(new, tau)
                        row["ops"][op][str(seed)] = {
                            "j_star_changed": int(np.count_nonzero(old["j_star"] != new["j_star"])),
                            "correctness_changed": int(np.count_nonzero((old["top1_geo_dist"] <= 5) != (new["top1_geo_dist"] <= 5))),
                            "n_candidates_changed": int(np.count_nonzero(old["n_candidates"] != new["n_candidates"])),
                            "c_spec_max_abs": float(np.max(np.abs(old["c_spec"] - new["c_spec"]))),
                            "score_abs_max_abs": float(np.max(np.abs(old["score_abs"] - new["score_abs"]))),
                            "metric_delta": {m: after[m] - before[m] for m in before},
                            "padded_ledger_sha256": sha256(padded_path)}
            result["per_sequence"][sequence] = row
            print(sequence, row["n_queries"], row["prefix_score_max_abs"], row["key_top100_set_changed"], flush=True)
    rows = list(result["per_sequence"].values())
    pairs = [r["ops"][op][str(s)] for r in rows for op in OPS for s in SEEDS]
    result["summary"] = {"n_queries": sum(r["n_queries"] for r in rows),
                         "prefix_score_max_abs": max(r["prefix_score_max_abs"] for r in rows),
                         "key_top1_changed": sum(r["key_top1_changed"] for r in rows),
                         "key_top100_set_changed": sum(r["key_top100_set_changed"] for r in rows),
                         "total_final_matches_compared": sum(r["n_queries"] for r in rows) * 6,
                         "final_match_changed": sum(p["j_star_changed"] for p in pairs),
                         "correctness_changed": sum(p["correctness_changed"] for p in pairs),
                         "metric_delta_max_abs": max(abs(v) for p in pairs for v in p["metric_delta"].values())}
    write_json(OUT / "zero_padding_check.json", result)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
