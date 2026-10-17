#!/usr/bin/env python3
"""공식 코드로 다시 돌린 비교 방법의 시험 기록을 주표(대칭 문맥)와 같은 질의로 재집계한다.

_pdv2_main_baselines.py(이전 기록: results/pdv2_c3/odom의 SC++·RING#-L)와 같은 규칙이다.
- 질의: 30 <= q < N-15. NSD 대칭 문맥 기록(results/pdv2_symmetric3_20261013/s1)과 질의 번호가 같아야 한다.
- 후보: j <= q-30. 비교 방법은 각자 기록에 적힌 입력만 쓴다(새 추론·선택 없음).
- 집계: 시퀀스 → 도시 6개 평균 → 매크로. 양성이 없는 시퀀스(KITTI 07)는 AP·재현율 평균에서 뺀다.
시드 1 기록이 46개 모두 있는 방법만 넣고, 없는 방법은 missing에 적는다.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_pilot import CITY, TEST
from _pr_eval_lib import AGG_METRICS, classify_queries, pr_sweep

OUT = REPO / "results/pdv2_main_official_baselines"
COMMON = REPO / "results/pdv2_symmetric3_20261013"

# key, 결과 폴더, 기록 접두어, 점수 이름, 요약 파일, 요약 안의 키
METHODS = (
    ("scpp", "results/pdv2_scpp_official_hsens/apc/odom", "dump_scpp", "score_scpp",
     "pr_summary_baselines.json", "scpp"),
    ("ring", "results/pdv2_ring_official/odom", "dump_ring", "score_corr",
     "pr_summary_ring_official.json", "ring"),
    ("ringpp", "results/pdv2_ring_official/odom", "dump_ringpp", "score_corr",
     "pr_summary_ring_official.json", "ringpp"),
    # 주 점수 score_trans(이동 재정렬 1위의 −오차, 사전 등록 §1.5); 공식 PR 방식 score_rotbranch는 비교용
    ("ringsharp", "results/pdv2_ring_official/odom", "dump_ringsharp", "score_trans",
     "pr_summary_ring_official.json", "ringsharp:score_trans"),
    ("deepring", "results/pdv2_deepring_reimpl/odom", "dump_deepring", "score_corr",
     "pr_summary_baselines.json", "deepring"),
    ("disco", "results/pdv2_disco_official/odom", "dump_disco", "score_disco",
     "pr_summary_baselines.json", "disco"),
    ("helios", "results/pdv2_helios_official/odom", "dump_helios", "score_cos",
     "pr_summary_baselines.json", "helios"),
    # 주표는 공식 가운데 정렬 창(2026-10-14 사용자 결정), 보고 거리는 공식 L2
    ("seqot", "results/pdv2_seqot_official_centered/odom", "dump_seqotl2", "score_l2",
     "pr_summary_baselines.json", "seqotl2"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}

    def track(path: Path) -> Path:
        sources[str(path.relative_to(REPO))] = sha256(path)
        return path

    track(Path(__file__))
    for name in ("_pr_eval_lib.py", "_pdv2_pilot.py"):
        track(REPO / "scripts" / name)
    report = {"scope": "공식 코드로 다시 돌린 비교 방법의 시드 1 시험 기록 재집계; 새 선택·추론 없음",
              "queries": "30 <= q < N-15", "candidates": "j <= q-30",
              "information_access": ("비교 방법은 각자 기록의 입력을 쓴다. SeqOT은 공식 가운데 정렬 창으로 "
                                     "질의 원시 스캔 뒤 최대 10장을 보며, 이는 NSD의 q+15 대기 안이다."),
              "methods": {}, "missing": []}
    metrics = (*AGG_METRICS, "R@1")
    for method, folder, prefix, score_name, summary_name, summary_key in METHODS:
        folder = REPO / folder
        paths = {(s, q): folder / f"{prefix}_{s}_{q}.npz" for s, q in TEST}
        if not all(p.exists() for p in paths.values()):
            report["missing"].append(method)
            print(method, "missing ledgers, skipped", flush=True)
            continue
        original = json.loads(track(folder / summary_name).read_text())[summary_key]
        per = {}
        for sensor, seq in TEST:
            ref = track(COMMON / f"s1/test_selected_symmetric_pbev128_main_{sensor}_{seq}.npz")
            with np.load(track(paths[(sensor, seq)])) as z, np.load(ref) as n:
                qc = classify_queries(z["min_past_gt_dist"])
                old = original["per_sequence"][f"{sensor}/{seq}"]
                if qc["n_pos"]:                       # 방법 자체 집계를 그대로 재현하는지 확인
                    sw = pr_sweep(z[score_name], z["top1_geo_dist"], qc)
                    for k in AGG_METRICS:
                        assert abs(sw[k] - old[k]) < 1e-12, (method, sensor, seq, k, sw[k], old[k])
                ix = np.searchsorted(z["query_idx"], n["query_idx"])
                np.testing.assert_array_equal(z["query_idx"][ix], n["query_idx"])
                np.testing.assert_array_equal(z["min_past_gt_dist"][ix], n["min_past_gt_dist"])
                assert len(ix) == len(z["query_idx"]) - 15
                assert np.all(z["j_star"][ix] <= n["query_idx"] - 30)
                qc = classify_queries(z["min_past_gt_dist"][ix])
                row = {"n_queries": len(ix), "n_positive": qc["n_pos"],
                       "n_negative": qc["n_neg"], "n_ambiguous": qc["n_ambiguous"]}
                if qc["n_pos"]:
                    sw = pr_sweep(z[score_name][ix], z["top1_geo_dist"][ix], qc)
                    row.update({k: float(sw[k]) for k in AGG_METRICS})
                    row["R@1"] = float((z["top1_geo_dist"][ix][qc["positive"]] <= 5).mean())
                else:
                    row.update({k: None for k in metrics})
                    row["negative_score_quantiles"] = np.quantile(
                        z[score_name][ix][qc["negative"]], [0, .5, .95, 1]).tolist()
                per[f"{sensor}/{seq}"] = row
        groups = {city: {k: float(np.mean([per[s][k] for s in names if per[s]["n_positive"]]))
                         for k in metrics} for city, names in CITY.items()}
        macro = {k: float(np.mean([g[k] for g in groups.values()])) for k in metrics}
        entry = {"source": str(folder.relative_to(REPO)), "ledger_prefix": prefix, "score": score_name,
                 "per_sequence": per, "city_groups": groups, "macro": macro,
                 "n_queries": sum(r["n_queries"] for r in per.values()),
                 "n_positive": sum(r["n_positive"] for r in per.values()),
                 "excluded_no_positive": [s for s, r in per.items() if not r["n_positive"]]}
        assert entry["n_queries"] == 195726
        assert entry["n_positive"] == 61682
        assert entry["excluded_no_positive"] == ["kitti/07"]
        report["methods"][method] = entry
        print(method, {k: round(v, 4) for k, v in macro.items()}, flush=True)
    report["input_sha256"] = sources
    report["checks_passed"] = True
    write_json(OUT / "baselines_common.json", report)


if __name__ == "__main__":
    main()
