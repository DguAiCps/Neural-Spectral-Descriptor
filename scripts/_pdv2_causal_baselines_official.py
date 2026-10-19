#!/usr/bin/env python3
"""부록 과거 전용 표(tab:causal_pr, tab:v2_causal_pilot의 주행 추정 열)용 공식 비교 방법 재집계.

- 방법: 주표와 같은 공식 시드 1 실행(_pdv2_main_baselines_official.METHODS). 한 장 방법은 질의
  키프레임만 쓰므로 기록 그대로 과거 전용이다. SeqOT은 주표의 가운데 정렬 창 대신 같은 공식 코드의
  과거 창 실행(results/pdv2_seqot_official, 원시 스캔 [s-21, s])을 쓴다.
- 질의: 각 시퀀스의 전체 질의(q >= 30). 질의 번호와 과거 최근접 거리가 NSD 과거 전용 기록
  (results/pdv2_c3_bevclip_s1/odom/dump_pdv2_causal_pbev128_main_s1_*)과 같아야 한다.
- 집계: 시퀀스 → 도시 6개 평균 → 매크로. 방법 자체 요약과 NSD 과거 전용 요약을 같은 규칙으로
  다시 계산해 일치를 확인한다. 새 추론·선택은 없다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_main_baselines_official import METHODS, sha256, write_json
from _pdv2_pilot import CITY, TEST
from _pr_eval_lib import AGG_METRICS, classify_queries, pr_sweep

OUT = REPO / "results/pdv2_causal_official_baselines"
NSD = REPO / "results/pdv2_c3_bevclip_s1/odom"
CAUSAL_METHODS = tuple(m for m in METHODS if m[0] != "seqot") + (
    ("seqot", "results/pdv2_seqot_official/odom", "dump_seqotl2", "score_l2",
     "pr_summary_baselines.json", "seqotl2"),
)
METRICS = (*AGG_METRICS, "R@1_revisit")


def city_macro(per: dict) -> tuple[dict, dict]:
    groups = {city: {k: float(np.mean([per[s][k] for s in names])) for k in METRICS}
              for city, names in CITY.items()}
    return groups, {k: float(np.mean([g[k] for g in groups.values()])) for k in METRICS}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}

    def track(path: Path) -> Path:
        sources[str(path.relative_to(REPO))] = sha256(path)
        return path

    track(Path(__file__))
    for name in ("_pr_eval_lib.py", "_pdv2_pilot.py", "_pdv2_main_baselines_official.py"):
        track(REPO / "scripts" / name)
    # 같은 집계 규칙인지: NSD 과거 전용 요약의 매크로를 시퀀스별 값에서 다시 만든다.
    nsd = json.loads(track(NSD / "pr_summary_align.json").read_text())
    for op in ("pbev128_main", "hspec12"):
        _, macro = city_macro(nsd[op]["per_sequence"])
        for k in METRICS:
            assert abs(macro[k] - nsd[op]["city_macro6"][k]) < 1e-12, (op, k)
    report = {"_meta": {
        "scope": "공식 코드 비교 방법의 시드 1 시험 기록을 과거 전용 표의 전체 질의로 재집계; 새 선택·추론 없음",
        "queries": "각 시퀀스의 전체 질의(q >= 30), NSD 과거 전용 기록과 같은 질의 번호",
        "candidates": "j <= q-30",
        "past_only": "한 장 방법은 질의 키프레임만 쓴다. SeqOT은 공식 코드의 과거 창(원시 스캔 [s-21, s])",
        "condition": "odom"}}
    for method, folder, prefix, score_name, summary_name, summary_key in CAUSAL_METHODS:
        folder = REPO / folder
        original = json.loads(track(folder / summary_name).read_text())[summary_key]
        per = {}
        for sensor, seq in TEST:
            name = f"{sensor}/{seq}"
            ref = track(NSD / f"dump_pdv2_causal_pbev128_main_s1_{sensor}_{seq}.npz")
            with np.load(track(folder / f"{prefix}_{sensor}_{seq}.npz")) as z, np.load(ref) as n:
                np.testing.assert_array_equal(z["query_idx"], n["query_idx"])
                np.testing.assert_array_equal(z["min_past_gt_dist"], n["min_past_gt_dist"])
                assert np.all(z["j_star"] <= z["query_idx"] - 30)
                qc = classify_queries(z["min_past_gt_dist"])
                assert qc["n_pos"], (method, name)
                sw = pr_sweep(z[score_name], z["top1_geo_dist"], qc)
                row = {k: float(sw[k]) for k in AGG_METRICS}
                row["R@1_revisit"] = float((z["top1_geo_dist"][qc["positive"]] <= 5).mean())
                row.update(n_queries=len(z["query_idx"]), n_positive=qc["n_pos"])
            for k in METRICS:                       # 방법 자체 요약을 그대로 재현하는지
                assert abs(row[k] - original["per_sequence"][name][k]) < 1e-12, (method, name, k)
            per[name] = row
        groups, macro = city_macro(per)
        for k in METRICS:
            assert abs(macro[k] - original["city_macro6"][k]) < 1e-12, (method, k)
        report[method] = {"source": str(folder.relative_to(REPO)), "ledger_prefix": prefix,
                          "score": score_name, "per_sequence": per, "city_groups": groups,
                          "city_macro6": macro,
                          "n_queries": sum(r["n_queries"] for r in per.values())}
        print(method, {k: round(v, 4) for k, v in macro.items()}, flush=True)
    report["_meta"]["input_sha256"] = sources
    report["_meta"]["checks_passed"] = True
    write_json(OUT / "pr_summary_baselines.json", report)


if __name__ == "__main__":
    main()
