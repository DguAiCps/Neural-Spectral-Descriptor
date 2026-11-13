#!/usr/bin/env python3
"""RING++ 논문 표 II·III의 R@1과 F1이 공개 평가 정의 아래 서로 양립하는지 검사한다(데이터 불필요).

호스트에서 실행한다.
    python3 scripts/_review_ring_followup_consistency.py

근거: 공개 compute_PR_pairs(_external/RING/evaluation/plot_PR_curve.py:18-49)의 임계값 격자 끝은 1.0이고
공식 거리 1−score는 항상 1보다 작으므로 마지막 임계값에서 모든 질의를 수락한다. 이때 TP=정답 1위 수,
FP=N−TP, FN=0이므로 precision=P=TP/N, recall=1, F1=2P/(1+P)이다. 따라서 같은 질의 집합의 최대 F1은
2P/(1+P) 이상이고 P ≤ F1/(2−F1)이다.
(가) 논문 R@1이 공개 evaluate()처럼 전체 질의 분모라면 R@1 ≤ F1/(2−F1)이어야 한다.
(나) 논문 식 30의 글대로 양성 질의 분모라면 R@1·(N_pos/N) = P ≤ F1/(2−F1)이므로
     양성 비율 N_pos/N ≤ (F1/(2−F1))/R@1 이다.
표 값은 소수 4자리 반올림이므로 ±0.00005 구간에서 가장 관대한 경계를 쓴다.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_reproduction_orchestration_20261106/ring/paper_consistency.json"
NATIVE = REPO / "results/pdv2_review_native_20261105"
HALF = 0.00005
# results/pdv2_review_native_20261105/ringpp_paper.txt:847-881 (표 II), 883-897 (표 III)
TABLE_II = {
    "NCLT": {"RING++": (.7321, .7890, .8374), "RING++ (SG)": (.6309, .7150, .7776), "RING": (.7098, .7658, .8246),
             "RING (SG)": (.6582, .7274, .7392), "M2DP": (.2811, .3966, .4127),
             "Fast Histogram": (.1840, .2785, .2136), "SC-50": (.5248, .6314, .5934),
             "SC++ (PC)": (.3862, .6495, .6465), "SC++ (CC)": (.1355, .2449, .1895)},
    "MulRan": {"RING++": (.7451, .8267, .8380), "RING++ (SG)": (.6592, .7625, .7701), "RING": (.7256, .8079, .8329),
               "RING (SG)": (.6487, .7549, .7370), "M2DP": (.0831, .1437, .0750),
               "Fast Histogram": (.2163, .3363, .2551), "SC-50": (.5342, .6658, .6359),
               "SC++ (PC)": (.3740, .6260, .6260), "SC++ (CC)": (.1829, .3347, .3572)},
    "Oxford": {"RING++": (.8274, .8937, .9438), "RING++ (SG)": (.7612, .8492, .8992), "RING": (.8013, .8772, .9234),
               "RING (SG)": (.7425, .8366, .8787), "M2DP": (.2997, .4496, .4638),
               "Fast Histogram": (.0642, .1181, .0741), "SC-50": (.4375, .5935, .5366),
               "SC++ (PC)": (.3993, .5561, .5295), "SC++ (CC)": (.3084, .4596, .4540)},
}
TABLE_III = {
    "MulRan*": {"RING++": (.6941, .7983, .8337), "PointNetVLAD": (.6968, .8214, .8436),
                "DiSCO": (.7425, .8630, .8973), "EgoNN": (.7260, .8412, .9060)},
    "NCLT": {"RING++": (.7321, .7890, .8374), "PointNetVLAD": (.3691, .5391, .6260),
             "DiSCO": (.6036, .7816, .7722), "EgoNN": (.5620, .6965, .7988)},
    "Oxford": {"RING++": (.8274, .8937, .9438), "PointNetVLAD": (.5247, .7825, .6114),
               "DiSCO": (.6562, .7890, .8566), "EgoNN": (.5920, .7438, .8471)},
}


def bound(f1: float) -> float:
    """최대 F1이 f1일 때 가능한 전체 분모 정답 비율의 상한(반올림 구간 상단)."""
    f = f1 + HALF
    return f / (2 - f)


def rows(table: dict) -> dict:
    """행마다 (가)·(나) 경계를 계산한다."""
    out = {}
    for ds, methods in table.items():
        out[ds] = {}
        for m, (r1, f1, auc) in methods.items():
            p_max = bound(f1)
            out[ds][m] = {"R@1": r1, "F1": f1, "AUC": auc,
                          "max_all_denominator_R@1_given_F1": p_max,
                          "violates_all_denominator_reading": (r1 - HALF) > p_max,
                          "max_positive_fraction_if_positive_only_R@1": p_max / (r1 - HALF)}
    return out


def main() -> None:
    """두 표의 경계와 공개 생성기 집합의 양성 비율을 함께 저장한다."""
    t2, t3 = rows(TABLE_II), rows(TABLE_III)
    nclt = t2["NCLT"]
    ours = json.loads((NATIVE / "summary.json").read_text())["variants"]
    sets = {k: {"queries": ours[f"{k}_ring"]["10.0"]["queries"],
                "positive_queries": ours[f"{k}_ring"]["10.0"]["positive_queries"]} for k in ("test", "all")}
    sampling = json.loads((OUT.parent / "sampling/report.json").read_text())["conditions"]["official_as_is"]
    for k, v in sets.items():
        v["positive_fraction"] = v["positive_queries"] / v["queries"]
        # 20m 필터로 빠진 질의는 20m 안에 지도가 없어 10m 양성도 없다.
        v["queries_without_20m_filter"] = sampling[k]["query_before_filter"]
        v["positive_fraction_without_20m_filter"] = v["positive_queries"] / v["queries_without_20m_filter"]
    ring_rows = [m for m in nclt if m.startswith("RING")]
    report = {
        "purpose": "논문 표의 R@1·F1이 공개 평가 정의 아래 양립하는지 검사; 데이터·설정 선택 없음",
        "derivation": __doc__,
        "table_II": t2, "table_III": t3,
        "nclt_table_II_violations_all_denominator": sorted(m for m, v in nclt.items()
                                                           if v["violates_all_denominator_reading"]),
        "nclt_positive_fraction_upper_bound": min(v["max_positive_fraction_if_positive_only_R@1"]
                                                  for v in nclt.values()),
        "nclt_positive_fraction_upper_bound_ring_rows_only": min(
            nclt[m]["max_positive_fraction_if_positive_only_R@1"] for m in ring_rows),
        "mulran_positive_fraction_bounds": {m: v["max_positive_fraction_if_positive_only_R@1"]
                                            for m, v in t2["MulRan"].items()},
        "released_generator_sets_10m": sets,
        "assumptions": ["표의 F1은 공개 compute_PR_pairs 격자에서의 최대 F1", "R@1과 F1이 같은 질의 집합",
                        "격자 끝 1.0에서 모든 질의 수락(공식 거리는 1 미만)"],
        "rigor": ("RING 계열 4행은 공개 circorr2 거리 1−score(score>0)라 마지막 가정이 코드로 보장된다. "
                  "M2DP·Fast Histogram·SC 계열은 저자 평가 코드의 거리 척도가 공개되지 않아 같은 가정이 성립할 때만 적용된다."),
        "nclt_ring_rows_violations_all_denominator": sorted(m for m in ring_rows
                                                            if nclt[m]["violates_all_denominator_reading"]),
        "interpretation": (
            "세 가정 아래 전체 질의 분모(공개 evaluate()) R@1은 표 II NCLT의 다수 행과 양립하지 않는다. "
            "양성 분모 R@1(논문 식 30의 글)이라면 NCLT 질의의 양성 비율이 상한 이하여야 하며, "
            "공개 생성기 test·all 집합(20m 필터 유무 두 상태)의 10m 양성 비율과 비교한다. 논문의 실제 질의 집합을 확정하지 않으며 "
            "이 경계에 맞추어 표본을 고르지 않는다."),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("nclt_table_II_violations_all_denominator",
                                             "nclt_positive_fraction_upper_bound",
                                             "nclt_positive_fraction_upper_bound_ring_rows_only",
                                             "released_generator_sets_10m")}, indent=1, ensure_ascii=False))
    for m, v in nclt.items():
        print(f"{m:15s} R1 {v['R@1']:.4f} F1 {v['F1']:.4f} maxP {v['max_all_denominator_R@1_given_F1']:.4f} "
              f"violate {v['violates_all_denominator_reading']} posfrac<= {v['max_positive_fraction_if_positive_only_R@1']:.4f}")


if __name__ == "__main__":
    main()
