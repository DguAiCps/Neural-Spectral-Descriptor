#!/usr/bin/env python3
"""공식 FreSCo 시험 46 원장의 공통 질의 검산·집계 (로컬 호스트, torch 없음, 2026-11-07).

  python3 scripts/fresco_official/aggregate.py
입력: results/pdv2_fresco_official_20261107/pipeline/ledgers/test/dump_fresco_<seq>.npz (서버에서 rsync).
_pdv2_main_baselines_official.aggregate와 같은 규칙: NSD 대칭 문맥 기록과 query_idx·min_past_gt_dist가
같은지, 원장 질의 수 - 15 = 공통 질의, 승자 j <= q-30, 점수 유한, 합 195,726 / 양성 61,682,
양성 없는 kitti/07 원장 포함(평균 제외), 45양성 시퀀스 -> 도시 6 -> 매크로.
주 점수 score_fresco, 보조 진단 score_negl1(같은 승자, L1만).
-> summary.json, verification.json (모두 통과해야 verification.passed)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CITY, REPO, RESULTS, TEST, name, sha256, write_json  # noqa: E402
from _pr_eval_lib import AGG_METRICS, classify_queries, pr_sweep  # noqa: E402

COMMON = REPO / "results/pdv2_symmetric3_20261013/s1"
LED = RESULTS / "ledgers/test"
METRICS = (*AGG_METRICS, "R@1")
SCORES = ("score_fresco", "score_negl1")


def rel(p: Path) -> str:
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def main():
    sources, checks, per = {}, {}, {s: {} for s in SCORES}
    nonfinite = {}
    for sensor, seq in TEST:
        k = f"{sensor}/{seq}"
        p = LED / f"dump_fresco_{name(sensor, seq)}.npz"
        ref = COMMON / f"test_selected_symmetric_pbev128_main_{sensor}_{seq}.npz"
        sources[rel(p)] = sha256(p)
        sources[rel(ref)] = sha256(ref)
        with np.load(p) as z, np.load(ref) as n:
            meta = json.loads(str(z["meta"]))
            q = z["query_idx"]
            ix = np.searchsorted(q, n["query_idx"])
            c = {"query_idx_equal": bool(np.array_equal(q[ix], n["query_idx"])),
                 "min_past_equal": bool(np.array_equal(z["min_past_gt_dist"][ix], n["min_past_gt_dist"])),
                 "common_is_ledger_minus_15": bool(len(ix) == len(q) - 15),
                 "ledger_queries_from_30": bool(q[0] == 30 and np.all(np.diff(q) == 1)),
                 "past_only": bool(np.all(z["j_star"] <= q - 30) and np.all(z["j_star"] >= 0)),
                 "scores_finite": bool(all(np.isfinite(z[s]).all() for s in SCORES)),
                 "kd_candidates": bool(np.all(z["n_kd_candidates"] == np.minimum(20, q - 29))),
                 "n_nonfinite_or_no_winner": int(meta["n_nonfinite_or_no_winner"])}
            c["passed"] = all(v for kk, v in c.items() if kk != "n_nonfinite_or_no_winner")
            checks[k] = c
            nonfinite[k] = c["n_nonfinite_or_no_winner"]
            qc = classify_queries(z["min_past_gt_dist"][ix])
            for s in SCORES:
                row = {"n_queries": int(len(ix)), "n_positive": qc["n_pos"], "n_negative": qc["n_neg"],
                       "n_ambiguous": qc["n_ambiguous"]}
                if qc["n_pos"]:
                    sw = pr_sweep(z[s][ix], z["top1_geo_dist"][ix], qc)
                    row.update({m: float(sw[m]) for m in AGG_METRICS})
                    row["R@1"] = float((z["top1_geo_dist"][ix][qc["positive"]] <= 5).mean())
                else:
                    row.update({m: None for m in METRICS})
                per[s][k] = row
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "method": "FreSCo official (MATLAB 2bf5cee + C++ preprocessing d94040e)",
           "protocol": "results/pdv2_fresco_official_20261107/pipeline/protocol.json",
           "primary_score": "score_fresco", "runs": 1, "scores": {}}
    for s in SCORES:
        groups = {city: {m: float(np.mean([per[s][q][m] for q in names if per[s][q]["n_positive"]]))
                         for m in METRICS} for city, names in CITY.items()}
        macro = {m: float(np.mean([g[m] for g in groups.values()])) for m in METRICS}
        out["scores"][s] = {"per_sequence": per[s], "city_groups": groups, "macro": macro,
                            "macro_percent": {m: round(100 * v, 3) for m, v in macro.items()},
                            "n_queries": sum(r["n_queries"] for r in per[s].values()),
                            "n_positive": sum(r["n_positive"] for r in per[s].values()),
                            "excluded_no_positive": [q for q, r in per[s].items() if not r["n_positive"]]}
    e = out["scores"]["score_fresco"]
    totals = {"n_queries_195726": e["n_queries"] == 195726, "n_positive_61682": e["n_positive"] == 61682,
              "excluded_kitti07_only": e["excluded_no_positive"] == ["kitti/07"],
              "n_ledgers_46": len(checks) == 46, "all_sequence_checks": all(c["passed"] for c in checks.values())}
    ver = {"created_utc": out["created_utc"], "per_sequence": checks, "totals": totals,
           "nonfinite_or_no_winner_total": int(sum(nonfinite.values())),
           "sources_sha256": sources, "passed": all(totals.values())}
    write_json(RESULTS / "summary.json", out)
    write_json(RESULTS / "verification.json", ver)
    print(json.dumps({"passed": ver["passed"], "totals": totals,
                      "macro_percent": e["macro_percent"],
                      "negl1_macro_percent": out["scores"]["score_negl1"]["macro_percent"]}, indent=1))
    return 0 if ver["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
