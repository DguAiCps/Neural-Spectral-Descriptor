#!/usr/bin/env python3
"""NCLT9 수정 원장 + 기존 37 원장의 공통 질의 검산·집계 (로컬 호스트, torch 없음, 2026-11-08).

  python3 scripts/fresco_nclt_fix_20261108/aggregate_fix.py
기존 scripts/fresco_official/aggregate.py의 main을 그대로 부른다(같은 검산·지표·도시 6 매크로). 바꾸는 것은 입력 폴더와
출력 폴더뿐이다: OUT/combined_ledgers/test/에 기존 37 원장(기존 결과 폴더, sha256을 preflight 원장 해시와 대조)과
새 NCLT9 원장(새 NAS에서 받은 mirror, 서버 sha256과 대조)으로 symlink를 만들고, aggregate.LED·RESULTS를 그 폴더와 OUT/final로 바꾼다.
추가 산출(final/fix_summary.json):
  * nclt9: 새 NCLT9만의 도시(ann_arbor) 평균과 시퀀스별 값
  * all46_macro: 기존 37 + 새 9의 도시 6 매크로(기존 aggregate 규칙)
  * diagnostic_delta_vs_g3: 기존 G3 집계와의 차(진단 전용, 선택에 쓰지 않는다)
NCLT9 원장이 9개 모두 검산을 통과하지 않으면 집계 값을 완료로 쓰지 않는다(passed false, 값은 partial 표시).
  --selftest-old-g3   집계기 자체 검사: NCLT 자리에 기존 G3 원장 9개를 넣어 OUT/prefreeze/agg_selftest/에 집계하고
                      기존 summary.json의 모든 시퀀스·도시·매크로 값과 정확히 같은지 본다(새 계산 없음).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixenv  # noqa: E402

fixenv.check_old_code()
sys.path.insert(0, str(fixenv.OLD_PY))
import aggregate  # noqa: E402
from common import CITY, TEST, name, sha256, write_json  # noqa: E402

OUT = fixenv.LOCAL_OUT
OLD = Path("/mnt/d/NSD_local/work/relocated_20261107/pdv2_fresco_official_20261107/pipeline")
MIRROR = OUT / "mirror/results/ledgers/test"
COMB = OUT / "combined_ledgers/test"
FINAL = OUT / "final"


def main(selftest=False):
    global MIRROR, COMB, FINAL
    pre = json.loads((OUT / "ledger_hashes_preflight.json").read_text())
    if selftest:
        MIRROR = OLD / "ledgers/test"
        COMB = OUT / "prefreeze/agg_selftest/combined_ledgers/test"
        FINAL = OUT / "prefreeze/agg_selftest/final"
        remote = pre["old_nclt9"]
    else:
        remote = json.loads((OUT / "mirror/ledger_sha256_server.json").read_text())
    COMB.mkdir(parents=True, exist_ok=True)
    src, problems = {}, []
    for sensor, seq in TEST:
        fn = f"dump_fresco_{name(sensor, seq)}.npz"
        if sensor == "nclt":
            p = MIRROR / fn
            ok = p.exists() and sha256(p) == remote.get(fn)
            kind = "new_nclt9_fix"
        else:
            p = OLD / "ledgers/test" / fn
            ok = p.exists() and sha256(p) == pre["old37"][fn]
            kind = "old37_reused"
        if not ok:
            problems.append(fn)
            continue
        link = COMB / fn
        if link.is_symlink() or link.exists():
            assert link.is_symlink() and Path(os.readlink(link)) == p, link
        else:
            os.symlink(p, link)
        src[fn] = {"kind": kind, "target": str(p), "sha256": sha256(p)}
    if problems:
        rec = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "passed": False,
               "status": "partial_not_aggregated", "missing_or_hash_mismatch": problems}
        write_json(FINAL / "fix_summary.json", rec)
        print(json.dumps(rec)); return 1
    aggregate.LED = COMB
    aggregate.RESULTS = FINAL
    aggregate.COMMON = fixenv.REPO / "results/pdv2_symmetric3_20261013/s1"
    rc = aggregate.main()
    summ = json.loads((FINAL / "summary.json").read_text())
    ver = json.loads((FINAL / "verification.json").read_text())
    old = json.loads((OLD / "summary.json").read_text())
    nclt = list(fixenv.NCLT9)
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "label": "FreSCo official + NCLT inverted-HDL-32E body-frame sensor adaptation (spec fresco_nclt_G3prime_v1); "
                    "not a native NCLT FreSCo benchmark; 37 non-NCLT ledgers reused unchanged",
           "aggregate_rc": rc, "verification_passed": ver["passed"], "sources": src, "scores": {}}
    for s in aggregate.SCORES:
        e, o = summ["scores"][s], old["scores"][s]
        n9 = {k: e["per_sequence"][k] for k in nclt}
        aa = e["city_groups"]["ann_arbor"]
        out["scores"][s] = {
            "nclt9": {"ann_arbor_mean_percent": {m: round(100 * v, 3) for m, v in aa.items()},
                      "per_sequence": n9},
            "all46_macro_percent": e["macro_percent"],
            "city_groups_percent": {c: {m: round(100 * v, 3) for m, v in g.items()} for c, g in e["city_groups"].items()},
            "diagnostic_delta_vs_g3": {
                "note": "진단 전용. 설정 선택·채택 근거로 쓰지 않는다",
                "nclt9_ann_arbor_pp": {m: round(100 * (aa[m] - o["city_groups"]["ann_arbor"][m]), 3) for m in aa},
                "all46_macro_pp": {m: round(e["macro_percent"][m] - o["macro_percent"][m], 3) for m in e["macro_percent"]},
                "old_g3_all46_macro_percent": o["macro_percent"],
                "old_g3_ann_arbor_percent": {m: round(100 * v, 3) for m, v in o["city_groups"]["ann_arbor"].items()}},
            "non_nclt_city_groups_unchanged": all(
                abs(e["city_groups"][c][m] - o["city_groups"][c][m]) == 0 for c in CITY if c != "ann_arbor"
                for m in e["city_groups"][c])}
    out["passed"] = bool(rc == 0 and ver["passed"] and out["scores"]["score_fresco"]["non_nclt_city_groups_unchanged"])
    if selftest:
        out["selftest_exact_equal_old_summary"] = all(
            summ["scores"][s][f] == old["scores"][s][f] for s in aggregate.SCORES
            for f in ("per_sequence", "city_groups", "macro", "macro_percent", "n_queries", "n_positive", "excluded_no_positive"))
        out["passed"] = bool(out["passed"] and out["selftest_exact_equal_old_summary"])
    write_json(FINAL / "fix_summary.json", out)
    print(json.dumps({"passed": out["passed"], "nclt9": out["scores"]["score_fresco"]["nclt9"]["ann_arbor_mean_percent"],
                      "all46": out["scores"]["score_fresco"]["all46_macro_percent"]}, indent=1))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(selftest="--selftest-old-g3" in sys.argv[1:]))
