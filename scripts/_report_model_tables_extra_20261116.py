#!/usr/bin/env python3
"""표 생성(단위 F1) 보충 계산: Past-only NSD − 정제 없음+과거 5 수락의 짝지은 차이(전체·순방향·역방향).

시간 창 차이표(appdx_temporal_window_deltas)의 v2.3 행 "Past-only NSD − no refining, past-five acceptance"를 새 원장으로
다시 만든다. 두 원장은 이미 있다: Past-only NSD = controls/ledgers/test/<seq>/pastonly@s{4,5,6}@u512.npz,
정제 없음+과거 5 수락 = ablations/ledgers/test/<seq>/zl_identity.npz. 짝지은 구간은 단위 C 공통 함수
(_report_model_common_20261116.paired, 도시 안 시퀀스 재표집 10,000회, 난수 20261104)를 그대로 쓴다.
같은 절차로 controls/summary.json의 pastonly−uniform_past10 구간과 window_side_summary.json의 절대값을 재현해 확인한다.
호스트에서 torch 없이 돈다: python3 scripts/_report_model_tables_extra_20261116.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _report_model_common_20261116 as C  # noqa: E402

BASE = ROOT / "results/pdv2_report_model_20261116"
OUT = BASE / "tables_f1/pastonly_minus_zl_identity.json"
FILES = {"pastonly": [BASE / f"controls/ledgers/test/{{s}}/pastonly@s{k}@u512.npz" for k in (4, 5, 6)],
         "uniform_past10": [BASE / "controls/ledgers/test/{s}/uniform_past10.npz"],
         "zl_identity": [BASE / "ablations/ledgers/test/{s}/zl_identity.npz"]}


def rows(arm: str, masks=None) -> list:
    """시드별 {seq: 지표}."""
    out = []
    for pattern in FILES[arm]:
        per = {}
        for s in C.SEQS:
            with np.load(str(pattern).format(s=s.replace("/", "_"))) as z:
                d = dict(score=z["score"].astype(np.float64), top1=z["top1_geo_dist"], gt=z["min_past_gt_dist"])
            per[s] = C.seq_metric(d, None if masks is None else masks[s])
        out.append(per)
    return out


def main() -> None:
    controls = json.loads((BASE / "controls/summary.json").read_text())
    window = json.loads((BASE / "ablations/window_side_summary.json").read_text())
    result, checks = {}, {}
    for group in ("all", "forward_only", "reverse_only"):
        masks = None if group == "all" else C.direction_masks(group)
        a, b, u = rows("pastonly", masks), rows("zl_identity", masks), rows("uniform_past10", masks)
        # 절대값 재현: 6도시 매크로가 기존 요약과 같아야 한다.
        for name, rs, ref in (("pastonly", a, [controls["test"][group]["arms"]["pastonly"][k]["mean"] for k in C.METRICS]),
                              ("zl_identity", b, [window["groups"][group]["percent"]["no_refining_past5_acceptance"][k] / 100 for k in C.METRICS])):
            got = [float(np.mean([C.macro(r)["macro"][k] for r in rs])) for k in C.METRICS]
            checks[f"{group}/{name}_macro_max_abs_err"] = max(abs(x - y) for x, y in zip(got, ref))
        # 구간 절차 재현: pastonly−uniform_past10은 controls/summary.json과 같아야 한다.
        ref = controls["test"][group]["differences"]["pastonly-uniform_past10"]
        rep = C.paired(a, u)
        checks[f"{group}/pastonly-uniform_past10_interval_max_abs_err_pp"] = max(
            abs(x - y) for k in C.METRICS for x, y in zip(rep["conditional_95_pp"][k], ref["conditional_95_pp"][k]))
        result[group] = C.paired(a, b)
    worst = max(checks.values())
    out = {"status": "passed" if worst < 1e-9 else "failed", "comparison": "pastonly - zl_identity",
           "a": "Past-only NSD (pastonly@s4..6@u512, controls/ledgers)", "b": "no refining + past-five acceptance (zl_identity, ablations/ledgers)",
           "procedure": "_report_model_common_20261116.paired: 시퀀스별 (a 시드 평균 − b) → 도시 안 시퀀스 재표집 10,000회 → 6도시 평균, 난수 20261104",
           "units": "%p", "checks": checks, "groups": result}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"status": out["status"], "worst_check": worst,
                      "all_AP": result["all"]["delta_pp"]["AP"], "all_AP_ci": result["all"]["conditional_95_pp"]["AP"]}))


if __name__ == "__main__":
    main()
