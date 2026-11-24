#!/usr/bin/env python3
"""BEVPlace++ + 같은 시간 증거(TE) 대조의 NSD 행과 짝지은 구간.

_pdv2_bev_temporal_highprec.py와 같은 원장·지표(AP·최대F1·R@95P·R@99P·R@100P·R@1)·구간 절차(도시 안 양성 시퀀스
복원추출 10,000회, seed 20261104). BEVPlace++·BEVPlace++ + TE 원장(results/pdv2_review_response_20261104/bev/test)은
BEVPlace++ 서술자와 고정 시간 증거 규칙(β=11/15, λ=0.25 원래 함수)만 쓰므로 NSD 모델과 무관하다.
먼저 old 꼬리표 원장(results/pdv2_diffattn_contrast_test_20261104)으로 results/pdv2_bev_temporal_highprec_20261115/highprec.json을
재현해 계산 경로를 확인한다.
출력: results/pdv2_report_model_20261116/analysis/bev_temporal/highprec.json (같은 스키마)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _report_model_common_20261116 as C  # noqa: E402

KEYS = ("AP", "max_F1", "R@95P", "R@99P", "R@100P", "R@1")
PREV = C.ROOT / "results/pdv2_bev_temporal_highprec_20261115/highprec.json"


def build(nsd_kind):
    rows = {k: [C.per_seq_rows(t, extra=True) for t in C.tags(kind)] for k, kind in (("diff", nsd_kind), ("bevt", "bevt"), ("bev", "bev"))}
    mean = {c: {m: float(np.mean([C.macro(r, KEYS)["macro"][m] for r in rows[c]])) for m in KEYS} for c in rows}
    sd = {c: {m: float(np.std([C.macro(r, KEYS)["macro"][m] for r in rows[c]], ddof=1)) for m in KEYS} for c in rows}
    comps = {}
    for a, b in (("diff", "bev"), ("diff", "bevt"), ("bevt", "bev")):
        p = C.paired(rows[a], rows[b], KEYS)
        comps[f"{a}-{b}"] = dict(delta_pp={m: 100 * (mean[a][m] - mean[b][m]) for m in KEYS}, conditional_95_pp=p["conditional_95_pp"])
    return mean, sd, comps


def main():
    prev = json.loads(PREV.read_text())
    mean, _, comps = build("old")
    err = max(abs(mean[c][m] - prev["mean"][c][m]) for c in mean for m in KEYS)
    err_iv = max(abs(x - y) for k in comps for m in KEYS for x, y in zip(comps[k]["conditional_95_pp"][m], prev["comparisons"][k]["conditional_95_pp"][m]))
    print("재현(이전 NSD): 평균 최대 오차", err, "구간 최대 오차(pp)", err_iv)
    assert err < 1e-12 and err_iv < 1e-9
    mean, sd, comps = build("new")
    assert all(abs(mean[c][m] - prev["mean"][c][m]) < 1e-12 for c in ("bev", "bevt") for m in KEYS)
    out = dict(queries_per_case=prev["queries_per_case"], mean=mean, sd=sd, comparisons=comps, nsd_seeds=list(C.NEW_SEEDS),
               intervals=prev["intervals"], reproduction=dict(previous_mean_max_abs_err=err, previous_interval_max_abs_err_pp=err_iv),
               note="bev·bevt 평균은 기존 highprec.json과 같다(검사). diff=새 NSD 시드 4~6.")
    C.write(C.OUT / "bev_temporal" / "highprec.json", out)
    for c, v in mean.items():
        print(c, {m: round(100 * x, 3) for m, x in v.items()})
    for k, v in comps.items():
        print(k, {m: (round(v["delta_pp"][m], 3), [round(x, 3) for x in v["conditional_95_pp"][m]]) for m in KEYS})


if __name__ == "__main__":
    main()
