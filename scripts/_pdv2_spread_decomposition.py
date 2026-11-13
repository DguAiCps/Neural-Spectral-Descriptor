#!/usr/bin/env python3
"""고정 검색 키에서 장소 내 퍼짐과 먼 장소 충돌을 나누고, 각 원인에서 NSD가 어떻게 되는지 센다.

키는 후보 검색에 쓰는 두 고정 키다(288D 스펙트럼 키 = 주, 128D BEV 크기 색인 = 보조).
양성 질의마다 적격 과거(j<=q-30)에서
  같은 장소 거리 d_same = 1 - max cos(5 m 안 관측), 먼 장소 거리 d_far = 1 - max cos(25 m 이상 관측).
d_far < d_same이면 키 수준 실패다. 실패의 원인은 같은 도시에서 키로 성공한 질의들의 분포에 대한 백분위로 정한다:
같은 장소 거리가 더 비정상적으로 멀면 퍼짐, 먼 장소 거리가 더 비정상적으로 가까우면 충돌(임계값 없음).
정의는 계산 전에 preregistration.json에 고정한다. 산출물은 D 드라이브(컨테이너 /workspace/data)에 둔다.
사용법(컨테이너 ring_baseline 안): python scripts/_pdv2_spread_decomposition.py
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
OUT = Path(os.environ.get("NSD_SPREAD_OUT",
                          "/workspace/data/nsd_results/pdv2_spread_decomposition_20261110"))
NSD = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
MAG = ROOT / "results/pdv2_macro_ablation_20261106"
STARTS = (1, 2, 3)
KEYS = ("spectral_key", "bev_magnitude")

PREREGISTRATION = dict(
    purpose="원고 서론이 앨리어싱과 함께 드는 장소 내 퍼짐(시점·표본·가림으로 같은 장소의 관측이 키 공간에서 흩어지는 현상)을 고정 검색 키에서 재고, 원인별로 NSD의 결과를 센다. 기술 통계이며 설정 선택에 쓰지 않는다.",
    keys="후보 검색에 쓰는 두 고정 키를 따로 본다. 주: 288D 재계량 스펙트럼 키(단위 정규화). 보조: 128D BEV 크기 색인(단위 정규화). 두 키를 섞은 점수는 질의별 min-max라 거리 척도가 없어 쓰지 않는다.",
    population="시험 46시퀀스의 양성 질의(min_past_gt_dist<=5 m). 적격 과거는 j<=q-30이며 위치는 정답 자세.",
    distances="d_same = 1 - (5 m 안 적격 관측과의 최대 코사인), d_far = 1 - (25 m 이상 적격 관측과의 최대 코사인). 5–25 m 관측은 두 거리 모두에서 뺀다. 25 m 이상 관측이 없으면 실패가 될 수 없다.",
    failure="키 수준 실패: d_far < d_same(먼 장소가 모든 같은 장소 관측보다 키에서 가깝다).",
    cause="실패마다 같은 도시에서 그 키로 성공한 질의들의 d_same 분포에서 이 질의 d_same보다 작은 비율 p_same, d_far 분포에서 이 질의 d_far보다 큰 비율 p_far를 구한다. p_same > p_far면 퍼짐(같은 장소가 유난히 멀다), 아니면 충돌(먼 장소가 유난히 가깝다). 임계값을 쓰지 않는다.",
    outcomes="원인별로 (1) 크기 점수만(표3 정렬 제거 원장)의 정답 반환 비율 (2) NSD(표2 원장, 세 시작점 평균)의 정답 반환 비율 (3) NSD 후보 합집합(색인별 top-100)에 정답이 있는 비율.",
    summaries="질의 합산과 도시 매크로(양성 시퀀스 균등 -> 6도시 균등). 키별 실패율, 원인 비율, 성공/실패 질의의 d_same·d_far 중앙값.",
    limits="퍼짐/충돌은 상대적 비정상성으로 정한 지배 원인 분류이며 두 현상이 함께 있을 수 있다. 수락 점수(오탐 수락)는 다루지 않는다.",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))


def sequence_rows(T, sequence: str) -> list:
    """한 시퀀스의 양성 질의마다 두 키의 거리와 NSD 결과를 모은다."""
    name = sequence.replace("/", "_")
    cache, pos, minimum, _, bank, _, _, _ = T.inputs("test", sequence)
    keys = {"spectral_key": T.pipeline.raw_key(cache["descriptors"]), "bev_magnitude": bank[0]}
    for k in keys.values():
        assert np.allclose(np.linalg.norm(k, axis=1), 1, atol=1e-4)
    ledger = np.load(NSD / "test" / name / "diff_s1.npz")
    nsd_ok = np.mean([np.load(NSD / "test" / name / f"diff_s{s}.npz")["top1_geo_dist"] <= 5
                      for s in STARTS], axis=0)
    mag_ok = np.load(MAG / "test" / name / "no_alignment_s1.npz")["top1_geo_dist"] <= 5
    pool = np.load(NSD / "test" / name / "candidate_pool.npz")
    q_all = ledger["query_idx"]
    np.testing.assert_array_equal(pool["query"], q_all)
    rows = []
    for n, q in enumerate(q_all):
        if not ledger["min_past_gt_dist"][n] <= 5:
            continue
        elig = np.arange(0, q - 29)
        dist = np.linalg.norm(pos[elig] - pos[q], axis=1)
        same, far = dist <= 5, dist >= 25
        assert same.any() and abs(dist.min() - minimum[q]) < 1e-6
        ids = pool["ids"][pool["offset"][n]:pool["offset"][n + 1]]
        row = dict(sequence=sequence, q=int(q), nsd_correct=float(nsd_ok[n]), mag_correct=bool(mag_ok[n]),
                   candidate_hit=bool((np.linalg.norm(pos[ids] - pos[q], axis=1) <= 5).any()))
        for key, K in keys.items():
            sims = K[elig] @ K[q]
            d_same = 1.0 - float(sims[same].max())
            d_far = 1.0 - float(sims[far].max()) if far.any() else np.inf
            row[key] = dict(d_same=d_same, d_far=d_far, fail=bool(d_far < d_same))
        rows.append(row)
    return rows


def main() -> None:
    pre = OUT / "preregistration.json"
    if not pre.exists():
        write(pre, dict(created_unix=time.time(), script_sha256=sha(Path(__file__)), **PREREGISTRATION))
    import _pdv2_contrast_pipeline_test as T
    import _pdv2_pilot as pilot

    sequences = T.prepare()["test"]
    city_of = {s: c for c, seqs in pilot.CITY.items() for s in seqs}
    rows = [r for s in sequences for r in sequence_rows(T, s)]
    report = dict(positives=len(rows))
    for key in KEYS:
        # 원인 분류: 같은 도시의 성공 질의 분포에 대한 백분위
        for city in pilot.CITY:
            sel = [r for r in rows if city_of[r["sequence"]] == city]
            ok = [r[key] for r in sel if not r[key]["fail"]]
            same_ref = np.sort([v["d_same"] for v in ok])
            far_ref = np.sort([v["d_far"] for v in ok if np.isfinite(v["d_far"])])
            for r in sel:
                v = r[key]
                if v["fail"]:
                    p_same = np.searchsorted(same_ref, v["d_same"], side="left") / len(same_ref)
                    p_far = 1 - np.searchsorted(far_ref, v["d_far"], side="right") / len(far_ref)
                    v.update(p_same=float(p_same), p_far=float(p_far),
                             cause="spread" if p_same > p_far else "collision")

        def stats(sel: list) -> dict:
            fails = [r for r in sel if r[key]["fail"]]
            out = dict(positives=len(sel), failure_rate=len(fails) / max(len(sel), 1),
                       median_d_same_success=float(np.median([r[key]["d_same"] for r in sel if not r[key]["fail"]])),
                       median_d_same_failure=float(np.median([r[key]["d_same"] for r in fails])) if fails else None,
                       median_d_far_success=float(np.median([r[key]["d_far"] for r in sel if not r[key]["fail"] and np.isfinite(r[key]["d_far"])])),
                       median_d_far_failure=float(np.median([r[key]["d_far"] for r in fails])) if fails else None)
            for cause in ("spread", "collision"):
                group = [r for r in fails if r[key]["cause"] == cause]
                out[cause] = dict(
                    share_of_failures=len(group) / max(len(fails), 1),
                    rate=len(group) / max(len(sel), 1),
                    count=len(group),
                    magnitude_only_correct=float(np.mean([r["mag_correct"] for r in group])) if group else None,
                    nsd_correct=float(np.mean([r["nsd_correct"] for r in group])) if group else None,
                    candidate_hit=float(np.mean([r["candidate_hit"] for r in group])) if group else None)
            return out

        pooled = stats(rows)
        cities = {c: stats([r for r in rows if city_of[r["sequence"]] == c]) for c in pilot.CITY}

        def seq_macro(field: str, cause: str = None) -> float:
            vals = []
            for c, seqs in pilot.CITY.items():
                per = []
                for s in seqs:
                    sel = [r for r in rows if r["sequence"] == s]
                    if not sel:
                        continue
                    if cause is None:
                        per.append(np.mean([r[key]["fail"] for r in sel]))
                    else:
                        per.append(np.mean([r[key]["fail"] and r[key]["cause"] == cause for r in sel]))
                vals.append(np.mean(per))
            return float(np.mean(vals))

        report[key] = dict(pooled=pooled, cities=cities,
                           city_macro=dict(failure_rate=seq_macro("fail"),
                                           spread_rate=seq_macro("fail", "spread"),
                                           collision_rate=seq_macro("fail", "collision")))
    report["candidate_hit_all_positives"] = float(np.mean([r["candidate_hit"] for r in rows]))
    report["sources"] = {str(p.relative_to(ROOT)): sha(p) for p in
                         (NSD / "protocol.json", MAG / "protocol.json", Path(__file__))}
    write(OUT / "result.json", report)
    np.savez_compressed(OUT / "rows.npz", rows=np.array(json.dumps(rows)))
    for key in KEYS:
        p, m = report[key]["pooled"], report[key]["city_macro"]
        print(key, f"실패 {100*p['failure_rate']:.2f}% (도시매크로 {100*m['failure_rate']:.2f}%)",
              f"퍼짐 {100*p['spread']['share_of_failures']:.1f}% / 충돌 {100*p['collision']['share_of_failures']:.1f}%",
              f"| d_same 중앙 성공 {p['median_d_same_success']:.4f} 실패 {p['median_d_same_failure']:.4f}",
              f"d_far 중앙 성공 {p['median_d_far_success']:.4f} 실패 {p['median_d_far_failure']:.4f}", flush=True)
        for cause in ("spread", "collision"):
            g = p[cause]
            print(f"  {cause}: 양성의 {100*g['rate']:.2f}%, 크기만 정답 {100*g['magnitude_only_correct']:.1f}%,"
                  f" NSD 정답 {100*g['nsd_correct']:.1f}%, 후보 포함 {100*g['candidate_hit']:.1f}%", flush=True)
    print("후보 합집합 포함(전체 양성)", round(100 * report["candidate_hit_all_positives"], 2), flush=True)


if __name__ == "__main__":
    main()
