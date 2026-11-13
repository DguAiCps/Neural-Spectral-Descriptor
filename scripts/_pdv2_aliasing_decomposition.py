#!/usr/bin/env python3
"""크기 색인 점수만으로 반환한 장소의 오류를 먼 장소 충돌과 근거리 오류로 나누고, NSD가 바로잡는 비율을 센다.

GPU 없이 기존 시험 원장만 읽는다.
- 크기만: `results/pdv2_macro_ablation_20261106/test/*/no_alignment_s{k}.npz`
  (두 색인 코사인으로만 반환, 원고 표3의 "정렬 없음" 행)
- NSD: `results/pdv2_diffattn_contrast_test_20261104/test/*/diff_s{k}.npz` (원고 표2)
정의와 보고 수치는 계산 전에 preregistration.json에 고정한다. 산출물은 D 드라이브에 둔다.
사용법: python3 scripts/_pdv2_aliasing_decomposition.py
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("NSD_ALIAS_OUT",
                          "/mnt/d/NSD_datasets/nsd_results/pdv2_aliasing_decomposition_20261110"))
NSD = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
MAG = ROOT / "results/pdv2_macro_ablation_20261106"
STARTS = (1, 2, 3)
POSITIVE_M, CORRECT_M, DISTANT_M = 5.0, 5.0, 25.0
CITY = {  # scripts/_pdv2_pilot.py CITY와 같은 묶음(아래에서 대조)
    "karlsruhe": ["kitti/00", "kitti/07", "kitti/08", "kitti360/0000"],
    "seoul": [f"helipr/Town0{i}" for i in (1, 2, 3)],
    "daejeon": [f"helipr/{p}0{i}" for p in ("KAIST", "DCC", "Riverside") for i in (4, 5, 6)]
               + [f"mulran/{p}0{i}" for p in ("KAIST", "DCC", "Riverside") for i in (1, 2, 3)],
    "ann_arbor": [f"nclt/{d}" for d in ("2012-01-08", "2012-02-04", "2012-03-17", "2012-05-11",
                                        "2012-08-04", "2012-11-04", "2012-11-16", "2013-01-10",
                                        "2013-02-23")],
    "stockholm": [f"mcd/kth_{s}" for s in ("day_06", "day_09", "day_10", "night_01", "night_04",
                                           "night_05")],
    "hamburg": [f"mcd/tuhh_{s}" for s in ("day_02", "day_03", "day_04", "night_07", "night_08",
                                          "night_09")],
}
PREREGISTRATION = dict(
    purpose="원고가 중심 개념으로 쓰는 spectral aliasing(크기 키에서 먼 장소가 비슷해지는 현상)이 시험셋에서 얼마나 일어나고 NSD가 얼마나 바로잡는지 센다. 기술 통계이며 설정 선택에 쓰지 않는다.",
    data="시험 46시퀀스, 세 시작점. 크기만 = 원고 표3 '정렬 없음' 원장(두 색인 코사인으로만 반환). NSD = 원고 표2 원장.",
    population="양성 질의: 적격 과거(j<=q-30) 안 최근접 정답 거리 min_past_gt_dist<=5 m. 원고 R@1과 같은 정의.",
    categories="반환 장소까지 거리 d. 정답 d<=5 m. 근거리 오류 5<d<25 m. 먼 장소 충돌 d>=25 m. 25 m는 원고의 모호 구간 경계이자 이전 AliasRate의 gamma와 같은 값이며 결과를 보고 바꾸지 않는다.",
    primary="세 시작점 평균, 46시퀀스 양성 질의 합산: (a) 크기만의 오류율과 그중 먼 장소 충돌 비율 (b) NSD의 같은 값 (c) 크기만에서 먼 장소 충돌이던 질의 중 NSD가 정답을 반환한 비율 (d) 크기만에서 정답이던 질의 중 NSD가 틀린 비율.",
    secondary="도시 매크로(양성이 있는 45시퀀스 -> 6도시 균등)로 같은 값, 시작점별 값, 도시별 표. 원고 R@1과의 대조(검산).",
    limits="반환 장소 거리만으로 나눈다. 정답이 키 공간에서 멀리 흩어진 경우(장소 내 퍼짐)는 서술자 거리가 필요해 따로 재지 않는다. 수락 점수(오탐 수락)는 다루지 않는다.",
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


def classify(d: np.ndarray) -> np.ndarray:
    """0 정답, 1 근거리 오류, 2 먼 장소 충돌."""
    return np.where(d <= CORRECT_M, 0, np.where(d < DISTANT_M, 1, 2))


def transitions() -> dict:
    """시작점별·시퀀스별 [크기만 범주, NSD 범주] 3×3 개수표를 원장에서 계산한다."""
    sequences = json.loads((NSD / "protocol.json").read_text())["test"]
    assert sorted(sequences) == sorted(s for v in CITY.values() for s in v)
    per_seq = {k: {} for k in STARTS}
    for s in sequences:
        name = s.replace("/", "_")
        for k in STARTS:
            mag = np.load(MAG / "test" / name / f"no_alignment_s{k}.npz")
            nsd = np.load(NSD / "test" / name / f"diff_s{k}.npz")
            np.testing.assert_array_equal(mag["query_idx"], nsd["query_idx"])
            np.testing.assert_array_equal(mag["min_past_gt_dist"], nsd["min_past_gt_dist"])
            pos = nsd["min_past_gt_dist"] <= POSITIVE_M
            a, b = classify(mag["top1_geo_dist"][pos]), classify(nsd["top1_geo_dist"][pos])
            table = np.zeros((3, 3), np.int64)
            np.add.at(table, (a, b), 1)
            per_seq[k][s] = table
    return per_seq


def city_rates(per_seq: dict) -> dict:
    """원고 집계와 같게 도시 안 양성 시퀀스를 균등 평균한 뒤 세 시작점을 평균한다(%)."""
    def rates(t: np.ndarray) -> dict:
        n = t.sum()
        return dict(magnitude_error=t[1:].sum() / n, magnitude_distant=t[2].sum() / n,
                    nsd_error=t[:, 1:].sum() / n, nsd_distant=t[:, 2].sum() / n)

    out = {}
    for city, seqs in CITY.items():
        out[city] = {key: 100 * float(np.mean([np.mean([rates(per_seq[k][s])[key] for s in seqs
                                                        if per_seq[k][s].sum()]) for k in STARTS]))
                     for key in ("magnitude_error", "magnitude_distant", "nsd_error", "nsd_distant")}
    out["macro"] = {key: float(np.mean([out[c][key] for c in CITY])) for key in out["karlsruhe"]}
    return out


def main() -> None:
    pre = OUT / "preregistration.json"
    if not pre.exists():
        write(pre, dict(created_unix=time.time(), script_sha256=sha(Path(__file__)), **PREREGISTRATION))
    sequences = json.loads((NSD / "protocol.json").read_text())["test"]
    assert sorted(sequences) == sorted(s for v in CITY.values() for s in v)
    pooled = {k: np.zeros((3, 3), np.int64) for k in STARTS}  # [크기만 범주, NSD 범주]
    per_seq = {k: {} for k in STARTS}
    for s in sequences:
        name = s.replace("/", "_")
        for k in STARTS:
            mag = np.load(MAG / "test" / name / f"no_alignment_s{k}.npz")
            nsd = np.load(NSD / "test" / name / f"diff_s{k}.npz")
            np.testing.assert_array_equal(mag["query_idx"], nsd["query_idx"])
            np.testing.assert_array_equal(mag["min_past_gt_dist"], nsd["min_past_gt_dist"])
            pos = nsd["min_past_gt_dist"] <= POSITIVE_M
            a, b = classify(mag["top1_geo_dist"][pos]), classify(nsd["top1_geo_dist"][pos])
            table = np.zeros((3, 3), np.int64)
            np.add.at(table, (a, b), 1)
            pooled[k] += table
            per_seq[k][s] = table
    # 크기만 반환이 시작점과 무관한지(색인은 고정) 확인
    same_mag = all(np.array_equal(per_seq[1][s].sum(1), per_seq[k][s].sum(1))
                   for s in sequences for k in STARTS)

    def summary(t: np.ndarray) -> dict:
        n = t.sum()
        mag_err, nsd_err = t[1:].sum(), t[:, 1:].sum()
        return dict(
            positives=int(n),
            magnitude_error_rate=float(mag_err / n),
            magnitude_distant_share_of_errors=float(t[2].sum() / max(mag_err, 1)),
            magnitude_distant_rate=float(t[2].sum() / n),
            nsd_error_rate=float(nsd_err / n),
            nsd_distant_share_of_errors=float(t[:, 2].sum() / max(nsd_err, 1)),
            nsd_distant_rate=float(t[:, 2].sum() / n),
            distant_fixed_by_nsd=float(t[2, 0] / max(t[2].sum(), 1)),
            near_fixed_by_nsd=float(t[1, 0] / max(t[1].sum(), 1)),
            correct_broken_by_nsd=float(t[0, 1:].sum() / max(t[0].sum(), 1)),
            transition=t.tolist(),
        )

    starts = {k: summary(pooled[k]) for k in STARTS}
    mean = {key: float(np.mean([starts[k][key] for k in STARTS]))
            for key, v in starts[1].items() if isinstance(v, float)}

    def city_macro(k: int, key: str) -> float:
        vals = []
        for city, seqs in CITY.items():
            rows = [summary(per_seq[k][s])[key] for s in seqs if per_seq[k][s].sum()]
            vals.append(np.mean(rows))
        return float(np.mean(vals))

    keys = ("magnitude_error_rate", "magnitude_distant_rate", "nsd_error_rate", "nsd_distant_rate")
    macro = {key: float(np.mean([city_macro(k, key) for k in STARTS])) for key in keys}
    # 검산: 원고 R@1(도시 매크로)의 크기만-NSD 차이 19.63점
    paper_check = 100 * (macro["magnitude_error_rate"] - macro["nsd_error_rate"])
    cities = {}
    for city, seqs in CITY.items():
        t = sum(pooled_city for pooled_city in
                (sum(per_seq[k][s] for s in seqs) for k in STARTS))
        cities[city] = summary(t)
    result = dict(pooled_mean_over_starts=mean, per_start=starts, city_macro=macro,
                  r1_gap_points_city_macro=paper_check, cities=cities,
                  magnitude_returns_identical_across_starts=same_mag,
                  sources={str(p.relative_to(ROOT)): sha(p) for p in
                           (NSD / "protocol.json", MAG / "protocol.json", Path(__file__))})
    write(OUT / "result.json", result)
    print(json.dumps(dict(pooled=mean, city_macro=macro, r1_gap=paper_check, same_mag=same_mag),
                     ensure_ascii=False, indent=1))
    for city, v in cities.items():
        print(city, {k: round(100 * v[k], 1) for k in ("magnitude_error_rate", "magnitude_distant_share_of_errors",
                                                         "distant_fixed_by_nsd", "nsd_error_rate")})


if __name__ == "__main__":
    main()
