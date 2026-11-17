#!/usr/bin/env python3
"""기하 편향 0 개입 결과의 독립 검산(numpy만, 파이프라인 코드 미사용).

원장(npz)의 점수·라벨에서 AP·최대 F1·R@100P·R@1을 따로 구현한 계산으로 다시 구하고,
검증 계열 매크로·시험 도시 매크로를 다시 집계해 result.json과 비교한다.
사용법: python3 scripts/_pdv2_geometry_bias_ablation_verify_20261116.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pdv2_geometry_bias_ablation_20261116"
CITY = {
    "karlsruhe": ["kitti/00", "kitti/07", "kitti/08", "kitti360/0000"],
    "seoul": ["helipr/Town01", "helipr/Town02", "helipr/Town03"],
    "daejeon": [f"helipr/{p}0{i}" for p in ("KAIST", "DCC", "Riverside") for i in (4, 5, 6)]
    + [f"mulran/{p}0{i}" for p in ("KAIST", "DCC", "Riverside") for i in (1, 2, 3)],
    "ann_arbor": ["nclt/2012-01-08", "nclt/2012-02-04", "nclt/2012-03-17", "nclt/2012-05-11",
                  "nclt/2012-08-04", "nclt/2012-11-04", "nclt/2012-11-16", "nclt/2013-01-10",
                  "nclt/2013-02-23"],
    "stockholm": [f"mcd/kth_day_{i:02d}" for i in (6, 9, 10)] + [f"mcd/kth_night_{i:02d}" for i in (1, 4, 5)],
    "hamburg": [f"mcd/tuhh_day_{i:02d}" for i in (2, 3, 4)] + [f"mcd/tuhh_night_{i:02d}" for i in (7, 8, 9)],
}


def metrics(path):
    d = np.load(path)
    gt, top, score = d["min_past_gt_dist"], d["top1_geo_dist"], d["score"]
    pos = gt <= 5
    use = pos | (gt > 25)
    hit = pos & (top <= 5)
    n_pos = int(pos.sum())
    if n_pos == 0:
        return None
    s, h = score[use], hit[use]
    thresholds = np.unique(s)[::-1]  # 내림차순 고유 점수
    curve = [(1.0, 0.0)]  # (precision, recall), 임계값 무한대
    tp = fp = 0
    order = np.argsort(-s, kind="mergesort")
    s, h = s[order], h[order]
    i = 0
    for t in thresholds:
        while i < len(s) and s[i] >= t:
            tp += int(h[i]); fp += int(not h[i]); i += 1
        curve.append((tp / (tp + fp), tp / n_pos, fp))
    prec = np.array([c[0] for c in curve]); rec = np.array([c[1] for c in curve])
    fps = np.array([0] + [c[2] for c in curve[1:]])
    ap = 0.0
    for k in range(1, len(curve)):
        ap += (rec[k] - rec[k - 1]) * prec[k:].max()
    f1 = max((2 * p * r / (p + r)) if p + r > 0 else 0.0 for p, r in zip(prec, rec))
    return dict(AP=ap, max_F1=f1, **{"R@100P": float(rec[fps == 0].max())}, **{"R@1": hit.sum() / n_pos})


def main():
    res = json.loads((OUT / "result.json").read_text())
    report = {}
    worst = 0.0
    # 검증: 시작점별 계열 매크로(양성 있는 시퀀스) → 세 시작점 평균
    val = json.loads((OUT / "val_eval/protocol.json").read_text())["val"]
    for arm in ("diff", "diff_nogeo", "convs_only", "convs_only_nogeo"):
        per = []
        for k in (1, 2, 3):
            fam = {}
            for s in val:
                m = metrics(OUT / "val_eval/val" / s.replace("/", "_") / f"{arm}_s{k}.npz")
                if m:
                    fam.setdefault("kitti" if s.startswith("kitti") else s.split("/")[0], []).append(m)
            per.append({m: np.mean([np.mean([r[m] for r in rows]) for rows in fam.values()])
                        for m in ("AP", "max_F1", "R@100P", "R@1")})
        mine = {m: 100 * float(np.mean([p[m] for p in per])) for m in per[0]}
        diff = {m: abs(mine[m] - res["val_percent"][arm]["mean"][m]) for m in mine}
        worst = max(worst, *diff.values())
        report[f"val/{arm}"] = dict(recomputed=mine, abs_diff_pp=diff)
    # 시험: 양성 45시퀀스 → 도시6 → 세 시작점
    for arm in ("diff", "diff_nogeo"):
        per = []
        for k in (1, 2, 3):
            cities = []
            for seqs in CITY.values():
                rows = [metrics(OUT / "test_eval/test" / s.replace("/", "_") / f"{arm}_s{k}.npz") for s in seqs]
                rows = [r for r in rows if r]
                cities.append({m: np.mean([r[m] for r in rows]) for m in rows[0]})
            per.append({m: np.mean([c[m] for c in cities]) for m in cities[0]})
        mine = {m: 100 * float(np.mean([p[m] for p in per])) for m in per[0]}
        diff = {m: abs(mine[m] - res["test_percent"][arm]["mean"][m]) for m in mine}
        worst = max(worst, *diff.values())
        report[f"test/{arm}"] = dict(recomputed=mine, abs_diff_pp=diff)
    report["max_abs_diff_pp"] = worst
    report["status"] = "passed" if worst < 1e-9 else "failed"
    (OUT / "independent_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
