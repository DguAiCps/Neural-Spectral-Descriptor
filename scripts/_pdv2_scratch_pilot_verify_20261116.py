#!/usr/bin/env python3
"""처음부터 학습 시범의 시험46 지표를 평가 모듈 없이 원장에서 독립 재계산한다(호스트, torch 없음).

python3 scripts/_pdv2_scratch_pilot_verify_20261116.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pdv2_scratch_pilot_20261116"
CITY = {
    "karlsruhe": ["kitti/00", "kitti/07", "kitti/08", "kitti360/0000"],
    "seoul": ["helipr/Town01", "helipr/Town02", "helipr/Town03"],
    "daejeon": [f"helipr/{p}0{i}" for p in ("KAIST", "DCC", "Riverside") for i in (4, 5, 6)]
    + [f"mulran/{p}0{i}" for p in ("KAIST", "DCC", "Riverside") for i in (1, 2, 3)],
    "ann_arbor": ["nclt/" + d for d in ("2012-01-08", "2012-02-04", "2012-03-17", "2012-05-11", "2012-08-04",
                                        "2012-11-04", "2012-11-16", "2013-01-10", "2013-02-23")],
    "stockholm": ["mcd/kth_day_06", "mcd/kth_day_09", "mcd/kth_day_10", "mcd/kth_night_01", "mcd/kth_night_04", "mcd/kth_night_05"],
    "hamburg": ["mcd/tuhh_day_02", "mcd/tuhh_day_03", "mcd/tuhh_day_04", "mcd/tuhh_night_07", "mcd/tuhh_night_08", "mcd/tuhh_night_09"],
}


def metrics(d):
    """임계값을 점수 내림차순으로 훑는 단순 구현. 동점은 한 임계값으로 묶는다."""
    pos = d["min_past_gt_dist"] <= 5
    keep = pos | (d["min_past_gt_dist"] > 25)
    ok = pos & (d["top1_geo_dist"] <= 5)
    s, y = d["score"][keep], ok[keep]
    npos = int(pos.sum())
    if npos == 0:
        return None
    precisions, recalls, fps = [1.0], [0.0], [0]
    tp = fp = 0
    for v in sorted(set(s.tolist()), reverse=True):
        sel = s == v
        tp += int(y[sel].sum())
        fp += int((~y[sel]).sum())
        precisions.append(tp / (tp + fp))
        recalls.append(tp / npos)
        fps.append(fp)
    p, r = np.array(precisions), np.array(recalls)
    ap = sum((r[i] - r[i - 1]) * p[i:].max() for i in range(1, len(r)))
    f1 = max(2 * a * b / (a + b) if a + b > 0 else 0 for a, b in zip(p, r))
    r100 = max(rr for rr, f in zip(r, fps) if f == 0)
    return {"AP": ap, "max_F1": f1, "R@100P": r100, "R@1": int(ok.sum()) / npos}


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    result = {}
    for tag in ("scratch_s1", "uniform_reselected"):
        cities = []
        for seqs in CITY.values():
            rows = []
            for s in seqs:
                d = dict(np.load(OUT / "step3/test" / s.replace("/", "_") / (tag + ".npz")))
                m = metrics(d)
                if m is not None:
                    rows.append(m)
            cities.append({k: np.mean([r[k] for r in rows]) for k in rows[0]})
        mine = {k: float(np.mean([c[k] for c in cities])) for k in cities[0]}
        ref = summary["test"]["results"][tag]["macro"]
        result[tag] = dict(independent=mine, pipeline=ref, max_abs_error=max(abs(mine[k] - ref[k]) for k in mine))
        print(tag, result[tag]["max_abs_error"], {k: round(100 * v, 3) for k, v in mine.items()})
    (OUT / "independent_verification.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
