#!/usr/bin/env python3
"""현재 주표 질의에서 15키프레임 관측 대기를 다시 집계한다."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"results/pdv2_review_response_20261104"


def stats(value: np.ndarray) -> dict:
    """단위 변환 없이 캐시의 초·미터 분위수를 요약한다."""
    assert np.isfinite(value).all() and (value >= 0).all()
    return dict(n=len(value), mean=float(value.mean()), median=float(np.median(value)),
                p95=float(np.percentile(value, 95)), max=float(value.max()))


def main() -> None:
    result = {}
    for role in ("val", "test"):
        per, seconds, meters = {}, [], []
        for path in sorted((RUN/"nsd"/role/"s1").glob("*.json")):
            record = json.loads(path.read_text())
            cache = next(ROOT/p for p in record["inputs"] if Path(p).name.startswith("cache_"))
            ledger = ROOT/record["records"]["main"]["path"]
            with np.load(cache) as c, np.load(ledger) as r:
                q = r["query_idx"]
                t = c["timestamps"]
                pose = c["poses"][:, :3, 3]
                distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pose, axis=0), axis=1))]
                sec, meter = t[q+15]-t[q], distance[q+15]-distance[q]
                per[record["sequence"]] = dict(seconds=stats(sec), odom_meters=stats(meter))
                seconds.append(sec); meters.append(meter)
        result[role] = dict(pooled=dict(seconds=stats(np.concatenate(seconds)),
                                       odom_meters=stats(np.concatenate(meters))), per_sequence=per)
    (RUN/"delay.json").write_text(json.dumps(dict(delay_keyframes=15, aggregation="질의별 합산 분위수; 모델·설정 선택 없음", **result),
                                           ensure_ascii=False, indent=2)+"\n")
    print({k:v["pooled"] for k,v in result.items()})


if __name__ == "__main__": main()
