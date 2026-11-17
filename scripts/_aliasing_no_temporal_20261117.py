"""수락 단계 앨리어싱: NSD 대 시간 증거 없음(중심 쌍만으로 수락) (2026-11-17).

정의는 scripts/_aliasing_accept_20261117.py와 같다(alias@r: 시퀀스마다 점수 내림차순으로 평가 질의를
수락하며 맞은 수락이 r×양성수에 처음 도달할 때까지 수락한 질의 중 반환 장소가 25 m 넘게 떨어진 비율).
시간 증거 없음 원장(ablations/ledgers/test/<seq>/no_temporal_s{4,5,6}.npz)은 NSD와 반환 장소가 같고
수락 점수만 중심 쌍이다. 구간: 시퀀스별 시드 평균 차이(NSD − 변형), 도시 안 시퀀스 재표집,
모든 실행이 도달한 시퀀스만 짝짓는다.
출력: results/pdv2_aliasing_accept_20261117/no_temporal.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _report_model_common_20261116 as C  # noqa: E402

ABL = C.ROOT / "results/pdv2_report_model_20261116/ablations/ledgers/test"
OUT = C.ROOT / "results/pdv2_aliasing_accept_20261117/no_temporal.json"
RECALLS = (0.5, 0.7)
KEYS = ["far"] + [f"alias@{r}" for r in RECALLS]

QIDX = {}
for q in C.SEQS:
    with np.load(C.DIRS / (q.replace("/", "_") + ".npz")) as z:
        QIDX[q] = z["query_idx"]


def ledger(path):
    with np.load(path) as z:
        return dict(q=z["query_idx"], score=z["score"].astype(float), top1=z["top1_geo_dist"], gt=z["min_past_gt_dist"])


def seq_values(d, q):
    ix = np.searchsorted(d["q"], QIDX[q])
    assert np.array_equal(d["q"][ix], QIDX[q]), q
    sc, t1, gt = d["score"][ix], d["top1"][ix], d["gt"][ix]
    pos, neg = gt <= 5, gt > 25
    if not pos.any():
        return None
    ev = pos | neg
    out = dict(far=float(np.mean(t1[pos] > 25)))
    o = np.argsort(-sc[ev], kind="stable")
    tps = np.cumsum((pos & (t1 <= 5))[ev][o])
    al = np.cumsum((t1 > 25)[ev][o])
    for r in RECALLS:
        k = int(np.searchsorted(tps, r * pos.sum()))
        out[f"alias@{r}"] = float(al[k] / (k + 1)) if k < len(tps) else None
    return out


def macro(rows, key):
    cities = []
    for v in C.CITY.values():
        vals = [rows[s][key] for s in v if rows.get(s) is not None and rows[s][key] is not None]
        if vals:
            cities.append(np.mean(vals))
    return 100 * float(np.mean(cities))


def paired(rows_a, rows_b, key):
    rng = np.random.default_rng(C.BOOT_SEED)
    draws, deltas, used = [], [], 0
    for v in C.CITY.values():
        ok = [s for s in v if all(r.get(s) is not None and r[s][key] is not None for r in rows_a + rows_b)]
        if not ok:
            continue
        x = np.array([100 * (np.mean([r[s][key] for r in rows_a]) - np.mean([r[s][key] for r in rows_b])) for s in ok])
        used += len(ok)
        deltas.append(x.mean())
        draws.append(x[rng.integers(0, len(ok), size=(C.BOOT, len(ok)))].mean(1))
    boot = np.stack(draws).mean(0)
    return dict(delta_pp=float(np.mean(deltas)), ci95_pp=np.quantile(boot, [0.025, 0.975]).tolist(),
                n_seq_paired=used, n_city=len(deltas))


runs = {
    "nsd": [{q: seq_values(C.load(f"new:{k}", q), q) for q in C.SEQS} for k in C.NEW_SEEDS],
    "no_temporal": [{q: seq_values(ledger(ABL / q.replace("/", "_") / f"no_temporal_s{k}.npz"), q) for q in C.SEQS}
                    for k in C.NEW_SEEDS],
}
res = {m: dict(mean={k: float(np.mean([macro(r, k) for r in rs])) for k in KEYS},
               per_run={k: [macro(r, k) for r in rs] for k in KEYS},
               n_excluded={k: [sum(1 for v in r.values() if v is not None and v[k] is None) for r in rs]
                           for k in KEYS[1:]})
       for m, rs in runs.items()}
res["nsd_minus_no_temporal"] = {k: paired(runs["nsd"], runs["no_temporal"], k) for k in KEYS}
res["definition"] = __doc__
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
for m in ("nsd", "no_temporal"):
    print(m, {k: round(v, 2) for k, v in res[m]["mean"].items()}, res[m]["n_excluded"])
for k, v in res["nsd_minus_no_temporal"].items():
    print(k, round(v["delta_pp"], 2), [round(x, 2) for x in v["ci95_pp"]], v["n_seq_paired"])
