"""앨리어싱 분석: 고르는 단계의 먼 곳 반환율과 수락된 루프 중 앨리어싱 비율(2026-11-17).

정의(고정):
- 평가 질의 = 방향 파일(C.DIRS)의 공통 질의. 양성 gt<=5 m, 음성 gt>25 m, 중간 제외.
- (a) far: 양성 중 반환 장소가 25 m 넘게 떨어진 비율(top1>25).
  검산용 err5: 양성 중 top1>5 비율(=1-R@1).
- (b) alias@r: 시퀀스마다 점수 내림차순으로 평가 질의(양성∪음성)를 수락하며, 맞은 수락(양성 & top1<=5)이
  r×양성수에 처음 도달하는 지점까지 수락한 질의 중 top1>25인 것의 비율. 도달 못 하면 그 실행에서 제외.
- 집계: 시퀀스 → 도시 평균 → 6도시 균등 평균 → 실행(시드) 평균.
- 구간: NSD − BEVPlace++, 시퀀스별 시드 평균 차이를 도시 안 시퀀스 재표집(C.BOOT회, C.BOOT_SEED) 95% 구간.
  (b)는 두 방법 모두 모든 시드에서 도달한 시퀀스만 짝짓는다.
출력: results/pdv2_aliasing_accept_20261117/summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _report_model_common_20261116 as C  # noqa: E402

SUMMARY = C.ROOT / "results/pdv2_deepring_scaleinit_integration_20261110/baseline_seeds_summary_seqot_stable_ringpp_i2_fresco_kgf_deepring_scaleinit.json"
OUT = C.ROOT / "results/pdv2_aliasing_accept_20261117/summary.json"
PREFIX = {"scpp": "dump_scpp", "fresco": "dump_fresco", "ring": "dump_ring", "ringpp": "dump_ringpp_i2",
          "disco": "dump_disco", "deepring": "dump_deepring", "ringsharp": "dump_ringsharp", "helios": "dump_helios",
          "seqot": "dump_seqotl2", "bevplace": "dump_bevplace"}
ORDER = ["scpp", "fresco", "ring", "ringpp", "disco", "deepring", "ringsharp", "helios", "seqot", "bevplace", "nsd"]
RECALLS = (0.5, 0.7)
KEYS = ["far", "err5"] + [f"alias@{r}" for r in RECALLS]

S = json.loads(SUMMARY.read_text())
QIDX = {}
for q in C.SEQS:
    with np.load(C.DIRS / (q.replace("/", "_") + ".npz")) as z:
        QIDX[q] = z["query_idx"]


def loaders(m):
    if m == "nsd":
        return {f"s{k}": (lambda q, t=f"new:{k}": C.load(t, q)) for k in C.NEW_SEEDS}
    e = S["methods"][m]
    out = {}
    for sd, v in e["per_seed"].items():
        def f(q, src=C.ROOT / v["source"], pre=PREFIX[m], key=e["score"]):
            with np.load(src / f"{pre}_{q.replace('/', '_')}.npz") as z:
                return dict(q=z["query_idx"], score=z[key].astype(float), top1=z["top1_geo_dist"], gt=z["min_past_gt_dist"])
        out[sd] = f
    return out


def seq_values(d, q):
    ix = np.searchsorted(d["q"], QIDX[q])
    assert np.array_equal(d["q"][ix], QIDX[q]), q
    sc, t1, gt = d["score"][ix], d["top1"][ix], d["gt"][ix]
    pos, neg = gt <= 5, gt > 25
    if not pos.any():
        return None
    ev = pos | neg
    out = dict(far=float(np.mean(t1[pos] > 25)), err5=float(np.mean(t1[pos] > 5)), n_pos=int(pos.sum()))
    o = np.argsort(-sc[ev], kind="stable")
    tps = np.cumsum((pos & (t1 <= 5))[ev][o])
    al = np.cumsum((t1 > 25)[ev][o])
    for r in RECALLS:
        k = int(np.searchsorted(tps, r * pos.sum()))
        out[f"alias@{r}"] = float(al[k] / (k + 1)) if k < len(tps) else None
    return out


def macro(rows, key):
    cities = []
    for c, v in C.CITY.items():
        vals = [rows[s][key] for s in v if rows.get(s) is not None and rows[s][key] is not None]
        if vals:
            cities.append(np.mean(vals))
    return float(np.mean(cities))


def paired(rows_a, rows_b, key):
    """시퀀스별 시드 평균 차이(%p), 도시 안 재표집. 두 방법 모든 시드에 값이 있는 시퀀스만."""
    rng = np.random.default_rng(C.BOOT_SEED)
    draws, deltas, used = [], [], 0
    for c, v in C.CITY.items():
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


res, rows_all = {}, {}
for m in ORDER:
    runs = {sd: {q: seq_values(f(q), q) for q in C.SEQS} for sd, f in loaders(m).items()}
    rows_all[m] = list(runs.values())
    per_run = {sd: {k: 100 * macro(rows, k) for k in KEYS} for sd, rows in runs.items()}
    excl = {sd: {f"alias@{r}": sorted(q for q, v in rows.items() if v is not None and v[f"alias@{r}"] is None)
                 for r in RECALLS} for sd, rows in runs.items()}
    res[m] = dict(
        runs=list(runs),
        mean={k: float(np.mean([p[k] for p in per_run.values()])) for k in KEYS},
        sd={k: (float(np.std([p[k] for p in per_run.values()], ddof=1)) if len(per_run) > 1 else None) for k in KEYS},
        per_run=per_run,
        excluded_sequences=excl,
        n_excluded={f"alias@{r}": {sd: len(e[f"alias@{r}"]) for sd, e in excl.items()} for r in RECALLS},
        n_seq_with_pos=sum(v is not None for v in next(iter(runs.values())).values()),
    )
    if m != "nsd":
        res[m]["table1_R@1_check"] = 100 * S["methods"][m]["mean_std"]["R@1"]["mean"]

ci = {k: paired(rows_all["nsd"], rows_all["bevplace"], k) for k in KEYS}
out = dict(
    definition=__doc__,
    nsd_runs=[f"new:{k}" for k in C.NEW_SEEDS], baseline_summary=str(SUMMARY.relative_to(C.ROOT)),
    eval_queries=str(C.DIRS.relative_to(C.ROOT)), boot=C.BOOT, boot_seed=C.BOOT_SEED,
    methods=res, nsd_minus_bevplace=ci,
)
C.write(OUT, out)

print(f"{'method':10s} {'far':>6s} {'1-R@1':>6s} {'(R@1)':>6s} {'a@50':>6s} {'a@70':>6s}  excl@50 excl@70")
for m in ORDER:
    v = res[m]["mean"]
    print(f"{m:10s} {v['far']:6.2f} {v['err5']:6.2f} {100 - v['err5']:6.2f} {v['alias@0.5']:6.2f} {v['alias@0.7']:6.2f}  "
          f"{res[m]['n_excluded']['alias@0.5']} {res[m]['n_excluded']['alias@0.7']}")
for k, v in ci.items():
    print(k, round(v["delta_pp"], 2), [round(x, 2) for x in v["ci95_pp"]], v["n_seq_paired"])
