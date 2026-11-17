#!/usr/bin/env python3
"""처음부터 학습 구성의 시험46 시드 평균±SD와 짝지은 구간(균등 감산·현재 NSD·BEVPlace++), 방향별 대조.

scripts/_pdv2_scratch_3seed_compare_20261116.py와 같은 정의다. 시드 수가 다르면 시퀀스별 시드 평균의 차이를 쓴다.
지표는 평가 파이프라인과 별개인 _pr_eval_lib.pr_sweep로 원장에서 다시 계산한다.

python3 scripts/_pdv2_scratch_search_compare_20261116.py <구성 이름> <갱신 수> <시드,시드,...>
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from _pr_eval_lib import classify_queries, pr_sweep  # noqa: E402

KEYS = ("AP", "max_F1", "R@100P", "R@1")
OUT = ROOT / "results/pdv2_scratch_search_20261116"
MAIN = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
BEV = ROOT / "results/pdv2_review_response_20261104/bev/test"
DIRS = ROOT / "results/pdv2_review_direction_yaw_20261105/directions"
CITY = json.loads((ROOT / "results/pdv2_scratch_pilot_20261116/defaultbn_3seed/city.json").read_text())


def metric(score, top1, minimum, mask=None):
    qc = classify_queries(minimum)
    if mask is not None:
        qc["positive"] &= mask
        qc["ambiguous"] = ~(qc["positive"] | qc["negative"])
        qc["n_pos"] = int(qc["positive"].sum())
        qc["n_ambiguous"] = int(qc["ambiguous"].sum())
    if not qc["n_pos"]:
        return None
    v = pr_sweep(score, top1, qc)
    return {**{k: float(v[k]) for k in KEYS[:3]}, "R@1": float(np.mean(top1[qc["positive"]] <= 5))}


def ledger(kind, seed, name, cfg, u):
    if kind == "scratch":
        p, f = OUT / "ledgers/test" / name / f"{cfg}@s{seed}@u{u}.npz", "score"
    elif kind == "uniform":
        p, f = OUT / "ledgers/test" / name / "uniform.npz", "score"
    elif kind == "nsd":
        p, f = MAIN / "test" / name / f"diff_s{seed}.npz", "score"
    else:
        p, f = BEV / f"s{seed}" / (name + ".npz"), "score_negl2"
    with np.load(p) as z:
        return z[f], z["top1_geo_dist"], z["min_past_gt_dist"], z["query_idx"]


def main(cfg, u, seeds):
    seedsets = dict(scratch=seeds, uniform=[0], nsd=[1, 2, 3], bev=[1, 2, 3])
    rows = {}
    for kind, ss in seedsets.items():
        for seed in ss:
            for seqs in CITY.values():
                for s in seqs:
                    name = s.replace("/", "_")
                    d = dict(np.load(DIRS / (name + ".npz")))
                    score, top1, minimum, q = ledger(kind, seed, name, cfg, u)
                    np.testing.assert_array_equal(q, d["query_idx"])
                    masks = dict(all=None, forward_only=d["group"] == 0, reverse_only=d["group"] == 1)
                    rows.setdefault(kind, {}).setdefault(seed, {})[s] = {g: metric(score, top1, minimum, m) for g, m in masks.items()}
    out = dict(cfg=cfg, u=u, seeds=seeds)
    for g in ("all", "forward_only", "reverse_only"):
        def macro(kind, seed):
            cities = []
            for seqs in CITY.values():
                v = [rows[kind][seed][s][g] for s in seqs if rows[kind][seed][s][g] is not None]
                if v:
                    cities.append({k: np.mean([x[k] for x in v]) for k in KEYS})
            return {k: float(np.mean([c[k] for c in cities])) for k in KEYS}
        per_seed = {kind: {sd: macro(kind, sd) for sd in ss} for kind, ss in seedsets.items()}
        means = {kind: {k: float(np.mean([v[k] for v in per_seed[kind].values()])) for k in KEYS} for kind in rows}
        sds = {kind: {k: float(np.std([v[k] for v in per_seed[kind].values()], ddof=1)) if len(per_seed[kind]) > 1 else None for k in KEYS}
               for kind in rows}
        comp = {}
        for ref in ("bev", "uniform", "nsd"):
            rng = np.random.default_rng(20261104)
            draws = []
            for seqs in CITY.values():
                avail = [s for s in seqs if rows["scratch"][seeds[0]][s][g] is not None]
                if not avail:
                    continue
                mat = np.array([[100 * (np.mean([rows["scratch"][sd][s][g][k] for sd in seeds])
                                        - np.mean([rows[ref][sd][s][g][k] for sd in seedsets[ref]])) for k in KEYS] for s in avail])
                idx = rng.integers(0, len(avail), size=(10000, len(avail)))
                draws.append(mat[idx].mean(1))
            boot = np.stack(draws).mean(0)
            comp[ref] = dict(point_pp={k: 100 * (means["scratch"][k] - means[ref][k]) for k in KEYS},
                             conditional_95_pp={k: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(KEYS)})
        out[g] = dict(mean=means, sd=sds, per_seed=per_seed, scratch_minus=comp)
        print(g)
        for kind in means:
            print("  ", kind, {k: (round(100 * means[kind][k], 3), round(100 * (sds[kind][k] or 0), 3)) for k in KEYS})
        for ref, v in comp.items():
            print("   scratch-" + ref, {k: (round(v["point_pp"][k], 3), [round(x, 3) for x in v["conditional_95_pp"][k]]) for k in KEYS})
    target = OUT / "compare" / f"{cfg}@u{u}@s{'-'.join(map(str, seeds))}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")])
