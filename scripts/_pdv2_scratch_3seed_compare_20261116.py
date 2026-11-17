#!/usr/bin/env python3
"""기본 BN 3시드 처음부터 모델의 시험46 짝지은 구간: BEVPlace++·균등 감산·현재 NSD, 전체·순방향만·역방향만.

방향 라벨은 원고 방향 표의 results/pdv2_review_direction_yaw_20261105/directions, 지표는
scripts/_review_direction_metrics.metric(선택 방향 양성 + 공통 음성)과 같은 정의를 쓴다.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from _pr_eval_lib import classify_queries, pr_sweep  # noqa: E402

KEYS = ("AP", "max_F1", "R@100P", "R@1")
LABELS = ("forward_only", "reverse_only", "mixed", "other_positive")
D3 = ROOT / "results/pdv2_scratch_pilot_20261116/defaultbn_3seed"
PILOT = ROOT / "results/pdv2_scratch_pilot_20261116/step3/test"
MAIN = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
BEV = ROOT / "results/pdv2_review_response_20261104/bev/test"
DIRS = ROOT / "results/pdv2_review_direction_yaw_20261105/directions"
CITY = json.loads((D3 / "city.json").read_text())


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


def ledger(kind, seed, name):
    if kind == "scratch":
        p, f = D3 / "step3/test" / name / f"s{seed}_u{U}.npz", "score"
    elif kind == "uniform":
        p, f = PILOT / name / "uniform_reselected.npz", "score"
    elif kind == "nsd":
        p, f = MAIN / "test" / name / f"diff_s{seed}.npz", "score"
    else:
        p, f = BEV / f"s{seed}" / (name + ".npz"), "score_negl2"
    with np.load(p) as z:
        return z[f], z["top1_geo_dist"], z["min_past_gt_dist"], z["query_idx"]


def main():
    rows = {}  # rows[kind][seed][seq][group]
    for kind in ("scratch", "uniform", "nsd", "bev"):
        for seed in (1, 2, 3):
            for seqs in CITY.values():
                for s in seqs:
                    name = s.replace("/", "_")
                    d = dict(np.load(DIRS / (name + ".npz")))
                    score, top1, minimum, q = ledger(kind, seed, name)
                    np.testing.assert_array_equal(q, d["query_idx"])
                    masks = dict(all=None, forward_only=d["group"] == 0, reverse_only=d["group"] == 1)
                    rows.setdefault(kind, {}).setdefault(seed, {})[s] = {g: metric(score, top1, minimum, m) for g, m in masks.items()}
    out = {}
    for g in ("all", "forward_only", "reverse_only"):
        def macro(kind, seed):
            cities = []
            for seqs in CITY.values():
                v = [rows[kind][seed][s][g] for s in seqs if rows[kind][seed][s][g] is not None]
                if v:
                    cities.append({k: np.mean([x[k] for x in v]) for k in KEYS})
            return {k: float(np.mean([c[k] for c in cities])) for k in KEYS}
        means = {kind: {k: float(np.mean([macro(kind, sd)[k] for sd in (1, 2, 3)])) for k in KEYS} for kind in rows}
        sds = {kind: {k: float(np.std([macro(kind, sd)[k] for sd in (1, 2, 3)], ddof=1)) for k in KEYS} for kind in rows}
        comp = {}
        for ref in ("bev", "uniform", "nsd"):
            rng = np.random.default_rng(20261104)
            draws = []
            for seqs in CITY.values():
                avail = [s for s in seqs if rows["scratch"][1][s][g] is not None]
                if not avail:
                    continue
                mat = np.array([[np.mean([100 * (rows["scratch"][sd][s][g][k] - rows[ref][sd][s][g][k]) for sd in (1, 2, 3)])
                                 for k in KEYS] for s in avail])
                idx = rng.integers(0, len(avail), size=(10000, len(avail)))
                draws.append(mat[idx].mean(1))
            boot = np.stack(draws).mean(0)
            comp[ref] = dict(point_pp={k: 100 * (means["scratch"][k] - means[ref][k]) for k in KEYS},
                             conditional_95_pp={k: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(KEYS)})
        out[g] = dict(mean=means, sd=sds, scratch_minus=comp)
        print(g)
        for kind in means:
            print("  ", kind, {k: (round(100 * means[kind][k], 3), round(100 * sds[kind][k], 3)) for k in KEYS})
        for ref, v in comp.items():
            print("   scratch-" + ref, {k: (round(v["point_pp"][k], 3), [round(x, 3) for x in v["conditional_95_pp"][k]]) for k in KEYS})
    (D3 / f"compare_u{U}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    U = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    main()
