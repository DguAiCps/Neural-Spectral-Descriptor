#!/usr/bin/env python3
"""재설계 B의 시험46 독립 재계산과 짝지은 구간(호스트, torch 없음).

python3 scripts/_pdv2_pretrain_redesign_B_compare_20261116.py [시드 목록, 예: 1 또는 1,2,3]

- 독립 재계산: _pr_eval_lib.pr_sweep 경로와 단순 임계값 훑기(_pdv2_scratch_pilot_verify_20261116.metrics)로
  원장에서 도시 매크로를 다시 계산해 파이프라인 summary.json과 대조한다.
- 구간: 도시 안 시퀀스 재표집 10,000회(rng 20261104). 비교 대상은 균등 감산(재선택, 시드 없음)과
  BEVPlace++(시퀀스별 3시드 평균). B 모델도 시퀀스별 시드 평균(시드가 같으면 시드 짝 차이의 평균과 같다).
  전체·순방향만·역방향만. 후보: 1단계 단독(stage1), 2단계 lr3e-4 4096(u4096)·1536(u1536).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from _pr_eval_lib import classify_queries, pr_sweep  # noqa: E402
from _pdv2_scratch_pilot_verify_20261116 import metrics as simple_metrics  # noqa: E402

KEYS = ("AP", "max_F1", "R@100P", "R@1")
OUT = ROOT / "results/pdv2_pretrain_redesign_20261116/B"
PILOT = ROOT / "results/pdv2_scratch_pilot_20261116"
MAIN = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
BEV = ROOT / "results/pdv2_review_response_20261104/bev/test"
DIRS = ROOT / "results/pdv2_review_direction_yaw_20261105/directions"
CITY = json.loads((PILOT / "defaultbn_3seed/city.json").read_text())


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


FILES = {1: dict(stage1="B_stage1", u4096="B_selected", u1536="B_s2_lr3e-4_u1536")}
for _sd in (2, 3):
    FILES[_sd] = dict(stage1="B_stage1", u4096="B_s2_lr3e-4_u4096", u1536="B_s2_lr3e-4_u1536")
MODELS = ("stage1", "u4096", "u1536")


def run_dir(seed):
    return OUT if seed == 1 else OUT / f"s{seed}"


def ledger(kind, name, seed=1):
    if kind == "uniform":
        p, f = PILOT / "step3/test" / name / "uniform_reselected.npz", "score"
    elif kind == "nsd":
        p, f = MAIN / "test" / name / f"diff_s{seed}.npz", "score"
    elif kind == "bev":
        p, f = BEV / f"s{seed}" / (name + ".npz"), "score_negl2"
    elif kind == "repro_uniform":
        p, f = run_dir(seed) / "eval/test" / name / "repro_uniform.npz", "score"
    else:
        p, f = run_dir(seed) / "eval/test" / name / (FILES[seed][kind] + ".npz"), "score"
    with np.load(p) as z:
        return z[f], z["top1_geo_dist"], z["min_past_gt_dist"], z["query_idx"]


def main():
    seeds = SEEDS
    summaries = {sd: json.loads((run_dir(sd) / "summary.json").read_text()) for sd in seeds}
    models = list(MODELS)
    kinds = models + ["repro_uniform", "uniform", "nsd", "bev"]
    rows = {}  # rows[kind][seq][group] (시드 평균)
    seed_rows = {}  # seed_rows[kind][seed][seq] (전체 group, 시드별)
    simple = {}
    for kind in kinds:
        kseeds = (1, 2, 3) if kind in ("nsd", "bev") else ((seeds[0],) if kind == "repro_uniform" else seeds)
        for seqs in CITY.values():
            for s in seqs:
                name = s.replace("/", "_")
                d = dict(np.load(DIRS / (name + ".npz")))
                per = []
                for sd in kseeds:
                    score, top1, minimum, q = ledger(kind, name, sd)
                    np.testing.assert_array_equal(q, d["query_idx"])
                    masks = dict(all=None, forward_only=d["group"] == 0, reverse_only=d["group"] == 1)
                    per.append({g: metric(score, top1, minimum, m) for g, m in masks.items()})
                    seed_rows.setdefault(kind, {}).setdefault(sd, {})[s] = per[-1]
                    if kind in models or kind == "repro_uniform":
                        simple.setdefault((kind, sd), {})[s] = simple_metrics(dict(score=score, top1_geo_dist=top1, min_past_gt_dist=minimum))
                rows.setdefault(kind, {})[s] = {
                    g: (None if per[0][g] is None else {k: float(np.mean([p[g][k] for p in per])) for k in KEYS})
                    for g in per[0]}

    def macro(kind, g, table=None):
        table = table or {s: rows[kind][s][g] for s in rows[kind]}
        cities = []
        for seqs in CITY.values():
            v = [table[s] for s in seqs if table[s] is not None]
            if v:
                cities.append({k: np.mean([x[k] for x in v]) for k in KEYS})
        return {k: float(np.mean([c[k] for c in cities])) for k in KEYS}

    # 독립 재계산(시드별, 파이프라인 summary.json 대비)
    indep = {}
    for (kind, sd), table in simple.items():
        tag = "repro_uniform" if kind == "repro_uniform" else FILES[sd][kind]
        ref = summaries[sd]["test"]["results"][tag]["macro"]
        a = macro(kind, "all", {s: seed_rows[kind][sd][s]["all"] for s in seed_rows[kind][sd]})
        b = macro(kind, "all", table)
        indep[f"{kind}_s{sd}"] = dict(pr_sweep_path=a, simple_sweep=b,
                                     max_abs_error_pr_sweep=max(abs(a[k] - ref[k]) for k in KEYS),
                                     max_abs_error_simple=max(abs(b[k] - ref[k]) for k in KEYS))
    for kind in ("uniform", "nsd", "bev"):
        indep[kind] = dict(macro=macro(kind, "all"))
    # 시드별 매크로와 평균±SD
    per_seed = {kind: {sd: {g: macro(kind, g, {s: seed_rows[kind][sd][s][g] for s in seed_rows[kind][sd]})
                            for g in ("all", "forward_only", "reverse_only")} for sd in seeds} for kind in models}
    spread = {kind: {g: {k: dict(mean=float(np.mean([per_seed[kind][sd][g][k] for sd in seeds])),
                                 sd=float(np.std([per_seed[kind][sd][g][k] for sd in seeds], ddof=1)) if len(seeds) > 1 else None,
                                 seeds=[per_seed[kind][sd][g][k] for sd in seeds]) for k in KEYS}
                     for g in ("all", "forward_only", "reverse_only")} for kind in models}
    out = dict(seeds=list(seeds), independent=indep, per_seed=per_seed, spread=spread, comparisons={})
    for g in ("all", "forward_only", "reverse_only"):
        comp = {}
        for kind in models:
            for ref in ("uniform", "bev"):
                rng = np.random.default_rng(20261104)
                draws = []
                for seqs in CITY.values():
                    avail = [s for s in seqs if rows[kind][s][g] is not None]
                    if not avail:
                        continue
                    mat = np.array([[100 * (rows[kind][s][g][k] - rows[ref][s][g][k]) for k in KEYS] for s in avail])
                    idx = rng.integers(0, len(avail), size=(10000, len(avail)))
                    draws.append(mat[idx].mean(1))
                boot = np.stack(draws).mean(0)
                a, b = macro(kind, g), macro(ref, g)
                comp[f"{kind}-{ref}"] = dict(
                    point_pp={k: 100 * (a[k] - b[k]) for k in KEYS},
                    conditional_95_pp={k: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(KEYS)})
        out["comparisons"][g] = dict(macro={kind: macro(kind, g) for kind in kinds}, deltas=comp)
        print(g)
        for kind in kinds:
            print("  ", kind, {k: round(100 * v, 3) for k, v in macro(kind, g).items()})
        for name, v in comp.items():
            print("   ", name, {k: (round(v["point_pp"][k], 3), [round(x, 3) for x in v["conditional_95_pp"][k]]) for k in KEYS})
    for kind, v in indep.items():
        if "max_abs_error_simple" in v:
            print("independent", kind, v["max_abs_error_pr_sweep"], v["max_abs_error_simple"])
    for kind in models:
        print("spread", kind, {k: (round(100 * v["mean"], 3), None if v["sd"] is None else round(100 * v["sd"], 3))
                               for k, v in spread[kind]["all"].items()})
    (OUT / ("compare_s" + "".join(map(str, seeds)) + ".json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    SEEDS = tuple(int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1").split(","))
    main()
