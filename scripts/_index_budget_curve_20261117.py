#!/usr/bin/env python3
"""후보 수 곡선: 총 후보 예산 K(20·50·100)에서 이중 색인(원키 top-K/2 ∪ BEV top-K/2)과 BEV 단독을 비교한다.

검색·점수 경로는 scripts/_report_model_ablations_20261116.py의 색인 예산 부분과 같다(승자 = 후보 집합 안에서 네 융합 채널
(2,1,2,2) 최대, 수락 = P.score(z, plan, λ=0.25)). K=200은 기존 원장(dual_100_100·aux_200·aux_matched)을 쓴다.
팔: dual_K(합집합), aux_K(BEV top-K), auxm_K(BEV top-|합집합|), keym_K(원키 top-|합집합|).
검사: K=200 팔을 같은 코드로 다시 만들어 기존 원장과 승자·점수가 같은지 시퀀스마다 확인한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_index_budget_curve_20261117.py <action>
  worker --sequence S | campaign --workers 2 | finish
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _report_model_ablations_20261116 as A  # noqa: E402

OUT = ROOT / "results/pdv2_index_budget_curve_20261117"
LED = OUT / "ledgers"
BUDGETS = (20, 50, 100)
SEEDS = A.SEEDS
KEYS = A.KEYS
GROUPS = A.GROUPS


def arms_for(key_scores, aux_scores, K):
    from _pdv2_contrast_index_budget import topk
    union = np.union1d(topk(key_scores, K // 2), topk(aux_scores, K // 2))
    n = len(union)
    return {f"dual_{K}": union, f"aux_{K}": np.sort(topk(aux_scores, K)),
            f"auxm_{K}": np.sort(topk(aux_scores, n)), f"keym_{K}": np.sort(topk(key_scores, n))}


def worker(sequence: str) -> None:
    import torch
    S = A.modules()
    P, run, prior, pipeline = S.P, S.run, S.prior, S.pipeline
    name = sequence.replace("/", "_")
    folder = LED / name
    folder.mkdir(parents=True, exist_ok=True)
    old = A.LED / "test" / name
    cache, pos, minimum, travel, bank, _, _, _ = P.inputs("test", sequence)
    graph, _, _ = prior.graph_inputs("test", sequence)
    raw = pipeline.raw_key(cache["descriptors"])
    new_w, lam = P.selected()
    base_pool = run.ARR(P.TEST / "test" / name / "candidate_pool.npz")
    q = base_pool["query"]
    groups = []
    for t in q:
        ks, as_ = raw[:t - 29] @ raw[t], bank[0][:t - 29] @ bank[0][t]
        g = {}
        for K in BUDGETS:
            g.update(arms_for(ks, as_, K))
        g["dual_200"] = arms_for(ks, as_, 200)["dual_200"]
        g["aux_200"] = arms_for(ks, as_, 200)["aux_200"]
        groups.append(g)
    names = list(groups[0])
    sizes = {a: np.array([len(g[a]) for g in groups]) for a in names}
    hits = {a: np.array([bool((np.linalg.norm(pos[g[a]] - pos[t], axis=1) <= 5).any()) for t, g in zip(q, groups)]) for a in names}

    def row_of(j):
        assert (j <= q - 30).all()
        return dict(query_idx=q, j_star=j, min_past_gt_dist=minimum[q],
                    top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1), group=np.zeros(len(q), np.int64))

    def score(z, plan):
        return P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(q)).astype(np.float64)

    tick = time.time()
    checks = {}
    with torch.no_grad():
        for k in SEEDS:
            if all((folder / f"{a}_s{k}.npz").exists() for a in names if not a.endswith("_200")):
                continue
            net = A.modules().load(A.CFG, k, A.UPD)
            z = net(graph)
            change = run.base.variant_inputs(z.cpu().numpy(), 8)
            sel = {a: [] for a in names}
            for t, g in zip(q, groups):
                all_ids = np.unique(np.concatenate(list(g.values())))
                sa = run.base.align_scores(change[1], change[2], int(t), all_ids)
                sc = run.base.column_align_scores(change[3], int(t), all_ids)
                for a, ids in g.items():
                    loc = np.searchsorted(all_ids, ids)
                    stat = (new_w[0] * run.base.minmax(raw[ids] @ raw[t]) + new_w[1] * run.base.minmax(bank[0][ids] @ bank[0][t])).astype(np.float32)
                    value = stat + new_w[2] * run.base.minmax(sa[loc]) + new_w[3] * run.base.minmax(sc[loc])
                    sel[a].append(int(ids[np.argmax(value)]))
            for a in names:
                ja = np.asarray(sel[a], np.int64)
                if a.endswith("_200"):  # 기존 원장 재현 검사
                    ref = np.load(old / ({"dual_200": "dual_100_100", "aux_200": "aux_200"}[a] + f"_s{k}.npz"))
                    checks[f"{a}_s{k}"] = bool(np.array_equal(ja, ref["j_star"]))
                    checks[f"{a}_s{k}_mismatch"] = int((ja != ref["j_star"]).sum()); checks[f"{a}_s{k}_n"] = len(ja)
                    continue
                row = dict(row_of(ja), candidate_count=sizes[a], candidate_hit=hits[a])
                np.savez_compressed(folder / f"{a}_s{k}.npz", **row, score=score(z, prior.base.prior.plan(travel, q, ja)))
            del net, z, change
    A.write(folder / "done.json", dict(seconds=time.time() - tick, winner_checks=checks))
    print("WORKER", sequence, round(time.time() - tick, 1), flush=True)


def campaign(workers: int) -> None:
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    tick = time.time()

    def one(s):
        with (OUT / "logs" / f"{s.replace('/', '_')}.log").open("a") as f:
            r = subprocess.run([sys.executable, __file__, "worker", "--sequence", s], stdout=f, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError(s)
        return s

    with ThreadPoolExecutor(workers) as ex:
        for i, s in enumerate(ex.map(one, A.sequences("test")), 1):
            print("DONE", i, s, round(time.time() - tick, 1), flush=True)
    A.write(OUT / "logs" / "campaign.done.json", dict(seconds=time.time() - tick, workers=workers))


def finish() -> None:
    from _pr_eval_lib import classify_queries, pr_sweep
    DIRS = ROOT / "results/pdv2_review_direction_yaw_20261105/directions"
    CITY = json.loads((ROOT / "results/pdv2_scratch_pilot_20261116/defaultbn_3seed/city.json").read_text())
    seqs = A.sequences("test")

    def path_of(c, s):
        name = s.replace("/", "_")
        arm, seed = c.rsplit("_s", 1)
        if arm.endswith("_200"):
            arm = {"dual_200": "dual_100_100", "aux_200": "aux_200", "auxm_200": "aux_matched", "keym_200": "key_matched"}[arm]
            return A.LED / "test" / name / f"{arm}_s{seed}.npz"
        return LED / name / f"{c}.npz"

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

    labels = {}
    for s in seqs:
        with np.load(DIRS / (s.replace("/", "_") + ".npz")) as d:
            labels[s] = (d["query_idx"].copy(), d["group"].copy())
    arms = [f"{a}_{K}" for K in BUDGETS + (200,) for a in ("dual", "aux", "auxm", "keym")]
    rows, cand = {}, {}
    for a in arms:
        for k in SEEDS:
            c = f"{a}_s{k}"
            rows[c], cand[c] = {}, {}
            for s in seqs:
                with np.load(path_of(c, s)) as z:
                    d = {kk: z[kk] for kk in z.files}
                qi, grp = labels[s]
                np.testing.assert_array_equal(d["query_idx"], qi)
                masks = dict(all=None, forward_only=grp == 0, reverse_only=grp == 1)
                rows[c][s] = {g: metric(d["score"], d["top1_geo_dist"], d["min_past_gt_dist"], m) for g, m in masks.items()}
                pos = d["min_past_gt_dist"] <= 5
                cand[c][s] = (float(np.mean(d["candidate_hit"][pos])) if pos.any() else None, float(np.mean(d["candidate_count"])))

    def macro(c, g, key):
        vals = []
        for ss in CITY.values():
            v = [rows[c][s][g][key] for s in ss if rows[c][s][g] is not None]
            if v:
                vals.append(np.mean(v))
        return float(np.mean(vals))

    def cmacro(c, i):
        vals = []
        for ss in CITY.values():
            v = [cand[c][s][i] for s in ss if cand[c][s][i] is not None]
            if v:
                vals.append(np.mean(v))
        return float(np.mean(vals))

    def paired(a, b, g):
        rng = np.random.default_rng(20261117)
        draws, cd = [], {}
        for city, ss in CITY.items():
            avail = [s for s in ss if rows[f"{a}_s4"][s][g] is not None]
            if not avail:
                continue
            mat = np.array([[100 * (np.mean([rows[f"{a}_s{k}"][s][g][m] for k in SEEDS]) - np.mean([rows[f"{b}_s{k}"][s][g][m] for k in SEEDS]))
                             for m in KEYS] for s in avail])
            cd[city] = mat.mean(0)
            idx = rng.integers(0, len(avail), size=(10000, len(avail)))
            draws.append(mat[idx].mean(1))
        boot = np.stack(draws).mean(0)
        point = np.mean(list(cd.values()), 0)
        return {m: dict(delta=float(point[i]), ci=np.quantile(boot[:, i], [0.025, 0.975]).tolist()) for i, m in enumerate(KEYS)}

    summary = {}
    for K in BUDGETS + (200,):
        e = {}
        for a in ("dual", "aux", "auxm", "keym"):
            tags = [f"{a}_{K}_s{k}" for k in SEEDS]
            e[a] = {g: {m: float(np.mean([macro(t, g, m) for t in tags])) * 100 for m in KEYS} for g in GROUPS}
            e[a]["candidate_recall"] = float(np.mean([cmacro(t, 0) for t in tags])) * 100
            e[a]["candidate_mean"] = float(np.mean([cmacro(t, 1) for t in tags]))
        e["dual_minus"] = {b: {g: paired(f"dual_{K}", f"{b}_{K}", g) for g in GROUPS} for b in ("aux", "auxm")}
        summary[str(K)] = e
    A.write(OUT / "summary.json", dict(
        model=dict(cfg=A.CFG, seeds=list(SEEDS), updates=A.UPD, fusion=[2, 1, 2, 2], lam=0.25),
        arms="dual_K = 원키 top-K/2 ∪ BEV top-K/2, aux_K = BEV top-K, auxm_K = BEV top-|합집합|, keym_K = 원키 top-|합집합|. K=200은 기존 절제 원장.",
        aggregation="양성 있는 시퀀스 -> 6도시 균등 -> 시드 4·5·6 평균. 구간은 도시 안 시퀀스 재표집 10,000회(시퀀스별 시드 평균 차이).",
        test_used_for_selection=False, summary=summary))
    for K in BUDGETS + (200,):
        e = summary[str(K)]
        print(K, {a: (round(e[a]["all"]["AP"], 2), round(e[a]["all"]["R@100P"], 2), round(e[a]["candidate_recall"], 2), round(e[a]["candidate_mean"], 1))
                  for a in ("dual", "aux", "auxm", "keym")})
        for b in ("aux", "auxm"):
            print("   dual-", b, {g: {m: (round(v["delta"], 2), [round(x, 2) for x in v["ci"]]) for m, v in e["dual_minus"][b][g].items() if m in ("AP", "R@100P")} for g in GROUPS})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("worker", "campaign", "finish"))
    ap.add_argument("--sequence")
    ap.add_argument("--workers", type=int, default=2)
    x = ap.parse_args()
    {"worker": lambda: worker(x.sequence), "campaign": lambda: campaign(x.workers), "finish": finish}[x.action]()
