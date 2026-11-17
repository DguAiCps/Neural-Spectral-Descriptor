#!/usr/bin/env python3
"""새 보고모델(처음부터 한 단계 학습, 시드 4·5·6, 고정 설정 융합 (2,1,2,2)·λ=0.25)의 추론 절제를 다시 계산한다.

검색·점수 경로는 scripts/_pdv2_scratch_search_20261116.py의 pipe()와 같다(재선택 융합으로 다시 계산한 후보 정적 점수,
pipeline.winners, prior.base.prior.plan, P.score(z, plan, λ)). 조건 정의는 기존 절제 스크립트를 따른다.

- 색인 예산(scripts/_pdv2_contrast_index_budget.py): 원키 top-200, BEV top-200, 이중 top-100+100(NSD), 원키 top-K_q, BEV top-K_q.
  후보 집합은 원키·BEV 크기 색인만으로 정해져 모델·융합과 무관하다. 승자는 각 후보 집합에서 네 융합 채널(새 융합)로 고른다.
  정적 부분합의 float32 변환 위치는 pipeline.winners와 같게 두어 이중 색인 조건이 NSD 승자와 비트 단위로 같게 했다.
- 거시 대조(scripts/_pdv2_macro_ablation.py): no_alignment(두 색인 코사인 부분합 최대로 반환, 새 융합), no_temporal(NSD 승자, 중심 쌍만으로 수락).
- 정제 없음(identity): z = 원래 복소 DFT 특징, 정렬 뱅크 = 입력 뱅크(pipeline.worker의 identity와 같음).
- 시간 창 방향(scripts/_pdv2_window_side_control.py): 균등 감산의 전달 칸을 과거 5개(칸 0~4)나 미래 5개(칸 6~10)로 줄인다.
  세기는 검증9 파이프라인 집계 AP 최대(동률이면 약한 세기)로 {-0.125,-0.25,-0.5,-1}에서 고른다.
- 무지연 정제 없음(scripts/_pdv2_zero_lag.py의 zl_identity): identity 승자, 수락의 질의 쪽 미래 오프셋 +1~+5를 없는 이웃으로 둔 plan.

재현 검사: 같은 경로로 NSD 시드 4~6 원장과 uniform 원장(S 원장)을 승자·점수 비트 단위로 재현한다. 옛 설정(융합 (1,1,2,1),
옛 후보 정적 점수)에서 기존 identity·zl_identity·past_0p5·future_0p25 원장도 재현해 조건 구현을 확인한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_report_model_ablations_20261116.py <action>
  worker --role test|val --sequence S --groups core,window_equal | window_selected | window_val
  campaign --role test|val --groups ... --workers 2
  select      검증9 세기 선택
  finish      집계·구간·요약 JSON
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

# 후보 색인 gemv(raw[:q-29] @ raw[q])는 BLAS 스레드 수에 따라 마지막 비트가 달라져 경계 동점이 바뀐다.
# 저장된 후보 합집합(candidate_pool.npz)과 옛 색인 예산 스크립트는 단일 스레드 결과와 같다(helipr/Riverside04 q=1663에서 확인).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
OUT = ROOT / "results/pdv2_report_model_20261116/ablations"
LED = OUT / "ledgers"
CFG = "x=log__n=ln__g=frozen__lr=1e-3__U=1536"
SEEDS = (4, 5, 6)
UPD = 512
ARMS = ("key_200", "aux_200", "dual_100_100", "key_matched", "aux_matched")
SIDES = ("past", "future")
STRENGTHS = (-0.125, -0.25, -0.5, -1.0)
KEYS = ("AP", "max_F1", "R@100P", "R@1")
GROUPS = ("all", "forward_only", "reverse_only")
DATA = Path("/workspace/data/nsd_results")
WSIDE = DATA / "pdv2_window_side_20261111"
ZLAG = DATA / "pdv2_zero_lag_20261112"
DENSE = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])  # neighbor_plans의 arc_dense 질의 오프셋 순서(_pdv2_zero_lag.py와 같음)
FUTURE = DENSE > 0


def tag(side: str, eta: float) -> str:
    return f"{side}_{abs(eta):g}".replace(".", "p")


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(path)


def read(path: Path):
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------- GPU 쪽
def modules():
    import torch  # noqa: F401
    import _pdv2_scratch_search_20261116 as S
    return S


def one_sided(prior, graph, side: str, eta: float):
    """_pdv2_window_side_control.patch_transform과 같은 한쪽 창 균등 감산."""
    full = graph.transport_mask
    mask = full.clone()
    if side == "past":
        mask[:, 6:] = 0
    else:
        mask[:, :5] = 0
    graph.transport_mask = mask
    try:
        return prior.filtered(graph, eta)
    finally:
        graph.transport_mask = full


def zero_lag_plan(sg, travel, q, j) -> dict:
    """_pdv2_zero_lag.zero_lag_plan과 같다: plan 안의 neighbor_plans에서 질의 쪽 미래 오프셋을 없는 이웃으로 바꾼다."""
    original = sg.neighbor_plans

    def past_neighbor_plans(path, query, candidate):
        result = original(path, query, candidate)
        qq, jj = np.asarray(query, np.int64), np.asarray(candidate, np.int64)
        qn, jn, valid = (v.copy() for v in result["arc_dense"])
        assert np.array_equal(qn - qq[None, :, None], np.broadcast_to(DENSE, qn.shape))
        qn[..., FUTURE] = qq[None, :, None]
        jn[..., FUTURE] = jj[None, :, None]
        valid[..., FUTURE] = False
        result["arc_dense"] = result["arc_unique"] = (qn, jn, valid)
        return result

    sg.neighbor_plans = past_neighbor_plans
    try:
        plan = sg.plan(travel, q, j)
    finally:
        sg.neighbor_plans = original
    assert (plan["query"] <= np.asarray(q)[:, None, None]).all()
    return plan


def center_only(plan: dict) -> dict:
    """_pdv2_macro_ablation.center_only와 같다."""
    weight = np.zeros_like(plan["weight"])
    weight[..., 0] = 1.0
    return dict(plan, weight=weight)


def worker(role: str, sequence: str, groups: list) -> None:
    import torch
    S = modules()
    P, run, prior, pipeline = S.P, S.run, S.prior, S.pipeline
    from _pdv2_contrast_index_budget import candidates as index_candidates
    sg = prior.base.prior
    name = sequence.replace("/", "_")
    folder = LED / role / name
    folder.mkdir(parents=True, exist_ok=True)
    src = (P.PIPE / "val" / name) if role == "val" else (P.TEST / "test" / name)
    sledger = S.LEDGERS / role / name
    cache, pos, minimum, travel, bank, _, _, _ = P.inputs(role, sequence)
    graph, _, _ = prior.graph_inputs(role, sequence)
    raw = pipeline.raw_key(cache["descriptors"])
    new_w, lam = P.selected()
    poolpath = src / "candidate_pool.npz"
    assert run.HASH(poolpath) == run.READ(poolpath.with_suffix(".json"))["sha256"]
    base_pool = run.ARR(poolpath)
    q = base_pool["query"]
    off = base_pool["offset"]
    static = []
    for n, t in enumerate(q):
        ids = base_pool["ids"][off[n]:off[n + 1]]
        static.append(new_w[0] * run.base.minmax(raw[ids] @ raw[t]) + new_w[1] * run.base.minmax(bank[0][ids] @ bank[0][t]))
    new_pool = dict(base_pool, static=np.concatenate(static).astype(np.float32))

    def row_of(j):
        assert (j <= q - 30).all()
        return dict(query_idx=q, j_star=j, min_past_gt_dist=minimum[q],
                    top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1), group=np.zeros(len(q), np.int64))

    def score(z, plan, lam_=lam):
        return P.batched(lambda pp: P.score(z, pp, lam_).cpu().numpy(), plan, len(q)).astype(np.float64)

    def metrics(row, value):
        m = run.base.independent_metrics(row, value)
        o = run.white.prior.BASE.exp.metrics(value, row)
        for k in KEYS:
            if m[k] is not None:
                np.testing.assert_allclose(m[k], o[k], atol=1e-12, rtol=0)
        return m

    def save(tagname, row, value, extra=None):
        np.savez_compressed(folder / (tagname + ".npz"), **row, score=value)
        write(folder / (tagname + ".json"), dict(metrics=metrics(row, value), sha256=run.HASH(folder / (tagname + ".npz")), **(extra or {})))

    def compare(tagname, row, value, ref_path):
        r = run.ARR(ref_path)
        rec = dict(reference=str(ref_path), query_equal=bool(np.array_equal(q, r["query_idx"])),
                   winners_equal=bool(np.array_equal(row["j_star"], r["j_star"])),
                   score_bitwise_equal=bool(np.array_equal(value, r["score"])),
                   score_max_error=float(np.abs(value - r["score"]).max()))
        write(folder / ("check_" + tagname + ".json"), rec)
        return rec

    def done(*tags):
        return all((folder / (t + ".json")).exists() for t in tags)

    tick = time.time()
    timing = {}
    with torch.no_grad():
        if "core" in groups:
            # --- 학습 없는 조건: uniform 재현, identity, zl_identity, 옛 설정 재현
            if not done("check_uniform"):
                z = prior.filtered(graph, -0.5)
                change = run.base.variant_inputs(z.cpu().numpy(), 8)
                j = pipeline.winners(new_pool, change, new_w)
                rec = compare("uniform", row_of(j), score(z, prior.base.prior.plan(travel, q, j)), sledger / "uniform.npz")
                assert rec["winners_equal"] and rec["score_bitwise_equal"], rec
            if not done("identity", "zl_identity") or (role == "test" and not done("check_old_identity", "check_old_zl_identity")):
                z = graph.spectrum
                j = pipeline.winners(new_pool, bank, new_w)
                save("identity", row_of(j), score(z, prior.base.prior.plan(travel, q, j)))
                save("zl_identity", row_of(j), score(z, zero_lag_plan(sg, travel, q, j)))
                if role == "test":
                    jo = pipeline.winners(base_pool, bank, P.CURRENT_W)
                    rec = compare("old_identity", row_of(jo), score(z, prior.base.prior.plan(travel, q, jo), 0.25), src / "identity.npz")
                    assert rec["winners_equal"] and rec["score_max_error"] < 8e-6, rec
                    rec = compare("old_zl_identity", row_of(jo), score(z, zero_lag_plan(sg, travel, q, jo), 0.25),
                                  ZLAG / "test_eval/test" / name / "zl_identity.npz")
                    assert rec["winners_equal"] and rec["score_max_error"] < 8e-6, rec
            timing["training_free"] = time.time() - tick
            # --- 모델 조건
            pools, sizes, hits = [], {a: [] for a in ARMS}, {a: [] for a in ARMS}
            if role == "test" and not done(*[f"{a}_s{k}" for a in ARMS for k in SEEDS]):
                for n, t in enumerate(q):
                    group = index_candidates(raw[:t - 29] @ raw[t], bank[0][:t - 29] @ bank[0][t])
                    np.testing.assert_array_equal(group["dual_100_100"], base_pool["ids"][off[n]:off[n + 1]])
                    for a, ids in group.items():
                        assert len(ids) <= 200 and (ids <= t - 30).all()
                        sizes[a].append(len(ids))
                        hits[a].append(bool((np.linalg.norm(pos[ids] - pos[t], axis=1) <= 5).any()))
                    pools.append(group)
            static_j = np.asarray([base_pool["ids"][a:b][np.argmax(new_pool["static"][a:b])] for a, b in zip(off[:-1], off[1:])], np.int64)
            for k in SEEDS:
                st = time.time()
                mtag = f"{CFG}@s{k}@u{UPD}"
                need_macro = role == "test" and not done(f"no_temporal_s{k}", f"no_alignment_s{k}", f"check_nsd_s{k}")
                need_index = role == "test" and not done(*[f"{a}_s{k}" for a in ARMS])
                if not (need_macro or need_index):
                    continue
                net = S.load(CFG, k, UPD)
                z = net(graph)
                change = run.base.variant_inputs(z.cpu().numpy(), 8)
                j = pipeline.winners(new_pool, change, new_w)
                plan = prior.base.prior.plan(travel, q, j)
                full = score(z, plan)
                rec = compare(f"nsd_s{k}", row_of(j), full, sledger / (mtag + ".npz"))
                assert rec["winners_equal"] and rec["score_bitwise_equal"], rec
                base_row = row_of(j)
                if need_macro:
                    pre = []
                    at = len(q) // 2
                    t = int(q[at])
                    end = t + 16
                    small = prior.graph_inputs(role, sequence, end)[0]
                    short = net(small)
                    sb = run.base.variant_inputs(short.cpu().numpy(), 8)
                    winner = int(run.cal.candidates(raw[:end], (bank[0][:end], *sb[1:]), new_w[None], t)[0])
                    assert winner == int(j[at])
                    nt = score(z, center_only(plan))
                    save(f"no_temporal_s{k}", base_row, nt)
                    na_row = row_of(static_j)
                    na = score(z, prior.base.prior.plan(travel, q, static_j))
                    save(f"no_alignment_s{k}", na_row, na)
                    for cond, jj, tf, ref in (("no_temporal", winner, center_only, nt), ("no_alignment", int(static_j[at]), lambda p: p, na)):
                        pp = {kk: torch.from_numpy(v).cuda() for kk, v in tf(prior.base.prior.plan(travel[:end], np.array([t]), np.array([jj]))).items()}
                        chk = P.score(short, pp, lam).cpu().numpy()
                        pre.append(dict(condition=f"{cond}_s{k}", query=t, winner=jj, score_error=float(abs(chk - ref[at]).max())))
                        assert pre[-1]["score_error"] < 8e-6, pre[-1]
                    write(folder / f"prefix_s{k}.json", pre)
                    del small, short
                if need_index:
                    selected = {a: [] for a in ARMS}
                    for n, (t, group) in enumerate(zip(q, pools)):
                        all_ids = np.unique(np.concatenate(list(group.values())))
                        sa = run.base.align_scores(change[1], change[2], int(t), all_ids)
                        sc = run.base.column_align_scores(change[3], int(t), all_ids)
                        for a, ids in group.items():
                            loc = np.searchsorted(all_ids, ids)
                            stat = (new_w[0] * run.base.minmax(raw[ids] @ raw[t]) + new_w[1] * run.base.minmax(bank[0][ids] @ bank[0][t])).astype(np.float32)
                            value = stat + new_w[2] * run.base.minmax(sa[loc]) + new_w[3] * run.base.minmax(sc[loc])
                            selected[a].append(int(ids[np.argmax(value)]))
                    np.testing.assert_array_equal(np.asarray(selected["dual_100_100"]), j)
                    for a in ARMS:
                        ja = np.asarray(selected[a], np.int64)
                        row = dict(row_of(ja), candidate_count=np.asarray(sizes[a], np.int64), candidate_hit=np.asarray(hits[a], bool))
                        value = full if a == "dual_100_100" else score(z, prior.base.prior.plan(travel, q, ja))
                        positive = row["min_past_gt_dist"] <= 5
                        save(f"{a}_s{k}", row, value, dict(
                            candidate_recall=float(np.mean(row["candidate_hit"][positive])) if positive.any() else None,
                            candidate_mean=float(np.mean(row["candidate_count"]))))
                del net, z, change
                timing[f"s{k}"] = time.time() - st
        # --- 시간 창
        etas = []
        if "window_equal" in groups:
            etas += [(s, -0.5) for s in SIDES]
        if "window_selected" in groups:
            sel = read(OUT / "window_selection.json")["selected"]
            etas += [(s, sel[s]) for s in SIDES]
        if "window_val" in groups:
            etas += [(s, e) for s in SIDES for e in STRENGTHS]
        for side, eta in dict.fromkeys(etas):
            t_ = tag(side, eta)
            if done(t_):
                continue
            z = one_sided(prior, graph, side, eta)
            change = run.base.variant_inputs(z.cpu().numpy(), 8)
            j = pipeline.winners(new_pool, change, new_w)
            save(t_, row_of(j), score(z, prior.base.prior.plan(travel, q, j)))
            if role == "test" and t_ in ("past_0p5", "future_0p5", "past_0p125", "future_0p25"):
                jo = pipeline.winners(base_pool, change, P.CURRENT_W)
                rec = compare("old_" + t_, row_of(jo), score(z, prior.base.prior.plan(travel, q, jo), 0.25), WSIDE / "test_eval/test" / name / (t_ + ".npz"))
                assert rec["winners_equal"] and rec["score_max_error"] < 8e-6, rec
        if "window_val" in groups and not done("check_uniform"):
            z = prior.filtered(graph, -0.5)
            change = run.base.variant_inputs(z.cpu().numpy(), 8)
            j = pipeline.winners(new_pool, change, new_w)
            rec = compare("uniform", row_of(j), score(z, prior.base.prior.plan(travel, q, j)), sledger / "uniform.npz")
            assert rec["winners_equal"] and rec["score_bitwise_equal"], rec
    timing["total"] = time.time() - tick
    tpath = folder / "timing.json"
    old = read(tpath) if tpath.exists() else {}
    old[",".join(groups)] = timing
    write(tpath, old)
    print("WORKER", role, sequence, groups, round(timing["total"], 1), flush=True)


def sequences(role: str) -> list:
    spec = json.loads((ROOT / "results/pdv2_diffattn_contrast_test_20261104/protocol.json").read_text())
    return spec["val"] if role == "val" else spec["test"]


def campaign(role: str, groups: str, workers: int) -> None:
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tick = time.time()

    def one(s):
        with (logs / f"{role}_{s.replace('/', '_')}.log").open("a") as f:
            r = subprocess.run([sys.executable, __file__, "worker", "--role", role, "--sequence", s, "--groups", groups],
                               stdout=f, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError(s)
        return s

    with ThreadPoolExecutor(workers) as ex:
        for i, s in enumerate(ex.map(one, sequences(role)), 1):
            print("DONE", i, s, round(time.time() - tick, 1), flush=True)
    write(OUT / "logs" / f"campaign_{role}_{groups.replace(',', '+')}.done.json", dict(seconds=time.time() - tick, workers=workers))


def select() -> None:
    """검증9 파이프라인 집계 AP 최대 세기(1e-9 안 동률이면 약한 세기). _pdv2_window_side_control.val과 같은 규칙."""
    S = modules()
    run = S.run
    table = {}
    for t_ in ["uniform"] + [tag(s, e) for s in SIDES for e in STRENGTHS]:
        per = {}
        for s in sequences("val"):
            p = (S.LEDGERS / "val" / s.replace("/", "_") / "uniform.json") if t_ == "uniform" else (LED / "val" / s.replace("/", "_") / (t_ + ".json"))
            per[s] = read(p)["metrics"]
        table[t_] = run.base.macro(per)["macro"]
    ref = read(S.OUT / "agg.json")["val"]["models"]["uniform"]["macro"]
    for k in KEYS:
        assert abs(table["uniform"][k] - ref[k]) < 1e-12
    chosen = {side: sorted(STRENGTHS, key=lambda e: (-round(table[tag(side, e)]["AP"], 9), abs(e)))[0] for side in SIDES}
    write(OUT / "window_selection.json", dict(
        selected=chosen, rule="검증9 파이프라인 집계 AP 최대인 세기를 창마다 하나 고른다(1e-9 안 동률이면 약한 세기). 시험셋은 쓰지 않는다. _pdv2_window_side_control.val과 같은 규칙, 새 고정 설정(융합 (2,1,2,2), λ=0.25).",
        val_macro_percent={t_: {k: 100 * v[k] for k in KEYS} for t_, v in table.items()}, test_used=False))
    print("SELECTED", chosen, {t_: round(100 * v["AP"], 3) for t_, v in table.items()})


# ----------------------------------------------------------------- 집계(torch 불필요)
def finish() -> None:
    from _pr_eval_lib import classify_queries, pr_sweep
    S_OUT = ROOT / "results/pdv2_scratch_search_20261116"
    DIRS = ROOT / "results/pdv2_review_direction_yaw_20261105/directions"
    CITY = json.loads((ROOT / "results/pdv2_scratch_pilot_20261116/defaultbn_3seed/city.json").read_text())
    seqs = sequences("test")
    assert sorted(seqs) == sorted(s for v in CITY.values() for s in v)
    mtag = f"{CFG}@s{{k}}@u{UPD}"

    def path_of(cond: str, s: str) -> Path:
        name = s.replace("/", "_")
        if cond.startswith("diff_s"):
            return S_OUT / "ledgers/test" / name / (mtag.format(k=cond[6:]) + ".npz")
        if cond == "uniform":
            return S_OUT / "ledgers/test" / name / "uniform.npz"
        return LED / "test" / name / (cond + ".npz")

    def metric(score, top1, minimum, mask=None):  # _pdv2_scratch_search_compare_20261116.metric와 같다
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

    sel = read(OUT / "window_selection.json")["selected"] if (OUT / "window_selection.json").exists() else None
    learned = {"diff": [f"diff_s{k}" for k in SEEDS], "no_alignment": [f"no_alignment_s{k}" for k in SEEDS],
               "no_temporal": [f"no_temporal_s{k}" for k in SEEDS]}
    learned.update({a: [f"{a}_s{k}" for k in SEEDS] for a in ARMS})
    free = ["uniform", "identity", "zl_identity", "past_0p5", "future_0p5"]
    if sel:
        free += [tag(s, sel[s]) for s in SIDES]
    free = list(dict.fromkeys(free))
    conds = [c for v in learned.values() for c in v] + free
    rows, cand, ledger_count, maxerr = {}, {}, {}, 0.0
    labels = {}
    for s in seqs:
        name = s.replace("/", "_")
        with np.load(DIRS / (name + ".npz")) as d:
            labels[s] = (d["query_idx"].copy(), d["group"].copy())
    for c in conds:
        rows[c], cand[c] = {}, {}
        for s in seqs:
            p = path_of(c, s)
            with np.load(p) as z:
                d = {k: z[k] for k in z.files}
            qi, grp = labels[s]
            np.testing.assert_array_equal(d["query_idx"], qi)
            masks = dict(all=None, forward_only=grp == 0, reverse_only=grp == 1)
            rows[c][s] = {g: metric(d["score"], d["top1_geo_dist"], d["min_past_gt_dist"], m) for g, m in masks.items()}
            if "candidate_hit" in d:
                pos = d["min_past_gt_dist"] <= 5
                cand[c][s] = (float(np.mean(d["candidate_hit"][pos])) if pos.any() else None, float(np.mean(d["candidate_count"])), len(d["query_idx"]))
            # 독립 재계산: 평가 모듈과 다른 pr_sweep 경로가 워커의 지표와 같은지
            js = p.with_suffix(".json")
            if js.exists() and rows[c][s]["all"] is not None:
                ref = read(js)["metrics"]
                maxerr = max(maxerr, max(abs(rows[c][s]["all"][k] - ref[k]) for k in KEYS))
            ledger_count[c] = ledger_count.get(c, 0) + 1
    assert maxerr < 1e-9, maxerr

    def city_macro(c: str, g: str, key: str) -> float:
        vals = []
        for seqs_ in CITY.values():
            v = [rows[c][s][g][key] for s in seqs_ if rows[c][s][g] is not None]
            if v:
                vals.append(np.mean(v))
        return float(np.mean(vals))

    def city_groups(c: str, g: str = "all") -> dict:
        return {city: {k: float(np.mean([rows[c][s][g][k] for s in seqs_ if rows[c][s][g] is not None])) for k in KEYS}
                for city, seqs_ in CITY.items()}

    def per_case(c: str) -> dict:
        return dict(macro={k: city_macro(c, "all", k) for k in KEYS}, groups=city_groups(c),
                    sequences={s: rows[c][s]["all"] for s in seqs})

    def kind_tags(kind: str) -> list:
        return learned.get(kind, [kind])

    def mean_sd(kind: str, g: str = "all") -> tuple:
        tags = kind_tags(kind)
        vals = {k: [city_macro(t_, g, k) for t_ in tags] for k in KEYS}
        return ({k: float(np.mean(v)) for k, v in vals.items()},
                {k: (float(np.std(v, ddof=1)) if len(v) > 1 else None) for k, v in vals.items()},
                {k: v for k, v in vals.items()})

    def paired(a: str, b: str, g: str, seed: int, extra: dict = None) -> dict:
        """도시 안 시퀀스 재표집 10,000회. 시드가 있는 쪽은 시퀀스별 시드 평균, 도시 균등(_pdv2_scratch_search_compare 규약)."""
        ta, tb = kind_tags(a), kind_tags(b)
        keys = list(KEYS) + (list(extra) if extra else [])
        rng = np.random.default_rng(seed)
        draws, city_delta = [], {}
        for city, seqs_ in CITY.items():
            avail = [s for s in seqs_ if rows[ta[0]][s][g] is not None and rows[tb[0]][s][g] is not None]
            if not avail:
                continue
            mat = []
            for s in avail:
                r = [100 * (np.mean([rows[x][s][g][k] for x in ta]) - np.mean([rows[x][s][g][k] for x in tb])) for k in KEYS]
                if extra:
                    r += [100 * (np.mean([extra[k][x][s] for x in ta]) - np.mean([extra[k][x][s] for x in tb])) for k in extra]
                mat.append(r)
            mat = np.array(mat)
            city_delta[city] = dict(zip(keys, mat.mean(0).tolist()))
            idx = rng.integers(0, len(avail), size=(10000, len(avail)))
            draws.append(mat[idx].mean(1))
        boot = np.stack(draws).mean(0)
        point = {k: float(np.mean([city_delta[c][k] for c in city_delta])) for k in keys}
        return dict(delta_pp=point, city_delta_pp=city_delta,
                    conditional_95_pp={k: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(keys)})

    def directions(kinds: list) -> dict:
        out = {}
        for g in GROUPS:
            out[g] = {}
            for kind in kinds:
                m, sd, v = mean_sd(kind, g)
                out[g][kind] = dict(mean=m, sd=sd, per_seed=v)
        return out

    model = dict(cfg=CFG, seeds=list(SEEDS), updates=UPD, fusion=[2.0, 1.0, 2.0, 2.0], lam=0.25,
                 checkpoints=f"results/pdv2_scratch_search_20261116/runs/{CFG}/s{{4,5,6}}/u512.pth")
    common = dict(model=model, queries_per_case=int(sum(cand.get("dual_100_100_s4", {}).get(s, (0, 0, 0))[2] for s in seqs)),
                  aggregation="양성 있는 45시퀀스 -> 6도시 균등 -> 시드 4·5·6 평균(SD ddof=1). 학습 없는 조건은 1회.",
                  bootstrap=dict(replicates=10000, unit="도시 안 시퀀스 복원추출, 시퀀스별 시드 평균 차이, 도시 균등",
                                 note="_pdv2_scratch_search_compare_20261116.py와 같은 벡터화 구현"),
                  test_used_for_selection=False, independent_metric_max_error=maxerr)
    repro = {}
    for s in seqs:
        f = LED / "test" / s.replace("/", "_")
        for p in f.glob("check_*.json"):
            r = read(p)
            key = p.stem[6:]
            e = repro.setdefault(key, dict(sequences=0, winners_equal=True, score_bitwise_equal=True, score_max_error=0.0))
            e["sequences"] += 1
            e["winners_equal"] &= r["winners_equal"]
            e["score_bitwise_equal"] &= r["score_bitwise_equal"]
            e["score_max_error"] = max(e["score_max_error"], r["score_max_error"])
    prefix = [x for s in seqs for k in SEEDS for x in read(LED / "test" / s.replace("/", "_") / f"prefix_s{k}.json")]
    # --- 거시 대조
    means, stds = {}, {}
    for kind in ("diff", "no_alignment", "no_temporal"):
        means[kind], stds[kind], _ = mean_sd(kind)
    base = read(S_OUT / "agg.json")["test"]["models"]  # 기준: S 집계의 시드 4~6 평균(그룹 항목은 시드 1~6을 함께 묶는다)
    for k in KEYS:
        ref = float(np.mean([base[mtag.format(k=sd)]["macro"][k] for sd in SEEDS]))
        assert abs(means["diff"][k] - ref) < 1e-9, (k, means["diff"][k], ref)
    r1_equal = all(np.array_equal(np.load(path_of(f"no_temporal_s{k}", s))["j_star"], np.load(path_of(f"diff_s{k}", s))["j_star"]) for s in seqs for k in SEEDS)
    macro_cmp = {f"diff-{c}": paired("diff", c, "all", 20261104) for c in ("no_alignment", "no_temporal")}
    macro_out = dict(status="passed", **common, prefix_checks=len(prefix), prefix_max_score_error=max(x["score_error"] for x in prefix),
                     no_temporal_winners_equal_nsd=r1_equal, mean=means, std=stds,
                     per_case={t_: per_case(t_) for kind in ("diff", "no_alignment", "no_temporal") for t_ in learned[kind]},
                     comparisons=macro_cmp, directions=directions(["diff", "no_alignment", "no_temporal"]),
                     direction_comparisons={g: {f"diff-{c}": paired("diff", c, g, 20261104) for c in ("no_alignment", "no_temporal")} for g in GROUPS},
                     reproduction={k: v for k, v in repro.items() if k.startswith("nsd_s")},
                     definitions=dict(no_alignment="반환 = 후보 합집합에서 새 융합 정적 부분합(2·minmax 원키 + 1·minmax BEV 크기) 최대. 수락 = 원래 시간 증거 식.",
                                      no_temporal="반환 = NSD 승자. 수락 = 같은 점수식에서 중심 쌍 가중치 1, 시간 이웃 0."))
    write(OUT / "macro_summary.json", macro_out)
    # --- 색인 예산
    recall = {"candidate_recall": {t_: {s: cand[t_][s][0] for s in seqs} for a in ARMS for t_ in learned[a]}}
    imeans, istds, ipc = {}, {}, {}
    for a in ARMS:
        imeans[a], istds[a], _ = mean_sd(a)
        crs = []
        for t_ in learned[a]:
            cm = float(np.mean([np.mean([cand[t_][s][0] for s in seqs_ if cand[t_][s][0] is not None]) for seqs_ in CITY.values()]))
            crs.append(cm)
            pc = per_case(t_)
            pc["macro"]["candidate_recall"] = cm
            ipc[t_] = pc
        imeans[a]["candidate_recall"] = float(np.mean(crs))
        istds[a]["candidate_recall"] = float(np.std(crs, ddof=1))
    mean_cand = {a: float(sum(cand[f"{a}_s4"][s][1] * cand[f"{a}_s4"][s][2] for s in seqs) / sum(cand[f"{a}_s4"][s][2] for s in seqs)) for a in ARMS}
    icmp = {f"dual-minus-{a}": paired("dual_100_100", a, "all", 20261106, recall) for a in ARMS if a != "dual_100_100"}
    for k in KEYS:
        assert abs(imeans["dual_100_100"][k] - means["diff"][k]) < 1e-12
    old_index = read(ROOT / "results/pdv2_contrast_index_budget_20261106/summary.json")
    index_out = dict(status="passed", **common, per_case=ipc, mean=imeans, std=istds, mean_candidates=mean_cand, comparisons=icmp,
                     directions=directions(list(ARMS)),
                     direction_comparisons={g: {f"dual-minus-{a}": paired("dual_100_100", a, g, 20261106) for a in ARMS if a != "dual_100_100"} for g in GROUPS},
                     original_main_metric_difference={k: imeans["dual_100_100"][k] - means["diff"][k] for k in KEYS},
                     candidate_sets_model_independent=True,
                     candidate_recall_vs_previous_model={a: imeans[a]["candidate_recall"] - old_index["mean"][a]["candidate_recall"] for a in ARMS},
                     mean_candidates_vs_previous_model={a: mean_cand[a] - old_index["mean_candidates"][a] for a in ARMS},
                     checks=dict(ledgers=sum(ledger_count[t_] for a in ARMS for t_ in learned[a]), queries=common["queries_per_case"],
                                 dual_winners_equal_nsd=repro.get("nsd_s4", {}).get("winners_equal"),
                                 maximum_metric_error=maxerr),
                     note="승자 선택의 정적 부분합은 pipeline.winners처럼 float32로 바꾼 뒤 정렬 채널을 더한다(옛 색인 예산 스크립트는 변환 없이 한 식).")
    write(OUT / "index_budget_summary.json", index_out)
    write(OUT / "index_budget_integrated_validation.json", dict(
        status="passed", ledgers=index_out["checks"]["ledgers"], queries=common["queries_per_case"],
        positive_queries=int(sum(int((np.load(path_of("dual_100_100_s4", s))["min_past_gt_dist"] <= 5).sum()) for s in seqs)),
        maximum_metric_error=maxerr, exact_matched_counts=True, past_only_candidates=True,
        original_winners_exact=all(repro[f"nsd_s{k}"]["winners_equal"] and repro[f"nsd_s{k}"]["score_bitwise_equal"] for k in SEEDS),
        summary_sha256=__import__("hashlib").sha256((OUT / "index_budget_summary.json").read_bytes()).hexdigest()))
    # --- 정제 없음
    nmeans = {"diff": means["diff"], "identity": mean_sd("identity")[0], "uniform": mean_sd("uniform")[0]}
    ncmp = {}
    for ref in ("identity", "uniform"):
        ncmp[f"diff-{ref}"] = {}
        for g in GROUPS:
            r = paired("diff", ref, g, 20261104)
            ncmp[f"diff-{ref}"][g] = dict(overall={k: dict(delta_pp=r["delta_pp"][k], ci95_pp=r["conditional_95_pp"][k]) for k in KEYS},
                                          city_delta_pp=r["city_delta_pp"])
    no_ref = dict(status="passed", **common, mean=nmeans, std={"diff": stds["diff"], "identity": dict.fromkeys(KEYS), "uniform": dict.fromkeys(KEYS)},
                  per_case={"identity": per_case("identity"), "uniform": per_case("uniform"), **{t_: per_case(t_) for t_ in learned["diff"]}},
                  comparisons=ncmp, directions=directions(["diff", "identity", "uniform"]),
                  reproduction={k: v for k, v in repro.items() if k in ("uniform", "old_identity")},
                  bev_reference_max_F1=0.88995, definitions=dict(identity="z = 원래 복소 DFT 특징(graph.spectrum), 정렬 뱅크 = 입력 뱅크, 새 융합·λ=0.25, 양방향 수락."))
    write(OUT / "no_refining_summary.json", no_ref)
    # --- 반환 오류(크기만 vs NSD)
    def classify(d):
        return np.where(d <= 5, 0, np.where(d < 25, 1, 2))
    per_seq = {k: {} for k in SEEDS}
    for s in seqs:
        for k in SEEDS:
            mag, nsd = np.load(path_of(f"no_alignment_s{k}", s)), np.load(path_of(f"diff_s{k}", s))
            np.testing.assert_array_equal(mag["query_idx"], nsd["query_idx"])
            pos = nsd["min_past_gt_dist"] <= 5
            t3 = np.zeros((3, 3), np.int64)
            np.add.at(t3, (classify(mag["top1_geo_dist"][pos]), classify(nsd["top1_geo_dist"][pos])), 1)
            per_seq[k][s] = t3

    def rates(t3):
        n = t3.sum()
        return dict(magnitude_error=t3[1:].sum() / n, magnitude_distant=t3[2].sum() / n, nsd_error=t3[:, 1:].sum() / n, nsd_distant=t3[:, 2].sum() / n)
    alias = {}
    for city, seqs_ in CITY.items():  # _pdv2_aliasing_decomposition.city_rates와 같은 집계
        alias[city] = {key: 100 * float(np.mean([np.mean([rates(per_seq[k][s])[key] for s in seqs_ if per_seq[k][s].sum()]) for k in SEEDS]))
                       for key in ("magnitude_error", "magnitude_distant", "nsd_error", "nsd_distant")}
    alias["macro"] = {key: float(np.mean([alias[c][key] for c in CITY])) for key in alias["karlsruhe"]}
    assert abs(alias["macro"]["magnitude_error"] - 100 * (1 - means["no_alignment"]["R@1"])) < 1e-9
    assert abs(alias["macro"]["nsd_error"] - 100 * (1 - means["diff"]["R@1"])) < 1e-9
    pooled = {k: sum(per_seq[k].values()) for k in SEEDS}
    same_mag = all(np.array_equal(per_seq[SEEDS[0]][s].sum(1), per_seq[k][s].sum(1)) for s in seqs for k in SEEDS)
    def summary(t3):  # _pdv2_aliasing_decomposition.main의 summary와 같다
        n, mag_err, nsd_err = t3.sum(), t3[1:].sum(), t3[:, 1:].sum()
        return dict(positives=int(n), magnitude_error_rate=float(mag_err / n), magnitude_distant_share_of_errors=float(t3[2].sum() / max(mag_err, 1)),
                    magnitude_distant_rate=float(t3[2].sum() / n), nsd_error_rate=float(nsd_err / n),
                    nsd_distant_share_of_errors=float(t3[:, 2].sum() / max(nsd_err, 1)), nsd_distant_rate=float(t3[:, 2].sum() / n),
                    distant_fixed_by_nsd=float(t3[2, 0] / max(t3[2].sum(), 1)), near_fixed_by_nsd=float(t3[1, 0] / max(t3[1].sum(), 1)),
                    correct_broken_by_nsd=float(t3[0, 1:].sum() / max(t3[0].sum(), 1)), transition=t3.tolist())
    starts = {str(k): summary(pooled[k]) for k in SEEDS}
    write(OUT / "aliasing_summary.json", dict(
        status="passed", **{k_: v for k_, v in common.items() if k_ != "bootstrap"}, city_rates_percent=alias,
        per_start=starts,
        pooled_mean_over_starts={key: float(np.mean([starts[str(k)][key] for k in SEEDS])) for key, v in starts[str(SEEDS[0])].items() if isinstance(v, float)},
        cities={city: summary(sum(per_seq[k][s] for s in seqs_ for k in SEEDS)) for city, seqs_ in CITY.items()},
        magnitude_returns_identical_across_seeds=same_mag,
        definition="_pdv2_aliasing_decomposition.city_rates와 같다. 크기만 = no_alignment 원장(새 융합), NSD = 시드 4~6 원장. 오류 d>5 m, 먼 오류 d>=25 m, 양성 질의만."))
    # --- 시간 창(학습 없는 조건)
    arms = dict(symmetric="uniform", past_same="past_0p5", future_same="future_0p5", no_refining="identity", no_refining_past5_acceptance="zl_identity")
    if sel:
        arms.update(past_selected=tag("past", sel["past"]), future_selected=tag("future", sel["future"]))
    wout = dict(status="passed", **common, selection=read(OUT / "window_selection.json") if sel else None, arms=arms, groups={})
    for g in GROUPS:
        block = dict(percent={a: {k: 100 * city_macro(t_, g, k) for k in KEYS} for a, t_ in arms.items()}, minus_symmetric={},
                     nsd_minus={}, nsd_percent={k: 100 * mean_sd("diff", g)[0][k] for k in KEYS})
        for a, t_ in arms.items():
            if a != "symmetric":
                r = paired(t_, "uniform", g, 20261111)
                block["minus_symmetric"][a] = {k: dict(mean_pp=r["delta_pp"][k], ci95_pp=r["conditional_95_pp"][k]) for k in KEYS}
            r = paired("diff", t_, g, 20261111)
            block["nsd_minus"][a] = {k: dict(mean_pp=r["delta_pp"][k], ci95_pp=r["conditional_95_pp"][k]) for k in KEYS}
        wout["groups"][g] = block
    wout["reproduction"] = {k: v for k, v in repro.items() if k.startswith("old_") or k == "uniform"}
    write(OUT / "window_side_summary.json", wout)
    # --- 요약 출력
    pct = lambda d: {k: round(100 * d[k], 3) for k in d}  # noqa: E731
    print("REPRO", json.dumps(repro))
    print("MACRO", {k: pct(v) for k, v in means.items()}, {k: pct(v) for k, v in stds.items()})
    for c, v in macro_cmp.items():
        print(" ", c, {k: (round(v["delta_pp"][k], 3), [round(x, 3) for x in v["conditional_95_pp"][k]]) for k in KEYS})
    print("INDEX", {a: pct(v) for a, v in imeans.items()}, mean_cand)
    for c, v in icmp.items():
        print(" ", c, {k: (round(v["delta_pp"][k], 3), [round(x, 3) for x in v["conditional_95_pp"][k]]) for k in v["delta_pp"]})
    print("NOREF", {k: pct(v) for k, v in nmeans.items()})
    for g in GROUPS:
        print(" ", g, {k: (round(v["delta_pp"], 3), [round(x, 3) for x in v["ci95_pp"]]) for k, v in ncmp["diff-identity"][g]["overall"].items()})
    print("ALIAS", alias["macro"])
    for g in GROUPS:
        print("WINDOW", g, {a: round(v["AP"], 3) for a, v in wout["groups"][g]["percent"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("worker", "campaign", "select", "finish"))
    ap.add_argument("--role", default="test")
    ap.add_argument("--sequence")
    ap.add_argument("--groups", default="core,window_equal")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.action == "worker":
        S = modules()
        S.P.e.configure()
        import torch
        torch.set_num_threads(4)
        worker(a.role, a.sequence, a.groups.split(","))
    elif a.action == "campaign":
        campaign(a.role, a.groups, a.workers)
    elif a.action == "select":
        select()
    else:
        finish()
