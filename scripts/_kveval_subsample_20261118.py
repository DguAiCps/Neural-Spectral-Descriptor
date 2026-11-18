#!/usr/bin/env python3
"""B7 키프레임 하위 표집 평가(학습 없음): 차이 K/V(NSD) 대 절대 K/V(abskv).

평가 하네스 scripts/_kveval_perturb_20261118.py(이하 H)를 import만 하고(파일은 고치지 않음) 그 evaluate와 같은
경로(C.pipe 대칭 창)를 메모리에서 다시 만든 입력으로 돈다. 디스크에 쓰는 것은 원장·진단 json뿐이다.

-------------------------------------------------------------------- 조건(결과 보기 전에 고정)
  b7_even  키프레임 색인 0, 2, 4, ...만 남긴다(2개 중 1개, 짝 실현).
  b7_odd   키프레임 색인 1, 3, 5, ...만 남긴다(홀 실현).
  b7_s1    검산용(stride 1, 하위 표집 없음): perturb/base 원장과 비트 일치해야 한다. campaign에 쓰지 않는다.
규약: 질의 arange(30, n-15), 후보 제외 창 q-29, 꼬리 15는 **새 색인 기준 그대로**(방법을 낮은 프레임률 입력에
그대로 적용하는 상황). 그래프 ±5 시간 창도 새 색인 ±5(실제 주행 거리로 약 두 배). 양성(<=5 m)·음성(>25 m) 판정
반경, R@1 반경 5 m는 그대로. 정답 위치는 poses_gt 참값.

-------------------------------------------------------------------- 메모리에서 다시 만드는 것
  cache   : 7개 배열(descriptors, poses, timestamps, scan_ids, keyframe_ids, spectral_entropies, poses_gt)의 부분집합.
  spectrum: testspec(sk) 계수 앞 8주파수 부분집합. bank = variant_inputs(spectrum, 8), nb·nc는 _pdv2_diffattn_comparable_test
            .inputs와 같은 식(nb·nc는 평가에 쓰이지 않지만 튜플 일관성을 위해 다시 계산).
  travel  : 부분집합 poses(KISS-ICP)로 다시 누적(H.travel_of).
  minimum : min_past_gt_dist[q] = min ||pos[:q-29] - pos[q]||(새 색인), q in arange(30, n-15). 나머지 nan.
  graph   : run.make_graph(x, poses, 0) + attach_transport(spectrum, lidar_yaw(poses, None), 0)
            (prior.graph_inputs의 시험 경로와 같은 식. graph_inputs는 스펙트럼을 디스크에서 전체 길이로 읽으므로 쓰지 않음).
  pool    : pipeline.candidate_pool(raw_key(descriptors), bank, weights)로 메모리에서(ids는 원키·bank[0] 상위 100 합집합,
            weights와 무관). static은 H.evaluate와 같게 P.selected() 가중으로 다시 계산.

실행(컨테이너 안):
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_kveval_subsample_20261118.py \
      eval --cond b7_s1 --sequence kitti/07 --models main --root results/kv_eval_conditions_20261118/tmp/w4_b7check
  ... campaign --cond b7_even --models main --workers 2
  ... summary          (perturb/summary3.json; B6 조건이 있으면 함께 집계)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import _kveval_perturb_20261118 as H  # noqa: E402

OUT = H.OUT
ROOT = H.ROOT
SUB = {"b7_even": (2, 0), "b7_odd": (2, 1), "b7_s1": (1, 0)}
DESC = {"b7_even": "B7 키프레임 하위 표집 2개 중 1개(색인 짝수), 제외 창·꼬리·그래프 창은 새 색인 기준(30, 15, ±5)",
        "b7_odd": "B7 키프레임 하위 표집 2개 중 1개(색인 홀수), 제외 창·꼬리·그래프 창은 새 색인 기준(30, 15, ±5)",
        "b7_s1": "B7 검산: stride 1(하위 표집 없음), 전용 평가 경로"}
CACHE_KEYS = ("descriptors", "poses", "timestamps", "scan_ids", "keyframe_ids", "spectral_entropies", "poses_gt")


# ----------------------------------------------------------------- 입력·그래프·후보 풀(메모리)
def min_past(pos: np.ndarray) -> np.ndarray:
    n = len(pos)
    out = np.full(n, np.nan)
    for q in range(30, n - 15):
        out[q] = np.linalg.norm(pos[:q - 29] - pos[q], axis=1).min()
    return out


def build_tuple(cache: dict, spectrum: np.ndarray, paths: tuple) -> tuple:
    """P.inputs와 같은 8-튜플을 주어진 cache·스펙트럼(N,16,8 complex64)으로 만든다(minimum은 새 색인 식)."""
    run = H.ctx()["run"]
    for k in CACHE_KEYS:
        assert len(cache[k]) == len(spectrum), k
    wx = run.white.transform(spectrum, run.ARR(run.white.OUT / "statistics.npz")["full"])
    bank = run.base.variant_inputs(spectrum, 8)
    nb, nc = [run.base.variant_inputs(run.power.transform(v, .5), 8) for v in (spectrum, wx)]
    pos = cache["poses_gt"][:, :3, 3]
    return cache, pos, min_past(pos), H.travel_of(cache["poses"]), bank, nb, nc, paths


def _eq_sub(a, b, idx) -> bool:
    """a == b[idx]를 (중첩 튜플까지) 비트 단위로 검사한다."""
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_eq_sub(x, y, idx) for x, y in zip(a, b))
    return bool(np.array_equal(a, np.asarray(b)[idx]))


def subsample_tuple(seq: str, stride: int, phase: int) -> tuple:
    c = H.ctx()
    P, run = c["P"], c["run"]
    tup0 = P.inputs("test", seq)
    cache0, paths = tup0[0], tup0[7]
    n0 = len(cache0["poses"])
    idx = np.arange(phase, n0, stride)
    assert set(CACHE_KEYS) <= set(cache0), sorted(cache0)
    cache = {k: np.asarray(cache0[k])[idx] for k in CACHE_KEYS}
    for k, v in cache0.items():  # 그 밖의 항목은 길이가 n0이면 같이 자르고 아니면 그대로
        if k not in cache:
            cache[k] = np.asarray(v)[idx] if hasattr(v, "__len__") and len(v) == n0 else v
    spec = run.white.coefficients(paths[1])
    assert len(spec) == n0
    tup = build_tuple(cache, spec[idx], paths)
    d = dict(n_orig=n0, n=len(idx), stride=stride, phase=phase,
             bank_equals_subset=_eq_sub(tup[4], tup0[4], idx),
             travel_total_orig=float(tup0[3][-1]), travel_total=float(tup[3][-1]),
             step_m_median=float(np.median(np.diff(tup[3]))) if len(idx) > 1 else None,
             step_m_median_orig=float(np.median(np.diff(tup0[3]))))
    if stride == 1:
        q = np.arange(30, n0 - 15)
        d["minimum_bitwise_equal_orig"] = bool(np.array_equal(tup[2][q], tup0[2][q]))
        d["minimum_max_abs_err_orig"] = float(np.nanmax(np.abs(tup[2][q] - tup0[2][q])))
        d["travel_bitwise_equal_orig"] = bool(np.array_equal(tup[3], tup0[3]))
    return tup, spec[idx], d


def build_graph(tup: tuple, spectrum: np.ndarray):
    """prior.graph_inputs의 시험 경로와 같은 식(외부 보정 없음). spectrum = (N,16,8) complex64."""
    prior = H.ctx()["prior"]
    run = H.ctx()["run"]
    cache = tup[0]
    x, poses = cache["descriptors"], cache["poses"]
    yaw = prior.lidar_yaw(poses, None)
    graph = run.make_graph(x, poses, np.zeros(len(x), np.int64)).cuda()
    prior.attach_transport(graph, spectrum, yaw, np.zeros(len(x), np.int64))
    return graph


def build_pool(raw: np.ndarray, bank: tuple) -> dict:
    """후보 풀 ids는 pipeline.candidate_pool로, static은 H.evaluate와 같은 P.selected() 가중으로."""
    c = H.ctx()
    P, run, pipeline = c["P"], c["run"], c["pipeline"]
    weights, _ = P.selected()
    pool = pipeline.candidate_pool(raw, bank, weights)
    static = []
    for n, q in enumerate(pool["query"]):
        lo, hi = pool["offset"][n:n + 2]
        ids = pool["ids"][lo:hi]
        static.append(weights[0] * run.base.minmax(raw[ids] @ raw[q]) + weights[1] * run.base.minmax(bank[0][ids] @ bank[0][q]))
    return dict(query=pool["query"], offset=pool["offset"], ids=pool["ids"], static=np.concatenate(static).astype(np.float32))


def disk_pool_check(seq: str, pool: dict) -> dict:
    c = H.ctx()
    run = c["run"]
    pp = c["C"].P.TEST / "test" / seq.replace("/", "_") / "candidate_pool.npz"
    base = run.ARR(pp)
    return {k: bool(np.array_equal(pool[k], base[k])) for k in ("query", "offset", "ids")}


# ----------------------------------------------------------------- 평가(H.evaluate의 모델 고리와 같다)
def evaluate_tuple(cond: str, sequence: str, models: list, root: Path, tup: tuple, graph, pool: dict, diag: dict,
                   prep: float, check_base: bool = False) -> None:
    c = H.ctx()
    torch, C, P, run, pipeline = c["torch"], c["C"], c["P"], c["run"], c["pipeline"]
    name = sequence.replace("/", "_")
    folder = root / cond / "ledgers" / "test" / name
    folder.mkdir(parents=True, exist_ok=True)
    _, pos, minimum, travel = tup[:4]
    weights, lam = P.selected()
    P.write(root / cond / "inputs" / "test" / (name + ".json"), dict(diag=diag, prep_seconds=prep))
    for m in models:
        if (folder / (m + ".json")).exists():
            continue
        t0 = time.time()
        net = H.load_model(m)
        with torch.no_grad():
            z = net(graph)
            change = run.base.variant_inputs(z.cpu().numpy(), 8)
            q = pool["query"]
            j = pipeline.winners(pool, change, weights)
            assert (j <= q - 30).all()
            row = dict(query_idx=q, j_star=j, min_past_gt_dist=minimum[q], top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1))
            plan = C.ORIGINAL["plan"](travel, q, j)
            value = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(q)).astype(np.float64)
        metric = run.base.independent_metrics(row, value)
        rec = dict(model=m, cond=cond, metrics=metric, seconds=time.time() - t0, prep_seconds=prep, n_queries=int(len(q)),
                   checkpoint=str(H.model_source(m)["ckpt"].relative_to(ROOT)), gpu=torch.cuda.get_device_name())
        if check_base:
            bp = OUT / "base/ledgers/test" / name / (m + ".npz")
            with np.load(bp) as r:
                rec["check_vs_perturb_base"] = dict(
                    reference=str(bp.relative_to(ROOT)),
                    **{f"{k}_bitwise_equal": bool(np.array_equal(row[k], r[k])) for k in H.FIELDS},
                    score_bitwise_equal=bool(np.array_equal(value, r["score"])),
                    score_max_error=float(np.abs(value - r["score"]).max()) if len(value) == len(r["score"]) else None)
        np.savez_compressed(folder / (m + ".npz"), **row, score=value)
        P.write(folder / (m + ".json"), rec)
        del net, z, change


def evaluate(cond: str, sequence: str, models: list, root: Path = OUT) -> None:
    c = H.ctx()
    pipeline = c["pipeline"]
    name = sequence.replace("/", "_")
    folder = root / cond / "ledgers" / "test" / name
    if all((folder / (m + ".json")).exists() for m in models):
        return
    tick = time.time()
    stride, phase = SUB[cond]
    tup, spectrum, diag = subsample_tuple(sequence, stride, phase)
    graph = build_graph(tup, spectrum)
    raw = pipeline.raw_key(tup[0]["descriptors"])
    pool = build_pool(raw, tup[4])
    diag["n_queries"] = int(len(pool["query"]))
    diag["n_pos"] = int(np.sum(tup[2][pool["query"]] <= 5))
    diag["n_neg"] = int(np.sum(tup[2][pool["query"]] > 25))
    if stride == 1:
        diag["pool_equal_disk"] = disk_pool_check(sequence, pool)
        g0, _, _ = c["prior"].graph_inputs("test", sequence)
        torch = c["torch"]
        diag["graph_equal_graph_inputs"] = {a: bool(torch.equal(getattr(graph, a), getattr(g0, a)))
                                            for a in ("x", "edge_index", "edge_attr", "edge_type", "transport_indices",
                                                      "transport_mask", "transport_phase", "spectrum")}
    prep = time.time() - tick
    evaluate_tuple(cond, sequence, models, root, tup, graph, pool, diag, prep, check_base=(stride == 1))
    print("EVAL", cond, sequence, round(time.time() - tick, 1), flush=True)


def campaign(cond: str, group: str, workers: int, script: str = __file__, desc: dict = None, params: dict = None) -> None:
    import concurrent.futures
    assert workers <= 2
    c = H.ctx()
    seqs = c["P"].TESTSEQ
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    c["P"].write(OUT / cond / "condition.json", dict(name=cond, desc=(desc or DESC)[cond], params=params or dict(
        family="B7", stride=SUB[cond][0], phase=SUB[cond][1], exclusion="new index (30, 15)", graph_window="new index ±5")))
    tick = time.time()

    def call(s):
        with (logs / f"{cond}_{s.replace('/', '_')}.log").open("a") as f:
            subprocess.run([sys.executable, script, "eval", "--cond", cond, "--sequence", s, "--models", group],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(call, s): s for s in seqs}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            fut.result()
            print("DONE", cond, group, i, len(seqs), futs[fut], round(time.time() - tick, 1), flush=True)
    c["P"].write(logs / f"campaign_{cond}_{group}.done.json", dict(seconds=time.time() - tick, workers=workers,
                                                                    models=H.model_names(group)))


# ----------------------------------------------------------------- 집계(H.summary 규약, perturb/summary3.json)
def summary() -> None:
    import _kv_subset_analysis_20261117 as K
    import _report_model_common_20261116 as RC

    keys = K.KEYS
    arms = {"main": {"nsd": [f"nsd@s{k}" for k in (4, 5, 6)], "abs": [f"abskv@s{k}" for k in (4, 5, 6)]},
            "ref": {"nsd": [f"nsd@s{k}" for k in (1, 2, 3)], "abs": [f"abskv@s{k}" for k in (1, 2, 3)]}}

    def ledger(cond, m, s):
        p = OUT / cond / "ledgers/test" / s.replace("/", "_") / (m + ".npz")
        if not p.exists():
            return None
        with np.load(p) as z:
            return dict(score=z["score"].astype(np.float64), top1=z["top1_geo_dist"], gt=z["min_past_gt_dist"], q=z["query_idx"], j=z["j_star"])

    def rows(cond, tags):
        out = []
        for t in tags:
            r = {}
            for s in RC.SEQS:
                d = ledger(cond, t, s)
                if d is None:
                    return None
                r[s] = K.metric(d)
            out.append(r)
        return out

    order = ["b7_even", "b7_odd", "b6_drop50", "b6_range50", "b6_beam2", "b6_occ90"]
    conds = [c for c in order if (OUT / c / "ledgers").exists()]
    res = dict(conventions=dict(
        aggregation="시퀀스 지표(_pr_eval_lib.pr_sweep, R@1=양성 top1<=5 m) -> 도시 안 평균 -> 6도시 동일 가중 -> 시드 평균 (양성 없는 시퀀스 제외)",
        interval="NSD-abs: 시퀀스별 (NSD 시드 평균 - abs 시드 평균), 도시 안 시퀀스 복원추출 10,000회, rng=default_rng(20261104) 비교마다 새로",
        D="D = (NSD-abs)_조건 - (NSD-abs)_base(로컬 재실행), 같은 시퀀스 -> boot_joint(도시 안 시퀀스 색인 공유), rng=default_rng(20261117). "
          "B7은 질의 집합이 기본과 다르지만 시퀀스 단위 지표의 차이로 같은 방식",
        units="%p", baseline="base = H로 로컬 RTX 5080에서 다시 돌린 교란 없음 조건(perturb/base)",
        source="scripts/_kveval_subsample_20261118.py (B7), scripts/_kveval_pointperturb_20261118.py (B6)"), conditions={})
    base_rows = {g: {a: rows("base", t) for a, t in arms[g].items()} for g in arms}
    for cond in conds:
        cj = OUT / cond / "condition.json"
        meta = json.loads(cj.read_text()) if cj.exists() else {}
        rec = dict(desc=meta.get("desc"), params=meta.get("params"))
        for g, ga in arms.items():
            rr = {a: rows(cond, t) for a, t in ga.items()}
            if any(v is None for v in rr.values()):
                continue
            per = {a: [K.macro(r, RC.CITY) for r in rs] for a, rs in rr.items()}
            mean = {a: {k: 100 * float(np.mean([p[k] for p in ps])) for k in keys} for a, ps in per.items()}
            per_model = {t: {k: 100 * p[k] for k in keys} for a in ga for t, p in zip(ga[a], per[a])}
            point, draws = K.boot_single(rr["nsd"], rr["abs"], RC.CITY, np.random.default_rng(RC.BOOT_SEED))
            out = dict(mean_pp=mean, per_model_pp=per_model, nsd_minus_abs_pp=dict(zip(keys, point.tolist())), nsd_minus_abs_ci95=K.ci(draws))
            any_rows = rr["nsd"][0]
            out["n_sequences_with_positives"] = int(sum(v is not None for v in any_rows.values()))
            out["n_pos_total"] = int(sum(v["n_pos"] for v in any_rows.values() if v is not None))
            out["n_neg_total"] = int(sum(v["n_neg"] for v in any_rows.values() if v is not None))
            br = base_rows[g]
            if all(v is not None for v in br.values()):
                bpoint, _ = K.boot_single(br["nsd"], br["abs"], RC.CITY, np.random.default_rng(RC.BOOT_SEED))
                d = K.boot_joint(rr["nsd"], rr["abs"], br["nsd"], br["abs"], RC.CITY, np.random.default_rng(20261117))
                out["D_pp"] = dict(zip(keys, (point - bpoint).tolist()))
                out["D_ci95"] = K.ci(d)
                bper = {a: [K.macro(r, RC.CITY) for r in rs] for a, rs in br.items()}
                out["arm_change_vs_base_pp"] = {a: {k: mean[a][k] - 100 * float(np.mean([p[k] for p in bper[a]])) for k in keys} for a in ga}
                agree = {}
                for a in ga:
                    for t in ga[a]:
                        same = tot = 0
                        for s in RC.SEQS:
                            x, y = ledger(cond, t, s), ledger("base", t, s)
                            if len(x["q"]) != len(y["q"]) or not np.array_equal(x["q"], y["q"]):
                                same = None
                                break
                            same += int((x["j"] == y["j"]).sum())
                            tot += len(x["j"])
                        agree[t] = None if same is None else same / tot
                out["winner_agreement_vs_base"] = agree
            rec[g] = out
        nq = {}
        for s in RC.SEQS:
            p = OUT / cond / "ledgers/test" / s.replace("/", "_") / "nsd@s4.npz"
            b = OUT / "base/ledgers/test" / s.replace("/", "_") / "nsd@s4.npz"
            if p.exists():
                with np.load(p) as z, np.load(b) as zb:
                    nq[s] = dict(n_queries=int(len(z["query_idx"])), n_queries_base=int(len(zb["query_idx"])),
                                 n_pos=int(np.sum(z["min_past_gt_dist"] <= 5)), n_pos_base=int(np.sum(zb["min_past_gt_dist"] <= 5)),
                                 n_neg=int(np.sum(z["min_past_gt_dist"] > 25)), n_neg_base=int(np.sum(zb["min_past_gt_dist"] > 25)))
        if nq:
            rec["queries"] = dict(total={k: int(sum(v[k] for v in nq.values())) for k in next(iter(nq.values()))}, per_sequence=nq)
        folder = OUT / cond / "inputs/test"
        dg = [json.loads(p.read_text())["diag"] for p in sorted(folder.glob("*.json"))] if folder.exists() else []
        if dg:
            num = [k for k, v in dg[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            rec["input_diag_median"] = {k: float(np.median([d[k] for d in dg if d.get(k) is not None])) for k in num}
            rec["input_diag_median"]["n_sequences"] = len(dg)
        res["conditions"][cond] = rec
    (OUT / "summary3.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for cond, rec in res["conditions"].items():
        for g in arms:
            if g in rec:
                o = rec[g]
                line = f"{cond:11s} {g:4s} nsd " + " ".join(f"{o['mean_pp']['nsd'][k]:.3f}" for k in keys) + " | abs " + \
                       " ".join(f"{o['mean_pp']['abs'][k]:.3f}" for k in keys) + " | n-a " + \
                       " ".join(f"{o['nsd_minus_abs_pp'][k]:+.2f}[{o['nsd_minus_abs_ci95'][k][0]:+.2f},{o['nsd_minus_abs_ci95'][k][1]:+.2f}]" for k in keys)
                if "D_pp" in o:
                    line += " | D " + " ".join(f"{o['D_pp'][k]:+.2f}[{o['D_ci95'][k][0]:+.2f},{o['D_ci95'][k][1]:+.2f}]" for k in keys)
                print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("eval", "campaign", "summary"))
    ap.add_argument("--cond", default="b7_even")
    ap.add_argument("--sequence")
    ap.add_argument("--models", default="main")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--root", default=str(OUT))
    a = ap.parse_args()
    if a.action == "summary":
        summary()
    else:
        assert a.cond in SUB
        assert a.action == "eval" or a.cond != "b7_s1", "b7_s1은 검산용"
        c = H.ctx()
        c["P"].e.configure()
        c["torch"].set_num_threads(4)
        if a.action == "eval":
            evaluate(a.cond, a.sequence, H.model_names(a.models), Path(a.root))
        else:
            campaign(a.cond, a.models, a.workers)
