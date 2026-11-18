#!/usr/bin/env python3
"""평가 때 입력을 메모리 안에서 흔드는 평가 하네스(학습 없음): 차이 K/V(NSD) 대 절대 K/V(abskv).

평가 경로는 scripts/_report_model_controls_20261116.py(C)의 pipe(대칭 창)를 그대로 옮겼다:
  P.inputs(캐시·정답·travel·뱅크) -> prior.graph_inputs(그래프·간선 속성·전달 위상) -> net -> 후보 풀 승자
  -> plan(travel 거리 대응) -> P.score. 접두(q+15) 검사는 하지 않는다(교란과 무관한 인과 검산이며 기존 원장에서 이미 통과).
기존 스크립트는 고치지 않고 import만 한다. 중간 그래프·디스크립터 캐시는 디스크에 쓰지 않는다.

정답(poses_gt에서 온 pos, min_past_gt_dist, top1 거리)은 항상 교란하지 않은 참값이다. 교란은 방법의 입력에만 들어간다.

-------------------------------------------------------------------- 모델(12개, 이름 -> 체크포인트)
  nsd@s4..6   원고 NSD(sym)      results/pdv2_scratch_search_20261116/runs/<CFG>/s{k}/u512.pth (S.load)
  abskv@s4..6 원고 abskv         results/pdv2_report_model_20261116/controls/runs/abskv/s{k}/u512.pth (C.load)
  nsd@s1..3   sym, 기본 조건      results/kv_diff_improve_20261118/sweep2/runs/base/sym/s{k}/u512.pth
  abskv@s1..3 abskv, 기본 조건    results/kv_diff_improve_20261118/sweep2/runs/base/abskv/s{k}/u512.pth
    (시드 1~3 두 모델은 같은 스크립트 C.train, 같은 로컬 RTX 5080에서 학습된 짝. training_complete.json의 gpu 필드)
  보조(기본 묶음에 없음): nsdL@s4..6, abskvL@s4..6 = results/kv_diff_improve_20261118/final_s456/runs/{sym,abskv}
묶음: main = nsd/abskv 시드 4~6, ref = 시드 1~3, all12 = main+ref, aux = 보조 6개.

-------------------------------------------------------------------- 조건(교란) 추가 방법
COND 사전에 Cond(name, desc, inputs=..., graph=..., params=...)를 하나 넣는다(register()).
  inputs(seq, tup, rng_key) -> (tup2, diag)
      tup = P.inputs("test", seq)의 8-튜플 (cache, pos, minimum, travel, bank, nb, nc, paths).
      cache는 dict(descriptors, poses, timestamps, scan_ids, keyframe_ids, spectral_entropies, poses_gt).
      교란한 새 튜플을 돌려준다(원본 배열을 제자리에서 바꾸지 말 것). pos·minimum·poses_gt는 바꾸지 않는다(assert로 검사).
      이 튜플은 그래프 생성(prior.graph_inputs가 run.inputs로 읽음)과 후보 선택(raw 키·bank)·plan(travel)에 함께 쓰인다.
      diag는 시퀀스별 진단(json으로 저장, 작은 값만).
  graph(seq, graph, tup2) -> (graph2, diag)   그래프를 만든 뒤 바꿀 것(간선 제거, graph.x·graph.spectrum 잡음 등).
      간선을 바꾸면 edge_attr·edge_type·transport_indices·transport_mask도 일관되게 고쳐야 한다(C.to_p10 참고).
      주의: graph.spectrum은 graph_inputs가 run.white.coefficients(paths[1])로 디스크에서 읽으므로 inputs 훅으로는 안 바뀐다.
  plan(seq, travel, q, j) -> plan (선택)        기본은 C.ORIGINAL["plan"].
  난수는 seed_of(seq, ...)로 시퀀스 이름에서 유도해 모든 모델에 같은 실현을 쓴다.

-------------------------------------------------------------------- 실행(컨테이너 안; GPU 1장이므로 동시 2개까지)
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_kveval_perturb_20261118.py \
      campaign --cond base --models main --workers 2
  ... eval --cond COND --sequence kitti/00 --models main        (시퀀스 하나)
  ... list                                                        (등록된 조건)
  python3 scripts/_kveval_perturb_20261118.py summary            (호스트 numpy로 돈다; torch 불필요)
출력: results/kv_eval_conditions_20261118/perturb/<조건>/{condition.json, inputs/test/<seq>.json,
      ledgers/test/<seq>/<model>.{npz,json}}, perturb/summary.json, perturb/logs/.
원장 npz(압축): query_idx, j_star, min_past_gt_dist, top1_geo_dist, score. json: 지표·시간·(base는) 기존 원장 대조.

-------------------------------------------------------------------- 주행 기하 교란(B1, B2)
방법이 주행 기하(캐시 poses = KISS-ICP 오도메트리)를 쓰는 곳:
  (1) 시간 간선 속성 pose_attributes(poses): log1p(거리)/5, 상대 회전각/pi -> edge_encoder -> 두 DiffAttnConv 층의
      bias_mlp(attention logit의 기하 항). 두 모델 공통.
  (2) lidar_yaw(poses) -> transport_phase: 이웃 복소 스펙트럼을 중심 좌표로 돌리는 위상(전달 차이 연산). 두 모델 공통.
  (3) travel = 누적 주행 거리(poses 위치) -> plan: 후보 쪽 이웃을 질의 쪽 ±5 이웃의 주행 거리 차로 대응(시간 증거 판정).
  쓰지 않는 곳: 후보 풀(candidate_pool.npz: 원키·원보조 디스크립터 상위 100의 합집합, 제외 창 q-29는 프레임 번호),
  승자 선택(디스크립터·모델 출력만), 노드 입력(디스크립터·스펙트럼). 키프레임 선택(캐시 생성 때 오도메트리로 이미 고정)은
  다시 하지 않는다(한계).
B1: 연속 키프레임 상대 변환 T_i = P_{i-1}^-1 P_i 의 병진에 등방 3차원 가우시안(축별 sigma = 비율 x |t_i|), 회전에
    요 잡음 Rz(delta), delta ~ N(0, sigma_yaw)을 왼쪽에 곱하고 P'_0 = P_0에서 다시 적분한다(드리프트 누적).
    표준 정규 난수는 시퀀스·실현마다 한 번 뽑아 네 단계에 같이 쓴다(공통 난수; 단계 간 비교가 덜 시끄럽다).
B2: 상대 병진에 전역 배율 s(회전 그대로) -> 다시 적분. travel이 s배가 되므로 plan의 거리 대응은 수학적으로 불변이고
    (부동소수 동점만 바뀔 수 있음) 간선 속성의 거리 항만 바뀐다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
OUT = ROOT / "results/kv_eval_conditions_20261118/perturb"
CFG = "x=log__n=ln__g=frozen__lr=1e-3__U=1536"
U = 512
SW2 = ROOT / "results/kv_diff_improve_20261118/sweep2"
FIN = ROOT / "results/kv_diff_improve_20261118/final_s456"
SEARCH_LEDGERS = ROOT / "results/pdv2_scratch_search_20261116/ledgers"
CTRL_LEDGERS = ROOT / "results/pdv2_report_model_20261116/controls/ledgers"
FIELDS = ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist")

_CTX = {}


def ctx():
    """torch를 쓰는 평가 모듈은 필요할 때만 불러온다(summary는 호스트 numpy로 돈다)."""
    if not _CTX:
        import torch
        import _report_model_controls_20261116 as C
        _CTX.update(torch=torch, C=C, S=C.S, P=C.P, run=C.P.run, prior=C.P.prior, pipeline=C.P.pipeline)
    return _CTX


# ----------------------------------------------------------------- 모델
def model_names(group: str) -> list:
    groups = {"main": [f"{a}@s{k}" for a in ("nsd", "abskv") for k in (4, 5, 6)],
              "ref": [f"{a}@s{k}" for a in ("nsd", "abskv") for k in (1, 2, 3)],
              "aux": [f"{a}@s{k}" for a in ("nsdL", "abskvL") for k in (4, 5, 6)]}
    groups["all12"] = groups["main"] + groups["ref"]
    return groups[group] if group in groups else group.split(",")


def model_source(m: str) -> dict:
    """체크포인트와 기존 원장(기본 조건 재현 대조용)의 경로."""
    arm, seed = m.split("@")
    k = int(seed[1:])
    canon = "x=log__n=ln__g=frozen__lr=1e-3__U=1536"
    if arm == "nsd" and k >= 4:
        return dict(variant="sym", ckpt=ROOT / f"results/pdv2_scratch_search_20261116/runs/{canon}/s{k}/u{U}.pth",
                    ref=lambda name: SEARCH_LEDGERS / "test" / name / f"{canon}@s{k}@u{U}.npz", loader="S")
    if arm == "abskv" and k >= 4:
        return dict(variant="abskv", ckpt=ROOT / f"results/pdv2_report_model_20261116/controls/runs/abskv/s{k}/u{U}.pth",
                    ref=lambda name: CTRL_LEDGERS / "test" / name / f"abskv@s{k}@u{U}.npz", loader="C")
    if arm in ("nsd", "abskv"):
        v = "sym" if arm == "nsd" else "abskv"
        return dict(variant=v, ckpt=SW2 / "runs/base" / v / f"s{k}" / f"u{U}.pth",
                    ref=lambda name: SW2 / "ledgers/test" / name / f"{v}@base@s{k}.npz", loader="build")
    if arm in ("nsdL", "abskvL"):
        v = "sym" if arm == "nsdL" else "abskv"
        return dict(variant=v, ckpt=FIN / "runs" / v / f"s{k}" / f"u{U}.pth",
                    ref=lambda name: FIN / "ledgers/test" / name / f"{v}@s{k}@u{U}.npz", loader="build")
    raise KeyError(m)


def load_model(m: str):
    c = ctx()
    torch, C, S = c["torch"], c["C"], c["S"]
    src = model_source(m)
    k = int(m.split("@")[1][1:])
    if src["loader"] == "S":
        net = S.load(CFG, k, U)
    elif src["loader"] == "C":
        net = C.load(src["variant"], k)
    else:
        net = C.build(src["variant"], k)
        net.load_state_dict(torch.load(src["ckpt"], map_location="cuda", weights_only=False)["model_state_dict"], strict=True)
    kind = "abs_kv" if src["variant"] == "abskv" else "diff"
    assert net.kind == kind, (m, net.kind)
    return net.eval()


# ----------------------------------------------------------------- 조건
@dataclass
class Cond:
    name: str
    desc: str
    params: dict = field(default_factory=dict)
    inputs: Optional[Callable] = None
    graph: Optional[Callable] = None
    plan: Optional[Callable] = None


COND: dict = {}


def register(c: Cond) -> None:
    assert c.name not in COND, c.name
    COND[c.name] = c


def seed_of(seq: str, *extra: int) -> list:
    """시퀀스 이름에서 유도한 고정 난수 시드(모든 모델·단계 공통)."""
    return [zlib.crc32(seq.encode()), 20261118, *extra]


def travel_of(poses: np.ndarray) -> np.ndarray:
    """P.inputs(_pdv2_diffattn_comparable_test.inputs)와 같은 식."""
    xyz = poses[:, :3, 3]
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]


def rot_z(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    r = np.zeros(a.shape + (3, 3))
    r[..., 0, 0], r[..., 0, 1], r[..., 1, 0], r[..., 1, 1], r[..., 2, 2] = c, -s, s, c, 1.0
    return r


def reintegrate(poses: np.ndarray, frac: float, yaw_deg: float, scale: float, eps_t: np.ndarray, eps_y: np.ndarray) -> np.ndarray:
    """상대 변환을 교란하고 P'_0 = P_0에서 다시 적분한다."""
    p = np.asarray(poses, np.float64)
    rel = np.linalg.inv(p[:-1]) @ p[1:]
    t = rel[:, :3, 3]
    dist = np.linalg.norm(t, axis=1)
    rel2 = rel.copy()
    rel2[:, :3, 3] = scale * t + frac * dist[:, None] * eps_t
    rel2[:, :3, :3] = rot_z(np.deg2rad(yaw_deg) * eps_y) @ rel[:, :3, :3]
    out = np.empty_like(p)
    out[0] = p[0]
    for i in range(1, len(p)):
        out[i] = out[i - 1] @ rel2[i - 1]
    return out


def _yaw(poses: np.ndarray) -> np.ndarray:
    from gnn.spectral_transport import lidar_yaw
    return lidar_yaw(poses, None)


def odometry_inputs(frac: float, yaw_deg: float, scale: float, realization: int) -> Callable:
    def hook(seq: str, tup: tuple, *_):
        cache = dict(tup[0])
        poses = np.asarray(cache["poses"], np.float64)
        n = len(poses)
        rng = np.random.default_rng(seed_of(seq, realization))
        eps_t, eps_y = rng.standard_normal((n - 1, 3)), rng.standard_normal(n - 1)
        new = reintegrate(poses, frac, yaw_deg, scale, eps_t, eps_y)
        zero = reintegrate(poses, 0.0, 0.0, 1.0, eps_t, eps_y)  # 적분기 검산(교란 0)
        cache["poses"] = new
        travel = travel_of(new)
        tr0 = np.asarray(tup[3])
        from _pdv2_input_alignment_lib import pose_attributes, temporal_edges
        edges = temporal_edges(np.zeros(n, np.int64))
        a0, a1 = pose_attributes(poses, edges), pose_attributes(new, edges)
        dyaw = np.angle(np.exp(1j * (_yaw(new) - _yaw(poses))))
        err = np.linalg.norm(new[:, :3, 3] - poses[:, :3, 3], axis=1)
        diag = dict(n=n, travel_total=float(tr0[-1]), travel_total_perturbed=float(travel[-1]),
                    end_drift_m=float(err[-1]), max_drift_m=float(err.max()), end_drift_frac_of_travel=float(err[-1] / max(tr0[-1], 1e-9)),
                    yaw_abs_dev_deg=dict(median=float(np.degrees(np.median(np.abs(dyaw)))), max=float(np.degrees(np.abs(dyaw).max()))),
                    edge_attr_abs_change_mean=[float(x) for x in np.abs(a1 - a0).mean(0)[:2]],
                    edge_attr_abs_change_p95=[float(x) for x in np.quantile(np.abs(a1 - a0), 0.95, axis=0)[:2]],
                    integrator_zero_noise_max_pos_err_m=float(np.abs(zero[:, :3, 3] - poses[:, :3, 3]).max()),
                    integrator_zero_noise_max_rot_err=float(np.abs(zero[:, :3, :3] - poses[:, :3, :3]).max()))
        return (cache, tup[1], tup[2], travel) + tuple(tup[4:]), diag
    return hook


register(Cond("base", "교란 없음(기본 조건, 로컬 재실행)"))
B1_LEVELS = {1: (0.01, 0.1), 2: (0.05, 0.5), 3: (0.10, 1.0), 4: (0.20, 2.0)}
B1_REALIZATIONS = {1: (0,), 2: (0,), 3: (0, 1, 2), 4: (0, 1, 2)}
for lv, (fr, yd) in B1_LEVELS.items():
    for r in B1_REALIZATIONS[lv]:
        register(Cond(f"b1_L{lv}_r{r}", f"B1 누적 오도메트리 잡음 단계 {lv}: 병진 sigma={fr:g}x|t|(축별 등방), 요 sigma={yd:g}도/키프레임, 실현 {r}",
                      dict(family="B1", level=lv, trans_frac=fr, yaw_deg=yd, realization=r), inputs=odometry_inputs(fr, yd, 1.0, r)))
for sc in (0.90, 0.95, 1.05, 1.10):
    register(Cond(f"b2_s{sc:.2f}", f"B2 오도메트리 병진 전역 배율 {sc:.2f}(회전 그대로)",
                  dict(family="B2", scale=sc), inputs=odometry_inputs(0.0, 0.0, sc, 0)))


# ----------------------------------------------------------------- 평가(C.pipe의 대칭 창 경로)
def evaluate(cond: str, sequence: str, models: list, root: Path = OUT) -> None:
    c = ctx()
    torch, C, P, run, prior, pipeline = c["torch"], c["C"], c["P"], c["run"], c["prior"], c["pipeline"]
    spec = COND[cond]
    name = sequence.replace("/", "_")
    folder = root / cond / "ledgers" / "test" / name
    todo = [m for m in models if not (folder / (m + ".json")).exists()]
    if not todo:
        return
    folder.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    tup0 = P.inputs("test", sequence)
    tup, diag = (spec.inputs(sequence, tup0) if spec.inputs else (tup0, {}))
    assert tup[1] is tup0[1] and tup[2] is tup0[2], "정답(pos, minimum)은 교란하지 않는다"
    assert np.array_equal(tup[0]["poses_gt"], tup0[0]["poses_gt"])
    cache, pos, minimum, travel, bank = tup[:5]
    saved = run.inputs
    run.inputs = lambda role, s: tup if (role, s) == ("test", sequence) else saved(role, s)
    try:
        graph, _, _ = prior.graph_inputs("test", sequence)
    finally:
        run.inputs = saved
    if spec.graph:
        graph, gdiag = spec.graph(sequence, graph, tup)
        diag["graph"] = gdiag
    raw = pipeline.raw_key(cache["descriptors"])
    weights, lam = P.selected()
    poolpath = C.P.TEST / "test" / name / "candidate_pool.npz"
    assert run.HASH(poolpath) == run.READ(poolpath.with_suffix(".json"))["sha256"]
    base_pool = run.ARR(poolpath)
    static = []
    for n, q in enumerate(base_pool["query"]):
        lo, hi = base_pool["offset"][n:n + 2]
        ids = base_pool["ids"][lo:hi]
        static.append(weights[0] * run.base.minmax(raw[ids] @ raw[q]) + weights[1] * run.base.minmax(bank[0][ids] @ bank[0][q]))
    pool = dict(base_pool, static=np.concatenate(static).astype(np.float32))
    prep = time.time() - tick
    P.write(root / cond / "inputs" / "test" / (name + ".json"), dict(diag=diag, prep_seconds=prep))
    planfn = spec.plan or (lambda s, tr, q, j: C.ORIGINAL["plan"](tr, q, j))
    for m in todo:
        t0 = time.time()
        net = load_model(m)
        with torch.no_grad():
            z = net(graph)
            change = run.base.variant_inputs(z.cpu().numpy(), 8)
            q = pool["query"]
            j = pipeline.winners(pool, change, weights)
            assert (j <= q - 30).all()
            row = dict(query_idx=q, j_star=j, min_past_gt_dist=minimum[q], top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1))
            plan = planfn(sequence, travel, q, j)
            value = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(q)).astype(np.float64)
        metric = run.base.independent_metrics(row, value)
        rec = dict(model=m, cond=cond, metrics=metric, seconds=time.time() - t0, prep_seconds=prep,
                   checkpoint=str(model_source(m)["ckpt"].relative_to(ROOT)), gpu=torch.cuda.get_device_name())
        if cond == "base":
            refp = model_source(m)["ref"](name)
            if refp.exists():
                with np.load(refp) as r:
                    rj = run.READ(refp.with_suffix(".json"))["metrics"]
                    rec["reproduction"] = dict(
                        reference=str(refp.relative_to(ROOT)), query_equal=bool(np.array_equal(q, r["query_idx"])),
                        winners_equal_frac=float(np.mean(j == r["j_star"])), score_bitwise_equal=bool(np.array_equal(value, r["score"])),
                        score_max_error=float(np.abs(value - r["score"]).max()),
                        metric_diff={k: (None if metric[k] is None or rj[k] is None else float(metric[k] - rj[k])) for k in metric if k in rj})
        np.savez_compressed(folder / (m + ".npz"), **row, score=value)
        P.write(folder / (m + ".json"), rec)
        del net, z, change
    print("EVAL", cond, sequence, len(todo), round(time.time() - tick, 1), flush=True)


def campaign(cond: str, group: str, workers: int) -> None:
    import concurrent.futures
    assert workers <= 2
    c = ctx()
    seqs = c["P"].TESTSEQ
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    spec = COND[cond]
    c["P"].write(OUT / cond / "condition.json", dict(name=cond, desc=spec.desc, params=spec.params))
    tick = time.time()

    def call(s):
        with (logs / f"{cond}_{s.replace('/', '_')}.log").open("a") as f:
            subprocess.run([sys.executable, __file__, "eval", "--cond", cond, "--sequence", s, "--models", group],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(call, s): s for s in seqs}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            fut.result()
            print("DONE", cond, group, i, len(seqs), futs[fut], round(time.time() - tick, 1), flush=True)
    c["P"].write(logs / f"campaign_{cond}_{group}.done.json", dict(seconds=time.time() - tick, workers=workers,
                                                                    models=model_names(group)))


# ----------------------------------------------------------------- 집계(_kv_subset_analysis_20261117 규약)
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

    conds = [c for c in COND if (OUT / c / "ledgers").exists()]
    res = dict(conventions=dict(
        aggregation="시퀀스 지표(_pr_eval_lib.pr_sweep, R@1=양성 top1<=5 m) -> 도시 안 평균 -> 6도시 동일 가중 -> 시드 평균 (KITTI 07 양성 없음)",
        interval="NSD-abs: 시퀀스별 (NSD 시드 평균 - abs 시드 평균), 도시 안 시퀀스 복원추출 10,000회, rng=default_rng(20261104) 비교마다 새로",
        D="D = (NSD-abs)_조건 - (NSD-abs)_base(로컬 재실행), 같은 시퀀스 -> boot_joint(도시 안 시퀀스 색인 공유), rng=default_rng(20261117)",
        units="%p", baseline="base = 이 하네스로 로컬 RTX 5080에서 다시 돌린 교란 없음 조건"), conditions={})
    base_rows = {g: {a: rows("base", t) for a, t in arms[g].items()} for g in arms}
    for cond in conds:
        rec = dict(desc=COND[cond].desc, params=COND[cond].params)
        for g, ga in arms.items():
            rr = {a: rows(cond, t) for a, t in ga.items()}
            if any(v is None for v in rr.values()):
                continue
            per = {a: [K.macro(r, RC.CITY) for r in rs] for a, rs in rr.items()}
            mean = {a: {k: 100 * float(np.mean([p[k] for p in ps])) for k in keys} for a, ps in per.items()}
            per_model = {t: {k: 100 * p[k] for k in keys} for a in ga for t, p in zip(ga[a], per[a])}
            point, draws = K.boot_single(rr["nsd"], rr["abs"], RC.CITY, np.random.default_rng(RC.BOOT_SEED))
            out = dict(mean_pp=mean, per_model_pp=per_model, nsd_minus_abs_pp=dict(zip(keys, point.tolist())), nsd_minus_abs_ci95=K.ci(draws))
            br = base_rows[g]
            if cond != "base" and all(v is not None for v in br.values()):
                bpoint, _ = K.boot_single(br["nsd"], br["abs"], RC.CITY, np.random.default_rng(RC.BOOT_SEED))
                d = K.boot_joint(rr["nsd"], rr["abs"], br["nsd"], br["abs"], RC.CITY, np.random.default_rng(20261117))
                out["D_pp"] = dict(zip(keys, (point - bpoint).tolist()))
                out["D_ci95"] = K.ci(d)
                # 기본 대비 각 팔의 변화(시드 평균 매크로)
                bper = {a: [K.macro(r, RC.CITY) for r in rs] for a, rs in br.items()}
                out["arm_change_vs_base_pp"] = {a: {k: mean[a][k] - 100 * float(np.mean([p[k] for p in bper[a]])) for k in keys} for a in ga}
                # 승자 일치율(질의 풀링)
                agree = {}
                for a in ga:
                    for t in ga[a]:
                        same = tot = 0
                        for s in RC.SEQS:
                            x, y = ledger(cond, t, s), ledger("base", t, s)
                            same += int((x["j"] == y["j"]).sum())
                            tot += len(x["j"])
                        agree[t] = same / tot
                out["winner_agreement_vs_base"] = agree
            rec[g] = out
        # 입력 진단(시퀀스 중앙값)
        dg = [json.loads(p.read_text())["diag"] for p in sorted((OUT / cond / "inputs/test").glob("*.json"))] if (OUT / cond / "inputs/test").exists() else []
        if dg and "end_drift_m" in dg[0]:
            rec["input_diag_median"] = dict(
                end_drift_m=float(np.median([d["end_drift_m"] for d in dg])),
                end_drift_frac_of_travel=float(np.median([d["end_drift_frac_of_travel"] for d in dg])),
                yaw_median_abs_dev_deg=float(np.median([d["yaw_abs_dev_deg"]["median"] for d in dg])),
                edge_attr_dist_abs_change_mean=float(np.median([d["edge_attr_abs_change_mean"][0] for d in dg])),
                edge_attr_angle_abs_change_mean=float(np.median([d["edge_attr_abs_change_mean"][1] for d in dg])),
                integrator_zero_noise_max_pos_err_m=float(max(d["integrator_zero_noise_max_pos_err_m"] for d in dg)),
                n_sequences=len(dg))
        res["conditions"][cond] = rec
    # 기본 조건 재현(기존 원장 대비)
    rep = {}
    for g, ga in arms.items():
        for t in ga["nsd"] + ga["abs"]:
            js = [OUT / "base/ledgers/test" / s.replace("/", "_") / (t + ".json") for s in RC.SEQS]
            if not all(p.exists() for p in js):
                continue
            rs = [json.loads(p.read_text()).get("reproduction") for p in js]
            if any(r is None for r in rs):
                continue
            nq = [len(np.load(p.with_suffix(".npz"))["query_idx"]) for p in js]
            rep[t] = dict(all_score_bitwise=all(r["score_bitwise_equal"] for r in rs), all_query_equal=all(r["query_equal"] for r in rs),
                          winners_equal_frac_pooled=float(np.sum([r["winners_equal_frac"] * n for r, n in zip(rs, nq)]) / np.sum(nq)),
                          score_max_error=max(r["score_max_error"] for r in rs), reference_example=rs[0]["reference"])
    # 기존 원장으로 같은 집계 -> 매크로 차이
    refmac = {}
    for g, ga in arms.items():
        for a, tags in ga.items():
            refs = []
            for t in tags:
                r = {}
                for s in RC.SEQS:
                    p = model_source(t)["ref"](s.replace("/", "_"))
                    with np.load(p) as z:
                        r[s] = K.metric(dict(score=z["score"].astype(np.float64), top1=z["top1_geo_dist"], gt=z["min_past_gt_dist"]))
                refs.append(r)
            refmac[f"{g}/{a}"] = {k: 100 * float(np.mean([K.macro(r, RC.CITY)[k] for r in refs])) for k in keys}
    res["base_reproduction"] = dict(per_model=rep, reference_ledger_macro_pp=refmac,
                                    expected_main=dict(nsd=K.EXPECT["nsd"], abs=K.EXPECT["abs"]))
    if "base" in res["conditions"]:
        for g in arms:
            if g in res["conditions"]["base"]:
                res["base_reproduction"][f"{g}_local_minus_reference_pp"] = {
                    a: {k: res["conditions"]["base"][g]["mean_pp"][a][k] - refmac[f"{g}/{a}"][k] for k in keys} for a in arms[g]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for cond, rec in res["conditions"].items():
        for g in arms:
            if g in rec:
                o = rec[g]
                line = f"{cond:14s} {g:4s} nsd " + " ".join(f"{o['mean_pp']['nsd'][k]:.3f}" for k in keys) + " | abs " + \
                       " ".join(f"{o['mean_pp']['abs'][k]:.3f}" for k in keys) + " | n-a " + \
                       " ".join(f"{o['nsd_minus_abs_pp'][k]:+.2f}[{o['nsd_minus_abs_ci95'][k][0]:+.2f},{o['nsd_minus_abs_ci95'][k][1]:+.2f}]" for k in keys)
                if "D_pp" in o:
                    line += " | D " + " ".join(f"{o['D_pp'][k]:+.2f}[{o['D_ci95'][k][0]:+.2f},{o['D_ci95'][k][1]:+.2f}]" for k in keys)
                print(line)
    print("REPRO", json.dumps(res["base_reproduction"], ensure_ascii=False)[:3000])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("eval", "campaign", "summary", "list"))
    p.add_argument("--cond", default="base")
    p.add_argument("--sequence")
    p.add_argument("--models", default="main")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--root", default=str(OUT), help="eval 출력 위치(결정성 검사 등은 tmp/ 아래로)")
    a = p.parse_args()
    if a.action == "list":
        for k, v in COND.items():
            print(k, "|", v.desc)
    elif a.action == "summary":
        summary()
    else:
        c = ctx()
        c["P"].e.configure()
        c["torch"].set_num_threads(4)
        if a.action == "eval":
            evaluate(a.cond, a.sequence, model_names(a.models), Path(a.root))
        else:
            campaign(a.cond, a.models, a.workers)
