#!/usr/bin/env python3
"""NSD(처음부터 한 단계 학습) 구성의 두 학습 대조: Past-only NSD(무지연, 과거 10)와 절대 K/V attention.

NSD 구성 = scripts/_pdv2_scratch_search_20261116.py(S)의 CFG "x=log__n=ln__g=frozen__lr=1e-3__U=1536" @512갱신,
고정 설정 P.selected() = 융합 (2,1,2,2), lambda=0.25. 학습·검색 경로는 S.train / S.pipe를 그대로 옮기고 다음만 바꾼다.

- variant "sym"     : NSD 자체(재현 검사용). 학습 그래프·표본·보정 모두 S와 같다.
- variant "pastonly": 정제 이웃과 수락의 시간 증거를 앞선 10키프레임으로 제한(지연 0, 질의 q=t).
    그래프·plan 변환은 scripts/_pdv2_zero_lag_p10.py의 to_p10 / p10_neighbor_plans를 그대로 옮겼다
    (간선 (i-d -> i), d=1..10, 전달 칸 (오프셋+5) mod 11, 질의 쪽 오프셋 -10..-1, 후보 쪽 거리 대응 반경 20).
    학습 표본의 (질의, 고정 승자)와 정답 표지는 P.build_pool이 정제와 무관한 원키·원보조 뱅크로 고르므로 대칭과 같고,
    대응 계획(plan_*)만 과거 10 규칙으로 다시 만든다. 손실의 고정 양의 아핀 보정은 이 조건의 초기 출력
    (과거 10 균등 감산 -0.5)에서 P.calibrate와 같은 절차로 다시 적합한다. 융합·lambda는 다시 고르지 않는다.
- variant "abskv"   : K와 V 입력을 차이 대신 이웃 절대 특징으로(ContrastSpectralDifference(ref, "abs_kv")).
    대칭 창, 같은 표본·보정·고정 설정. 0 초기화 규칙이 같아 초기 출력은 균등 감산과 같다(기존 abs_kv와 같은 규칙).

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_report_model_controls_20261116.py <action>
  pools | fit | train --variant V --seed K | pipe --role R --sequence S --models FILE | pipe_campaign | summary
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import _pdv2_scratch_search_20261116 as S  # noqa: E402
from gnn.contrast_spectral_difference import ContrastSpectralDifference  # noqa: E402

P = S.P
run, prior, pipeline = P.run, P.prior, P.pipeline
ROOT = P.ROOT
CFG = "x=log__n=ln__g=frozen__lr=1e-3__U=1536"
U = 512
OUT = ROOT / "results/pdv2_report_model_20261116/controls"
RUNS, LEDGERS, POOLS, FITS = OUT / "runs", OUT / "ledgers", OUT / "pools", OUT / "fit"
TRAIN, VAL, TESTSEQ, METRICS = P.TRAIN, P.VAL, P.TESTSEQ, P.METRICS
VARIANTS = {"sym": dict(kind="diff", window="sym"), "pastonly": dict(kind="diff", window="past10"),
            "abskv": dict(kind="abs_kv", window="sym")}
WINDOW, RADIUS = 10, 20
OFFSETS = np.arange(-WINDOW, 0)
SG = prior.base.prior  # _pdv2_diffattn_spectral_gain: plan()이 모듈 전역 neighbor_plans를 부른다
ORIGINAL = dict(plan=SG.plan, neighbor_plans=SG.neighbor_plans)


# ----------------------------------------------------------------- 과거 10 (scripts/_pdv2_zero_lag_p10.py에서 옮김)
def past_edges(n: int) -> np.ndarray:
    """(i -> i+d), d=1..10. 메시지는 앞선 프레임에서 뒤 프레임으로 간다(한 시퀀스)."""
    source = np.broadcast_to(np.arange(n)[:, None], (n, WINDOW))
    target = source + np.arange(1, WINDOW + 1)[None]
    valid = target < n
    return np.stack((source[valid], target[valid]))


def to_p10(graph, poses: np.ndarray):
    """노드 입력은 그대로 두고 시간 간선·속성·전달 칸만 과거 10개로 바꾼다."""
    from _pdv2_input_alignment_lib import pose_attributes

    n = graph.num_nodes
    assert len(poses) == n
    old = graph.edge_index.cpu().numpy()
    assert np.array_equal(pose_attributes(poses, old), graph.edge_attr.cpu().numpy())
    edges = past_edges(n)
    device = graph.x.device
    g = graph.clone()
    g.edge_index = torch.as_tensor(edges, device=device)
    g.edge_attr = torch.as_tensor(pose_attributes(poses, edges), device=device)
    g.edge_type = torch.zeros(edges.shape[1], dtype=graph.edge_type.dtype, device=device)
    src, tgt = edges
    slot = (src - tgt + 5) % 11
    index = np.broadcast_to(np.arange(n)[:, None], (n, 11)).copy()
    mask = np.zeros((n, 11), np.float32)
    index[tgt, slot], mask[tgt, slot] = src, 1.0
    assert int(mask.sum()) == len(src) and not mask[:, 5].any()
    g.transport_indices = torch.as_tensor(index, device=device)
    g.transport_mask = torch.as_tensor(mask, device=device)
    s, t = g.edge_index
    assert bool((s < t).all()) and int((t - s).max()) <= WINDOW
    return g


def p10_neighbor_plans(path, query, candidate) -> dict:
    from retrieval.sequence_evidence import distance_indices

    q, j = np.asarray(query, np.int64), np.asarray(candidate, np.int64)
    qs, js, masks = [], [], []
    for direction in (1, -1):
        qn = q[:, None] + OFFSETS[None]
        qsafe = np.clip(qn, 0, len(path) - 1)
        jn = distance_indices(path, j[:, None], direction * (path[qsafe] - path[q, None]), radius=RADIUS)
        valid = (qn >= 0) & (qn < len(path)) & (jn >= 0) & (jn < len(path)) & (jn != j[:, None])
        qs.append(qn)
        js.append(jn)
        masks.append(valid)
    dense = np.stack(qs), np.stack(js), np.stack(masks)
    return {"arc_dense": dense, "arc_unique": dense}


def p10_plan(travel, q, j) -> dict:
    SG.neighbor_plans = p10_neighbor_plans
    try:
        plan = ORIGINAL["plan"](travel, q, j)
    finally:
        SG.neighbor_plans = ORIGINAL["neighbor_plans"]
    assert (plan["query"] <= np.asarray(q)[:, None, None]).all()
    return plan


def window(variant: str) -> str:
    return VARIANTS[variant]["window"]


def graph_for(variant: str, graph, poses):
    return to_p10(graph, np.asarray(poses)) if window(variant) == "past10" else graph


def plan_for(variant: str, travel, q, j) -> dict:
    return p10_plan(travel, q, j) if window(variant) == "past10" else ORIGINAL["plan"](travel, q, j)


def train_graph(variant: str, s: str):
    g = torch.load(P.recal.OUT / "train_inputs" / (s.replace("/", "_") + ".pt"), map_location="cuda", weights_only=False)
    return graph_for(variant, g, P.inputs("train", s)[0]["poses"]) if window(variant) == "past10" else g


def pool_dir(variant: str) -> Path:
    return POOLS / "past10" if window(variant) == "past10" else P.NEWPOOLS


def fit_path(variant: str) -> Path:
    return FITS / "past10.json" if window(variant) == "past10" else P.OUT / "step2/loss_fit.json"


# ----------------------------------------------------------------- 표본·보정
def pools() -> None:
    """과거 10 표본: 대칭 표본(P.NEWPOOLS)의 질의·고정 승자·표지를 그대로 두고 plan_*만 과거 10 규칙으로."""
    rec = {}
    for role, seqs in (("train", TRAIN), ("val", VAL)):
        for s in seqs:
            name = s.replace("/", "_")
            target = POOLS / "past10" / role / name / "pool.npz"
            d = dict(run.ARR(P.NEWPOOLS / role / name / "pool.npz"))
            travel = P.inputs(role, s)[3]
            q, j = d["query_idx"], d["j_star"]
            again = ORIGINAL["plan"](travel, q, j)
            assert all(np.array_equal(again[k], d["plan_" + k]) for k in again), s
            plan = p10_plan(travel, q, j)
            w = plan["weight"]
            d.update({"plan_" + k: v for k, v in plan.items()})
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **d)
            rec[s] = dict(rows=len(q), sha256=run.HASH(target), source_sha256=run.HASH(P.NEWPOOLS / role / name / "pool.npz"),
                          center_weight_mean=float(w[..., 0].mean()), weighted_neighbors_mean=float((w[..., 1:] > 0).sum(-1).mean()),
                          max_query_offset=int((plan["query"] - q[:, None, None]).max()),
                          max_candidate_offset_vs_query=int((plan["candidate"] - q[:, None, None]).max()))
            print("POOL", role, s, rec[s], flush=True)
    P.write(POOLS / "past10_summary.json", dict(
        rule="P.NEWPOOLS의 query_idx·j_star·min_past_gt_dist·top1_geo_dist·group·eligible·correct 그대로, plan_*만 과거 10 plan. "
             "P.build_pool은 원키·원보조(정제 없음) 뱅크로 고정 승자를 고르므로 정제 창과 무관하다.", sequences=rec))


def build(variant: str, seed: int, stats: dict = None) -> nn.Module:
    """S.build(CFG)와 같다(P.scratch_model의 무작위·0 초기화 -> LayerNorm 3개 -> log1p 입력). kind만 variant에 따른다."""
    cfg = S.parse(CFG)
    assert cfg["n"] == "ln" and cfg["x"] == "log" and cfg["g"] == "frozen"
    torch.manual_seed(seed)
    ref = run.source.factory(288)
    net = ContrastSpectralDifference(ref, VARIANTS[variant]["kind"]).cuda().eval()
    net.input_norm = nn.LayerNorm(128)
    net.batch_norms = nn.ModuleList([nn.LayerNorm(128), nn.LayerNorm(128)])
    if stats is None:
        z = torch.zeros(288)
        stats = dict(lin=dict(mean=z, std=z + 1), log=dict(mean=z, std=z + 1))
    st = stats["log"]
    net.input_proj = nn.Sequential(S.XNorm(cfg["x"], st["mean"], st["std"]), net.input_proj)
    return net.cuda().eval()


def load(variant: str, seed: int) -> nn.Module:
    net = build(variant, seed)
    path = RUNS / variant / f"s{seed}" / f"u{U}.pth"
    net.load_state_dict(torch.load(path, map_location="cuda", weights_only=False)["model_state_dict"], strict=True)
    return net


def fit() -> None:
    """P.calibrate와 같은 절차: 초기 출력(=균등 감산)의 학습8 점수에 균형 가중 양의 로지스틱 척도. 대칭은 재현 검사."""
    from _pdv2_contrast_calibration import fit_score

    _, lam = P.selected()
    out = {}
    for variant in ("sym", "pastonly"):
        xs, ys, ws, init = [], [], [], []
        for s in TRAIN:
            name = s.replace("/", "_")
            data = run.ARR(pool_dir(variant) / "train" / name / "pool.npz")
            graph = train_graph(variant, s)
            plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
            with torch.no_grad():
                z = prior.filtered(graph, -0.5)
                for v2 in ([variant, "abskv"] if variant == "sym" else [variant]):
                    init.append(bool(torch.equal(build(v2, 1)(graph), z)))
                v = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(data["query_idx"])).astype(np.float64)
            keep = data["eligible"]
            y = data["correct"][keep].astype(np.float64)
            b = np.zeros(len(y))
            classes = np.unique(y)
            for c in classes:
                b[y == c] = 1.0 / (len(classes) * int((y == c).sum()))
            xs.append(v[keep]); ys.append(y); ws.append(b)
            del graph, z
        x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
        out[variant] = dict(fit=fit_score(x, y, w / w.sum()), init_bitwise_equal_uniform=all(init), n=len(x))
        print("FIT", variant, out[variant], flush=True)
    ref = run.READ(P.OUT / "step2/loss_fit.json")["fit"]
    same = all(out["sym"]["fit"][k] == ref[k] for k in ("mean", "scale", "coefficients", "intercept"))
    assert same, (out["sym"]["fit"], ref)
    P.write(FITS / "past10.json", dict(fit=out["pastonly"]["fit"], lam=lam, symmetric_refit=out["sym"]["fit"],
                                       symmetric_refit_equals_existing=same, existing=str(P.OUT / "step2/loss_fit.json"),
                                       init_bitwise_equal_uniform=dict(sym_and_abskv=out["sym"]["init_bitwise_equal_uniform"],
                                                                       pastonly=out["pastonly"]["init_bitwise_equal_uniform"]),
                                       validation_used=False, test_used=False))


# ----------------------------------------------------------------- 학습 (S.train과 같은 순서·난수)
def train(variant: str, seed: int) -> None:
    cfg = S.parse(CFG)
    folder = RUNS / variant / f"s{seed}"
    if (folder / "training_complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    _, lam = P.selected()
    fitrec = run.READ(fit_path(variant))["fit"]
    net = build(variant, seed, S.xstats())
    if variant == "sym":  # S.build와 같은 초기 모델인지
        ref = S.build(cfg, seed, S.xstats())
        assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), ref.state_dict().values()))
        del ref
    assert not [m for m in net.modules() if isinstance(m, nn.BatchNorm1d)]
    for n_, p in net.named_parameters():
        p.requires_grad_(S.trainable(cfg, n_))
    graphs = {s: train_graph(variant, s) for s in TRAIN}
    with torch.no_grad():
        init = {s: bool(torch.equal(net(g), prior.filtered(g, -0.5))) for s, g in graphs.items()}
        init_err = {s: float((net(g) - prior.filtered(g, -0.5)).abs().max()) for s, g in graphs.items()}
    assert init[TRAIN[0]], init_err
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    pls = {s: run.ARR(pool_dir(variant) / "train" / s.replace("/", "_") / "pool.npz") for s in TRAIN}
    plans = {s: {k: torch.from_numpy(d["plan_" + k]).cuda() for k in ("query", "candidate", "weight")} for s, d in pls.items()}
    batch = int(cfg["B"])
    records, updates, tick = [], 0, time.time()
    for epoch in range(1, U // 64 + 1):
        for ordinal, s in enumerate(TRAIN):
            data = pls[s]
            sets = [np.flatnonzero(data["eligible"] & (data["correct"] == c)) for c in (0, 1)]
            sets = [v for v in sets if len(v)]
            rng = np.random.default_rng((seed, epoch, ordinal))
            losses, grads = [], []
            for _ in range(8):
                n = batch // len(sets)
                chosen = rng.permutation(np.concatenate([rng.choice(v, n, replace=len(v) < n) for v in sets]))
                ix = torch.from_numpy(chosen).cuda()
                for group in opt.param_groups:
                    group["lr"] = S.lr_at(cfg, updates)
                opt.zero_grad(set_to_none=True)
                z = net(graphs[s])
                value = P.score(z, {k: v[ix] for k, v in plans[s].items()}, lam)
                logit = (value - fitrec["mean"][0]) * fitrec["coefficients"][0] / fitrec["scale"][0] + fitrec["intercept"]
                loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(data["correct"][chosen]).cuda().float())
                assert torch.isfinite(loss)
                loss.backward()
                g = torch.nn.utils.clip_grad_norm_(params, 1.0)
                assert torch.isfinite(g)
                opt.step()
                updates += 1
                losses.append(float(loss.detach()))
                grads.append(float(g))
            st = net.last_strengths.detach()
            records.append(dict(epoch=epoch, sequence=s, loss=float(np.mean(losses)), grad=float(np.mean(grads)),
                                strength_mean=float(st.mean()), strength_min=float(st.min()), strength_max=float(st.max())))
    assert updates == U
    path = folder / f"u{U}.pth"
    torch.save(dict(model_state_dict=net.state_dict(), updates=updates, cfg=cfg, variant=variant, seed=seed), path)
    seconds = time.time() - tick
    changed = [k for k, v in net.state_dict().items() if not torch.equal(v, initial[k])]
    repro = None
    if variant == "sym":
        old = torch.load(S.RUNS / S.canon(CFG) / f"s{seed}" / f"u{U}.pth", map_location="cuda", weights_only=False)["model_state_dict"]
        repro = dict(state_bitwise_equal_existing=all(torch.equal(v, old[k]) for k, v in net.state_dict().items()),
                     max_abs=max(float((v.float() - old[k].float()).abs().max()) for k, v in net.state_dict().items()))
    del graphs, plans, opt
    # 검증9 고정 승자 AP(S.train과 같은 기록용)
    per = {}
    with torch.no_grad():
        for s in VAL:
            vname = s.replace("/", "_")
            graph, _, _ = prior.graph_inputs("val", s)
            graph = graph_for(variant, graph, P.inputs("val", s)[0]["poses"])
            data = run.ARR(pool_dir(variant) / "val" / vname / "pool.npz")
            row = {k: data[k] for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist")}
            plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
            z = net(graph)
            v = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(row["query_idx"])).astype(np.float64)
            per[s] = run.base.independent_metrics(row, v)
            del graph
    P.write(folder / "training_complete.json", dict(
        status="passed", variant=variant, kind=VARIANTS[variant]["kind"], window=window(variant), cfg=cfg, name=CFG, seed=seed,
        updates=updates, checkpoint=path.name, sha256=run.HASH(path), records=records, changed=changed,
        init_bitwise_equal_uniform=init, init_max_abs_vs_uniform=init_err,
        trainable_parameters=sum(p.numel() for p in params), total_parameters=sum(p.numel() for p in net.parameters()),
        fit=fitrec, fit_path=str(fit_path(variant).relative_to(ROOT)), pools=str(pool_dir(variant).relative_to(ROOT)),
        reproduction_vs_scratch_search=repro, val_fixed=dict(macro=run.base.macro(per)["macro"], sequences=per),
        seconds=seconds, torch=torch.__version__, gpu=torch.cuda.get_device_name()))
    print("TRAIN", variant, seed, round(seconds, 1), "trainable", sum(p.numel() for p in params), "repro", repro,
          "valfixedAP", round(100 * run.base.macro(per)["macro"]["AP"], 3), flush=True)


# ----------------------------------------------------------------- 실제 검색 (S.pipe와 같은 경로)
def model_spec(m: str):
    """원장 이름 -> (variant, 신경망 또는 None, 재현 대상 원장 경로 또는 None)."""
    if m == "uniform_past10":
        return "pastonly", None, None
    if m == "repro_uniform":
        return "sym", None, "uniform"
    if m.startswith("repro_nsd@"):
        seed = int(m.split("@")[1][1:])
        return "sym", S.load(CFG, seed, U), f"{S.canon(CFG)}@s{seed}@u{U}"
    variant, seed, _ = m.split("@")
    return variant, load(variant, int(seed[1:])), None


def pipe(role: str, sequence: str, models: list, nprefix: int) -> None:
    name = sequence.replace("/", "_")
    folder = LEDGERS / role / name
    todo = [m for m in models if not (folder / (m + ".json")).exists()]
    if not todo:
        return
    folder.mkdir(parents=True, exist_ok=True)
    src = (P.PIPE / "val" / name) if role == "val" else (P.TEST / "test" / name)
    cache, pos, minimum, travel, bank, _, _, _ = P.inputs(role, sequence)
    poses = np.asarray(cache["poses"])
    sym_graph, _, _ = prior.graph_inputs(role, sequence)
    graphs = {"sym": sym_graph}
    raw = pipeline.raw_key(cache["descriptors"])
    weights, lam = P.selected()
    poolpath = src / "candidate_pool.npz"
    assert run.HASH(poolpath) == run.READ(poolpath.with_suffix(".json"))["sha256"]
    base_pool = run.ARR(poolpath)
    static = []
    for n, q in enumerate(base_pool["query"]):
        lo, hi = base_pool["offset"][n:n + 2]
        ids = base_pool["ids"][lo:hi]
        static.append(weights[0] * run.base.minmax(raw[ids] @ raw[q]) + weights[1] * run.base.minmax(bank[0][ids] @ bank[0][q]))
    pool = dict(base_pool, static=np.concatenate(static).astype(np.float32))
    small = {}
    for m in todo:
        tick = time.time()
        variant, net, ref = model_spec(m)
        w = window(variant)
        if w not in graphs:
            graphs[w] = to_p10(sym_graph, poses)
        graph = graphs[w]
        with torch.no_grad():
            z = prior.filtered(graph, -0.5) if net is None else net(graph)
            change = run.base.variant_inputs(z.cpu().numpy(), 8)
            q = pool["query"]
            j = pipeline.winners(pool, change, weights)
            assert (j <= q - 30).all()
            row = dict(query_idx=q, j_star=j, min_past_gt_dist=minimum[q],
                       top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1), group=np.zeros(len(q), np.int64))
            plan = plan_for(variant, travel, q, j)
            value = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(q)).astype(np.float64)
            pre = []
            # 접두 검사: 대칭 창은 q+15(S.pipe와 같다), 과거 10은 q+0(질의 시점까지의 접두). 재현 사례는 생략.
            lag = 15 if w == "sym" else 0
            for k in range(nprefix if ref is None and (net is not None or w == "past10") else 0):
                at = (k + 1) * len(q) // (nprefix + 1)
                t = int(q[at])
                end = t + lag + 1
                key = (w, end)
                if key not in small:
                    g0 = prior.graph_inputs(role, sequence, end)[0]
                    small[key] = to_p10(g0, poses[:end]) if w == "past10" else g0
                short = prior.filtered(small[key], -0.5) if net is None else net(small[key])
                sb = run.base.variant_inputs(short.cpu().numpy(), 8)
                winner = run.cal.candidates(raw[:end], (bank[0][:end], *sb[1:]), weights[None], t)
                sp = plan_for(variant, travel[:end], np.array([t]), winner)
                fp = plan_for(variant, travel, np.array([t]), j[at:at + 1])
                pp = {kk: torch.from_numpy(v).cuda() for kk, v in sp.items()}
                chk = P.score(short, pp, lam).cpu().numpy()
                pre.append(dict(query=t, prefix_frames=end, lag=lag, winner_equal=int(winner[0]) == int(j[at]),
                                score_error=float(abs(chk - value[at]).max()),
                                output_max_error=float((short[: t + 1] - z[: t + 1]).abs().max()),
                                plan_equal=all(np.array_equal(sp[kk], fp[kk]) for kk in sp),
                                max_plan_frame_minus_query=int(max(sp["query"].max(), sp["candidate"].max()) - t)))
        metric = run.base.independent_metrics(row, value)
        other = run.white.prior.BASE.exp.metrics(value, row)
        for k in METRICS:
            if metric[k] is not None:
                np.testing.assert_allclose(metric[k], other[k], atol=1e-12, rtol=0)
        rec = dict(metrics=metric, prefix=pre, seconds=time.time() - tick, variant=variant, window=w)
        if ref:
            r = run.ARR(S.LEDGERS / role / name / (ref + ".npz"))
            rj = run.READ(S.LEDGERS / role / name / (ref + ".json"))
            rec["reproduction"] = dict(
                reference=str((S.LEDGERS / role / name / ref).relative_to(ROOT)),
                winners_equal=bool(np.array_equal(j, r["j_star"])), query_equal=bool(np.array_equal(q, r["query_idx"])),
                score_bitwise_equal=bool(np.array_equal(value, r["score"])), score_max_error=float(np.abs(value - r["score"]).max()),
                metrics_equal=all(metric[k] == rj["metrics"][k] for k in METRICS))
        np.savez_compressed(folder / (m + ".npz"), **row, score=value)
        P.write(folder / (m + ".json"), rec)
        del net, z, change
    print("PIPE", role, sequence, len(todo), flush=True)


def pipe_campaign(role: str, models_file: str, workers: int, nprefix: int) -> None:
    import concurrent.futures

    seqs = VAL if role == "val" else TESTSEQ
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tick = time.time()

    def call(s):
        with (logs / f"pipe_{role}_{s.replace('/', '_')}.log").open("a") as f:
            subprocess.run([sys.executable, __file__, "pipe", "--role", role, "--sequence", s, "--models", models_file,
                            "--prefix", str(nprefix)], stdout=f, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(concurrent.futures.as_completed([ex.submit(call, s) for s in seqs]), 1):
            fut.result()
            print("DONE", role, i, len(seqs), round(time.time() - tick, 1), flush=True)
    P.write(logs / f"pipe_campaign_{role}_{Path(models_file).stem}.done.json", dict(seconds=time.time() - tick, workers=workers))


# ----------------------------------------------------------------- 집계·구간 (compare 스크립트와 같은 규약)
NEW = {"pastonly": [f"pastonly@s{k}@u{U}" for k in (4, 5, 6)], "abskv": [f"abskv@s{k}@u{U}" for k in (4, 5, 6)]}


def ledger_file(role: str, name: str, tag: str) -> Path:
    if tag.startswith(("pastonly@", "abskv@", "uniform_past10", "repro_")):
        return LEDGERS / role / name / (tag + ".npz")
    return S.LEDGERS / role / name / (tag + ".npz")  # 새 NSD 시드·uniform


def summary() -> None:
    import _pdv2_scratch_search_compare_20261116 as C
    import _pdv2_scratch_search_verify_20261116 as V

    keys = C.KEYS
    nsd = [f"{S.canon(CFG)}@s{k}@u{U}" for k in (4, 5, 6)]
    arms = {"nsd": nsd, "pastonly": NEW["pastonly"], "abskv": NEW["abskv"], "uniform": ["uniform"], "uniform_past10": ["uniform_past10"]}
    out = dict(config=CFG, updates=U, fixed=dict(weights=P.selected()[0].tolist(), lam=P.selected()[1]), arms=arms)
    # 검증9: 파이프라인 집계(S.macro와 같은 계열 평균)
    val = {}
    for arm, tags in arms.items():
        per = {t: S.macro("val", {s: run.READ(ledger_file("val", s.replace("/", "_"), t).with_suffix(".json")) for s in VAL}) for t in tags}
        val[arm] = dict(per_model=per, **{k: dict(mean=float(np.mean([per[t][k] for t in tags])),
                                                   sd=float(np.std([per[t][k] for t in tags], ddof=1)) if len(tags) > 1 else None) for k in METRICS})
    out["val"] = val
    # 시험46: 방향별(전체·순방향만·역방향만), compare 스크립트의 metric/CITY/난수 규약
    rows = {}
    for arm, tags in arms.items():
        for t in tags:
            for seqs in C.CITY.values():
                for s in seqs:
                    name = s.replace("/", "_")
                    d = dict(np.load(C.DIRS / (name + ".npz")))
                    with np.load(ledger_file("test", name, t)) as z:
                        np.testing.assert_array_equal(z["query_idx"], d["query_idx"])
                        masks = dict(all=None, forward_only=d["group"] == 0, reverse_only=d["group"] == 1)
                        rows.setdefault(t, {})[s] = {g: C.metric(z["score"], z["top1_geo_dist"], z["min_past_gt_dist"], m)
                                                     for g, m in masks.items()}
    pairs = [("pastonly", "nsd"), ("pastonly", "uniform"), ("pastonly", "uniform_past10"), ("nsd", "pastonly"),
             ("abskv", "nsd"), ("abskv", "uniform"), ("nsd", "abskv"), ("uniform_past10", "uniform")]
    test = {}
    for g in ("all", "forward_only", "reverse_only"):
        def macro(t):
            cities = []
            for seqs in C.CITY.values():
                v = [rows[t][s][g] for s in seqs if rows[t][s][g] is not None]
                if v:
                    cities.append({k: np.mean([x[k] for x in v]) for k in keys})
            return {k: float(np.mean([c[k] for c in cities])) for k in keys}
        per_model = {t: macro(t) for tags in arms.values() for t in tags}
        stats = {arm: {k: dict(mean=float(np.mean([per_model[t][k] for t in tags])),
                               sd=float(np.std([per_model[t][k] for t in tags], ddof=1)) if len(tags) > 1 else None,
                               values=[per_model[t][k] for t in tags]) for k in keys} for arm, tags in arms.items()}
        comp = {}
        for a, b in pairs:
            rng = np.random.default_rng(20261104)
            draws, city = [], {}
            for cname, seqs in C.CITY.items():
                avail = [s for s in seqs if rows[arms[a][0]][s][g] is not None]
                if not avail:
                    continue
                mat = np.array([[100 * (np.mean([rows[t][s][g][k] for t in arms[a]]) - np.mean([rows[t][s][g][k] for t in arms[b]]))
                                 for k in keys] for s in avail])
                city[cname] = dict(zip(keys, mat.mean(0).tolist()))
                idx = rng.integers(0, len(avail), size=(10000, len(avail)))
                draws.append(mat[idx].mean(1))
            boot = np.stack(draws).mean(0)
            comp[f"{a}-{b}"] = dict(point_pp={k: 100 * (stats[a][k]["mean"] - stats[b][k]["mean"]) for k in keys},
                                    conditional_95_pp={k: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(keys)},
                                    city_delta_pp=city)
        test[g] = dict(per_model=per_model, arms=stats, differences=comp)
    out["test"] = test
    # 검산: 파이프라인 집계(S.macro) · compare 경로(pr_sweep) · 독립 단순 구현(verify 스크립트)
    checks = {}
    for arm, tags in arms.items():
        for t in tags:
            pipe_macro = S.macro("test", {s: run.READ(ledger_file("test", s.replace("/", "_"), t).with_suffix(".json")) for s in TESTSEQ})
            cities = []
            for seqs in V.CITY.values():
                rr = [V.metrics(dict(np.load(ledger_file("test", s.replace("/", "_"), t)))) for s in seqs]
                rr = [r for r in rr if r is not None]
                cities.append({k: np.mean([r[k] for r in rr]) for k in rr[0]})
            indep = {k: float(np.mean([c[k] for c in cities])) for k in cities[0]}
            checks[t] = dict(pipeline=pipe_macro, independent=indep, compare_path=test["all"]["per_model"][t],
                             max_abs_independent_vs_pipeline=max(abs(indep[k] - pipe_macro[k]) for k in keys),
                             max_abs_compare_vs_pipeline=max(abs(test["all"]["per_model"][t][k] - pipe_macro[k]) for k in keys))
    out["metric_checks"] = checks
    # 재현·접두·학습 기록
    rep = {}
    for role, seqs in (("val", VAL), ("test", TESTSEQ)):
        for t in ("repro_uniform", "repro_nsd@s4@u512"):
            recs = [run.READ(LEDGERS / role / s.replace("/", "_") / (t + ".json"))["reproduction"] for s in seqs]
            rep[f"{role}/{t}"] = dict(all_exact=all(r["winners_equal"] and r["score_bitwise_equal"] and r["metrics_equal"] for r in recs),
                                      max_score_error=max(r["score_max_error"] for r in recs), sequences=len(recs))
    out["ledger_reproduction"] = rep
    pre = {}
    for role, seqs in (("val", VAL), ("test", TESTSEQ)):
        for t in NEW["pastonly"] + NEW["abskv"] + ["uniform_past10"]:
            xs = [x for s in seqs for x in run.READ(LEDGERS / role / s.replace("/", "_") / (t + ".json"))["prefix"]]
            pre[f"{role}/{t}"] = dict(n=len(xs), lag=sorted({x["lag"] for x in xs}), all_winner_equal=all(x["winner_equal"] for x in xs),
                                      all_plan_equal=all(x["plan_equal"] for x in xs), max_score_error=max(x["score_error"] for x in xs),
                                      max_output_error=max(x["output_max_error"] for x in xs),
                                      max_plan_frame_minus_query=max(x["max_plan_frame_minus_query"] for x in xs))
    out["prefix_checks"] = pre
    tr = {}
    for variant in ("pastonly", "abskv", "sym"):
        for k in (4, 5, 6):
            p = RUNS / variant / f"s{k}" / "training_complete.json"
            if p.exists():
                r = run.READ(p)
                tr[f"{variant}@s{k}"] = dict(sha256=r["sha256"], local_sha256=run.HASH(RUNS / variant / f"s{k}" / f"u{U}.pth"),
                                             trainable_parameters=r["trainable_parameters"], total_parameters=r["total_parameters"],
                                             init_bitwise_equal_uniform=all(r["init_bitwise_equal_uniform"].values()),
                                             init_max_abs=max(r["init_max_abs_vs_uniform"].values()), seconds=r["seconds"], gpu=r["gpu"],
                                             final_loss=float(np.mean([x["loss"] for x in r["records"][-8:]])),
                                             strength_range=[min(x["strength_min"] for x in r["records"]), max(x["strength_max"] for x in r["records"])],
                                             reproduction=r["reproduction_vs_scratch_search"],
                                             val_fixed_AP=r["val_fixed"]["macro"]["AP"])
    out["training"] = tr
    out["fit"] = run.READ(FITS / "past10.json")
    out["conventions"] = dict(
        aggregation="KITTI 07(양성 없음) 제외 45시퀀스 -> 6도시 평균 -> 시드 평균, SD ddof=1. 방향별은 리뷰 방향 표지(group 0 순방향만, 1 역방향만).",
        intervals="a-b: 시퀀스별 (a 시드 평균 - b 시드 평균), 도시 안 시퀀스 재표집 10,000회, 도시 평균, 2.5/97.5% 분위, 쌍·묶음마다 난수 20261104 새로 시작(compare 스크립트와 같다).",
        metrics="test.*: _pr_eval_lib.pr_sweep(compare 스크립트 metric). metric_checks: 평가 모듈 집계·verify 스크립트 단순 구현과 대조.",
        units="0~1 비율(구간·차이는 %p)")
    P.write(OUT / "summary.json", out)
    for g, v in test.items():
        print("==", g)
        for arm, st in v["arms"].items():
            print(f"  {arm:16s}", {k: (round(100 * st[k]["mean"], 3), round(100 * (st[k]["sd"] or 0), 3)) for k in keys})
        for c, d in v["differences"].items():
            print(f"  {c:28s}", {k: (round(d["point_pp"][k], 3), [round(x, 3) for x in d["conditional_95_pp"][k]]) for k in keys})
    print("VAL", {a: {k: round(100 * v[k]["mean"], 3) for k in METRICS} for a, v in val.items()})
    print("CHECKS", {t: (c["max_abs_independent_vs_pipeline"], c["max_abs_compare_vs_pipeline"]) for t, c in checks.items()})
    print("REPRO", rep)
    print("PREFIX", pre)
    print("TRAIN", {k: (v["trainable_parameters"], v["init_bitwise_equal_uniform"], v["sha256"] == v["local_sha256"], v["reproduction"]) for k, v in tr.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action")
    p.add_argument("--variant", choices=list(VARIANTS))
    p.add_argument("--seed", type=int)
    p.add_argument("--role", default="val")
    p.add_argument("--sequence")
    p.add_argument("--models")
    p.add_argument("--prefix", type=int, default=3)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    P.e.configure()
    torch.set_num_threads(4)
    if a.action == "pools":
        pools()
    elif a.action == "fit":
        fit()
    elif a.action == "train":
        train(a.variant, a.seed)
    elif a.action == "pipe":
        pipe(a.role, a.sequence, Path(a.models).read_text().split(), a.prefix)
    elif a.action == "pipe_campaign":
        pipe_campaign(a.role, a.models, a.workers, a.prefix)
    elif a.action == "summary":
        summary()
    else:
        raise SystemExit(a.action)
