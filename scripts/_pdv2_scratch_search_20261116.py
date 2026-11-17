#!/usr/bin/env python3
"""처음부터(사전학습 없음) 한 단계 학습 구성 탐색.

시범(scripts/_pdv2_scratch_pilot_20261116.py)의 고정 설정·표본·손실 척도·검색 경로를 그대로 쓰고
입력 정규화, 정규화 층 처리, 학습 범위, 학습률·일정만 바꾼다. 계획은
results/pdv2_scratch_search_20261116/protocol.json에 먼저 적었다.

docker exec -w <repo> ring_baseline python3 scripts/_pdv2_scratch_search_20261116.py <action>
"""
from __future__ import annotations

import argparse
import json
import math
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
import _pdv2_scratch_pilot_20261116 as P  # noqa: E402
from gnn.spectral_difference import transport_difference  # noqa: E402

run, prior, pipeline = P.run, P.prior, P.pipeline
ROOT = P.ROOT
OUT = ROOT / "results/pdv2_scratch_search_20261116"
RUNS = OUT / "runs"
LEDGERS = OUT / "ledgers"
METRICS = P.METRICS
TRAIN, VAL, TESTSEQ = P.TRAIN, P.VAL, P.TESTSEQ
DEFAULT = dict(x="none", n="bn0", sc="all", g="free", lr="3e-4", sch="const", U="4096", B="64", wd="1e-5", do="off", sv="std")
ORDER = list(DEFAULT)


def parse(name: str) -> dict:
    """'x=logstd__n=ln' 꼴 이름을 설정으로 바꾼다. 빠진 항목은 시범 기본값이다."""
    cfg = dict(DEFAULT)
    for part in name.split("__"):
        k, v = part.split("=")
        assert k in cfg, k
        cfg[k] = v
    assert cfg["x"] in ("none", "log", "logstd", "lin")
    assert cfg["n"] in ("bn0", "bnest", "bnt", "ln")
    assert cfg["sc"] in ("all", "convs")
    assert cfg["g"] in ("free", "frozen", "weak")
    assert cfg["sch"] in ("const", "cos")
    assert cfg["do"] in ("off", "on") and cfg["sv"] in ("std", "fine")
    return cfg


def canon(name: str) -> str:
    cfg = parse(name)
    return "__".join(f"{k}={cfg[k]}" for k in ORDER if cfg[k] != DEFAULT[k]) or "x=none"


def model_tag(name: str, seed: int, u: int) -> str:
    return f"{canon(name)}@s{seed}@u{u}"


# ----------------------------------------------------------------- 모델
class XNorm(nn.Module):
    """스캔 하나의 입력만 쓰는 고정 변환. 통계는 학습8에서 구해 버퍼로 모델에 넣는다."""

    def __init__(self, mode: str, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.mode = mode
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "lin":
            return (x - self.mean) / self.std
        y = torch.log1p(x)
        return (y - self.mean) / self.std if self.mode == "logstd" else y


class WeakGate(P.ContrastSpectralDifference):
    """감산을 약하게만 할 수 있는 게이트: eta = max(-sigmoid(g), -0.5), 범위 [-0.5, 0).

    초기 g=0에서 -sigmoid(0) = -0.5라 초기 출력은 균등 감산과 비트 단위로 같다. clamp는 경계에서
    기울기를 그대로 넘기므로 첫 갱신부터 학습된다. 나머지 forward는 부모와 같다.
    """

    def forward(self, graph) -> torch.Tensor:
        h = F.relu(self.input_norm(self.input_proj(graph.x)))
        first = h
        attr = self.edge_encoder(graph.edge_attr, graph.edge_type)
        weights = []
        src, tgt = graph.edge_index
        for layer, (conv, norm) in enumerate(zip(self.convs, self.batch_norms)):
            previous = h
            h = norm(conv(h, graph.edge_index, edge_attr=attr))
            if layer == 0:
                h = F.relu(h)
            h = h + previous
            dense = torch.zeros(len(h), 11, 4, dtype=h.dtype, device=h.device)
            dense[tgt, src - tgt + 5] = conv.attention
            weights.append(dense)
        strengths = torch.clamp(-torch.stack((self.gate(first).sigmoid(), self.gate(h).sigmoid())), min=-0.5)
        self.last_strengths = strengths
        return transport_difference(graph.spectrum, graph.transport_indices, graph.transport_mask,
                                      graph.transport_phase, strengths, torch.stack(weights))


def xstats() -> dict:
    """학습8 시퀀스 균등 가중의 좌표별 평균·표준편차(원값과 log1p)."""
    path = OUT / "xstats.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    acc = {k: [] for k in ("lin_m", "lin_q", "log_m", "log_q")}
    for s in TRAIN:
        g = torch.load(P.recal.OUT / "train_inputs" / (s.replace("/", "_") + ".pt"), map_location="cpu", weights_only=False)
        x = g.x.double()
        assert float(x.min()) >= 0
        y = torch.log1p(x)
        acc["lin_m"].append(x.mean(0)); acc["lin_q"].append((x * x).mean(0))
        acc["log_m"].append(y.mean(0)); acc["log_q"].append((y * y).mean(0))
    out = {}
    for k in ("lin", "log"):
        m = torch.stack(acc[k + "_m"]).mean(0)
        q = torch.stack(acc[k + "_q"]).mean(0)
        out[k] = dict(mean=m.float(), std=(q - m * m).clamp_min(1e-12).sqrt().float())
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(out, tmp)
    tmp.replace(path)
    return out


def build(cfg: dict, seed: int, stats: dict = None) -> nn.Module:
    """시범과 같은 무작위 초기화·0 초기화 뒤 정규화 관련 부분만 바꾼다."""
    net = P.scratch_model(seed)
    if cfg["n"] == "ln":
        net.input_norm = nn.LayerNorm(128)
        net.batch_norms = nn.ModuleList([nn.LayerNorm(128), nn.LayerNorm(128)])
    if cfg["g"] == "weak":
        net.__class__ = WeakGate
    if cfg["x"] != "none":
        if stats is None:  # 체크포인트의 버퍼가 덮어쓴다
            z = torch.zeros(288)
            stats = dict(lin=dict(mean=z, std=z + 1), log=dict(mean=z, std=z + 1))
        st = stats["lin" if cfg["x"] == "lin" else "log"]
        net.input_proj = nn.Sequential(XNorm(cfg["x"], st["mean"], st["std"]), net.input_proj)
    return net.cuda().eval()


def load(name: str, seed: int, u: int) -> nn.Module:
    cfg = parse(name)
    net = build(cfg, seed)
    path = RUNS / canon(name) / f"s{seed}" / f"u{u}.pth"
    net.load_state_dict(torch.load(path, map_location="cuda", weights_only=False)["model_state_dict"], strict=True)
    return net


def trainable(cfg: dict, name: str) -> bool:
    if name.startswith("gate."):
        return cfg["g"] == "free"
    return cfg["sc"] == "all" or name.startswith("convs.")


def lr_at(cfg: dict, step: int) -> float:
    lr, total = float(cfg["lr"]), int(cfg["U"])
    if cfg["sch"] == "const":
        return lr
    warm = max(1, total // 20)
    if step < warm:
        return lr * (step + 1) / warm
    return lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def train(name: str, seed: int) -> None:
    cfg = parse(name)
    folder = RUNS / canon(name) / f"s{seed}"
    if (folder / "training_complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    _, lam = P.selected()
    fit = run.READ(P.OUT / "step2/loss_fit.json")["fit"]
    net = build(cfg, seed, xstats())
    bns = [m for m in net.modules() if isinstance(m, nn.BatchNorm1d)]
    assert bool(bns) == (cfg["n"] != "ln")
    if cfg["n"] == "bnest":
        P.estimate_bn(net)
    for n_, p in net.named_parameters():
        p.requires_grad_(trainable(cfg, n_))
    graphs = {s: torch.load(P.recal.OUT / "train_inputs" / (s.replace("/", "_") + ".pt"), map_location="cuda", weights_only=False) for s in TRAIN}
    with torch.no_grad():
        assert torch.equal(net(graphs[TRAIN[0]]), prior.filtered(graphs[TRAIN[0]], -0.5))
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    pools = {s: run.ARR(P.NEWPOOLS / "train" / s.replace("/", "_") / "pool.npz") for s in TRAIN}
    plans = {s: {k: torch.from_numpy(d["plan_" + k]).cuda() for k in ("query", "candidate", "weight")} for s, d in pools.items()}
    total, batch = int(cfg["U"]), int(cfg["B"])
    assert total % 64 == 0
    grid = (512, 1536, 4096) if cfg["sv"] == "std" else (64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096)
    saves = {total} if cfg["sch"] == "cos" else {u for u in grid if u <= total} | {total}
    if cfg["n"] == "bnt":
        for m in bns:
            m.train()
    drops = [m for m in net.modules() if isinstance(m, nn.Dropout)]
    if cfg["do"] == "on":  # 학습 중에만 dropout 0.1(주의 가중치·간선 부호기)을 켠다
        assert drops
        for m in drops:
            m.train()
    records, hashes, states, updates, tick = [], {}, {}, 0, time.time()
    for epoch in range(1, total // 64 + 1):
        for ordinal, s in enumerate(TRAIN):
            data = pools[s]
            sets = [np.flatnonzero(data["eligible"] & (data["correct"] == c)) for c in (0, 1)]
            sets = [v for v in sets if len(v)]
            rng = np.random.default_rng((seed, epoch, ordinal))
            losses, grads = [], []
            for _ in range(8):
                n = batch // len(sets)
                chosen = rng.permutation(np.concatenate([rng.choice(v, n, replace=len(v) < n) for v in sets]))
                ix = torch.from_numpy(chosen).cuda()
                for group in opt.param_groups:
                    group["lr"] = lr_at(cfg, updates)
                opt.zero_grad(set_to_none=True)
                z = net(graphs[s])
                value = P.score(z, {k: v[ix] for k, v in plans[s].items()}, lam)
                logit = (value - fit["mean"][0]) * fit["coefficients"][0] / fit["scale"][0] + fit["intercept"]
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
        if updates in saves:
            if cfg["n"] == "bnt":  # 추론용 고정 통계: 현재 가중치로 학습8 누적 평균(시퀀스 균등)
                P.estimate_bn(net)
            path = folder / f"u{updates}.pth"
            torch.save(dict(model_state_dict=net.state_dict(), updates=updates, cfg=cfg, seed=seed), path)
            if cfg["n"] == "bnt":  # 학습은 배치 통계를 쓰므로 누적 통계 교체는 학습에 영향이 없다
                for m in bns:
                    m.train()
            hashes[path.name] = run.HASH(path)
            states[updates] = path
            print("TRAIN", canon(name), seed, updates, round(time.time() - tick, 1), round(float(np.mean([r["loss"] for r in records[-8:]])), 5), flush=True)
    seconds = time.time() - tick
    changed = [k for k, v in net.state_dict().items() if not torch.equal(v, initial[k])]
    del graphs, plans, opt
    # 원키 승자를 고정한 검증9 AP
    tick2 = time.time()
    per = {u: {} for u in states}
    for s in VAL:
        vname = s.replace("/", "_")
        graph, _, _ = prior.graph_inputs("val", s)
        data = run.ARR(P.NEWPOOLS / "val" / vname / "pool.npz")
        row = {k: data[k] for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist")}
        plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
        with torch.no_grad():
            for u in states:
                z = load(name, seed, u)(graph)
                v = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(row["query_idx"])).astype(np.float64)
                per[u][s] = run.base.independent_metrics(row, v)
        del graph
    val = {str(u): dict(macro=run.base.macro(v)["macro"], families=run.base.macro(v)["families"], sequences=v) for u, v in per.items()}
    P.write(folder / "training_complete.json", dict(
        status="passed", cfg=cfg, name=canon(name), seed=seed, updates=updates, checkpoints=hashes, records=records, changed=changed,
        trainable_parameters=sum(p.numel() for p in params), total_parameters=sum(p.numel() for p in net.parameters()),
        val_fixed=val, seconds=seconds, val_seconds=time.time() - tick2))
    print("VALFIXED", canon(name), seed, {u: round(100 * v["macro"]["AP"], 3) for u, v in val.items()}, flush=True)


# ----------------------------------------------------------------- 실제 검색
def pipe(role: str, sequence: str, models: list, nprefix: int) -> None:
    """matching_only 실제 검색(시범 d3_pipe와 같은 경로). 모델별 원장을 따로 남긴다."""
    name = sequence.replace("/", "_")
    folder = LEDGERS / role / name
    todo = [m for m in models if not (folder / (m + ".json")).exists()]
    if not todo:
        return
    folder.mkdir(parents=True, exist_ok=True)
    src = (P.PIPE / "val" / name) if role == "val" else (P.TEST / "test" / name)
    cache, pos, minimum, travel, bank, _, _, _ = P.inputs(role, sequence)
    graph, _, _ = prior.graph_inputs(role, sequence)
    raw = pipeline.raw_key(cache["descriptors"])
    new_w, new_lam = P.selected()
    poolpath = src / "candidate_pool.npz"
    assert run.HASH(poolpath) == run.READ(poolpath.with_suffix(".json"))["sha256"]
    base_pool = run.ARR(poolpath)
    static = []
    for n, q in enumerate(base_pool["query"]):
        lo, hi = base_pool["offset"][n:n + 2]
        ids = base_pool["ids"][lo:hi]
        static.append(new_w[0] * run.base.minmax(raw[ids] @ raw[q]) + new_w[1] * run.base.minmax(bank[0][ids] @ bank[0][q]))
    new_pool = dict(base_pool, static=np.concatenate(static).astype(np.float32))
    small = {}
    for m in todo:
        tick = time.time()
        ref = None
        weights, lam, pool = new_w, new_lam, new_pool
        with torch.no_grad():
            if m == "uniform":
                net = None
                z = prior.filtered(graph, -0.5)
            elif m == "repro_fixed":
                net, ref = None, "fixed"
                weights, lam, pool = P.CURRENT_W, 0.25, base_pool
                z = prior.filtered(graph, -0.5)
            elif m == "repro_diff_s1":
                net, ref = pipeline.load_model(P.SPEC["cases"]["diff_s1"]), "diff_s1"
                weights, lam, pool = P.CURRENT_W, 0.25, base_pool
                z = net(graph)
            else:
                cname, seed, u = m.split("@")
                net = load(cname, int(seed[1:]), int(u[1:]))
                z = net(graph)
            change = run.base.variant_inputs(z.cpu().numpy(), 8)
            q = pool["query"]
            j = pipeline.winners(pool, change, weights)
            assert (j <= q - 30).all()
            row = dict(query_idx=q, j_star=j, min_past_gt_dist=minimum[q],
                       top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1), group=np.zeros(len(q), np.int64))
            plan = prior.base.prior.plan(travel, q, j)
            value = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(q)).astype(np.float64)
            pre = []
            for k in range(nprefix if net is not None and ref is None else 0):
                at = (k + 1) * len(q) // (nprefix + 1)
                t = int(q[at])
                end = t + 16
                if end not in small:
                    small[end] = prior.graph_inputs(role, sequence, end)[0]
                short = net(small[end])
                sb = run.base.variant_inputs(short.cpu().numpy(), 8)
                winner = run.cal.candidates(raw[:end], (bank[0][:end], *sb[1:]), weights[None], t)
                pp = {kk: torch.from_numpy(v).cuda() for kk, v in prior.base.prior.plan(travel[:end], np.array([t]), winner).items()}
                chk = P.score(short, pp, lam).cpu().numpy()
                pre.append(dict(query=t, winner_equal=int(winner[0]) == int(j[at]), score_error=float(abs(chk - value[at]).max()),
                                output_max_error=float((short[: t + 1] - z[: t + 1]).abs().max())))
        metric = run.base.independent_metrics(row, value)
        other = run.white.prior.BASE.exp.metrics(value, row)
        for k in METRICS:
            if metric[k] is not None:
                np.testing.assert_allclose(metric[k], other[k], atol=1e-12, rtol=0)
        rec = dict(metrics=metric, prefix=pre, seconds=time.time() - tick)
        if ref:
            r = run.ARR(src / (ref + ".npz"))
            rec["reproduction"] = dict(
                winners_equal=bool(np.array_equal(j, r["j_star"])), query_equal=bool(np.array_equal(q, r["query_idx"])),
                score_bitwise_equal=bool(np.array_equal(value, r["score"])), score_max_error=float(np.abs(value - r["score"]).max()),
                metrics_equal=all(metric[k] == run.READ(src / (ref + ".json"))["metrics"][k] for k in METRICS))
        np.savez_compressed(folder / (m + ".npz"), **row, score=value)
        P.write(folder / (m + ".json"), rec)
        del net, z, change
    print("PIPE", role, sequence, len(todo), flush=True)


def eta(models: list) -> None:
    """학습된 감산 세기 eta의 분포(층별, 검증9·시험46 스캔 전체)."""
    path = OUT / "eta.json"
    out = run.READ(path) if path.exists() else {}
    nets = {m: load(m.split("@")[0], int(m.split("@")[1][1:]), int(m.split("@")[2][1:])) for m in models if m not in out}
    acc = {m: {"val": [], "test": []} for m in nets}
    for role, seqs in (("val", VAL), ("test", TESTSEQ)):
        for s in seqs:
            graph, _, _ = prior.graph_inputs(role, s)
            with torch.no_grad():
                for m, net in nets.items():
                    net(graph)
                    acc[m][role].append(net.last_strengths[..., 0].cpu().numpy())
            del graph
    qs = (0.01, 0.1, 0.5, 0.9, 0.99)
    for m in nets:
        out[m] = {}
        for role in ("val", "test"):
            v = np.concatenate(acc[m][role], 1)
            out[m][role] = dict(
                mean=v.mean(1).tolist(), quantiles={str(q): np.quantile(v, q, axis=1).tolist() for q in qs},
                frac_stronger_than_half=(v < -0.5 - 1e-6).mean(1).tolist(), frac_weaker_than_half=(v > -0.5 + 1e-6).mean(1).tolist(),
                sequence_mean_min=[float(min(a[i].mean() for a in acc[m][role])) for i in (0, 1)],
                sequence_mean_max=[float(max(a[i].mean() for a in acc[m][role])) for i in (0, 1)])
        print(m, "test mean", [round(x, 3) for x in out[m]["test"]["mean"]], "stronger", [round(x, 3) for x in out[m]["test"]["frac_stronger_than_half"]],
              "q01/q50/q99 L1", [round(out[m]["test"]["quantiles"][q][0], 3) for q in ("0.01", "0.5", "0.99")],
              "L2", [round(out[m]["test"]["quantiles"][q][1], 3) for q in ("0.01", "0.5", "0.99")], flush=True)
    P.write(path, out)


def collect(role: str, model: str):
    seqs = VAL if role == "val" else TESTSEQ
    per = {}
    for s in seqs:
        path = LEDGERS / role / s.replace("/", "_") / (model + ".json")
        if not path.exists():
            return None
        per[s] = run.READ(path)
    return per


def macro(role: str, per: dict) -> dict:
    m = {s: v["metrics"] for s, v in per.items()}
    if role == "val":
        return run.base.macro(m)["macro"]
    return run.current.adoption.previous.macro(m, run.current.CITY)["macro"]


def agg() -> None:
    """원장이 다 있는 모델의 검증9·시험46 지표와 구성별 시드 평균±SD."""
    out = {}
    for role in ("val", "test"):
        first = LEDGERS / role / (VAL if role == "val" else TESTSEQ)[0].replace("/", "_")
        models = sorted(p.stem for p in first.glob("*.json")) if first.exists() else []
        res, groups = {}, {}
        for m in models:
            per = collect(role, m)
            if per is None:
                continue
            pre = [x for v in per.values() for x in v["prefix"]]
            res[m] = dict(macro=macro(role, per), prefix_n=len(pre), prefix_all_winner_equal=all(x["winner_equal"] for x in pre),
                          prefix_max_score_error=max([x["score_error"] for x in pre], default=0.0))
            if "reproduction" in next(iter(per.values())):
                rp = [v["reproduction"] for v in per.values()]
                res[m]["reproduction_exact"] = all(r["winners_equal"] and r["score_bitwise_equal"] and r["metrics_equal"] for r in rp)
            if "@" in m:
                c, s, u = m.split("@")
                groups.setdefault(c + "@" + u, {})[int(s[1:])] = res[m]["macro"]
        summ = {}
        for gname, by in groups.items():
            seeds = sorted(by)
            summ[gname] = dict(seeds=seeds, **{k: dict(mean=float(np.mean([by[s][k] for s in seeds])),
                                                         sd=float(np.std([by[s][k] for s in seeds], ddof=1)) if len(seeds) > 1 else None,
                                                         values=[by[s][k] for s in seeds]) for k in METRICS})
        out[role] = dict(models=res, groups=summ)
        print("==", role)
        for m in ("uniform", "repro_fixed", "repro_diff_s1"):
            if m in res:
                print(f"{m:60s}", {k: round(100 * res[m]["macro"][k], 3) for k in METRICS}, res[m].get("reproduction_exact"))
        for gname, v in sorted(summ.items(), key=lambda kv: -kv[1]["AP"]["mean"]):
            print(f"{gname:60s} n={len(v['seeds'])}", {k: (round(100 * v[k]["mean"], 3), round(100 * (v[k]["sd"] or 0), 3)) for k in METRICS})
    P.write(OUT / "agg.json", out)


def valtable() -> None:
    """학습 폴더의 검증9 고정 승자 AP를 구성별로 모은다."""
    rows = {}
    for path in sorted(RUNS.glob("*/s*/training_complete.json")):
        rec = run.READ(path)
        for u, v in rec["val_fixed"].items():
            rows.setdefault((rec["name"], int(u)), {})[rec["seed"]] = v["macro"]
    table = []
    for (name, u), by in rows.items():
        seeds = sorted(by)
        ap = [100 * by[s]["AP"] for s in seeds]
        r1 = [100 * by[s]["R@1"] for s in seeds]
        table.append(dict(name=name, u=u, seeds=seeds, AP=ap, AP_mean=float(np.mean(ap)), AP_min=float(np.min(ap)),
                          AP_sd=float(np.std(ap, ddof=1)) if len(ap) > 1 else None,
                          F1_mean=float(np.mean([100 * by[s]["max_F1"] for s in seeds])),
                          R100P_mean=float(np.mean([100 * by[s]["R@100P"] for s in seeds])), R1_mean=float(np.mean(r1))))
    table.sort(key=lambda r: -r["AP_mean"])
    ref = run.READ(P.OUT / "step2/val_fixed_winner.json")["results"]
    P.write(OUT / "val_fixed_table.json", dict(uniform=ref["uniform"]["macro"], current_diff_s1=ref["current_diff_s1"]["macro"], rows=table))
    print("uniform", {k: round(100 * ref["uniform"]["macro"][k], 3) for k in METRICS})
    for r in table:
        print(f"{r['name']:58s} u{r['u']:<5d} n={len(r['seeds'])} AP {r['AP_mean']:.3f} min {r['AP_min']:.3f} sd {(r['AP_sd'] or 0):.3f} "
              f"F1 {r['F1_mean']:.3f} R100P {r['R100P_mean']:.3f} | " + " ".join(f"{a:.2f}" for a in r["AP"]))


# ----------------------------------------------------------------- 병렬 실행
def fanout(commands: list, gpus: list, per_gpu: int, logdir: Path) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    slots = [g for g in gpus for _ in range(per_gpu)]
    running, queue, failed = {}, list(enumerate(commands)), []
    while queue or running:
        while queue and slots:
            i, (label, cmd) = queue.pop(0)
            g = slots.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            f = (logdir / (label + ".log")).open("a")
            running[subprocess.Popen([sys.executable, __file__, *cmd], stdout=f, stderr=subprocess.STDOUT, env=env)] = (g, label, f)
        time.sleep(1)
        for proc in list(running):
            if proc.poll() is not None:
                g, label, f = running.pop(proc)
                f.close()
                slots.append(g)
                if proc.returncode:
                    failed.append(label)
                print("DONE" if not proc.returncode else "FAIL", label, flush=True)
    assert not failed, failed


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action")
    p.add_argument("--cfg")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--role", default="val")
    p.add_argument("--sequence")
    p.add_argument("--models")
    p.add_argument("--jobs")
    p.add_argument("--gpus", default="0")
    p.add_argument("--per-gpu", type=int, default=1)
    p.add_argument("--prefix", type=int, default=1)
    p.add_argument("--stamp", default="x")
    a = p.parse_args()
    P.e.configure()
    torch.set_num_threads(4)
    gpus = [int(x) for x in a.gpus.split(",")]
    if a.action == "xstats":
        print({k: {kk: (float(vv.min()), float(vv.max())) for kk, vv in v.items()} for k, v in xstats().items()})
    elif a.action == "train":
        train(a.cfg, a.seed)
    elif a.action == "train_campaign":
        tick = time.time()
        xstats()
        jobs = [line.split() for line in Path(a.jobs).read_text().splitlines() if line.strip()]
        fanout([(f"{canon(c)}@s{s}", ["train", "--cfg", c, "--seed", s]) for c, s in jobs], gpus, a.per_gpu, OUT / "logs/train")
        P.write(OUT / "logs" / f"train_campaign_{a.stamp}.done.json", dict(jobs=len(jobs), seconds=time.time() - tick))
    elif a.action == "pipe":
        pipe(a.role, a.sequence, Path(a.models).read_text().split(), a.prefix)
    elif a.action == "pipe_campaign":
        tick = time.time()
        seqs = VAL if a.role == "val" else TESTSEQ
        fanout([(f"{a.role}_{s.replace('/', '_')}", ["pipe", "--role", a.role, "--sequence", s, "--models", a.models, "--prefix", str(a.prefix)])
                for s in seqs], gpus, a.per_gpu, OUT / "logs/pipe")
        P.write(OUT / "logs" / f"pipe_campaign_{a.role}_{a.stamp}.done.json", dict(models=Path(a.models).read_text().split(), seconds=time.time() - tick))
    elif a.action == "eta":
        eta(Path(a.models).read_text().split())
    elif a.action == "agg":
        agg()
    elif a.action == "valtable":
        valtable()
    else:
        raise SystemExit(a.action)
