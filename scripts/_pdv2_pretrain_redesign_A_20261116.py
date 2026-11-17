#!/usr/bin/env python3
"""현재 시간 모듈(diff)의 두 단계 학습 재설계 A, 1시드.

1단계: 입력 투영·간선 부호화·두 DiffAttnConv를 장소 대조(InfoNCE)로 학습한다(h2 → 투영 머리, 머리는 버림).
2단계: Q 투영·기하 최종층·게이트 최종층을 0으로 되돌려 균등 감산에서 시작하고, 시범과 같은 BCE로 학습한다.

docker exec -w <repo> ring_baseline python3 scripts/_pdv2_pretrain_redesign_A_20261116.py <action>
규칙은 results/pdv2_pretrain_redesign_20261116/A/protocol.json에 결과 전에 적었다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import _pdv2_scratch_pilot_20261116 as P  # noqa: E402

run, prior, pipeline = P.run, P.prior, P.pipeline
BASE = ROOT / "results/pdv2_pretrain_redesign_20261116/A"
OUT = BASE if P.SEED == 1 else BASE / f"s{P.SEED}"  # 시드 2·3(조정자 추가 요청, protocol amendment_1)
PREPARED = ROOT / "results/pdv2_spectral_keys_20261025/prepared_octave288.pt"
TRAIN_GRAPHS = P.recal.OUT / "train_inputs"
TRAIN, VAL, TESTSEQ = P.TRAIN, P.VAL, P.TESTSEQ
SEED = P.SEED
METRICS = P.METRICS
S1 = dict(updates=900, warmup=135, lr=5e-4, weight_decay=1e-5, temperature=0.1, pairs=4096, chunk=1024,
          pos_radius=5.0, min_gap=30, neg_mask_radius=10.0, clip=1.0)
LRS = {"lr3e-4": 3e-4, "lr1e-4": 1e-4}
PRIMARY = "A_lr3e-4_u4096"
EVAL_CASES = ("A_lr3e-4_u4096", "A_lr3e-4_u1536", "A_lr1e-4_u4096", "A_lr1e-4_u1536") if P.SEED == 1 else ("A_lr3e-4_u4096", "A_lr3e-4_u1536")
write = P.write


def timing(step: str, seconds: float) -> None:
    path = OUT / "timing.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault(step, []).append(round(seconds, 1))
    write(path, data)


def load_train(s: str):
    return torch.load(TRAIN_GRAPHS / (s.replace("/", "_") + ".pt"), map_location="cuda", weights_only=False)


def concat_train():
    """학습8 그래프를 시간 간선 그대로 이어 붙인다(시퀀스 사이 간선 없음)."""
    xs, eis, eas, ets, off = [], [], [], [], [0]
    for s in TRAIN:
        g = load_train(s)
        src, tgt = g.edge_index
        assert bool((g.edge_type == 0).all()) and int((src - tgt).abs().max()) <= 5
        xs.append(g.x); eis.append(g.edge_index + off[-1]); eas.append(g.edge_attr); ets.append(g.edge_type)
        off.append(off[-1] + len(g.x))
        del g
    return SimpleNamespace(x=torch.cat(xs), edge_index=torch.cat(eis, 1), edge_attr=torch.cat(eas),
                           edge_type=torch.cat(ets)), np.asarray(off)


def node_features(net, graph) -> torch.Tensor:
    """ContrastSpectralDifference.forward의 h2 계산과 같은 순서(src/gnn/contrast_spectral_difference.py:16-31)."""
    h = F.relu(net.input_norm(net.input_proj(graph.x)))
    attr = net.edge_encoder(graph.edge_attr, graph.edge_type)
    for layer, (conv, norm) in enumerate(zip(net.convs, net.batch_norms)):
        previous = h
        h = norm(conv(h, graph.edge_index, edge_attr=attr))
        if layer == 0:
            h = F.relu(h)
        h = h + previous
    return h


def bns(net):
    return [m for m in net.modules() if isinstance(m, nn.BatchNorm1d)]


def zero_layers(net) -> None:
    """LearnedSpectralDifference.__init__의 0 초기화와 같은 세 층."""
    with torch.no_grad():
        for conv in net.convs:
            conv.W_q.weight.zero_()
            if conv.W_q.bias is not None:
                conv.W_q.bias.zero_()
            final = [v for v in conv.bias_mlp.modules() if isinstance(v, nn.Linear)][-1]
            final.weight.zero_(); final.bias.zero_()
        last = [v for v in net.gate.modules() if isinstance(v, nn.Linear)][-1]
        last.weight.zero_(); last.bias.zero_()


# ----------------------------------------------------------------- 준비
def prep() -> None:
    """양성(GT 자세 5 m 안·30키프레임 이상)과 음성 마스크 그룹을 만든다."""
    tick = time.time()
    state = torch.load(PREPARED, map_location="cpu", weights_only=False)
    seqs = np.asarray(state["sequences"])
    poses = np.asarray(state["poses"])
    pos_all, group, ptr, idx, rec = [], [], [0], [], {}
    off = 0
    intra = state.get("intra")
    for i, s in enumerate(TRAIN):
        m = seqs == i
        g = load_train(s)
        x_equal = bool(torch.equal(g.x.cpu(), torch.from_numpy(np.asarray(state["descriptors"])[m])))
        assert x_equal, s
        n = len(g.x)
        assert n == int(m.sum())
        del g
        p = poses[m][:, :3, 3]
        pt = torch.from_numpy(p).cuda()
        frame = torch.arange(n, device="cuda")
        cnt = 0
        for lo in range(0, n, 2048):
            d = torch.cdist(pt[lo:lo + 2048], pt)
            ok = (d <= S1["pos_radius"]) & ((frame[lo:lo + 2048, None] - frame[None]).abs() >= S1["min_gap"])
            for r in range(len(ok)):
                j = torch.nonzero(ok[r]).flatten().cpu().numpy() + off
                idx.append(j); ptr.append(ptr[-1] + len(j)); cnt += len(j) > 0
        pos_all.append(p)
        group += [0 if s.startswith("helipr/") else i + 1] * n  # HeLiPR Bridge는 같은 좌표계를 공유
        rec[s] = dict(nodes=n, anchors=cnt, x_equal_prepared=x_equal)
        off += n
    pos_all = np.concatenate(pos_all)
    ptr = np.asarray(ptr, np.int64)
    idx = np.concatenate(idx).astype(np.int64)
    anchors = np.flatnonzero(np.diff(ptr) > 0)
    # 기존 사전학습 양성 풀과 대조(있으면)
    cmp = None
    if isinstance(intra, list) and len(intra) == len(seqs):
        same = sum(int(np.array_equal(np.sort(np.asarray(intra[a], np.int64)), idx[ptr[a]:ptr[a + 1]])) for a in range(len(seqs)))
        cmp = dict(nodes_identical=same, nodes=len(seqs))
    np.savez_compressed(OUT / "stage1/positives.npz", pos=pos_all, group=np.asarray(group), ptr=ptr, idx=idx, anchors=anchors)
    write(OUT / "stage1/positives.json", dict(sequences=rec, anchors=len(anchors), positive_pairs=len(idx),
                                             compare_prepared_intra=cmp, seconds=time.time() - tick))
    print(rec, len(anchors), cmp)


# ----------------------------------------------------------------- 1단계
def stage1() -> None:
    folder = OUT / "stage1"
    if (folder / "training_complete.json").exists():
        return
    tick = time.time()
    d = run.ARR(BASE / "stage1/positives.npz")  # 시드와 무관
    pos_t = torch.from_numpy(d["pos"]).cuda()
    grp_t = torch.from_numpy(d["group"]).cuda()
    ptr, idx, anchors = d["ptr"], d["idx"], d["anchors"]
    graph, _ = concat_train()
    net = P.scratch_model(SEED)  # torch.manual_seed(SEED) 뒤 무작위 초기화 + 세 층 0
    head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128)).cuda()  # 같은 RNG 흐름에서 이어서 초기화
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    trained = [p for n, p in net.named_parameters() if not n.startswith("gate.")]
    params = trained + list(head.parameters())
    opt = torch.optim.Adam(params, lr=S1["lr"], weight_decay=S1["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: (u + 1) / S1["warmup"] if u < S1["warmup"] else
                                              0.5 * (1 + math.cos(math.pi * (u - S1["warmup"]) / (S1["updates"] - S1["warmup"]))))
    net.eval(); head.train()
    for m in bns(net):
        m.train()  # 1단계 BN은 배치(학습8 전체 노드) 통계
    records = []
    for u in range(S1["updates"]):
        rng = np.random.default_rng((SEED, 7001, u))
        a = rng.choice(anchors, S1["pairs"], replace=False)
        cnt = ptr[a + 1] - ptr[a]
        p = idx[ptr[a] + rng.integers(0, cnt)]
        h = node_features(net, graph)
        ea = F.normalize(head(h[torch.from_numpy(a).cuda()]), dim=1)
        ep = F.normalize(head(h[torch.from_numpy(p).cuda()]), dim=1)
        at, ptt = torch.from_numpy(a).cuda(), torch.from_numpy(p).cuda()
        losses, accs, masked = [], [], 0
        for lo in range(0, S1["pairs"], S1["chunk"]):
            sl = slice(lo, lo + S1["chunk"])
            logits = ea[sl] @ ep[sl].T / S1["temperature"]
            same = (grp_t[at[sl]][:, None] == grp_t[ptt[sl]][None]) & (torch.cdist(pos_t[at[sl]], pos_t[ptt[sl]]) <= S1["neg_mask_radius"])
            same.fill_diagonal_(False)
            masked += int(same.sum())
            logits = logits.masked_fill(same, float("-inf"))
            target = torch.arange(len(logits), device="cuda")
            losses.append(F.cross_entropy(logits, target))
            accs.append(float((logits.argmax(1) == target).float().mean()))
        loss = torch.stack(losses).mean()
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g = torch.nn.utils.clip_grad_norm_(params, S1["clip"])
        assert torch.isfinite(g)
        opt.step(); sched.step()
        records.append(dict(update=u + 1, loss=float(loss.detach()), acc=float(np.mean(accs)), grad=float(g),
                            masked_per_chunk=masked / 4, lr=opt.param_groups[0]["lr"]))
        if (u + 1) % 50 == 0 or u == 0:
            print("S1", u + 1, round(records[-1]["loss"], 4), round(records[-1]["acc"], 3), round(time.time() - tick, 1), flush=True)
    # BN 누적 통계: 1단계 최종 가중치로 학습8 전체 노드(한 번, 노드 가중)에서 정확히 추정해 고정
    for m in bns(net):
        m.reset_running_stats(); m.momentum = None; m.train()
    with torch.no_grad():
        node_features(net, graph)
    for m in bns(net):
        m.momentum = 0.1; m.eval()
    net.eval()
    path = folder / "model.pth"
    torch.save(dict(model_state_dict=net.state_dict(), head_state_dict=head.state_dict(), seed=SEED, config=S1), path)
    changed = [k for k, v in net.state_dict().items() if not torch.equal(v, initial[k])]
    write(folder / "training_complete.json", dict(
        status="passed", seed=SEED, config=S1, checkpoint=run.HASH(path), records=records, changed=changed,
        trainable_parameters_net=sum(p.numel() for p in trained), head_parameters=sum(p.numel() for p in head.parameters()),
        bn_batches=[int(m.num_batches_tracked) for m in bns(net)], seconds=time.time() - tick))
    timing("stage1", time.time() - tick)


# ----------------------------------------------------------------- 2단계
def stage2_init() -> None:
    """1단계 가중치에서 세 층을 0으로 → 균등 감산과 비트 단위 같음 확인, 척도 적합."""
    tick = time.time()
    net = P.scratch_model(SEED)
    net.load_state_dict(torch.load(OUT / "stage1/model.pth", map_location="cuda", weights_only=False)["model_state_dict"], strict=True)
    # h2 함수가 forward 내부 h2와 같은지: 게이트 최종층을 임시 무작위로 두고 둘째 세기 비교
    g0 = load_train(TRAIN[2])
    with torch.no_grad():
        last = net.gate[2]
        last.weight.normal_(); last.bias.normal_()
        net(g0)
        h2_check = float((net.last_strengths[1] + net.gate(node_features(net, g0)).sigmoid()).abs().max())
    zero_layers(net)
    path = OUT / "stage2/init.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_state_dict=net.state_dict(), seed=SEED), path)
    rec = dict(h2_function_vs_forward_max_abs=h2_check)
    with torch.no_grad():
        for role, s in (("train", TRAIN[0]), ("train", TRAIN[4]), ("val", "kitti360/0004"), ("test", "kitti/00")):
            graph = load_train(s) if role == "train" else prior.graph_inputs(role, s)[0]
            z = net(graph)
            fixed = prior.filtered(graph, -0.5)
            rec[f"{role}:{s}"] = dict(bitwise_equal_uniform=bool(torch.equal(z, fixed)), max_abs=float((z - fixed).abs().max()),
                                      strengths_all_minus_half=bool(torch.all(net.last_strengths == -0.5)))
            del graph
    assert all(v["bitwise_equal_uniform"] for k, v in rec.items() if ":" in k)
    # 양의 아핀 척도(학습8): 초기 출력이 균등 감산과 같으므로 시범 적합과 같아야 한다
    _, lam = P.selected()
    xs, ys, ws = [], [], []
    for s in TRAIN:
        name = s.replace("/", "_")
        data = run.ARR(P.NEWPOOLS / "train" / name / "pool.npz")
        graph = load_train(s)
        plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
        with torch.no_grad():
            z = net(graph)
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
    fit = P.fit_score(x, y, w / w.sum())
    old = run.READ(P.OUT / "step2/loss_fit.json")["fit"]
    rec["loss_fit"] = fit
    rec["loss_fit_equals_pilot"] = bool(json.dumps(fit, sort_keys=True) == json.dumps(old, sort_keys=True))
    rec["init_checkpoint"] = run.HASH(path)
    write(OUT / "stage2/init_identity.json", rec)
    timing("stage2_init", time.time() - tick)
    print(json.dumps(rec, indent=1)[:2000])


def stage2(tag: str) -> None:
    """시범 train()과 같은 표본·순서·손실, 출발점만 2단계 초기 모델(BN 고정 통계 포함)."""
    lr = LRS[tag]
    folder = OUT / "stage2" / tag
    if (folder / "training_complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    _, lam = P.selected()
    fit = run.READ(OUT / "stage2/init_identity.json")["loss_fit"]
    net = P.load_scratch(OUT / "stage2/init.pth")
    with torch.no_grad():
        g0 = load_train(TRAIN[0])
        assert torch.equal(net(g0), prior.filtered(g0, -0.5))
        del g0
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    graphs = {s: load_train(s) for s in TRAIN}
    pools = {s: run.ARR(P.NEWPOOLS / "train" / s.replace("/", "_") / "pool.npz") for s in TRAIN}
    plans = {s: {k: torch.from_numpy(d["plan_" + k]).cuda() for k in ("query", "candidate", "weight")} for s, d in pools.items()}
    records, hashes, updates, tick = [], {}, 0, time.time()
    for epoch in range(1, max(P.SAVE_EPOCHS) + 1):
        for ordinal, s in enumerate(TRAIN):
            data = pools[s]
            sets = [np.flatnonzero(data["eligible"] & (data["correct"] == c)) for c in (0, 1)]
            sets = [v for v in sets if len(v)]
            rng = np.random.default_rng((SEED, epoch, ordinal))
            losses, grads = [], []
            for _ in range(8):
                n = 64 // len(sets)
                chosen = rng.permutation(np.concatenate([rng.choice(v, n, replace=len(v) < n) for v in sets]))
                ix = torch.from_numpy(chosen).cuda()
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
                losses.append(float(loss.detach())); grads.append(float(g))
            records.append(dict(epoch=epoch, sequence=s, loss=float(np.mean(losses)), grad=float(np.mean(grads)),
                                strength_mean=float(net.last_strengths.detach().mean())))
        if epoch in P.SAVE_EPOCHS:
            assert updates == P.SAVE_EPOCHS[epoch]
            path = folder / f"u{updates}.pth"
            torch.save(dict(model_state_dict=net.state_dict(), epoch=epoch, updates=updates, lr=lr, seed=SEED), path)
            hashes[path.name] = run.HASH(path)
            print("TRAIN", tag, epoch, updates, round(time.time() - tick, 1), np.mean([r["loss"] for r in records[-8:]]), flush=True)
    changed = [k for k, v in net.state_dict().items() if not torch.equal(v, initial[k])]
    write(folder / "training_complete.json", dict(
        status="passed", lr=lr, seed=SEED, updates=updates, checkpoints=hashes, records=records, changed=changed,
        trainable_parameters=sum(p.numel() for p in params), seconds=time.time() - tick))
    timing("stage2_" + tag, time.time() - tick)


def ckpt(case: str) -> Path:
    tag, u = case[2:].rsplit("_u", 1)
    return OUT / "stage2" / tag / f"u{u}.pth"


def val_fixed() -> None:
    """원키 승자를 고정한 검증9 AP(시범 val_fixed와 같은 표본)."""
    tick = time.time()
    _, lam = P.selected()
    cases = {"uniform": None, "A_init_u0": OUT / "stage2/init.pth"}
    for tag in (LRS if P.SEED == 1 else ["lr3e-4"]):
        for u in P.SAVE_EPOCHS.values():
            cases[f"A_{tag}_u{u}"] = OUT / "stage2" / tag / f"u{u}.pth"
    per = {k: {} for k in cases}
    for s in VAL:
        graph, _, _ = prior.graph_inputs("val", s)
        data = run.ARR(P.NEWPOOLS / "val" / s.replace("/", "_") / "pool.npz")
        row = {k: data[k] for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist")}
        plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
        with torch.no_grad():
            for tag, ck in cases.items():
                z = prior.filtered(graph, -0.5) if ck is None else P.load_scratch(ck)(graph)
                v = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(row["query_idx"])).astype(np.float64)
                per[tag][s] = run.base.independent_metrics(row, v)
    res = {k: run.base.macro(v) for k, v in per.items()}
    pilot = run.READ(P.OUT / "step2/val_fixed_winner.json")["results"]["uniform"]["macro"]
    write(OUT / "stage2/val_fixed_winner.json", dict(results=res, per_sequence=per, lam=lam,
                                                     uniform_equals_pilot=res["uniform"]["macro"] == pilot, test_used=False))
    timing("val_fixed", time.time() - tick)
    for k in cases:
        print(k, {m: round(100 * res[k]["macro"][m], 3) for m in METRICS})


# ----------------------------------------------------------------- 평가
def pipe(role: str, sequence: str) -> None:
    name = sequence.replace("/", "_")
    folder = OUT / "eval" / role / name
    if (folder / "summary.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    src = (P.PIPE / "val" / name) if role == "val" else (P.TEST / "test" / name)
    pilot_ledger = P.OUT / "step3" / role / name / "uniform_reselected.npz"
    cache, pos, minimum, travel, bank, _, _, _ = P.inputs(role, sequence)
    graph, _, _ = prior.graph_inputs(role, sequence)
    raw = pipeline.raw_key(cache["descriptors"])
    poolpath = src / "candidate_pool.npz"
    assert run.HASH(poolpath) == run.READ(poolpath.with_suffix(".json"))["sha256"]
    base_pool = run.ARR(poolpath)
    w_sel, lam_sel = P.selected()
    static = []
    for n, q in enumerate(base_pool["query"]):
        lo, hi = base_pool["offset"][n:n + 2]
        ids = base_pool["ids"][lo:hi]
        static.append(w_sel[0] * run.base.minmax(raw[ids] @ raw[q]) + w_sel[1] * run.base.minmax(bank[0][ids] @ bank[0][q]))
    sel_pool = dict(base_pool, static=np.concatenate(static).astype(np.float32))
    cases = {"repro_diff_s1": ("current", P.CURRENT_W, 0.25, base_pool),
             "repro_uniform_reselected": ("fixed", w_sel, lam_sel, sel_pool)}
    for c in EVAL_CASES:
        cases[c] = ("A", w_sel, lam_sel, sel_pool)
    if P.SEED != 1:  # 재현 대조는 시드 1에서만
        cases = {k: v for k, v in cases.items() if not k.startswith("repro_")}
    metrics, checks = {}, {}
    for tag, (kind, weights, lam, pool) in cases.items():
        weights = np.asarray(weights, np.float32)
        with torch.no_grad():
            if kind == "fixed":
                net, z = None, prior.filtered(graph, -0.5)
            elif kind == "current":
                net = pipeline.load_model(P.SPEC["cases"]["diff_s1"]); z = net(graph)
            else:
                net = P.load_scratch(ckpt(tag)); z = net(graph)
            change = run.base.variant_inputs(z.cpu().numpy(), 8)
            q = pool["query"]
            j = pipeline.winners(pool, change, weights)
            assert (j <= q - 30).all()
            row = dict(query_idx=q, j_star=j, min_past_gt_dist=minimum[q],
                       top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1), group=np.zeros(len(q), np.int64))
            plan = prior.base.prior.plan(travel, q, j)
            value = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(q)).astype(np.float64)
        metric = run.base.independent_metrics(row, value)
        other = run.white.prior.BASE.exp.metrics(value, row)
        for m in METRICS:
            if metric[m] is not None:
                np.testing.assert_allclose(metric[m], other[m], atol=1e-12, rtol=0)
        if tag.startswith("repro_"):
            ref = run.ARR(src / "diff_s1.npz") if tag == "repro_diff_s1" else run.ARR(pilot_ledger)
            checks[tag] = dict(winners_equal=bool(np.array_equal(j, ref["j_star"])),
                               query_equal=bool(np.array_equal(q, ref["query_idx"])),
                               score_bitwise_equal=bool(np.array_equal(value, ref["score"])),
                               score_max_error=float(np.abs(value - ref["score"]).max()))
        if tag == PRIMARY:
            pre = []
            for at in (len(q) // 3, 2 * len(q) // 3):
                t = int(q[at]); end = t + 16
                with torch.no_grad():
                    small, _, _ = prior.graph_inputs(role, sequence, end)
                    short = net(small)
                    sb = run.base.variant_inputs(short.cpu().numpy(), 8)
                    winner = run.cal.candidates(raw[:end], (bank[0][:end], *sb[1:]), weights[None], t)
                    pp = {k: torch.from_numpy(v).cuda() for k, v in prior.base.prior.plan(travel[:end], np.array([t]), winner).items()}
                    chk = P.score(short, pp, lam).cpu().numpy()
                pre.append(dict(query=t, winner_equal=int(winner[0]) == int(j[at]), score_error=float(abs(chk - value[at]).max())))
            checks["prefix_primary"] = pre
        np.savez_compressed(folder / (tag + ".npz"), **row, score=value)
        metrics[tag] = metric
        del net, z, change
    write(folder / "summary.json", dict(metrics=metrics, checks=checks, seconds=time.time() - tick))
    print("PIPE", role, sequence, round(time.time() - tick, 1), json.dumps(checks)[:300], flush=True)


def campaign(role: str, workers: int) -> None:
    import concurrent.futures
    import subprocess
    tick = time.time()
    seqs = VAL if role == "val" else TESTSEQ
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    def call(s):
        with (OUT / "logs" / f"{role}_{s.replace('/', '_')}.log").open("a") as f:
            subprocess.run([sys.executable, __file__, "pipe", "--role", role, "--sequence", s], stdout=f, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(concurrent.futures.as_completed([ex.submit(call, s) for s in seqs]), 1):
            fut.result()
            print("DONE", role, i, len(seqs), flush=True)
    timing("eval_" + role, time.time() - tick)


def finish() -> None:
    """검증9·시험46 집계, 재현 검사, 접두 검사, 방향별 짝지은 구간(비교 스크립트 지표 경로)."""
    import _pdv2_scratch_3seed_compare_20261116 as C
    tags = ["repro_diff_s1", "repro_uniform_reselected", *EVAL_CASES]
    out = {}
    for role, seqs in (("val", VAL), ("test", TESTSEQ)):
        per = {t: {} for t in tags}
        checks = {}
        for s in seqs:
            rec = run.READ(OUT / "eval" / role / s.replace("/", "_") / "summary.json")
            for t in tags:
                per[t][s] = rec["metrics"][t]
            checks[s] = rec["checks"]
        if role == "val":
            res = {t: dict(**run.base.macro(per[t]), sequences=per[t]) for t in tags}
        else:
            res = {t: dict(**run.current.adoption.previous.macro(per[t], run.current.CITY), sequences=per[t]) for t in tags}
            n = sum(len(run.ARR(OUT / "eval/test" / s.replace("/", "_") / f"{PRIMARY}.npz")["query_idx"]) for s in seqs)
            assert n == 195726, n
        repro = {t: dict(all_exact=all(c[t]["winners_equal"] and c[t]["query_equal"] and c[t]["score_bitwise_equal"] for c in checks.values()),
                         max_score_error=max(c[t]["score_max_error"] for c in checks.values()))
                 for t in ("repro_diff_s1", "repro_uniform_reselected")}
        prefix = [x for c in checks.values() for x in c["prefix_primary"]]
        out[role] = dict(results=res, reproduction=repro,
                         prefix_checks=dict(n=len(prefix), all_winner_equal=all(x["winner_equal"] for x in prefix),
                                            max_score_error=max(x["score_error"] for x in prefix),
                                            over_8e6=sum(x["score_error"] > 8e-6 for x in prefix)))
    pilot_val = run.READ(P.OUT / "summary.json")["val"]["results"]["uniform_reselected"]["macro"]
    out["val"]["uniform_macro_equals_pilot"] = out["val"]["results"]["repro_uniform_reselected"]["macro"] == pilot_val
    old = run.READ(P.TEST / "summary.json")
    out["test"]["diff_s1_macro_equals_ledger"] = all(abs(out["test"]["results"]["repro_diff_s1"]["macro"][m] - old["per_case"]["diff_s1"]["macro"][m]) < 1e-12 for m in METRICS)
    # 방향별·전체 지표를 비교 스크립트 경로(_pr_eval_lib)로 원장에서 다시 계산
    keys = C.KEYS
    rows = {}
    city = C.CITY
    for seqs in city.values():
        for s in seqs:
            name = s.replace("/", "_")
            dd = dict(np.load(C.DIRS / (name + ".npz")))
            masks = dict(all=None, forward_only=dd["group"] == 0, reverse_only=dd["group"] == 1)
            srcs = {c: [(OUT / "eval/test" / name / f"{c}.npz", "score")] for c in EVAL_CASES}
            srcs["uniform"] = [(OUT / "eval/test" / name / "repro_uniform_reselected.npz", "score")]
            srcs["nsd"] = [(C.MAIN / "test" / name / f"diff_s{sd}.npz", "score") for sd in (1, 2, 3)]
            srcs["bev"] = [(C.BEV / f"s{sd}" / (name + ".npz"), "score_negl2") for sd in (1, 2, 3)]
            for kind, lst in srcs.items():
                vals = []
                for path, field in lst:
                    with np.load(path) as z:
                        np.testing.assert_array_equal(z["query_idx"], dd["query_idx"])
                        vals.append({g: C.metric(z[field], z["top1_geo_dist"], z["min_past_gt_dist"], m) for g, m in masks.items()})
                rows.setdefault(kind, {})[s] = vals  # 시드 목록
    nseed = {k: len(next(iter(v.values()))) for k, v in rows.items()}

    def macro(kind, g, sd):
        cities = []
        for seqs in city.values():
            v = [rows[kind][s][sd][g] for s in seqs if rows[kind][s][sd][g] is not None]
            if v:
                cities.append({k: np.mean([x[k] for x in v]) for k in keys})
        return {k: float(np.mean([c[k] for c in cities])) for k in keys}

    def seqmean(kind, s, g, k):
        return float(np.mean([rows[kind][s][sd][g][k] for sd in range(nseed[kind])]))
    comp = {}
    for g in ("all", "forward_only", "reverse_only"):
        means = {kind: {k: float(np.mean([macro(kind, g, sd)[k] for sd in range(nseed[kind])])) for k in keys} for kind in rows}
        cmpg = {}
        for a in EVAL_CASES:
            for ref in ("uniform", "bev", "nsd"):
                rng = np.random.default_rng(20261104)
                draws = []
                for seqs in city.values():
                    avail = [s for s in seqs if rows[a][s][0][g] is not None]
                    if not avail:
                        continue
                    mat = np.array([[100 * (rows[a][s][0][g][k] - seqmean(ref, s, g, k)) for k in keys] for s in avail])
                    ix = rng.integers(0, len(avail), size=(10000, len(avail)))
                    draws.append(mat[ix].mean(1))
                boot = np.stack(draws).mean(0)
                cmpg[f"{a}-{ref}"] = dict(point_pp={k: 100 * (means[a][k] - means[ref][k]) for k in keys},
                                          conditional_95_pp={k: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(keys)})
        comp[g] = dict(mean=means, minus=cmpg)
    # 독립 재계산 대조: 전체 지표가 파이프라인 집계와 일치하는지
    indep = {}
    for c, t in [(c, c) for c in EVAL_CASES] + [("uniform", "repro_uniform_reselected")]:
        indep[c] = max(abs(comp["all"]["mean"][c][k] - out["test"]["results"][t]["macro"][k]) for k in keys)
    indep["nsd_vs_ledger_mean"] = max(abs(comp["all"]["mean"]["nsd"][k] - old["mean"]["diff"][k]) for k in keys)
    indep["bev_vs_ledger_mean"] = max(abs(comp["all"]["mean"]["bev"][k] - old["mean"]["bev_native"][k]) for k in keys)
    out["direction_compare"] = comp
    out["independent_recompute_max_abs"] = indep
    out["reference_rows_pct"] = dict(uniform_reselected=[89.198, 89.244, 68.820, 91.488], current_nsd_3start=[89.431, 89.330, 69.648, 91.733],
                                     bevplace_pp=[89.846, 88.995, 57.940, 93.900], scratch_single_stage_3seed=[88.388, 89.412, 67.971, 90.379])
    out["primary"] = PRIMARY
    out["test_used_for_selection"] = False
    write(OUT / "summary.json", out)
    for role in ("val", "test"):
        print(role, out[role]["reproduction"], out[role]["prefix_checks"])
        for t in tags:
            print("  ", t, {m: round(100 * out[role]["results"][t]["macro"][m], 3) for m in METRICS})
    print("indep", indep)
    for g, v in comp.items():
        print(g)
        for k2, x in v["minus"].items():
            print("  ", k2, {m: (round(x["point_pp"][m], 3), [round(y, 3) for y in x["conditional_95_pp"][m]]) for m in keys})


def three_seed() -> None:
    """시드 1·2·3의 lr3e-4 1536·4096: 검증9·시험46 평균±SD, 짝지은 구간(전체·순방향·역방향)."""
    import _pdv2_scratch_3seed_compare_20261116 as C
    keys, city = C.KEYS, C.CITY
    roots = {1: BASE, 2: BASE / "s2", 3: BASE / "s3"}
    old = run.READ(P.TEST / "summary.json")
    out = dict(seeds=[1, 2, 3])
    for u in (1536, 4096):
        tag = f"A_lr3e-4_u{u}"
        res = {}
        for role, seqs in (("val", VAL), ("test", TESTSEQ)):
            per, pre = {}, []
            for sd, r in roots.items():
                ps = {}
                for s in seqs:
                    rec = run.READ(r / "eval" / role / s.replace("/", "_") / "summary.json")
                    ps[s] = rec["metrics"][tag]
                    pre += rec["checks"].get("prefix_primary", []) if u == 4096 else []
                per[sd] = run.base.macro(ps)["macro"] if role == "val" else run.current.adoption.previous.macro(ps, run.current.CITY)["macro"]
            res[role] = dict(per_seed={str(k): v for k, v in per.items()},
                             mean={m: float(np.mean([per[sd][m] for sd in roots])) for m in METRICS},
                             sd={m: float(np.std([per[sd][m] for sd in roots], ddof=1)) for m in METRICS})
            if u == 4096:
                res[role]["prefix"] = dict(n=len(pre), all_winner_equal=all(x["winner_equal"] for x in pre),
                                           max_score_error=max(x["score_error"] for x in pre), over_8e6=sum(x["score_error"] > 8e-6 for x in pre))
        rows = {}
        for seqs in city.values():
            for s in seqs:
                name = s.replace("/", "_")
                dd = dict(np.load(C.DIRS / (name + ".npz")))
                masks = dict(all=None, forward_only=dd["group"] == 0, reverse_only=dd["group"] == 1)
                srcs = {"A": [(roots[sd] / "eval/test" / name / f"{tag}.npz", "score") for sd in (1, 2, 3)],
                        "uniform": [(BASE / "eval/test" / name / "repro_uniform_reselected.npz", "score")] * 3,
                        "nsd": [(C.MAIN / "test" / name / f"diff_s{sd}.npz", "score") for sd in (1, 2, 3)],
                        "bev": [(C.BEV / f"s{sd}" / (name + ".npz"), "score_negl2") for sd in (1, 2, 3)]}
                for kind, lst in srcs.items():
                    vals = []
                    for path, field in lst:
                        with np.load(path) as z:
                            np.testing.assert_array_equal(z["query_idx"], dd["query_idx"])
                            vals.append({g: C.metric(z[field], z["top1_geo_dist"], z["min_past_gt_dist"], m) for g, m in masks.items()})
                    rows.setdefault(kind, {})[s] = vals

        def macro(kind, g, sd):
            cs = []
            for seqs in city.values():
                v = [rows[kind][s][sd][g] for s in seqs if rows[kind][s][sd][g] is not None]
                if v:
                    cs.append({k: np.mean([x[k] for x in v]) for k in keys})
            return {k: float(np.mean([c[k] for c in cs])) for k in keys}
        comp = {}
        for g in ("all", "forward_only", "reverse_only"):
            means = {kind: {k: float(np.mean([macro(kind, g, sd)[k] for sd in range(3)])) for k in keys} for kind in rows}
            sds = {kind: {k: float(np.std([macro(kind, g, sd)[k] for sd in range(3)], ddof=1)) for k in keys} for kind in rows}
            cm = {}
            for ref in ("uniform", "bev", "nsd"):
                rng = np.random.default_rng(20261104)
                draws = []
                for seqs in city.values():
                    avail = [s for s in seqs if rows["A"][s][0][g] is not None]
                    if not avail:
                        continue
                    mat = np.array([[np.mean([100 * (rows["A"][s][sd][g][k] - rows[ref][s][sd][g][k]) for sd in range(3)])
                                     for k in keys] for s in avail])
                    ix = rng.integers(0, len(avail), size=(10000, len(avail)))
                    draws.append(mat[ix].mean(1))
                boot = np.stack(draws).mean(0)
                cm[ref] = dict(point_pp={k: 100 * (means["A"][k] - means[ref][k]) for k in keys},
                               conditional_95_pp={k: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(keys)})
            comp[g] = dict(mean=means, sd=sds, A_minus=cm)
        res["indep_all_vs_pipeline_max_abs"] = max(abs(comp["all"]["mean"]["A"][k] - res["test"]["mean"][k]) for k in keys)
        res["nsd_vs_ledger_max_abs"] = max(abs(comp["all"]["mean"]["nsd"][k] - old["mean"]["diff"][k]) for k in keys)
        res["direction_compare"] = comp
        out[f"u{u}"] = res
        print(tag)
        for role in ("val", "test"):
            print("  ", role, {m: (round(100 * res[role]["mean"][m], 3), round(100 * res[role]["sd"][m], 3)) for m in METRICS}, res[role].get("prefix"))
        print("   indep", res["indep_all_vs_pipeline_max_abs"], res["nsd_vs_ledger_max_abs"])
        for g, v in comp.items():
            for ref, x in v["A_minus"].items():
                print("  ", g, "A-" + ref, {k: (round(x["point_pp"][k], 3), [round(y, 3) for y in x["conditional_95_pp"][k]]) for k in keys})
    out["test_used_for_selection"] = False
    write(BASE / "three_seed_summary.json", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action")
    p.add_argument("--sequence", default="kitti360/0004")
    p.add_argument("--role", default="val")
    p.add_argument("--tag", default="lr3e-4")
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    P.e.configure()
    (OUT / "stage1").mkdir(parents=True, exist_ok=True)
    assert (BASE / "protocol.json").exists(), "protocol 먼저"
    {"prep": prep, "stage1": stage1, "stage2_init": stage2_init, "val_fixed": val_fixed, "finish": finish, "three_seed": three_seed}.get(a.action, lambda: None)()
    if a.action == "stage2":
        stage2(a.tag)
    elif a.action == "pipe":
        pipe(a.role, a.sequence)
    elif a.action == "campaign":
        campaign(a.role, a.workers)
