#!/usr/bin/env python3
"""수락 점수 학습의 두 고침 시험(3시드).

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  python3 scripts/_pdv2_acceptance_fix_20261116.py <action> [--fix f4|f13|f13c] [--tag lr1e-4] [--seed 1]

f4  : 처음부터, 두 DiffAttnConv(`convs.*`)만 학습, 게이트 고정(eta=-0.5), 시범과 같은 BCE.
f13 : 처음부터, 전체 모듈(151,049). 학습 목표의 반환 장소를 현재 수정 계수로 전체 후보에서 다시
      고르고(고침 1), 시퀀스 안 짝 순위 손실(고침 3)로 학습.
f13c: f13과 같은 목표·손실, 학습 범위는 f4처럼 `convs.*`만, 게이트 고정(선택 변형).
f13b: f13과 같되 게이트를 약화 쪽으로만 묶음: eta = clamp(-sigmoid(g), min=-0.5) (조정자 추가 요청).
규칙은 OUT/protocol.json에 결과 전에 적는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import _pdv2_scratch_pilot_20261116 as P  # noqa: E402
import _pdv2_pretrain_redesign_B_20261116 as B  # noqa: E402
from gnn.contrast_spectral_difference import ContrastSpectralDifference  # noqa: E402
from gnn.spectral_difference import transport_difference  # noqa: E402

run, prior, pipeline, e, recal = P.run, P.prior, P.pipeline, P.e, P.recal
OUT = ROOT / "results/pdv2_acceptance_fix_20261116"
TRAIN, VAL, TESTSEQ, METRICS = P.TRAIN, P.VAL, P.TESTSEQ, P.METRICS
LRS = {"lr1e-4": 1e-4, "lr3e-4": 3e-4}
SAVE = (1536, 4096)
SEEDS = (1, 2, 3)
PEN = {"f13p1": 1.0, "f13p10": 10.0, "f13p100": 100.0}  # amendment_2: 벌점형 약화 변형
FIXES = ("f4", "f13", "f13c", "f13b", *PEN)


def lrs(fix: str) -> dict:
    """f13p*는 f13 검증 선택의 학습률 하나만(amendment_2)."""
    if fix in PEN:
        tag = run.READ(OUT / "f13p_lr.json")["tag"]
        return {tag: LRS[tag]}
    return LRS
Q_REVISIT = 64   # f13: 갱신당 재방문 질의(min_past<=5)
Q_NEGATIVE = 64  # f13: 갱신당 비재방문 질의(min_past>25)
ETA_BINS = np.linspace(-1.0, 0.0, 201)
if os.environ.get("NSD_ACC_SMOKE"):  # 코드 점검용 짧은 실행(결과로 쓰지 않음)
    SAVE = (16, 60)
    OUT = Path(os.environ["NSD_ACC_SMOKE"])


def write(path: Path, obj) -> None:
    P.write(path, obj)


def timing(step: str, seconds: float, **extra) -> None:
    path = OUT / "timing.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault(step, []).append(dict(seconds=round(seconds, 1), **extra))
    write(path, data)


def name_of(s: str) -> str:
    return s.replace("/", "_")


def convs_only(fix: str) -> bool:
    return fix in ("f4", "f13c")


# ----------------------------------------------------------------- 모델
class WeakenOnly(ContrastSpectralDifference):
    """ContrastSpectralDifference.forward와 같고 세기만 eta = clamp(-sigmoid(g), min=-0.5).

    초기 g=0에서 -0.5로 같다(비트 일치). clamp는 경계에서 기울기를 그대로 통과시켜 약화 방향으로
    움직일 수 있고, 강화 쪽(g>0)은 -0.5에 묶여 기울기 0이다.
    """

    def forward(self, graph) -> torch.Tensor:
        h = F.relu(self.input_norm(self.input_proj(graph.x)))
        first = h
        attr = self.edge_encoder(graph.edge_attr, graph.edge_type) if self.edge_encoder is not None else None
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
        strengths = (-torch.stack((self.gate(first).sigmoid(), self.gate(h).sigmoid()))).clamp(min=-0.5)
        self.last_strengths = strengths
        return transport_difference(graph.spectrum, graph.transport_indices, graph.transport_mask,
                                    graph.transport_phase, strengths, torch.stack(weights))


def base_model(seed: int, fix: str) -> torch.nn.Module:
    net = P.scratch_model(seed)
    if fix == "f13b":
        net.__class__ = WeakenOnly
    return net


def new_model(seed: int, fix: str):
    """시드 무작위 초기화 + 0 초기화 세 층, BN은 초기 가중치로 학습8에서 추정 후 고정, 학습 범위 지정."""
    net = base_model(seed, fix)
    bn = P.estimate_bn(net)
    if convs_only(fix):
        for n, p in net.named_parameters():
            p.requires_grad_(n.startswith("convs."))
    return net, bn


def load(path: Path, fix: str) -> torch.nn.Module:
    net = base_model(1, fix)
    net.load_state_dict(torch.load(path, map_location="cuda", weights_only=False)["model_state_dict"], strict=True)
    return net.eval()


def ckpt(fix: str, seed: int, tag: str, u: int) -> Path:
    return OUT / fix / f"s{seed}" / tag / f"u{u}.pth"


def model_tags(fix: str) -> dict:
    return {f"{tag}_u{u}_s{sd}": ckpt(fix, sd, tag, u) for tag in lrs(fix) for u in SAVE for sd in SEEDS}


def configs(fix: str) -> list:
    return [f"{tag}_u{u}" for tag in lrs(fix) for u in SAVE]


# ----------------------------------------------------------------- 사전 규칙
def protocol() -> None:
    path = OUT / "protocol.json"
    if path.exists():
        print("protocol exists")
        return
    w, lam = P.selected()
    fit = run.READ(P.OUT / "step2/loss_fit.json")["fit"]
    write(path, dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        written_before_results=True,
        purpose="수락 점수 학습의 두 고침(고침 4; 고침 1+3)을 처음부터, 3시드로 시험",
        common=dict(
            structure="현재 diff(ContrastSpectralDifference 'diff'), 매개변수 151,049. torch.manual_seed(시드) 무작위 초기화 뒤 Q 투영·기하 MLP 최종층·게이트 최종층 0 -> 초기 출력 = 균등 감산(eta=-0.5) 정확히.",
            fixed_settings=dict(fusion_weights=w.tolist(), lam=lam, source="results/pdv2_scratch_pilot_20261116/step1/selection.json"),
            normalization="BN·dropout eval 고정. BN 누적 통계는 학습 전 초기 가중치로 학습8을 한 번 통과시켜 추정(P.estimate_bn, 시범 주 조건). 이유: 입력 절댓값 ~2.6e4라 기본 통계(0/1)는 투영 출력이 매우 크고, 시범·재설계 B의 주 조건과 같아야 비교가 된다. 시범의 기본 BN 우위는 사후 진단이었고 3시드에서 반복되지 않았다.",
            seeds=list(SEEDS),
            seed_role="시드가 무작위 초기화(입력 투영·BN 앞 가중치·간선 부호화·K/V 등)와 표본 순서를 함께 정한다. 학습하지 않는 층도 시드별로 결정적이다.",
            calibration=dict(fit=fit, source="results/pdv2_scratch_pilot_20261116/step2/loss_fit.json", note="초기 출력(균등 감산)은 시드와 무관하므로 시범 척도를 그대로 쓴다. logit = (value-mean)*coef/scale + intercept."),
            optimizer="AdamW, 상수 학습률, weight decay 1e-5, 기울기 절단 1.0 (시범과 같음)",
            grid=dict(lr=list(LRS.values()), updates=list(SAVE), note="각 (lr, 시드)마다 4096갱신 한 번 학습, 1536·4096 저장"),
        ),
        fix4=dict(
            trainable="convs.* 두 DiffAttnConv만(100,680). 입력 투영·입력 BN·층 BN·간선 부호화·게이트는 시드별 무작위 초기값(게이트 최종층 0)으로 고정 -> eta=-0.5 고정.",
            objective="시범과 같은 BCE: 학습8 고정 원키 색인 승자 표본(results/pdv2_scratch_pilot_20261116/step2/pools, 재선택 융합), 갱신당 64쌍(정답·오답 균등), 에포크마다 학습8 순서로 시퀀스당 8갱신, rng (시드, epoch, ordinal).",
        ),
        fix13=dict(
            trainable="f13: 전체 151,049. f13c(선택 변형, 시간이 되면): convs.*만, 게이트 고정.",
            fix1_target="매 갱신마다 현재 모델 출력 z'로 표본 질의의 전체 후보(두 색인 상위 100+100 합집합, 재설계 B cands)에 재선택 융합 장소 선택 점수(재설계 B selection_score, 기울기 없음)를 계산해 argmax 반환 장소 j_hat을 고른다. 표본 질의의 수락 점수는 (q, j_hat)에서 계산한다(평가와 같음).",
            labels="양성 = 재방문 질의(min_past<=5) 중 j_hat이 기준 자세 5 m 안. 음성 = 재방문인데 j_hat이 5 m 밖 + 비재방문(min_past>25). 모호(5~25 m) 제외. pr_sweep의 TP/FP 정의와 같다.",
            fix3_loss="시퀀스 안 짝 로지스틱 순위 손실: L = mean_{i in 양성, k in 음성} softplus(logit_k - logit_i), logit은 위 고정 척도(절편은 상쇄). 한 갱신 = 한 시퀀스이므로 짝은 항상 같은 시퀀스 안.",
            sampling=f"갱신당 한 시퀀스에서 재방문 질의 {Q_REVISIT}개 + 비재방문 질의 {Q_NEGATIVE}개(각각 균등, 부족하면 복원). 시퀀스 균형: 에포크마다 학습8 순서로 시퀀스당 8갱신. 재방문 질의가 없는 kitti/01은 짝을 만들 수 없어 제외(7시퀀스, 에포크당 56갱신). 저장은 갱신 수 1536·4096에서(에포크 중간 가능). rng (시드, 313, epoch, ordinal). 양성이나 음성이 0인 표본은 그 갱신의 step을 건너뛰고 기록.",
        ),
        evaluation=dict(
            models="각 고침의 lr x 갱신 x 시드 12개 전부를 검증9·시험46 실제 검색(matching_only, 재선택 융합·lam 0.25)에 적용.",
            selection="고침마다 설정(lr, 갱신) 중 검증9 실제 검색 3계열 AP의 3시드 평균 최대. 동률은 적은 갱신, 그다음 작은 lr. 시험은 선택에 쓰지 않는다(시험은 모든 설정을 보고).",
            reproduction="같은 실행에서 균등 감산을 다시 계산해 시범 원장 uniform_reselected와 승자·점수 비트 일치 확인.",
            intervals="최적 설정의 3시드 평균 ± SD, 도시 안 시퀀스 재표집 10,000회(rng 20261104) 짝지은 95% 구간: 균등 감산 대비, BEVPlace++(시드 짝) 대비, 순방향만·역방향만 BEVPlace++ 대비(_pdv2_scratch_3seed_compare_20261116의 방향 정의).",
            checks="0 초기화 항등(시드 1·2·3, 추정 BN), torch 재선택 승자와 numpy 파이프라인 승자 일치(학습 시퀀스), 시험 지표 독립 재계산(pr_sweep), q+15 접두 검사(모든 모델, 시퀀스당 2점; 최적 설정 보고).",
        ),
    ))
    print("protocol written")


# ----------------------------------------------------------------- 검사
def verify() -> None:
    tick = time.time()
    rec = {}
    lam = P.selected()[1]
    seqs = (("train", TRAIN[0]), ("train", "helipr/Bridge01"), ("val", "kitti360/0004"))
    graphs = {s: (B.train_graph(s) if r == "train" else prior.graph_inputs(r, s)[0]) for r, s in seqs}
    init = {}
    with torch.no_grad():
        for seed in SEEDS:
            for fix in ("f4", "f13", "f13b"):
                net, _ = new_model(seed, fix)
                for r, s in seqs:
                    out = net(graphs[s])
                    ref = prior.filtered(graphs[s], -0.5)
                    init[f"{fix}_s{seed}_{name_of(s)}"] = dict(
                        bitwise_equal_uniform=bool(torch.equal(out, ref)), max_abs=float((out - ref).abs().max()),
                        strengths_all_minus_half=bool(torch.all(net.last_strengths == -0.5)))
                rec[f"trainable_{fix}"] = sum(p.numel() for p in net.parameters() if p.requires_grad)
    rec["init_identity"] = init
    rec["init_identity_all"] = all(v["bitwise_equal_uniform"] and v["strengths_all_minus_half"] for v in init.values())
    # torch 재선택 승자 vs numpy 파이프라인(균등 감산), 고정 원키 승자와의 차이
    w, _ = P.selected()
    sel = {}
    for s in ("kitti/05", "helipr/Bridge01"):
        d = B.build_cands("train", s)
        with torch.no_grad():
            z = prior.filtered(graphs[s] if s in graphs else B.train_graph(s), -0.5)
            change = run.base.variant_inputs(z.cpu().numpy(), 8)
            pool = dict(query=d["query"], offset=d["offset"], ids=d["ids"], static=(w[0] * d["sn"] + w[1] * d["sh"]).astype(np.float32))
            j_np = pipeline.winners(pool, change, w)
            tab = B.tables(z)
            wins = []
            for start in range(0, len(d["query"]), 512):
                b = B.padded(d, np.arange(start, min(start + 512, len(d["query"]))))
                sc = B.selection_score(tab, b, B.weights_t()).masked_fill(~b["mask"], float("-inf"))
                wins.append(b["ids"].gather(1, sc.argmax(-1)[:, None])[:, 0].cpu().numpy())
            j_t = np.concatenate(wins)
        pl = run.ARR(P.NEWPOOLS / "train" / name_of(s) / "pool.npz")
        assert np.array_equal(pl["query_idx"], d["query"])
        m = d["min_past_gt_dist"]
        sel[s] = dict(queries=len(j_np), torch_equal_numpy=int((j_t == j_np).sum()),
                      reselected_vs_fixed_key_winner_differ=int((j_t != pl["j_star"]).sum()),
                      revisit_queries=int((m <= 5).sum()),
                      revisit_differ=int(((j_t != pl["j_star"]) & (m <= 5)).sum()))
    rec["reselection_at_init"] = sel
    # 기울기 범위 검사(f4: convs만)
    net, _ = new_model(1, "f4")
    data = run.ARR(P.NEWPOOLS / "train" / name_of(TRAIN[0]) / "pool.npz")
    plan = {k: torch.from_numpy(data["plan_" + k][:64]).cuda() for k in ("query", "candidate", "weight")}
    fit = run.READ(P.OUT / "step2/loss_fit.json")["fit"]
    v = P.score(net(graphs[TRAIN[0]]), plan, lam)
    logit = (v - fit["mean"][0]) * fit["coefficients"][0] / fit["scale"][0] + fit["intercept"]
    F.binary_cross_entropy_with_logits(logit, torch.from_numpy(data["correct"][:64]).cuda().float()).backward()
    rec["f4_grad_tensors"] = sorted(n for n, p in net.named_parameters() if p.grad is not None and float(p.grad.abs().max()) > 0)
    rec["f4_nongrad_trainable"] = sorted(n for n, p in net.named_parameters() if p.requires_grad and (p.grad is None or float(p.grad.abs().max()) == 0))
    rec["seconds"] = time.time() - tick
    write(OUT / "verify.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "init_identity"}, indent=1))


# ----------------------------------------------------------------- 학습
def logit_of(value, fit):
    return (value - fit["mean"][0]) * fit["coefficients"][0] / fit["scale"][0] + fit["intercept"]


def train(fix: str, tag: str, seed: int) -> None:
    folder = OUT / fix / f"s{seed}" / tag
    if (folder / "training_complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    lr = LRS[tag]
    _, lam = P.selected()
    fit = run.READ(P.OUT / "step2/loss_fit.json")["fit"]
    net, bn = new_model(seed, fix)
    g0 = B.train_graph(TRAIN[0])
    with torch.no_grad():
        assert torch.equal(net(g0), prior.filtered(g0, -0.5))
    del g0
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    graphs = {s: B.train_graph(s) for s in TRAIN}
    records, hashes, updates, skipped, tick = [], {}, 0, 0, time.time()

    def save(epoch):
        path = folder / f"u{updates}.pth"
        torch.save(dict(model_state_dict=net.state_dict(), epoch=epoch, updates=updates, lr=lr, seed=seed, fix=fix), path)
        hashes[path.name] = run.HASH(path)
        print("TRAIN", fix, tag, seed, epoch, updates, round(time.time() - tick, 1), flush=True)

    if fix == "f4":
        pools = {s: run.ARR(P.NEWPOOLS / "train" / name_of(s) / "pool.npz") for s in TRAIN}
        plans = {s: {k: torch.from_numpy(d["plan_" + k]).cuda() for k in ("query", "candidate", "weight")} for s, d in pools.items()}
        epoch = 0
        while updates < max(SAVE):
            epoch += 1
            for ordinal, s in enumerate(TRAIN):
                data = pools[s]
                sets = [np.flatnonzero(data["eligible"] & (data["correct"] == c)) for c in (0, 1)]
                sets = [v for v in sets if len(v)]
                rng = np.random.default_rng((seed, epoch, ordinal))
                losses, grads = [], []
                for _ in range(8):
                    n = 64 // len(sets)
                    chosen = rng.permutation(np.concatenate([rng.choice(v, n, replace=len(v) < n) for v in sets]))
                    ix = torch.from_numpy(chosen).cuda()
                    opt.zero_grad(set_to_none=True)
                    z = net(graphs[s])
                    value = P.score(z, {k: v[ix] for k, v in plans[s].items()}, lam)
                    loss = F.binary_cross_entropy_with_logits(logit_of(value, fit), torch.from_numpy(data["correct"][chosen]).cuda().float())
                    assert torch.isfinite(loss)
                    loss.backward()
                    g = torch.nn.utils.clip_grad_norm_(params, 1.0)
                    assert torch.isfinite(g)
                    opt.step()
                    updates += 1
                    losses.append(float(loss.detach()))
                    grads.append(float(g))
                records.append(dict(epoch=epoch, sequence=s, loss=float(np.mean(losses)), grad=float(np.mean(grads)),
                                    strength_mean=float(net.last_strengths.detach().mean())))
            if updates in SAVE:
                save(epoch)
    else:
        w = B.weights_t()
        cands = {s: B.build_cands("train", s) for s in TRAIN}
        travel = {s: P.inputs("train", s)[3] for s in TRAIN}
        fixed_j = {s: run.ARR(P.NEWPOOLS / "train" / name_of(s) / "pool.npz")["j_star"] for s in TRAIN}
        rows, full = {}, {}
        for s in TRAIN:
            m = cands[s]["min_past_gt_dist"]
            rows[s] = (np.flatnonzero(m <= 5), np.flatnonzero(m > 25))
            if len(rows[s][0]):
                full[s] = B.padded(cands[s])
        active = [s for s in TRAIN if s in full]
        epoch = 0
        while updates < max(SAVE):
            epoch += 1
            for ordinal, s in enumerate(TRAIN):
                if s not in full or updates >= max(SAVE):
                    continue
                rev, neg = rows[s]
                rng = np.random.default_rng((seed, 313, epoch, ordinal))
                losses, grads, stats = [], [], []
                for _ in range(8):
                    if updates >= max(SAVE):
                        break
                    r = rev[rng.choice(len(rev), Q_REVISIT, replace=len(rev) < Q_REVISIT)]
                    n = neg[rng.choice(len(neg), Q_NEGATIVE, replace=len(neg) < Q_NEGATIVE)]
                    chosen = np.concatenate((r, n))
                    ix = torch.from_numpy(chosen).cuda()
                    opt.zero_grad(set_to_none=True)
                    z = net(graphs[s])
                    with torch.no_grad():
                        b = B.take(full[s], ix)
                        sc = B.selection_score(B.tables(z.detach()), b, w).masked_fill(~b["mask"], float("-inf"))
                        win = sc.argmax(-1)
                        j_hat = b["ids"].gather(1, win[:, None])[:, 0].cpu().numpy()
                        hit = b["pos"].gather(1, win[:, None])[:, 0].cpu().numpy()
                    y = np.zeros(len(chosen), bool)
                    y[:len(r)] = hit[:len(r)]
                    qv = cands[s]["query"][chosen]
                    assert (j_hat <= qv - 30).all()
                    pp = {k: torch.from_numpy(v).cuda() for k, v in prior.base.prior.plan(travel[s], qv, j_hat).items()}
                    logit = logit_of(P.score(z, pp, lam), fit)
                    yt = torch.from_numpy(y).cuda()
                    stats.append((float(y.mean()), float((j_hat[:len(r)] != fixed_j[s][r]).mean())))
                    if y.all() or not y.any():
                        skipped += 1
                        updates += 1
                        if updates in SAVE:
                            save(epoch)
                        continue
                    loss = F.softplus(logit[~yt][None, :] - logit[yt][:, None]).mean()
                    if fix in PEN:  # eta = last_strengths(두 층, 시퀀스 전체 노드); -0.5보다 강한 감산만 벌점
                        loss = loss + PEN[fix] * F.relu(-0.5 - net.last_strengths).pow(2).mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    g = torch.nn.utils.clip_grad_norm_(params, 1.0)
                    assert torch.isfinite(g)
                    opt.step()
                    updates += 1
                    losses.append(float(loss.detach()))
                    grads.append(float(g))
                    if updates in SAVE:
                        save(epoch)
                records.append(dict(epoch=epoch, sequence=s, loss=float(np.mean(losses)) if losses else None,
                                    grad=float(np.mean(grads)) if grads else None,
                                    positive_fraction=float(np.mean([a for a, _ in stats])),
                                    revisit_winner_changed_vs_fixed=float(np.mean([c for _, c in stats])),
                                    strength_mean=float(net.last_strengths.detach().mean())))
        assert set(hashes) == {f"u{u}.pth" for u in SAVE}, hashes
    if convs_only(fix):
        assert torch.all(net.last_strengths == -0.5)
    changed = [k for k, v in net.state_dict().items() if not torch.equal(v, initial[k])]
    write(folder / "training_complete.json", dict(
        status="passed", fix=fix, lr=lr, seed=seed, updates=updates, skipped_steps=skipped, bn_estimated_batches=bn,
        checkpoints=hashes, records=records, changed=changed, trainable_parameters=sum(p.numel() for p in params),
        seconds=time.time() - tick, peak_cuda_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
        sequences=active if fix != "f4" else list(TRAIN)))
    timing(f"train_{fix}_{tag}_s{seed}", time.time() - tick)


# ----------------------------------------------------------------- 실제 검색 평가
def pipe(fix: str, role: str, sequence: str) -> None:
    name = name_of(sequence)
    folder = OUT / fix / "eval" / role / name
    if (folder / "summary.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    src = (P.PIPE / "val" / name) if role == "val" else (P.TEST / "test" / name)
    cache, pos, minimum, travel, bank, _, _, _ = P.inputs(role, sequence)
    graph, _, _ = prior.graph_inputs(role, sequence)
    raw = pipeline.raw_key(cache["descriptors"])
    poolpath = src / "candidate_pool.npz"
    assert run.HASH(poolpath) == run.READ(poolpath.with_suffix(".json"))["sha256"]
    base_pool = run.ARR(poolpath)
    weights, lam = P.selected()
    static = []
    for n, q in enumerate(base_pool["query"]):
        lo, hi = base_pool["offset"][n:n + 2]
        ids = base_pool["ids"][lo:hi]
        static.append(weights[0] * run.base.minmax(raw[ids] @ raw[q]) + weights[1] * run.base.minmax(bank[0][ids] @ bank[0][q]))
    pool = dict(base_pool, static=np.concatenate(static).astype(np.float32))
    q = pool["query"]
    cases = {"repro_uniform": None, **model_tags(fix)}
    pts = [len(q) // 3, 2 * len(q) // 3]
    smalls = {}
    metrics, checks, eta = {}, {}, {}
    for tag, ck in cases.items():
        with torch.no_grad():
            net = None if ck is None else load(ck, fix)
            z = prior.filtered(graph, -0.5) if net is None else net(graph)
            change = run.base.variant_inputs(z.cpu().numpy(), 8)
            j = pipeline.winners(pool, change, weights)
            assert (j <= q - 30).all()
            row = dict(query_idx=q, j_star=j, min_past_gt_dist=minimum[q],
                       top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1), group=np.zeros(len(q), np.int64))
            plan = prior.base.prior.plan(travel, q, j)
            value = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(q)).astype(np.float64)
            if net is not None:
                st = net.last_strengths.detach().reshape(2, -1).double().cpu().numpy()
                eta[tag] = [dict(n=int(len(x)), mean=float(x.mean()), hist=np.histogram(x, bins=ETA_BINS)[0].tolist(),
                                 below=int((x < -0.5).sum()), above=int((x > -0.5).sum()), min=float(x.min()), max=float(x.max()))
                            for x in st]
        metric = run.base.independent_metrics(row, value)
        other = run.white.prior.BASE.exp.metrics(value, row)
        for m in METRICS:
            if metric[m] is not None:
                np.testing.assert_allclose(metric[m], other[m], atol=1e-12, rtol=0)
        if net is None:
            ref = run.ARR(P.OUT / "step3" / role / name / "uniform_reselected.npz")
            checks[tag] = dict(winners_equal=bool(np.array_equal(j, ref["j_star"])),
                               score_bitwise_equal=bool(np.array_equal(value, ref["score"])),
                               score_max_error=float(np.abs(value - ref["score"]).max()))
        else:
            pre = []
            for at in pts:
                t = int(q[at])
                end = t + 16
                with torch.no_grad():
                    if end not in smalls:
                        smalls[end] = prior.graph_inputs(role, sequence, end)[0]
                    short = net(smalls[end])
                    sb = run.base.variant_inputs(short.cpu().numpy(), 8)
                    winner = run.cal.candidates(raw[:end], (bank[0][:end], *sb[1:]), weights[None], t)
                    pp = {k: torch.from_numpy(v).cuda() for k, v in prior.base.prior.plan(travel[:end], np.array([t]), winner).items()}
                    chk = P.score(short, pp, lam).cpu().numpy()
                pre.append(dict(query=t, winner_equal=int(winner[0]) == int(j[at]), score_error=float(abs(chk - value[at]).max())))
            checks[tag] = dict(prefix=pre)
        np.savez_compressed(folder / (tag + ".npz"), **row, score=value)
        metrics[tag] = metric
        del net, z, change
    write(folder / "summary.json", dict(metrics=metrics, checks=checks, eta=eta, seconds=time.time() - tick))
    print("PIPE", fix, role, sequence, round(time.time() - tick, 1), flush=True)


def campaign(fix: str, role: str, workers: int) -> None:
    import concurrent.futures
    import subprocess
    tick = time.time()
    seqs = VAL if role == "val" else TESTSEQ
    (OUT / fix / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3")

    def call(s):
        with (OUT / fix / "logs" / f"{role}_{name_of(s)}.log").open("a") as f:
            subprocess.run([sys.executable, __file__, "pipe", "--fix", fix, "--role", role, "--sequence", s],
                           stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(concurrent.futures.as_completed([ex.submit(call, s) for s in seqs]), 1):
            fut.result()
            print("DONE", fix, role, i, len(seqs), flush=True)
    timing(f"eval_{fix}_{role}", time.time() - tick, workers=workers)


# ----------------------------------------------------------------- 집계
def finish(fix: str) -> None:
    import _pdv2_scratch_3seed_compare_20261116 as C
    tags = ["repro_uniform", *model_tags(fix)]
    out = {}
    for role, seqs in (("val", VAL), ("test", TESTSEQ)):
        per = {t: {} for t in tags}
        chk = {}
        for s in seqs:
            rec = run.READ(OUT / fix / "eval" / role / name_of(s) / "summary.json")
            for t in tags:
                per[t][s] = rec["metrics"][t]
            chk[s] = rec["checks"]
        if role == "val":
            res = {t: dict(**run.base.macro(per[t]), sequences=per[t]) for t in tags}
        else:
            res = {t: dict(**run.current.adoption.previous.macro(per[t], run.current.CITY), sequences=per[t]) for t in tags}
            n = sum(len(run.ARR(OUT / fix / "eval/test" / name_of(s) / "repro_uniform.npz")["query_idx"]) for s in seqs)
            assert n == 195726, n
        repro = all(c["repro_uniform"]["winners_equal"] and c["repro_uniform"]["score_bitwise_equal"] for c in chk.values())
        prefix = {t: [x for c in chk.values() for x in c[t]["prefix"]] for t in tags if t != "repro_uniform"}
        three = {}
        for cfg in configs(fix):
            vals = {m: [res[f"{cfg}_s{sd}"]["macro"][m] for sd in SEEDS] for m in METRICS}
            three[cfg] = {m: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), seeds=v) for m, v in vals.items()}
        etas = {}
        for t in tags[1:]:
            layers = []
            for li in range(2):
                recs = [run.READ(OUT / fix / "eval" / role / name_of(s) / "summary.json")["eta"][t][li] for s in seqs]
                n = sum(r["n"] for r in recs)
                cdf = np.cumsum(np.sum([r["hist"] for r in recs], 0)) / n
                qs = {f"p{int(round(100 * qq))}": float(ETA_BINS[1:][np.searchsorted(cdf, qq)]) for qq in (0.05, 0.5, 0.95)}
                layers.append(dict(mean=float(sum(r["mean"] * r["n"] for r in recs) / n), **qs,
                                   frac_stronger=sum(r["below"] for r in recs) / n, frac_weaker=sum(r["above"] for r in recs) / n,
                                   min=min(r["min"] for r in recs), max=max(r["max"] for r in recs)))
            etas[t] = layers
        out[role] = dict(results=res, three_seed=three, eta=etas, reproduction_uniform_exact=repro,
                         prefix={t: dict(n=len(v), all_winner_equal=all(x["winner_equal"] for x in v),
                                         max_score_error=max(x["score_error"] for x in v)) for t, v in prefix.items()})
    rank = sorted(configs(fix), key=lambda c: (-out["val"]["three_seed"][c]["AP"]["mean"], int(c.split("_u")[1]), LRS[c.split("_u")[0]]))
    best = rank[0]
    out["selected"] = best
    out["selection_rule"] = "검증9 실제 검색 3계열 AP 3시드 평균 최대; 시험 미사용"
    out["test_used_for_selection"] = False
    out["best_prefix"] = {role: {f"{best}_s{sd}": out[role]["prefix"][f"{best}_s{sd}"] for sd in SEEDS} for role in ("val", "test")}

    # 방향 분할·짝지은 구간·독립 재계산(_pdv2_scratch_3seed_compare_20261116와 같은 정의)
    def ledger(kind, seed, name):
        if kind == "model":
            p, f = OUT / fix / "eval/test" / name / f"{best}_s{seed}.npz", "score"
        elif kind == "uniform":
            p, f = C.PILOT / name / "uniform_reselected.npz", "score"
        elif kind == "nsd":
            p, f = C.MAIN / "test" / name / f"diff_s{seed}.npz", "score"
        else:
            p, f = C.BEV / f"s{seed}" / (name + ".npz"), "score_negl2"
        with np.load(p) as zz:
            return zz[f], zz["top1_geo_dist"], zz["min_past_gt_dist"], zz["query_idx"]

    rows = {}
    for kind in ("model", "uniform", "nsd", "bev"):
        for sd in SEEDS:
            for seqs in C.CITY.values():
                for s in seqs:
                    name = name_of(s)
                    d = dict(np.load(C.DIRS / (name + ".npz")))
                    sc, top1, minimum, qq = ledger(kind, sd, name)
                    np.testing.assert_array_equal(qq, d["query_idx"])
                    masks = dict(all=None, forward_only=d["group"] == 0, reverse_only=d["group"] == 1)
                    rows.setdefault(kind, {}).setdefault(sd, {})[s] = {g: C.metric(sc, top1, minimum, mk) for g, mk in masks.items()}
    comp = {}
    for g in ("all", "forward_only", "reverse_only"):
        def macro(kind, sd):
            cities = []
            for seqs in C.CITY.values():
                v = [rows[kind][sd][s][g] for s in seqs if rows[kind][sd][s][g] is not None]
                if v:
                    cities.append({k: np.mean([x[k] for x in v]) for k in C.KEYS})
            return {k: float(np.mean([c[k] for c in cities])) for k in C.KEYS}
        means = {kind: {k: float(np.mean([macro(kind, sd)[k] for sd in SEEDS])) for k in C.KEYS} for kind in rows}
        sds = {kind: {k: float(np.std([macro(kind, sd)[k] for sd in SEEDS], ddof=1)) for k in C.KEYS} for kind in rows}
        cmp = {}
        for ref in ("bev", "uniform", "nsd"):
            rng = np.random.default_rng(20261104)
            draws = []
            for seqs in C.CITY.values():
                avail = [s for s in seqs if rows["model"][1][s][g] is not None]
                if not avail:
                    continue
                mat = np.array([[np.mean([100 * (rows["model"][sd][s][g][k] - rows[ref][sd][s][g][k]) for sd in SEEDS])
                                 for k in C.KEYS] for s in avail])
                idx = rng.integers(0, len(avail), size=(10000, len(avail)))
                draws.append(mat[idx].mean(1))
            boot = np.stack(draws).mean(0)
            cmp[ref] = dict(point_pp={k: 100 * (means["model"][k] - means[ref][k]) for k in C.KEYS},
                            conditional_95_pp={k: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(C.KEYS)})
        comp[g] = dict(mean=means, sd=sds, model_minus=cmp)
    # 독립 재계산: pr_sweep 경로 전체 지표 3시드 평균 = 파이프라인 3시드 평균
    mapping = {"AP": "AP", "max_F1": "max_F1", "R@100P": "R@100P", "R@1": "R@1"}
    pipe_mean = {m: out["test"]["three_seed"][best][m]["mean"] for m in METRICS}
    indep = {k: comp["all"]["mean"]["model"][k] for k in C.KEYS}
    out["independent_recompute"] = dict(
        pipeline_mean=pipe_mean, pr_sweep_mean=indep,
        max_abs_error=max(abs(pipe_mean[mapping[k]] - indep[k]) for k in C.KEYS),
        uniform_pr_sweep=comp["all"]["mean"]["uniform"], bev_pr_sweep=comp["all"]["mean"]["bev"], nsd_pr_sweep=comp["all"]["mean"]["nsd"])
    out["direction_compare"] = comp
    write(OUT / fix / "summary.json", out)
    for role in ("val", "test"):
        print(role, "repro", out[role]["reproduction_uniform_exact"])
        print("   uniform", {m: round(100 * out[role]["results"]["repro_uniform"]["macro"][m], 3) for m in METRICS})
        for c, v in out[role]["three_seed"].items():
            print("  ", c, {m: (round(100 * x["mean"], 3), round(100 * x["sd"], 3)) for m, x in v.items()})
    print("SELECTED", best, "indep err", out["independent_recompute"]["max_abs_error"])
    for g, v in comp.items():
        for ref, x in v["model_minus"].items():
            print(g, "-" + ref, {k: (round(x["point_pp"][k], 3), [round(y, 3) for y in x["conditional_95_pp"][k]]) for k in C.KEYS})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action")
    p.add_argument("--fix", default="f4", choices=FIXES)
    p.add_argument("--tag", default="lr1e-4")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sequence", default="kitti360/0004")
    p.add_argument("--role", default="val")
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    e.configure()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.action == "protocol":
        protocol()
    elif a.action == "verify":
        verify()
    elif a.action == "train":
        train(a.fix, a.tag, a.seed)
    elif a.action == "pipe":
        pipe(a.fix, a.role, a.sequence)
    elif a.action == "campaign":
        campaign(a.fix, a.role, a.workers)
    elif a.action == "finish":
        finish(a.fix)
    else:
        raise SystemExit(a.action)
