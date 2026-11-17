#!/usr/bin/env python3
"""현재 시간 모듈을 사전학습 없이 한 단계로 학습하는 1시드 시범.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  python3 scripts/_pdv2_scratch_pilot_20261116.py <action>

규칙은 results/pdv2_scratch_pilot_20261116/protocol.json에 결과 전에 적었다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import _pdv2_contrast_pipeline_test as T  # noqa: E402
import _pdv2_diffattn_contrast_recalibrated as recal  # noqa: E402
from _pdv2_contrast_calibration import fit_score  # noqa: E402
from gnn.contrast_spectral_difference import ContrastSpectralDifference  # noqa: E402

pipeline = T.pipeline
e = pipeline.e
prior = e.prior
run = e.run
ORIGINAL_SCORE = recal.ORIGINAL_SCORE
OUT = ROOT / "results/pdv2_scratch_pilot_20261116"
PIPE = ROOT / "results/pdv2_diffattn_contrast_pipeline_20261104"
TEST = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
POOLS = prior.PARENT / "pools"  # gain_readout 고정 승자 표본(현재 융합)
NEWPOOLS = ROOT / "results/pdv2_scratch_pilot_20261116/step2/pools"  # 재선택 융합
SPEC = run.READ(TEST / "protocol.json")
VAL, TRAIN, TESTSEQ = SPEC["val"], SPEC["train"], SPEC["test"]
CURRENT_TUPLE = [1.0, 1.0, 2.0, 1.0]
CURRENT_W = run.cal.fusion_weights({"tuple": CURRENT_TUPLE})[run.cal.FUSIONS.index("aux_half")]
LAMBDAS = (0.0, 0.25, 0.5, 1.0)
SEED = int(os.environ.get("NSD_SCRATCH_SEED", "1"))
SUB = os.environ.get("NSD_SCRATCH_SUB")  # 예: defaultbn_3seed
LRS = {"lr1e-4": 1e-4, "lr3e-4": 3e-4, "lr3e-5": 3e-5}
SAVE_EPOCHS = {8: 512, 24: 1536, 64: 4096}
METRICS = run.base.METRICS

# scripts/_height_candidate_fusion.py:334-342 그대로
WEIGHTS = []
for wa in (0.0, 0.25, 0.5, 1.0, 2.0):
    for wc in (0.0, 0.25, 0.5, 1.0, 2.0):
        WEIGHTS.append((0.0, 1.0, wa, wc))
for wn in (0.25, 0.5, 1.0, 2.0):
    for wa in (0.0, 0.5, 1.0, 2.0):
        for wc in (0.0, 0.5, 1.0, 2.0):
            WEIGHTS.append((wn, 1.0, wa, wc))
WEIGHTS = sorted(set(WEIGHTS))


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(path)


def timing(step: str, seconds: float) -> None:
    path = OUT / "timing.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault(step, []).append(round(seconds, 1))
    write(path, data)


def inputs(role: str, sequence: str):
    """검증은 기존 입력, 시험은 기존 시험 입력 함수를 쓴다."""
    if role == "test":
        assert sequence in TESTSEQ
        return T.test_inputs.inputs(role, sequence)
    return T.VAL_INPUTS(role, sequence)


run.inputs = inputs  # graph_inputs가 시험 자료도 열 수 있게 한다


# ----------------------------------------------------------------- 점수
def components(z: torch.Tensor, plan: dict) -> tuple:
    """원본 뱅크만·보정 뱅크만의 최종 점수(같은 양의 아핀 척도)."""
    c = prior.coefficients(z)
    dummy = torch.zeros(len(z), 416, device=z.device)
    dummy[:, 0] = 1
    a0 = prior.base.score(dummy, torch.stack((c[:, 0], c[:, 0]), 1), plan, strength=0.0)
    a1 = prior.base.score(dummy, torch.stack((c[:, 1], c[:, 1]), 1), plan, strength=0.0)
    return a0, a1


def score(z: torch.Tensor, plan: dict, lam: float) -> torch.Tensor:
    """lambda=0.25는 원래 함수, 나머지는 선형 혼합."""
    if lam == 0.25:
        return ORIGINAL_SCORE(z, plan)
    a0, a1 = components(z, plan)
    return (1 - lam) * a0 + lam * a1


def batched(fn, plan_np: dict, n: int) -> np.ndarray:
    out = []
    for start in range(0, n, 128):
        pp = {k: torch.from_numpy(v[start:start + 128]).cuda() for k, v in plan_np.items()}
        out.append(fn(pp))
    return np.concatenate(out)


# ----------------------------------------------------------------- 모델
def scratch_model(seed: int = SEED) -> torch.nn.Module:
    """사전학습 없이 같은 구조를 무작위 초기화하고 같은 0 초기화를 적용한다."""
    torch.manual_seed(seed)
    ref = run.source.factory(288)
    return ContrastSpectralDifference(ref, "diff").cuda().eval()


def estimate_bn(net: torch.nn.Module) -> dict:
    """초기 가중치로 학습8을 한 번씩 통과시켜 BN 누적 통계를 구한다(시퀀스 균등)."""
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    with torch.no_grad():
        for s in TRAIN:
            graph = torch.load(recal.OUT / "train_inputs" / (s.replace("/", "_") + ".pt"), map_location="cuda", weights_only=False)
            net(graph)
            del graph
    for m in bns:
        m.momentum = 0.1
        m.eval()
    net.eval()
    return {n: int(m.num_batches_tracked) for n, m in net.named_modules() if isinstance(m, torch.nn.BatchNorm1d)}


def load_scratch(path: Path) -> torch.nn.Module:
    net = scratch_model()
    net.load_state_dict(torch.load(path, map_location="cuda", weights_only=False)["model_state_dict"], strict=True)
    return net


# ----------------------------------------------------------------- 1단계
def verify_init() -> None:
    """Q·기하 최종층·게이트 최종층 0이면 나머지 무작위 가중치와 무관하게 출력이 같다."""
    tick = time.time()
    rec = {}
    for role, s in (("train", TRAIN[0]), ("train", TRAIN[4]), ("val", "kitti360/0004")):
        graph, _, _ = prior.graph_inputs(role, s)
        fixed = prior.filtered(graph, -0.5)
        outs = {}
        with torch.no_grad():
            for seed in (1, 2, 3, 77):
                net = scratch_model(seed)
                outs[seed] = net(graph)
                strengths = net.last_strengths
                assert torch.all(strengths == -0.5)
        same = all(torch.equal(outs[1], v) for v in outs.values())
        rec[s] = dict(
            nodes=len(graph.x),
            bitwise_identical_across_random_inits=bool(same),
            max_abs_vs_uniform_subtraction=max(float((v - fixed).abs().max()) for v in outs.values()),
            input_abs_max=float(graph.x.abs().max()),
        )
    net = scratch_model(1)
    rec["trainable_parameters"] = sum(p.numel() for p in net.parameters() if p.requires_grad)
    rec["bn_running_mean_abs_max"] = max(float(m.running_mean.abs().max()) for m in net.modules() if isinstance(m, torch.nn.BatchNorm1d))
    rec["mechanism"] = "W_q=0 -> content=0; 기하 MLP 최종층 0 -> geometry=0; interaction 결합 0 -> softmax 균등; 게이트 최종층 0 -> sigmoid(0)=0.5 -> eta=-0.5. 0 * 유한값 = 0이므로 다른 가중치와 무관."
    write(OUT / "step1/init_identity.json", rec)
    timing("step1_verify_init", time.time() - tick)
    print(json.dumps(rec, indent=1))


def tuple_scores(W: np.ndarray, comp: dict) -> np.ndarray:
    """pipeline.candidate_pool/winners와 같은 float32 연산 순서."""
    static = (W[:, :1] * comp["sn"][None] + W[:, 1:2] * comp["sh"][None]).astype(np.float32)
    return static + W[:, 2:3] * comp["sa"][None] + W[:, 3:4] * comp["sc"][None]


def grid_sequence(sequence: str) -> None:
    """균등 감산에서 모든 후보 튜플의 승자와 두 뱅크 점수를 저장한다."""
    name = sequence.replace("/", "_")
    target = OUT / "step1/val" / (name + ".npz")
    if target.exists():
        return
    tick = time.time()
    cache, pos, minimum, travel, bank, _, _, _ = inputs("val", sequence)
    graph, _, _ = prior.graph_inputs("val", sequence)
    with torch.no_grad():
        z = prior.filtered(graph, -0.5)
    change = run.base.variant_inputs(z.cpu().numpy(), 8)
    raw = pipeline.raw_key(cache["descriptors"])
    pool = run.ARR(PIPE / "val" / name / "candidate_pool.npz")
    q = pool["query"]
    perturb = {}
    for w in WEIGHTS:
        for i, f in enumerate(run.cal.FUSIONS):
            row = run.cal.fusion_weights({"tuple": list(w)})[i]
            perturb[tuple(float(x) for x in row)] = 1
    tuples = [tuple(float(x) for x in w) for w in WEIGHTS]
    tuples += [t for t in perturb if t not in tuples]
    W = np.asarray(tuples, np.float32)
    winners = np.zeros((len(W), len(q)), np.int64)
    static_err = 0.0
    for n, t in enumerate(q):
        lo, hi = pool["offset"][n:n + 2]
        ids = pool["ids"][lo:hi]
        comp = dict(
            sn=run.base.minmax(raw[ids] @ raw[t]),
            sh=run.base.minmax(bank[0][ids] @ bank[0][t]),
            sa=run.base.minmax(run.base.align_scores(change[1], change[2], int(t), ids)),
            sc=run.base.minmax(run.base.column_align_scores(change[3], int(t), ids)),
        )
        cur = (CURRENT_W[0] * comp["sn"] + CURRENT_W[1] * comp["sh"]).astype(np.float32)
        static_err = max(static_err, float(np.abs(cur - pool["static"][lo:hi]).max()))
        winners[:, n] = ids[np.argmax(tuple_scores(W, comp), axis=1)]
    pairs = np.unique(np.stack((np.broadcast_to(q, winners.shape).ravel(), winners.ravel()), 1), axis=0)
    p = prior.base.prior.plan(travel, pairs[:, 0], pairs[:, 1])
    with torch.no_grad():
        a = batched(lambda pp: torch.stack(components(z, pp), 1).cpu().numpy(), p, len(pairs))
        orig = batched(lambda pp: ORIGINAL_SCORE(z, pp).cpu().numpy(), p, len(pairs))
    mix_err = float(np.abs(0.75 * a[:, 0] + 0.25 * a[:, 1] - orig).max())
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, query=q, tuples=W, winners=winners, pairs=pairs, a0=a[:, 0].astype(np.float64),
        a1=a[:, 1].astype(np.float64), original=orig.astype(np.float64),
        min_past_gt_dist=minimum[q], pos=pos,
    )
    ref = run.ARR(PIPE / "val" / name / "fixed.npz")
    cur_i = tuples.index(tuple(float(x) for x in CURRENT_W))
    write(target.with_suffix(".json"), dict(
        sequence=sequence, queries=len(q), tuples=len(W), unique_pairs=len(pairs),
        static_max_error_vs_pool=static_err, mixture_linearity_max_error=mix_err,
        current_tuple_winners_equal_existing_fixed=bool(np.array_equal(winners[cur_i], ref["j_star"])),
        seconds=time.time() - tick))
    timing("step1_grid_" + name, time.time() - tick)
    print("GRID", sequence, len(pairs), static_err, mix_err, flush=True)


def grid_metrics(lam_list, tuple_list) -> dict:
    """튜플·lambda별 시퀀스 지표."""
    out = {}
    for s in VAL:
        d = run.ARR(OUT / "step1/val" / (s.replace("/", "_") + ".npz"))
        tuples = [tuple(float(x) for x in w) for w in d["tuples"]]
        index = {(int(a), int(b)): i for i, (a, b) in enumerate(d["pairs"])}
        q, pos = d["query"], d["pos"]
        for t in tuple_list:
            j = d["winners"][tuples.index(t)]
            ix = np.fromiter((index[(int(a), int(b))] for a, b in zip(q, j)), np.int64, len(q))
            row = dict(query_idx=q, j_star=j, min_past_gt_dist=d["min_past_gt_dist"],
                       top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1))
            for lam in lam_list:
                value = d["original"][ix] if lam == 0.25 else (1 - lam) * d["a0"][ix] + lam * d["a1"][ix]
                out.setdefault((t, lam), {})[s] = run.base.independent_metrics(row, value)
            positive = row["min_past_gt_dist"] <= 5
            out.setdefault((t, "r1count"), {})[s] = (int((positive & (row["top1_geo_dist"] <= 5)).sum()), int(positive.sum()))
    return out


def select_fixed() -> None:
    tick = time.time()
    fam = run.base.macro.__globals__["family"]
    # S1: WEIGHTS 격자, 계열 안 질의 합산 R@1의 계열 평균 argmax
    tw = [tuple(float(x) for x in w) for w in WEIGHTS]
    m = grid_metrics([0.25], tw)
    s1 = {}
    for t in tw:
        counts = m[(t, "r1count")]
        fams = {}
        for s, (c, n) in counts.items():
            f = fam(s)
            fams.setdefault(f, [0, 0])
            fams[f][0] += c
            fams[f][1] += n
        s1[t] = float(np.mean([c / n for c, n in fams.values() if n]))
    base_t = max(tw, key=lambda t: s1[t])  # 동률은 WEIGHTS 순서 첫째(max는 첫 최대 반환)
    ranking = sorted(tw, key=lambda t: -s1[t])[:8]
    # S2: 혼합
    m2 = grid_metrics(LAMBDAS, [base_t])
    mac = {lam: run.base.macro(m2[(base_t, lam)]) for lam in LAMBDAS}

    def blend_pass(lam):
        b, c = mac[0.0], mac[lam]
        d = {k: c["macro"][k] - b["macro"][k] for k in METRICS}
        fam_loss = min(c["families"][f]["AP"] - b["families"][f]["AP"] for f in b["families"])
        ok = d["AP"] >= 0.001 and d["AP"] > 0 and d["R@100P"] >= -0.001 and fam_loss >= -0.0025
        return ok, d, fam_loss
    s2 = {lam: blend_pass(lam) for lam in LAMBDAS[1:]}
    passing = [lam for lam in LAMBDAS[1:] if s2[lam][0]]
    lam_sel = max(passing, key=lambda lam: (mac[lam]["macro"]["AP"], -lam)) if passing else 0.0
    # S3: 한 축 섭동
    rows = run.cal.fusion_weights({"tuple": list(base_t)})
    pert = [tuple(float(x) for x in r) for r in rows]
    m3 = grid_metrics([lam_sel], pert)
    mac3 = {t: run.base.macro(m3[(t, lam_sel)]) for t in pert}
    b = mac3[pert[0]]

    def pert_pass(t):
        c = mac3[t]
        d = {k: c["macro"][k] - b["macro"][k] for k in METRICS}
        fam_loss = min(c["families"][f]["AP"] - b["families"][f]["AP"] for f in b["families"])
        ok = (d["AP"] >= 0.005 and d["max_F1"] >= 0.005 and d["AP"] >= -1e-11 and d["R@1"] >= -1e-11
              and d["R@100P"] >= -0.01 and fam_loss >= -0.005)
        return ok, d, fam_loss
    s3 = {t: pert_pass(t) for t in pert[1:]}
    passing3 = [t for t in pert[1:] if s3[t][0]]
    t_sel = max(passing3, key=lambda t: mac3[t]["macro"]["AP"]) if passing3 else pert[0]
    # 참고: 현재 설정 (1,.5,2,1), 0.25 의 균등 감산 검증 지표
    cur = tuple(float(x) for x in CURRENT_W)
    mcur = grid_metrics([0.25, lam_sel], [cur, t_sel])
    rec = dict(
        S1=dict(selected=base_t, family_macro_R1=s1[base_t], current_base=CURRENT_TUPLE,
                current_base_R1=s1[tuple(CURRENT_TUPLE)],
                top8=[dict(tuple=t, R1=s1[t]) for t in ranking]),
        S2=dict(selected_lambda=lam_sel, macro={str(k): v for k, v in mac.items()},
                checks={str(k): dict(passed=v[0], delta=v[1], worst_family_AP=v[2]) for k, v in s2.items()}),
        S3=dict(selected=t_sel, baseline=pert[0], macro={str(k): v for k, v in mac3.items()},
                checks={str(k): dict(passed=v[0], delta=v[1], worst_family_AP=v[2]) for k, v in s3.items()}),
        selected=dict(weights=list(t_sel), lam=lam_sel),
        current=dict(weights=list(cur), lam=0.25,
                     macro=run.base.macro(mcur[(cur, 0.25)])),
        selected_macro=run.base.macro(mcur[(t_sel, lam_sel)]),
        differs_from_current=bool(list(t_sel) != list(cur) or lam_sel != 0.25),
        test_used=False,
    )
    write(OUT / "step1/selection.json", rec)
    timing("step1_select", time.time() - tick)
    print(json.dumps({k: rec[k] for k in ("selected", "current", "differs_from_current")}, indent=1))
    print("S1", base_t, s1[base_t], "S2", lam_sel, "S3", t_sel)


def selected() -> tuple:
    rec = run.READ(OUT / "step1/selection.json")["selected"]
    return np.asarray(rec["weights"], np.float32), float(rec["lam"])


def build_pool(role: str, sequence: str, weights: np.ndarray, folder: Path) -> dict:
    """gain_readout.pool과 같은 원키·원보조 고정 색인 승자 표본을 주어진 융합 가중치로 만든다."""
    target = folder / role / sequence.replace("/", "_") / "pool.npz"
    if target.exists():
        return run.ARR(target)
    gr = prior.base
    cache, pos, minimum, travel, bank, _, _, _ = inputs(role, sequence)
    frozen = gr.prior.previous.prior.gate.MODEL(gr.prior.PARENT / "recenter_frozen/epoch0.pth")
    key = run.source.make_keys(frozen, cache["descriptors"], cache["poses"])
    raw = key[:, :288].copy()
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    base = run.base
    q = np.arange(30, len(raw) - 15, dtype=np.int64)
    winner = []
    for t in q:
        ids = np.unique(np.r_[base.prefix_topk(raw, t, t - 29, 100), base.prefix_topk(bank[0], t, t - 29, 100)])
        a = base.align_scores(bank[1], bank[2], int(t), ids)
        c = base.column_align_scores(bank[3], int(t), ids)
        sc = (weights[0] * base.minmax(raw[ids] @ raw[t]) + weights[1] * base.minmax(bank[0][ids] @ bank[0][t])
              + weights[2] * base.minmax(a) + weights[3] * base.minmax(c))
        winner.append(ids[np.argmax(sc)])
    j = np.asarray(winner, np.int64)
    assert (j <= q - 30).all()
    row = dict(query_idx=q, j_star=j, min_past_gt_dist=minimum[q],
               top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1), group=np.zeros(len(q), np.int64))
    p = gr.prior.plan(travel, row["query_idx"], row["j_star"])
    eligible = (row["min_past_gt_dist"] <= 5) | (row["min_past_gt_dist"] > 25)
    correct = (row["min_past_gt_dist"] <= 5) & (row["top1_geo_dist"] <= 5)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **row, **{f"plan_{k}": v for k, v in p.items()}, eligible=eligible, correct=correct)
    return run.ARR(target)


def pools(mode: str) -> None:
    """mode=check: 현재 가중치로 기존 표본 재현. mode=build: 재선택 가중치로 학습8·검증9 표본."""
    tick = time.time()
    if mode == "check":
        rec = {}
        for role, s in (("train", "kitti/05"), ("val", "kitti360/0004")):
            d = build_pool(role, s, CURRENT_W, OUT / "step2/pool_check")
            ref = run.ARR(POOLS / role / s.replace("/", "_") / "pool.npz")
            rec[s] = {k: bool(np.array_equal(d[k], ref[k])) for k in ref}
        write(OUT / "step2/pool_check.json", rec)
        print(rec)
    else:
        w, _ = selected()
        rec = {}
        for role, seqs in (("train", TRAIN), ("val", VAL)):
            for s in seqs:
                d = build_pool(role, s, w, OUT / "step2/pools")
                ref = run.ARR(POOLS / role / s.replace("/", "_") / "pool.npz")
                rec[s] = dict(rows=len(d["query_idx"]), correct=int(d["correct"].sum()), eligible=int(d["eligible"].sum()),
                              changed_winners_vs_current=int((d["j_star"] != ref["j_star"]).sum()),
                              correct_current=int(ref["correct"].sum()))
                print("POOL", role, s, rec[s], flush=True)
        write(OUT / "step2/pools_summary.json", dict(weights=w.tolist(), sequences=rec))
    timing("step2_pools_" + mode, time.time() - tick)


# ----------------------------------------------------------------- 2단계
def calibrate() -> None:
    """초기 출력(=균등 감산)의 학습8 점수에 양의 로지스틱 척도를 적합해 고정한다."""
    tick = time.time()
    _, lam = selected()
    xs, ys, ws = [], [], []
    for s in TRAIN:
        name = s.replace("/", "_")
        data = run.ARR(NEWPOOLS / "train" / name / "pool.npz")
        graph = torch.load(recal.OUT / "train_inputs" / (name + ".pt"), map_location="cuda", weights_only=False)
        plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
        with torch.no_grad():
            z = scratch_model()(graph)
            v = batched(lambda pp: score(z, pp, lam).cpu().numpy(), plan, len(data["query_idx"])).astype(np.float64)
        keep = data["eligible"]
        y = data["correct"][keep].astype(np.float64)
        b = np.zeros(len(y))
        classes = np.unique(y)
        for c in classes:
            b[y == c] = 1.0 / (len(classes) * int((y == c).sum()))
        xs.append(v[keep]); ys.append(y); ws.append(b)
        del graph, z
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    fit = fit_score(x, y, w / w.sum())
    old = run.READ(recal.OUT / "protocol.json")["loss_fit"]
    write(OUT / "step2/loss_fit.json", dict(
        fit=fit, lam=lam, existing_current_fit=old,
        same_as_existing=bool(abs(fit["mean"][0] - old["mean"][0]) < 1e-6 and abs(fit["coefficients"][0] - old["coefficients"][0]) < 1e-6),
        validation_used=False))
    timing("step2_calibrate", time.time() - tick)
    print("FIT", fit, "OLD", old)


def train(tag: str) -> None:
    """선택한 혼합·고정 척도로 처음부터 4096회 학습, 512/1536/4096 저장."""
    default_bn = tag.startswith("bn0_")
    lr = LRS[tag.replace("bn0_", "")]
    folder = OUT / "step2" / tag if not SUB else OUT / SUB / f"s{SEED}" / tag
    if (folder / "training_complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    _, lam = selected()
    fit = run.READ(OUT / "step2/loss_fit.json")["fit"]
    net = scratch_model(SEED)
    bn = None if default_bn else estimate_bn(net)
    with torch.no_grad():
        g0 = torch.load(recal.OUT / "train_inputs" / (TRAIN[0].replace("/", "_") + ".pt"), map_location="cuda", weights_only=False)
        assert torch.equal(net(g0), prior.filtered(g0, -0.5))
        del g0
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    graphs = {s: torch.load(recal.OUT / "train_inputs" / (s.replace("/", "_") + ".pt"), map_location="cuda", weights_only=False) for s in TRAIN}
    pools = {s: run.ARR(NEWPOOLS / "train" / s.replace("/", "_") / "pool.npz") for s in TRAIN}
    plans = {s: {k: torch.from_numpy(d["plan_" + k]).cuda() for k in ("query", "candidate", "weight")} for s, d in pools.items()}
    records, hashes, updates, tick = [], {}, 0, time.time()
    for epoch in range(1, max(SAVE_EPOCHS) + 1):
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
                value = score(z, {k: v[ix] for k, v in plans[s].items()}, lam)
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
            records.append(dict(epoch=epoch, sequence=s, loss=float(np.mean(losses)), grad=float(np.mean(grads)),
                                strength_mean=float(net.last_strengths.detach().mean())))
        if epoch in SAVE_EPOCHS:
            assert updates == SAVE_EPOCHS[epoch]
            path = folder / f"u{updates}.pth"
            torch.save(dict(model_state_dict=net.state_dict(), epoch=epoch, updates=updates, lr=lr, seed=SEED), path)
            hashes[path.name] = run.HASH(path)
            print("TRAIN", tag, epoch, updates, round(time.time() - tick, 1), np.mean([r["loss"] for r in records[-8:]]), flush=True)
    changed = [k for k, v in net.state_dict().items() if not torch.equal(v, initial[k])]
    write(folder / "training_complete.json", dict(
        status="passed", lr=lr, seed=SEED, updates=updates, bn_estimated_batches=bn, checkpoints=hashes, records=records,
        changed=changed, trainable_parameters=sum(p.numel() for p in params), seconds=time.time() - tick))
    timing("step2_train_" + tag, time.time() - tick)


def val_fixed() -> None:
    """원키 검색 승자를 고정한 검증9에서 최종 판정만 다시 계산해 AP로 고른다."""
    tick = time.time()
    _, lam = selected()
    cases = {"uniform": None}
    for tag in LRS:
        for u in SAVE_EPOCHS.values():
            cases[f"{tag}_u{u}"] = OUT / "step2" / tag / f"u{u}.pth"
    for tag in LRS:
        for u in SAVE_EPOCHS.values():
            cases[f"bn0_{tag}_u{u}"] = OUT / "step2" / ("bn0_" + tag) / f"u{u}.pth"
    cases["current_diff_s1"] = "current"
    per = {k: {} for k in cases}
    for s in VAL:
        name = s.replace("/", "_")
        graph, _, _ = prior.graph_inputs("val", s)
        data = run.ARR(NEWPOOLS / "val" / name / "pool.npz")
        row = {k: data[k] for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist")}
        plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
        with torch.no_grad():
            for tag, ck in cases.items():
                if ck is None:
                    z = prior.filtered(graph, -0.5)
                elif ck == "current":
                    z = pipeline.load_model(SPEC["cases"]["diff_s1"])(graph)
                else:
                    z = load_scratch(ck)(graph)
                v = batched(lambda pp: score(z, pp, lam).cpu().numpy(), plan, len(row["query_idx"])).astype(np.float64)
                per[tag][s] = run.base.independent_metrics(row, v)
        print("VALFIXED", s, flush=True)
    res = {k: run.base.macro(v) for k, v in per.items()}
    order = [k for k in cases if k.startswith("lr")]
    # 동률: 적은 갱신, 그다음 현재 학습률 순서
    rank = sorted(order, key=lambda k: (-res[k]["macro"]["AP"], int(k.split("_u")[1]), list(LRS).index(k.split("_u")[0])))
    write(OUT / "step2/val_fixed_winner.json", dict(results=res, per_sequence=per, selected=rank[0], lam=lam, test_used=False))
    timing("step2_val_fixed", time.time() - tick)
    for k in cases:
        print(k, {m: round(100 * res[k]["macro"][m], 3) for m in METRICS})
    print("SELECTED", rank[0])


# ----------------------------------------------------------------- 3단계
def cases_for_eval() -> dict:
    w, lam = selected()
    sel = run.READ(OUT / "step2/val_fixed_winner.json")["selected"]
    tag, u = sel.rsplit("_u", 1)
    return {
        "repro_fixed": dict(kind="fixed", weights=CURRENT_W, lam=0.25, ref="fixed"),
        "repro_diff_s1": dict(kind="current", weights=CURRENT_W, lam=0.25, ref="diff_s1"),
        "uniform_reselected": dict(kind="fixed", weights=w, lam=lam, ref=None),
        "scratch_s1": dict(kind="scratch", weights=w, lam=lam, ref=None, checkpoint=str(OUT / "step2" / tag / f"u{u}.pth")),
    }


def pipe(role: str, sequence: str) -> None:
    """matching_only 실제 검색(합집합 후보·새 정렬 승자)과 최종 판정."""
    name = sequence.replace("/", "_")
    folder = OUT / "step3" / role / name
    if (folder / "summary.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    src = (PIPE / "val" / name) if role == "val" else (TEST / "test" / name)
    cache, pos, minimum, travel, bank, _, _, _ = inputs(role, sequence)
    graph, _, _ = prior.graph_inputs(role, sequence)
    raw = pipeline.raw_key(cache["descriptors"])
    poolpath = src / "candidate_pool.npz"
    assert run.HASH(poolpath) == run.READ(poolpath.with_suffix(".json"))["sha256"]
    base_pool = run.ARR(poolpath)
    metrics, checks = {}, {}
    for tag, case in cases_for_eval().items():
        weights = np.asarray(case["weights"], np.float32)
        if np.array_equal(weights, CURRENT_W):
            pool = base_pool
        else:
            static = []
            for n, q in enumerate(base_pool["query"]):
                lo, hi = base_pool["offset"][n:n + 2]
                ids = base_pool["ids"][lo:hi]
                static.append(weights[0] * run.base.minmax(raw[ids] @ raw[q]) + weights[1] * run.base.minmax(bank[0][ids] @ bank[0][q]))
            pool = dict(base_pool, static=np.concatenate(static).astype(np.float32))
        with torch.no_grad():
            if case["kind"] == "fixed":
                net = None
                z = prior.filtered(graph, -0.5)
            elif case["kind"] == "current":
                net = pipeline.load_model(SPEC["cases"]["diff_s1"])
                z = net(graph)
            else:
                net = load_scratch(Path(case["checkpoint"]))
                z = net(graph)
            change = run.base.variant_inputs(z.cpu().numpy(), 8)
            q = pool["query"]
            j = pipeline.winners(pool, change, weights)
            assert (j <= q - 30).all()
            row = dict(query_idx=q, j_star=j, min_past_gt_dist=minimum[q],
                       top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1), group=np.zeros(len(q), np.int64))
            plan = prior.base.prior.plan(travel, q, j)
            value = batched(lambda pp: score(z, pp, case["lam"]).cpu().numpy(), plan, len(q)).astype(np.float64)
        metric = run.base.independent_metrics(row, value)
        other = run.white.prior.BASE.exp.metrics(value, row)
        for m in METRICS:
            if metric[m] is not None:
                np.testing.assert_allclose(metric[m], other[m], atol=1e-12, rtol=0)
        if case["ref"]:
            ref = run.ARR(src / (case["ref"] + ".npz"))
            checks[tag] = dict(
                winners_equal=bool(np.array_equal(j, ref["j_star"])),
                query_equal=bool(np.array_equal(q, ref["query_idx"])),
                score_bitwise_equal=bool(np.array_equal(value, ref["score"])),
                score_max_error=float(np.abs(value - ref["score"]).max()),
                metrics_equal=all(metric[m] == run.READ(src / (case["ref"] + ".json"))["metrics"][m] for m in METRICS),
            )
        if tag == "scratch_s1":
            # q+15 접두 검사 두 지점
            pre = []
            for at in (len(q) // 3, 2 * len(q) // 3):
                t = int(q[at])
                end = t + 16
                with torch.no_grad():
                    small, _, _ = prior.graph_inputs(role, sequence, end)
                    short = net(small)
                    sb = run.base.variant_inputs(short.cpu().numpy(), 8)
                    winner = run.cal.candidates(raw[:end], (bank[0][:end], *sb[1:]), weights[None], t)
                    pp = {k: torch.from_numpy(v).cuda() for k, v in prior.base.prior.plan(travel[:end], np.array([t]), winner).items()}
                    chk = score(short, pp, case["lam"]).cpu().numpy()
                pre.append(dict(query=t, winner_equal=int(winner[0]) == int(j[at]), score_error=float(abs(chk - value[at]).max())))
            checks["prefix_scratch"] = pre
        np.savez_compressed(folder / (tag + ".npz"), **row, score=value)
        metrics[tag] = metric
        del net, z, change
    write(folder / "summary.json", dict(metrics=metrics, checks=checks, seconds=time.time() - tick))
    print("PIPE", role, sequence, round(time.time() - tick, 1), json.dumps(checks)[:300], flush=True)


def finish() -> None:
    """검증9·시험46 집계, 짝지은 조건부 구간."""
    tags = list(cases_for_eval())
    out = {}
    for role, seqs in (("val", VAL), ("test", TESTSEQ)):
        per = {t: {} for t in tags}
        checks = {}
        for s in seqs:
            rec = run.READ(OUT / "step3" / role / s.replace("/", "_") / "summary.json")
            for t in tags:
                per[t][s] = rec["metrics"][t]
            checks[s] = rec["checks"]
        if role == "val":
            res = {t: dict(**run.base.macro(per[t]), sequences=per[t]) for t in tags}
        else:
            res = {t: dict(**run.current.adoption.previous.macro(per[t], run.current.CITY), sequences=per[t]) for t in tags}
            n = sum(len(run.ARR(OUT / "step3/test" / s.replace("/", "_") / "scratch_s1.npz")["query_idx"]) for s in seqs)
            assert n == 195726, n
        repro = {t: all(c[t]["winners_equal"] and c[t]["score_bitwise_equal"] and c[t]["metrics_equal"] for c in checks.values())
                 for t in ("repro_fixed", "repro_diff_s1")}
        prefix = [x for c in checks.values() for x in c["prefix_scratch"]]
        out[role] = dict(results=res, reproduction_exact=repro,
                         reproduction_max_score_error={t: max(c[t]["score_max_error"] for c in checks.values()) for t in repro},
                         prefix_checks=dict(n=len(prefix), all_winner_equal=all(x["winner_equal"] for x in prefix),
                                            max_score_error=max(x["score_error"] for x in prefix)))
    # 구간: _pdv2_contrast_pipeline_report와 같은 방식(모델 하나씩 고정)
    per = out["test"]["results"]
    intervals = {}
    for a, b in (("scratch_s1", "repro_diff_s1"), ("scratch_s1", "uniform_reselected"), ("uniform_reselected", "repro_fixed")):
        rng = np.random.default_rng(20261104)
        draws, city = [], {}
        for c, seqs in run.current.CITY.items():
            avail = [q for q in seqs if per[a]["sequences"][q]["n_pos"]]
            mat = np.array([[100 * (per[a]["sequences"][q][m] - per[b]["sequences"][q][m]) for m in METRICS] for q in avail])
            city[c] = dict(zip(METRICS, mat.mean(0).tolist()))
            idx = rng.integers(0, len(avail), size=(10000, len(avail)))
            draws.append(mat[idx].mean(1))
        boot = np.stack(draws).mean(0)
        intervals[f"{a}-{b}"] = dict(
            point_pp={m: 100 * (per[a]["macro"][m] - per[b]["macro"][m]) for m in METRICS},
            conditional_95_pp={m: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, m in enumerate(METRICS)},
            city_delta_pp=city)
    old = run.READ(TEST / "summary.json")
    out["intervals"] = intervals
    out["existing"] = dict(diff_s1=old["per_case"]["diff_s1"]["macro"], diff_mean=old["mean"]["diff"],
                           diff_std=old["std"]["diff"], fixed=old["per_case"]["fixed"]["macro"],
                           bev_native_mean=old["mean"]["bev_native"], bev_native_std=old["std"]["bev_native"])
    out["test_used_for_selection"] = False
    write(OUT / "summary.json", out)
    for role in ("val", "test"):
        print(role, out[role]["reproduction_exact"], out[role]["prefix_checks"])
        for t in tags:
            print("  ", t, {m: round(100 * out[role]["results"][t]["macro"][m], 3) for m in METRICS})
    for k, v in intervals.items():
        print(k, {m: (round(v["point_pp"][m], 3), [round(x, 3) for x in v["conditional_95_pp"][m]]) for m in METRICS})


D3 = OUT / "defaultbn_3seed"
D3_TAGS = {f"s{sd}_u{u}": D3 / f"s{sd}" / "bn0_lr3e-4" / f"u{u}.pth" for sd in (1, 2, 3) for u in (1536, 4096)}


def d3_val_fixed() -> None:
    tick = time.time()
    _, lam = selected()
    per = {k: {} for k in D3_TAGS}
    for s in VAL:
        name = s.replace("/", "_")
        graph, _, _ = prior.graph_inputs("val", s)
        data = run.ARR(NEWPOOLS / "val" / name / "pool.npz")
        row = {k: data[k] for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist")}
        plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
        with torch.no_grad():
            for tag, ck in D3_TAGS.items():
                z = load_scratch(ck)(graph)
                v = batched(lambda pp: score(z, pp, lam).cpu().numpy(), plan, len(row["query_idx"])).astype(np.float64)
                per[tag][s] = run.base.independent_metrics(row, v)
    res = {k: run.base.macro(v) for k, v in per.items()}
    write(D3 / "val_fixed_winner.json", dict(results=res, per_sequence=per, lam=lam))
    timing("d3_val_fixed", time.time() - tick)
    for k in D3_TAGS:
        print(k, {m: round(100 * res[k]["macro"][m], 3) for m in METRICS})


def d3_pipe(role: str, sequence: str) -> None:
    name = sequence.replace("/", "_")
    folder = D3 / "step3" / role / name
    if (folder / "summary.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    src = (PIPE / "val" / name) if role == "val" else (TEST / "test" / name)
    cache, pos, minimum, travel, bank, _, _, _ = inputs(role, sequence)
    graph, _, _ = prior.graph_inputs(role, sequence)
    raw = pipeline.raw_key(cache["descriptors"])
    weights, lam = selected()
    base_pool = run.ARR(src / "candidate_pool.npz")
    static = []
    for n, q in enumerate(base_pool["query"]):
        lo, hi = base_pool["offset"][n:n + 2]
        ids = base_pool["ids"][lo:hi]
        static.append(weights[0] * run.base.minmax(raw[ids] @ raw[q]) + weights[1] * run.base.minmax(bank[0][ids] @ bank[0][q]))
    pool = dict(base_pool, static=np.concatenate(static).astype(np.float32))
    q = pool["query"]
    metrics, checks = {}, {}
    for tag, ck in D3_TAGS.items():
        with torch.no_grad():
            net = load_scratch(ck)
            z = net(graph)
            change = run.base.variant_inputs(z.cpu().numpy(), 8)
            j = pipeline.winners(pool, change, weights)
            assert (j <= q - 30).all()
            row = dict(query_idx=q, j_star=j, min_past_gt_dist=minimum[q],
                       top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1), group=np.zeros(len(q), np.int64))
            plan = prior.base.prior.plan(travel, q, j)
            value = batched(lambda pp: score(z, pp, lam).cpu().numpy(), plan, len(q)).astype(np.float64)
            at = len(q) // 2
            t = int(q[at])
            small, _, _ = prior.graph_inputs(role, sequence, t + 16)
            short = net(small)
            sb = run.base.variant_inputs(short.cpu().numpy(), 8)
            winner = run.cal.candidates(raw[:t + 16], (bank[0][:t + 16], *sb[1:]), weights[None], t)
            pp = {k: torch.from_numpy(v).cuda() for k, v in prior.base.prior.plan(travel[:t + 16], np.array([t]), winner).items()}
            chk = score(short, pp, lam).cpu().numpy()
        metric = run.base.independent_metrics(row, value)
        other = run.white.prior.BASE.exp.metrics(value, row)
        for m in METRICS:
            if metric[m] is not None:
                np.testing.assert_allclose(metric[m], other[m], atol=1e-12, rtol=0)
        checks[tag] = dict(prefix_winner_equal=int(winner[0]) == int(j[at]), prefix_score_error=float(abs(chk - value[at]).max()))
        np.savez_compressed(folder / (tag + ".npz"), **row, score=value)
        metrics[tag] = metric
        del net, z, change, small, short
    write(folder / "summary.json", dict(metrics=metrics, checks=checks, seconds=time.time() - tick))
    print("D3PIPE", role, sequence, round(time.time() - tick, 1), flush=True)


def d3_campaign(role: str) -> None:
    import subprocess
    import concurrent.futures
    tick = time.time()
    seqs = VAL if role == "val" else TESTSEQ
    (D3 / "logs").mkdir(parents=True, exist_ok=True)

    def call(s):
        with (D3 / "logs" / f"{role}_{s.replace('/', '_')}.log").open("a") as f:
            subprocess.run([sys.executable, __file__, "d3_pipe", "--role", role, "--sequence", s], stdout=f, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        for fut in concurrent.futures.as_completed([ex.submit(call, s) for s in seqs]):
            fut.result()
    timing("d3_" + role, time.time() - tick)


def d3_finish() -> None:
    pilot = run.READ(OUT / "summary.json")
    old = run.READ(TEST / "summary.json")
    out = {}
    for role, seqs in (("val", VAL), ("test", TESTSEQ)):
        per = {t: {} for t in D3_TAGS}
        chk = []
        for s in seqs:
            rec = run.READ(D3 / "step3" / role / s.replace("/", "_") / "summary.json")
            for t in D3_TAGS:
                per[t][s] = rec["metrics"][t]
            chk += list(rec["checks"].values())
        if role == "val":
            res = {t: dict(**run.base.macro(per[t]), sequences=per[t]) for t in D3_TAGS}
        else:
            res = {t: dict(**run.current.adoption.previous.macro(per[t], run.current.CITY), sequences=per[t]) for t in D3_TAGS}
        summ = {}
        for u in (1536, 4096):
            vals = {m: [res[f"s{sd}_u{u}"]["macro"][m] for sd in (1, 2, 3)] for m in METRICS}
            summ[f"u{u}"] = {m: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), seeds=v) for m, v in vals.items()}
        out[role] = dict(results=res, three_seed=summ, prefix_all_equal=all(c["prefix_winner_equal"] for c in chk),
                         prefix_max_error=max(c["prefix_score_error"] for c in chk))
    # 구간
    test = out["test"]["results"]
    uni = pilot["test"]["results"]["uniform_reselected"]["sequences"]
    intervals = {}
    for u in (1536, 4096):
        for other in ("uniform_reselected", "current_nsd_3start"):
            rng = np.random.default_rng(20261104)
            draws = []
            for c, seqs in run.current.CITY.items():
                avail = [q for q in seqs if uni[q]["n_pos"]]
                mat = []
                for q in avail:
                    rowv = []
                    for m in METRICS:
                        diffs = []
                        for sd in (1, 2, 3):
                            a = test[f"s{sd}_u{u}"]["sequences"][q][m]
                            b = uni[q][m] if other == "uniform_reselected" else old["per_case"][f"diff_s{sd}"]["sequences"][q][m]
                            diffs.append(100 * (a - b))
                        rowv.append(np.mean(diffs))
                    mat.append(rowv)
                mat = np.array(mat)
                idx = rng.integers(0, len(avail), size=(10000, len(avail)))
                draws.append(mat[idx].mean(1))
            boot = np.stack(draws).mean(0)
            ref = (pilot["test"]["results"]["uniform_reselected"]["macro"] if other == "uniform_reselected" else old["mean"]["diff"])
            intervals[f"u{u}-{other}"] = dict(
                point_pp={m: 100 * (out["test"]["three_seed"][f"u{u}"][m]["mean"] - ref[m]) for m in METRICS},
                conditional_95_pp={m: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, m in enumerate(METRICS)})
    out["intervals"] = intervals
    write(D3 / "summary.json", out)
    for role in ("val", "test"):
        print(role, out[role]["prefix_all_equal"], out[role]["prefix_max_error"])
        for u, v in out[role]["three_seed"].items():
            print("  ", u, {m: (round(100 * x["mean"], 3), round(100 * x["sd"], 3), [round(100 * y, 3) for y in x["seeds"]]) for m, x in v.items()})
    for k, v in intervals.items():
        print(k, {m: (round(v["point_pp"][m], 3), [round(x, 3) for x in v["conditional_95_pp"][m]]) for m in METRICS})


def campaign(role: str) -> None:
    import subprocess
    import concurrent.futures
    tick = time.time()
    seqs = VAL if role == "val" else TESTSEQ
    (OUT / "logs").mkdir(exist_ok=True)

    def call(s):
        with (OUT / "logs" / f"{role}_{s.replace('/', '_')}.log").open("a") as f:
            subprocess.run([sys.executable, __file__, "pipe", "--role", role, "--sequence", s], stdout=f, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        for i, fut in enumerate(concurrent.futures.as_completed([ex.submit(call, s) for s in seqs]), 1):
            fut.result()
            print("DONE", role, i, len(seqs), flush=True)
    timing("step3_" + role, time.time() - tick)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action")
    p.add_argument("--sequence", default="kitti360/0004")
    p.add_argument("--role", default="val")
    p.add_argument("--tag", default="lr1e-4")
    a = p.parse_args()
    e.configure()
    if a.action == "verify_init":
        verify_init()
    elif a.action == "grid":
        for s in VAL:
            grid_sequence(s)
    elif a.action == "grid_one":
        grid_sequence(a.sequence)
    elif a.action == "select_fixed":
        select_fixed()
    elif a.action == "pools":
        pools(a.role)
    elif a.action == "calibrate":
        calibrate()
    elif a.action == "train":
        train(a.tag)
    elif a.action == "val_fixed":
        val_fixed()
    elif a.action == "pipe":
        pipe(a.role, a.sequence)
    elif a.action == "d3_val_fixed":
        d3_val_fixed()
    elif a.action == "d3_pipe":
        d3_pipe(a.role, a.sequence)
    elif a.action == "d3_campaign":
        d3_campaign(a.role)
    elif a.action == "d3_finish":
        d3_finish()
    elif a.action == "finish":
        finish()
    elif a.action == "campaign":
        campaign(a.role)
    else:
        raise SystemExit(a.action)
