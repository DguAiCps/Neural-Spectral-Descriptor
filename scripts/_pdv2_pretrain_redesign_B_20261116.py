#!/usr/bin/env python3
"""두 단계 학습 재설계 B: 1단계 정렬 순위 대조 학습 -> 2단계 수락 점수 BCE (1시드).

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  python3 scripts/_pdv2_pretrain_redesign_B_20261116.py <action>

1단계: 학습8 질의마다 두 색인 상위 100+100 합집합 후보 전부(최대 200)에서, 재선택 융합
(2,1,2,2)의 장소 선택 점수 s(q,j)를 온도 TAU로 나눈 softmax 교차 엔트로피(양성 log-sum-exp)를
줄인다. 두 코사인 항은 고정, 두 정렬 항만 수정 계수에서 미분 가능하게 계산한다.
2단계: 1단계 가중치에서 그대로 이어서 처음부터 학습 시범과 같은 BCE.
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
from _pdv2_contrast_calibration import fit_score  # noqa: E402

run, prior, pipeline, e, recal = P.run, P.prior, P.pipeline, P.e, P.recal
OUT = ROOT / "results/pdv2_pretrain_redesign_20261116/B"
PILOT = P.OUT
TRAIN, VAL, TESTSEQ = P.TRAIN, P.VAL, P.TESTSEQ
METRICS = P.METRICS
SEED = int(os.environ.get("NSD_B_SEED", "1"))
RUN = OUT if SEED == 1 else OUT / f"s{SEED}"  # amendment_2: 시드 2·3은 하위 폴더
TAU = 0.1
S1_LR = 3e-4
S1_QUERIES = 128  # 갱신당 질의 수(각 질의의 후보 전부)
S1_SAVE = {8: 448, 16: 896, 32: 1792}  # amendment_1: 적격 질의 없는 kitti/01 제외, 7시퀀스
S1_FINAL = 1792
S2_LRS = {"lr3e-4": 3e-4, "lr1e-4": 1e-4} if SEED == 1 else {"lr3e-4": 3e-4}  # amendment_2
S2_SAVE = P.SAVE_EPOCHS  # {8: 512, 24: 1536, 64: 4096}
KMAX = 200


def write(path: Path, obj) -> None:
    P.write(path, obj)


def timing(step: str, seconds: float, **extra) -> None:
    path = RUN / "timing.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault(step, []).append(dict(seconds=round(seconds, 1), **extra))
    write(path, data)


def name_of(s: str) -> str:
    return s.replace("/", "_")


def train_graph(s: str):
    return torch.load(recal.OUT / "train_inputs" / (name_of(s) + ".pt"), map_location="cuda", weights_only=False)


# ----------------------------------------------------------------- 모델
def new_model() -> torch.nn.Module:
    """시드 1 무작위 초기화 + Q·기하 최종층·게이트 최종층 0, BN은 학습8에서 초기 가중치로 추정 후 고정."""
    net = P.scratch_model(SEED)
    bn = P.estimate_bn(net)
    return net, bn


def load(path: Path) -> torch.nn.Module:
    net = P.scratch_model(SEED)
    net.load_state_dict(torch.load(path, map_location="cuda", weights_only=False)["model_state_dict"], strict=True)
    return net.eval()


# ----------------------------------------------------------------- 미분 가능한 장소 선택 점수
def tables(z: torch.Tensor):
    """_xs_eval_lib.variant_inputs(z, 8)의 hfft·hnorm·열 정렬 뱅크를 torch로."""
    z8 = z[:, :, :8]
    n = len(z8)
    zero1 = torch.zeros(n, 16, 1, dtype=z8.dtype, device=z8.device)
    zero22 = torch.zeros(n, 16, 22, dtype=z8.dtype, device=z8.device)
    hfft = torch.cat((zero1, z8, zero22), -1)
    hnorm = torch.linalg.norm(z8.abs().reshape(n, -1), dim=1).clamp_min(1e-8)
    recon = torch.fft.irfft(hfft, n=60, dim=-1)
    coln = torch.linalg.norm(recon, dim=1, keepdim=True)
    valid = (coln[:, 0] > 1e-8).to(recon.dtype)
    cfft = torch.fft.rfft(recon / coln.clamp_min(1e-8), dim=-1)
    mfft = torch.fft.rfft(valid, dim=-1)
    return hfft, hnorm, cfft, mfft


def masked_minmax(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lo = x.masked_fill(~mask, float("inf")).amin(-1, keepdim=True)
    hi = x.masked_fill(~mask, float("-inf")).amax(-1, keepdim=True)
    return (x - lo) / (hi - lo).clamp_min(1e-8)


def align_terms(tab, q: torch.Tensor, ids: torch.Tensor):
    """두 정렬 점수(정규화 전). q:(B,), ids:(B,K)."""
    hfft, hnorm, cfft, mfft = tab
    a = torch.fft.irfft(hfft[ids] * hfft[q].conj()[:, None], n=60, dim=-1).sum(-2).amax(-1)
    a = a / (hnorm[ids] * hnorm[q][:, None])
    c = torch.fft.irfft(cfft[ids] * cfft[q].conj()[:, None], n=60, dim=-1).sum(-2)
    cnt = torch.fft.irfft(mfft[ids] * mfft[q].conj()[:, None], n=60, dim=-1)
    c = torch.where(cnt > 0.5, c / cnt.clamp_min(1.0), torch.full_like(c, -1.0)).amax(-1)
    return a, c


def selection_score(tab, batch: dict, w: torch.Tensor) -> torch.Tensor:
    a, c = align_terms(tab, batch["q"], batch["ids"])
    m = batch["mask"]
    return w[0] * batch["sn"] + w[1] * batch["sh"] + w[2] * masked_minmax(a, m) + w[3] * masked_minmax(c, m)


def ranking_loss(score: torch.Tensor, batch: dict) -> tuple:
    logit = (score / TAU).masked_fill(~batch["mask"], float("-inf"))
    lse_all = torch.logsumexp(logit, -1)
    lse_pos = torch.logsumexp(logit.masked_fill(~batch["pos"], float("-inf")), -1)
    win = logit.argmax(-1)
    top1 = batch["pos"].gather(1, win[:, None])[:, 0]
    return (lse_all - lse_pos).mean(), top1


# ----------------------------------------------------------------- 후보 집합
def build_cands(role: str, s: str) -> dict:
    """pipeline.candidate_pool과 같은 합집합 후보, 고정 두 코사인의 minmax 값, 기준 자세 5 m 양성."""
    target = OUT / "cands" / role / (name_of(s) + ".npz")
    if target.exists():
        return run.ARR(target)
    tick = time.time()
    cache, pos, minimum, travel, bank, _, _, _ = P.inputs(role, s)
    raw = pipeline.raw_key(cache["descriptors"])
    queries = np.arange(30, len(raw) - 15, dtype=np.int64)
    offsets, ids_all, sn_all, sh_all, pos_all = [0], [], [], [], []
    for q in queries:
        ids = np.unique(np.r_[run.base.prefix_topk(raw, int(q), int(q) - 29, 100),
                              run.base.prefix_topk(bank[0], int(q), int(q) - 29, 100)])
        ids_all.append(ids)
        sn_all.append(run.base.minmax(raw[ids] @ raw[q]))
        sh_all.append(run.base.minmax(bank[0][ids] @ bank[0][q]))
        pos_all.append(np.linalg.norm(pos[ids] - pos[q], axis=1) <= 5)
        offsets.append(offsets[-1] + len(ids))
    rec = dict(query=queries, offset=np.asarray(offsets, np.int64), ids=np.concatenate(ids_all),
               sn=np.concatenate(sn_all), sh=np.concatenate(sh_all), positive=np.concatenate(pos_all),
               min_past_gt_dist=minimum[queries])
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **rec)
    print("CANDS", role, s, len(queries), round(time.time() - tick, 1), flush=True)
    return run.ARR(target)


def padded(d: dict, rows: np.ndarray | None = None) -> dict:
    """질의별 가변 후보를 KMAX로 채운 GPU 텐서. rows=None이면 전체 질의."""
    off = d["offset"]
    n = len(d["query"])
    rows = np.arange(n) if rows is None else rows
    k = KMAX
    ids = np.zeros((len(rows), k), np.int64)
    mask = np.zeros((len(rows), k), bool)
    sn = np.zeros((len(rows), k), np.float64)
    sh = np.zeros((len(rows), k), np.float64)
    pos = np.zeros((len(rows), k), bool)
    for r, i in enumerate(rows):
        lo, hi = off[i], off[i + 1]
        m = hi - lo
        assert m <= k
        ids[r, :m] = d["ids"][lo:hi]
        mask[r, :m] = True
        sn[r, :m] = d["sn"][lo:hi]
        sh[r, :m] = d["sh"][lo:hi]
        pos[r, :m] = d["positive"][lo:hi]
    t = lambda x: torch.from_numpy(x).cuda()  # noqa: E731
    return dict(q=t(d["query"][rows]), ids=t(ids), mask=t(mask), sn=t(sn).float(), sh=t(sh).float(), pos=t(pos))


def eligible(d: dict) -> np.ndarray:
    """양성과 음성 후보가 모두 있는 질의."""
    off = d["offset"]
    cnt = np.add.reduceat(d["positive"].astype(np.int64), off[:-1])
    size = np.diff(off)
    return np.flatnonzero((cnt > 0) & (cnt < size))


def take(b: dict, ix) -> dict:
    return {k: v[ix] for k, v in b.items()}


def weights_t() -> torch.Tensor:
    w, _ = P.selected()
    return torch.tensor(w, dtype=torch.float32, device="cuda")


# ----------------------------------------------------------------- 사전 규칙
def protocol() -> None:
    path = OUT / "protocol.json"
    if path.exists():
        print("protocol exists")
        return
    w, lam = P.selected()
    write(path, dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        written_before_results=True,
        purpose="재설계 B: 같은 diff 구조(151,049)를 무작위 초기화(시드1)+Q/기하 최종층/게이트 최종층 0으로 시작해 1단계 정렬 순위 대조 학습, 2단계 수락 점수 BCE. 1시드.",
        fixed_settings=dict(fusion_weights=w.tolist(), lam=lam, source="results/pdv2_scratch_pilot_20261116/step1/selection.json (재선택 설정)"),
        normalization="BN·dropout은 eval 모드 고정. BN 누적 통계는 1단계 시작 전 초기 가중치로 학습8을 한 번 통과시켜 추정(시퀀스 균등, 시범 주 조건과 같은 P.estimate_bn)하고 두 단계 모두 그대로 고정(2단계에서 다시 추정하지 않음). 이유: 입력 절댓값 ~2.6e4에서 정규화 없는 기본 통계는 첫 투영 출력이 매우 커서 tanh/sigmoid 포화가 쉽고, 사전학습 모델도 데이터 척도로 정규화된 BN 통계를 가진다.",
        stage1=dict(
            samples="학습8 각 시퀀스의 질의 q=30..N-16. 후보 C_q = 원키288·원보조128 두 색인 과거(j<=q-30) 상위 100+100 합집합 전부(최대 200, 중복 제거·보충 없음, pipeline.candidate_pool과 같은 규칙). 양성 = 기준 자세 거리 <=5 m 후보, 나머지 전부 음성. 양성과 음성이 모두 있는 질의만 사용.",
            score="s(q,j) = 2*mm(원키 코사인) + 1*mm(보조 코사인) + 2*mm(스펙트럼 정렬) + 2*mm(열 정렬). mm은 질의별 후보 내 최소·최대 정규화. 앞 두 항은 고정, 뒤 두 항은 모델 출력 z'에서 _xs_eval_lib.align_scores/column_align_scores와 같은 계산을 torch로 미분 가능하게(최대값은 amax, 열 유효 마스크는 미분 없음).",
            loss="L = logsumexp_{j in C_q}(s/TAU) - logsumexp_{j in P_q}(s/TAU), 질의 평균",
            temperature=TAU,
            optimizer="AdamW, lr 3e-4 상수, weight decay 1e-5, 기울기 절단 1.0",
            batch=f"갱신당 한 시퀀스의 질의 {S1_QUERIES}개(비복원), 각 질의의 후보 전부. 시퀀스 그래프 전체를 매 갱신 순전파.",
            schedule="에포크마다 학습8 순서대로 시퀀스당 8갱신, 32에포크 = 2048갱신. 512·1024·2048 저장. 2단계 시작점은 2048 체크포인트(선택 없음). 512·1024는 진단만.",
            rng="표본 순서 np.random.default_rng((1, 101, epoch, ordinal)); 초기화 torch.manual_seed(1).",
            diagnostics="저장 시점마다 학습8·검증9 적격 질의 전체의 순위 손실과 top-1 양성 비율을 기록(선택에 쓰지 않음).",
        ),
        stage2=dict(
            start="1단계 2048 체크포인트 가중치·BN 버퍼 그대로 이어서(재초기화 없음; 0 초기화는 1단계 시작에만 적용).",
            calibration="1단계 모델의 학습8 고정 승자 표본(재선택 융합, 시범 step2/pools) 수락 점수(lam 0.25)에 양의 로지스틱 척도를 적합해 고정(시범 calibrate와 같은 함수·가중).",
            loss="시범과 같은 BCE, 64쌍(정답·오답 균등), 시퀀스당 8갱신, AdamW wd 1e-5, 절단 1, rng (1, epoch, ordinal).",
            lrs=list(S2_LRS.values()),
            updates="4096, 512·1536·4096 저장",
            selection="검증9 고정 승자(시범 step2/pools val) 3계열 AP 최대인 2단계 체크포인트 6개 중 하나를 주 모델로. 동률은 적은 갱신, 그다음 lr3e-4 우선. 시험은 선택에 쓰지 않음.",
        ),
        evaluation=dict(
            models="1단계 단독(2048), 2단계 선택 모델, 그리고 고정 참고점 lr3e-4 1536·4096(중복이면 생략)을 검증9·시험46 실제 검색(matching_only)에 적용.",
            reproduction="같은 실행에서 균등 감산(재선택 설정)을 다시 계산해 시범 원장 uniform_reselected와 승자·점수·지표 비트 일치 확인.",
            intervals="도시 안 시퀀스 재표집 10,000회(rng 20261104), 균등 감산 대비·BEVPlace++(3시드 시퀀스 평균) 대비, 전체·순방향만·역방향만.",
            test_use="시험은 보고만 하고 어떤 선택에도 쓰지 않는다.",
        ),
    ))
    print("protocol written")


def amend() -> None:
    path = OUT / "protocol.json"
    spec = run.READ(path)
    if "amendment_1" in spec:
        return
    spec["amendment_1"] = dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        before_training_results=True,
        seen_before=" 1단계 시작 시 초기 상태(균등 감산) 순위 진단만(학습8 손실 1.297·top1 0.909, 검증9 0.205·0.965). 학습 결과는 없음.",
        change="kitti/01은 양성·음성이 모두 있는 질의가 0개라 1단계에서 건너뛴다(시퀀스 순번 ordinal은 TRAIN 순서 그대로). 32에포크 일정은 유지하므로 1단계 갱신은 7x8x32=1792, 저장 448·896·1792, 2단계 시작점은 1792. 2단계는 그대로(kitti/01 포함).")
    write(path, spec)


def amend2() -> None:
    path = OUT / "protocol.json"
    spec = run.READ(path)
    if "amendment_2" in spec:
        return
    spec["amendment_2"] = dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source="조정자(사용자 결정) 요청",
        seen_before="시드1: 1단계 순위 진단, 2단계 학습 손실, 검증9 고정 승자 AP 전 조건(선택 s2_lr3e-4_u4096), 검증9 실제 검색 실행 완료(지표는 아직 읽지 않음). 시드1 시험46 결과는 보지 않음(실행 중).",
        change="시드 2·3을 같은 규칙으로 추가(amendment_1의 kitti/01 제외, 1단계 1792갱신, 2단계는 lr3e-4만 1단계에서 이어 4096갱신, 1536·4096 저장). 1단계 단독(1792, 2단계 없음)을 2단계 모델과 같은 지위의 후보로 취급하고, 세 시드 모두 검증9(고정 승자·실제 검색)와 시험46에서 1단계 단독과 2단계(1536·4096) 모두 평가. 엄격한 단일 선택 규칙은 요구하지 않음. 시드1의 lr1e-4와 검증 선택 기록은 그대로 둔다. 3시드 평균±SD와 짝지은 구간(균등 감산·BEVPlace++ 대비, 순방향만·역방향만)을 1단계 단독과 2단계 4096(시드1 선택과 같은 조건)·1536에 대해 보고.",
        outputs="시드2·3: results/pdv2_pretrain_redesign_20261116/B/s{2,3}/")
    write(path, spec)


# ----------------------------------------------------------------- 검사
def verify() -> None:
    """0 초기화 항등(기본 BN·추정 BN), torch 선택 점수와 numpy 파이프라인 승자 일치, 후보 집합 재현."""
    tick = time.time()
    rec = {}
    graph = train_graph(TRAIN[0])
    fixed = prior.filtered(graph, -0.5)
    outs = {}
    with torch.no_grad():
        for seed in (1, 2, 3, 77):
            net = P.scratch_model(seed)
            P.estimate_bn(net)
            outs[seed] = net(graph)
            assert torch.all(net.last_strengths == -0.5)
    rec["init_identity_estimated_bn"] = dict(
        sequence=TRAIN[0], bitwise_identical_across_seeds=all(torch.equal(outs[1], v) for v in outs.values()),
        bitwise_equal_uniform_subtraction=all(torch.equal(v, fixed) for v in outs.values()),
        max_abs_vs_uniform=max(float((v - fixed).abs().max()) for v in outs.values()),
        dtype=str(fixed.dtype), shape=list(fixed.shape))
    net = P.scratch_model(1)
    rec["trainable_parameters"] = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # 후보 집합 재현: 검증 시퀀스 하나에서 PIPE 후보 풀과 비교
    s = "kitti360/0004"
    d = build_cands("val", s)
    ref = run.ARR(P.PIPE / "val" / name_of(s) / "candidate_pool.npz")
    cw = P.CURRENT_W
    static = (cw[0] * d["sn"] + cw[1] * d["sh"]).astype(np.float32)
    rec["candidate_pool_vs_pipeline"] = dict(
        query_equal=bool(np.array_equal(d["query"], ref["query"])), offset_equal=bool(np.array_equal(d["offset"], ref["offset"])),
        ids_equal=bool(np.array_equal(d["ids"], ref["ids"])), static_max_error=float(np.abs(static - ref["static"]).max()))
    # torch 선택 점수 vs numpy 파이프라인 승자(균등 감산, 재선택 가중치)
    w, _ = P.selected()
    vg, _, _ = prior.graph_inputs("val", s)
    with torch.no_grad():
        z = prior.filtered(vg, -0.5)
        change = run.base.variant_inputs(z.cpu().numpy(), 8)
        pool = dict(query=d["query"], offset=d["offset"], ids=d["ids"],
                    static=(w[0] * d["sn"] + w[1] * d["sh"]).astype(np.float32))
        j_np = pipeline.winners(pool, change, w)
        tab = tables(z)
        wins, aerr, cerr = [], 0.0, 0.0
        for start in range(0, len(d["query"]), 512):
            rows = np.arange(start, min(start + 512, len(d["query"])))
            b = padded(d, rows)
            sc = selection_score(tab, b, weights_t()).masked_fill(~b["mask"], float("-inf"))
            wins.append(b["ids"].gather(1, sc.argmax(-1)[:, None])[:, 0].cpu().numpy())
            a, c = align_terms(tab, b["q"], b["ids"])
            for r, i in enumerate(rows):
                lo, hi = d["offset"][i], d["offset"][i + 1]
                ids = d["ids"][lo:hi]
                a_np = run.base.align_scores(change[1], change[2], int(d["query"][i]), ids)
                c_np = run.base.column_align_scores(change[3], int(d["query"][i]), ids)
                aerr = max(aerr, float(np.abs(a[r, :hi - lo].cpu().numpy() - a_np).max()))
                cerr = max(cerr, float(np.abs(c[r, :hi - lo].cpu().numpy() - c_np).max()))
        j_t = np.concatenate(wins)
    rec["torch_selection_vs_numpy"] = dict(
        sequence=s, queries=len(j_np), winners_equal=int((j_t == j_np).sum()),
        spectral_align_max_error=aerr, column_align_max_error=cerr)
    # 기울기가 흐르는지
    net, _ = new_model()
    gb = padded(d, eligible(d)[:16])
    zz = net(vg)
    loss, _ = ranking_loss(selection_score(tables(zz), gb, weights_t()), gb)
    loss.backward()
    gn = {n: float(p.grad.norm()) for n, p in net.named_parameters() if p.grad is not None and float(p.grad.norm()) > 0}
    rec["gradient_check"] = dict(loss=float(loss), nonzero_grad_tensors=len(gn),
                                 zero_init_grads={k: v for k, v in gn.items() if "W_q" in k or "gate" in k or "bias_mlp" in k})
    rec["seconds"] = time.time() - tick
    write(OUT / "verify.json", rec)
    print(json.dumps(rec, indent=1))


# ----------------------------------------------------------------- 1단계
def cands_all() -> None:
    for role, seqs in (("train", TRAIN), ("val", VAL)):
        for s in seqs:
            build_cands(role, s)
    rec = {}
    for role, seqs in (("train", TRAIN), ("val", VAL)):
        for s in seqs:
            d = build_cands(role, s)
            el = eligible(d)
            size = np.diff(d["offset"])
            rec[s] = dict(role=role, queries=len(d["query"]), eligible=len(el), mean_candidates=float(size.mean()),
                          max_candidates=int(size.max()), mean_positives_eligible=float(np.add.reduceat(d["positive"].astype(np.int64), d["offset"][:-1])[el].mean()) if len(el) else 0.0)
    write(OUT / "cands_summary.json", rec)
    print(json.dumps(rec, indent=1))


def rank_diag(net, role_seqs) -> dict:
    """적격 질의 전체의 순위 손실·top-1 양성 비율(진단)."""
    w = weights_t()
    out = {}
    with torch.no_grad():
        for role, s in role_seqs:
            d = build_cands(role, s)
            el = eligible(d)
            if not len(el):
                continue
            graph = train_graph(s) if role == "train" else prior.graph_inputs(role, s)[0]
            z = prior.filtered(graph, -0.5) if net is None else net(graph)
            tab = tables(z)
            ls, tp = [], []
            for start in range(0, len(el), 512):
                b = padded(d, el[start:start + 512])
                loss, top1 = ranking_loss(selection_score(tab, b, w), b)
                ls.append(float(loss) * len(top1))
                tp.append(int(top1.sum()))
            out[s] = dict(n=len(el), loss=sum(ls) / len(el), top1=sum(tp) / len(el))
            del graph, z, tab
    for role in ("train", "val"):
        v = [out[s] for r, s in role_seqs if r == role and s in out]
        if v:
            out["_" + role] = dict(loss=float(np.mean([x["loss"] for x in v])), top1=float(np.mean([x["top1"] for x in v])))
    return out


def stage1() -> None:
    folder = RUN / "stage1"
    if (folder / "training_complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    net, bn = new_model()
    g0 = train_graph(TRAIN[0])
    with torch.no_grad():
        assert torch.equal(net(g0), prior.filtered(g0, -0.5))
    del g0
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=S1_LR, weight_decay=1e-5)
    w = weights_t()
    graphs = {s: train_graph(s) for s in TRAIN}
    data = {s: build_cands("train", s) for s in TRAIN}
    elig = {s: eligible(d) for s, d in data.items()}
    full = {s: padded(data[s], elig[s]) for s in TRAIN}
    role_seqs = [("train", s) for s in TRAIN] + [("val", s) for s in VAL]
    diag = {"u0": rank_diag(None, role_seqs)}
    print("DIAG u0", diag["u0"]["_train"], diag["u0"]["_val"], flush=True)
    records, hashes, updates, tick = [], {}, 0, time.time()
    train_seconds = 0.0
    for epoch in range(1, max(S1_SAVE) + 1):
        t_epoch = time.time()
        for ordinal, s in enumerate(TRAIN):
            if not len(elig[s]):
                continue  # amendment_1
            rng = np.random.default_rng((SEED, 101, epoch, ordinal))
            losses, grads, accs = [], [], []
            for _ in range(8):
                chosen = rng.choice(len(elig[s]), S1_QUERIES, replace=len(elig[s]) < S1_QUERIES)
                ix = torch.from_numpy(chosen).cuda()
                b = take(full[s], ix)
                opt.zero_grad(set_to_none=True)
                z = net(graphs[s])
                loss, top1 = ranking_loss(selection_score(tables(z), b, w), b)
                assert torch.isfinite(loss)
                loss.backward()
                g = torch.nn.utils.clip_grad_norm_(params, 1.0)
                assert torch.isfinite(g)
                opt.step()
                updates += 1
                losses.append(float(loss.detach()))
                grads.append(float(g))
                accs.append(float(top1.float().mean()))
            records.append(dict(epoch=epoch, sequence=s, loss=float(np.mean(losses)), grad=float(np.mean(grads)),
                                top1=float(np.mean(accs)), strength_mean=float(net.last_strengths.detach().mean())))
        train_seconds += time.time() - t_epoch
        if epoch in S1_SAVE:
            assert updates == S1_SAVE[epoch]
            path = folder / f"u{updates}.pth"
            torch.save(dict(model_state_dict=net.state_dict(), epoch=epoch, updates=updates, lr=S1_LR, seed=SEED, tau=TAU), path)
            hashes[path.name] = run.HASH(path)
            diag[f"u{updates}"] = rank_diag(net, role_seqs)
            print("S1", epoch, updates, round(time.time() - tick, 1), np.mean([r["loss"] for r in records[-8:]]),
                  diag[f"u{updates}"]["_train"], diag[f"u{updates}"]["_val"], flush=True)
    changed = [k for k, v in net.state_dict().items() if not torch.equal(v, initial[k])]
    peak = torch.cuda.max_memory_allocated() / 2**30
    write(folder / "training_complete.json", dict(
        status="passed", lr=S1_LR, tau=TAU, seed=SEED, queries_per_update=S1_QUERIES, updates=updates,
        bn_estimated_batches=bn, eligible_queries={s: int(len(v)) for s, v in elig.items()},
        checkpoints=hashes, records=records, diagnostics=diag, changed=changed,
        trainable_parameters=sum(p.numel() for p in params), seconds_total=time.time() - tick,
        seconds_training_only=train_seconds, peak_cuda_memory_gib=peak, gpu=torch.cuda.get_device_name()))
    timing("stage1", time.time() - tick, training_only=round(train_seconds, 1), peak_gib=round(peak, 2))


# ----------------------------------------------------------------- 2단계
def calibrate() -> None:
    path = RUN / "stage2/loss_fit.json"
    if path.exists():
        return
    tick = time.time()
    _, lam = P.selected()
    net = load(RUN / "stage1" / f"u{S1_FINAL}.pth")
    xs, ys, ws = [], [], []
    for s in TRAIN:
        data = run.ARR(P.NEWPOOLS / "train" / name_of(s) / "pool.npz")
        graph = train_graph(s)
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
    fit = fit_score(x, y, w / w.sum())
    write(path, dict(fit=fit, lam=lam, model=f"stage1/u{S1_FINAL}.pth",
                     pilot_initial_fit=run.READ(PILOT / "step2/loss_fit.json")["fit"], validation_used=False))
    timing("stage2_calibrate", time.time() - tick)
    print("FIT", fit)


def stage2(tag: str) -> None:
    lr = S2_LRS[tag]
    folder = RUN / "stage2" / tag
    if (folder / "training_complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    _, lam = P.selected()
    fit = run.READ(RUN / "stage2/loss_fit.json")["fit"]
    net = load(RUN / "stage1" / f"u{S1_FINAL}.pth")
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    graphs = {s: train_graph(s) for s in TRAIN}
    pools = {s: run.ARR(P.NEWPOOLS / "train" / name_of(s) / "pool.npz") for s in TRAIN}
    plans = {s: {k: torch.from_numpy(d["plan_" + k]).cuda() for k in ("query", "candidate", "weight")} for s, d in pools.items()}
    records, hashes, updates, tick = [], {}, 0, time.time()
    for epoch in range(1, max(S2_SAVE) + 1):
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
                losses.append(float(loss.detach()))
                grads.append(float(g))
            records.append(dict(epoch=epoch, sequence=s, loss=float(np.mean(losses)), grad=float(np.mean(grads)),
                                strength_mean=float(net.last_strengths.detach().mean())))
        if epoch in S2_SAVE:
            assert updates == S2_SAVE[epoch]
            path = folder / f"u{updates}.pth"
            torch.save(dict(model_state_dict=net.state_dict(), epoch=epoch, updates=updates, lr=lr, seed=SEED), path)
            hashes[path.name] = run.HASH(path)
            print("S2", tag, epoch, updates, round(time.time() - tick, 1), np.mean([r["loss"] for r in records[-8:]]), flush=True)
    changed = [k for k, v in net.state_dict().items() if not torch.equal(v, initial[k])]
    write(folder / "training_complete.json", dict(
        status="passed", lr=lr, seed=SEED, start=f"stage1/u{S1_FINAL}.pth", updates=updates, checkpoints=hashes,
        records=records, changed=changed, seconds=time.time() - tick,
        peak_cuda_memory_gib=torch.cuda.max_memory_allocated() / 2**30))
    timing("stage2_" + tag, time.time() - tick)


def ckpt_cases() -> dict:
    cases = {f"s1_u{u}": RUN / "stage1" / f"u{u}.pth" for u in S1_SAVE.values()}
    for tag in S2_LRS:
        for u in S2_SAVE.values():
            cases[f"s2_{tag}_u{u}"] = RUN / "stage2" / tag / f"u{u}.pth"
    return cases


def val_fixed() -> None:
    tick = time.time()
    _, lam = P.selected()
    cases = {"uniform": None, **ckpt_cases()}
    per = {k: {} for k in cases}
    for s in VAL:
        graph, _, _ = prior.graph_inputs("val", s)
        data = run.ARR(P.NEWPOOLS / "val" / name_of(s) / "pool.npz")
        row = {k: data[k] for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist")}
        plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
        with torch.no_grad():
            for tag, ck in cases.items():
                z = prior.filtered(graph, -0.5) if ck is None else load(ck)(graph)
                v = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(row["query_idx"])).astype(np.float64)
                per[tag][s] = run.base.independent_metrics(row, v)
    res = {k: run.base.macro(v) for k, v in per.items()}
    order = [k for k in cases if k.startswith("s2_")]
    rank = sorted(order, key=lambda k: (-res[k]["macro"]["AP"], int(k.split("_u")[1]), list(S2_LRS).index(k.split("_")[1])))
    pilot_uniform = run.READ(PILOT / "step2/val_fixed_winner.json")["results"]["uniform"]["macro"]["AP"]
    write(RUN / "val_fixed_winner.json", dict(results=res, per_sequence=per, selected=rank[0], lam=lam, test_used=False,
                                              uniform_equals_pilot=bool(res["uniform"]["macro"]["AP"] == pilot_uniform)))
    timing("val_fixed", time.time() - tick)
    for k in cases:
        print(k, {m: round(100 * res[k]["macro"][m], 3) for m in METRICS})
    print("SELECTED", rank[0])


# ----------------------------------------------------------------- 실제 검색 평가
def eval_cases() -> dict:
    if SEED != 1:  # amendment_2: 선택 없이 1단계 단독과 2단계 1536·4096을 같은 지위로
        return {"B_stage1": RUN / "stage1" / f"u{S1_FINAL}.pth",
                **{f"B_s2_lr3e-4_u{u}": ckpt_cases()[f"s2_lr3e-4_u{u}"] for u in (1536, 4096)}}
    sel = run.READ(RUN / "val_fixed_winner.json")["selected"]
    out = {"B_stage1": RUN / "stage1" / f"u{S1_FINAL}.pth", "B_selected": ckpt_cases()[sel]}
    for u in (1536, 4096):
        k = f"s2_lr3e-4_u{u}"
        if k != sel:
            out["B_" + k] = ckpt_cases()[k]
    return out


def pipe(role: str, sequence: str) -> None:
    name = name_of(sequence)
    folder = RUN / "eval" / role / name
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
    cases = {"repro_uniform": None, **eval_cases()}
    pts = [len(q) // 3, 2 * len(q) // 3]
    smalls = {}
    metrics, checks = {}, {}
    for tag, ck in cases.items():
        with torch.no_grad():
            net = None if ck is None else load(ck)
            z = prior.filtered(graph, -0.5) if net is None else net(graph)
            change = run.base.variant_inputs(z.cpu().numpy(), 8)
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
        if net is None:
            ref = run.ARR(PILOT / "step3" / role / name / "uniform_reselected.npz")
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
    write(folder / "summary.json", dict(metrics=metrics, checks=checks, seconds=time.time() - tick))
    print("PIPE", role, sequence, round(time.time() - tick, 1), flush=True)


def campaign(role: str, workers: int) -> None:
    import concurrent.futures
    import subprocess
    tick = time.time()
    seqs = VAL if role == "val" else TESTSEQ
    (RUN / "logs").mkdir(parents=True, exist_ok=True)

    def call(s):
        with (RUN / "logs" / f"{role}_{name_of(s)}.log").open("a") as f:
            subprocess.run([sys.executable, __file__, "pipe", "--role", role, "--sequence", s], stdout=f, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(concurrent.futures.as_completed([ex.submit(call, s) for s in seqs]), 1):
            fut.result()
            print("DONE", role, i, len(seqs), flush=True)
    timing("eval_" + role, time.time() - tick, workers=workers)


def finish() -> None:
    tags = ["repro_uniform", *eval_cases()]
    out = {}
    for role, seqs in (("val", VAL), ("test", TESTSEQ)):
        per = {t: {} for t in tags}
        chk = {}
        for s in seqs:
            rec = run.READ(RUN / "eval" / role / name_of(s) / "summary.json")
            for t in tags:
                per[t][s] = rec["metrics"][t]
            chk[s] = rec["checks"]
        if role == "val":
            res = {t: dict(**run.base.macro(per[t]), sequences=per[t]) for t in tags}
        else:
            res = {t: dict(**run.current.adoption.previous.macro(per[t], run.current.CITY), sequences=per[t]) for t in tags}
            n = sum(len(run.ARR(RUN / "eval/test" / name_of(s) / "repro_uniform.npz")["query_idx"]) for s in seqs)
            assert n == 195726, n
        repro = all(c["repro_uniform"]["winners_equal"] and c["repro_uniform"]["score_bitwise_equal"] for c in chk.values())
        prefix = {t: [x for c in chk.values() for x in c[t]["prefix"]] for t in tags if t != "repro_uniform"}
        out[role] = dict(results=res, reproduction_uniform_exact=repro,
                         prefix={t: dict(n=len(v), all_winner_equal=all(x["winner_equal"] for x in v),
                                         max_score_error=max(x["score_error"] for x in v)) for t, v in prefix.items()})
    pilot = run.READ(PILOT / "summary.json")
    out["pilot_uniform_test_macro"] = pilot["test"]["results"]["uniform_reselected"]["macro"]
    out["test_used_for_selection"] = False
    write(RUN / "summary.json", out)
    for role in ("val", "test"):
        print(role, out[role]["reproduction_uniform_exact"], out[role]["prefix"])
        for t in tags:
            print("  ", t, {m: round(100 * out[role]["results"][t]["macro"][m], 3) for m in METRICS})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action")
    p.add_argument("--sequence", default="kitti360/0004")
    p.add_argument("--role", default="val")
    p.add_argument("--tag", default="lr3e-4")
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    e.configure()
    RUN.mkdir(parents=True, exist_ok=True)
    {"protocol": protocol, "amend": amend, "amend2": amend2, "verify": verify, "cands": cands_all, "stage1": stage1, "calibrate": calibrate,
     "val_fixed": val_fixed, "finish": finish}.get(a.action, lambda: None)()
    if a.action == "stage2":
        stage2(a.tag)
    elif a.action == "pipe":
        pipe(a.role, a.sequence)
    elif a.action == "campaign":
        campaign(a.role, a.workers)
