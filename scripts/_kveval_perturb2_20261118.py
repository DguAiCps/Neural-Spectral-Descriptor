#!/usr/bin/env python3
"""평가 하네스(_kveval_perturb_20261118, 이하 H)에 그래프 문맥·노드 입력 교란 조건(B3, B4, B5)을 더한다(학습 없음).

H 파일은 고치지 않는다(돌고 있는 체인이 다시 import한다). 이 파일이 H를 import해 조건을 등록하고 같은 명령
(eval, campaign, summary, list)을 그대로 제공한다. 원장 규약도 H와 같다:
  results/kv_eval_conditions_20261118/perturb/<조건>/ledgers/test/<seq>/<model>.{npz,json}
집계는 perturb/summary2.json(기본 조건은 perturb/base의 로컬 재실행 원장).

-------------------------------------------------------------------- 그래프·전달 구조(코드에서 확인)
graph = prior.graph_inputs(...) = run.make_graph(x, poses) + attach_transport(spectrum, yaw):
  x        : 캐시 디스크립터(N, 288, 양수, 1e-4 ~ 1e4대). 모델 첫 연산이 XNorm("log") = log1p(x).
  edge_index: 같은 시퀀스 ±5 대칭 시간 간선(temporal_edges, (src -> tgt)), edge_attr = pose_attributes(poses), edge_type 0.
  transport_indices/mask (N, 11): 칸 = src - tgt + 5(5는 자기 자신, 항상 0). 빈 칸의 색인은 자기 자신.
  transport_phase: lidar_yaw(poses)의 위상. spectrum: testspec의 sk 복소 계수 앞 8주파수(N, 16, 8), 디스크에서 읽는다.
모델(ContrastSpectralDifference, 두 팔 공통): h = LN(proj(log1p x)) -> 두 DiffAttnConv 층(K/V 입력만 diff/abs_kv로 다름)
  -> 층별 attention을 dense[tgt, src-tgt+5]에 펼침 -> transport_difference(spectrum, indices, mask, phase, 게이트 세기,
  attention): 각 노드 스펙트럼에서 이웃(마스크 칸) 차이를 attention 가중으로 두 번 더한다. 출력 z(N,16,8)가
  변화 키(승자 선택)와 시간 증거 점수(plan의 질의·후보 ±이웃 대응)에 쓰인다. 후보 풀·정적 키(raw, bank)는 모델과 무관.

-------------------------------------------------------------------- 조건(수준은 결과 보기 전에 고정)
B3 b3_p{p}  이웃 프레임 누락, p in {0.1, 0.3, 0.5}. u ~ U(0,1)을 시퀀스마다 한 번 뽑아(seed_of(seq, 3)) 누락 = u < p
            (단계 간 포개진 공통 난수). 누락 프레임이 메시지 원천(src)인 시간 간선을 지우고, 간선 속성·종류와
            전달 칸(transport_indices/mask)을 남은 간선에서 attach_transport와 같은 규칙으로 다시 만든다.
            누락 프레임 자신은 여전히 질의·후보·tgt로 평가된다(이웃 문맥만 듬성). plan(시간 증거)은 그대로.
    b3p_p{p} 같은 누락 실현 + plan에서도 누락 프레임을 무효로: neighbor_plans의 arc_unique 유효 마스크에
            ~누락[질의 이웃] & ~누락[후보 이웃]을 곱한다(중심 질의·후보는 그대로; 무효가 된 자리를 다른 프레임으로
            다시 대응하지 않는다). 가중은 원래 plan 식(repeated_candidate_weights, 중심 4/15)이 남은 칸으로 다시 정규화.
B4 b4_k{k}  평가 때 시간 창 축소 |src - tgt| <= k, k in {1, 2, 3}(학습은 ±5). 간선·속성·전달 칸을 같은 방식으로 지운다. plan 그대로.
B5 b5_s{s}  노드 입력 잡음, s in {0.10, 0.25, 0.50, 1.00} x (시퀀스 안 특징별 표준편차). 표준 정규 난수는 시퀀스마다 한 번
            (seed_of(seq, 5)) 뽑아 단계 공통.
            - 디스크립터 x: 모델이 실제 받는 log1p(x) 단계에 더한다(XNorm 출력에 forward hook). x 원 단위는 꼬리가 길어
              (최대 ~1e4) 원 단위 가우시안이 음수를 만들고 log1p가 깨지므로 원 단위에는 넣지 않았다. 표준편차 = log1p(x)의 특징별.
            - 스펙트럼: graph.spectrum(복소 16x8)의 실수부·허수부에 각 원소별 표준편차로(그래프 훅; 디스크 읽기 뒤).
            x(디스크립터 캐시)와 스펙트럼(testspec)은 서로 다른 전처리 파일이라 한 원천에서 일관되게 만들 수 없다.
              두 잡음은 독립이다.
            한계: 후보 풀, 정적 키(raw, bank, 승자 선택의 고정 융합 항)는 깨끗한 입력을 쓴다. 모델 출력 z(잡음 영향)는
              승자 선택의 변화 키와 시간 증거 점수에 들어간다.
교란 0 검산용(campaign에는 쓰지 않음): b3_p0.00, b3p_p0.00, b4_k5, b5_s0.00 -> kitti/07에서 base 원장과 비트 일치.
정답(pos, min_past_gt_dist, top1 거리)은 H.evaluate가 그대로 참값을 쓴다.

실행(컨테이너 안):
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_kveval_perturb2_20261118.py \
      campaign --cond b4_k1 --models main --workers 2
  python3 scripts/_kveval_perturb2_20261118.py summary      (호스트 numpy)
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
B3_P = (0.10, 0.30, 0.50)
B4_K = (1, 2, 3)
B5_S = (0.10, 0.25, 0.50, 1.00)
TRAIN_RADIUS = 5
_NOISE: dict = {}  # B5: 이번 eval 프로세스(시퀀스 하나)의 log1p(x) 잡음 텐서


def b3_names():
    return [f"b3_p{p:.2f}" for p in B3_P] + [f"b3p_p{p:.2f}" for p in B3_P]


NEW = b3_names() + [f"b4_k{k}" for k in B4_K] + [f"b5_s{s:.2f}" for s in B5_S]
ZERO = ["b3_p0.00", "b3p_p0.00", f"b4_k{TRAIN_RADIUS}", "b5_s0.00"]


# ----------------------------------------------------------------- 간선 제한(간선·속성·종류·전달 칸 일관)
def restrict(graph, keep: np.ndarray):
    """keep(간선 bool)만 남기고 전달 칸을 attach_transport 규칙으로 다시 만든다. 위상·스펙트럼·x는 그대로."""
    torch = H.ctx()["torch"]
    n = graph.num_nodes
    src, tgt = graph.edge_index.cpu().numpy()
    old_index, old_mask = graph.transport_indices.cpu().numpy(), graph.transport_mask.cpu().numpy()
    slot = src - tgt + 5
    assert (old_mask[tgt, slot] == 1).all() and (old_index[tgt, slot] == src).all() and int(old_mask.sum()) == len(src)
    s, t, k = src[keep], tgt[keep], slot[keep]
    index = np.broadcast_to(np.arange(n)[:, None], (n, 11)).copy()
    mask = np.zeros((n, 11), np.float32)
    index[t, k], mask[t, k] = s, 1.0
    assert int(mask.sum()) == len(s) and not mask[:, 5].any()
    dev = graph.x.device
    kt = torch.as_tensor(keep, device=dev)
    g = graph.clone()
    g.edge_index = graph.edge_index[:, kt]
    g.edge_attr = graph.edge_attr[kt]
    g.edge_type = graph.edge_type[kt]
    g.transport_indices = torch.as_tensor(index, dtype=graph.transport_indices.dtype, device=dev)
    g.transport_mask = torch.as_tensor(mask, dtype=graph.transport_mask.dtype, device=dev)
    if keep.all():  # 교란 0: 그래프가 원본과 같아야 한다
        for a in ("edge_index", "edge_attr", "edge_type", "transport_indices", "transport_mask"):
            assert torch.equal(getattr(g, a), getattr(graph, a)), a
    incoming = mask.sum(1)
    diag = dict(n=n, edges=int(len(src)), edges_kept=int(keep.sum()), edges_kept_frac=float(keep.mean()),
                isolated_frac=float(np.mean(incoming == 0)), incoming_mean=float(incoming.mean()),
                incoming_orig_mean=float(old_mask.sum(1).mean()))
    return g, diag


def dropped_of(seq: str, n: int, p: float) -> np.ndarray:
    return np.random.default_rng(H.seed_of(seq, 3)).random(n) < p


def drop_graph(p: float):
    def hook(seq, graph, tup):
        drop = dropped_of(seq, graph.num_nodes, p)
        src = graph.edge_index[0].cpu().numpy()
        g, d = restrict(graph, ~drop[src])
        d.update(p=p, dropped_frac=float(drop.mean()))
        return g, d
    return hook


def drop_plan(p: float):
    def fn(seq, travel, q, j):
        c = H.ctx()["C"]
        drop = dropped_of(seq, len(travel), p)
        original = c.ORIGINAL["neighbor_plans"]

        def nb(path, query, candidate):
            r = dict(original(path, query, candidate))
            qn, jn, valid = r["arc_unique"]
            n = len(path)
            ok = ~drop[np.clip(qn, 0, n - 1)] & ~drop[np.clip(jn, 0, n - 1)]
            r["arc_unique"] = (qn, jn, valid & ok)
            return r
        c.SG.neighbor_plans = nb
        try:
            plan = c.ORIGINAL["plan"](travel, q, j)
        finally:
            c.SG.neighbor_plans = original
        return plan
    return fn


def window_graph(k: int):
    def hook(seq, graph, tup):
        src, tgt = graph.edge_index.cpu().numpy()
        g, d = restrict(graph, np.abs(src - tgt) <= k)
        d.update(k=k)
        return g, d
    return hook


def noise_graph(s: float):
    def hook(seq, graph, tup):
        torch = H.ctx()["torch"]
        x = graph.x.cpu().numpy()
        n = len(x)
        rng = np.random.default_rng(H.seed_of(seq, 5))
        ex = rng.standard_normal(x.shape)
        spec = graph.spectrum.cpu().numpy()
        er, ei = rng.standard_normal(spec.shape), rng.standard_normal(spec.shape)
        ly = np.log1p(x.astype(np.float64))
        sx = ly.std(0)
        sr, si = spec.real.astype(np.float64).std(0), spec.imag.astype(np.float64).std(0)
        nx = (s * sx[None] * ex).astype(np.float32)
        nz = (s * sr[None] * er + 1j * s * si[None] * ei).astype(np.complex64)
        _NOISE["x"] = torch.as_tensor(nx, device=graph.x.device)
        g = graph.clone()
        g.spectrum = graph.spectrum + torch.as_tensor(nz, device=graph.x.device)
        mag = np.abs(spec).reshape(n, -1)
        d = dict(sigma_mult=s, n=n, logx_feature_std_median=float(np.median(sx)),
                 logx_noise_rms=float(np.sqrt(np.mean(nx.astype(np.float64) ** 2))),
                 logx_rms=float(np.sqrt(np.mean(ly ** 2))),
                 spec_noise_rms=float(np.sqrt(np.mean(np.abs(nz.astype(np.complex128)) ** 2))),
                 spec_rms=float(np.sqrt(np.mean(mag.astype(np.float64) ** 2))),
                 node_snr_spec_median=float(np.median(np.linalg.norm(mag, axis=1) /
                                                      np.maximum(np.linalg.norm(np.abs(nz).reshape(n, -1), axis=1), 1e-30))))
        return g, d
    return hook


_load = H.load_model


def load_model(m: str):
    """B5일 때만 XNorm(log1p) 출력에 이번 시퀀스의 잡음을 더한다(다른 조건은 H.load_model 그대로)."""
    net = _load(m)
    if "x" in _NOISE:
        c = H.ctx()
        xn = net.input_proj[0]
        assert isinstance(xn, (c["S"].XNorm,)) and xn.mode == "log", type(xn)
        noise = _NOISE["x"]

        def add(module, inp, out):
            assert out.shape == noise.shape
            return out + noise
        xn.register_forward_hook(add)
    return net


H.load_model = load_model  # 이 프로세스 안에서만(파일은 그대로)

for p in (0.0,) + B3_P:
    H.register(H.Cond(f"b3_p{p:.2f}", f"B3 이웃 프레임 누락 p={p:g}: 누락 프레임이 원천인 시간 간선·전달 칸 제거, plan 그대로",
                      dict(family="B3", p=p, plan="unchanged"), graph=drop_graph(p)))
    H.register(H.Cond(f"b3p_p{p:.2f}", f"B3p 이웃 프레임 누락 p={p:g}: 간선·전달 칸 제거 + plan의 이웃 대응에서도 누락 프레임 무효",
                      dict(family="B3p", p=p, plan="dropped_invalid"), graph=drop_graph(p), plan=drop_plan(p)))
for k in B4_K + (TRAIN_RADIUS,):
    H.register(H.Cond(f"b4_k{k}", f"B4 평가 시간 창 축소 |i-j|<={k}(학습 ±{TRAIN_RADIUS}), plan 그대로",
                      dict(family="B4", k=k), graph=window_graph(k)))
for s in (0.0,) + B5_S:
    H.register(H.Cond(f"b5_s{s:.2f}", f"B5 노드 입력 가우시안 잡음 {s:g} x 시퀀스 안 특징별 표준편차(log1p(x)와 스펙트럼 실·허수부)",
                      dict(family="B5", sigma_mult=s), graph=noise_graph(s)))


# ----------------------------------------------------------------- campaign(H.campaign과 같고 하위 프로세스만 이 파일)
def campaign(cond: str, group: str, workers: int) -> None:
    import concurrent.futures
    assert workers <= 2
    c = H.ctx()
    seqs = c["P"].TESTSEQ
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    spec = H.COND[cond]
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
                                                                    models=H.model_names(group)))


# ----------------------------------------------------------------- 집계(H.summary와 같은 규약, 조건만 base + NEW)
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

    conds = [c for c in ["base"] + NEW if (OUT / c / "ledgers").exists()]
    res = dict(conventions=dict(
        aggregation="시퀀스 지표(_pr_eval_lib.pr_sweep, R@1=양성 top1<=5 m) -> 도시 안 평균 -> 6도시 동일 가중 -> 시드 평균 (KITTI 07 양성 없음)",
        interval="NSD-abs: 시퀀스별 (NSD 시드 평균 - abs 시드 평균), 도시 안 시퀀스 복원추출 10,000회, rng=default_rng(20261104) 비교마다 새로",
        D="D = (NSD-abs)_조건 - (NSD-abs)_base(로컬 재실행), 같은 시퀀스 -> boot_joint(도시 안 시퀀스 색인 공유), rng=default_rng(20261117)",
        units="%p", baseline="base = H로 로컬 RTX 5080에서 다시 돌린 교란 없음 조건(perturb/base)",
        source="scripts/_kveval_perturb2_20261118.py (B3, B3p, B4, B5)"), conditions={})
    base_rows = {g: {a: rows("base", t) for a, t in arms[g].items()} for g in arms}
    for cond in conds:
        rec = dict(desc=H.COND[cond].desc, params=H.COND[cond].params)
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
                bper = {a: [K.macro(r, RC.CITY) for r in rs] for a, rs in br.items()}
                out["arm_change_vs_base_pp"] = {a: {k: mean[a][k] - 100 * float(np.mean([p[k] for p in bper[a]])) for k in keys} for a in ga}
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
        folder = OUT / cond / "inputs/test"
        dg = [json.loads(p.read_text())["diag"].get("graph", {}) for p in sorted(folder.glob("*.json"))] if folder.exists() else []
        if dg and dg[0]:
            num = [k for k, v in dg[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            rec["input_diag_median"] = {k: float(np.median([d[k] for d in dg])) for k in num}
            rec["input_diag_median"]["n_sequences"] = len(dg)
        res["conditions"][cond] = rec
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary2.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
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
    ap.add_argument("action", choices=("eval", "campaign", "summary", "list"))
    ap.add_argument("--cond", default="base")
    ap.add_argument("--sequence")
    ap.add_argument("--models", default="main")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--root", default=str(OUT), help="eval 출력 위치(교란 0 검산은 tmp/ 아래로)")
    a = ap.parse_args()
    if a.action == "list":
        for k in NEW + ZERO:
            print(k, "|", H.COND[k].desc)
    elif a.action == "summary":
        summary()
    else:
        assert a.cond in NEW + ZERO or a.action == "eval", "campaign은 이 파일의 새 조건만"
        c = H.ctx()
        c["P"].e.configure()
        c["torch"].set_num_threads(4)
        if a.action == "eval":
            H.evaluate(a.cond, a.sequence, H.model_names(a.models), Path(a.root))
        else:
            campaign(a.cond, a.models, a.workers)
