#!/usr/bin/env python3
"""보고모델 DiffAttnConv 주의 가중치 alpha_ij 분석(2026-11-17).

보고 구성 x=log__n=ln__g=frozen__lr=1e-3__U=1536, u512, 시드 4·5·6을 시험46 입력 그래프(원장과 같은
prior.graph_inputs 경로)에 통과시켜 두 층의 alpha(노드, 오프셋 11칸, 헤드 4)를 모은다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_attention_analysis_20261117.py collect
docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_attention_analysis_20261117.py analyze
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import _pdv2_scratch_search_20261116 as S  # noqa: E402
import _report_model_common_20261116 as C  # noqa: E402
from _pr_eval_lib import classify_queries  # noqa: E402

P, prior = S.P, S.prior
CFG = C.CFG
SEEDS = C.NEW_SEEDS
OUT = C.OUT / "attention"
RAWDIR = OUT / "raw"
SLOTS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]          # 오프셋 -5..-1, +1..+5
OFFS = [s - 5 for s in SLOTS]


def rot_angle(Ra, Rb):
    R = Ra @ np.transpose(Rb, (0, 2, 1))
    tr = np.clip(np.trace(R, axis1=1, axis2=2), -1., 3.)
    return np.arccos(np.clip((tr - 1.) / 2., -1., 1.))


def dense_from_edges(graph, values, n):
    src, tgt = graph.edge_index
    out = torch.zeros((n, 11) + tuple(values.shape[1:]), dtype=values.dtype, device=values.device)
    out[tgt, src - tgt + 5] = values
    return out


def collect() -> None:
    RAWDIR.mkdir(parents=True, exist_ok=True)
    nets = {s: S.load(CFG, s, 512) for s in SEEDS}
    _, lam = P.selected()
    checks = dict(init=[], ledger=[], mask=[])
    init_net = S.build(S.parse(CFG), SEEDS[0])     # 학습 전(0 초기화) 상태
    for k, seq in enumerate(S.TESTSEQ):
        name = seq.replace("/", "_")
        graph, travel, _ = prior.graph_inputs("test", seq)
        n = len(graph.x)
        cache = P.inputs("test", seq)[0]
        poses = np.asarray(cache["poses"])[:n]
        mask = graph.transport_mask.float()
        # 간선 속성(입력과 같은 정의): 거리, 3D 회전각
        attr = graph.edge_attr
        dist = dense_from_edges(graph, torch.expm1(attr[:, 0] * 5.), n).cpu().numpy()
        ang = dense_from_edges(graph, attr[:, 1] * np.pi, n).cpu().numpy()
        att = []
        with torch.no_grad():
            if k < 3:   # 초기화 모델: alpha가 균등 감산 가중치와 같고 출력이 균등 감산과 비트 단위로 같은가
                z0 = init_net(graph)
                a0 = [dense_from_edges(graph, c.attention, n) for c in init_net.convs]
                uni = mask / mask.sum(1, keepdim=True).clamp_min(1)
                interior = mask.sum(1) == 10
                checks["init"].append(dict(
                    sequence=seq,
                    alpha_equals_uniform_weights=all(bool(torch.equal(a, uni[..., None].expand_as(a))) for a in a0),
                    interior_alpha_all_exactly_0p1=all(bool((a[interior][:, SLOTS] == torch.tensor(0.1, dtype=a.dtype)).all()) for a in a0),
                    output_equals_uniform=bool(torch.equal(z0, prior.filtered(graph, -0.5)))))
            for s in SEEDS:
                z = nets[s](graph)
                a = torch.stack([dense_from_edges(graph, c.attention, n) for c in nets[s].convs])  # (2,N,11,4)
                checks["mask"].append(bool(((a.sum(-2) - 1).abs() < 1e-5).all()) and bool((a.permute(0, 1, 3, 2)[..., 5] == 0).all())
                                      and bool((a[:, mask == 0] == 0).all()))
                att.append(a.cpu().numpy())
                if k % 9 == 0:   # 원장 점수 재현(같은 승자 j로 점수 재계산)
                    led = np.load(S.LEDGERS / "test" / name / f"{CFG}@s{s}@u512.npz")
                    q, j = led["query_idx"], led["j_star"]
                    plan = prior.base.prior.plan(travel, q, j)
                    v = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(q)).astype(np.float64)
                    checks["ledger"].append(dict(sequence=seq, seed=s, n=int(len(q)), bitwise=bool(np.array_equal(v, led["score"])),
                                                 max_abs_err=float(np.abs(v - led["score"]).max())))
        # 노드 주변 진행: i-5와 i+5 사이 회전각·이동거리
        turn = np.full(n, np.nan)
        span = np.full(n, np.nan)
        i = np.arange(5, n - 5)
        if len(i):
            turn[i] = rot_angle(poses[i - 5, :3, :3], poses[i + 5, :3, :3])
            span[i] = np.linalg.norm(poses[i + 5, :3, 3] - poses[i - 5, :3, 3], axis=1)
        np.savez_compressed(RAWDIR / f"{name}.npz", att=np.stack(att).astype(np.float32), mask=mask.cpu().numpy(),
                            dist=dist, ang=ang, turn=turn, span=span)
        print("SEQ", k, seq, n, flush=True)
        del graph
    checks["mask_all_ok"] = all(checks["mask"])
    checks.pop("mask")
    C.write(OUT / "checks.json", checks)
    print(json.dumps(checks, indent=1), flush=True)


# ----------------------------------------------------------------- 분석
def qstats(x):
    x = np.asarray(x, float)
    return dict(median=float(np.median(x)), q25=float(np.quantile(x, .25)), q75=float(np.quantile(x, .75)), mean=float(x.mean()))


def load_all():
    rows = []
    for seq in S.TESTSEQ:
        z = np.load(RAWDIR / (seq.replace("/", "_") + ".npz"))
        rows.append({k: z[k] for k in z.files} | dict(seq=seq))
    return rows


def entropy(a):
    """a: (...,10) 합 1. 정규화 엔트로피(균등=1)."""
    p = np.clip(a, 1e-12, 1)
    return -(p * np.log(p)).sum(-1) / np.log(10)


def analyze() -> None:
    rows = load_all()
    # 내부 노드(이웃 10개 모두)만
    A, D, G, T, SP, SEQ, IDX = [], [], [], [], [], [], []
    for r in rows:
        keep = r["mask"][:, SLOTS].sum(1) == 10
        A.append(r["att"][:, :, keep][:, :, :, SLOTS])        # (3,2,n,10,4)
        D.append(r["dist"][keep][:, SLOTS]); G.append(np.degrees(r["ang"][keep][:, SLOTS]))
        T.append(np.degrees(r["turn"][keep])); SP.append(r["span"][keep])
        SEQ += [r["seq"]] * int(keep.sum()); IDX.append(np.flatnonzero(keep))
    A = np.concatenate(A, 2); D = np.concatenate(D); G = np.concatenate(G); T = np.concatenate(T); SP = np.concatenate(SP)
    n = A.shape[2]
    Ah = A.mean(-1)                                           # 헤드 평균 (3,2,n,10)
    res = dict(n_interior_nodes=int(n), n_all_nodes=int(sum(len(r["mask"]) for r in rows)), offsets=OFFS, uniform=0.1)

    # 1. 오프셋별 평균 alpha
    prof = {}
    for L in range(2):
        hm = Ah[:, L].mean(1)                                 # (3,10)
        ph = A[:, L].mean(1)                                  # (3,10,4)
        prof[f"layer{L + 1}"] = dict(
            head_mean=dict(mean=hm.mean(0).tolist(), seed_sd=hm.std(0, ddof=1).tolist()),
            per_head=ph.mean(0).T.tolist(),
            near_pm1=float(hm.mean(0)[[4, 5]].sum()), far_pm5=float(hm.mean(0)[[0, 9]].sum()),
            past_total=float(hm.mean(0)[:5].sum()), future_total=float(hm.mean(0)[5:].sum()),
            per_seed_past_total=hm[:, :5].sum(1).tolist(),
            per_seed_near_pm1=hm[:, [4, 5]].sum(1).tolist(), per_seed_far_pm5=hm[:, [0, 9]].sum(1).tolist())
    res["offset_profile"] = prof

    # 2. 집중도
    conc = {}
    for L in range(2):
        Hh = entropy(Ah[:, L])                                # (3,n) 헤드 평균 alpha의 엔트로피
        Hp = entropy(np.moveaxis(A[:, L], -1, -2))            # (3,n,4) 헤드별
        mx = Ah[:, L].max(-1)
        mxp = A[:, L].max(-2)                                 # (3,n,4)
        conc[f"layer{L + 1}"] = dict(
            head_mean_alpha=dict(H_norm=qstats(Hh), eff_neighbors=qstats(np.exp(Hh * np.log(10))), max_alpha=qstats(mx),
                                 frac_max_gt_0p2=float((mx > .2).mean()), frac_max_gt_0p5=float((mx > .5).mean()),
                                 per_seed_H_median=np.median(Hh, 1).tolist()),
            per_head=[dict(H_norm=qstats(Hp[..., h]), eff_neighbors=qstats(np.exp(Hp[..., h] * np.log(10))), max_alpha=qstats(mxp[..., h]),
                           frac_max_gt_0p5=float((mxp[..., h] > .5).mean())) for h in range(4)])
    res["concentration"] = conc

    # 3. 기하 의존: alpha*10(균등=1)과 같은 오프셋 평균 대비 비(오프셋 효과 제거)
    dbins = [0, .5, 2, 5, 10, 20, 40, 1e9]
    gbins = [0, 2, 5, 10, 20, 45, 90, 181]
    geo = {}
    for L in range(2):
        a = Ah[:, L].mean(0)                                  # 시드 평균 (n,10)
        rel = a / a.mean(0, keepdims=True)                   # 같은 오프셋 평균 대비
        out = {}
        for key, val, bins in (("distance_m", D, dbins), ("rotation_deg", G, gbins)):
            b = np.digitize(val, bins) - 1
            out[key] = [dict(bin=[bins[k], bins[k + 1]], n_edges=int((b == k).sum()),
                             alpha_x10=float(10 * a[b == k].mean()) if (b == k).any() else None,
                             rel_to_offset_mean=float(rel[b == k].mean()) if (b == k).any() else None)
                        for k in range(len(bins) - 1)]
        # 회전 구간: 같은 노드 안에서 회전 큰 이웃 대 작은 이웃
        big = G > 20
        mixed = big.any(1) & (~big).any(1)
        out["within_node_rot_gt20_vs_le20"] = dict(
            n_nodes=int(mixed.sum()),
            mean_alpha_rot_gt20=float((a * big)[mixed].sum() / big[mixed].sum()),
            mean_alpha_rot_le20=float((a * ~big)[mixed].sum() / (~big)[mixed].sum()),
            rel_gt20=float(rel[mixed][big[mixed]].mean()), rel_le20=float(rel[mixed][~big[mixed]].mean()))
        still = D < .5
        mixed = still.any(1) & (~still).any(1)
        out["within_node_dist_lt0p5_vs_ge0p5"] = dict(
            n_nodes=int(mixed.sum()),
            mean_alpha_lt0p5=float((a * still)[mixed].sum() / max(1, still[mixed].sum())),
            mean_alpha_ge0p5=float((a * ~still)[mixed].sum() / (~still)[mixed].sum()))
        # 간선 단위 상관(오프셋 효과 제거한 rel 기준)
        from scipy.stats import spearmanr
        out["spearman_rel_vs_rotation"] = float(spearmanr(rel.ravel(), G.ravel())[0])
        out["spearman_rel_vs_distance"] = float(spearmanr(rel.ravel(), D.ravel())[0])
        ps = Ah[:, L] / Ah[:, L].mean(1, keepdims=True)       # 시드별 같은 오프셋 평균 대비
        out["per_seed_rel"] = {k: [float(ps[i][m].mean()) for i in range(len(SEEDS))] for k, m in
                               (("rot_gt45", G > 45), ("rot_10_45", (G > 10) & (G <= 45)), ("rot_le2", G <= 2), ("dist_lt0p5", D < .5))}
        out["frac_edges_rot_gt45"] = float((G > 45).mean())
        geo[f"layer{L + 1}"] = out
    res["geometry"] = geo
    res["edge_distance_quantiles"] = {str(q): float(np.quantile(D, q)) for q in (.01, .1, .5, .9, .99)}
    res["edge_rotation_deg_quantiles"] = {str(q): float(np.quantile(G, q)) for q in (.5, .9, .99)}

    # 4. 진행 상황: 노드 i-5~i+5 회전각
    tb = [0, 5, 30, 181]
    prog = {}
    for L in range(2):
        a = Ah[:, L]
        Hh = entropy(a).mean(0)
        mx = a.max(-1).mean(0)
        b = np.digitize(T, tb) - 1
        prog[f"layer{L + 1}"] = [dict(turn_deg=[tb[k], tb[k + 1]], n_nodes=int((b == k).sum()), H_norm=qstats(Hh[b == k]),
                                      max_alpha=qstats(mx[b == k]), profile=a.mean(0)[b == k].mean(0).tolist())
                                 for k in range(len(tb) - 1)]
        stop = SP < 1.0
        prog[f"layer{L + 1}_stationary_span_lt1m"] = dict(n_nodes=int(stop.sum()),
                                                          H_norm=qstats(Hh[stop]) if stop.any() else None)
    res["progress"] = prog

    # 5. 수락 결과: 질의 노드 집중도와 학습 이득
    node_key = {}
    off = 0
    for r, idx in zip(rows, IDX):
        node_key[r["seq"]] = (off, idx)
        off += len(idx)
    Hq = entropy(Ah.mean(0)).mean(0)                          # 시드 평균, 두 층 평균의 노드 엔트로피(헤드 평균 alpha)
    acc = dict(H=[], d=[], pos=[], neg=[], correct=[])
    for seq in S.TESTSEQ:
        o, idx = node_key[seq]
        pos_of = -np.ones(int(idx.max()) + 1 if len(idx) else 1, int)
        pos_of[idx] = np.arange(len(idx)) + o
        u = C.load("uniform", seq)
        news = [C.load(f"new:{s}", seq) for s in SEEDS]
        q = u["q"]
        assert all(np.array_equal(q, x["q"]) for x in news)
        ok = q < len(pos_of)
        ok[ok] = pos_of[q[ok]] >= 0
        qc = classify_queries(u["gt"])
        # 시퀀스 안 백분위 순위(척도 차이 제거)
        def pr(v):
            return np.argsort(np.argsort(v)) / max(1, len(v) - 1)
        dlt = np.mean([pr(x["score"]) for x in news], 0) - pr(u["score"])
        corr = np.mean([x["top1"] <= 5 for x in news], 0)
        acc["H"].append(Hq[pos_of[q[ok]]]); acc["d"].append(dlt[ok]); acc["pos"].append(qc["positive"][ok])
        acc["neg"].append(qc["negative"][ok]); acc["correct"].append(corr[ok])
    H, dl, pos, neg, cor = (np.concatenate(acc[k]) for k in ("H", "d", "pos", "neg", "correct"))
    from scipy.stats import spearmanr
    terc = np.quantile(H[pos], [1 / 3, 2 / 3])
    tb_ = np.digitize(H, terc)
    res["acceptance"] = dict(
        note="질의 노드의 엔트로피(시드·층 평균, 헤드 평균 alpha). 이득 = 시퀀스 안 점수 백분위(NSD 3시드 평균) - 균등 감산 백분위. 양성에서 +는 학습이 순위를 올림, 음성에서 -는 학습이 순위를 내림.",
        n_queries=int(len(H)), n_pos=int(pos.sum()), n_neg=int(neg.sum()),
        pos_spearman_H_vs_gain=float(spearmanr(H[pos], dl[pos])[0]),
        neg_spearman_H_vs_gain=float(spearmanr(H[neg], dl[neg])[0]),
        pos_by_H_tercile=[dict(n=int(((tb_ == k) & pos).sum()), H_median=float(np.median(H[(tb_ == k) & pos])),
                               mean_gain=float(dl[(tb_ == k) & pos].mean()),
                               mean_gain_correct_top1=float(dl[(tb_ == k) & pos & (cor > .5)].mean())) for k in range(3)],
        neg_by_H_tercile=[dict(n=int(((tb_ == k) & neg).sum()), mean_gain=float(dl[(tb_ == k) & neg].mean())) for k in range(3)],
        H_pos_correct=qstats(H[pos & (cor > .5)]), H_pos_wrong=qstats(H[pos & (cor <= .5)]), H_neg=qstats(H[neg]))
    C.write(OUT / "summary.json", res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    P.e.configure()
    torch.set_num_threads(4)
    {"collect": collect, "analyze": analyze}[sys.argv[1]]()
