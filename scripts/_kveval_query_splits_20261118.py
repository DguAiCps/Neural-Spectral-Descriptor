#!/usr/bin/env python3
"""차이 K/V(NSD) 대 절대 K/V(abskv): 평가 조건을 질의 단위로 나눠 NSD가 앞서는 조건이 있는지 본다(2026-11-18).
원장만 읽는 CPU 분석(호스트 numpy·scipy), 재학습·GPU 없음.

  python3 scripts/_kveval_query_splits_20261118.py meta      # 1단계: 공변량만 계산·분포 출력(점수 안 봄)
  python3 scripts/_kveval_query_splits_20261118.py compare   # 2단계: 조건별 NSD−abskv
출력: results/kv_eval_conditions_20261118/query_splits/{meta.npz, meta_distribution.json, summary.json, tables.csv}

==================================================================== 사전 고정
(1단계 `meta`로 공변량 분포만 본 뒤, NSD·abskv 점수를 조건별로 보기 전에 적음. 분포 수치는 meta_distribution.json.
 균등 감산(uniform)은 A5·A7 정의에 쓰는 학습 없는 기준선이라 1단계에서 그 top1 정오·점수 분포를 보았다.)

비교: 시드 4~6 = NSD(new:{4,5,6}, results/pdv2_scratch_search_20261116/ledgers) − abskv(abskv@s{4,5,6}@u512,
      results/pdv2_report_model_20261116/controls/ledgers). 시드 4는 두 쪽 모두 서버 RTX PRO 6000 재학습과 비트 일치
      (results/kv_diff_improve_20261118/sweep1/repro_check/repro_checks.json; 시드 5·6은 같은 캠페인으로 보며 따로 확인하지 않음),
      원장은 로컬 RTX 5080(docs/학습과 선택.md).
      시드 1~3(방향 확인) = results/kv_diff_improve_20261118/sweep1/ledgers의 sym@s{1,2,3}@u512 − abskv@s{1,2,3}@u512.
      두 쪽 모두 같은 스크립트로 서버 RTX PRO 6000에서 학습(sweep1/summary.json training.*.gpu)했고 원장은 같은 캠페인
      (sweep1/logs, 로컬 리포 폴더)에서 만들었다. 같은 탐색 폴더의 NSD 시드 1·2는 RTX PRO 5000에서 학습된 것이라
      (repro_checks.json) 짝 조건이 다르므로 쓰지 않는다(전체 수치만 참고로 낸다).
집계·구간: _kv_subset_analysis_20261117의 metric·macro·boot_single·boot_joint·subset_report를 그대로 쓴다.
      시퀀스 지표 → 도시 안 평균 → 6도시 동일 가중 → 시드 평균. 단일 구간 rng=default_rng(20261104),
      차이의 차이 D = (NSD−abs)_T − (NSD−abs)_C는 같은 시퀀스의 두 질의 부분집합이므로 boot_joint(rng 20261117, 짝지은 재표집).
계산 방식: A1~A5 = 방향 부분집합식(양성만 제한, 음성 전체; A5b만 음성을 제한하고 양성 전체).
      A6·A7 = 질의 자체 속성이라 양성·음성 모두 제한.
정답 프레임 g(q) = argmin_{j <= q-30} |p_gt[j] − p_gt[q]| (3D GT 위치; 원장 min_past_gt_dist와 1e-6 안 일치 확인).
판정: 'NSD 앞섬' = 시드 4~6 NSD−abs의 주 지표(AP; A9는 해당 지표) 95% 구간 하한 > 0 이고 시드 1~3 점추정도 양수.
      D 구간도 같이 낸다. 시도한 조각은 전부 기록하고, 조각 수에 대해 우연 기대치를 함께 따진다.

A1 재방문 시간 간격 Δt = t[q] − t[g(q)] (캐시 timestamps, 초).
   주: 시퀀스 안 양성 질의의 Δt 중앙값으로 긺(T, Δt > 중앙값) 대 짧음(C). 시퀀스별 중앙값 143.7~1147.9 s.
   민감도: 시험 양성 전체 풀(61,682개)의 삼분위 절대 경계 276.60 s, 480.79 s → T1 ≤ 276.60 < T2 ≤ 480.79 < T3. D = T3 − T1.
A2 질의–정답 거리 min_past_gt_dist, 시험 양성 풀 사분위 0.3757, 0.7055, 2.7953 m →
   Q1 ≤ 0.3757 < Q2 ≤ 0.7055 < Q3 ≤ 2.7953 < Q4 ≤ 5. D = Q4 − Q1.
A3 방위 차이: 진행 방향 = GT 위치를 지면 평면(GT 위치의 최소 분산 주축을 법선, 시퀀스마다)에 투영한 p[t] − p[t−5],
   t ≥ 5이고 길이 ≥ 1 m일 때만 유효(_review_direction_metrics의 유효 조건). 질의 q와 g(q)의 진행 방향 사이 각(0~180도).
   유효 양성 61,500개(무효 182개는 A3 모든 구간에서 뺌). 구간: [0,30) 순방향 35,242, [30,150] 교차 2,956, (150,180] 역방향 23,302.
   D = 교차 − 순방향, 역방향 − 순방향. 연속 경향: 30도 6구간(개수 35,242/468/336/588/1,564/23,302), D 없음.
   측방 오프셋 = 평면 위 변위 p[g] − p[q]의 질의 진행 방향 수직 성분 크기. 유효 양성 풀 중앙값 0.4967 m로 큼(T, >) 대 작음(C).
   (측방 오프셋 ≤ 질의–정답 거리이므로 A2와 겹친다는 점을 해석에 적는다.)
A4 과거 방문 횟수: 질의 GT 위치 5 m 안의 과거 프레임(j ≤ q−30)을 색인 순서로 놓고 간격 > 30에서 끊은 묶음 수.
   양성 분포 1회 52,551 / 2회 8,475 / 3회 656. T = 2회 이상, C = 1회.
A5 앨리어싱 위험(균등 감산 uniform 원장으로 정의; 두 모델과 후보 풀을 공유하므로 정의는 두 모델에 대칭이지만
   조건이 모델 출력(uniform의 반환 장소·점수)에 의존함을 해석에 적는다).
   (a) 양성 중 uniform top1이 틀린 질의(top1_geo_dist > 5 m, 6,302개, T) 대 맞은 질의(C). 음성 전체.
   (b) 음성을 시퀀스 안 uniform 점수 중앙값 이상(어려운 음성, T) 대 미만(C)으로 제한, 양성 전체. 시퀀스당 음성 최소 190개.
A6 국소 주행(질의는 30..n−16이라 ±5 창이 잘리지 않음).
   (a) 회전 = 주행 추정 자세(KISS-ICP, 방법이 쓰는 것) q−5와 q+5의 상대 회전 크기(도, 축 무관; 주행에서는 대부분 요).
       HeLiPR 주행 추정 자세의 z축이 세계 z와 맞지 않는 구간이 있어 요 성분 대신 축 무관 크기를 쓴다.
       T = 시퀀스 안 상위 20%(>= 80% 분위), C = 나머지. 시퀀스별 80% 분위 1.30~35.45도.
   (b) 시간 불규칙 = t[q−5..q+5]의 10개 간격 표준편차, T = 시퀀스 안 상위 20%(>= 80% 분위), C = 나머지.
       80% 분위가 KITTI-360 0(부동소수 잡음 수준), HeLiPR KAIST06·DCC·Riverside와 MulRan 전부 0.5~3 ms라 이들 시퀀스의
       T는 실제 불규칙이 아니다. 민감도: 80% 분위 >= 0.01 s인 시퀀스(KITTI 00·07·08, HeLiPR Town01~03·KAIST05, NCLT 9, MCD 12)만.
A7 시퀀스 경계. 질의 쪽: 판정에 필요한 마지막 관측 q+15가 시퀀스 끝을 넘는 질의는 0개(원장 질의가 q <= n−16) → 수만 보고.
   후보 쪽: 후보 시간 이웃 창 j±10(sequence_evidence.distance_indices radius 10)이 시작에서 잘리는 질의.
   승자 j는 모델마다 다르므로 uniform의 j_star로 정의(대칭). T = uniform j_star < 10(양성 177·음성 2,277, 양성 있는 시퀀스 18),
   C = 나머지, 양성·음성 모두 제한. 민감도: j_star < 20(이웃의 수정 계수 수용역 j−20까지 잘림; 양성 343, 시퀀스 23).
A8 규약 변경(현재: 양성 <= 5 m, 음성 > 25 m, 정답 반환 <= 5 m, R@1 <= 5 m; _pr_eval_lib).
   (a) 양성 반경 r ∈ {3, 10, 25} m, 정답 반환 반경도 r(분야 관행: 양성 판정과 정답 판정 반경이 같음), 음성 > 25 m 유지.
       pr_sweep에 top1 × 5/r를 넣어 정답 판정을 r로 바꾼다(라이브러리 수정 없음). D = (r 규약) − (현재 규약), 짝지은 재표집.
   (b) R@1 반경 2, 10 m(양성 집합은 현재 그대로). R@1만 의미가 있다. D = (반경) − (5 m).
   (c) 도시 안 질의 풀링: 도시 안 시퀀스 질의를 합쳐 한 번의 임계 스윕 → 6도시 평균 → 시드 평균.
       구간: 도시 안 시퀀스 복원추출(10,000회, rng 20261104) 후 풀링·스윕. D 없음.
A9 다른 작동점: R@99P, R@95P(pr_sweep), R@90P(정밀도 >= 0.90인 최대 재현율, 같은 정의),
   P@R50·P@R90 = 재현율 >= 0.5/0.9인 스윕 점의 최대 정밀도(보간 정밀도), 그 재현율에 못 닿으면 0.
   같은 질의의 다른 요약 통계라 T/C가 없어 D 없음.
====================================================================
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _kv_subset_analysis_20261117 as K  # noqa: E402
import _report_model_common_20261116 as C  # noqa: E402

OUT = ROOT / "results/kv_eval_conditions_20261118/query_splits"
GAP = 30            # 과거 후보 = j <= q-30 (_pr_eval_lib.MIN_INDEX_GAP, controls pipe의 assert j <= q-30)
POS_R = 5.0         # _pr_eval_lib.POSITIVE_MAX_M


def cache_arrays(seq: str) -> dict:
    """원장이 쓴 캐시(K.motion과 같은 경로 찾기): GT 위치·GT 자세·주행 추정 자세·시각."""
    name = seq.replace("/", "_")
    audit = json.loads((C.OLD / "test" / name / "summary.json").read_text())["input_audit"]
    (cache,) = [k for k in audit["inputs"] if "/cache_" in k]
    with np.load(ROOT / cache, allow_pickle=False) as z:
        out = dict(gt=z["poses_gt"][:, :3, 3].astype(np.float64), odo=z["poses"][:, :3, :3].astype(np.float64),
                   t=z["timestamps"].astype(np.float64))
    assert audit["nodes"] == len(out["t"])
    return out


def rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """두 회전의 상대 회전 크기(도, 축 무관)."""
    tr = np.einsum("nji,nji->n", Ra, Rb)  # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def seq_meta(seq: str) -> dict:
    name = seq.replace("/", "_")
    with np.load(C.SEARCH / "ledgers" / "test" / name / "uniform.npz") as z:
        q, gtd = z["query_idx"].astype(np.int64), z["min_past_gt_dist"].astype(np.float64)
        uj, utop1, uscore = z["j_star"].astype(np.int64), z["top1_geo_dist"].astype(np.float64), z["score"].astype(np.float64)
    a = cache_arrays(seq)
    pos, t, n = a["gt"], a["t"], len(a["t"])
    # 정답 프레임: 과거(j <= q-30) 중 GT 위치가 가장 가까운 프레임. 원장 min_past_gt_dist와 일치 확인.
    g = np.empty(len(q), np.int64)
    dmin = np.empty(len(q))
    for i, qq in enumerate(q):
        d = np.linalg.norm(pos[: qq - GAP + 1] - pos[qq], axis=1)
        g[i] = int(np.argmin(d))
        dmin[i] = d[g[i]]
    assert np.allclose(dmin, gtd, atol=1e-6), (seq, float(np.abs(dmin - gtd).max()))
    positive = gtd <= POS_R
    # 지면 평면: GT 위치의 최소 분산 주축을 법선으로(데이터셋마다 GT 축 규약이 달라서)
    x = pos - pos.mean(0)
    ev, evec = np.linalg.eigh(np.cov(x.T))
    assert ev[0] < 0.01 * ev[1], (seq, ev)
    nrm = evec[:, 0]
    flat = pos - np.outer(pos @ nrm, nrm)
    idx = np.arange(n)
    hv = flat - flat[np.maximum(idx - 5, 0)]
    hl = np.linalg.norm(hv, axis=1)
    hvalid = (idx >= 5) & (hl >= 1.0)  # _review_direction_metrics.classify_direction와 같은 유효 조건(평면 투영)
    h = hv / np.maximum(hl[:, None], 1e-12)
    yawdiff = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", h[q], h[g]), -1, 1)))
    yvalid = hvalid[q] & hvalid[g]
    disp = flat[g] - flat[q]
    along = np.einsum("ij,ij->i", disp, h[q])
    lateral = np.linalg.norm(disp - along[:, None] * h[q], axis=1)
    # 과거 방문 횟수: 질의 5 m 안의 과거 프레임(j <= q-30)을 색인 간격 > 30에서 끊은 묶음 수
    tree = cKDTree(pos)
    visits = np.zeros(len(q), np.int64)
    for i in np.flatnonzero(positive):
        ids = np.sort(np.asarray(tree.query_ball_point(pos[q[i]], POS_R), np.int64))
        ids = ids[ids <= q[i] - GAP]
        assert len(ids)
        visits[i] = 1 + int((np.diff(ids) > GAP).sum())
    # 국소 주행: ±5 키프레임(질의는 30..n-16이라 잘리지 않음)
    assert q.min() - 5 >= 0 and q.max() + 5 <= n - 1
    rot = rot_angle(a["odo"][q - 5], a["odo"][q + 5])           # 주행 추정 자세(방법이 쓰는 것)의 상대 회전 크기
    win = q[:, None] + np.arange(-5, 6)[None]
    dtstd = np.diff(t[win], axis=1).std(1)
    return dict(q=q, gtd=gtd, positive=positive, n=n, g=g, dt=t[q] - t[g], yawdiff=yawdiff, yvalid=yvalid,
                lateral=lateral, visits=visits, rot=rot, dtstd=dtstd, q_end_cut=(q + 15 > n - 1),
                uj=uj, uwrong=positive & ~(utop1 <= 5.0), uscore=uscore)


def q_(a, p):
    return [round(float(v), 4) for v in np.quantile(a, p)]


def meta() -> None:
    tick = time.time()
    M = {s: seq_meta(s) for s in C.SEQS}
    OUT.mkdir(parents=True, exist_ok=True)
    flat = {f"{s}|{k}": (np.asarray(v) if not np.isscalar(v) else np.array(v)) for s, m in M.items() for k, v in m.items()}
    np.savez_compressed(OUT / "meta.npz", **flat)
    P = {k: np.concatenate([M[s][k][M[s]["positive"]] for s in C.SEQS]) for k in ("dt", "gtd", "yawdiff", "yvalid", "lateral", "visits")}
    dist = dict(n_seq=len(M), n_queries=int(sum(len(m["q"]) for m in M.values())), n_pos=int(len(P["dt"])),
                n_neg=int(sum((m["gtd"] > 25).sum() for m in M.values())),
                seq_with_pos=int(sum(m["positive"].any() for m in M.values())))
    dist["A1_dt_pos_pool_quantiles_0_1_25_33_50_67_75_99_100"] = q_(P["dt"], [0, .01, .25, 1 / 3, .5, 2 / 3, .75, .99, 1])
    dist["A1_dt_seq_median"] = {s: round(float(np.median(m["dt"][m["positive"]])), 2) for s, m in M.items() if m["positive"].any()}
    dist["A2_gtd_pos_pool_quantiles_0_25_50_75_100"] = q_(P["gtd"], [0, .25, .5, .75, 1])
    yv = P["yawdiff"][P["yvalid"]]
    dist["A3_yaw_valid_pos"] = int(P["yvalid"].sum())
    dist["A3_yaw_invalid_pos"] = int((~P["yvalid"]).sum())
    dist["A3_yaw_hist_30deg"] = np.histogram(yv, bins=np.arange(0, 181, 30))[0].tolist()
    dist["A3_yaw_bins_fwd_cross_rev"] = [int((yv < 30).sum()), int(((yv >= 30) & (yv <= 150)).sum()), int((yv > 150).sum())]
    dist["A3_lateral_pos_pool_quantiles_0_25_50_75_100"] = q_(P["lateral"][P["yvalid"]], [0, .25, .5, .75, 1])
    dist["A4_visits_pos_counts"] = {int(v): int((P["visits"] == v).sum()) for v in np.unique(P["visits"])}
    dist["A5a_uniform_wrong_pos"] = int(sum(m["uwrong"].sum() for m in M.values()))
    dist["A5b_neg_per_seq_min"] = int(min((m["gtd"] > 25).sum() for m in M.values()))
    dist["A6_rot_deg_seq_p80"] = {s: round(float(np.quantile(m["rot"], .8)), 2) for s, m in M.items()}
    dist["A6_rot_deg_all_quantiles_50_80_95"] = q_(np.concatenate([m["rot"] for m in M.values()]), [.5, .8, .95])
    dist["A6_dtstd_seq_p80"] = {s: round(float(np.quantile(m["dtstd"], .8)), 4) for s, m in M.items()}
    dist["A6_dtstd_seq_p50"] = {s: round(float(np.quantile(m["dtstd"], .5)), 4) for s, m in M.items()}
    dist["A6_dtstd_ties_at_p80_frac"] = {s: round(float(np.mean(m["dtstd"] == np.quantile(m["dtstd"], .8))), 3) for s, m in M.items()}
    dist["A7_q_end_cut"] = int(sum(m["q_end_cut"].sum() for m in M.values()))
    for lim in (10, 20):
        dist[f"A7_uniform_j_lt{lim}_all"] = int(sum((m["uj"] < lim).sum() for m in M.values()))
        dist[f"A7_uniform_j_lt{lim}_pos"] = int(sum(((m["uj"] < lim) & m["positive"]).sum() for m in M.values()))
        dist[f"A7_uniform_j_lt{lim}_neg"] = int(sum(((m["uj"] < lim) & (m["gtd"] > 25)).sum() for m in M.values()))
        dist[f"A7_uniform_j_lt{lim}_seqs_with_pos"] = int(sum((((m["uj"] < lim) & m["positive"]).any()) for m in M.values()))
    dist["seconds"] = round(time.time() - tick, 1)
    C.write(OUT / "meta_distribution.json", dist)
    print(json.dumps({k: v for k, v in dist.items() if not k.endswith(("seq_median", "seq_p80", "seq_p50", "p80_frac"))}, ensure_ascii=False, indent=1))


# ===================================================================== 2단계: 조건별 비교
SWEEP1 = ROOT / "results/kv_diff_improve_20261118/sweep1/ledgers/test"
SWEEP1_SUMMARY = ROOT / "results/kv_diff_improve_20261118/sweep1/summary.json"
KEYS = K.KEYS
A9_KEYS = ("R@99P", "R@95P", "R@90P", "P@R50", "P@R90")


def load_sweep1(tag: str, seq: str) -> dict:
    with np.load(SWEEP1 / seq.replace("/", "_") / f"{tag}.npz") as z:
        return dict(score=z["score"].astype(np.float64), top1=z["top1_geo_dist"], gt=z["min_past_gt_dist"], q=z["query_idx"])


def load_meta() -> dict:
    M = {}
    with np.load(OUT / "meta.npz") as z:
        for key in z.files:
            s, k = key.split("|")
            M.setdefault(s, {})[k] = z[key]
    return M


class keys_as:
    """K의 함수들(macro·seq_delta·boot_*·ci·subset_report)이 읽는 모듈 전역 KEYS를 잠시 바꾼다(A9)."""
    def __init__(self, keys):
        self.keys = keys

    def __enter__(self):
        self.old, K.KEYS = K.KEYS, self.keys

    def __exit__(self, *a):
        K.KEYS = self.old


def metric_radius(d: dict, r: float) -> dict | None:
    """A8(a): 양성 <= r, 음성 > 25, 정답 반환 <= r. pr_sweep의 5 m 정답 판정에 top1 × 5/r를 넣는다."""
    pos, neg = d["gt"] <= r, d["gt"] > 25.0
    qc = dict(positive=pos, negative=neg, ambiguous=~(pos | neg), n_pos=int(pos.sum()), n_neg=int(neg.sum()))
    if not qc["n_pos"]:
        return None
    v = K.pr_sweep(d["score"], d["top1"] * (5.0 / r), qc)
    out = {k: float(v[k]) for k in KEYS[:3]}
    out["R@1"] = float(np.mean(d["top1"][pos] <= r))
    out["n_pos"], out["n_neg"] = qc["n_pos"], qc["n_neg"]
    return out


def metric_r1(d: dict, r1: float) -> dict | None:
    """A8(b): 현재 규약에 R@1 반경만 r1."""
    out = K.metric(d)
    if out is not None:
        pos = K.classify_queries(d["gt"])["positive"]
        out["R@1"] = float(np.mean(d["top1"][pos] <= r1))
    return out


def ops_from_sweep(v: dict) -> dict:
    p, r = v["precision"], v["recall"]
    pint = np.maximum.accumulate(p[::-1])[::-1]
    out = {"R@99P": float(v["R@99P"]), "R@95P": float(v["R@95P"]),
           "R@90P": float(r[p >= 0.90].max()) if (p >= 0.90).any() else 0.0}
    for name, x in (("P@R50", 0.5), ("P@R90", 0.9)):
        m = r >= x - 1e-12
        out[name] = float(pint[m].max()) if m.any() else 0.0
    return out


def metric_ops(d: dict) -> dict | None:
    """A9: 현재 규약의 다른 작동점."""
    qc = K.classify_queries(d["gt"])
    if not qc["n_pos"]:
        return None
    v = K.pr_sweep(d["score"], d["top1"], qc)
    out = ops_from_sweep(v)
    out["n_pos"], out["n_neg"] = qc["n_pos"], qc["n_neg"]
    return out


def rows_fn(data: dict, fn) -> dict:
    return {a: [{s: fn(d[s]) for s in C.SEQS} for d in ds] for a, ds in data.items()}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k != "_draws"}
    return o


def pair_report(data: dict, mT: tuple, mC: tuple | None, label: str, Tname="T", Cname="C") -> dict:
    """mT, mC = (pos_mask_dict|None, neg_mask_dict|None). D = T − C(같은 시퀀스 부분집합, 짝지은 재표집)."""
    rT = K.rows_for(data, C.SEQS, *mT)
    out = {"definition": label, Tname: K.subset_report(rT, C.CITY)}
    if mC is not None:
        rC = K.rows_for(data, C.SEQS, *mC)
        out[Cname] = K.subset_report(rC, C.CITY)
        d = K.boot_joint(rT["nsd"], rT["abs"], rC["nsd"], rC["abs"], C.CITY, np.random.default_rng(20261117))
        out["D_name"] = f"{Tname}-{Cname}"
        out["D_pp"] = {k: out[Tname]["nsd_minus_abs_pp"][k] - out[Cname]["nsd_minus_abs_pp"][k] for k in K.KEYS}
        out["D_ci95_pp"] = K.ci(d)
    return out


def bins_report(data: dict, masks: dict, neg_restrict: bool, label: str, dpairs=()) -> dict:
    """여러 구간: masks = {구간: {seq: mask}}. dpairs = [(T, C), ...]."""
    rows = {b: K.rows_for(data, C.SEQS, m, m if neg_restrict else None) for b, m in masks.items()}
    out = {"definition": label, "bins": {b: K.subset_report(r, C.CITY) for b, r in rows.items()}, "D": {}}
    for t, c in dpairs:
        d = K.boot_joint(rows[t]["nsd"], rows[t]["abs"], rows[c]["nsd"], rows[c]["abs"], C.CITY, np.random.default_rng(20261117))
        out["D"][f"{t}-{c}"] = dict(D_pp={k: out["bins"][t]["nsd_minus_abs_pp"][k] - out["bins"][c]["nsd_minus_abs_pp"][k] for k in K.KEYS},
                                    D_ci95_pp=K.ci(d))
    return out


# ------------------------------------------------------------------ A8(c) 도시 안 풀링
def _pooled_city(dlist: list, seqs: list) -> dict:
    sc = np.concatenate([dlist[s]["score"] for s in seqs])
    t1 = np.concatenate([dlist[s]["top1"] for s in seqs])
    gt = np.concatenate([dlist[s]["gt"] for s in seqs])
    qc = K.classify_queries(gt)
    v = K.pr_sweep(sc, t1, qc)
    out = [float(v[k]) for k in KEYS[:3]] + [float(np.mean(t1[qc["positive"]] <= 5))]
    return np.array(out)


_POOL = {}


def _pooled_rep(args):
    seed, n = args
    data, groups = _POOL["data"], _POOL["groups"]
    rng = np.random.default_rng(seed)
    res = np.empty((n, len(KEYS)))
    for i in range(n):
        city = []
        for seqs in groups.values():
            pick = [seqs[k] for k in rng.integers(0, len(seqs), size=len(seqs))]
            a = np.mean([_pooled_city(d, pick) for d in data["nsd"]], 0)
            b = np.mean([_pooled_city(d, pick) for d in data["abs"]], 0)
            city.append(100 * (a - b))
        res[i] = np.mean(city, 0)
    return res


def pooled_report(data: dict) -> dict:
    import multiprocessing as mp
    groups = {c: [s for s in v if any(K.classify_queries(data["nsd"][0][s]["gt"])["positive"])] for c, v in C.CITY.items()}
    groups = {c: v for c, v in groups.items() if v}
    means = {}
    for a, ds in data.items():
        per = [np.mean([_pooled_city(d, v) for v in groups.values()], 0) for d in ds]
        means[a] = dict(zip(KEYS, (100 * np.mean(per, 0)).tolist()))
    point = {k: means["nsd"][k] - means["abs"][k] for k in KEYS}
    _POOL.update(data=data, groups=groups)
    ss = np.random.SeedSequence(C.BOOT_SEED).spawn(16)
    chunks = [(int(s.generate_state(1)[0]), C.BOOT // 16) for s in ss]
    with mp.get_context("fork").Pool(12) as pool:
        draws = np.concatenate(pool.map(_pooled_rep, chunks))
    return dict(mean_pp=means, nsd_minus_abs_pp=point, ci95_pp=K.ci(draws), n_boot=len(draws),
                note="풀링 재표집은 SeedSequence(20261104)를 16조각으로 나눈 병렬 난수(단일 구간 규약의 순차 난수와 다름)")


# ------------------------------------------------------------------ 조건 정의(사전 고정 경계)
A1_TERT = (276.60, 480.79)
A2_QUART = (0.3757, 0.7055, 2.7953)
A3_LAT = 0.4967


def conditions(M: dict) -> list:
    """(이름, 종류, 인자). 경계는 docstring의 사전 고정 값. 마스크는 질의 순서(원장 query_idx)."""
    S = C.SEQS
    pos = {s: M[s]["positive"] for s in S}
    every = {s: np.ones(len(M[s]["q"]), bool) for s in S}
    out = []
    # A1
    long_ = {s: pos[s] & (M[s]["dt"] > (np.median(M[s]["dt"][pos[s]]) if pos[s].any() else np.inf)) for s in S}
    short = {s: pos[s] & ~long_[s] for s in S}
    out.append(("A1_dt_seqmedian", "pair", dict(mT=(long_, None), mC=(short, None), label="Δt 시퀀스 안 양성 중앙값: 긺(T) 대 짧음(C), 양성만 제한", Tname="long", Cname="short")))
    lo, hi = A1_TERT
    t3 = {"T1": {s: M[s]["dt"] <= lo for s in S}, "T2": {s: (M[s]["dt"] > lo) & (M[s]["dt"] <= hi) for s in S},
          "T3": {s: M[s]["dt"] > hi for s in S}}
    out.append(("A1_dt_tertile_sens", "bins", dict(masks=t3, neg_restrict=False, label="민감도: Δt 풀 삼분위 276.60/480.79 s", dpairs=[("T3", "T1")])))
    # A2
    e = (0.0,) + A2_QUART + (POS_R,)
    q4 = {f"Q{i + 1}": {s: (M[s]["gtd"] > e[i] if i else M[s]["gtd"] >= 0) & (M[s]["gtd"] <= e[i + 1]) for s in S} for i in range(4)}
    out.append(("A2_gtdist_quartile", "bins", dict(masks=q4, neg_restrict=False, label="질의–정답 거리 풀 사분위 0.3757/0.7055/2.7953 m", dpairs=[("Q4", "Q1")])))
    # A3
    yv = {s: M[s]["yvalid"] & pos[s] for s in S}
    y = {s: M[s]["yawdiff"] for s in S}
    a3 = {"forward": {s: yv[s] & (y[s] < 30) for s in S}, "cross": {s: yv[s] & (y[s] >= 30) & (y[s] <= 150) for s in S},
          "reverse": {s: yv[s] & (y[s] > 150) for s in S}}
    out.append(("A3_yaw_3bins", "bins", dict(masks=a3, neg_restrict=False, label="방위 차이 [0,30) / [30,150] / (150,180]", dpairs=[("cross", "forward"), ("reverse", "forward")])))
    b6 = {f"{a}-{a + 30}": {s: yv[s] & (y[s] >= a) & ((y[s] < a + 30) if a < 150 else (y[s] <= 180)) for s in S} for a in range(0, 180, 30)}
    out.append(("A3_yaw_30deg_trend", "bins", dict(masks=b6, neg_restrict=False, label="방위 차이 30도 6구간(연속 경향, D 없음)", dpairs=[])))
    lat_hi = {s: yv[s] & (M[s]["lateral"] > A3_LAT) for s in S}
    lat_lo = {s: yv[s] & (M[s]["lateral"] <= A3_LAT) for s in S}
    out.append(("A3_lateral_median", "pair", dict(mT=(lat_hi, None), mC=(lat_lo, None), label="측방 오프셋 풀 중앙값 0.4967 m: 큼(T) 대 작음(C)", Tname="large", Cname="small")))
    # A4
    multi = {s: pos[s] & (M[s]["visits"] >= 2) for s in S}
    single = {s: pos[s] & (M[s]["visits"] == 1) for s in S}
    out.append(("A4_visits", "pair", dict(mT=(multi, None), mC=(single, None), label="과거 방문 2회 이상(T) 대 1회(C)", Tname="multi", Cname="single")))
    # A5
    uw = {s: M[s]["uwrong"] for s in S}
    ur = {s: pos[s] & ~M[s]["uwrong"] for s in S}
    out.append(("A5a_uniform_top1", "pair", dict(mT=(uw, None), mC=(ur, None), label="uniform top1 틀린 양성(T) 대 맞은 양성(C), 음성 전체", Tname="uniform_wrong", Cname="uniform_right")))
    neg = {s: M[s]["gtd"] > 25 for s in S}
    med = {s: np.median(M[s]["uscore"][neg[s]]) for s in S}
    hard = {s: neg[s] & (M[s]["uscore"] >= med[s]) for s in S}
    easy = {s: neg[s] & (M[s]["uscore"] < med[s]) for s in S}
    out.append(("A5b_hard_negatives", "pair", dict(mT=(None, hard), mC=(None, easy), label="음성을 uniform 점수 시퀀스 안 중앙값 이상(T)/미만(C)으로 제한, 양성 전체", Tname="hard_neg", Cname="easy_neg")))
    # A6
    top = lambda m: {s: m[s] >= np.quantile(m[s], 0.8) for s in S}  # noqa: E731
    rot, irr = top({s: M[s]["rot"] for s in S}), top({s: M[s]["dtstd"] for s in S})
    out.append(("A6a_rotation_top20", "pair", dict(mT=(rot, rot), mC=({s: ~rot[s] for s in S},) * 2, label="±5 상대 회전 크기 시퀀스 안 상위 20%(T) 대 나머지(C), 양성·음성 제한", Tname="turn", Cname="rest")))
    out.append(("A6b_dt_irregular_top20", "pair", dict(mT=(irr, irr), mC=({s: ~irr[s] for s in S},) * 2, label="±5 시간 간격 표준편차 상위 20%(T) 대 나머지(C), 양성·음성 제한", Tname="irregular", Cname="rest")))
    real = [s for s in S if np.quantile(M[s]["dtstd"], 0.8) >= 0.01]
    nil = {s: np.zeros(len(M[s]["q"]), bool) for s in S}
    irr2 = {s: irr[s] if s in real else nil[s] for s in S}
    rest2 = {s: ~irr[s] if s in real else nil[s] for s in S}
    out.append(("A6b_sens_real_irregular_seqs", "pair", dict(mT=(irr2, irr2), mC=(rest2, rest2), label=f"민감도: 80% 분위 >= 0.01 s 시퀀스 {len(real)}개만", Tname="irregular", Cname="rest")))
    # A7
    for lim in (10, 20):
        cut = {s: M[s]["uj"] < lim for s in S}
        out.append((f"A7_candidate_start_j_lt{lim}" + ("" if lim == 10 else "_sens"), "pair",
                    dict(mT=(cut, cut), mC=({s: ~cut[s] for s in S},) * 2, label=f"uniform j_star < {lim}(T) 대 나머지(C), 양성·음성 제한",
                         Tname="cut", Cname="rest")))
    return out


def run_group(data: dict, M: dict) -> dict:
    res = {}
    full = K.rows_for(data, C.SEQS)
    res["test_all"] = K.subset_report(full, C.CITY)
    for name, kind, arg in conditions(M):
        res[name] = pair_report(data, **arg) if kind == "pair" else bins_report(data, **arg)
    # A8(a)
    a8 = {}
    for r in (3.0, 10.0, 25.0):
        rr = rows_fn(data, lambda d, r=r: metric_radius(d, r))
        rep = K.subset_report(rr, C.CITY)
        dd = K.boot_joint(rr["nsd"], rr["abs"], full["nsd"], full["abs"], C.CITY, np.random.default_rng(20261117))
        a8[f"pos{int(r)}m"] = dict(report=rep, D_name="r규약-현재", D_pp={k: rep["nsd_minus_abs_pp"][k] - res["test_all"]["nsd_minus_abs_pp"][k] for k in KEYS},
                                   D_ci95_pp=K.ci(dd))
    res["A8a_positive_radius"] = a8
    a8b = {}
    for r1 in (2.0, 10.0):
        rr = rows_fn(data, lambda d, r1=r1: metric_r1(d, r1))
        rep = K.subset_report(rr, C.CITY)
        dd = K.boot_joint(rr["nsd"], rr["abs"], full["nsd"], full["abs"], C.CITY, np.random.default_rng(20261117))
        a8b[f"R1_{int(r1)}m"] = dict(report=rep, D_name="반경-5m", D_pp={k: rep["nsd_minus_abs_pp"][k] - res["test_all"]["nsd_minus_abs_pp"][k] for k in KEYS},
                                     D_ci95_pp=K.ci(dd), note="R@1만 의미; 나머지 지표는 현재 규약과 같음")
    res["A8b_R1_radius"] = a8b
    res["A8c_pooled_city"] = pooled_report({a: v for a, v in data.items() if a in ("nsd", "abs")})
    with keys_as(A9_KEYS):
        res["A9_operating_points"] = K.subset_report(rows_fn(data, metric_ops), C.CITY)
    return res


def compare() -> None:
    tick = time.time()
    M = load_meta()
    d456 = {"nsd": [{s: C.load(f"new:{k}", s) for s in C.SEQS} for k in C.NEW_SEEDS],
            "abs": [{s: K.load(f"abskv@s{k}@u512", s, "test") for s in C.SEQS} for k in C.NEW_SEEDS],
            "uniform": [{s: C.load("uniform", s) for s in C.SEQS}]}
    d123 = {"nsd": [{s: load_sweep1(f"sym@s{k}@u512", s) for s in C.SEQS} for k in (1, 2, 3)],
            "abs": [{s: load_sweep1(f"abskv@s{k}@u512", s) for s in C.SEQS} for k in (1, 2, 3)],
            "uniform": d456["uniform"]}
    for s in C.SEQS:   # 같은 질의·같은 정답 라벨
        for D in (d456, d123):
            for ds in D.values():
                for d in ds:
                    np.testing.assert_array_equal(d[s]["q"], M[s]["q"])
                    np.testing.assert_array_equal(d[s]["gt"], M[s]["gtd"])
    out = dict(conventions=__doc__.split("====================================================================")[1].strip())
    out["s456"] = run_group(d456, M)
    out["s123"] = run_group(d123, M)
    # 재현 확인
    got = {a: tuple(round(out["s456"]["test_all"]["mean_pp"][a][k], 3) for k in KEYS) for a in ("nsd", "abs")}
    assert got["nsd"] == K.EXPECT["nsd"] and got["abs"] == K.EXPECT["abs"], got
    ref = json.loads((ROOT / "results/pdv2_report_model_20261116/controls/summary.json").read_text())["test"]["all"]["differences"]["nsd-abskv"]
    assert all(abs(ref["point_pp"][k] - out["s456"]["test_all"]["nsd_minus_abs_pp"][k]) < 1e-9 for k in KEYS)
    assert all(np.allclose(ref["conditional_95_pp"][k], out["s456"]["test_all"]["ci95_pp"][k], atol=1e-9) for k in KEYS)
    sw = json.loads(SWEEP1_SUMMARY.read_text())["test"]
    for a, arm in (("nsd", "sym"), ("abs", "abskv")):
        for k in KEYS:
            assert abs(out["s123"]["test_all"]["mean_pp"][a][k] - 100 * sw["arms"][arm][k]["mean"]) < 1e-9, (a, k)
    sref = sw["differences"]["sym-abskv"]
    assert all(np.allclose(sref["conditional_95_pp"][k], out["s123"]["test_all"]["ci95_pp"][k], atol=1e-9) for k in KEYS)
    # 참고: 원 탐색 폴더 NSD 시드 1~3(시드 1·2는 PRO 5000 학습) − sweep1 abskv 시드 1~3, 전체만
    dref = {"nsd": [{s: C.load(f"new:{k}", s) for s in C.SEQS} for k in (1, 2, 3)], "abs": d123["abs"]}
    out["ref_scratch_nsd_s123_minus_sweep1_abskv_all"] = strip(K.subset_report(K.rows_for(dref, C.SEQS), C.CITY))
    out["check"] = dict(s456_test_all_matches_EXPECT=True, s456_ci_matches_controls_summary=True,
                        s123_matches_sweep1_summary=True, queries_and_labels_identical_all_ledgers=True,
                        A7_query_end_cut_queries=int(sum(M[s]["q_end_cut"].sum() for s in C.SEQS)))
    out["seconds"] = time.time() - tick
    out = strip(out)
    C.write(OUT / "summary.json", out)
    write_tables(out)
    print("seconds", round(out["seconds"], 1))


def write_tables(out: dict) -> None:
    head = ["condition", "slice", "kind", "n_seq", "n_pos", "n_neg"]
    keys4 = list(KEYS)
    cols = head + [f"s456_{a}_{k}" for a in ("nsd", "abs") for k in keys4] + \
        [f"s456_{x}_{k}" for k in keys4 for x in ("d", "lo", "hi")] + [f"s123_{x}_{k}" for k in keys4 for x in ("d", "lo", "hi")]
    lines = [",".join(cols)]

    def fmt(v):
        return f"{v:.3f}"

    def rep_line(cond, sl, r4, r1, keys=keys4):
        row = [cond, sl, "slice", str(r4["n_sequences_with_pos"]), str(r4["n_pos"]), str(r4["n_neg"])]
        row += [fmt(r4["mean_pp"][a][k]) for a in ("nsd", "abs") for k in keys]
        row += [fmt(x) for k in keys for x in (r4["nsd_minus_abs_pp"][k], *r4["ci95_pp"][k])]
        row += [fmt(x) for k in keys for x in (r1["nsd_minus_abs_pp"][k], *r1["ci95_pp"][k])]
        lines.append(",".join(row))

    def d_line(cond, name, d4, d1, keys=keys4):
        row = [cond, "D=" + name, "D", "", "", ""] + [""] * (2 * len(keys))
        row += [fmt(x) for k in keys for x in (d4["D_pp"][k], *d4["D_ci95_pp"][k])]
        row += [fmt(x) for k in keys for x in (d1["D_pp"][k], *d1["D_ci95_pp"][k])]
        lines.append(",".join(row))

    a, b = out["s456"], out["s123"]
    rep_line("test_all", "all", a["test_all"], b["test_all"])
    for cond, v in a.items():
        if cond in ("test_all",) or cond.startswith(("A8", "A9")):
            continue
        if "bins" in v:
            for sl, r in v["bins"].items():
                rep_line(cond, sl, r, b[cond]["bins"][sl])
            for dn, dv in v["D"].items():
                d_line(cond, dn, dv, b[cond]["D"][dn])
        else:
            for sl in (k for k in v if isinstance(v[k], dict) and "mean_pp" in v[k]):
                rep_line(cond, sl, v[sl], b[cond][sl])
            if "D_pp" in v:
                d_line(cond, v["D_name"], v, b[cond])
    for sl, v in a["A8a_positive_radius"].items():
        rep_line("A8a_positive_radius", sl, v["report"], b["A8a_positive_radius"][sl]["report"])
        d_line("A8a_positive_radius", sl + "-current", v, b["A8a_positive_radius"][sl])
    for sl, v in a["A8b_R1_radius"].items():
        rep_line("A8b_R1_radius", sl, v["report"], b["A8b_R1_radius"][sl]["report"])
        d_line("A8b_R1_radius", sl + "-5m", v, b["A8b_R1_radius"][sl])
    p4, p1 = a["A8c_pooled_city"], b["A8c_pooled_city"]
    lines.append(",".join(["A8c_pooled_city", "pooled", "slice", "", "", ""] + [fmt(p4["mean_pp"][x][k]) for x in ("nsd", "abs") for k in keys4]
                          + [fmt(x) for k in keys4 for x in (p4["nsd_minus_abs_pp"][k], *p4["ci95_pp"][k])]
                          + [fmt(x) for k in keys4 for x in (p1["nsd_minus_abs_pp"][k], *p1["ci95_pp"][k])]))
    lines.append("")
    a9, b9 = a["A9_operating_points"], b["A9_operating_points"]
    lines.append("A9 metric,n_seq,n_pos,n_neg,s456_nsd,s456_abs,s456_d,s456_lo,s456_hi,s123_d,s123_lo,s123_hi")
    for k in A9_KEYS:
        lines.append(",".join([k, str(a9["n_sequences_with_pos"]), str(a9["n_pos"]), str(a9["n_neg"]), fmt(a9["mean_pp"]["nsd"][k]), fmt(a9["mean_pp"]["abs"][k]),
                               *(fmt(x) for x in (a9["nsd_minus_abs_pp"][k], *a9["ci95_pp"][k])),
                               *(fmt(x) for x in (b9["nsd_minus_abs_pp"][k], *b9["ci95_pp"][k]))]))
    (OUT / "tables.csv").write_text("\n".join(lines) + "\n")


def posthoc() -> None:
    """사후 확인(결과를 본 뒤 추가, 판정·경계에 쓰지 않음): A6a 회전 조각이 양성 쪽인지 음성 쪽인지와 도시별 분포.
    경계는 바꾸지 않았다(시퀀스 안 상위 20% 그대로)."""
    M = load_meta()
    S = C.SEQS
    d456 = {"nsd": [{s: C.load(f"new:{k}", s) for s in S} for k in C.NEW_SEEDS],
            "abs": [{s: K.load(f"abskv@s{k}@u512", s, "test") for s in S} for k in C.NEW_SEEDS]}
    d123 = {"nsd": [{s: load_sweep1(f"sym@s{k}@u512", s) for s in S} for k in (1, 2, 3)],
            "abs": [{s: load_sweep1(f"abskv@s{k}@u512", s) for s in S} for k in (1, 2, 3)]}
    rot = {s: M[s]["rot"] >= np.quantile(M[s]["rot"], 0.8) for s in S}
    rest = {s: ~rot[s] for s in S}
    out = {}
    for g, data in (("s456", d456), ("s123", d123)):
        r = {}
        r["pos_only"] = pair_report(data, (rot, None), (rest, None), "회전 상위 20%: 양성만 제한, 음성 전체", "turn", "rest")
        r["neg_only"] = pair_report(data, (None, rot), (None, rest), "회전 상위 20%: 음성만 제한, 양성 전체", "turn", "rest")
        rows = K.rows_for(data, S, rot, rot)
        r["turn_by_city"] = {c: K.subset_report(K.rows_for(data, v, rot, rot), {c: v})["nsd_minus_abs_pp"] for c, v in C.CITY.items()}
        rows = K.rows_for(data, S, rest, rest)
        r["rest_by_city"] = {c: K.subset_report(K.rows_for(data, v, rest, rest), {c: v})["nsd_minus_abs_pp"] for c, v in C.CITY.items()}
        out[g] = strip(r)
    out["note"] = posthoc.__doc__
    C.write(OUT / "posthoc_A6a.json", out)
    for g in ("s456", "s123"):
        for k in ("pos_only", "neg_only"):
            v = out[g][k]
            print(g, k, "turn", {m: (round(v["turn"]["nsd_minus_abs_pp"][m], 3), [round(x, 3) for x in v["turn"]["ci95_pp"][m]]) for m in KEYS})
            print(g, k, "D", {m: (round(v["D_pp"][m], 3), [round(x, 3) for x in v["D_ci95_pp"][m]]) for m in KEYS})
        print(g, "turn_by_city AP", {c: round(x["AP"], 3) for c, x in out[g]["turn_by_city"].items()})
        print(g, "rest_by_city AP", {c: round(x["AP"], 3) for c, x in out[g]["rest_by_city"].items()})


if __name__ == "__main__":
    {"meta": meta, "compare": compare, "posthoc": posthoc}[sys.argv[1]]()
