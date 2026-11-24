#!/usr/bin/env python3
"""질의 속성별 비교의 시드·도시·시퀀스 견고성 확인(2026-11-18). 원장만 읽는 CPU 분석.

  python3 scripts/_kveval_split_robustness_20261118.py
출력: results/kv_eval_conditions_20261118/query_splits_robustness/summary.json

==================================================================== 규약
조각 정의·경계는 scripts/_kveval_query_splits_20261118.py의 사전 고정값 그대로(새 경계·새 교집합 없음):
  A2_Q1   양성 중 min_past_gt_dist <= 0.3757 m, 음성 전체
  A2_Q2   양성 중 0.3757 < d <= 0.7055 m, 음성 전체
  A3_small 방위 유효 양성 중 측방 오프셋 <= 0.4967 m, 음성 전체
  A6a_turn ±5 주행 추정 자세 상대 회전 크기 시퀀스 안 상위 20%(>= 80% 분위), 양성·음성 모두 제한.
           D = turn − rest(같은 시퀀스 두 부분집합, 짝지은 재표집 rng 20261117).
  시험 마스크는 앞선 스크립트의 conditions(meta)와 같은지 검산한다.
집계·구간: _kv_subset_analysis_20261117 함수(rows_for·subset_report·boot_joint) 그대로.
  시퀀스 지표 → 묶음 안 평균 → 묶음 동일 가중 → 시드 평균. 단일 구간 rng 20261104, 묶음 안 시퀀스 재표집 10,000회.
  시험 묶음 = 6도시(city.json), 검증 묶음 = H4와 같은 3계열(helipr·kitti[KITTI-360 포함]·mulran).
비교 짝:
  test_s456   NSD new:{4,5,6} − abskv@s{4,5,6}(controls)                        (앞선 스크립트와 같음)
  test_s123   sweep1 sym@s{1,2,3} − sweep1 abskv@s{1,2,3}                         (앞선 스크립트와 같음)
  val_s456    검증9 NSD new:{4,5,6}(pdv2_scratch_search val) − abskv@s{4,5,6}(controls val)  (H4와 같음)
  val_s123    검증9 sweep1 sym@s{1,2,3} − sweep1 abskv@s{1,2,3}(val)              (방향 확인)
  aux_s456    final_s456(로컬 RTX 5080 재학습) sym@s{4,5,6} − abskv@s{4,5,6}, 시험46
  aux 참고    final_s456 diffk_absv·sum_kv − abskv(시드 4~6), 시험46
검증 메타데이터: 앞선 스크립트의 seq_meta를 그대로 쓴다. seq_meta가 시험 경로를 고정해 두었으므로 tmp에 검증 원장·
  검증 감사 폴더(results/pdv2_diffattn_contrast_pipeline_20261104/val)를 가리키는 심볼릭 링크를 만들고 C.SEARCH·C.OLD를
  잠시 그쪽으로 돌린다(파일 복사 없음, 끝나면 링크 삭제). 경계는 시험 풀에서 정한 절대값 0.3757·0.7055·0.4967 m를 그대로,
  회전은 검증 시퀀스 안 상위 20%.
시드별 값: 시드 k의 (NSD 매크로 − abskv 매크로), 짝은 위와 같음(시드 1~3 = sweep1 sym/abskv, 시드 4~6 = new/controls abskv).
도시별: 묶음 안 시퀀스 평균의 (NSD 시드 평균 − abs 시드 평균). leave-one-city-out = 나머지 5도시 동일 가중 평균.
  leave-one-sequence-out = 시퀀스 하나를 뺀 6도시 매크로(시퀀스 영향 확인).
A6a 공변량: 점수와 무관한 메타 통계(양성의 방위 구간·측방 오프셋·정답 거리 분포, 양성 비율, 도시 구성).
====================================================================
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _kveval_query_splits_20261118 as P  # noqa: E402

K, C = P.K, P.C
KEYS = K.KEYS
BASE = ROOT / "results/kv_eval_conditions_20261118"
OUT = BASE / "query_splits_robustness"
SHIM = BASE / "tmp/robust_valshim"
FINAL = ROOT / "results/kv_diff_improve_20261118/final_s456/ledgers"
SWEEP1_VAL = ROOT / "results/kv_diff_improve_20261118/sweep1/ledgers/val"
VAL_AUDIT = ROOT / "results/pdv2_diffattn_contrast_pipeline_20261104/val"
SLICES = ("A2_Q1", "A2_Q2", "A3_small", "A6a_turn")


def ld(p: Path) -> dict:
    with np.load(p) as z:
        return dict(score=z["score"].astype(np.float64), top1=z["top1_geo_dist"], gt=z["min_past_gt_dist"], q=z["query_idx"])


def slice_masks(M: dict, seqs: list) -> dict:
    """이름 -> (pos_mask|None, neg_mask|None). 앞선 스크립트의 사전 고정 정의."""
    q1, q2, _ = P.A2_QUART
    out = {}
    out["A2_Q1"] = ({s: (M[s]["gtd"] >= 0) & (M[s]["gtd"] <= q1) for s in seqs}, None)
    out["A2_Q2"] = ({s: (M[s]["gtd"] > q1) & (M[s]["gtd"] <= q2) for s in seqs}, None)
    out["A3_small"] = ({s: M[s]["yvalid"] & M[s]["positive"] & (M[s]["lateral"] <= P.A3_LAT) for s in seqs}, None)
    turn = {s: M[s]["rot"] >= np.quantile(M[s]["rot"], 0.8) for s in seqs}
    rest = {s: ~turn[s] for s in seqs}
    out["A6a_turn"] = (turn, turn)
    out["A6a_rest"] = (rest, rest)
    return out


def check_masks_against_prior(M: dict, sm: dict) -> None:
    cond = {n: a for n, _, a in P.conditions(M)}
    for s in C.SEQS:
        np.testing.assert_array_equal(sm["A2_Q1"][0][s], cond["A2_gtdist_quartile"]["masks"]["Q1"][s])
        np.testing.assert_array_equal(sm["A2_Q2"][0][s], cond["A2_gtdist_quartile"]["masks"]["Q2"][s])
        np.testing.assert_array_equal(sm["A3_small"][0][s], cond["A3_lateral_median"]["mC"][0][s])
        np.testing.assert_array_equal(sm["A6a_turn"][0][s], cond["A6a_rotation_top20"]["mT"][0][s])
        np.testing.assert_array_equal(sm["A6a_rest"][0][s], cond["A6a_rotation_top20"]["mC"][0][s])
    assert cond["A2_gtdist_quartile"]["neg_restrict"] is False and cond["A3_lateral_median"]["mC"][1] is None


def r3(x):
    return round(float(x), 3)


def detail(rows: dict, groups: dict) -> dict:
    """도시별·시퀀스별·LOCO·LOSO·시드별."""
    A, B = rows["nsd"], rows["abs"]
    seq_d, city_d = {}, {}
    for c, seqs in groups.items():
        av = [s for s in seqs if A[0].get(s) is not None]
        if not av:
            continue
        for s in av:
            seq_d[s] = K.seq_delta(A, B, s)
        city_d[c] = np.mean([seq_d[s] for s in av], 0)
    cities = list(city_d)
    macro = np.mean([city_d[c] for c in cities], 0)
    loco = {c: np.mean([city_d[x] for x in cities if x != c], 0) for c in cities}
    loso = {}
    for s in seq_d:
        cd = {}
        for c, seqs in groups.items():
            av = [x for x in seqs if x in seq_d and x != s]
            if av:
                cd[c] = np.mean([seq_d[x] for x in av], 0)
        loso[s] = np.mean(list(cd.values()), 0)
    ap = {s: v[0] for s, v in seq_d.items()}
    order = sorted(ap, key=ap.get)
    npos = {s: A[0][s]["n_pos"] for s in seq_d}
    nneg = {s: A[0][s]["n_neg"] for s in seq_d}
    per_seed = [100 * (K.macro(a, groups)["AP"] - K.macro(b, groups)["AP"]) for a, b in zip(A, B)]
    per_seed_all = [{k: 100 * (K.macro(a, groups)[k] - K.macro(b, groups)[k]) for k in KEYS} for a, b in zip(A, B)]
    loso_ap = {s: v[0] for s, v in loso.items()}
    return dict(
        macro_pp=dict(zip(KEYS, map(r3, macro))),
        city_pp={c: dict(zip(KEYS, map(r3, v))) for c, v in city_d.items()},
        city_n_seq={c: len([s for s in groups[c] if s in seq_d]) for c in cities},
        n_city_positive={k: int(sum(city_d[c][i] > 0 for c in cities)) for i, k in enumerate(KEYS)},
        n_city=len(cities),
        loco_pp={c: dict(zip(KEYS, map(r3, v))) for c, v in loco.items()},
        loco_AP_range=[r3(min(v[0] for v in loco.values())), r3(max(v[0] for v in loco.values()))],
        seq_AP_pp={s: r3(ap[s]) for s in order},
        seq_n_pos={s: int(npos[s]) for s in order},
        seq_n_neg={s: int(nneg[s]) for s in order},
        n_seq=len(seq_d), n_seq_AP_positive=int(sum(v > 0 for v in ap.values())),
        seq_AP_median=r3(np.median(list(ap.values()))),
        top3_AP=[(s, r3(ap[s]), int(npos[s])) for s in order[::-1][:3]],
        bottom3_AP=[(s, r3(ap[s]), int(npos[s])) for s in order[:3]],
        loso_AP_range=[r3(min(loso_ap.values())), r3(max(loso_ap.values()))],
        loso_AP_min_by=min(loso_ap, key=loso_ap.get), loso_AP_max_by=max(loso_ap, key=loso_ap.get),
        per_seed_AP_pp=[r3(x) for x in per_seed],
        per_seed_pp=[{k: r3(v) for k, v in d.items()} for d in per_seed_all],
    )


def detail_D(rT: dict, rC: dict, groups: dict) -> dict:
    """D = (NSD−abs)_T − (NSD−abs)_C의 도시별 값·LOCO·LOSO(AP 중심)."""
    def city_vals(drop=None):
        out = {}
        for c, seqs in groups.items():
            t = [K.seq_delta(rT["nsd"], rT["abs"], s) for s in seqs if s != drop and rT["nsd"][0].get(s) is not None]
            u = [K.seq_delta(rC["nsd"], rC["abs"], s) for s in seqs if s != drop and rC["nsd"][0].get(s) is not None]
            if t or u:
                out[c] = (np.mean(t, 0) if t else np.full(len(KEYS), np.nan)) - (np.mean(u, 0) if u else np.full(len(KEYS), np.nan))
        return out
    cd = city_vals()
    cities = [c for c in cd if not np.isnan(cd[c][0])]
    loco = {c: np.mean([cd[x] for x in cities if x != c], 0) for c in cities}
    seqs = [s for v in groups.values() for s in v if rT["nsd"][0].get(s) is not None or rC["nsd"][0].get(s) is not None]
    loso = {}
    for s in seqs:
        v = city_vals(s)
        loso[s] = float(np.nanmean([x[0] for x in v.values()]))
    return dict(macro_pp=dict(zip(KEYS, map(r3, np.nanmean([cd[c] for c in cities], 0)))),
                city_pp={c: dict(zip(KEYS, map(r3, cd[c]))) for c in cities},
                n_city_positive={k: int(sum(cd[c][i] > 0 for c in cities)) for i, k in enumerate(KEYS)}, n_city=len(cities),
                loco_AP={c: r3(v[0]) for c, v in loco.items()},
                loco_AP_range=[r3(min(v[0] for v in loco.values())), r3(max(v[0] for v in loco.values()))],
                loso_AP_range=[r3(min(loso.values())), r3(max(loso.values()))],
                loso_AP_min_by=min(loso, key=loso.get), loso_AP_max_by=max(loso, key=loso.get),
                per_seed_AP_pp=[r3(100 * ((K.macro(a, groups)["AP"] - K.macro(b, groups)["AP"]) - (K.macro(x, groups)["AP"] - K.macro(y, groups)["AP"])))
                                for a, b, x, y in zip(rT["nsd"], rT["abs"], rC["nsd"], rC["abs"])])


def group_report(data: dict, seqs: list, groups: dict, sm: dict, full_detail=True) -> dict:
    out = {}
    full = K.rows_for(data, seqs)
    rep = P.strip(K.subset_report(full, groups))
    out["all"] = dict(report=rep, detail=detail(full, groups) if full_detail else None)
    rows = {}
    for name in SLICES + ("A6a_rest",):
        pm, nm = sm[name]
        rows[name] = K.rows_for(data, seqs, pm, nm)
        out[name] = dict(report=P.strip(K.subset_report(rows[name], groups)),
                         detail=detail(rows[name], groups) if full_detail else None)
    d = K.boot_joint(rows["A6a_turn"]["nsd"], rows["A6a_turn"]["abs"], rows["A6a_rest"]["nsd"], rows["A6a_rest"]["abs"],
                     groups, np.random.default_rng(20261117))
    t, r = out["A6a_turn"]["report"]["nsd_minus_abs_pp"], out["A6a_rest"]["report"]["nsd_minus_abs_pp"]
    out["A6a_D_turn_minus_rest"] = dict(D_pp={k: t[k] - r[k] for k in KEYS}, D_ci95_pp=K.ci(d),
                                        detail=detail_D(rows["A6a_turn"], rows["A6a_rest"], groups) if full_detail else None)
    return out


# ------------------------------------------------------------------ 검증 메타(앞선 seq_meta를 검증 경로로)
def val_meta(vseqs: list) -> dict:
    (SHIM / "search/ledgers/test").mkdir(parents=True, exist_ok=True)
    (SHIM / "old/test").mkdir(parents=True, exist_ok=True)
    for s in vseqs:
        n = s.replace("/", "_")
        for link, target in ((SHIM / "search/ledgers/test" / n, C.SEARCH / "ledgers/val" / n), (SHIM / "old/test" / n, VAL_AUDIT / n)):
            if not link.is_symlink():
                os.symlink(target, link)
    old = (C.SEARCH, C.OLD)
    try:
        C.SEARCH, C.OLD = SHIM / "search", SHIM / "old"
        M = {s: P.seq_meta(s) for s in vseqs}
    finally:
        C.SEARCH, C.OLD = old
    return M


def remove_shim() -> None:
    for sub in ("search/ledgers/test", "old/test"):
        d = SHIM / sub
        if d.exists():
            for p in d.iterdir():
                assert p.is_symlink()
                p.unlink()
    for sub in ("search/ledgers/test", "search/ledgers", "search", "old/test", "old", ""):
        d = SHIM / sub
        if d.exists():
            d.rmdir()


# ------------------------------------------------------------------ A6a 공변량(점수 무관)
def covariates(M: dict, seqs: list, groups: dict, sm: dict) -> dict:
    turn = sm["A6a_turn"][0]
    out = {}
    for name, m in (("turn", turn), ("rest", {s: ~turn[s] for s in seqs})):
        pos = {s: M[s]["positive"] & m[s] for s in seqs}
        neg = {s: (M[s]["gtd"] > 25) & m[s] for s in seqs}
        yv = {s: pos[s] & M[s]["yvalid"] for s in seqs}
        cats = {"forward_0_30": lambda s: yv[s] & (M[s]["yawdiff"] < 30),
                "cross_30_150": lambda s: yv[s] & (M[s]["yawdiff"] >= 30) & (M[s]["yawdiff"] <= 150),
                "reverse_150_180": lambda s: yv[s] & (M[s]["yawdiff"] > 150),
                "yaw_invalid": lambda s: pos[s] & ~M[s]["yvalid"]}
        npos = int(sum(pos[s].sum() for s in seqs))
        r = dict(n_queries=int(sum(m[s].sum() for s in seqs)), n_pos=npos, n_neg=int(sum(neg[s].sum() for s in seqs)))
        r["pos_rate_pooled"] = round(npos / max(r["n_queries"], 1), 4)
        r["dir_counts_pos"] = {c: int(sum(f(s).sum() for s in seqs)) for c, f in cats.items()}
        r["dir_frac_pos_pooled"] = {c: round(v / max(npos, 1), 4) for c, v in r["dir_counts_pos"].items()}

        # 매크로(시퀀스 비율 → 도시 평균 → 도시 동일 가중), 양성 있는 시퀀스만
        def mac(fn):
            cv = []
            for c, ss in groups.items():
                v = [fn(s) for s in ss if pos[s].any()]
                if v:
                    cv.append(np.nanmean(v))
            return round(float(np.nanmean(cv)), 4)
        r["dir_frac_pos_macro"] = {c: mac(lambda s, f=f: f(s).sum() / pos[s].sum()) for c, f in cats.items()}
        gtd = np.concatenate([M[s]["gtd"][pos[s]] for s in seqs])
        lat = np.concatenate([M[s]["lateral"][yv[s]] for s in seqs])
        r["gtd_pos_quantiles_10_25_50_75_90"] = P.q_(gtd, [.1, .25, .5, .75, .9])
        r["lateral_pos_quantiles_10_25_50_75_90"] = P.q_(lat, [.1, .25, .5, .75, .9])
        q1, q2, q3 = P.A2_QUART
        r["gtd_pos_frac_Q1_Q2_Q3_Q4_pooled"] = [round(float(np.mean(gtd <= q1)), 4), round(float(np.mean((gtd > q1) & (gtd <= q2))), 4),
                                               round(float(np.mean((gtd > q2) & (gtd <= q3))), 4), round(float(np.mean(gtd > q3)), 4)]
        r["gtd_pos_frac_Q1_macro"] = mac(lambda s: np.mean(M[s]["gtd"][pos[s]] <= q1))
        r["gtd_pos_frac_le_Q2_macro"] = mac(lambda s: np.mean(M[s]["gtd"][pos[s]] <= q2))
        r["lateral_small_frac_pooled"] = round(float(np.mean(lat <= P.A3_LAT)), 4)
        r["lateral_small_frac_macro"] = mac(lambda s: np.mean(M[s]["lateral"][yv[s]] <= P.A3_LAT) if yv[s].any() else np.nan)
        r["rot_deg_quantiles_10_50_90"] = P.q_(np.concatenate([M[s]["rot"][m[s]] for s in seqs]), [.1, .5, .9])
        r["pos_share_by_city"] = {c: round(int(sum(pos[s].sum() for s in ss)) / max(npos, 1), 4) for c, ss in groups.items()}
        # 기존 방향 표지(group 0 forward_only, 1 reverse_only, 2 mixed; pdv2_review_direction_yaw_20261105)
        dg = {}
        for s in (seqs if all((C.DIRS / (x.replace("/", "_") + ".npz")).exists() for x in seqs) else []):
            with np.load(C.DIRS / (s.replace("/", "_") + ".npz")) as z:
                np.testing.assert_array_equal(z["query_idx"], M[s]["q"])
                g = z["group"][pos[s]]
            for code, nm in ((0, "forward_only"), (1, "reverse_only"), (2, "mixed")):
                dg[nm] = dg.get(nm, 0) + int((g == code).sum())
        r["review_direction_group_pos"] = dg or "검증 시퀀스에는 방향 표지 파일이 없음"
        out[name] = r
    return out


def main() -> None:
    tick = time.time()
    M = P.load_meta()
    S = C.SEQS
    sm = slice_masks(M, S)
    check_masks_against_prior(M, sm)
    out = dict(conventions=__doc__.split("====================================================================")[1].strip())

    # ---------------------------------------------------------------- 시험46 (원고 체크포인트)
    d456 = {"nsd": [{s: C.load(f"new:{k}", s) for s in S} for k in C.NEW_SEEDS],
            "abs": [{s: K.load(f"abskv@s{k}@u512", s, "test") for s in S} for k in C.NEW_SEEDS]}
    d123 = {"nsd": [{s: P.load_sweep1(f"sym@s{k}@u512", s) for s in S} for k in (1, 2, 3)],
            "abs": [{s: P.load_sweep1(f"abskv@s{k}@u512", s) for s in S} for k in (1, 2, 3)]}
    aux = {v: [{s: ld(FINAL / "test" / s.replace("/", "_") / f"{v}@s{k}@u512.npz") for s in S} for k in C.NEW_SEEDS]
           for v in ("sym", "abskv", "diffk_absv", "sum_kv")}
    for s in S:
        for ds in list(d456.values()) + list(d123.values()) + list(aux.values()):
            for d in ds:
                np.testing.assert_array_equal(d[s]["q"], M[s]["q"])
                np.testing.assert_array_equal(d[s]["gt"], M[s]["gtd"])
    res = {}
    res["test_s456"] = group_report(d456, S, C.CITY, sm)
    res["test_s123"] = group_report(d123, S, C.CITY, sm)
    res["aux_s456_sym_minus_abskv"] = group_report({"nsd": aux["sym"], "abs": aux["abskv"]}, S, C.CITY, sm)
    for v in ("diffk_absv", "sum_kv"):
        res[f"aux_s456_{v}_minus_abskv"] = group_report({"nsd": aux[v], "abs": aux["abskv"]}, S, C.CITY, sm, full_detail=False)

    # 검산: 앞선 요약과 일치
    prior = json.loads((P.OUT / "summary.json").read_text())
    ref = {"A2_Q1": ("A2_gtdist_quartile", "bins", "Q1"), "A2_Q2": ("A2_gtdist_quartile", "bins", "Q2"),
           "A3_small": ("A3_lateral_median", "small"), "A6a_turn": ("A6a_rotation_top20", "turn"),
           "A6a_rest": ("A6a_rotation_top20", "rest")}
    for g in ("s456", "s123"):
        for name, path in ref.items():
            r = prior[g]
            for p in path:
                r = r[p]
            mine = res[f"test_{g}"][name]["report"]
            for k in KEYS:
                assert abs(r["nsd_minus_abs_pp"][k] - mine["nsd_minus_abs_pp"][k]) < 1e-9, (g, name, k)
                assert np.allclose(r["ci95_pp"][k], mine["ci95_pp"][k], atol=1e-9), (g, name, k)
                assert abs(res[f"test_{g}"][name]["detail"]["macro_pp"][k] - mine["nsd_minus_abs_pp"][k]) < 2e-3
        pd = prior[g]["A6a_rotation_top20"]
        for k in KEYS:
            assert abs(pd["D_pp"][k] - res[f"test_{g}"]["A6a_D_turn_minus_rest"]["D_pp"][k]) < 1e-9
            assert np.allclose(pd["D_ci95_pp"][k], res[f"test_{g}"]["A6a_D_turn_minus_rest"]["D_ci95_pp"][k], atol=1e-9)
    fin = json.loads((FINAL.parent / "summary.json").read_text())["s456"]["test"]["all"]
    for a, arm in (("nsd", "sym"), ("abs", "abskv")):
        for k in KEYS:
            assert abs(res["aux_s456_sym_minus_abskv"]["all"]["report"]["mean_pp"][a][k] - 100 * fin["arms"][arm][k]["mean"]) < 1e-9
    fd = fin["differences"]["sym-abskv"]
    for k in KEYS:
        assert np.allclose(fd["conditional_95_pp"][k], res["aux_s456_sym_minus_abskv"]["all"]["report"]["ci95_pp"][k], atol=1e-9)

    # ---------------------------------------------------------------- 검증9
    vdir = C.SEARCH / "ledgers" / "val"
    vseqs = sorted(p.name.replace("_", "/", 1) for p in vdir.iterdir() if p.is_dir())
    vfam = {f: [s for s in vseqs if ("kitti" if s.startswith("kitti") else s.split("/")[0]) == f] for f in K.VAL_FAMILY_ORDER}
    try:
        VM = val_meta(vseqs)
    finally:
        remove_shim()
    vsm = slice_masks(VM, vseqs)
    v456 = {"nsd": [{s: K.load(f"new:{k}", s, "val") for s in vseqs} for k in C.NEW_SEEDS],
            "abs": [{s: K.load(f"abskv@s{k}@u512", s, "val") for s in vseqs} for k in C.NEW_SEEDS]}
    v123 = {a: [{s: ld(SWEEP1_VAL / s.replace("/", "_") / f"{t}@s{k}@u512.npz") for s in vseqs} for k in (1, 2, 3)]
            for a, t in (("nsd", "sym"), ("abs", "abskv"))}
    for s in vseqs:
        for ds in list(v456.values()) + list(v123.values()):
            for d in ds:
                np.testing.assert_array_equal(d[s]["q"], VM[s]["q"])
                np.testing.assert_array_equal(d[s]["gt"], VM[s]["gtd"])
    res["val_s456"] = group_report(v456, vseqs, vfam, vsm)
    res["val_s123"] = group_report(v123, vseqs, vfam, vsm)
    h4 = json.loads((K.OUT / "summary.json").read_text())["H4"]
    for k in KEYS:
        assert abs(h4["nsd_minus_abs_pp"][k] - res["val_s456"]["all"]["report"]["nsd_minus_abs_pp"][k]) < 1e-9
        assert np.allclose(h4["ci95_pp"][k], res["val_s456"]["all"]["report"]["ci95_pp"][k], atol=1e-9)
    sw = json.loads(P.SWEEP1_SUMMARY.read_text())["val"]
    for a, arm in (("nsd", "sym"), ("abs", "abskv")):
        for k in KEYS:
            assert abs(res["val_s123"]["all"]["report"]["mean_pp"][a][k] - 100 * sw[arm][k]["mean"]) < 1e-6, (a, k)
    res["val_meta"] = dict(
        families=vfam,
        n_pos={s: int(VM[s]["positive"].sum()) for s in vseqs},
        n_neg={s: int((VM[s]["gtd"] > 25).sum()) for s in vseqs},
        n_A2_Q1_pos={s: int((vsm["A2_Q1"][0][s] & VM[s]["positive"]).sum()) for s in vseqs},
        n_A2_Q2_pos={s: int((vsm["A2_Q2"][0][s] & VM[s]["positive"]).sum()) for s in vseqs},
        n_A3_small_pos={s: int(vsm["A3_small"][0][s].sum()) for s in vseqs},
        rot_deg_seq_p80={s: round(float(np.quantile(VM[s]["rot"], .8)), 2) for s in vseqs},
        gtd_pos_pool_quantiles_25_50_75=P.q_(np.concatenate([VM[s]["gtd"][VM[s]["positive"]] for s in vseqs]), [.25, .5, .75]),
        lateral_pos_pool_median=P.q_(np.concatenate([VM[s]["lateral"][VM[s]["positive"] & VM[s]["yvalid"]] for s in vseqs]), [.5]),
        note="경계는 시험 풀 절대값 그대로. 검증 풀 분위는 참고로만 적음.")

    # ---------------------------------------------------------------- 시드 6개 모음
    seeds = {}
    for name in ("all",) + SLICES:
        v = res["test_s123"][name]["detail"]["per_seed_AP_pp"] + res["test_s456"][name]["detail"]["per_seed_AP_pp"]
        seeds[name] = dict(seeds=[1, 2, 3, 4, 5, 6], AP_pp=v, n_positive=int(sum(x > 0 for x in v)),
                           mean=r3(np.mean(v)), sd_ddof1=r3(np.std(v, ddof=1)))
    v = [1, 2, 3, 4, 5, 6]
    for g in ("val_s123", "val_s456"):
        for name in ("all",) + SLICES:
            seeds.setdefault("val:" + name, dict(seeds=v, AP_pp=[]))["AP_pp"] += res[g][name]["detail"]["per_seed_AP_pp"]
    for name in ("all",) + SLICES:
        x = seeds["val:" + name]["AP_pp"]
        seeds["val:" + name].update(n_positive=int(sum(t > 0 for t in x)), mean=r3(np.mean(x)), sd_ddof1=r3(np.std(x, ddof=1)))
    a = res["aux_s456_sym_minus_abskv"]
    for name in ("all",) + SLICES:
        x = a[name]["detail"]["per_seed_AP_pp"]
        seeds["aux:" + name] = dict(seeds=[4, 5, 6], AP_pp=x, n_positive=int(sum(t > 0 for t in x)), mean=r3(np.mean(x)), sd_ddof1=r3(np.std(x, ddof=1)))
    x = res["test_s123"]["A6a_D_turn_minus_rest"]["detail"]["per_seed_AP_pp"] + res["test_s456"]["A6a_D_turn_minus_rest"]["detail"]["per_seed_AP_pp"]
    seeds["A6a_D_turn_minus_rest"] = dict(seeds=[1, 2, 3, 4, 5, 6], AP_pp=x, n_positive=int(sum(t > 0 for t in x)), mean=r3(np.mean(x)), sd_ddof1=r3(np.std(x, ddof=1)))
    res["per_seed"] = seeds
    res["A6a_covariates_test"] = covariates(M, S, C.CITY, sm)
    res["A6a_covariates_val"] = covariates(VM, vseqs, vfam, vsm)
    res["check"] = dict(test_masks_equal_prior_conditions=True, test_slices_equal_prior_summary_point_and_ci=True,
                        aux_all_matches_final_s456_summary=True, val_s456_all_matches_H4=True, val_s123_all_matches_sweep1_summary=True,
                        queries_and_labels_identical_all_ledgers=True)
    out["results"] = res
    out["seconds"] = round(time.time() - tick, 1)
    C.write(OUT / "summary.json", out)
    brief(res)


def brief(res: dict) -> None:
    def f(r):
        return f"{r['nsd_minus_abs_pp']['AP']:+.2f} [{r['ci95_pp']['AP'][0]:+.2f}, {r['ci95_pp']['AP'][1]:+.2f}]"
    for g in ("test_s456", "test_s123", "val_s456", "val_s123", "aux_s456_sym_minus_abskv", "aux_s456_diffk_absv_minus_abskv", "aux_s456_sum_kv_minus_abskv"):
        r = res[g]
        print(g, "all", f(r["all"]["report"]), *(f"| {n} {f(r[n]['report'])} npos={r[n]['report']['n_pos']}" for n in SLICES),
              f"| D {r['A6a_D_turn_minus_rest']['D_pp']['AP']:+.2f} [{r['A6a_D_turn_minus_rest']['D_ci95_pp']['AP'][0]:+.2f}, {r['A6a_D_turn_minus_rest']['D_ci95_pp']['AP'][1]:+.2f}]")


if __name__ == "__main__":
    main()
