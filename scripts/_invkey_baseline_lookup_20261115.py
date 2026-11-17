#!/usr/bin/env python3
"""invkey 진단: 같은 시퀀스에서 원고 표2 비교 방법 10종·현재 NSD의 기존 원장을 invkey 규약으로 다시 센다 (2026-11-15).

새 추론·학습 없음. 이미 있는 원장(질의별 1위 거리·점수)만 읽는다. 호스트 python3(numpy, scipy)로 돈다.

규약 두 가지를 같은 원장에서 센다.
  inv  invkey 규약: 질의 q >= 30 (끝까지), 후보 j <= q-30, 양성 d_min <= 5 m, 음성 > 25 m, 5-25 m 제외,
       R@1 = 양성 중 1위 <= 5 m, AP·최대 F1·R@100P = scripts/_pr_eval_lib.py pr_sweep(1위 점수).
       (scripts/_invkey_train_linear_20261114.py eval_seq와 같다)
  ms   원고 공통 질의: 위와 같고 질의만 30 <= q < N-15 (scripts/_pdv2_main_baselines_official.py aggregate).
방향(같은/반대/교차/미정): 양성 질의 q와 과거 최근접 GT 키프레임 jnn(j <= q-30 중 최소 거리)의 진행 방향
  (GT 위치 P[i+3]-P[i-3], 1 m 미만 이동은 미정; 30도 미만 같은, 150도 초과 반대, 나머지 교차).
  scripts/_invkey_data_survey_20261114.py headings·classify를 그대로 불러 쓴다(_invkey_submap_probe_20261115.py와 같음).
GT 위치: 원장이 쓴 키프레임 캐시 data/preprocessed_v2_odom/cache_*의 poses_gt 평행 이동(float64 직접 차).

원장
  시험(원고 시험 46 중 대상 16): 표2 최종 요약
    results/pdv2_deepring_scaleinit_integration_20261110/baseline_seeds_summary_seqot_stable_ringpp_i2_fresco_kgf_deepring_scaleinit.json
    의 methods[*].per_seed[*].source 폴더. NSD는 results/pdv2_diffattn_contrast_test_20261104/test/<seq>/diff_s{1,2,3}.npz.
  검증(kitti/09, kitti360/0004·0006): 표2와 같은 체크포인트·구현의 검증 원장이 있는 것만(VAL_LEDGERS, VAL_JSON).
검산
  (1) 시험 원장 전부(표2 22실행 x 46, NSD 3 x 46)를 ms 규약으로 다시 모아 6도시 매크로가 표2 요약·NSD summary와 같은지.
  (2) 대상 시퀀스 원장의 min_past_gt_dist가 캐시 GT로 다시 잰 값과 같은지, 질의 번호가 30..N-1(또는 N-16)인지.
  (3) inv 규약의 양성·음성·제외 수와 방향 수가 invkey 결과(scope3/metrics.json raw)와 같은지,
      양성 질의의 jnn·방향이 fov_survey/recall.json의 pos_jnn·pos_dir와 같은지.
  (4) invkey 묶음 평균 정의(R@1은 양성 합산, AP 등은 시퀀스 평균, 시드 평균·최소·최대)가
      scope3/stepctl/metrics.json table을 재현하는지.

실행 (리포 루트, 호스트):  python3 scripts/_invkey_baseline_lookup_20261115.py
출력: results/invkey_study_20261114/baseline_lookup/lookup.json, tables.md
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _invkey_data_survey_20261114 import DIRS, classify, headings  # noqa: E402
from _pdv2_pilot import CITY, TEST  # noqa: E402
from _pr_eval_lib import classify_queries, pr_sweep  # noqa: E402

OUT = REPO / "results/invkey_study_20261114/baseline_lookup"
INV = REPO / "results/invkey_study_20261114"
T2 = REPO / ("results/pdv2_deepring_scaleinit_integration_20261110/"
             "baseline_seeds_summary_seqot_stable_ringpp_i2_fresco_kgf_deepring_scaleinit.json")
NSD = REPO / "results/pdv2_diffattn_contrast_test_20261104"
NSD_VAL = REPO / "results/pdv2_diffattn_contrast_pipeline_20261104/val"
CACHE = REPO / "data/preprocessed_v2_odom"
METHODS = ("scpp", "fresco", "ring", "ringpp", "disco", "deepring", "ringsharp", "helios", "seqot", "bevplace")
PREFIX = {"scpp": "dump_scpp", "fresco": "dump_fresco", "ring": "dump_ring", "ringpp": "dump_ringpp_i2",
          "disco": "dump_disco", "deepring": "dump_deepring", "ringsharp": "dump_ringsharp",
          "helios": "dump_helios", "seqot": "dump_seqotl2", "bevplace": "dump_bevplace"}
M4 = ("AP", "max_F1", "R@100P")

GROUPS = {
    "A_tuhh": [f"mcd/{s}" for s in ("tuhh_day_02", "tuhh_day_03", "tuhh_day_04",
                                    "tuhh_night_07", "tuhh_night_08", "tuhh_night_09")],
    "B_val": ["kitti/09", "kitti360/0004", "kitti360/0006"],
    "C_kitti_dev": ["kitti/00", "kitti/07", "kitti/08"],
    "C_kth": [f"mcd/{s}" for s in ("kth_day_06", "kth_day_09", "kth_day_10",
                                   "kth_night_01", "kth_night_04", "kth_night_05")],
    "C_k360_0000": ["kitti360/0000"],
}
TARGET = [s for g in GROUPS.values() for s in g]
VALSEQ = set(GROUPS["B_val"])

# 검증 원장: 표2와 같은 체크포인트(체크포인트 해시를 원장 meta에서 대조)·구현
_RET = "results/pdv2_review_selection_full_20261105/retained/val"
_I2V = "/mnt/d/NSD_local/results/pdv2_overnight_repro_20261108/ringpp_norm_validation_20261109/result/ledgers"
VAL_LEDGERS = {   # method -> [(seed, path template with {s} = sensor_seq, score)]
    "bevplace": [("1", "results/pdv2_bevplace_val_odom_20261018/dump_bevplace_{s}.npz", "score_negl2"),
                 ("2", "results/pdv2_bevplace_val_odom_seeds_20261101/dump_bevplace_s2_{s}.npz", "score_negl2"),
                 ("3", "results/pdv2_bevplace_val_odom_seeds_20261101/dump_bevplace_s3_{s}.npz", "score_negl2")],
    "disco": [("1", f"{_RET}/disco/s1/e001_eca498df1bab/dump_{{s}}.npz", "score_disco"),
              ("2", f"{_RET}/disco/s2/e001_80c4b8bda050/dump_{{s}}.npz", "score_disco"),
              ("3", f"{_RET}/disco/s3/e001_1d7ace8e54a6/dump_{{s}}.npz", "score_disco")],
    "ringpp": [("-", f"{_I2V}/dump_ringpp_i2_{{s}}.npz", "score_corr")],
}
VAL_CKPT = {("bevplace", "1"): "b0dd3a16c329bd8fc847e4f44ad5b848d98e821772cd6e39097f41940f90a96f",
            ("bevplace", "2"): "c08180f99b78c7596185d91bfe1b6e0e024c421acce793cb4224583d0ac015fa",
            ("bevplace", "3"): "e55452024826ac67c22ce73845e96aca3ca043b0248d604aec20d9a683d5b80f",
            ("disco", "1"): "80cdb08811707462205da226a594daaed9de52b7b1e30ec78ffe9ecfabd66608",
            ("disco", "2"): "6bd92cfa8d2359c2f9ef41488f624a2c8fd2a485f5387e8cf1b9386259bc0d9a",
            ("disco", "3"): "3ebeb577c429aa1305db63ce3653d891a6a6ae7a7af5ff69287fff81c198e3f2"}
# 검증 요약만 있는 것(원장 npz가 디스크에 없음): HeLiOS 표2 체크포인트 s1 e010, s2 e008, s3 e010
VAL_JSON = {"helios": [("1", f"{_RET}/helios/s1/e010_5ec26f621a9c/{{s}}.json"),
                       ("2", f"{_RET}/helios/s2/e008_d7a45d80d7c9/{{s}}.json"),
                       ("3", f"{_RET}/helios/s3/e010_dee1c31ff027/{{s}}.json")]}
HELIOS_CKPT = {"1": "3a6424dbd89f835ae708bb819dc804c0a274ad1ce56c58204df676fa0a188258",
               "2": "ac779d5d434fde3d0d8738d164447963569e1a2e84b9ec067963a5c74ce91e29",
               "3": "315de47cc6139f145aebffc0daed8e63c77793c03a10c4cd3f61256a3395f2ed"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rpath(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def fs(key):
    return key.replace("/", "_")


# ---------------------------------------------------------------- GT
_GT = {}


def gt(key):
    """캐시 GT 위치, 질의 q >= 30의 과거 최근접 거리·색인, 진행 방향."""
    if key in _GT:
        return _GT[key]
    sensor, seq = key.split("/")
    role = "val" if key in VALSEQ else "test"
    (path,) = CACHE.glob(f"cache_*_{sensor}_{role}_{seq}.npz")
    with np.load(path) as z:
        P = np.asarray(z["poses_gt"], dtype=np.float64)[:, :3, 3]
    n = len(P)
    dmin = np.full(n, np.nan)
    jnn = np.full(n, -1)
    for q in range(30, n):
        d = np.linalg.norm(P[: q - 29] - P[q], axis=1)
        jnn[q] = int(np.argmin(d))
        dmin[q] = d[jnn[q]]
    u, moving = headings(P)
    _GT[key] = {"P": P, "n": n, "dmin": dmin, "jnn": jnn, "u": u, "moving": moving,
                "cache": str(path.relative_to(REPO)), "cache_sha256": sha256(path)}
    return _GT[key]


def seq_metrics(q, score, top1, dmin_ledger, key, check_gt=True):
    """원장 한 개(질의 q 배열)의 지표. q는 이미 규약으로 거른 것."""
    cls = classify_queries(dmin_ledger)
    pos = cls["positive"]
    hit = top1 <= 5.0
    m = {"n_queries": int(len(q)), "n_pos": cls["n_pos"], "n_neg": cls["n_neg"],
         "n_ambiguous": cls["n_ambiguous"], "n_hit1": int(hit[pos].sum())}
    if cls["n_pos"]:
        sw = pr_sweep(score, top1, cls)
        m.update({k: float(sw[k]) for k in M4})
        m["R@1"] = float(hit[pos].mean())
    else:
        m.update({k: None for k in (*M4, "R@1")})
    if check_gt:
        g = gt(key)
        qp = q[pos]
        dirs = classify(g["u"], g["moving"], qp, g["jnn"][qp])
        m["dir_n"] = {d: int((dirs == i).sum()) for i, d in enumerate(DIRS)}
        m["dir_hit1"] = {d: int(hit[pos][dirs == i].sum()) for i, d in enumerate(DIRS)}
    return m


def load_ledger(path, score_name, key):
    """원장을 읽고 질의 번호·과거 최근접 거리를 캐시 GT와 대조한다. 반환: q, score, top1, dmin, 질의 범위 표시."""
    with np.load(path) as z:
        q = np.asarray(z["query_idx"], dtype=np.int64)
        s = np.asarray(z[score_name], dtype=np.float64)
        t = np.asarray(z["top1_geo_dist"], dtype=np.float64)
        d = np.asarray(z["min_past_gt_dist"], dtype=np.float64)
        js = np.asarray(z["j_star"], dtype=np.int64)
    g = gt(key)
    n = g["n"]
    if np.array_equal(q, np.arange(30, n)):
        span = "q>=30 (all)"
    elif np.array_equal(q, np.arange(30, n - 15)):
        span = "30<=q<N-15"
    else:
        raise SystemExit(f"STOP: unexpected query set {path}")
    assert np.all(js <= q - 30), path
    dd = float(np.max(np.abs(d - g["dmin"][q])))
    assert dd < 1e-6, (path, dd)
    return q, s, t, d, span, dd


def both_rules(path, score_name, key):
    q, s, t, d, span, dd = load_ledger(path, score_name, key)
    n = gt(key)["n"]
    out = {"ledger": str(Path(path).relative_to(REPO)) if str(path).startswith(str(REPO)) else str(path),
           "query_span": span, "dmin_max_abs_diff_vs_cache": dd}
    ms = q < n - 15
    out["ms"] = seq_metrics(q[ms], s[ms], t[ms], d[ms], key)
    out["inv"] = seq_metrics(q, s, t, d, key) if span == "q>=30 (all)" else None
    return out


# ---------------------------------------------------------------- 검산 (1): 표2 재현
def macro_ms(per):
    groups = {c: {k: float(np.mean([per[s][k] for s in names if per[s]["n_pos"]]))
                  for k in (*M4, "R@1")} for c, names in CITY.items()}
    return {k: float(np.mean([g[k] for g in groups.values()])) for k in (*M4, "R@1")}


def check_table2(t2, nsd_summary):
    res = {}
    for m in METHODS:
        e = t2["methods"][m]
        for seed, ps in e["per_seed"].items():
            per = {}
            for sensor, seq in TEST:
                p = REPO / ps["source"] / f"{PREFIX[m]}_{sensor}_{seq}.npz"
                with np.load(p) as z:
                    q, n = z["query_idx"], len(z["query_idx"]) + 30
                    ms = q < n - 15
                    assert ms.sum() == len(q) - 15, p
                    cls = classify_queries(z["min_past_gt_dist"][ms])
                    r = {"n_pos": cls["n_pos"], "n_q": int(ms.sum())}
                    if cls["n_pos"]:
                        sw = pr_sweep(z[e["score"]][ms], z["top1_geo_dist"][ms], cls)
                        r.update({k: float(sw[k]) for k in M4})
                        r["R@1"] = float((z["top1_geo_dist"][ms][cls["positive"]] <= 5).mean())
                per[f"{sensor}/{seq}"] = r
            mac = macro_ms(per)
            diff = max(abs(mac[k] - ps["macro"][k]) for k in (*M4, "R@1"))
            res[f"{m}/{seed}"] = {"max_abs_diff_macro": diff, "n_queries": sum(r["n_q"] for r in per.values()),
                                  "n_pos": sum(r["n_pos"] for r in per.values())}
            assert diff < 1e-12 and res[f"{m}/{seed}"]["n_queries"] == 195726, (m, seed, diff)
    for i in (1, 2, 3):
        per = {}
        for sensor, seq in TEST:
            with np.load(NSD / f"test/{sensor}_{seq}/diff_s{i}.npz") as z:
                cls = classify_queries(z["min_past_gt_dist"])
                r = {"n_pos": cls["n_pos"], "n_q": len(z["query_idx"])}
                if cls["n_pos"]:
                    sw = pr_sweep(z["score"], z["top1_geo_dist"], cls)
                    r.update({k: float(sw[k]) for k in M4})
                    r["R@1"] = float((z["top1_geo_dist"][cls["positive"]] <= 5).mean())
            per[f"{sensor}/{seq}"] = r
        mac = macro_ms(per)
        ref = nsd_summary["per_case"][f"diff_s{i}"]["macro"]
        diff = max(abs(mac[k] - ref[k]) for k in (*M4, "R@1"))
        res[f"nsd/{i}"] = {"max_abs_diff_macro": diff, "n_queries": sum(r["n_q"] for r in per.values())}
        assert diff < 1e-12 and res[f"nsd/{i}"]["n_queries"] == 195726, (i, diff)
    return res


# ---------------------------------------------------------------- 묶음 평균
def group_stat(per_seq, keys, rule):
    """invkey 정의: R@1·방향 R@1 = 양성 합산, AP·최대 F1·R@100P = 양성 있는 시퀀스 평균."""
    rows = [per_seq[k][rule] if rule else per_seq[k] for k in keys if k in per_seq]
    rows = [r for r in rows if r is not None]
    if len(rows) != len(keys):
        return None
    npos = sum(r["n_pos"] for r in rows)
    out = {"n_seq": len(rows), "n_pos": npos,
           "R@1_pooled": (sum(r["n_hit1"] for r in rows) / npos) if npos else None}
    for k in M4:
        v = [r[k] for r in rows if r["n_pos"] and r.get(k) is not None]
        out[k] = float(np.mean(v)) if v else None
    if all("dir_n" in r for r in rows):
        for d in DIRS:
            nn = sum(r["dir_n"][d] for r in rows)
            out[f"n_{d}"] = nn
            out[f"R@1_{d}_pooled"] = sum(r["dir_hit1"][d] for r in rows) / nn if nn else None
    return out


def seed_spread(per_seed_groups):
    """{seed: group_stat} -> {지표: mean/min/max}. 시드 하나면 그 값."""
    vals = [v for v in per_seed_groups.values() if v is not None]
    if not vals:
        return None
    out = {"seeds": list(per_seed_groups)}
    for k in vals[0]:
        xs = [v[k] for v in vals]
        if any(x is None for x in xs) or not isinstance(xs[0], (int, float)):
            out[k] = None if any(x is None for x in xs) else xs[0]
            continue
        out[k] = {"mean": float(np.mean(xs)), "min": float(np.min(xs)), "max": float(np.max(xs))}
    return out


# ---------------------------------------------------------------- invkey 쪽
def invkey_rows():
    s3 = json.load(open(INV / "scope3/metrics.json"))["per_seq"]
    st = json.load(open(INV / "scope3/stepctl/metrics.json"))
    rc = json.load(open(INV / "fov_survey/recall.json"))
    rows = {"invkey_raw": {"-": {}}, "invkey_lin16_old": {"0": {}, "1": {}, "2": {}},
            "invkey_lin16_new3_st5000": {"0": {}, "1": {}, "2": {}}}
    for k in TARGET:
        if k in s3:
            rows["invkey_raw"]["-"][k] = {"inv": s3[k]["raw"]}
            for i in "012":
                rows["invkey_lin16_old"][i][k] = {"inv": s3[k][f"lin16_old_s{i}"]}
                rows["invkey_lin16_new3_st5000"][i][k] = {"inv": st["per_seq"][k][f"new3_st5000last_s{i}"]}
        else:   # kitti360/0000: 학습 자료라 scope3 평가 밖. fov_survey recall(raw, 기존 선형 16 seed 0)만 있다
            rows["invkey_raw"]["-"][k] = {"inv": {x: v for x, v in rc[k]["raw"].items() if not x.startswith("pos_")}}
            rows["invkey_lin16_old"]["0"][k] = {"inv": {x: v for x, v in rc[k]["lin16_s0"].items()
                                                        if not x.startswith("pos_")}}
    # ms 규약의 R@1·방향 R@1.
    #  (a) recall.json에 양성 질의별 기록이 있는 MCD(raw, lin16_s0): q < N-15만 다시 센다.
    #  (b) 그 밖: ms 양성 집합이 inv 양성 집합과 같으면(ms는 inv의 부분 집합이므로 개수가 같으면 같은 집합)
    #      R@1·방향 R@1은 inv 값과 같다. 다르면 셀 수 없어 비운다.
    for name, seeds in rows.items():
        for seed, per in seeds.items():
            for k, d in per.items():
                g = gt(k)
                n_ms = int(np.sum(g["dmin"][30:g["n"] - 15] <= 5.0))
                rk = {("invkey_raw", "-"): "raw", ("invkey_lin16_old", "0"): "lin16_s0"}.get((name, seed))
                if rk and "pos_q" in rc[k][rk]:
                    r = rc[k][rk]
                    assert r["n_map"] == g["n"], k
                    q, h, dr = np.array(r["pos_q"]), np.array(r["pos_hit1"], bool), np.array(r["pos_dir"])
                    keep = q < r["n_map"] - 15
                    assert int(keep.sum()) == n_ms, k
                    m = {"n_pos": n_ms, "n_hit1": int(h[keep].sum()),
                         "R@1": float(h[keep].mean()) if keep.any() else None,
                         "dir_n": {x: int((dr[keep] == i).sum()) for i, x in enumerate(DIRS)},
                         "dir_hit1": {x: int(h[keep][dr[keep] == i].sum()) for i, x in enumerate(DIRS)},
                         "source": "fov_survey/recall.json per-positive records, q<N-15"}
                elif d["inv"]["n_pos"] == n_ms:
                    a = d["inv"]
                    m = {"n_pos": n_ms, "n_hit1": a["n_hit1"], "R@1": a["R@1"], "dir_n": a["dir_n"],
                         "dir_hit1": a["dir_hit1"], "source": "identical positive set (inv n_pos == ms n_pos)"}
                else:
                    continue
                m.update({x: None for x in M4})
                d["ms_r1_only"] = m
    return rows, st, rc, s3


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t2 = json.load(open(T2))
    nsd_summary = json.load(open(NSD / "summary.json"))
    checks = {"table2_ms_reproduction": check_table2(t2, nsd_summary)}
    print("check (1) table2 / NSD macro reproduced:", len(checks["table2_ms_reproduction"]), "runs", flush=True)

    per = {}    # method -> seed -> seq -> {ms, inv, ...}
    runs = {}
    for m in METHODS:
        e = t2["methods"][m]
        per[m] = {}
        runs[m] = {"score": e["score"], "seeds": {}}
        for seed, ps in e["per_seed"].items():
            per[m][seed] = {}
            runs[m]["seeds"][seed] = {"test_source": ps["source"]}
            for k in TARGET:
                if k in VALSEQ:
                    continue
                per[m][seed][k] = both_rules(REPO / ps["source"] / f"{PREFIX[m]}_{fs(k)}.npz", e["score"], k)
    # 검증 원장
    for m, lst in VAL_LEDGERS.items():
        for seed, tmpl, sc in lst:
            for k in GROUPS["B_val"]:
                p = rpath(tmpl.format(s=fs(k)))
                with np.load(p) as z:
                    meta = json.loads(str(z["meta"]))
                ck = meta.get("checkpoint_sha256") or meta.get("candidate", {}).get("sha256")
                if (m, seed) in VAL_CKPT:
                    assert ck == VAL_CKPT[(m, seed)], (p, ck)
                if m == "ringpp":
                    assert meta["method"] == "ringpp_i2", p
                r = both_rules(p, sc, k)
                r["checkpoint_sha256"] = ck
                per[m][seed][k] = r
            runs[m]["seeds"][seed]["val_source"] = tmpl
    for m, lst in VAL_JSON.items():
        for seed, tmpl in lst:
            for k in GROUPS["B_val"]:
                j = json.load(open(rpath(tmpl.format(s=fs(k)))))
                shas = {j["candidate"]["sha256"]} | {a["sha256"] for a in j["candidate"]["aliases"]}
                assert HELIOS_CKPT[seed] in shas, (m, seed, k)
                mm = j["metrics"]
                r1 = mm.get("R@1")
                per[m][seed][k] = {"ledger": None, "summary_json": tmpl.format(s=fs(k)),
                                   "query_span": "30<=q<N-15 (summary json; ledger npz absent)", "inv": None,
                                   "ms": {"n_queries": mm["n_queries"], "n_pos": mm["n_pos"],
                                          "n_hit1": int(round(r1 * mm["n_pos"])) if r1 is not None else 0,
                                          **{x: mm.get(x) for x in (*M4, "R@1")}}}
            runs[m]["seeds"][seed]["val_source"] = tmpl
    # NSD
    per["nsd"] = {}
    runs["nsd"] = {"score": "score", "seeds": {}}
    for i in ("1", "2", "3"):
        per["nsd"][i] = {}
        runs["nsd"]["seeds"][i] = {"test_source": str((NSD / "test").relative_to(REPO)) + f"/<seq>/diff_s{i}.npz",
                                   "val_source": str(NSD_VAL.relative_to(REPO)) + f"/<seq>/diff_s{i}.npz"}
        for k in TARGET:
            p = (NSD_VAL / fs(k) / f"diff_s{i}.npz") if k in VALSEQ else (NSD / "test" / fs(k) / f"diff_s{i}.npz")
            per["nsd"][i][k] = both_rules(p, "score", k)
            if k not in VALSEQ:   # 시퀀스별 값이 NSD summary와 같은지
                ref = nsd_summary["per_case"][f"diff_s{i}"]["sequences"][k]
                got = per["nsd"][i][k]["ms"]
                assert ref["n_pos"] == got["n_pos"], k
                if ref["n_pos"]:
                    assert max(abs(ref[x] - got[x]) for x in (*M4, "R@1")) < 1e-12, k

    # invkey 쪽과 검산 (3)
    inv, st, rc, s3 = invkey_rows()
    c3 = {}
    ref_led = per["bevplace"]["1"]
    for k in TARGET:
        if k in VALSEQ:
            src = per["ringpp"]["-"][k]      # 검증 중 전 질의를 가진 원장
        else:
            src = ref_led[k]
        a = src["inv"]
        b = rc[k]["raw"]
        c3[k] = {"inv_counts_ledger": [a["n_pos"], a["n_neg"], a["n_ambiguous"]],
                 "inv_counts_invkey": [b["n_pos"], b["n_neg"], b["n_ambiguous"]],
                 "dir_n_ledger": a["dir_n"], "dir_n_invkey": b["dir_n"],
                 "ms_n_pos": src["ms"]["n_pos"]}
        g = gt(k)
        if "pos_q" in b:   # MCD만 질의별 기록이 있다
            qp = np.array(b["pos_q"])
            c3[k]["pos_q_equal"] = bool(np.array_equal(np.flatnonzero(g["dmin"] <= 5.0), qp))
            c3[k]["pos_jnn_equal"] = bool(np.array_equal(g["jnn"][qp], np.array(b["pos_jnn"])))
            dd = classify(g["u"], g["moving"], qp, g["jnn"][qp])
            c3[k]["pos_dir_equal"] = bool(np.array_equal(dd, np.array(b["pos_dir"])))
            assert c3[k]["pos_q_equal"] and c3[k]["pos_dir_equal"], (k, c3[k])
        assert c3[k]["inv_counts_ledger"] == c3[k]["inv_counts_invkey"], (k, c3[k])
        assert a["dir_n"] == b["dir_n"], (k, c3[k])
        if k in s3:
            assert s3[k]["raw"]["n_pos"] == b["n_pos"] and s3[k]["raw"]["n_hit1"] == b["n_hit1"], k
        # 같은 시퀀스의 모든 원장이 같은 질의 분류를 쓰는지
        for m in per:
            for seed in per[m]:
                r = per[m][seed].get(k)
                if r and r.get("inv"):
                    assert r["inv"]["n_pos"] == a["n_pos"] and r["inv"]["dir_n"] == a["dir_n"], (m, seed, k)
                if r and r["ms"]:
                    assert r["ms"]["n_pos"] == src["ms"]["n_pos"], (m, seed, k)
    checks["invkey_query_classes"] = c3
    print("check (2)(3) ledger dmin == cache GT, invkey query classes/directions equal", flush=True)

    # 묶음
    allrows = {**{m: per[m] for m in per}, **inv}
    groups = {}
    for gname, keys in GROUPS.items():
        groups[gname] = {}
        for name, seeds in allrows.items():
            g = {}
            for rule in ("inv", "ms", "ms_r1_only"):
                ps = {sd: group_stat(rows, keys, rule) for sd, rows in seeds.items()
                      if all(k in rows and rows[k].get(rule) is not None for k in keys)}
                if ps:
                    g[rule] = {"per_seed": ps, "spread": seed_spread(ps)}
            groups[gname][name] = g
    # 검산 (4): invkey 묶음 평균 정의가 stepctl table을 재현하는지
    c4 = {}
    for gname, sname in (("A_tuhh", "mcd_tuhh"), ("B_val", "kitti_k360")):
        ref = st["table"]["new3_st5000last"][sname]
        got = groups[gname]["invkey_lin16_new3_st5000"]["inv"]["spread"]
        c4[gname] = max(abs(ref[k][x] - got[k][x]) for k in ("R@1_pooled", "AP", "R@100P")
                        for x in ("mean", "min", "max"))
        assert c4[gname] < 1e-12, (gname, c4[gname])
    checks["invkey_group_definition_reproduces_stepctl_table"] = c4

    out = {"date": "2026-11-15", "rules": {
        "inv": "invkey: q>=30 to end, j<=q-30, pos d_min<=5, neg >25, 5-25 excluded, top-1 score, _pr_eval_lib",
        "ms": "manuscript common queries: same but 30<=q<N-15",
        "ms_r1_only": "invkey raw / old linear-16 seed 0 positives restricted to q<N-15 (fov_survey/recall.json); R@1 only",
        "direction": "query q vs nearest past GT keyframe jnn heading (P[i+3]-P[i-3]); same<30, reverse>150 deg"},
        "group_definition": "R@1 and directional R@1 pooled over positives; AP/max_F1/R@100P = mean over sequences "
                            "with positives; seeds -> mean/min/max",
        "groups": GROUPS, "runs": runs, "per_seq": per, "invkey_per_seq": inv, "group_stats": groups,
        "checks": checks,
        "gt_caches": {k: {"cache": gt(k)["cache"], "sha256": gt(k)["cache_sha256"], "n_frames": gt(k)["n"]}
                      for k in TARGET},
        "inputs_sha256": {str(p.relative_to(REPO)): sha256(p) for p in (
            T2, NSD / "summary.json", INV / "scope3/metrics.json", INV / "scope3/stepctl/metrics.json",
            INV / "fov_survey/recall.json", Path(__file__))}}
    (OUT / "lookup.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # 표
    names = {"scpp": "SC++", "fresco": "FreSCo", "ring": "RING", "ringpp": "RING++(I2)", "disco": "DiSCO",
             "deepring": "DeepRING", "ringsharp": "RING#-L", "helios": "HeLiOS", "seqot": "SeqOT",
             "bevplace": "BEVPlace++", "nsd": "NSD(diff)", "invkey_raw": "invkey raw spec",
             "invkey_lin16_old": "invkey 기존 선형16", "invkey_lin16_new3_st5000": "invkey 새범위 5000스텝 선형16"}
    lines = []

    def cell(v):
        if v is None:
            return "–"
        if isinstance(v, dict):
            if abs(v["max"] - v["min"]) < 1e-12:
                return f"{100 * v['mean']:.2f}"
            return f"{100 * v['mean']:.2f} [{100 * v['min']:.2f}, {100 * v['max']:.2f}]"
        return f"{100 * v:.2f}"

    for gname in GROUPS:
        for rule in ("inv", "ms", "ms_r1_only"):
            rows = [(n, groups[gname][n][rule]["spread"]) for n in names if rule in groups[gname].get(n, {})]
            if not rows:
                continue
            lines.append(f"\n### {gname} / {rule}\n")
            lines.append("| 방법 | 시드 | 양성 | R@1 | AP | 최대F1 | R@100P | 같은 | 반대 | 교차 |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for n, s in rows:
                npos = s["n_pos"]["mean"] if isinstance(s["n_pos"], dict) else s["n_pos"]
                lines.append(f"| {names[n]} | {','.join(s['seeds'])} | {int(npos)} | {cell(s['R@1_pooled'])} | "
                             + " | ".join(cell(s[x]) for x in M4) + " | "
                             + " | ".join(cell(s.get(f"R@1_{d}_pooled")) for d in DIRS[:3]) + " |")
            dn = rows[0][1]
            lines.append(f"\n방향 양성 수: " + ", ".join(
                f"{d} {dn[f'n_{d}']['mean'] if isinstance(dn.get(f'n_{d}'), dict) else dn.get(f'n_{d}')}"
                for d in DIRS if f"n_{d}" in dn))
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
