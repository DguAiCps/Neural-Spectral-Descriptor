#!/usr/bin/env python3
"""RING#-L 표2(분할 v2): 재정렬 깊이 n_rerank = 1000(공개 코드·README) 대 후보 전체(논문 Eq.38·표 VIII).

바꾸는 것은 재정렬 깊이 하나뿐이다. 체크포인트(시드 1·2·3, 선택 에폭 2·1·29), BEV, 회전 분기,
이동 분기, 수락 점수(score_trans), 질의·후보 규칙, 집계기는 표2 실행(scripts/_ringsharp_official.py)과 같다.

한 질의에서 회전 분기(모든 후보)는 한 번만 계산하고, 이동 분기는 회전 순서의 앞 min(1000, |후보|)개를
표2 코드와 같은 호출로 계산한 뒤(=n_rerank 1000 결과, 표2 원장과 대조) 나머지 후보를 같은 함수로 따로
계산해 붙인다(=후보 전체 결과). 두 결과가 같은 회전 분기·같은 앞 1000개 오차를 공유하므로 차이는
재정렬 깊이에서만 나온다. 시간은 회전·질의 특징·앞부분 이동·나머지 이동으로 나눠 잰다.

하위 명령(컨테이너 ring_baseline, GPU는 val·test만):
  count                        후보 수 > 1000 질의 비율(시험46: 표2 원장, 검증9: 키프레임) (CPU)
  val  --seed K --seq S        검증9 선택 규약(GT 키프레임, 재방문 5 m, |i-j|>30, 4번째 질의마다) R@1
  test --seq S [--seeds ...]   시험46 원장 두 벌(n1000·all), 표2 원장과 n1000 대조
  aggregate                    검증 가족 매크로 R@1, 시험 공통 질의 도시 매크로 4지표, 순위 (CPU)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("SEED", "1")         # _ringsharp_official 의 경로 상수만 시드1로 둔다(쓰지 않음)
import _ringsharp_official as rso          # noqa: E402
import _ring_official_lib as rol           # noqa: E402
from _pr_eval_lib import MIN_INDEX_GAP     # noqa: E402
from _pdv2_pilot import TEST, VAL          # noqa: E402

OUT = REPO / "results/pdv2_ringsharp_rerank_all_20261118"
N_RERANK = 1000
SEEDS = (1, 2, 3)
SRC = {1: REPO / "results/pdv2_ring_official", 2: REPO / "results/pdv2_ring_official_s2",
       3: REPO / "results/pdv2_ring_official_s3"}
SUMMARY = REPO / ("results/pdv2_deepring_scaleinit_integration_20261110/"
                  "baseline_seeds_summary_seqot_stable_ringpp_i2_fresco_kgf_deepring_scaleinit.json")
TABLE2 = ("scpp", "fresco", "ring", "ringpp", "ringsharp", "deepring", "disco", "helios", "seqot", "bevplace")


def _weight(seed):
    sel = json.loads((SRC[seed] / "ringsharp_ckpt_selection.json").read_text())
    assert sel["query_translation_path"] == rso.TRANS_QUERY_PATH
    p = REPO / sel["selected_weight"].split("/Neural-Spectral-Codec/", 1)[1]
    assert rso._sha_file(p) == sel["weight_sha256"], p
    return p, sel


def _timer():
    import torch
    torch.cuda.synchronize()
    return time.perf_counter()


def _query(base, model, trans_cnn, specs, bts, qspec, qpacked, cand, device, tm, xcheck, do_x):
    """회전 분기 → 앞 m개 이동 오차(표2와 같은 호출) → 나머지 이동 오차. 반환 (dist, ang, order, m, e1, e2)."""
    t0 = _timer()
    dist, ang = rso.rotation_branch(base, specs, qspec, cand)
    order = np.argsort(dist)                               # 표2·공식: np.argsort
    t1 = _timer()
    m = min(N_RERANK, len(cand))
    top, rest = order[:m], order[m:]
    qfeat = rso.query_features(base, model, qpacked, device)
    tq = _timer()
    if do_x:
        x = {"first": rso.xcheck_translation(base, trans_cnn, bts, qfeat, cand[top], ang[top])}
        if len(rest):
            x["rest"] = rso.xcheck_translation(base, trans_cnn, bts, qfeat, cand[rest], ang[rest])
        xcheck.append(x)
    t2 = _timer()
    e1 = rso.translation_branch(base, trans_cnn, bts, qfeat, cand[top], ang[top], False)
    t3 = _timer()
    e2 = (rso.translation_branch(base, trans_cnn, bts, qfeat, cand[rest], ang[rest], False)
          if len(rest) else np.zeros(0, np.float32))
    t4 = _timer()
    tm["rot"] += t1 - t0
    tm["qfeat"] += tq - t1                                 # 교차 검산(xcheck) 시간은 넣지 않는다
    tm["trans_first"] += t3 - t2
    tm["trans_rest"] += t4 - t3
    tm["n_trans_first"] += m
    tm["n_trans_rest"] += len(rest)
    return dist, ang, order, m, e1, e2


def _timing(tm, nq):
    ms = lambda k: round(1e3 * tm[k] / max(nq, 1), 2)  # noqa: E731
    return {"n_queries": nq, "rotation_ms_per_query": ms("rot"), "query_feature_ms_per_query": ms("qfeat"),
            "translation_first_ms_per_query": ms("trans_first"), "translation_rest_ms_per_query": ms("trans_rest"),
            "n1000_ms_per_query": round(1e3 * (tm["rot"] + tm["qfeat"] + tm["trans_first"]) / max(nq, 1), 2),
            "all_ms_per_query": round(1e3 * (tm["rot"] + tm["qfeat"] + tm["trans_first"] + tm["trans_rest"])
                                      / max(nq, 1), 2),
            "mean_translation_candidates_n1000": round(tm["n_trans_first"] / max(nq, 1), 1),
            "mean_translation_candidates_all": round((tm["n_trans_first"] + tm["n_trans_rest"]) / max(nq, 1), 1),
            "seconds": {k: round(v, 1) for k, v in tm.items() if not k.startswith("n_")}}


def _new_tm():
    return dict(rot=0.0, qfeat=0.0, trans_first=0.0, trans_rest=0.0, n_trans_first=0, n_trans_rest=0)


# ------------------------------------------------------------------ count
def cmd_count():
    from baselines.eval_utils import _find_revisit_queries
    from _pdv2_main_baselines_official import COMMON
    rep = {"rule": "후보 = 질의 t의 과거 키프레임 j <= t-30 (시험, 표2 원장 n_candidates, 공통 질의 30<=t<N-15); "
                   "검증 = |i-j|>30 인 모든 키프레임(선택 규약, 4번째 재방문 질의마다)",
           "test": {}, "val": {}}
    tot = big = 0
    for s, q in TEST:
        with np.load(SRC[1] / f"odom/dump_ringsharp_{s}_{q}.npz") as z, \
                np.load(COMMON / f"s1/test_selected_symmetric_pbev128_main_{s}_{q}.npz") as n:
            ix = np.searchsorted(z["query_idx"], n["query_idx"])
            nc = z["n_candidates"][ix]
            assert np.array_equal(z["n_trans"][ix], np.minimum(nc, N_RERANK))
        b = int((nc > N_RERANK).sum())
        rep["test"][f"{s}/{q}"] = {"n_queries": int(len(nc)), "n_over_1000": b, "frac_over_1000": b / len(nc),
                                   "max_candidates": int(nc.max())}
        tot, big = tot + len(nc), big + b
    rep["test_total"] = {"n_queries": tot, "n_over_1000": big, "frac_over_1000": big / tot}
    tot = big = 0
    for s, q in VAL:
        _, poses = rol.load_keyframes(rol.split_cache("gt", "val", s, q))
        n = len(poses)
        qs = _find_revisit_queries(poses[:, :3, 3].astype(np.float64), 5.0, 30)[::rso.VAL_QUERY_STRIDE]
        nc = np.array([n - (min(i + 30, n - 1) - max(i - 30, 0) + 1) for i, _ in qs])
        b = int((nc > N_RERANK).sum())
        rep["val"][f"{s}/{q}"] = {"n_frames": n, "n_queries": int(len(nc)), "n_over_1000": b,
                                  "frac_over_1000": b / max(len(nc), 1)}
        tot, big = tot + len(nc), big + b
    rep["val_total"] = {"n_queries": tot, "n_over_1000": big, "frac_over_1000": big / tot}
    rso_write(OUT / "counts.json", rep)
    print(json.dumps({"test": rep["test_total"], "val": rep["val_total"]}))


def rso_write(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, p)


# -------------------------------------------------------------- validation
def cmd_val(seed, seq):
    import torch
    from baselines.eval_utils import compute_recall_cosine_then_rerank
    s, q = rso._split(seq, VAL)
    out = OUT / f"val/s{seed}/{s}_{q}.json"
    if out.exists():
        print(f"SKIP {out}", flush=True)
        return
    base = rso._base()
    w, sel = _weight(seed)
    epoch = int(sel["selected_epoch"][1:])
    ref = json.loads((SRC[seed] / f"ringsharp_val/val_e{epoch}_{s}_{q}.json").read_text())
    device = "cuda"
    model, trans_cnn = base.load_model(str(w), device)
    scan_ids, poses = rol.load_keyframes(rol.split_cache("gt", "val", s, q))
    z = np.load(rso.VAL_BEV / f"{s}_{q}.npz")
    assert np.array_equal(z["scan_ids"], scan_ids) and np.allclose(z["bounds"], rol.ringsharp_bounds(s))
    t0 = time.time()
    g, specs, bts = rso.features(base, model, trans_cnn, rso.bev_batches_from_packed(base, z["packed"]),
                                 len(scan_ids), device)
    t_feat = time.time() - t0
    specs, bts = specs.to(device), bts.to(device)
    tm, xcheck, ranked_all = _new_tm(), [], {}
    chg = {"top1": 0, "top1_from_rest": 0, "top10_set": 0}

    def rerank_fn(qi, cand):
        cand = np.asarray(cand)
        do_x = len(xcheck) < rso.N_XCHECK and min(N_RERANK, len(cand)) >= 2 * rso.BATCH
        dist, ang, order, m, e1, e2 = _query(base, model, trans_cnn, specs, bts, specs[qi:qi + 1],
                                             z["packed"][qi:qi + 1], cand, device, tm, xcheck, do_x)
        o1000 = order.copy()
        o1000[:m] = order[:m][np.argsort(e1)]                       # 표2 cmd_val 과 같은 줄
        ea = np.concatenate([e1, e2])
        ranked_all[qi] = cand[order[np.argsort(ea)]]
        chg["top1"] += int(ranked_all[qi][0] != cand[o1000[0]])
        chg["top1_from_rest"] += int(int(np.argmin(ea)) >= m)
        chg["top10_set"] += int(set(ranked_all[qi][:10].tolist()) != set(cand[o1000[:10]].tolist()))
        return cand[o1000]

    t0 = time.time()
    r1000, nq = compute_recall_cosine_then_rerank(g, rerank_fn, poses, k_values=[1, 5, 10], n_coarse=len(g),
                                                  query_stride=rso.VAL_QUERY_STRIDE)
    t_ret = time.time() - t0

    def lookup(qi, cand):
        r = ranked_all[qi]
        assert len(r) == len(cand) and set(r.tolist()) == set(np.asarray(cand).tolist())
        return r
    rall, nq2 = compute_recall_cosine_then_rerank(g, lookup, poses, k_values=[1, 5, 10], n_coarse=len(g),
                                                  query_stride=rso.VAL_QUERY_STRIDE)
    assert nq == nq2 == ref["n_queries"]
    rso_write(out, {
        "sensor": s, "sequence": q, "seed": seed, "epoch": epoch, "n_frames": len(scan_ids), "n_queries": nq,
        "n1000": {f"recall@{k}": r1000[k] for k in (1, 5, 10)},
        "all": {f"recall@{k}": rall[k] for k in (1, 5, 10)},
        "table2_val_json": {f"recall@{k}": ref[f"recall@{k}"] for k in (1, 5, 10)},
        "n1000_reproduces_table2_val": all(abs(r1000[k] - ref[f"recall@{k}"]) < 1e-12 for k in (1, 5, 10)),
        "weight": str(w.relative_to(REPO)), "weight_sha256": sel["weight_sha256"],
        "changed_queries": chg,
        "translation_xcheck": xcheck, "val_query_stride": rso.VAL_QUERY_STRIDE,
        "protocol": "revisit<5m, |i-j|>30, intra-sequence (baselines/eval_utils), GT keyframes",
        "timing": {"feature_s": round(t_feat, 1), "retrieval_s": round(t_ret, 1), **_timing(tm, nq)},
    })
    print(f"VAL_DONE s{seed} {seq}: R@1 n1000 {r1000[1]:.4f} (표2 {ref['recall@1']:.4f}) all {rall[1]:.4f}; "
          f"{_timing(tm, nq)['n1000_ms_per_query']} / {_timing(tm, nq)['all_ms_per_query']} ms/q", flush=True)
    del specs, bts
    torch.cuda.empty_cache()


# -------------------------------------------------------------------- test
def _ledger_path(seed, depth, s, q):
    return OUT / f"ledgers/s{seed}/{depth}/dump_ringsharp_{s}_{q}.npz"


def cmd_test(seq, seeds):
    import torch
    s, q = rso._split(seq, TEST)
    todo = [k for k in seeds if not all(_ledger_path(k, d, s, q).exists() for d in ("n1000", "all"))]
    if not todo:
        print(f"SKIP {seq}", flush=True)
        return
    base = rso._base()
    device = "cuda"
    cache = rol.split_cache("odom", "test", s, q)
    scan_ids, poses = rol.load_keyframes(cache)
    positions = poses[:, :3, 3]
    min_past = rol.min_past_dists(positions, MIN_INDEX_GAP)
    n = len(scan_ids)
    xq = set(sorted({n // 4, n // 2, n - 1})[:rso.N_XCHECK])
    packed = None
    for seed in todo:
        w, sel = _weight(seed)
        model, trans_cnn = base.load_model(str(w), device)
        t0 = time.time()
        if packed is None:
            sink = []
            loader = rol.make_loader(s, q)
            _, specs, bts = rso.features(base, model, trans_cnn,
                                         rso.tee_packed(rso.bev_batches_from_loader(
                                             base, loader, scan_ids, rol.ringsharp_bounds(s)), sink), n, device)
            packed = np.concatenate(sink)
            del loader
        else:
            _, specs, bts = rso.features(base, model, trans_cnn, rso.bev_batches_from_packed(base, packed),
                                         n, device)
        t_feat = time.time() - t0
        specs, bts = specs.to(device), bts.to(device)
        tm, xcheck, res_all = _new_tm(), [], {}

        def rerank_top1(t, npf):
            cand = np.arange(npf)
            dist, ang, order, m, e1, e2 = _query(base, model, trans_cnn, specs, bts, specs[t:t + 1],
                                                 packed[t:t + 1], cand, device, tm, xcheck, t in xq)
            i1 = int(np.argmin(e1))                                    # 표2 cmd_dump 과 같은 줄
            ea = np.concatenate([e1, e2])
            ia = int(np.argmin(ea))
            res_all[t] = {"j": int(order[ia]), "err": float(ea[ia]), "rot_of_j": float(dist[order[ia]]),
                          "j_rot": int(order[0]), "rot_min": float(dist[order[0]]), "n_trans": npf}
            return {"j": int(order[i1]), "err": float(e1[i1]), "rot_of_j": float(dist[order[i1]]),
                    "j_rot": int(order[0]), "rot_min": float(dist[order[0]]), "n_trans": m}

        t0 = time.time()
        a1000 = rso.ledger(n, positions, min_past, rerank_top1)
        t_led = time.time() - t0
        aall = rso.ledger(n, positions, min_past, lambda t, npf: res_all[t])
        nq = len(a1000["query_idx"])
        with np.load(SRC[seed] / f"odom/dump_ringsharp_{s}_{q}.npz") as r:
            same_j = float(np.mean(r["j_star"] == a1000["j_star"]))
            dscore = float(np.nanmax(np.abs(r["score_trans"] - a1000["score_trans"])))
            same_jrot = float(np.mean(r["j_star_rot"] == a1000["j_star_rot"]))
        timing = _timing(tm, nq)
        for depth, arr in (("n1000", a1000), ("all", aall)):
            meta = {"schema": "pr-dump-v1", "method": "ringsharp", "sensor": s, "sequence": q, "condition": "odom",
                    "n_frames": n, "n_queries": nq, "min_index_gap": MIN_INDEX_GAP,
                    "n_rerank": N_RERANK if depth == "n1000" else "all", "batch": rso.BATCH,
                    "retrieval": ("rotation branch ranks every keyframe j <= t - 30; translation branch re-orders "
                                  + ("the first min(1000, |prefix|)" if depth == "n1000" else "all of them")),
                    "query_translation_path": rso.TRANS_QUERY_PATH, "translation_errors": "trans_errors_exact",
                    "translation_xcheck": xcheck, "seed": seed,
                    "artifacts": {"weight": str(w.relative_to(REPO)), "weight_sha256": sel["weight_sha256"],
                                  "keyframe_cache": str(cache.relative_to(REPO))},
                    "table2_check": {"j_star_equal_frac": same_j, "score_trans_max_abs_diff": dscore,
                                     "j_star_rot_equal_frac": same_jrot},
                    "timing": {"feature_s_total": round(t_feat, 1), "ledger_s_total": round(t_led, 1), **timing}}
            p = _ledger_path(seed, depth, s, q)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(".tmp_" + p.name)
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, meta=json.dumps(meta), **arr)
            os.replace(tmp, p)
        changed = float(np.mean(a1000["j_star"] != aall["j_star"]))
        print(f"TEST_DONE s{seed} {seq}: 표2 j일치 {same_j:.4f} |Δscore| {dscore:.2e}; 1위 바뀜 {changed:.3f}; "
              f"feat {t_feat:.0f}s ledger {t_led:.0f}s; {timing['n1000_ms_per_query']} / "
              f"{timing['all_ms_per_query']} ms/q", flush=True)
        del specs, bts, model, trans_cnn
        torch.cuda.empty_cache()


# --------------------------------------------------------------- aggregate
def _agg_test(folder_fmt, score="score_trans"):
    """_pdv2_main_baselines_official.aggregate 의 공통 질의 집계(30<=q<N-15, 도시 6개 → 매크로)."""
    from _pdv2_main_baselines_official import COMMON, METRICS
    from _pdv2_pilot import CITY
    from _pr_eval_lib import AGG_METRICS, classify_queries, pr_sweep
    per = {}
    for s, q in TEST:
        with np.load(folder_fmt(s, q)) as z, \
                np.load(COMMON / f"s1/test_selected_symmetric_pbev128_main_{s}_{q}.npz") as n:
            ix = np.searchsorted(z["query_idx"], n["query_idx"])
            np.testing.assert_array_equal(z["query_idx"][ix], n["query_idx"])
            np.testing.assert_array_equal(z["min_past_gt_dist"][ix], n["min_past_gt_dist"])
            assert np.all(z["j_star"][ix] <= n["query_idx"] - 30)
            qc = classify_queries(z["min_past_gt_dist"][ix])
            row = {"n_queries": len(ix), "n_positive": qc["n_pos"]}
            if qc["n_pos"]:
                sw = pr_sweep(z[score][ix], z["top1_geo_dist"][ix], qc)
                row.update({k: float(sw[k]) for k in AGG_METRICS})
                row["R@1"] = float((z["top1_geo_dist"][ix][qc["positive"]] <= 5).mean())
            per[f"{s}/{q}"] = row
    groups = {c: {k: float(np.mean([per[s][k] for s in names if per[s]["n_positive"]])) for k in METRICS}
              for c, names in CITY.items()}
    return {"macro": {k: float(np.mean([g[k] for g in groups.values()])) for k in METRICS},
            "city_groups": groups, "per_sequence": per}


def _ms(vals):
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "values": a.tolist()}


def _sources():
    from _pdv2_main_baselines_official import sha256
    return {str(p.relative_to(REPO)): sha256(p) for p in
            [Path(__file__), REPO / "scripts/_ringsharp_official.py", REPO / "scripts/_pr_eval_lib.py",
             REPO / "scripts/_pdv2_main_baselines_official.py", SUMMARY]}


def _cost_estimate():
    """검증9 실측 단가(같은 공유 GPU0) × 시험46 후보 수(표2 원장 전체 질의). BEV 생성은 표2 시드1 실측."""
    v = [json.loads(p.read_text())["timing"] for p in sorted((OUT / "val").glob("s*/*.json"))]
    nq = sum(t["n_queries"] for t in v)
    cand_all = sum(t["mean_translation_candidates_all"] * t["n_queries"] for t in v)
    cand_rest = cand_all - sum(t["mean_translation_candidates_n1000"] * t["n_queries"] for t in v)
    sec = lambda k: sum(t["seconds"][k] for t in v)  # noqa: E731
    unit = {"rotation_ms_per_candidate": 1e3 * sec("rot") / cand_all,
            "query_feature_ms_per_query": 1e3 * sec("qfeat") / nq,
            "translation_first_ms_per_query": 1e3 * sec("trans_first") / nq,
            "translation_rest_ms_per_candidate": 1e3 * sec("trans_rest") / cand_rest,
            "feature_s_per_frame_from_packed_bev": sum(t["feature_s"] for t in v) /
            sum(json.loads(p.read_text())["n_frames"] for p in sorted((OUT / "val").glob("s*/*.json")))}
    nqt = c_all = c_1000 = frames = 0
    bev_s = 0.0
    for s, q in TEST:
        with np.load(SRC[1] / f"odom/dump_ringsharp_{s}_{q}.npz") as z:
            nc = z["n_candidates"].astype(float)
            m = json.loads(str(z["meta"]))
        nqt, c_all, c_1000 = nqt + len(nc), c_all + nc.sum(), c_1000 + np.minimum(nc, N_RERANK).sum()
        frames += m["n_frames"]
        bev_s += m["timing"]["feature_s_total"]
    per_seed = (unit["rotation_ms_per_candidate"] * c_all + unit["query_feature_ms_per_query"] * nqt
                + unit["translation_first_ms_per_query"] * nqt) / 3.6e6
    rest = unit["translation_rest_ms_per_candidate"] * (c_all - c_1000) / 3.6e6
    feat = unit["feature_s_per_frame_from_packed_bev"] * frames / 3600
    total = bev_s / 3600 + 2 * feat + 3 * (per_seed + rest)
    return {"unit_costs_val9": unit,
            "test46": {"n_queries": int(nqt), "n_frames": int(frames), "translation_candidates_n1000": float(c_1000),
                       "translation_candidates_all": float(c_all)},
            "estimate_gpu_hours": {"per_seed_search_n1000": per_seed, "per_seed_search_all": per_seed + rest,
                                   "bev_from_raw_once_table2_seed1": bev_s / 3600, "features_per_extra_seed": feat,
                                   "three_seeds_all_total": total, "threshold": 15.0},
            "note": "GPU0 공유(다른 학습·평가 작업과 동시) 상태 실측. 검색은 질의마다 순차 실행."}


def cmd_aggregate():
    from _pdv2_c3 import _family_pooled
    from _pdv2_main_baselines_official import METRICS
    rep = {"protocol": str((OUT / "protocol.md").relative_to(REPO)),
           "counts": json.loads((OUT / "counts.json").read_text())}
    # 검증9
    val = {}
    for seed in SEEDS:
        rows = {d: {} for d in ("n1000", "all", "table2")}
        tim = []
        for s, q in VAL:
            v = json.loads((OUT / f"val/s{seed}/{s}_{q}.json").read_text())  # noqa: E501
            assert v["n1000_reproduces_table2_val"], (seed, s, q)
            for d, key in (("n1000", "n1000"), ("all", "all"), ("table2", "table2_val_json")):
                rows[d][f"{s}/{q}"] = {"recall@1": v[key]["recall@1"], "n_queries": v["n_queries"]}
            tim.append(v["timing"])
        sel = json.loads((SRC[seed] / "ringsharp_ckpt_selection.json").read_text())
        nq = sum(t["n_queries"] for t in tim)
        val[seed] = {"epoch": sel["selected_epoch"],
                     "selection_json_family_macro_r1": sel["epochs"][sel["selected_epoch"]]["family_macro_r1"],
                     **{d: _family_pooled(rows[d]) for d in rows},
                     "per_seq": {k: {"n1000": rows["n1000"][k]["recall@1"], "all": rows["all"][k]["recall@1"],
                                     "n_queries": rows["n1000"][k]["n_queries"]} for k in rows["n1000"]},
                     "ms_per_query": {
                         "n1000": sum(t["n1000_ms_per_query"] * t["n_queries"] for t in tim) / nq,
                         "all": sum(t["all_ms_per_query"] * t["n_queries"] for t in tim) / nq}}
        assert abs(val[seed]["table2"] - val[seed]["selection_json_family_macro_r1"]) < 1e-12
    rep["val9"] = {"metric": "선택 규약 가족 매크로 5 m R@1(가족 안 질의 합침, 가족 동일 가중)",
                   "per_seed": val,
                   "mean": {d: _ms([val[k][d] for k in SEEDS]) for d in ("n1000", "all")},
                   "delta_pp": _ms([100 * (val[k]["all"] - val[k]["n1000"]) for k in SEEDS])}
    rep["val9"]["changed_queries"] = {
        f"s{k}": {f"{s_}/{q_}": json.loads((OUT / f"val/s{k}/{s_}_{q_}.json").read_text()).get("changed_queries")
                  for s_, q_ in VAL} for k in SEEDS}
    rep["test46_cost_estimate"] = _cost_estimate()
    if not all(_ledger_path(k, d, s_, q_).exists() for k in SEEDS for d in ("n1000", "all") for s_, q_ in TEST):
        rep["test46"] = "실행 안 함: 추정 비용이 규칙 4의 15 GPU 시간 이상(test46_cost_estimate)"
        probe = {}
        for p in sorted((OUT / "ledgers").glob("s*/n1000/*.npz")):
            pa = p.parents[1] / "all" / p.name
            with np.load(p) as z1, np.load(pa) as za:
                m1 = json.loads(str(z1["meta"]))
                probe[f"{p.parents[1].name}/{m1['sensor']}/{m1['sequence']}"] = {
                    "table2_check_n1000": m1["table2_check"],
                    "top1_changed": int((z1["j_star"] != za["j_star"]).sum()), "n_queries": m1["n_queries"],
                    "n1000_ms_per_query": m1["timing"]["n1000_ms_per_query"],
                    "all_ms_per_query": m1["timing"]["all_ms_per_query"]}
        rep["test46_probe_sequences"] = {"note": "규칙 1 검산용 시험 시퀀스(시드1 실행 중 GPU0 경합으로 시간 과대)",
                                         "rows": probe}
        rso_write(OUT / "summary.json", {**rep, "table2_changed": False, "input_sha256": _sources()})
        print("val9", {d: round(100 * rep["val9"]["mean"][d]["mean"], 3) for d in ("n1000", "all")},
              "delta", rep["val9"]["delta_pp"])
        print("cost", json.dumps(rep["test46_cost_estimate"]["estimate_gpu_hours"], ensure_ascii=False))
        return
    # 시험46
    summ = json.loads(SUMMARY.read_text())["methods"]
    test, sources = {}, {}
    for seed in SEEDS:
        t2 = _agg_test(lambda s, q: SRC[seed] / f"odom/dump_ringsharp_{s}_{q}.npz")
        for k in METRICS:
            assert abs(t2["macro"][k] - summ["ringsharp"]["per_seed"][str(seed)]["macro"][k]) < 1e-12, (seed, k)
        e = {"table2_ledger": t2["macro"]}
        for d in ("n1000", "all"):
            e[d] = _agg_test(lambda s, q: _ledger_path(seed, d, s, q))
        chk, tim, chg = [], [], []
        for s, q in TEST:
            with np.load(_ledger_path(seed, "n1000", s, q)) as z1, np.load(_ledger_path(seed, "all", s, q)) as za:
                m = json.loads(str(z1["meta"]))
                chk.append(m["table2_check"])
                tim.append(m["timing"])
                chg.append(int((z1["j_star"] != za["j_star"]).sum()))
        nq = sum(t["n_queries"] for t in tim)
        e["table2_check"] = {"min_j_star_equal_frac": min(c["j_star_equal_frac"] for c in chk),
                             "max_score_trans_abs_diff": max(c["score_trans_max_abs_diff"] for c in chk)}
        e["top1_changed_queries"] = {"n": int(sum(chg)), "frac_all_ledger_queries": sum(chg) / nq}
        e["timing"] = {"ms_per_query_n1000": sum(t["n1000_ms_per_query"] * t["n_queries"] for t in tim) / nq,
                       "ms_per_query_all": sum(t["all_ms_per_query"] * t["n_queries"] for t in tim) / nq,
                       "gpu_hours_n1000_ledger": sum(t["n1000_ms_per_query"] * t["n_queries"] for t in tim) / 3.6e6,
                       "gpu_hours_all_ledger": sum(t["all_ms_per_query"] * t["n_queries"] for t in tim) / 3.6e6,
                       "gpu_hours_features": sum(t["feature_s_total"] for t in tim) / 3600}
        test[seed] = e
    agg = {d: {k: _ms([test[s][d]["macro"][k] for s in SEEDS]) for k in METRICS} for d in ("n1000", "all")}
    delta = {k: _ms([100 * (test[s]["all"]["macro"][k] - test[s]["n1000"]["macro"][k]) for s in SEEDS])
             for k in METRICS}
    # 순위(표2 비교 방법 10종, 시드 평균): n1000 = 표2, all = RING#-L 행만 바꿈
    ranks = {}
    for k in ("AP", "max_F1", "R@100P", "R@1"):
        others = {m: summ[m]["mean_std"][k]["mean"] for m in TABLE2 if m != "ringsharp"}
        r = {}
        for d in ("n1000", "all"):
            vals = {**others, "ringsharp": agg[d][k]["mean"]}
            order = sorted(vals, key=lambda m: -vals[m])
            r[d] = {"rank_of_ringsharp": order.index("ringsharp") + 1, "order": order}
        ranks[k] = r
    rep["test46"] = {"metric": "공통 질의(30<=q<N-15) 도시 6개 매크로, 양성 과거 5 m·음성 25 m, score_trans",
                     "per_seed": test, "mean_std": agg, "delta_all_minus_n1000_pp": delta,
                     "rank_among_table2_baselines": ranks}
    rep["input_sha256"] = _sources()
    rep["table2_changed"] = False
    rso_write(OUT / "summary.json", rep)
    print("val9", {d: round(100 * rep["val9"]["mean"][d]["mean"], 3) for d in ("n1000", "all")})
    for d in ("n1000", "all"):
        print("test46", d, {k: f"{100 * agg[d][k]['mean']:.2f}±{100 * agg[d][k]['std']:.2f}" for k in METRICS})
    print("delta pp", {k: round(delta[k]["mean"], 3) for k in METRICS})
    print("ranks", {k: (ranks[k]["n1000"]["rank_of_ringsharp"], ranks[k]["all"]["rank_of_ringsharp"]) for k in ranks})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("count")
    sub.add_parser("aggregate")
    p = sub.add_parser("val")
    p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    p.add_argument("--seq", required=True)
    p = sub.add_parser("test")
    p.add_argument("--seq", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = ap.parse_args()
    if a.cmd == "count":
        cmd_count()
    elif a.cmd == "aggregate":
        cmd_aggregate()
    elif a.cmd == "val":
        cmd_val(a.seed, a.seq)
    else:
        cmd_test(a.seq, a.seeds)


if __name__ == "__main__":
    main()
