#!/usr/bin/env python3
"""08-20→02-04 날짜 쌍 진단의 독립 검산과 보고서.

서버 컨테이너 CPU 4스레드로 실행한다.
    docker exec -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=4 -w /workspace/Neural-Spectral-Codec ring_baseline \
        python3 scripts/_review_ring_followup_datepair_report.py verify
    ... report

verify: 입력(질의 재구성·원시 해시·수신 대조), 지도 TIRING과 기존 기술자 캐시의 비트 일치, 고정 부분 질의의
        FFT 없는 순환행렬 상관 재계산, 공식 출력 R@1·F1과 별도 구현 재계수를 확인한다.
report: 03-17(기존 공식 evaluate() 원장)과 08-20을 같은 함수로 집계해 나란히 보고한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _review_ring_followup_datepair as dp   # noqa: E402
from _review_ring_followup_verify import direct_dists   # noqa: E402

OUT = dp.OUT
RING_ROOT = OUT.parent
OLD = RING_ROOT / "official_eval"
CONDS = {"occ_official": ("RING", "ring", dp.NATIVE / "descriptor_cache"),
         "feat_maxfix": ("RING++ 정확max", "ringpp", dp.NATIVE / "maxfix/descriptor_cache")}
PAPER = {"RING": {"R@1": 0.7098, "F1": 0.7658, "AUC": 0.8246},
         "RING++ 정확max": {"R@1": 0.7321, "F1": 0.7890, "AUC": 0.8374}}
TABLE_II_NCLT_R1 = [0.7321, 0.6309, 0.7098, 0.6582, 0.2811, 0.1840, 0.5248, 0.3862, 0.1355]
TH = np.concatenate([np.linspace(0, .35, 10), np.linspace(.35, .55, 80), np.linspace(.55, 1, 10)])


def digest(path: Path) -> str:
    """파일의 SHA256."""
    return dp.digest(path)


def norm(x: np.ndarray) -> np.ndarray:
    """채널마다 평균0·편향분산1(eps 1e-5), 공식 circorr2의 InstanceNorm과 같은 정의."""
    x = x.astype(np.float64)
    mu = x.mean(axis=(-2, -1), keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(-2, -1), keepdims=True) + 1e-5)


def metrics(pairs: np.ndarray, qxy: np.ndarray, mxy: np.ndarray, radius: float) -> dict:
    """공식 정의를 numpy로 다시 쓴 지표. R@1은 query_radius(≤r), PR의 TP는 실제 거리<r(공식과 같음)."""
    geo = np.sqrt(((qxy[:, None, :] - mxy[None, :, :]) ** 2).sum(-1))
    top1 = pairs.argmin(axis=1)
    d1 = geo[np.arange(len(geo)), top1]
    pos = (geo <= radius).any(axis=1)
    hit_r1 = d1 <= radius
    best = pairs.min(axis=1)
    acc = best[:, None] < TH[None]
    tp = (acc & (d1 < radius)[:, None]).sum(0).astype(float)
    fp = (acc & ~(d1 < radius)[:, None]).sum(0).astype(float)
    fn = (~acc & pos[:, None]).sum(0).astype(float)
    p = tp / (tp + fp + 1e-10)
    r = tp / (tp + fn + 1e-10)
    f = 2 * p * r / (p + r + 1e-10)
    fixed_r = tp / max(int(pos.sum()), 1)
    fixed_p = tp / np.maximum(tp + fp, 1)
    fixed_f = 2 * fixed_p * fixed_r / np.maximum(fixed_p + fixed_r, 1e-10)
    auc = float(sum((r[i + 1] - r[i]) * (p[i + 1] + p[i]) / 2 for i in range(len(p) - 1)))
    wrong_pos = pos & ~hit_r1
    return {"queries": int(len(geo)), "positive_queries": int(pos.sum()), "correct_top1": int(hit_r1.sum()),
            "R@1_all_queries": float(hit_r1.mean()), "R@1_positive_only": float(hit_r1[pos].mean()),
            "official_grid_max_F1": float(f.max()), "official_grid_max_F1_threshold": float(TH[int(f.argmax())]),
            "official_PR_AUC": auc,
            "fixed_denominator_max_F1": float(fixed_f.max()),
            "positive_wrong": int(wrong_pos.sum()),
            "positive_wrong_within_20m": int((wrong_pos & (d1 <= 20.0)).sum()),
            "positive_wrong_over_20m": int((wrong_pos & (d1 > 20.0)).sum()),
            "_f1s": f}


def sets_for(date: str, split: str) -> tuple[np.ndarray, np.ndarray, list]:
    """질의·지도 위치(float32 → float64)와 질의 행."""
    ref = json.loads((dp.NATIVE / "evaluation_sets.json").read_text())[split]
    q = ref["query"] if date == "03-17" else json.loads((OUT / "query_sets.json").read_text())[split]["query"]
    qxy = np.array([e["xy"] for e in q], np.float32).astype(np.float64)
    mxy = np.array([e["xy"] for e in ref["map"]], np.float32).astype(np.float64)
    return qxy, mxy, q


def verify() -> dict:
    """독립 검산."""
    proto = json.loads((OUT / "protocol.json").read_text())
    cov = json.loads((OUT / "coverage.json").read_text())
    fc = json.loads((OUT / "fetch_complete.json").read_text())
    info = dp.members()["members"]
    pre = f"{dp.QUERY_SEQ}/velodyne_sync/"
    sets = json.loads((OUT / "query_sets.json").read_text())
    disco_hash = {r["name"]: r["sha256"] for r in json.loads((dp.DISCO / "complete.json").read_text())["raw_sha256"]}
    fetched_hash = {r["name"]: r["sha256"] for r in fc["raw_sha256"]}
    checks = {"protocol": {
        "datepair_script_unchanged": proto["script_sha256"] == digest(REPO / "scripts/_review_ring_followup_datepair.py"),
        "maxfix_unchanged": proto["maxfix_sha256"] == digest(REPO / "scripts/_review_native_ring_maxfix.py"),
        "query_sets_unchanged": proto["inputs"]["query_sets_sha256"] == digest(OUT / "query_sets.json"),
        "evaluation_sets_unchanged": proto["inputs"]["evaluation_sets_sha256"] == digest(dp.NATIVE / "evaluation_sets.json"),
        "official_sources_unchanged": all(digest(dp.OFFICIAL / p) == h
                                          for p, h in proto["inputs"]["official_source_sha256"].items()),
        "frozen_utc": proto["frozen_utc"]}}
    checks["query_reconstruction"] = {s: cov["splits"][s]["independent_reconstruction"] for s in dp.SPLITS}
    raw = {"queries_checked": 0, "fetched_sha_match": 0, "disco_sha_match": 0, "size_match": 0, "missing": 0}
    for name in sorted({Path(e["raw_path"]).name for s in dp.SPLITS for e in sets[s]["query"]}):
        path = dp.raw_path(f"{pre}{name}")
        raw["queries_checked"] += 1
        if not path.exists():
            raw["missing"] += 1
            continue
        h = digest(path)
        raw["size_match"] += int(path.stat().st_size == info[pre + name]["size"])
        if path.parent == dp.FETCHED:
            raw["fetched_sha_match"] += int(fetched_hash.get(name) == h)
        else:
            raw["disco_sha_match"] += int(disco_hash.get(name) == h)
    raw["control_reextraction"] = fc["control_reextraction"]
    raw["network_bytes"] = fc["network_bytes"]
    checks["raw"] = raw
    conds = {}
    for split in dp.SPLITS:
        qxy, mxy, q = sets_for("08-20", split)
        ref = json.loads((dp.NATIVE / "evaluation_sets.json").read_text())[split]
        for tag, (method, key, cache) in CONDS.items():
            name = f"{split}_{tag}"
            cond = OUT / name
            if not (cond / "report.json").exists():
                continue
            rep = json.loads((cond / "report.json").read_text())
            pairs = np.load(cond / "pair_dists.npy")
            with np.load(cond / "tirings.npz") as z:
                maps_t, sub_t, picks = z["maps"], z["query_subset"], list(z["query_subset_index"])
            cached, raw_ok = [], 0
            for e in ref["map"]:
                with np.load(cache / (e["raw_path"].replace("/", "_").replace(".bin", ".npz"))) as c:
                    cached.append(c[key])
                    raw_ok += int(str(c["raw_sha256"]) == digest(dp.NATIVE / "native_data" / e["raw_path"]))
            cached = np.stack(cached)
            err, top_ok = 0.0, 0
            mn = norm(maps_t)
            for j, i in enumerate(picks):
                d_abs, _ = direct_dists(norm(sub_t[j]), mn)
                err = max(err, float(np.abs(d_abs - pairs[i]).max()))
                top_ok += int(d_abs.argmin() == pairs[i].argmin())
            printed_r1 = np.array(rep["printed_recalls_k"])[:, 0]
            printed_f1 = np.array(rep["printed_f1s"])
            rec_r1, rec_f1 = [], []
            for r in dp.RADII:
                m = metrics(pairs, qxy, mxy, r)
                rec_r1.append(m["R@1_all_queries"])
                rec_f1.append(m["_f1s"])
            conds[name] = {
                "report_after_protocol": rep["created_utc"] > proto["frozen_utc"],
                "map_tiring_vs_existing_cache_bitwise": bool(maps_t.shape == cached.shape and (maps_t == cached).all()),
                "map_tiring_max_abs_diff": float(np.abs(maps_t - cached).max()) if maps_t.shape == cached.shape else None,
                "map_cache_raw_sha256_matches": f"{raw_ok}/{len(ref['map'])}",
                "subset_queries": len(picks), "direct_vs_official_pair_dist_max_abs_diff": err,
                "subset_top1_agreement": f"{top_ok}/{len(picks)}",
                "printed_vs_recount_R@1_max_abs_err": float(np.abs(printed_r1 - np.array(rec_r1)).max()),
                "printed_vs_recount_F1_max_abs_err": float(np.abs(printed_f1 - np.stack(rec_f1)).max()),
                "pair_dists_finite": bool(np.isfinite(pairs).all()), "shape": list(pairs.shape),
                "queries_match_query_sets": pairs.shape[0] == len(q)}
            print(name, json.dumps(conds[name], ensure_ascii=False), flush=True)
    checks["conditions"] = conds
    p = checks["protocol"]
    checks["passed"] = bool(
        all(v for k, v in p.items() if k != "frozen_utc")
        and all(v["timestamps_identical"] and v["xy_max_abs_diff_m"] == 0.0 for v in checks["query_reconstruction"].values())
        and raw["missing"] == 0 and raw["size_match"] == raw["queries_checked"]
        and raw["fetched_sha_match"] + raw["disco_sha_match"] == raw["queries_checked"]
        and raw["control_reextraction"]["all_match_disco_sha256"]
        and "all_occ_official" in conds and "all_feat_maxfix" in conds
        and all(v["report_after_protocol"] and v["map_tiring_vs_existing_cache_bitwise"]
                and v["map_cache_raw_sha256_matches"].split("/")[0] == v["map_cache_raw_sha256_matches"].split("/")[1]
                and v["direct_vs_official_pair_dist_max_abs_diff"] < 1e-4
                and v["printed_vs_recount_R@1_max_abs_err"] == 0.0 and v["printed_vs_recount_F1_max_abs_err"] < 1e-12
                and v["pair_dists_finite"] and v["queries_match_query_sets"] for v in conds.values()))
    checks["script_sha256"] = digest(Path(__file__))
    checks["created_utc"] = dp.utc()
    dp.save_json(OUT / "verification.json", checks)
    return checks


def compatible(n: int) -> int:
    """표II NCLT 9개 R@1 중 k/n의 소수4자리 반올림으로 나올 수 있는 값의 개수."""
    ok = 0
    for v in TABLE_II_NCLT_R1:
        k = int(round(v * n))
        ok += int(any(abs(c / n - v) <= 0.00005 + 1e-12 for c in (k - 1, k, k + 1) if c >= 0))
    return ok


def report() -> None:
    """03-17과 08-20을 같은 함수로 집계해 나란히 보고한다."""
    proto = json.loads((OUT / "protocol.json").read_text())
    ver = json.loads((OUT / "verification.json").read_text())
    cov = json.loads((OUT / "coverage.json").read_text())
    fc = json.loads((OUT / "fetch_complete.json").read_text())
    rows, counts = [], {}
    for split in dp.SPLITS:
        for date in ("03-17", "08-20"):
            qxy, mxy, _ = sets_for(date, split)
            for tag, (method, _, _) in CONDS.items():
                path = (OLD / f"{split}_{tag}" / "pair_dists.npy") if date == "03-17" else (OUT / f"{split}_{tag}" / "pair_dists.npy")
                if not path.exists():
                    continue
                pairs = np.load(path)
                by_r = {}
                for r in dp.RADII:
                    m = metrics(pairs, qxy, mxy, r)
                    m.pop("_f1s")
                    by_r[str(r)] = m
                m10 = by_r["10.0"]
                paper = PAPER[method]
                rows.append({"split": split, "query_date": f"2012-{date}", "map_date": "2012-02-04", "method": method,
                             "pair_dists": str(path.relative_to(REPO)), "pair_dists_sha256": digest(path),
                             "radii": by_r,
                             "gap_vs_paper_10m_pp": {
                                 "R@1_all": 100 * (m10["R@1_all_queries"] - paper["R@1"]),
                                 "R@1_positive_only": 100 * (m10["R@1_positive_only"] - paper["R@1"]),
                                 "official_grid_max_F1": 100 * (m10["official_grid_max_F1"] - paper["F1"]),
                                 "fixed_denominator_max_F1": 100 * (m10["fixed_denominator_max_F1"] - paper["F1"])}})
                counts[f"{split}_{date}"] = {"queries": m10["queries"], "positive_10m": m10["positive_queries"]}
    denom = {k: {"queries": v["queries"], "queries_compatible_of_9": compatible(v["queries"]),
                 "positive_10m": v["positive_10m"], "positive_compatible_of_9": compatible(v["positive_10m"])}
             for k, v in counts.items()}
    old = json.loads((OLD / "summary.json").read_text())["rows"]
    consistency = []
    for r in rows:
        if r["query_date"] != "2012-03-17":
            continue
        name = f"{r['split']}_{'occ_official' if r['method'] == 'RING' else 'feat_maxfix'}"
        prev = next(x for x in old if x["condition"] == name)
        consistency.append({"condition": name, "existing_official_R@1_10m": prev["official_R@1_10m"],
                            "recomputed": r["radii"]["10.0"]["R@1_all_queries"],
                            "equal": prev["official_R@1_10m"] == r["radii"]["10.0"]["R@1_all_queries"]})
    value = {
        "purpose": proto["purpose"], "protocol_sha256": digest(OUT / "protocol.json"),
        "verification_passed": ver["passed"],
        "inputs": {"query_counts": {s: cov["splits"][s] for s in dp.SPLITS},
                   "fetch": {k: fc[k] for k in ("fetched", "network_bytes", "planned_network_bytes", "elapsed_s",
                                                "block_retries", "etag", "control_reextraction")}},
        "rows": rows, "existing_03-17_consistency": consistency,
        "denominator_compatibility_table_II": {
            "assumption": "표II NCLT 9개 R@1이 같은 질의 집합의 정수 성공 수/분모를 소수4자리로 반올림한 값이라는 가정(기존 진단과 같음)",
            "existing_min_compatible": 989, "counts": denom,
            "note": "분모 일치 여부는 표본 선택에 쓰지 않는 사후 점검이다"},
        "paper_table_II_nclt": PAPER, "paper_evidence": proto["paper_evidence"],
        "created_utc": dp.utc(), "script_sha256": digest(Path(__file__)),
    }
    dp.save_json(OUT / "report.json", value)
    print(json.dumps([{k: r[k] for k in ("split", "query_date", "method")} | {
        "R1": r["radii"]["10.0"]["R@1_all_queries"], "R1pos": r["radii"]["10.0"]["R@1_positive_only"],
        "F1": r["radii"]["10.0"]["official_grid_max_F1"], "F1fixed": r["radii"]["10.0"]["fixed_denominator_max_F1"]}
        for r in rows], ensure_ascii=False, indent=1), flush=True)
    print(json.dumps({"denominators": denom, "consistency": consistency}, ensure_ascii=False), flush=True)


def pct(v: float) -> str:
    return f"{100 * v:.2f}"


def text(narrative: Path) -> None:
    """report.json·verification.json·해석 파일(narrative.json)로 report_ko.txt를 쓴다. 수치는 JSON에서만 읽는다."""
    rep = json.loads((OUT / "report.json").read_text())
    ver = json.loads((OUT / "verification.json").read_text())
    cov = json.loads((OUT / "coverage.json").read_text())
    nar = json.loads(narrative.read_text())
    L = ["RING·RING++ NCLT 날짜 쌍 진단: 08-20→02-04와 03-17→02-04 (공개 생성기·공식 점수)", "",
         f"작성 {rep['created_utc']}, protocol 고정 {ver['protocol']['frozen_utc']}, 검산 통과 {ver['passed']}", "",
         "[범위]", nar["scope"], "", "[논문 근거와 증거 수준]"] + [f"- {x}" for x in nar["evidence"]] + ["", "[입력]"]
    for s in dp.SPLITS:
        c = cov["splits"][s]
        L.append(f"- {s}: 08-20 5m 질의 {c['query_before_20m_filter']} → 20m 필터 {c['queries']}(10m 양성 "
                 f"{c['positive_queries_10m']}), 지도 02-04 {c['maps']}. DiSCO 보유분 {c['present_in_disco_selection']}, "
                 f"수신 {c['missing']}. 독립 재구성 시각 일치 {c['independent_reconstruction']['timestamps_identical']}")
    f = rep["inputs"]["fetch"]
    L.append(f"- 수신: {f['fetched']}장, 네트워크 {f['network_bytes'] / 1e9:.2f}GB(계획 {f['planned_network_bytes'] / 1e9:.2f}GB), "
             f"{f['elapsed_s'] / 60:.1f}분, ETag {f['etag']}, 보유 스캔 재추출 대조 {f['control_reextraction']['scans']}장 "
             f"일치 {f['control_reextraction']['all_match_disco_sha256']}")
    L += ["", "[결과: 10m 정답, 공식 점수(circorr2 |corr| 합), %]",
          "split  질의날짜    방법            질의  양성  R@1전체  R@1양성  공식격자maxF1  고정분모maxF1  공식AUC  | 논문 R@1/F1"]
    for r in rep["rows"]:
        m = r["radii"]["10.0"]
        p = rep["paper_table_II_nclt"][r["method"]]
        L.append(f"{r['split']:<6} {r['query_date']:<11} {r['method']:<14} {m['queries']:>5} {m['positive_queries']:>5} "
                 f"{pct(m['R@1_all_queries']):>8} {pct(m['R@1_positive_only']):>8} {pct(m['official_grid_max_F1']):>13} "
                 f"{pct(m['fixed_denominator_max_F1']):>13} {pct(m['official_PR_AUC']):>8}  | {pct(p['R@1'])}/{pct(p['F1'])}")
    L += ["", "[논문과의 차이(all, 10m, %p)]"]
    for r in rep["rows"]:
        if r["split"] == "all":
            g = r["gap_vs_paper_10m_pp"]
            L.append(f"- {r['query_date']} {r['method']}: R@1전체 {g['R@1_all']:+.2f}, R@1양성 {g['R@1_positive_only']:+.2f}, "
                     f"공식격자F1 {g['official_grid_max_F1']:+.2f}, 고정분모F1 {g['fixed_denominator_max_F1']:+.2f}")
    L += ["", "[검산]"] + [f"- {x}" for x in nar["verification"]]
    d = rep["denominator_compatibility_table_II"]["counts"]
    L.append("- 표II 9개 R@1 분모 호환(사후 점검, 표본 선택 없음): " + ", ".join(
        f"{k} 질의{v['queries']}→{v['queries_compatible_of_9']}/9·양성{v['positive_10m']}→{v['positive_compatible_of_9']}/9"
        for k, v in d.items()) + "; 최소 호환 989는 9/9")
    L += ["", "[해석]"] + [f"- {x}" for x in nar["interpretation"]]
    L += ["", "[남은 차이·미확정]"] + [f"- {x}" for x in nar["unresolved"]]
    L += ["", "[재실행]"] + [f"- {x}" for x in nar["commands"]]
    (OUT / "report_ko.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["verify", "report", "text"])
    parser.add_argument("--narrative", type=Path, default=OUT / "narrative.json")
    args = parser.parse_args()
    if args.stage == "verify":
        verify()
    elif args.stage == "report":
        report()
    else:
        text(args.narrative)
