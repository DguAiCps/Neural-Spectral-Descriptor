#!/usr/bin/env python3
"""중심항 진단의 독립 코사인 검산과 계열별 집계."""
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_center_information as run
from _pdv2_no_context_ablation import sha256, write_json

FEATURES = tuple(f"l{layer}_{name}" for layer in (1, 2)
                 for name in ("input", "center", "average", "difference", "post")) + ("context", "raw", "key")


def summarize(rows: dict) -> dict:
    """통합 표본과 시퀀스→계열 등비중 평균을 구분한다."""
    def one(values):
        result = {"n": len(values["triplets"]), "features": {}, "comparisons": {}}
        for name in FEATURES:
            valid = values[name+"_valid"]
            margin = values[name+"_margin"][valid]
            result["features"][name] = {"valid": int(valid.sum()),
                "accuracy": float(np.mean(margin > 0)) if len(margin) else None,
                "margin": float(margin.mean()) if len(margin) else None,
                "positive": float(values[name+"_positive"][valid].mean()) if len(margin) else None,
                "negative": float(values[name+"_negative"][valid].mean()) if len(margin) else None}
        for layer in (1, 2):
            d = f"l{layer}_difference"
            for other in ("center", "average", "input", "post"):
                a = f"l{layer}_{other}"
                valid = values[d+"_valid"] & values[a+"_valid"]
                dm, am = values[d+"_margin"][valid], values[a+"_margin"][valid]
                result["comparisons"][a+"_minus_difference"] = {
                    "valid": int(valid.sum()), "margin": float(np.mean(am-dm)) if len(am) else None,
                    "accuracy": float(np.mean(am > 0)-np.mean(dm > 0)) if len(am) else None,
                    "restored": int(np.sum((am > 0) & (dm <= 0))),
                    "lost": int(np.sum((am <= 0) & (dm > 0)))}
        return result

    def average(items):
        items = [x for x in items if x["n"]]
        if not items:
            return {"n": 0, "features": {}, "comparisons": {}}
        return {"n": sum(x["n"] for x in items),
            "features": {name: {field: float(np.mean([x["features"][name][field] for x in items
                if x["features"][name][field] is not None])) for field in ("accuracy", "margin", "positive", "negative")}
                for name in FEATURES},
            "comparisons": {name: {field: float(np.mean([x["comparisons"][name][field] for x in items
                if x["comparisons"][name][field] is not None])) for field in ("accuracy", "margin")}
                for name in items[0]["comparisons"]}}

    per_sequence = {seq: one(x) for seq, x in rows.items()}
    families = {f: average([x for seq, x in per_sequence.items() if seq.startswith(f)])
                for f in ("kitti", "helipr", "mulran")}
    pooled = one({k: np.concatenate([r[k] for r in rows.values()]) for k in next(iter(rows.values()))})
    return {"pooled": pooled, "sequences": per_sequence, "families": families,
            "macro": average(list(families.values()))}


def subset(rows: dict, selector) -> dict:
    """동일 행을 모든 특징에 적용한다."""
    return {seq: {k: v[selector(r)] for k, v in r.items()} for seq, r in rows.items()}


def main() -> None:
    """저장된 모든 통계와 표본 원벡터를 검산한 뒤 보고서를 만든다."""
    out = run.OUT
    spec = run.previous.read(out/"protocol.json")
    done = run.previous.read(out/"measurement_complete.json")
    pairs = run.previous.read(out/"pairs_frozen.json")
    assert done["protocol_sha256"] == pairs["protocol_sha256"] == sha256(out/"protocol.json")
    sources = {**spec["sources"], **done["sources"], **pairs["sources"]}
    for p, h in sources.items():
        assert sha256(REPO/p) == h, p
    cosine_checks = 0
    cosine_error = 0.
    bundles = 0
    for path in sorted(out.glob("*.npz")):
        if path.name.startswith(("pairs_", "audit_")):
            continue
        x = run.run.exp.arrays(path)
        audit = run.run.exp.arrays(path.with_name("audit_"+path.name))
        for name in FEATURES:
            np.testing.assert_allclose(x[name+"_positive"]-x[name+"_negative"], x[name+"_margin"], atol=1e-14, rtol=0)
            assert all(np.isfinite(x[name+"_"+field]).all() for field in ("positive", "negative", "margin"))
            for index, (q, p, n) in zip(audit["row"], audit[name].astype(float)):
                def cosine(a, b):
                    return np.dot(a, b)/max(float(np.sqrt(np.dot(a, a))), 1e-12)/max(float(np.sqrt(np.dot(b, b))), 1e-12)
                positive, negative = cosine(q, p), cosine(q, n)
                cosine_error = max(cosine_error, abs(positive-x[name+"_positive"][index]), abs(negative-x[name+"_negative"][index]))
                assert abs(positive-x[name+"_positive"][index]) < 1e-12
                assert abs(negative-x[name+"_negative"][index]) < 1e-12
                expected_valid = all(np.linalg.norm(v) > 1e-12 for v in (q, p, n))
                assert expected_valid == bool(x[name+"_valid"][index])
                cosine_checks += 2
        bundles += 1
    pair_checks = 0
    val_sequences = run.previous.protocol()["val"]
    for seq in val_sequences:
        plan = run.run.exp.arrays(out/("pairs_"+seq.replace("/", "_")+".npz"))
        _, _, paths, _ = run.run.exp.old.base.inputs("val", seq)
        cache = run.run.exp.arrays(paths["cache"])
        xyz = cache["poses_gt"][:, :3, 3].astype(float)
        raw = (cache["descriptors"]*np.load(run.run.exp.old.base.RVEC)).astype(float)
        raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
        q, p, n = plan["triplets"].T
        assert bool((p <= q-30).all() and (n <= q-30).all())
        assert bool((np.linalg.norm(xyz[q]-xyz[p], axis=1) <= 5).all())
        assert bool((np.linalg.norm(xyz[q]-xyz[n], axis=1) > 25).all())
        for idx in np.unique(np.linspace(0, len(q)-1, min(32, len(q))).astype(int)):
            i = q[idx]
            distances = np.linalg.norm(xyz[:i-29]-xyz[i], axis=1)
            assert int(np.argmin(distances)) == p[idx]
            scores = raw[:i-29] @ raw[i]
            scores[distances <= 25] = -np.inf
            assert int(np.argmax(scores)) == n[idx]
            pair_checks += 1
    val, train, cases, csv_rows = {}, {}, {}, []
    for arm in run.ARMS:
        rows = {seq: run.run.exp.arrays(out/("val_"+arm+"_"+seq.replace("/", "_")+".npz")) for seq in val_sequences}
        for seq, r in rows.items():
            plan = run.run.exp.arrays(out/("pairs_"+seq.replace("/", "_")+".npz"))
            for field in ("triplets", "group", *(a+"_correct" for a in run.ARMS)):
                np.testing.assert_array_equal(r[field], plan[field])
        val[arm] = {"all": summarize(rows)}
        for group in (0, 1, 2):
            val[arm][f"direction_group_{group}"] = summarize(subset(rows, lambda r: r["group"] == group))
        for case_arm, ref in (("diff_tanh", "standard_tanh"), ("diff", "standard")):
            val[arm]["lost_"+case_arm] = summarize(subset(rows, lambda r: r[ref+"_correct"] & ~r[case_arm+"_correct"]))
            val[arm]["kept_"+case_arm] = summarize(subset(rows, lambda r: r[ref+"_correct"] & r[case_arm+"_correct"]))
            case_rows = {seq: run.run.exp.arrays(out/("cases_"+case_arm+"_"+arm+"_"+seq.replace("/", "_")+".npz")) for seq in val_sequences}
            for seq, r in case_rows.items():
                plan = run.run.exp.arrays(out/("pairs_"+seq.replace("/", "_")+".npz"))
                np.testing.assert_array_equal(r["triplets"], plan["cases_"+case_arm])
                np.testing.assert_array_equal(r["negative_distance"], plan["cases_"+case_arm+"_distance"])
            cases.setdefault(case_arm, {})[arm] = summarize(case_rows)
            cases[case_arm][arm]["gray_negatives"] = sum(int((r["negative_distance"] <= 25).sum()) for r in case_rows.values())
        tr = run.run.exp.arrays(out/("train_"+arm+".npz"))
        tp = run.run.exp.arrays(out/"pairs_train.npz")
        np.testing.assert_array_equal(tr["triplets"], tp["triplets"])
        train[arm] = summarize({("kitti/" if i < 4 else "helipr/")+str(i):
            {k: v[tr["sequence"] == i] for k, v in tr.items()} for i in np.unique(tr["sequence"])})
        for scope, report in (("train", train[arm]), ("val", val[arm]["all"]), ("lost_cases", cases["diff_tanh"][arm])):
            for name in FEATURES:
                for aggregation, stats in (("pooled", report["pooled"]), ("macro", report["macro"]), *report["families"].items()):
                    if stats["n"]:
                        csv_rows.append({"scope": scope, "arm": arm, "aggregation": aggregation, "feature": name,
                            "n": stats["n"], **{k: v for k, v in stats["features"][name].items() if k != "valid"}})
    old = run.previous.read(run.previous.OUT/"summary.json")
    write_json(out/"summary.json", {"train": train, "val": val, "cases": cases,
        "final_metrics_unchanged": {a: old["per_seed"]["1"][a]["selected"] for a in run.ARMS},
        "test_used": False, "new_training": False,
        "caveat": "고정 삼중쌍에서 중간 특징의 순위 진단이며 최종 AP나 정보량이 아니다."})
    with (out/"center_information.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    sources.update({str(p.relative_to(REPO)): sha256(p) for p in
        (Path(__file__), out/"summary.json", out/"center_information.csv", out/"protocol.json", out/"pairs_frozen.json", out/"measurement_complete.json")})
    write_json(out/"completion_validation.json", {"created_unix": time.time(), "sources": sources,
        "bundles": bundles, "independent_cosine_checks": cosine_checks, "max_cosine_error": cosine_error,
        "independent_exact_pair_checks": pair_checks, "all_saved_pairs_geometry_checked": True,
        **{k: done[k] for k in ("max_output_reconstruction_error", "max_saved_key_error", "prefix_checks", "max_prefix_feature_error")}})
    print("PRIMARY_VAL", val["diff_tanh"]["all"]["macro"], flush=True)
    print("PRIMARY_CASES", cases["diff_tanh"]["diff_tanh"]["pooled"], flush=True)
    print("AUDIT", bundles, cosine_checks, cosine_error, pair_checks, flush=True)


if __name__ == "__main__":
    main()
