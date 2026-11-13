#!/usr/bin/env python3
"""공개와 재생성 HeLiOS native 원장의 중복을 제거한 범위를 기록한다."""
import json
from pathlib import Path
import time

import numpy as np

import _review_native_helios as native
from _review_native_helios_regen_eval import OUT as REGEN, MODELS


def main() -> None:
    """검산된 쌍만 합치고 완전한6쌍 그룹에서만 공식 평균을 만든다."""
    frozen = native.OUT / "coverage_protocol.json"
    design = json.loads(frozen.read_text())
    assert design["implementation_sha256"] == native.sha(Path(__file__))
    primary = json.loads((native.OUT.parent / "learned_independent_verification.json").read_text())["helios"]
    regenerated = json.loads((REGEN / "independent_verification.json").read_text())["rows"]
    passed = {(r["model"], r["database"], r["query"]): r["passed"] for r in primary}
    new_passed = {(r["model"], r["database"], r["query"]): r["passed"] for r in regenerated}
    rows, missing = [], []
    for db in native.DATABASES:
        for name in native.NAMES:
            public_ready = all(passed.get((m, db, name), False) for m in MODELS)
            new = not public_ready and name in ["Roundabout01-Velodyne", "Roundabout02-Velodyne"]
            source = REGEN if new else native.OUT
            verified = new_passed if new else passed
            paths = {m: source / m / "native_pairs" / f"{db}__{name}.npz" for m in MODELS}
            if not all(p.exists() and verified.get((m, db, name), False) for m, p in paths.items()):
                missing.append({"database": db, "query": name})
                continue
            assert all(verified[m, db, name] for m in MODELS)
            vectors = {m: dict(np.load(p)) for m, p in paths.items()}
            common = np.logical_and.reduce([z["evaluated"] for z in vectors.values()])
            recalls = {m: float(vectors[m]["first_correct"][common].mean()) for m in MODELS}
            official = {m: {"r1": float(vectors[m]["recall_hits"][0] / vectors[m]["evaluated"].sum()),
                            "r5": float(vectors[m]["recall_hits"][4] / vectors[m]["evaluated"].sum()),
                            "positive": int(vectors[m]["evaluated"].sum())} for m in MODELS}
            seeds = [recalls[m] for m in MODELS[1:]]
            rows.append({"database": db, "query": name, "input_source": "official_toolbox_regenerated_raw" if new else "author_released_processed",
                         "paired_positive_queries": int(common.sum()), "author_r1": recalls["author"],
                         "v2_mean_r1": float(np.mean(seeds)), "v2_sample_std_r1": float(np.std(seeds, ddof=1)),
                         "v2_r1": seeds, "official_metrics": official,
                         "predictions": {m: {"path": str(p.relative_to(native.REPO)), "sha256": native.sha(p)} for m, p in paths.items()}})
    assert len(rows) + len(missing) == 24
    groups = []
    published = native.check()["published_table_I"]
    for db in native.DATABASES:
        for group, sensors in [("Narrow", ["Aeva", "Avia"]), ("Wide", ["Ouster", "Velodyne"])]:
            chosen = [r for r in rows if r["database"] == db and r["query"].split("-")[-1] in sensors]
            entry = {"database": db, "group": group, "available_pairs": len(chosen), "required_pairs": 6, "complete": len(chosen) == 6}
            if entry["complete"]:
                entry["official_mean"] = {model: {metric: float(np.mean([r["official_metrics"][model][metric] for r in chosen])) for metric in ["r1", "r5"]} for model in MODELS}
                entry["published_reference"] = dict(zip(["r1", "r5"], published[db][group]))
                entry["regenerated_pairs"] = sum(r["input_source"] == "official_toolbox_regenerated_raw" for r in chosen)
            groups.append(entry)
    output = {"available_native_pairs": len(rows), "required_native_pairs": 24, "all_pairs_complete": len(rows) == 24,
        "released_input_pairs": sum(r["input_source"] == "author_released_processed" for r in rows),
        "new_regenerated_input_pairs": sum(r["input_source"] == "official_toolbox_regenerated_raw" for r in rows), "fixed_models_per_pair": 4,
        "available_model_pair_ledgers": len(rows) * 4, "rows": rows, "missing_pairs": missing, "groups": groups,
        "coverage_protocol_sha256": native.sha(frozen),
        "excluded_duplicates": "재생성Round03 8개 모델원장과 공개DB자체16개 모델원장은 일관성 대조로만 사용. 추가Velodyne지도 확장은 원래benchmark24쌍에 포함 안 함.",
        "interpretation": "출처는 공개가공 우선, 부족한Velodyne01/02만 원시재생성. 그룹6쌍완료전 평균미계산. 공식분모 평균과 네모델공통양성의 쌍별값을 구분. 장소는v2검증셋이며 원래선택모델 진단."}
    target = native.OUT / "coverage_with_regeneration.json"
    if target.exists() and json.loads(target.read_text()) != output:
        history = native.OUT / "coverage_history"
        history.mkdir(exist_ok=True)
        (history / (str(time.time_ns()) + ".json")).write_bytes(target.read_bytes())
    native.write(target, output)
    if output["all_pairs_complete"]:
        lines = [r"\paragraph{Completed HeLiOS group coverage.}",
                 "All 24 original Roundabout database/query pairs are now evaluated with the same four fixed models. " + str(output["released_input_pairs"]) + " pairs use author-released processed scans and " + str(output["new_regenerated_input_pairs"]) + " use official-toolbox regeneration. We average the six official query results in each group; each model retains the released evaluator's denominator. Paired-query results are recorded separately. Roundabout is a validation location for our shared-data models, and author weights do not replace the place-disjoint main comparison.",
                 r"\begin{center}\begin{tabular}{llrrrr}", r"Database & Queries & Paper R@1 & Author R@1 & Shared R@1 & Shared R@5 \\", r"\hline"]
        for group in groups:
            values = group["official_mean"]
            cells = []
            for metric in ["r1", "r5"]:
                seed_values = [values[m][metric] for m in MODELS[1:]]
                cells.append(f"${100*np.mean(seed_values):.2f}\\pm{100*np.std(seed_values, ddof=1):.2f}$")
            lines.append(f"{group['database'].split('-')[-1]} & {group['group']} & {100*group['published_reference']['r1']:.2f} & {100*values['author']['r1']:.2f} & " + " & ".join(cells) + r" \\")
        lines.extend([r"\end{tabular}\end{center}", r"Values are percentages; shared-data entries give the mean and sample standard deviation over three seeds. Input sources and all prediction hashes are retained. Numerical agreement with a paper value does not by itself establish identical preprocessing or training conditions."])
        (native.OUT.parent / "paper_proposal/helios_full_coverage_addendum.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({"pairs": len(rows), "models": len(MODELS), "missing": missing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
