#!/usr/bin/env python3
"""공개/재생성 HeLiOS 입력을 같은 가중치·질의에서 대조한다."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

from _review_native_helios_regen_eval import OUT, RELEASED, MODELS
import _review_native_helios as native


def main() -> None:
    """독립 검산을 완료한 원장만 읽고 부분 센서 범위를 명시한다."""
    verified = json.loads((OUT / "independent_verification.json").read_text())
    assert len(verified["rows"]) == 40 and all(r["passed"] for r in verified["rows"])
    rows = []
    comparisons = []
    hashes = {}
    for db in native.DATABASES:
        for sequence in [f"Roundabout{i:02d}-Velodyne" for i in (1, 2, 3)]:
            source = {}
            for model in MODELS:
                path = OUT / model / "native_pairs" / f"{db}__{sequence}.npz"
                hashes[str(path.relative_to(OUT))] = native.sha(path)
                source[model] = dict(np.load(path))
            common = np.logical_and.reduce([x["evaluated"] for x in source.values()])
            assert common.sum() > 0
            recalls = {k: float(v["first_correct"][common].mean()) for k, v in source.items()}
            scores = [recalls[k] for k in MODELS[1:]]
            rows.append({"database": db, "query": sequence, "common_positive_queries": int(common.sum()), "author_r1": recalls["author"], "v2_mean_r1": float(np.mean(scores)), "v2_sample_std_r1": float(np.std(scores, ddof=1)), "v2_r1": scores})
            if sequence == "Roundabout03-Velodyne":
                for model in MODELS:
                    path = RELEASED / model / "native_pairs" / f"{db}__{sequence}.npz"
                    prior = dict(np.load(path))
                    hashes[str(path.relative_to(RELEASED.parent))] = native.sha(path)
                    current = source[model]
                    paired = prior["evaluated"] & current["evaluated"]
                    assert paired.sum() == prior["evaluated"].sum() == current["evaluated"].sum()
                    a, b = prior["first_correct"][paired], current["first_correct"][paired]
                    comparisons.append({"database": db, "model": model, "paired_queries": int(paired.sum()), "released_r1": float(a.mean()), "regenerated_r1": float(b.mean()), "delta_r1_pp": float((b.mean()-a.mean())*100), "correct_to_wrong": int(np.sum(a & ~b)), "wrong_to_correct": int(np.sum(~a & b)), "top1_index_changes": int(np.sum(prior["indices"][paired,0] != current["indices"][paired,0]))})
    report = {"complete": True, "rows": rows, "released_round03_paired_comparisons": comparisons, "source_sha256": hashes, "scope": "공개 지도2개 × 공식 재생성 Velodyne 질의3개. Round03만 공개 가공입력과 짝지은 대조. Round01/02는 공개 archive 부족을 원자료로 보충. 전체4센서·24쌍 논문 표를 대체하지 않음. v2에서는 이곳이 검증 장소이므로 독립 시험 성능이 아님.", "fixed_weights": "original shared-data choice + author; native 결과로 선택하지 않음", "input_difference": "공식 toolbox의 시간 기반 무작위 부족점 채움 때문에 공개파일과 완전히 같지 않음; Round03은 입력 전체 해시·점 대응 대조 별도 기록", "verification_sha256": native.sha(OUT / "independent_verification.json")}
    native.write(OUT / "comparison_report.json", report)
    with (OUT / "comparison_rows.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [r"\paragraph{Regenerating unavailable Velodyne inputs.}", r"We additionally process Roundabout01--03 Velodyne scans with the unchanged official toolbox and the released query timestamps, keeping both public databases and all model weights fixed. The toolbox randomly fills points after voxel downsampling, so regenerated scans are not identical to the released files. On Roundabout03, the mean matching XYZ prefix is 99.41\%; we compare retrieval on exactly the same positive queries below. Roundabout is a validation location for our shared-data models, and these six pairs do not replace the full 24-pair multi-sensor benchmark.", r"\begin{center}\begin{tabular}{llrr}", r"Database & Velodyne query & Author & Shared-data (3 seeds) \\", r"\hline"]
    for row in rows:
        lines.append(f"{row['database'].split('-')[-1]} & {row['query'].split('-')[0]} & {100*row['author_r1']:.2f} & ${100*row['v2_mean_r1']:.2f}\\pm{100*row['v2_sample_std_r1']:.2f}$" + r" \\")
    lines.extend([r"\end{tabular}\end{center}", "Across the eight matched Roundabout03 model/database comparisons, the regeneration-induced R@1 change ranges from " + f"{min(r['delta_r1_pp'] for r in comparisons):+.2f} to {max(r['delta_r1_pp'] for r in comparisons):+.2f}" + r" percentage points. This isolates preprocessing variation while retaining the same weights and query labels."])
    (OUT.parent / "paper_proposal/helios_regeneration_addendum.tex").write_text("\n".join(lines)+"\n")
    print(json.dumps({"rows": len(rows), "paired_comparisons": len(comparisons), "delta_range_pp": [min(r['delta_r1_pp'] for r in comparisons), max(r['delta_r1_pp'] for r in comparisons)]}))


if __name__ == "__main__":
    main()
