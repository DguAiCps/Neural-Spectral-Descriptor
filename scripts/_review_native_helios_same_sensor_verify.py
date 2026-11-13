#!/usr/bin/env python3
"""새 같은센서 원장을 독립 재계산하고 공통 양성 질의만 비교한다."""
import json
import pickle

import numpy as np

from _review_native_helios_same_sensor import OUT, REGEN, MODELS, DATABASE, QUERIES
from _review_native_learned_verify import independent_pair
import _review_native_helios as native


def main() -> None:
    """모든 정수 hit·혼동행렬 검산 후 세 지도를 같은 질의로 집계한다."""
    metadata = pickle.loads((OUT / "metadata.pickle").read_bytes())
    report = json.loads((OUT / "report.json").read_text())
    assert report["complete"] and len(report["rows"]) == 8
    verified = []
    sources = {}
    for row in report["rows"]:
        path = OUT / row["model"] / (row["query"] + ".npz")
        assert native.sha(path) == row["prediction_sha256"]
        z = np.load(path)
        valid, good, hits, cells, fixed = independent_pair(z["indices"], z["distances"], metadata["queries"][row["query"]], metadata["database"], 0, False, 30, False)
        for a, key in [(valid, "evaluated"), (good, "first_correct"), (hits, "recall_hits"), (cells, "tp_fp_tn_fn")]:
            np.testing.assert_array_equal(a, z[key])
        assert valid.sum() == fixed == row["positive_evaluated"]
        assert float(hits[0] / fixed) == row["r1"]
        assert float(hits[4] / fixed) == row["r5"]
        verified.append({"model": row["model"], "query": row["query"], "passed": True, "positive": fixed})
        sources[str(path.relative_to(OUT.parent))] = native.sha(path)
    rows = []
    for name in QUERIES:
        pairs = {}
        for db in native.DATABASES + [DATABASE]:
            for model in MODELS:
                path = OUT / model / (name + ".npz") if db == DATABASE else REGEN / model / "native_pairs" / f"{db}__{name}.npz"
                pairs[db, model] = dict(np.load(path))
                sources[str(path.relative_to(OUT.parent))] = native.sha(path)
        common = np.logical_and.reduce([z["evaluated"] for z in pairs.values()])
        assert common.sum() > 0
        for db in native.DATABASES + [DATABASE]:
            values = {model: float(pairs[db, model]["first_correct"][common].mean()) for model in MODELS}
            seeds = [values[m] for m in MODELS[1:]]
            rows.append({"query": name, "database": db, "paired_positive_queries": int(common.sum()), "author_r1": values["author"], "v2_mean_r1": float(np.mean(seeds)), "v2_sample_std_r1": float(np.std(seeds, ddof=1)), "v2_r1": seeds,
                         "database_frames": 1692 if db == DATABASE else (1694 if db.endswith("Aeva") else 1693)})
    # Counts are checked against the actual frozen metadata, not assumed equal.
    old = pickle.loads((native.OUT / "data/database.pickle").read_bytes())
    for row in rows:
        row["database_frames"] = len(metadata["database"]) if row["database"] == DATABASE else len(old[native.DATABASES.index(row["database"])])
    output = {"complete": True, "verification": verified, "paired_rows": rows, "source_sha256": sources,
              "protocol_sha256": native.sha(OUT / "protocol.json"),
              "limits": "같은 질의·고정 가중치 대조지만 지도센서마다 프레임과 정답집합은 다르다. 완전한 센서 단독 인과 대조가 아니며 원래 논문 지도/24쌍 표를 대체하지 않는다. Roundabout은 v2 검증 장소다."}
    native.write(OUT / "independent_verification.json", output)
    lines = [r"\paragraph{Database-sensor coverage diagnostic.}",
             r"Using the same fixed model weights and regenerated Velodyne queries, we additionally build a Velodyne database from Roundabout01 and retain the released overlap threshold ($>0.5$). The table restricts all three database conditions and all four models to the same positive-query intersection. This extends the original benchmark, which uses Aeva and Ouster databases; it does not reproduce another published table. Database frame counts and positive sets still differ, so the comparison does not isolate sensor coverage causally. Roundabout remains a validation location for our shared-data models.",
             r"\begin{center}\begin{tabular}{llrrr}", r"Query & Database & Queries & Author & Shared-data \\", r"\hline"]
    for row in rows:
        lines.append(f"{row['query'].split('-')[0]} & {row['database'].split('-')[-1]} & {row['paired_positive_queries']} & {100*row['author_r1']:.2f} & ${100*row['v2_mean_r1']:.2f}\\pm{100*row['v2_sample_std_r1']:.2f}$" + r" \\")
    lines.extend([r"\end{tabular}\end{center}", r"R@1 values are percentages; shared-data entries show the three-seed mean and sample standard deviation. All new predictions were independently recounted from the released overlap labels."])
    (OUT.parent / "paper_proposal/helios_same_sensor_addendum.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
