#!/usr/bin/env python3
"""완료된 BEVPlace++ 검증의 전체 과거 검색 표본과 NSD 집계를 재검사한다."""
from __future__ import annotations

from pathlib import Path
import time

import faiss
import numpy as np

import _pdv2_bevplace_val_odom as base
from _pdv2_bevplace_val_odom_fast import amendment


def main() -> None:
    """새 선택 없이 동일 질의·지표·CPU 전체 검색을 확인한다."""
    started = time.time()
    base.prepare()
    amendment()
    assert base.read(base.OUT / "status.json")["status"] == "complete"
    assert base.read(base.OUT / "checkpoint_acquisition.json")["matching_test_ledgers"] == 46
    summary = base.read(base.OUT / "summary.json")
    faiss.omp_set_num_threads(1)
    references = {seed: {method: {} for method in ("raw", "arc_unique_b100")} for seed in (1, 2, 3)}
    audits, sources = {}, {}
    correct_overlap = {seed: {} for seed in (1, 2, 3)}
    for ordinal, (sensor, seq) in enumerate(base.VAL):
        name = sensor + "/" + seq
        path = base.OUT / f"dump_bevplace_{sensor}_{seq}.npz"
        features = base.OUT / f"descriptors_{sensor}_{seq}.npy"
        saved = base.read(base.OUT / f"{sensor}_{seq}.json")
        assert base.bev.sha256(features) == saved["descriptors_sha256"]
        with np.load(path) as z:
            arrays = {key: z[key] for key in z.files if key != "meta"}
        vectors = np.load(features, mmap_mode="r")
        rng = np.random.default_rng(20261018 + ordinal)
        count = len(arrays["query_idx"])
        rows = np.unique(np.r_[np.linspace(0, count - 1, 16, dtype=int),
                               rng.choice(count, 16, replace=False),
                               np.argsort(arrays["score_negl2"])[-4:]])
        index = faiss.IndexFlatL2(vectors.shape[1])
        for row in sorted(rows):
            query = int(arrays["query_idx"][row])
            end = query - 29
            index.add(vectors[index.ntotal:end])
            distance, candidate = index.search(vectors[query:query + 1], 1)
            assert candidate[0, 0] == arrays["j_star"][row], (name, query)
            exact = -float(np.sqrt(max(float(distance[0, 0]), 0.)))
            assert exact == arrays["score_negl2"][row], (name, query)
        audits[name] = {"queries": len(rows), "candidate_and_score_exact": True}
        for seed in (1, 2, 3):
            with np.load(base.NSD / f"s{seed}/standard_{sensor}_{seq}.npz") as z:
                for key in ("query_idx", "min_past_gt_dist"):
                    np.testing.assert_array_equal(z[key], arrays[key])
                positive = arrays["min_past_gt_dist"] <= 5
                nsd_correct = z["top1_geo_dist"][positive] <= 5
                bev_correct = arrays["top1_geo_dist"][positive] <= 5
                overlap = {
                    "both_correct": int((nsd_correct & bev_correct).sum()),
                    "nsd_only": int((nsd_correct & ~bev_correct).sum()),
                    "bevplace_only": int((~nsd_correct & bev_correct).sum()),
                    "both_wrong": int((~nsd_correct & ~bev_correct).sum()),
                }
                assert sum(overlap.values()) == int(positive.sum())
                correct_overlap[seed][name] = overlap
                for method in references[seed]:
                    data = {key: z[key] for key in ("min_past_gt_dist", "top1_geo_dist")}
                    data["score_negl2"] = z[method]
                    references[seed][method][name] = base.metrics(data)
        sources[str(path.relative_to(base.REPO))] = base.bev.sha256(path)
        print("FULL_CPU_CHECK", name, len(rows), flush=True)
    family_comparison = {}
    for method in references[1]:
        per_seed = {}
        for seed in (1, 2, 3):
            families = {}
            for family in ("kitti", "helipr", "mulran"):
                rows = [r for name, r in references[seed][method].items()
                        if name.startswith(family) and r["n_pos"]]
                families[family] = {metric: float(np.mean([r[metric] for r in rows])) for metric in base.METRICS}
            per_seed[seed] = families
        family_comparison[method] = {}
        for family in ("kitti", "helipr", "mulran"):
            family_comparison[method][family] = {
                metric: float(np.mean([per_seed[s][family][metric] for s in (1, 2, 3)]))
                for metric in base.METRICS
            }
        for metric in base.METRICS:
            values = [np.mean([per_seed[s][f][metric] for f in ("kitti", "helipr", "mulran")]) for s in (1, 2, 3)]
            expected = summary["nsd_three_seed"][method][metric]
            np.testing.assert_allclose([np.mean(values), np.std(values, ddof=1)],
                                       [expected["mean"], expected["std"]], atol=1e-12, rtol=0)
    family_delta = {
        family: {metric: 100 * (family_comparison["arc_unique_b100"][family][metric]
                               - summary["bevplace"]["families"][family][metric])
                 for metric in base.METRICS}
        for family in ("kitti", "helipr", "mulran")
    }
    base.write(base.OUT / "completion_report_validation.json", {
        "status": "passed", "runtime_s": time.time() - started,
        "script_sha256": base.bev.sha256(Path(__file__)),
        "full_cpu_reference": audits, "queries": sum(r["queries"] for r in audits.values()),
        "nsd_same_queries_seeds": [1, 2, 3], "nsd_mean_and_std_recomputed": True,
        "nsd_family_metrics": family_comparison, "nsd_sequence_minus_bevplace_family_pp": family_delta,
        "positive_top1_overlap_counts": correct_overlap,
        "ledgers_sha256": sources, "summary_sha256": base.bev.sha256(base.OUT / "summary.json"),
        "test_used": False,
    })
    print("VERIFIED", family_delta, flush=True)


if __name__ == "__main__":
    main()
