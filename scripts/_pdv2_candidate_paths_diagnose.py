#!/usr/bin/env python3
"""후보 경로의 최초 오수락 경계와 시드별 정답 손익을 요약한다."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_candidate_paths as run
from _pdv2_candidate_paths_finish import scalar_score
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """모든 최초 오답의 최종 점수와 R@100P 경계를 별도로 확인한다."""
    spec = run.protocol()
    summary = run.exp.read(run.OUT / "summary.json")
    diagnostic = run.exp.read(run.OUT / "diagnostics.json")["per_seed"]
    sources, boundary, pooled = {}, {}, {}
    checked = 0
    max_error = 0.
    for seed, models in summary["per_seed"].items():
        boundary[seed], pooled[seed] = {}, {}
        conf = run.exp.read(run.exp.old.base.BASE / f"s{seed}/confidence_selection.json")["ops"][run.exp.OP]
        for sequence in spec["val"]:
            sensor, seq, paths, _ = run.exp.old.base.inputs("val", sequence)
            cache = run.exp.arrays(paths["cache"])
            xyz = cache["poses"][:, :3, 3]
            travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
            with np.load(paths["spec"]) as z:
                bank = run.exp.old.base.bank_for(run.exp.OP, z)
            boundary[seed][sequence] = {}
            for model in models:
                boundary[seed][sequence][model] = {}
                for arm in run.CFG["arms"]:
                    p = run.OUT / f"s{seed}/{model}_{arm}_{sequence.replace('/', '_')}.npz"
                    a = run.exp.arrays(p)
                    sources[str(p.relative_to(REPO))] = sha256(p)
                    positive = a["min_past_gt_dist"] <= 5
                    eligible = positive | (a["min_past_gt_dist"] > 25)
                    correct = positive & (a["top1_geo_dist"] <= 5)
                    wrong = eligible & np.isfinite(a["score"]) & ~correct
                    if wrong.any():
                        index = np.flatnonzero(wrong)[np.argmax(a["score"][wrong])]
                        q, j = int(a["query_idx"][index]), int(a["j_star"][index])
                        value = scalar_score(bank, travel, q, j, conf["decision"], float(a["raw"][index]))
                        max_error = max(max_error, abs(value-a["score"][index]))
                        assert abs(value-a["score"][index]) < 2e-7
                        count = int((correct & (a["score"] > a["score"][index])).sum())
                        first_wrong = {"q": q, "j": j, "score": float(a["score"][index]),
                                       "distance": float(a["top1_geo_dist"][index]),
                                       "positive": bool(positive[index])}
                    else:
                        count = int((correct & np.isfinite(a["score"])).sum())
                        first_wrong = None
                    expected = run.exp.metrics(a["score"], a)["R@100P"]
                    assert abs(count/max(positive.sum(), 1)-expected) < 1e-12
                    boundary[seed][sequence][model][arm] = {
                        "first_wrong": first_wrong, "R100_correct": count,
                        "positive": int(positive.sum())}
                    checked += 1
        for model in models:
            rows = list(diagnostic[seed][model]["sequences"].values())
            pooled[seed][model] = {
                "positive": sum(r["positive"] for r in rows),
                "union_reachable": sum(r["union_reachable"] for r in rows),
                "shortlist_reachable": sum(r["shortlist_reachable"] for r in rows),
                "arms": {}}
            for arm in run.CFG["arms"]:
                metrics = ("correct", "gains", "losses", "null", "positive_null", "winner_changes")
                pooled[seed][model]["arms"][arm] = {
                    **{k: sum(r["arms"][arm][k] for r in rows) for k in metrics},
                    "directions": {str(g): {k: sum(r["arms"][arm]["directions"][str(g)][k] for r in rows)
                                           for k in ("n", "correct", "gains", "losses")} for g in (0, 1, 2)}}
    mean_std = {}
    for model in run.CFG["models"]:
        mean_std[model] = {}
        for arm in run.CFG["arms"]:
            mean_std[model][arm] = {
                m: {"mean": float(np.mean([s[model][arm]["macro"][m] for s in summary["per_seed"].values()])),
                    "std": float(np.std([s[model][arm]["macro"][m] for s in summary["per_seed"].values()], ddof=1))}
                for m in run.exp.METRICS}
    for p in (Path(__file__), REPO / "scripts/_pdv2_candidate_paths_finish.py",
              run.OUT / "summary.json", run.OUT / "diagnostics.json"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    write_json(run.OUT / "high_precision_diagnosis.json", {
        "status": "passed", "boundary_checks": checked, "scalar_score_max_error": max_error,
        "boundary": boundary, "pooled": pooled, "mean_std": mean_std, "sources": sources,
        "scope": "정답 수는 시퀀스통합이며 AP계열평균과다름. 동일장소의3시드를 독립질의로 해석하지않음.",
        "test_used": False})
    print("BOUNDARIES_VERIFIED", checked, "ERROR", max_error, flush=True)


if __name__ == "__main__":
    main()
