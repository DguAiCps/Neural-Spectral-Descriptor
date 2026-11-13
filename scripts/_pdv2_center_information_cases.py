#!/usr/bin/env python3
"""중심항 진단 뒤 실제 손실 사례의 검색·융합 단계를 재현한다."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_center_information as run
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_temporal_test_control_finish import scalar_score


def main() -> None:
    """고정 체크포인트의 실제 후보·융합 승자와 최종 판정을 검산한다."""
    spec = run.previous.protocol(True)
    protocol = run.previous.read(run.OUT/"protocol.json")
    completed = run.previous.read(run.OUT/"completion_validation.json")
    for p, h in completed["sources"].items():
        assert sha256(REPO/p) == h, p
    conf = spec["confidence"]["1"]
    weights = np.asarray(conf["tuple"], dtype=np.float32)
    report, checks, error = {}, 0, 0.
    for arm in ("diff_tanh", "diff"):
        records = []
        for seq in spec["val"]:
            plan = run.run.exp.arrays(run.OUT/("pairs_"+seq.replace("/", "_")+".npz"))
            triplets = plan["cases_"+arm]
            if not len(triplets):
                continue
            _, _, paths, _ = run.run.exp.old.base.inputs("val", seq)
            cache = run.run.exp.arrays(paths["cache"])
            with np.load(paths["spec"]) as z:
                np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
                bank = run.run.exp.old.base.bank_for(run.run.exp.OP, z)
            hkey, hfft, hnorm, column = bank
            epoch = protocol["epochs"][arm]
            p = run.run.root(1, arm)/f"epoch{epoch}"/(seq.replace("/", "_")+".npz")
            ledger = run.run.exp.arrays(p)
            key = run.run.exp.arrays(p.with_name("keys_"+p.name))["embedding"]
            xyz = cache["poses"][:, :3, 3]
            travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
            for q, positive, negative in triplets:
                a = run.run.prefix_topk(key, q, q-29, 100)
                b = run.run.prefix_topk(hkey, q, q-29, 100)
                candidates = np.unique(np.r_[a, b])
                raw_a = run.run.align_scores(hfft, hnorm, q, candidates)
                raw_c = run.run.column_align_scores(column, q, candidates)
                parts = np.stack([run.run.minmax(key[candidates]@key[q]),
                    run.run.minmax(hkey[candidates]@hkey[q]), run.run.minmax(raw_a), run.run.minmax(raw_c)])
                score = weights[0]*parts[0]+weights[1]*parts[1]+weights[2]*parts[2]+weights[3]*parts[3]
                winner = int(np.argsort(-score, kind="stable")[0])
                assert int(candidates[winner]) == negative
                idx = np.searchsorted(ledger["query_idx"], q)
                assert ledger["j_star"][idx] == negative
                assert len(candidates) <= 200 and bool((candidates <= q-30).all())
                final = scalar_score(bank, travel, int(q), int(negative), conf["decision"], float(ledger["raw"][idx]))
                error = max(error, abs(final-float(ledger[run.run.METHOD][idx])))
                assert error < 1e-7
                present = bool(positive in candidates)
                margin = float(key[q]@key[positive]-key[q]@key[negative])
                if present:
                    pi = int(np.flatnonzero(candidates == positive)[0])
                    contributions = (weights*(parts[:, pi]-parts[:, winner])).tolist()
                    category = "present_key_prefers_positive" if margin > 0 else "present_key_not_prefer_positive"
                    fused_margin = float(score[pi]-score[winner])
                    assert fused_margin <= 0
                else:
                    contributions, fused_margin, category = [], None, "positive_absent"
                gt = cache["poses_gt"][:, :3, 3]
                records.append({"sequence": seq, "q": int(q), "p": int(positive), "n": int(negative),
                    "category": category, "key_margin": margin, "fused_margin": fused_margin,
                    "weighted_channel_margins": contributions,
                    "any_positive_in_union": bool((np.linalg.norm(gt[candidates]-gt[q], axis=1) <= 5).any())})
                checks += 1
        counts = {c: sum(x["category"] == c for x in records) for c in
                  ("positive_absent", "present_key_prefers_positive", "present_key_not_prefer_positive")}
        report[arm] = {"records": records, "counts": counts,
            "all_positives_absent": sum(not x["any_positive_in_union"] for x in records),
            "families": {f: {c: sum(x["category"] == c and x["sequence"].startswith(f) for x in records)
                              for c in counts} for f in ("kitti", "helipr", "mulran")}}
    write_json(run.OUT/"case_pipeline.json", {"conditions": report, "winner_replay_checks": checks,
        "max_final_score_error": error, "test_used": False,
        "scope": "중심항 주진단을 본 뒤 추가한 사후 설명. p는 표준의 실제 정답 승자이며 모든 정답을 대표하지 않는다. 키가 선호한 p의 융합 역전도 설정 변경의 개선 효과를 증명하지 않는다.",
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in
                    (Path(__file__), run.OUT/"summary.json", run.OUT/"completion_validation.json")}})
    print("CASE_PIPELINE", {a: {k:v for k,v in r.items() if k != "records"} for a,r in report.items()}, checks, error)


if __name__ == "__main__":
    main()
