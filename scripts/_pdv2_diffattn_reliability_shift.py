#!/usr/bin/env python3
"""고정 승자에서 정답·오답 정렬 증거의 감점이 장소에 따라 달라지는지 본다."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import _pdv2_diffattn_reliability as exp
from _pdv2_context_accept_lib import auc


def main() -> None:
    """같은 승자·정렬 증거에 학습 전후 문맥만 교체해 감점률을 비교한다."""
    spec = exp.activate()
    models = {
        "frozen": exp.model(),
        "learned": exp.model(exp.OUT / "attenuate_lr4/epoch8.pth"),
    }
    results = {}
    for role in ("train", "val"):
        per = {}
        for sequence in spec[role]:
            name = sequence.replace("/", "_")
            if role == "train":
                row = exp.run.ARR(exp.OUT / "pools" / name / "pool.npz")
                data = {
                    k: row["plan_" + k]
                    for k in ("query", "candidate", "weight", "peaks")
                }
            else:
                folder = exp.OUT / "frozen/epoch0/val"
                row = exp.run.ARR(folder / (name + ".npz"))
                data = exp.run.ARR(folder / (name + "_evidence.npz"))
            cache, *_ = exp.run.inputs(role, sequence)
            eligible = (row["min_past_gt_dist"] <= 5) | (row["min_past_gt_dist"] > 25)
            correct = (row["min_past_gt_dist"] <= 5) & (row["top1_geo_dist"] <= 5)
            negative = eligible & ~correct
            records = {}
            for label, net in models.items():
                key = exp.run.source.make_keys(
                    net, cache["descriptors"], cache["poses"]
                )
                scores = exp.score_all(key, data, ("attenuate",))
                base, changed = scores["attenuate_0"], scores["attenuate_1"]
                keep = eligible & (base > 1e-8)
                ratio = changed / np.maximum(base, 1e-8)
                records[label] = dict(
                    correct_ratio_mean=(
                        float(ratio[correct].mean()) if correct.any() else None
                    ),
                    wrong_ratio_mean=(
                        float(ratio[negative].mean()) if negative.any() else None
                    ),
                    ratio_correct_auc=auc(ratio[keep], correct[keep]),
                    uniform_AP=exp.run.base.independent_metrics(row, base)["AP"],
                    attenuated_AP=exp.run.base.independent_metrics(row, changed)["AP"],
                )
            per[sequence] = dict(
                correct=int(correct.sum()), wrong=int(negative.sum()), records=records
            )
        # 정답과 오답이 함께 존재하는 시퀀스만 같은 가중치로 비교한다.
        included = [v for v in per.values() if v["correct"] and v["wrong"]]
        mean = {
            label: {
                m: float(np.mean([r["records"][label][m] for r in included]))
                for m in ("correct_ratio_mean", "wrong_ratio_mean", "ratio_correct_auc")
            }
            for label in models
        }
        results[role] = dict(
            sequences=per, positive_sequence_mean=mean, included=len(included)
        )
    exp.run.WRITE(
        exp.OUT / "confidence_shift.json",
        dict(
            results=results,
            scope="강한 감점 학습8에포크의 사후 원인 진단. 학습·검증 모두 초기승자 고정. gamma1 점수/기존점수의 유지비율. 정답·오답을 함께 가진 시퀀스 균등 평균. 학습/검증의 자료 구성 차이를 원인 하나로 확정하지 않는다.",
            source_sha256=exp.run.HASH(Path(__file__)),
            test_used=False,
        ),
    )
    print(
        "SHIFT",
        {role: r["positive_sequence_mean"] for role, r in results.items()},
        flush=True,
    )


if __name__ == "__main__":
    exp.run.configure()
    main()
