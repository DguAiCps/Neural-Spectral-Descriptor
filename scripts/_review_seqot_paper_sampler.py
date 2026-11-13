#!/usr/bin/env python3
"""SeqOT 논문 Eq.13 분모와 비자기 양성 선택의 같은 기준 스캔 대조다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import _review_seqot_sampler_repair as experiment

OUT = experiment.ROOT / "results/pdv2_seqot_ring_repair_20261106/seqot_paper"


def prepare() -> dict:
    """논문 양성과 두 정의 모두 음성인 기존 쌍을 결과 전에 동결한다."""
    import _seqot_official_prep as p
    import _seqot_official_train as s

    if (OUT / "plan.json").exists():
        return json.loads((OUT / "plan.json").read_text())
    sources = [s.StoreSource("train", sensor, seq) for sensor, seq in s.TRAIN]
    valid = lambda d, a: all(sources[d].exists(int(x)) for x in s.centered(a, s.SEQL))
    original = s.anchor_plan(s.load_train_indices(), valid, *s.USE1)
    pools, negative_conflicts, rows, hashes = {}, set(), {}, {}
    for d, (sensor, seq) in enumerate(s.TRAIN):
        i, j, overlap = p.load_explicit(sensor, seq, "paper")
        count = int(np.load(p.label_dir(sensor, seq) / "frames.npz")["poses"].shape[0])
        positive = (overlap > .3) & (i != j)
        ii, jj = i[positive] * 5, j[positive] * 5
        order = np.argsort(ii, kind="stable")
        keys, first = np.unique(ii[order], return_index=True)
        for a, lo, hi in zip(keys, first, np.append(first[1:], len(order))):
            partners = [int(b) for b in jj[order[lo:hi]] if valid(d, int(b))]
            if valid(d, int(a)) and partners:
                pools[d, int(a)] = partners
        pair_keys = i * count + j
        for dd, a, pos, neg in original:
            if dd != d:
                continue
            query = (a // 5) * count + np.asarray(neg) // 5
            slots = np.searchsorted(pair_keys, query)
            exists = slots < len(pair_keys)
            exists[exists] &= pair_keys[slots[exists]] == query[exists]
            # 생략된 쌍은 공식 범위 단축으로 증명한 겹침0이다.
            values = np.zeros(len(neg))
            values[exists] = overlap[slots[exists]]
            if np.any(values >= .3):
                negative_conflicts.add((d, a))
        rows[f"{sensor}/{seq}"] = {"nonself_positive_pairs": int(positive.sum()),
                                     "eligible_anchors": sum(key[0] == d for key in pools),
                                     "negative_conflict_anchors": sum(key[0] == d for key in negative_conflicts)}
        for path in [p.label_dir(sensor, seq) / "normalized_data.npy",
                     *p.label_dir(sensor, seq).glob("ovc_*of*.npz")]:
            hashes[str(path.relative_to(experiment.ROOT))] = experiment.digest(path)
    eligible = [row for row in original if (row[0], row[1]) in pools
                and (row[0], row[1]) not in negative_conflicts]
    rng = np.random.default_rng(20261106)
    ids = rng.permutation(len(eligible))[:4096]
    assert len(ids) == 4096
    control, repair = [], []
    for index in ids:
        d, a, pos, neg = eligible[int(index)]
        choices = pools[d, a]
        new = rng.choice(choices, len(pos), replace=len(choices) < len(pos)).tolist()
        control.append([d, a, pos, neg])
        repair.append([d, a, new, neg])
    result = {"control": control, "repair": repair, "steps": len(control),
              "train_sequences": s.TRAIN, "original_steps": len(original),
              "eligible_steps": len(eligible), "negative_conflict_anchors": len(negative_conflicts),
              "control_self_only_steps": sum(all(x == a for x in pos) for _, a, pos, _ in control),
              "repair_self_only_steps": 0, "sequence_audit": rows,
              "sources_sha256": hashes, "overlap_formula": "paper",
              "wrapper_source_sha256": experiment.digest(Path(__file__)),
              "unchanged": "같은 기준스캔·음성·양성수·순서·초기가중치·4096단계·모델·손실·학습률",
              "changed": "repair는 Eq13의 min(valid_anchor,valid_reprojected_ref)로 나눈 overlap>.3 비자기 양성 선택. control은 공개 코드 원래 선택 양성",
              "negative_rule": "기존 음성이 두 정의 모두 <.3인 기준스캔만 사용해 양성/음성 충돌을 방지",
              "test_used": False}
    experiment.save(OUT / "plan.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("prepare", "control", "repair"))
    args = parser.parse_args()
    experiment.OUT = OUT
    experiment.prepare = prepare
    if args.arm == "prepare":
        value = prepare()
        print(json.dumps({k: v for k, v in value.items() if k not in ("control", "repair", "sources_sha256")}, ensure_ascii=False))
    else:
        experiment.run(args.arm)
