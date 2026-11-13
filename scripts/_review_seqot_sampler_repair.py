#!/usr/bin/env python3
"""SeqOT의 같은 기준·음성 쌍에서 비자기 양성 선택을 대조한다.

학습 정답·모델·손실은 유지한다. ring_baseline 안에서 실행한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/pdv2_seqot_ring_repair_20261106/seqot"


def save(path: Path, data: dict) -> None:
    """검산 가능한 결과를 원자적으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path: Path) -> str:
    """파일 내용의 SHA256을 반환한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> dict:
    """원래 학습 overlap에서 자기 자신을 제외한 양성만 복구한다."""
    import _seqot_official_train as s
    import _seqot_official_prep as p

    if (OUT / "plan.json").exists():
        return json.loads((OUT / "plan.json").read_text())
    sources = [s.StoreSource("train", sensor, seq) for sensor, seq in s.TRAIN]
    valid = lambda d, a: all(sources[d].exists(int(x)) for x in s.centered(a, s.SEQL))
    original = s.anchor_plan(s.load_train_indices(), valid, *s.USE1)
    pools, audit, hashes = {}, {}, {}
    for d, (sensor, seq) in enumerate(s.TRAIN):
        i, j, overlap = p.load_explicit(sensor, seq, "code")
        mask = (overlap > s.OVERLAP_THRESH) & (i != j)
        ii, jj = i[mask] * 5, j[mask] * 5
        order = np.argsort(ii, kind="stable")
        keys, first = np.unique(ii[order], return_index=True)
        ends = np.append(first[1:], len(order))
        for key, lo, hi in zip(keys, first, ends):
            partners = [int(x) for x in jj[order[lo:hi]] if valid(d, int(x))]
            if valid(d, int(key)) and partners:
                pools[(d, int(key))] = partners
        audit[f"{sensor}/{seq}"] = {"nonself_positive_pairs": int(mask.sum()),
                                     "eligible_anchors": sum(k[0] == d for k in pools)}
        for path in [p.label_dir(sensor, seq) / "normalized_data.npy",
                     *p.label_dir(sensor, seq).glob("ovc_*of*.npz")]:
            hashes[str(path.relative_to(ROOT))] = digest(path)
    eligible = [row for row in original if (row[0], row[1]) in pools]
    rng = np.random.default_rng(20261106)
    ids = rng.permutation(len(eligible))[:4096]
    assert len(ids) == 4096
    control, repair = [], []
    for index in ids:
        d, a, pos, neg = eligible[int(index)]
        candidates = pools[(d, a)]
        new = rng.choice(candidates, size=len(pos), replace=len(candidates) < len(pos)).tolist()
        assert all(x != a and x in candidates for x in new)
        control.append([d, a, pos, neg])
        repair.append([d, a, new, neg])
    assert all(c[:2] == r[:2] and c[3] == r[3] and len(c[2]) == len(r[2])
               for c, r in zip(control, repair))
    result = {"control": control, "repair": repair, "train_sequences": s.TRAIN,
              "original_steps": len(original), "eligible_steps": len(eligible),
              "dropped_no_nonself_positive": len(original) - len(eligible),
              "steps": len(control), "sequence_audit": audit,
              "control_self_only_steps": sum(all(x == a for x in pos) for _, a, pos, _ in control),
              "repair_self_only_steps": 0, "sources_sha256": hashes,
              "scope": "동일 기준 프레임·음성·양성 수·순서·초기 가중치. 기존 전체 방향성 overlap>0.3 양성에서 비자기 쌍만 선택; 정답 반경 변경 없음.",
              "unresolved": "원래 overlap에 비자기 양성이 없는 기준 프레임은 두 조건 모두 제외. 센서 희소성과 학습 장소 부족을 해결했다는 대조가 아님.",
              "test_used": False}
    save(OUT / "plan.json", result)
    return result


def run(arm: str) -> None:
    """동일 예산 추가 학습과 전체 공통 검증을 완료한다."""
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    import torch
    import _seqot_official_train as s
    from _review_selection_full_inputs import SeqSource
    from _review_selection_full_eval import query_base
    from _review_selection_full_l2 import ledger_exact
    from _pr_eval_lib import classify_queries, pr_sweep

    torch.set_num_threads(1)
    data = prepare()
    plan = data[arm]
    home = OUT / arm
    home.mkdir(parents=True, exist_ok=True)
    s.seed_all()
    device = torch.device("cuda")
    selected = json.loads((ROOT / "results/pdv2_seqot_official_centered/train_selection.json").read_text())
    checkpoints = {stage: selected[stage]["chosen"] for stage in ("stage1", "stage2")}
    for row in checkpoints.values():
        assert digest(ROOT / row["checkpoint"]) == row["sha256"]
    protocol = {"arm": arm, "steps": len(plan), "plan_sha256": digest(OUT / "plan.json"),
                "initial_checkpoints": checkpoints, "seed": 1, "lr": 5e-7,
                "optimizer": "새 Adam; scheduler 없음; GeM 고정", "test_used": False,
                "validation": "전체9시퀀스, KISS·과거 후보·공식 L2·3계열 평균",
                "selection": "검증 R@1 우선. 최종 AP/F1/R@100P도 동시 보고; 시험 사용 없음",
                "source_sha256": digest(Path(__file__))}
    if (home / "protocol.json").exists():
        assert json.loads((home / "protocol.json").read_text()) == protocol
    else:
        save(home / "protocol.json", protocol)
    sources = [s.StoreSource("train", sensor, seq) for sensor, seq in s.TRAIN]
    val_sources = [((sensor, seq), SeqSource(sensor, seq)) for sensor, seq in s.VAL]
    checkpoint = home / "model.pth.tar"
    start = time.time()
    if not checkpoint.exists():
        model = s._load_fe(ROOT / checkpoints["stage1"]["checkpoint"], device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-7)
        loss_fn = s.official()["loss"].triplet_loss
        losses, distances = [], []
        for step, (d, anchor, pos, neg) in enumerate(plan, 1):
            arr = s.stage1_batch(sources[d], anchor, pos, neg, s.TRAIN_WINDOW)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            feat = model(torch.from_numpy(arr).to(device))
            q, positive, negative = torch.split(feat, [1, len(pos), len(neg)])
            loss = loss_fn(q, positive, negative, s.MARGIN_1, lazy=False)
            assert torch.isfinite(loss)
            distances.append(float(((positive - q) ** 2).sum(1).max().detach()))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 256 == 0 or step == len(plan):
                status = {"stage": "training", "arm": arm, "completed": step, "total": len(plan),
                          "elapsed_s": time.time() - start, "loss": float(np.mean(losses[-256:])),
                          "positive_distance": float(np.mean(distances[-256:]))}
                save(home / "status.json", status)
                print(json.dumps(status), flush=True)
        torch.save({"state_dict": model.state_dict(), "protocol": protocol}, checkpoint)
        save(home / "train.json", {"elapsed_s": time.time() - start,
                                   "checkpoint_sha256": digest(checkpoint),
                                   "mean_loss": float(np.mean(losses)),
                                   "positive_distance_nonzero_steps": int(np.count_nonzero(distances)),
                                   "active_loss_steps": int(np.count_nonzero(losses))})
    model = s._load_fe(checkpoint, device)
    gem = s._load_gem(ROOT / checkpoints["stage2"]["checkpoint"], device)
    rows = {}
    for (sensor, seq), src in val_sources:
        target = home / f"val_{sensor}_{seq}.json"
        if target.exists():
            row = json.loads(target.read_text())
            assert row["checkpoint_sha256"] == digest(checkpoint)
            rows[f"{sensor}/{seq}"] = row["metrics"]
            continue
        tick = time.time()
        descriptors, _, _ = s.centered_final_descriptors(model, gem, src, src.kf, device, batch=64)
        arrays, audit = ledger_exact(descriptors, src.kf_poses[:, :3, 3], query_base(sensor, seq, "val"))
        query = classify_queries(arrays["min_past_gt_dist"])
        metrics = {"n_queries": len(arrays["query_idx"]), "n_positive": int(query["n_pos"])}
        sweep = pr_sweep(arrays["score_disco"], arrays["top1_geo_dist"], query)
        metrics.update({k: float(sweep[k]) if query["n_pos"] else None for k in ("AP", "max_F1", "R@100P")})
        metrics["R@1"] = float((arrays["top1_geo_dist"][query["positive"]] <= 5).mean()) if query["n_pos"] else None
        ledger = home / f"val_{sensor}_{seq}.npz"
        np.savez_compressed(ledger, **arrays)
        save(target, {"metrics": metrics, "seconds": time.time() - tick, "search_audit": audit,
                      "ledger_sha256": digest(ledger), "checkpoint_sha256": digest(checkpoint)})
        rows[f"{sensor}/{seq}"] = metrics
        save(home / "status.json", {"stage": "validation", "completed": len(rows), "total": 9})
        print("VAL", arm, sensor, seq, metrics, flush=True)
    keys = ("AP", "max_F1", "R@100P", "R@1")
    family = lambda name: "kitti" if name.startswith("kitti") else name.split("/")[0]
    families = {f: {k: float(np.mean([r[k] for name, r in rows.items()
                                    if family(name) == f and r[k] is not None])) for k in keys}
                for f in ("kitti", "helipr", "mulran")}
    assert sum(r["n_queries"] for r in rows.values()) == 99236
    report = {"arm": arm, "sequences": rows, "families": families,
              "macro": {k: float(np.mean([v[k] for v in families.values()])) for k in keys},
              "elapsed_s": time.time() - start, "protocol": protocol, "passed": True}
    save(home / "report.json", report)
    save(home / "status.json", {"stage": "complete", "metrics": report["macro"]})
    print("DONE", arm, report["macro"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("prepare", "control", "repair"))
    args = parser.parse_args()
    if args.arm == "prepare":
        data = prepare()
        print(json.dumps({k: v for k, v in data.items() if k not in ("control", "repair", "sources_sha256")}, ensure_ascii=False))
    else:
        run(args.arm)
