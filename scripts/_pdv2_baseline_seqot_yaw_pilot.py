#!/usr/bin/env python3
"""같은 SeqOT 학습 예산에서 회전 증강 하나만 바꾸고 공통 검증 최종 성능을 잰다.

ring_baseline 안에서 --arm control|yaw를 각각 실행한다. 시험셋은 읽지 않는다.
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
OUT = ROOT / "results/pdv2_baseline_common_diagnosis_20261106/yaw_pilot"


def save(path: Path, data: dict) -> None:
    """진행과 결과를 원자적으로 저장한다."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main() -> None:
    """고정한 두 조건을 같은 4,096단계와 검증9 전체에서 비교한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["control", "yaw"], required=True)
    args = parser.parse_args()
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    import torch
    import _seqot_official_train as s
    from _review_selection_full_inputs import SeqSource
    from _review_selection_full_eval import query_base
    from _review_selection_full_l2 import ledger_exact
    from _pr_eval_lib import classify_queries, pr_sweep

    torch.set_num_threads(1)
    home = OUT / args.arm
    home.mkdir(parents=True, exist_ok=True)
    s.seed_all()
    device = torch.device("cuda")
    selected = json.loads((ROOT / "results/pdv2_seqot_official_centered/train_selection.json").read_text())
    checkpoints = {stage: selected[stage]["chosen"] for stage in ["stage1", "stage2"]}
    for row in checkpoints.values():
        assert hashlib.sha256((ROOT / row["checkpoint"]).read_bytes()).hexdigest() == row["sha256"]
    sources = [s.StoreSource("train", sensor, seq) for sensor, seq in s.TRAIN]
    plan = s.anchor_plan(s.load_train_indices(),
                        lambda d, a: all(sources[d].exists(int(x)) for x in s.centered(a, s.SEQL)), *s.USE1)
    selection_rng = np.random.default_rng(20261106)
    ids = selection_rng.permutation(len(plan))[:4096]
    plan = [plan[int(i)] for i in ids]
    plan_bytes = json.dumps(plan, separators=(",", ":")).encode()
    protocol = {"arm": args.arm, "steps": len(plan), "training_seed": 1,
                "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "initial_checkpoints": checkpoints,
                "optimizer": "새 Adam, lr5e-7; 두 조건 동일. 추가4096단계; scheduler 없음",
                "only_difference": "yaw 조건은 샘플마다 독립 수평 circular roll, 세 시간 프레임에 같은 이동. control은 원본.",
                "unchanged": "원래 overlap 라벨·선택 쌍·순서·양/음 개수·모델 구조·GeM 가중치·검색·수락 점수",
                "validation": "KISS 검증9 전체, q+15 시한, j<=q−30, 원래 L2 최종 점수, 3계열 동일 가중치",
                "test_used": False, "adoption": "단일 시드 진단. 자동 주표 채택이나 시험 평가 없음",
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if (home / "protocol.json").exists():
        assert json.loads((home / "protocol.json").read_text()) == protocol
    else:
        save(home / "protocol.json", protocol)
    # 추론 입력 전체가 준비됐는지 학습 전에 확인한다.
    val_sources = [((sensor, seq), SeqSource(sensor, seq)) for sensor, seq in s.VAL]
    for _, src in val_sources:
        need = s.centered_need(src, src.kf)
        raw = np.unique(np.concatenate([need - 1, need, need + 1]))
        assert all(int(r) in src.lookup for r in raw if 0 <= r < src.n_raw)
    model = s._load_fe(ROOT / checkpoints["stage1"]["checkpoint"], device)
    checkpoint = home / "pilot.pth.tar"
    start = time.time()
    if not checkpoint.exists():
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-7)
        loss_fn = s.official()["loss"].triplet_loss
        augmentation_rng = np.random.default_rng(20261107)
        losses = []
        for step, (d, anchor, pos, neg) in enumerate(plan, 1):
            arr = s.stage1_batch(sources[d], anchor, pos, neg, s.TRAIN_WINDOW)
            if args.arm == "yaw":
                for i, shift in enumerate(augmentation_rng.integers(0, arr.shape[-1], len(arr))):
                    arr[i] = np.roll(arr[i], int(shift), axis=-1)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            features = model(torch.from_numpy(arr).to(device))
            q, p, n = torch.split(features, [1, len(pos), len(neg)])
            loss = loss_fn(q, p, n, s.MARGIN_1, lazy=False)
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step()
            losses.append(float(loss.item()))
            if step % 256 == 0 or step == len(plan):
                status = {"arm": args.arm, "stage": "train", "completed": step,
                          "total": len(plan), "elapsed_s": time.time() - start,
                          "mean_last256_loss": float(np.mean(losses[-256:]))}
                save(home / "status.json", status)
                print(json.dumps(status), flush=True)
        torch.save({"state_dict": model.state_dict(), "steps": len(plan), "protocol": protocol}, checkpoint)
        save(home / "train.json", {"seconds": time.time() - start, "mean_loss": float(np.mean(losses)),
                                   "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    model = s._load_fe(checkpoint, device)
    gem = s._load_gem(ROOT / checkpoints["stage2"]["checkpoint"], device)
    rows = {}
    for (sensor, seq), src in val_sources:
        target = home / f"val_{sensor}_{seq}.json"
        if target.exists():
            rows[f"{sensor}/{seq}"] = json.loads(target.read_text())["metrics"]
            continue
        tick = time.time()
        descriptors, _, _ = s.centered_final_descriptors(model, gem, src, src.kf, device, batch=64)
        inference_s = time.time() - tick
        tick = time.time()
        arrays, audit = ledger_exact(descriptors, src.kf_poses[:, :3, 3], query_base(sensor, seq, "val"))
        query = classify_queries(arrays["min_past_gt_dist"])
        metrics = {"n_queries": len(arrays["query_idx"]), "n_positive": int(query["n_pos"])}
        sweep = pr_sweep(arrays["score_disco"], arrays["top1_geo_dist"], query)
        metrics.update({k: float(sweep[k]) if query["n_pos"] else None for k in ["AP", "max_F1", "R@100P"]})
        metrics["R@1"] = float((arrays["top1_geo_dist"][query["positive"]] <= 5).mean()) if query["n_pos"] else None
        ledger = home / f"val_{sensor}_{seq}.npz"
        np.savez_compressed(ledger, **arrays)
        save(target, {"metrics": metrics, "inference_s": inference_s, "search_s": time.time() - tick,
                      "search_audit": audit, "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest()})
        rows[f"{sensor}/{seq}"] = metrics
        save(home / "status.json", {"arm": args.arm, "stage": "validation", "completed": len(rows), "total": len(s.VAL)})
        print("VAL", args.arm, sensor, seq, metrics, "infer_s", inference_s, flush=True)
    assert sum(r["n_queries"] for r in rows.values()) == 99236
    family = lambda name: "kitti" if name.startswith("kitti") else name.split("/")[0]
    keys = ["AP", "max_F1", "R@100P", "R@1"]
    families = {f: {k: float(np.mean([r[k] for name, r in rows.items() if family(name) == f and r[k] is not None]))
                    for k in keys} for f in ["kitti", "helipr", "mulran"]}
    report = {"arm": args.arm, "sequences": rows, "families": families,
              "macro": {k: float(np.mean([r[k] for r in families.values()])) for k in keys},
              "queries": 99236, "elapsed_total_s": time.time() - start,
              "protocol": protocol, "checks_passed": True}
    save(home / "report.json", report)
    save(home / "status.json", {"arm": args.arm, "stage": "complete", "metrics": report["macro"]})
    print("DONE", args.arm, report["macro"], flush=True)


if __name__ == "__main__":
    main()
