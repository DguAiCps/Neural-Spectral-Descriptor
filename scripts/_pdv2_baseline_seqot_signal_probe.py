#!/usr/bin/env python3
"""실제 SeqOT 학습 입력·고정 가중치로 자기 양성의 손실 신호를 확인한다.

ring_baseline 안에서 실행한다. 학습·가중치 저장·시험셋 접근을 하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/pdv2_baseline_common_diagnosis_20261106"


def main() -> None:
    """각 학습 시퀀스에서 자기쌍·비자기쌍 조건을 균등 간격으로 조사한다."""
    import torch
    import _seqot_official_train as s

    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    s.seed_all()
    device = torch.device("cuda")
    sources = [s.StoreSource("train", sensor, seq) for sensor, seq in s.TRAIN]
    plan = s.anchor_plan(s.load_train_indices(),
                        lambda d, a: all(sources[d].exists(int(x)) for x in s.centered(a, s.SEQL)), *s.USE1)
    chosen = json.loads((ROOT / "results/pdv2_seqot_official_centered/train_selection.json").read_text())["stage1"]["chosen"]
    path = ROOT / chosen["checkpoint"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == chosen["sha256"]
    model = s._load_fe(path, device)
    official_loss = s.official()["loss"].triplet_loss
    selected = []
    for d in range(len(s.TRAIN)):
        for all_self in [True, False]:
            subset = [p for p in plan if p[0] == d and all(x == p[1] for x in p[2]) == all_self]
            ids = np.unique(np.linspace(0, len(subset) - 1, min(16, len(subset)), dtype=int))
            selected.extend(subset[i] for i in ids)
    protocol = {"seed": 1, "checkpoint": chosen, "steps": len(selected),
                "sampling": "8학습 시퀀스×모든 양성이 자기쌍/그 외×최대16기준프레임을 순서상 균등 간격 선택",
                "scope": "고정 모델의 학습 신호 진단. train/eval 모드 비교; optimizer step 및 시험 접근 없음.",
                "official_loss": "triplet_loss use_min=False: max positive squared distance, margin0.5",
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "seqot_signal_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    rows = []
    start = time.time()
    for d, anchor, pos, neg in selected:
        arr = s.stage1_batch(sources[d], anchor, pos, neg, s.TRAIN_WINDOW)
        row = {"sequence": "/".join(s.TRAIN[d]), "anchor": anchor, "positives": pos, "negatives": neg,
               "all_positive_self": all(x == anchor for x in pos),
               "inputs_sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
        # 두 모드 모두 같은 체크포인트 통계를 사용한다. train forward의 BN 갱신은 복구한다.
        buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        for mode in ["eval", "train"]:
            model.train(mode == "train")
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(buffers[name])
                emb = model(torch.from_numpy(arr).to(device))
            emb = emb.detach().requires_grad_(True)
            query, positive, negative = torch.split(emb, [1, len(pos), len(neg)])
            loss = official_loss(query, positive, negative, s.MARGIN_1, lazy=False)
            gradient = torch.autograd.grad(loss, emb)[0]
            max_pos = ((positive - query) ** 2).sum(1).max().item()
            row[mode] = {"loss": float(loss.item()), "max_positive_squared_distance": max_pos,
                         "positive_output_gradient_norm": float(gradient[1:1 + len(pos)].norm().item()),
                         "query_output_gradient_norm": float(gradient[:1].norm().item()),
                         "negative_output_gradient_norm": float(gradient[1 + len(pos):].norm().item())}
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        rows.append(row)
    summary = {}
    for all_self in [True, False]:
        subset = [row for row in rows if row["all_positive_self"] == all_self]
        summary[str(all_self)] = {"samples": len(subset)}
        for mode in ["eval", "train"]:
            summary[str(all_self)][mode] = {
                "active_loss_fraction": float(np.mean([r[mode]["loss"] > 1e-8 for r in subset])),
                "mean_positive_distance": float(np.mean([r[mode]["max_positive_squared_distance"] for r in subset])),
                "max_positive_gradient_norm": max(r[mode]["positive_output_gradient_norm"] for r in subset),
                "mean_positive_gradient_norm": float(np.mean([r[mode]["positive_output_gradient_norm"] for r in subset]))}
            if all_self:
                assert summary[str(all_self)][mode]["mean_positive_distance"] < 1e-8
                assert summary[str(all_self)][mode]["max_positive_gradient_norm"] < 1e-5
    report = {"protocol": protocol, "rows": rows, "summary": summary,
              "elapsed_s": time.time() - start, "checks_passed": True,
              "interpretation": "자기쌍만 있는 단계는 같은 장소의 다른 스캔을 당기는 양성 신호가 없다. 음성 밀어내기 신호까지 항상0이라는 뜻은 아니다."}
    (OUT / "seqot_signal_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": summary, "elapsed_s": report["elapsed_s"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
