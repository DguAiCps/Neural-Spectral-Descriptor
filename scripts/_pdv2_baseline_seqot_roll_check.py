#!/usr/bin/env python3
"""SeqOT의 수평 회전 증강이 실제 출력에 차이를 만드는지 확인한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    """고정한 학습 입력32개와 세 회전각에서 FP32 출력 오차를 잰다."""
    import torch
    import _seqot_official_train as s

    torch.set_num_threads(1)
    selected = json.loads((ROOT / "results/pdv2_seqot_official_centered/train_selection.json").read_text())["stage1"]["chosen"]
    checkpoint = ROOT / selected["checkpoint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == selected["sha256"]
    model = s._load_fe(checkpoint, "cuda").eval()
    shifts = [75, 375, 450]  # 30, 150, 180 degrees at 900 columns.
    rows = []
    for sensor, sequence in s.TRAIN:
        src = s.StoreSource("train", sensor, sequence)
        for anchor in np.linspace(1, src.n_raw - 2, 4, dtype=int):
            original = src.rows(s.centered(int(anchor), 3)).astype(np.float32)
            arr = np.stack([original] + [np.roll(original, shift, axis=-1) for shift in shifts])
            with torch.no_grad():
                desc = model(torch.from_numpy(arr).cuda()).cpu().numpy()
            for shift, value in zip(shifts, desc[1:]):
                delta = value.astype(float) - desc[0].astype(float)
                rows.append({"sequence": f"{sensor}/{sequence}", "anchor": int(anchor), "shift_columns": shift,
                             "max_absolute_difference": float(np.abs(delta).max()), "squared_l2": float(delta @ delta)})
    report = {"checkpoint": selected, "inputs": 32, "comparisons": len(rows), "rows": rows,
              "max_absolute_difference": max(r["max_absolute_difference"] for r in rows),
              "max_squared_l2": max(r["squared_l2"] for r in rows),
              "structure": "공식 폭1 convolution·위치부호 없는 attention·순서 없는 NetVLAD 합산: 수평 순환 이동은 토큰 순열이다.",
              "scope": "고정 입력의 열 순환 이동 검사이며 실제 역주행의 관측 변화에 대한 검사와 구별한다.",
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = ROOT / "results/pdv2_baseline_common_diagnosis_20261106/seqot_roll_check.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print({k: v for k, v in report.items() if k not in ["rows", "checkpoint"]}, flush=True)


if __name__ == "__main__":
    main()
