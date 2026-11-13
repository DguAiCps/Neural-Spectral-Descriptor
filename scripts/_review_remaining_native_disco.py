#!/usr/bin/env python3
"""서버 ring_baseline에서 DiSCO 저자 가중치와 원본 입력 경로를 점검한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results/pdv2_remaining_native_20261106/disco"


def main() -> None:
    """실제 스캔의 공식 전처리와 공개 모델 추론을 반복 확인한다."""
    import torch
    from _disco_official_dump import official_model
    from _disco_official_prep import generator_functions
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    ck = OUT.parent / "weights/disco_author_model.ckpt"
    cfg, _, _, model = official_model(ck)
    import gputransform
    functions, _ = generator_functions(gputransform)
    root = REPO / "results/pdv2_review_native_20261105/native_data"
    files = sorted(root.glob("*/velodyne_sync/*.bin"))
    selected = [files[i] for i in [0, len(files) // 2, len(files) - 1]]
    model.eval()
    rows = []
    for raw in selected:
        polar = functions["load_pc_file"](str(raw))
        tensor = torch.from_numpy(np.asarray(polar, dtype=np.float32)).reshape(1, cfg.num_height, cfg.num_ring, cfg.num_sector).cuda()
        with torch.no_grad():
            a = model(tensor)[0].cpu().numpy()
            b = model(tensor)[0].cpu().numpy()
        assert a.shape == (1, 1024) and np.isfinite(a).all()
        assert np.array_equal(a, b)
        rows.append({"raw": str(raw.relative_to(REPO)),
                     "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                     "polar_shape": list(tensor.shape), "descriptor_shape": list(a.shape),
                     "finite": True, "repeat_max_abs_difference": float(np.max(np.abs(a-b)))})
    report = {"passed": True, "scope": "공개 가중치 적재·공식 원시스캔 전처리·실제 GPU 추론 점검; 논문 평가 수치 재현 아님",
              "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(), "rows": rows,
              "paper_evaluation_sessions": ["2012-03-17", "2012-05-26", "2012-06-15", "2012-08-20", "2012-09-28", "2012-10-28"],
              "missing_local_sessions": ["2012-05-26", "2012-06-15", "2012-08-20", "2012-09-28", "2012-10-28"],
              "remaining": "6평가 세션의 공식 생성기 입력·3m 표본·1.5m 정답과 세션 간 평균을 함께 재현해야 논문89.08%와 비교할 수 있다."}
    (OUT / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
