#!/usr/bin/env python3
"""AMP에서 재계량의 숫자 정밀도 혼입 가능성을 실제 학습 특징으로 검사한다."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _pdv2_key_metric_alignment as run
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash


def main() -> None:
    """단위 재계량이 원 특징 키를 바꾸지 않는지 전수 확인한다."""
    run.protocol()
    state = torch.load(run.PREPARED, map_location="cpu", weights_only=False)
    x = torch.from_numpy(state["descriptors"].astype(np.float32)).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        # SpectralGNN.forward의 x_raw 조기 변환과 키 블록 연산을 그대로 재현한다.
        raw = x.to(torch.float16)
        plain = F.normalize(raw, p=2, dim=-1)
        unity = F.normalize(
            raw * torch.ones(288, dtype=torch.float32, device="cuda"), p=2, dim=-1
        )
    assert plain.dtype == unity.dtype == torch.float32
    assert torch.equal(plain, unity)
    assert torch.isfinite(plain).all()
    write_json(
        run.OUT / "precision_audit.json",
        {
            "status": "passed",
            "training_nodes": len(x),
            "raw_dtype": str(raw.dtype),
            "plain_output_dtype": str(plain.dtype),
            "unit_scale_output_dtype": str(unity.dtype),
            "max_absolute_difference": float((plain - unity).abs().max()),
            "plain_sha256": array_hash(plain.cpu().numpy()),
            "unit_scale_sha256": array_hash(unity.cpu().numpy()),
            "interpretation": "실제 AMP 경로에서 F.normalize는 이미 FP32를 반환한다. 학습 전체72306노드에서 단위FP32재계량과 기존원특징키는 비트단위로 같다. 따라서 고정원특징 경로의 FP32승격이 비교의 추가변경이라는 우려는 성립하지 않으며 별도동일학습을 반복하지 않는다. CUDA누적비결정성의 일반한계는 남는다.",
            "source_sha256": sha256(Path(__file__)),
            "model_source_sha256": sha256(run.REPO / "src/gnn/model.py"),
            "prepared_sha256": sha256(run.PREPARED),
            "test_used": False,
        },
    )
    print("PRECISION_AUDIT_PASSED", len(x), str(plain.dtype), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
