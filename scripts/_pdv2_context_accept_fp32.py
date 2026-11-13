#!/usr/bin/env python3
"""기존 평가와 같은 TF32 비활성 정밀도로 미완료 대조를 수행한다."""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_context_accept as run
from _pdv2_no_context_ablation import sha256, write_json

ORIGINAL_TRAIN_ROOT = run.train_root
ORIGINAL_FEATURE_FILE = run.feature_file


def configure() -> None:
    """근사 계산 산출물을 보존하고 교정 산출물은 새 경로에 둔다."""
    torch.backends.cuda.matmul.allow_tf32 = False
    run.train_root = lambda seed: run.PRIOR if seed == 1 else run.OUT/f"train_fp32/s{seed}"
    run.feature_file = lambda role, seed, sequence: (
        run.OUT/f"features/train_fp32/s{seed}/{sequence.replace('/', '_')}.npz"
        if role == "train" and seed != 1 else ORIGINAL_FEATURE_FILE(role, seed, sequence))


def main() -> None:
    """검증 결과를 보기 전에 정밀도 오류를 교정하고 원 실험을 재개한다."""
    configure()
    target = run.OUT / "fp32_amendment.json"
    if not target.exists():
        assert not (run.OUT/"fitted.json").exists()
        assert not (run.OUT/"features/val").exists()
        probe = run.exp.read(run.OUT/"precision_probe.json")
        assert max(r["max_error"] for r in probe["records"]["32"]["prefix"]) < 3e-6
        assert max(r["max_error"] for r in probe["records"]["64"]["prefix"]) < 1e-10
        assert probe["fp32_fp64_all_error"] < 3e-6
        sources = (Path(__file__), run.OUT/"protocol.json", run.OUT/"numerical_amendment.json",
                   run.OUT/"precision_probe.json", REPO/"scripts/_pdv2_context_accept_precision_probe.py",
                   REPO/"logs/pdv2_context_accept_20261018.log", REPO/"logs/pdv2_context_accept_resume_20261018.log")
        write_json(target, {"reason": "현환경torch의TF32기본값True. 원실험main이기존평가의False설정을누락. 첫정밀도추가검사도같은근사계산을유지해실패. TF32False에서FP32접두최대1.49e-7,FP64접두5.97e-16,전체교차8.78e-7확인.",
            "change": "TF32False명시. 시드2/3학습키·검색·특징을새경로에재생성. 기존False생성시드1·검증키재사용. 원atol3e-6/rtol3e-5접두검사유지. 점수대조·적합·선택규칙변경없음.",
            "excluded": ["train/s2", "features/train/s2", "precision_probe_fp32.npy", "precision_probe_fp64.npy"],
            "active_training": ["기존pdv2_train_failure시드1", "train_fp32/s2", "train_fp32/s3"],
            "sources": {str(p.relative_to(REPO)): sha256(p) for p in sources},
            "test_used": False, "validation_features_computed": False})
    for p, expected in run.exp.read(target)["sources"].items():
        assert sha256(REPO/p) == expected
    run.main()


if __name__ == "__main__":
    main()
