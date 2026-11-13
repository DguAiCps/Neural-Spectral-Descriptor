#!/usr/bin/env python3
"""고정 BEVPlace++ 평가의 검색 계산만 검증한 가속 경로로 실행한다."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import _pdv2_bevplace_val_odom as base

ACCELERATOR = base.REPO / "scripts/_bevplace_prefix_exact.py"


def amendment() -> dict:
    """가중치·점수 정의를 유지한 실행 변경과 대조 검사를 고정한다."""
    base.prepare()
    path = base.OUT / "execution_amendment.json"
    if path.exists():
        record = base.read(path)
        for source, digest in record["sources"].items():
            assert base.bev.sha256(base.REPO / source) == digest, source
        return record
    preflight = base.read(base.OUT / "accelerated_preflight.json")
    assert preflight["status"] == "passed"
    assert preflight["source_sha256"] == base.bev.sha256(ACCELERATOR)
    sources = [Path(__file__), ACCELERATOR, base.OUT / "accelerated_preflight.json"]
    record = {
        "created_unix": time.time(),
        "reason": "GENERIC CPU FAISS의 검색 시간. 가중치·전처리·후보 규칙·L2 정의는 유지.",
        "algorithm": "과거 GPU 색인으로 넉넉한 후보를 제안하고 공식 CPU L2로 재계산. 반올림 여유 밖 후보만 배제. 배제 불가 시 전체 CPU 검색.",
        "scope": "GPU 검색은 별도 프로세스에서 TF32/FP16을 끈다. 원래 신경망 추론 환경은 바꾸지 않는다.",
        "reference": preflight,
        "sources": {str(p.relative_to(base.REPO)): base.bev.sha256(p) for p in sources},
        "test_used": False, "model_reselected": False,
    }
    base.write(path, record)
    return record


def evaluate(sequence: str) -> None:
    """원래 평가의 검색 함수만 별도 프로세스로 교체한다."""
    amendment()
    name = sequence.replace("/", "_")

    def child_search(descriptors: np.ndarray, positions: np.ndarray) -> dict:
        feature_path = base.OUT / f"descriptors_{name}.npy"
        position_path = base.OUT / f"search_positions_{name}.npy"
        target = base.OUT / f"search_full_{name}.npz"
        np.testing.assert_array_equal(descriptors, np.load(feature_path, mmap_mode="r"))
        np.save(position_path, positions)
        environment = dict(os.environ)
        environment["NVIDIA_TF32_OVERRIDE"] = "0"
        command = [sys.executable, str(ACCELERATOR), "--descriptors", str(feature_path),
                   "--positions", str(position_path), "--out", str(target)]
        subprocess.run(command, env=environment, check=True)
        with np.load(target) as archive:
            return {key: archive[key] for key in archive.files}

    base.ledger_l2 = child_search
    base.evaluate(sequence)


def run() -> None:
    """영상 준비와 고정 선택을 재사용해 남은 평가를 완료한다."""
    record = amendment()
    assert base.bev.cmd_status([base.ROLE])
    started = time.time()
    for count, (sensor, seq) in enumerate(base.VAL, 1):
        sequence = sensor + "/" + seq
        with (base.LOG / f"fast_evaluate_{sensor}_{seq}.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__)), "evaluate", sequence],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        base.write(base.OUT / "status.json", {"status": "evaluate", "done": count,
                                              "total": len(base.VAL), "execution_amendment": True})
        print("EVALUATED", count, len(base.VAL), sequence, flush=True)
    base.report()
    amendment()
    base.write(base.OUT / "status.json", {"status": "complete", "amended_runtime_s": time.time() - started,
                                          "execution_amendment": True})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        evaluate(sys.argv[2])
    else:
        run()
