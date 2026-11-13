#!/usr/bin/env python3
"""동결된 BEVPlace++ 두 후보 검증을 재개 가능하게 실행한다."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import _review_selection_bev as task


def worker() -> None:
    """평가 코드는 유지하고 검산된 묶음 검색 실행만 연결한다."""
    original = subprocess.run

    def amended(command: list, **kwargs):
        if command[1].endswith("_review_selection_prefix_exact.py"):
            command = list(command)
            command[1] = str(task.ROOT / "scripts/_review_selection_prefix_batch.py")
        return original(command, **kwargs)

    task.subprocess.run = amended
    task.evaluate(int(sys.argv[2]), sys.argv[3], sys.argv[4], "val")


def main() -> None:
    """검증 결과 선택 전에 두 최종 후보의 모든 검증 원장을 만든다."""
    amendment = task.OUT / "execution_amendment.json"
    sources = [Path(__file__), task.ROOT / "scripts/_review_selection_preflight.py", task.ROOT / "scripts/_review_selection_prefix_batch.py",
               task.OUT / "preflight/batch_preflight.json"]
    if not amendment.exists():
        checks = task.read(sources[-1])
        assert all(row["all_fields_exact"] for row in checks.values())
        task.write(amendment, dict(created_unix=time.time(), protocol_sha256=task.sha(task.OUT / "protocol.json"),
            changes="128질의의 공통 과거접두에서 후보만 묶음 제안; 각 질의의 추가 과거후보는 모두 CPU L2로 계산",
            unchanged="체크포인트·질의·후보규칙·공식 CPU 최종점수·AP/R@1 선택규칙",
            source_sha256={str(p.relative_to(task.ROOT)): task.sha(p) for p in sources}, checks=checks))
    record = task.read(amendment)
    for source, digest in record["source_sha256"].items():
        assert task.sha(task.ROOT / source) == digest
    logs = task.OUT / "logs"
    logs.mkdir(exist_ok=True)
    jobs = [(1, "best", "kitti360/0004")]
    jobs += [(s, "final", q) for s in (1, 2, 3) for q in task.read(task.OUT / "protocol.json")["val"]]
    for done, (seed, tag, sequence) in enumerate(jobs):
        task.write(task.OUT / "status.json", dict(status="running", pid=os.getpid(), done=done,
            total=len(jobs), current=[seed, tag, sequence], updated_unix=time.time()))
        with (logs / f"s{seed}_{tag}_{sequence.replace('/', '_')}.log").open("a") as log:
            subprocess.run([sys.executable, str(Path(__file__)), "worker", str(seed), tag, sequence],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print("DONE", done + 1, len(jobs), seed, tag, sequence, flush=True)
    task.write(task.OUT / "status.json", dict(status="validation_complete", pid=os.getpid(),
        done=len(jobs), total=len(jobs), updated_unix=time.time()))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker()
    else:
        main()
