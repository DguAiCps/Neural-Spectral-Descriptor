#!/usr/bin/env python3
"""특징 생성·간선 분류기 재학습·최종 검증·감사를 도커에서 실행한다."""
from __future__ import annotations

import concurrent.futures as futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_edge_labels3d as exp
from _pdv2_no_context_ablation import write_json, sha256

LOG = REPO / "logs/edge_labels3d_20261016"


def call(args: list, label: str) -> None:
    """로그를 남기고 실패를 진행 상태로 전달한다."""
    path = LOG / (label + ".log")
    with path.open("a") as out:
        p = subprocess.run([sys.executable, "-u", *args], cwd=REPO, stdout=out, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"{label}: {path.read_text()[-4000:]}")


def main() -> None:
    """설정 동결 이후 각 단계를 완료하고 최종 지표를 검사한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      PYTHONDONTWRITEBYTECODE="1")
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    call(["scripts/_pdv2_edge_labels3d.py", "prepare"], "prepare")
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(call, ["scripts/_pdv2_edge_labels3d.py", "features", "--sequence", s + "/" + q],
                            f"features_{s}_{q}"): (s, q) for s, q in exp.TRAIN}
        for i, job in enumerate(futures.as_completed(jobs), 1):
            job.result()
            write_json(exp.OUT / "status.json", {"status": "running", "stage": "training_features", "done": i, "total": 8})
            print("FEATURES", i, flush=True)
    for arm in ("xy", "xyz"):
        write_json(exp.OUT / "status.json", {"status": "running", "stage": "classifier_training", "arm": arm})
        call(["scripts/_pdv2_edge_labels3d.py", "train", "--arm", arm], "train_" + arm)
    assert exp.read(exp.OUT / "classifiers/xy.json")["initial_state_sha256"] == exp.read(exp.OUT / "classifiers/xyz.json")["initial_state_sha256"]
    extracted = scored = 0
    with futures.ThreadPoolExecutor(max_workers=2) as gpu, futures.ThreadPoolExecutor(max_workers=4) as cpu:
        pending = {gpu.submit(call, ["scripts/_pdv2_edge_labels3d.py", "keys", "--model", n, "--sequence", s + "/" + q],
                              f"keys_{n}_{s}_{q}"): (n, s, q) for n in exp.MODELS for s, q in exp.VAL}
        scores = set()
        while pending or scores:
            ready, _ = futures.wait(set(pending) | scores, return_when=futures.FIRST_COMPLETED)
            for job in ready:
                job.result()
                if job in pending:
                    n, s, q = pending.pop(job)
                    extracted += 1
                    scores.update(cpu.submit(call, ["scripts/_pdv2_edge_labels3d.py", "score", "--model", n,
                                                    "--arm", arm, "--sequence", s + "/" + q],
                                             f"score_{n}_{arm}_{s}_{q}") for arm in exp.ARMS)
                else:
                    scores.remove(job)
                    scored += 1
                write_json(exp.OUT / "status.json", {"status": "running", "stage": "validation", "test_used": False,
                           "keys_done": extracted, "keys_total": 27, "score_done": scored, "score_total": 108})
                print("PROGRESS", extracted, scored, flush=True)
    call(["scripts/_pdv2_edge_labels3d.py", "report"], "report")
    write_json(exp.OUT / "status.json", {"status": "running", "stage": "audit", "test_used": False})
    call(["scripts/_pdv2_edge_labels3d_verify.py"], "verify")
    call(["scripts/_pdv2_edge_labels3d_plot.py"], "plot")
    validation = exp.read(exp.OUT / "completion_validation.json")
    write_json(exp.OUT / "status.json", {"status": "complete", "test_used": False, "finished_unix": time.time(),
               "runtime_s": time.time() - started, "verification_status": validation["status"],
               "summary_sha256": sha256(exp.OUT / "summary.json")})
    print("COMPLETE", time.time() - started, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(exp.OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
