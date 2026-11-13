#!/usr/bin/env python3
"""진단의 결정적 연산 개정 기록을 보존하고 동결된 학습·평가를 이어간다."""
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

import _pdv2_key_metric_alignment as run
import _pdv2_key_metric_alignment_run as driver
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """원래 비교 허용 오차를 유지하며 GPU 진단만 결정적으로 실행한다."""
    target = run.OUT / "diagnostic_amendment.json"
    record = {
        "reason": "첫 FP32 진단의 동일 문맥 출력 비교에서 CUDA scatter_add 누적 순서 때문에 최대 2.3469328880310059e-7 차이가 발생했다. 비교 허용 오차를 완화하지 않고 진단에만 결정적 연산을 켠다.",
        "failure_log": str((driver.LOG / "diagnose.log").relative_to(run.REPO)),
        "failure_log_sha256": sha256(driver.LOG / "diagnose.log"),
        "wrapper_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(run.OUT / "protocol.json"),
        "diagnostic_only": {
            "torch.use_deterministic_algorithms": True,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "training_evaluation_unchanged": True,
        "tolerance_unchanged": True,
        "test_used": False,
    }
    if target.exists():
        assert run.read(target) == record
    else:
        write_json(target, record)
    if not (run.OUT / "diagnosis.json").exists():
        command = "import sys; sys.path.insert(0,'scripts'); import _pdv2_key_metric_alignment as r; import torch; torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.use_deterministic_algorithms(True); r.diagnose()"
        with (driver.LOG / "diagnose_deterministic.log").open("x") as handle:
            subprocess.run(
                [sys.executable, "-u", "-c", command],
                cwd=run.REPO,
                env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
    assert (
        run.read(run.OUT / "diagnosis.json")["protocol_sha256"]
        == record["protocol_sha256"]
    )
    started = time.time()
    for seed in run.SEEDS:
        driver.call(f"train_s{seed}", ["train", "--seed", str(seed)])
        print("TRAIN_DONE", seed, round(time.time() - started, 1), flush=True)
    jobs = [
        (s, e, q) for s in run.SEEDS for e in run.EPOCHS for q in run.protocol()["val"]
    ]

    def evaluate(job):
        seed, epoch, sequence = job
        driver.call(
            f"val_s{seed}_e{epoch}_{sequence.replace('/', '_')}",
            ["val", "--seed", str(seed), "--epoch", str(epoch), "--sequence", sequence],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(evaluate, j) for j in jobs]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print(
                "VAL_DONE",
                count,
                len(jobs),
                round(time.time() - started, 1),
                flush=True,
            )
    driver.call("finish", ["finish"])
    write_json(
        run.OUT / "integrated_validation.json",
        {
            "status": "passed",
            "amendment_sha256": sha256(target),
            "completion_sha256": sha256(run.OUT / "completion_validation.json"),
            "diagnosis_sha256": sha256(run.OUT / "diagnosis.json"),
            "test_used": False,
        },
    )
    print("ALL_COMPLETE", round(time.time() - started, 1), flush=True)


if __name__ == "__main__":
    main()
