#!/usr/bin/env python3
"""전체12센서 입력 완료를 기다려 기존 원장을 보존하며24쌍을 집계한다."""
import json
import os
from pathlib import Path
import subprocess
import time

from _review_native_helios import OUT, REPO, NAMES, write


def main() -> None:
    """GPU를 사용하지 않고 모델별 전체 추론 완료 뒤 한 번만 계산한다."""
    pidfile = OUT / "full_score_queue.pid"
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit("전체 집계기가 이미 실행 중이다")
    pidfile.write_text(str(os.getpid()))
    models = ["author", "v2_s1", "v2_s2", "v2_s3"]
    while True:
        manifest = json.loads((OUT / "input_manifest.json").read_text())
        available = sum(row["available"] for row in manifest["sequences"].values())
        for model in models:
            complete = OUT / model / "full_release_preservation.json"
            if complete.exists() or available != 12:
                continue
            if not all((OUT / model / "desc" / name / "done.json").exists() for name in NAMES):
                continue
            write(OUT / "full_score_queue_status.json", {"pid": os.getpid(), "active": model, "unix_time": time.time()})
            with (OUT / f"full_score_{model}.log").open("a") as log:
                subprocess.run(["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=", "-e", "OMP_NUM_THREADS=2",
                                "-e", "OPENBLAS_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2",
                                "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "nice", "-n", "10",
                                "python3", "-u", "scripts/_review_native_helios_immutable_score.py", "--model", model],
                               cwd=REPO, stdout=log, stderr=log, check=True)
        done = [m for m in models if (OUT / m / "full_release_preservation.json").exists()]
        write(OUT / "full_score_queue_status.json", {"pid": os.getpid(), "active": None, "completed": done,
              "total": 4, "available_input_sequences": available, "unix_time": time.time(), "complete": len(done) == 4})
        if len(done) == 4:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
