#!/usr/bin/env python3
"""완료된 HeLiOS 센서 묶음의 공식 CPU 검색과 독립 집계를 실행한다."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105/helios"


def main() -> None:
    history = []
    last = {}
    while True:
        manifest = json.loads((OUT / "input_manifest.json").read_text())
        names = [name for name, row in manifest["sequences"].items() if row["available"]]
        for model in ["author", "v2_s1", "v2_s2", "v2_s3"]:
            if last.get(model) == names:
                continue
            if not all((OUT / model / "desc" / name / "done.json").exists() for name in names):
                continue
            log = OUT / f"score_{model}_{len(names)}sequences.log"
            with log.open("a") as stream:
                result = subprocess.run(["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=", "-e", "OMP_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2", "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "nice", "-n", "10", "python3", "-u", "scripts/_review_native_helios.py", "score", "--model", model], stdout=stream, stderr=stream)
            last[model] = names
            history.append({"model": model, "sequences": names, "returncode": result.returncode, "unix_time": time.time()})
            (OUT / "score_queue_status.json").write_text(json.dumps({"history": history}, indent=2))
        if all((OUT / model / "native_report.json").exists() for model in ["author", "v2_s1", "v2_s2", "v2_s3"]):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
