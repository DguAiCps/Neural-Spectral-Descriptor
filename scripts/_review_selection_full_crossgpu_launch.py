#!/usr/bin/env python3
"""작업 경계에서 검산 하나를 실행하고 기존 작업열을 반드시 재개한다."""
import argparse
from pathlib import Path
import subprocess
import time

from _review_selection_bev import read, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pdv2_review_selection_full_20261105"


def main(host: str) -> None:
    """계산 중인 작업을 중단하지 않고 경계 대기 후 smoke를 실행한다."""
    local = host == "local"
    state = OUT / ("local_gpu_state.json" if local else "queue_state.json")
    control = OUT / ("local_gpu_control.json" if local else "queue_control.json")
    expected = "held" if local else "held_pending_work"
    try:
        deadline = time.time() + 1800
        while read(state)["status"] != expected:
            if time.time() > deadline:
                raise TimeoutError("queue did not reach requested boundary")
            time.sleep(5)
        subprocess.run(["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=0", "-e", "SEED=1",
                        "-e", "OMP_NUM_THREADS=4", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=4",
                        "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "python3", "-u",
                        "scripts/_review_selection_full_crossgpu.py", "--tag", host], check=True)
    finally:
        if local:
            value = read(control)
            value.update(held_methods=[], reason="교차GPU 검산 종료; 통과한 방법 작업열 재개", updated_unix=time.time())
        else:
            value = read(OUT / "before_imlpr_smoke_control.json")
            value.update(updated_unix=time.time())
        write(control, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("local", "server"), required=True)
    main(parser.parse_args().host)
