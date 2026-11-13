#!/usr/bin/env python3
"""ImLPR 원래 RIV의 로컬 CPU6 병렬 실행을 서버 완성 파일과 검산한다."""
from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "src")]
from _review_selection_bev import read, write, sha
from _review_selection_full_inputs import OUT, cache
from _review_selection_full_server_riv import initialize, calculate


def block(rows: list) -> dict:
    """공식 계산을 유지하고 자식별 최대 RSS만 덧붙인다."""
    result = calculate(rows)
    result["peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return result


def main() -> None:
    """새 임시 위치에서만 병렬 계산하고 모든 NPY SHA를 확인한다."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    plan_path = OUT / "local_riv_smoke_plan.json"
    plan = read(plan_path)
    assert sha(OUT / "retained/selection_imlpr.json") == plan["selection_sha256"]
    sensor, seq = plan["sequence"].split("/")
    ids = np.load(cache(sensor, seq, "test"))["scan_ids"]
    home = OUT / "inputs/local_riv_smoke"
    assert home.is_symlink() and str(home.resolve()).startswith("/workspace/data/")
    rows = [(p, int(ids[p])) for p in plan["positions"]]
    assert all(not (home / f"{p:06d}.npy").exists() for p, _ in rows)
    blocks = [rows[i:i + 4] for i in range(0, len(rows), 4)]
    tick = time.time()
    completed = []
    with multiprocessing.get_context("spawn").Pool(6, initializer=initialize,
            initargs=("test", sensor, seq, str(home))) as pool:
        for result in pool.imap_unordered(block, blocks):
            completed.append(result)
            print("LOCAL_RIV_SMOKE", sum(len(x["positions"]) for x in completed), len(rows),
                  round(time.time() - tick, 2), flush=True)
    elapsed = time.time() - tick
    for pos, _ in rows:
        assert sha(home / f"{pos:06d}.npy") == plan["server_image_sha256"][str(pos)], pos
    write(OUT / "local_riv_smoke_verified.json", dict(completed_unix=time.time(),
        all_files_SHA_exact=True, plan_sha256=sha(plan_path), frames=len(rows),
        elapsed_s=elapsed, observed_frames_per_s=len(rows) / elapsed,
        child_observations=completed, processes=6, GPU_used=False,
        source_implementation_sha256=sha(ROOT / "scripts/_imlpr_official_riv.py"),
        note="공유 실행 중 작은 병렬 묶음 관측이며 전체시퀀스 처리율이나 논문 효율 수치가 아니다."))


if __name__ == "__main__":
    main()
