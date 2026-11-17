#!/usr/bin/env python3
"""서버 HeLiOS 완료 후 원장 전수 검산과 로컬 전달을 GPU 없이 잇는다."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _review_selection_bev import TEST, read, write, sha

OUT = ROOT / "results/pdv2_review_selection_full_20261105"
REMOTE = "/home/qkrrlans00/Neural-Spectral-Codec"
REL = OUT.relative_to(ROOT)


def main() -> None:
    """서버 보고서를 기다려 독립 검산과 로컬 전달을 잇는다."""
    lock = (OUT / "helios_delivery.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = OUT / "helios_delivery_state.json"
    report = OUT / "retained/report_helios.json"
    while not report.exists():
        write(status, dict(status="waiting_server_report", pid=os.getpid(), updated_unix=time.time()))
        time.sleep(30)
    frozen = read(OUT / "retained/selection_helios.json")
    assert not frozen["test_used_for_selection"]
    assert read(report)["selection"] == frozen
    # 서버에는 현재 NSD 전체 결과가 없으므로 동결 이후 공통 질의·정답만 무손실 복사한다.
    home = OUT / "test_query_reference"
    home.mkdir(exist_ok=True)
    rows = {}
    for sensor, seq in TEST:
        source = ROOT / f"results/pdv2_diffattn_contrast_test_20261104/test/{sensor}_{seq}/diff_s1.npz"
        dest = home / f"{sensor}_{seq}.npz"
        with np.load(source) as original:
            arrays = {k: original[k] for k in ("query_idx", "min_past_gt_dist")}
        np.savez_compressed(dest, **arrays)
        with np.load(dest) as restored:
            for key, value in arrays.items():
                np.testing.assert_array_equal(value, restored[key])
        rows[sensor + "/" + seq] = dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source),
                                       copy=str(dest.relative_to(ROOT)), copy_sha256=sha(dest))
    write(home / "manifest.json", dict(created_unix=time.time(), all_arrays_exact=True,
                                      selection_helios_sha256=sha(OUT / "retained/selection_helios.json"), sequences=rows))
    subprocess.run(["rsync", "-a", str(ROOT / "scripts/_review_selection_full_verify.py"),
                    "joms-gpu:" + REMOTE + "/scripts/"], check=True)
    subprocess.run(["rsync", "-a", str(home), "joms-gpu:" + REMOTE + "/" + str(REL) + "/"], check=True)
    write(status, dict(status="server_CPU_verification", pid=os.getpid(), updated_unix=time.time()))
    with (OUT / "logs/independent_verification_helios.log").open("a") as log:
        subprocess.run(["ssh", "joms-gpu", "docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=",
            "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
            "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "python3", "-u",
            "scripts/_review_selection_full_verify.py", "--method", "helios"], stdout=log,
            stderr=subprocess.STDOUT, check=True)
    proof_path = OUT / "independent_verification_helios.json"
    subprocess.run(["rsync", "-a", "joms-gpu:" + REMOTE + "/" + str(proof_path.relative_to(ROOT)), str(proof_path)], check=True)
    assert read(proof_path)["report_sha256"] == sha(report)
    from _review_selection_full_delivery import summarize
    summarize("helios")
    target = OUT / "final_delivery_helios.json"
    subprocess.run(["rsync", "-a", str(target), "joms-gpu:" + REMOTE + "/" + str(REL) + "/"], check=True)
    write(status, dict(status="complete", completed_unix=time.time(), main_table_changed=False))


if __name__ == "__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    main()
