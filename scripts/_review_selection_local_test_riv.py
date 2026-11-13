#!/usr/bin/env python3
"""서버에 아직 배정되지 않은 시험 입력 한 시퀀스를 로컬 CPU에서 복구한다."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _review_selection_bev import read, write, sha

OUT = ROOT / "results/pdv2_review_selection_full_20261105"


def main(plan_path: Path, state_path: Path) -> None:
    """검산된 원래 함수와 CPU6개를 사용하고 완성 뒤에만 입력을 공개한다."""
    lock = (OUT / "local_test_riv.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(plan_path)
    assert plan["processes"] == 6 and not plan["GPU_used"]
    assert sha(OUT / "local_riv_smoke_verified.json") == plan["smoke_sha256"]
    smoke = read(OUT / "local_riv_smoke_verified.json")
    assert smoke["all_files_SHA_exact"]
    assert sha(ROOT / "scripts/_imlpr_official_riv.py") == smoke["source_implementation_sha256"]
    selection = OUT / "retained/selection_imlpr.json"
    assert sha(selection) == plan["selection_sha256"]
    sensor, seq = plan["sequence"].split("/")
    name = "imlpr_" + sensor + "_" + seq
    owner_path = OUT / "test_external_input_owners.json"
    owner = read(owner_path)
    assert owner[name]["plan_sha256"] == sha(plan_path)
    target = OUT / "inputs/test_imlpr/data/test_odom" / f"{sensor}_{seq}"
    target.mkdir(parents=True, exist_ok=True)
    base = ["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=", "-e", "OMP_NUM_THREADS=1",
            "-e", "MKL_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
            "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "python3", "-u"]
    owner[name].update(status="running", controller_pid=os.getpid())
    write(owner_path, owner)
    try:
        for stage, script, extra in (
            ("rawpack", "_review_selection_full_rawpack.py", []),
            ("original_RIV_CPU", "_review_selection_full_server_riv.py", ["--processes", "6"]),
        ):
            with (OUT / "logs" / f"local_test_riv_{sensor}_{seq}.log").open("a") as log:
                child = subprocess.Popen(base + ["scripts/" + script, "--role", "test",
                    "--sequence", plan["sequence"], *extra], stdout=log, stderr=subprocess.STDOUT)
                write(state_path, dict(status=stage, pid=os.getpid(), child_pid=child.pid,
                    sequence=plan["sequence"], plan_sha256=sha(plan_path), updated_unix=time.time()))
                assert child.wait() == 0, stage
        hashes = read(target / "server_generated_hashes.json")
        done = read(target / "done.json")
        assert len(hashes) == done["n_frames"]
        assert sha(target / "server_generated_hashes.json") == done["generated_hash_manifest_sha256"]
        for filename, digest in hashes.items():
            assert sha(target / filename) == digest, filename
        raw = OUT / "inputs/rawpack/test" / f"{sensor}_{seq}"
        write(OUT / "test_input_provenance" / (name + ".json"), dict(completed_unix=time.time(),
            selection_sha256=sha(selection), rawpack=read(raw / "done.json"),
            rawpack_verification=read(raw / "verified.json"), done=done,
            generated_RIV_all_files_SHA_exact=True, generated_RIV_file_sha256=hashes,
            execution_origin="local_CPU_no_network_transfer", local_parallel_smoke_sha256=sha(OUT / "local_riv_smoke_verified.json"),
            original_RIV_equivalence_proof_sha256=sha(OUT / "server_riv_smoke/validated.json")))
        write(target / "transfer_complete.json", dict(done_sha256=sha(target / "done.json"),
            completed_unix=time.time(), host="local", network_transfer_required=False))
        write(OUT / "test_relay_status" / (name + ".json"), dict(status="synced", host="local_CPU",
            method="imlpr", sequence=plan["sequence"], updated_unix=time.time()))
        owner = read(owner_path)
        owner[name].update(status="complete", completed_unix=time.time())
        write(owner_path, owner)
        write(state_path, dict(status="complete", sequence=plan["sequence"], frames=done["n_frames"],
            all_files_SHA_exact=True, GPU_used=False, completed_unix=time.time()))
    except Exception as exc:
        owner = read(owner_path)
        owner[name].update(status="worker_failed", error=str(exc))
        write(owner_path, owner)
        write(state_path, dict(status="worker_failed", error=str(exc), pid=os.getpid(), updated_unix=time.time()))
        raise


if __name__ == "__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=OUT / "local_test_riv_plan.json")
    parser.add_argument("--state", type=Path, default=OUT / "local_test_riv_state.json")
    args = parser.parse_args()
    main(args.plan, args.state)
