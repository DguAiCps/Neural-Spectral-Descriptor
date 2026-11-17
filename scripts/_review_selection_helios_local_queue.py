#!/usr/bin/env python3
"""검산·분담 동결 뒤 HeLiOS 후보를 ImLPR 유휴 GPU에서 이어 평가한다."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "src")]
from _review_selection_bev import read, write, sha
from _review_selection_helios_local_smoke import imlpr_ready

OUT = ROOT / "results/pdv2_review_selection_full_20261105"
STATE_ROOT = OUT / "local_helios_queue"
REMOTE = "joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/"


def worker(position: int, role: str) -> None:
    """원래 평가기의 경로만 동일 입력 사본으로 연결한다."""
    from _review_selection_local_reservation import require_available
    require_available()
    import _review_selection_full_eval as evaluator
    plan = read(OUT / ("local_helios_plan.json" if role == "val" else "local_helios_test_plan.json"))
    job = plan["jobs"][position]
    assert os.environ["REVIEW_SELECTION_EXECUTOR"] == plan["executor"]
    assert sha(OUT / "retained_protocol.json") == plan["protocol_sha256"]
    inputs = OUT / ("local_helios_inputs" if role == "val" else "local_helios_test_inputs")
    assert read(inputs / "complete.json")["all_files_SHA_exact"]
    root = Path(read(inputs / "plan.json")["target_container_root"])
    original_configure = evaluator.configure

    def configure(method: str):
        module = original_configure(method)
        module.DATA = root
        return module

    evaluator.configure = configure
    if role == "test":
        assert sha(OUT / "retained/selection_helios.json") == plan["selection_sha256"]
        import _helios_official_prep as prep
        prep.DATA = root
    args = argparse.Namespace(method="helios", seed=job["seed"], candidate=job["candidate"],
        sequence=job["sequence"], role=role, scope="retained", query_budget=128,
        query_offset=0, forward_budget=plan["forward_budget"])
    try:
        evaluator.main(args)
    finally:
        folder = evaluator.folder("helios", args.seed, args.candidate, "retained", role)
        key = args.sequence.replace("/", "_")
        for path in (folder / f"descriptors_{key}.npy", folder / f"descriptor_progress_{key}.json"):
            if path.exists():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())


def upload(job: dict, role: str) -> None:
    """완료 JSON과 연결된 원장을 같이 전송한 뒤 서버 해시를 확인한다."""
    home = OUT / f"retained/{role}/helios/s{job['seed']}" / job["candidate"]
    record_path = home / (job["sequence"].replace("/", "_") + ".json")
    record = read(record_path)
    ledger = ROOT / record["ledger"]
    assert sha(ledger) == record["ledger_sha256"]
    listing = STATE_ROOT / "upload_files.txt"
    for artifact in (ledger, record_path):
        listing.write_text(str(artifact.relative_to(ROOT)) + "\n")
        subprocess.run(["rsync", "-aO", "--no-owner", "--no-group", "--no-perms", "--fsync",
            "--files-from=" + str(listing), str(ROOT) + "/", REMOTE], check=True)
    # 경로는 동결 명세에서 온 리포 내부 경로이며 shell을 통하지 않는 SSH 인자다.
    result = subprocess.check_output(["ssh", "joms-gpu", "sha256sum",
        "/home/qkrrlans00/Neural-Spectral-Codec/" + record["ledger"]], text=True).split()[0]
    assert result == record["ledger_sha256"]


def verify_input_durability(role: str) -> None:
    """이 부팅에서 입력 전체 SHA와 디스크 반영을 확인한다."""
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    proof_path = STATE_ROOT / "input_current_boot_verified.json"
    if proof_path.exists() and read(proof_path).get("boot_id") == boot:
        return
    source = OUT / ("local_helios_inputs" if role == "val" else "local_helios_test_inputs")
    assert read(source / "complete.json")["all_files_SHA_exact"]
    root = Path(read(source / "plan.json")["target_host_root"])
    total = 0
    tick = time.time()
    for path in sorted((source / "source").glob("*.json")):
        manifest = read(path)
        key = manifest["sequence"].replace("/", "_")
        for name, digest in manifest["files"].items():
            frame = root / "test_odom" / key / name
            assert sha(frame) == digest
            with frame.open("rb") as stream:
                os.fsync(stream.fileno())
            total += 1
        write(STATE_ROOT / "input_verification_progress.json", dict(sequence=manifest["sequence"],
            files=total, elapsed_s=time.time()-tick, updated_unix=time.time()))
    write(proof_path, dict(boot_id=boot, files=total, all_SHA_exact=True, fsynced=True,
        completed_unix=time.time(), source_complete_sha256=sha(source / "complete.json")))


def run(role: str) -> None:
    """고정한 후보 집합을 바꾸지 않고 완료 경계에서 GPU를 반납한다."""
    STATE_ROOT.mkdir(exist_ok=True)
    lock = (STATE_ROOT / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = OUT / ("local_helios_plan.json" if role == "val" else "local_helios_test_plan.json")
    plan = read(plan_path)
    proof = read(OUT / "helios_local_execution_smoke/accepted.json")
    assert proof["accepted_for_execution"] and sha(OUT / "helios_local_execution_smoke/accepted.json") == plan["execution_proof_sha256"]
    assert sha(OUT / "retained_protocol.json") == plan["protocol_sha256"]
    state_path = STATE_ROOT / "state.json"
    state = read(state_path) if state_path.exists() else dict(position=0, failures={})
    state.update(pid=os.getpid(), started_unix=time.time(), plan_sha256=sha(plan_path))
    control_path = OUT / "local_gpu_control.json"
    previous_path = STATE_ROOT / "held_control.json"
    if previous_path.exists():
        previous = read(previous_path)
        old_cmd = Path(f"/proc/{previous['pid']}/cmdline")
        if not old_cmd.exists() or b"_review_selection_helios_local_queue.py" not in old_cmd.read_bytes():
            if read(control_path) == previous["held"]:
                write(control_path, previous["before"])
    state.update(status="CPU_input_verification", updated_unix=time.time())
    write(state_path, state)
    inputs = OUT / ("local_helios_inputs" if role == "val" else "local_helios_test_inputs")
    while not (inputs / "complete.json").exists():
        state.update(status="waiting_full_input_copy", updated_unix=time.time())
        write(state_path, state)
        time.sleep(10)
    state.update(status="CPU_input_verification", updated_unix=time.time())
    write(state_path, state)
    verify_input_durability(role)
    while state["position"] < len(plan["jobs"]):
        assert sha(plan_path) == state["plan_sha256"]
        job = plan["jobs"][state["position"]]
        result = OUT / f"retained/{role}/helios/s{job['seed']}" / job["candidate"] / (job["sequence"].replace("/", "_") + ".json")
        if result.exists():
            state.update(status="uploading", active=job, updated_unix=time.time())
            write(state_path, state)
            upload(job, role)
            state["position"] += 1
            state.update(status="uploaded", active=None, updated_unix=time.time())
            write(state_path, state)
            continue
        from _review_selection_local_reservation import held as user_gpu_reserved
        if user_gpu_reserved():
            state.update(status="held_for_user_local_gpu", active=None, updated_unix=time.time())
            write(state_path, state)
            time.sleep(5)
            continue
        before = read(control_path)
        queue = read(OUT / "local_gpu_state.json")
        own_control = STATE_ROOT / "control.json"
        held_by_operator = own_control.exists() and read(own_control).get("hold_new", False)
        if queue.get("active") or before.get("held_methods") or imlpr_ready() or held_by_operator:
            state.update(status="waiting_priority", active=None, updated_unix=time.time())
            write(state_path, state)
            time.sleep(5)
            continue
        nonce = f"helios_local_{time.time_ns()}"
        held = dict(before, held_methods=read(OUT / "local_gpu_plan.json")["methods"],
                    local_helios_nonce=nonce, updated_unix=time.time())
        write(previous_path, dict(before=before, held=held, pid=os.getpid()))
        write(control_path, held)
        try:
            deadline = time.time() + 90
            while time.time() < deadline:
                assert read(control_path) == held
                queue = read(OUT / "local_gpu_state.json")
                if (not queue.get("active") and queue.get("status") == "held"
                        and queue["updated_unix"] > held["updated_unix"]):
                    break
                time.sleep(1)
            else:
                raise RuntimeError("새 GPU 유휴 경계 확인 실패")
            if imlpr_ready():
                continue
            command = ["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=0", "-e", "NVIDIA_TF32_OVERRIDE=0",
                "-e", "OMP_NUM_THREADS=4", "-e", "MKL_NUM_THREADS=4", "-e", "OPENBLAS_NUM_THREADS=1",
                "-e", f"SEED={job['seed']}", "-e", "REVIEW_SELECTION_EXECUTOR=" + plan["executor"],
                "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "python3", "-u",
                "scripts/_review_selection_helios_local_queue.py", "worker", "--position", str(state["position"]), "--role", role]
            with (STATE_ROOT / f"worker_{state['position']:03d}.log").open("a") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                state.update(status="GPU_chunk", active=job, child_pid=child.pid, updated_unix=time.time())
                write(state_path, state)
                code = child.wait()
            if code == 139:
                key = str(state["position"])
                failures = state.setdefault("failures", {})
                failures[key] = failures.get(key, 0) + 1
                repeated = state.setdefault("consecutive_segfaults", {})
                repeated[key] = repeated.get(key, 0) + 1
                state.update(status="retry_saved_cursor_after_segfault", returncode=code,
                             updated_unix=time.time())
                write(state_path, state)
                if repeated[key] >= 3:
                    raise RuntimeError(f"HeLiOS local worker failed 3 consecutive times: {job}, {code}")
            elif code not in (0, 75):
                state.update(status="worker_failed", returncode=code, updated_unix=time.time())
                write(state_path, state)
                raise RuntimeError(f"HeLiOS local worker failed: {job}, {code}")
            else:
                state.setdefault("consecutive_segfaults", {})[str(state["position"])] = 0
            state.update(last_returncode=code, updated_unix=time.time())
            write(state_path, state)
        finally:
            if read(control_path) == held:
                write(control_path, before)
            else:
                raise RuntimeError("외부 GPU 제어 변경; 기존 제어 복원 생략")
        time.sleep(2)
    state.update(status=f"complete_{role}_partition", active=None, completed_unix=time.time())
    write(state_path, state)


if __name__ == "__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker"))
    parser.add_argument("--position", type=int)
    parser.add_argument("--role", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if args.role == "test":
        STATE_ROOT = OUT / "local_helios_test_queue"
    run(args.role) if args.action == "run" else worker(args.position, args.role)
