#!/usr/bin/env python3
"""검증 선택 동결 후 HeLiOS 시험을 시드 단위로 고정 배정한다."""
from __future__ import annotations

import argparse
import fcntl
import itertools
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "src")]
from _review_selection_bev import TEST, read, write, sha
from _review_selection_full_finish import evaluation_choices, runtime_control_needed

OUT = ROOT / "results/pdv2_review_selection_full_20261105"
REMOTE_ROOT = "/home/qkrrlans00/Neural-Spectral-Codec"
STATE_ROOT = OUT / "helios_test_partition"


def live(pid: int | None, needle: bytes) -> bool:
    """PID 재사용과 좀비를 제외한다."""
    try:
        process = Path(f"/proc/{pid}")
        return needle in (process / "cmdline").read_bytes() and (process / "stat").read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def local_seeds(counts: dict, local_weight: float) -> list:
    """고유 추론모델 개수만으로 두 장치의 예상 최장 시간을 줄인다."""
    seeds = sorted(counts, key=int)
    candidates = []
    for flags in itertools.product((False, True), repeat=len(seeds)):
        chosen = [seed for seed, take in zip(seeds, flags) if take and counts[seed]]
        local_count = sum(counts[seed] for seed in chosen)
        server_count = sum(counts.values()) - local_count
        candidates.append((max(server_count, local_count * local_weight), local_count, chosen))
    return min(candidates)[2]


def server_inventory() -> None:
    """서버 CPU 입력 목록 생성만 중복 없이 시작한다."""
    home = OUT / "local_helios_test_inputs"
    (home / "source").mkdir(parents=True, exist_ok=True)
    marker = home / "source_launch.json"
    previous = read(marker) if marker.exists() else {}
    if live(previous.get("pid"), b"_review_selection_helios_local_inputs.py"):
        print(previous["pid"], flush=True)
        return
    with (home / "inventory.log").open("a") as log:
        child = subprocess.Popen([sys.executable, "-u", "scripts/_review_selection_helios_local_inputs.py",
            "inventory", "--role", "test"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    write(marker, dict(pid=child.pid, started_unix=time.time()))
    print(child.pid, flush=True)


def sync_paths(paths: list) -> None:
    """시험 분담 명세와 실행 코드만 서버로 동기화한다."""
    paths = list(dict.fromkeys([*paths, ROOT / "scripts/_review_selection_local_reservation.py"]))
    listing = STATE_ROOT / "upload.txt"
    listing.write_text("\n".join(str(path.relative_to(ROOT)) for path in paths) + "\n")
    subprocess.run(["rsync", "-aO", "--no-owner", "--no-group", "--no-perms", "--fsync",
        "--files-from=" + str(listing), str(ROOT) + "/", "joms-gpu:" + REMOTE_ROOT + "/"], check=True)


def run() -> None:
    """시험값을 읽지 않고 선택 이후의 입력 복구·평가를 자동으로 잇는다."""
    STATE_ROOT.mkdir(exist_ok=True)
    lock = (STATE_ROOT / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = STATE_ROOT / "state.json"
    state = read(status_path) if status_path.exists() else {}
    state.update(pid=os.getpid(), started_unix=time.time())
    selection_path = OUT / "retained/selection_helios.json"
    while not selection_path.exists():
        from _review_selection_local_reservation import held as user_gpu_reserved
        state.update(status="held_for_user_local_gpu" if user_gpu_reserved() else "waiting_frozen_common_validation", updated_unix=time.time())
        write(status_path, state)
        time.sleep(30)
    from _review_selection_local_reservation import held as user_gpu_reserved
    while user_gpu_reserved():
        state.update(status="held_for_user_local_gpu", updated_unix=time.time())
        write(status_path, state)
        time.sleep(15)
    frozen = read(selection_path)
    assert not frozen["test_used_for_selection"]
    remote_sha = subprocess.check_output(["ssh", "joms-gpu", "sha256sum",
        REMOTE_ROOT + "/" + str(selection_path.relative_to(ROOT))], text=True).split()[0]
    assert remote_sha == sha(selection_path)
    policy_path = OUT / "helios_test_partition_policy.json"
    policy = read(policy_path)
    target = OUT / "local_helios_test_plan.json"
    if not target.exists():
        choices = {seed: evaluation_choices("helios", seed, "retained", frozen) for seed in frozen["selected"]}
        counts = {seed: (len({row["id"] for row in rows.values()})
            if runtime_control_needed("helios", seed, "retained", frozen) else 0)
            for seed, rows in choices.items()}
        chosen_seeds = local_seeds(counts, policy["relative_local_model_time"])
        jobs = [dict(method="helios", seed=int(seed), candidate=name, sequence=f"{sensor}/{seq}", role="test")
                for sensor, seq in TEST for seed in chosen_seeds
                for name in dict.fromkeys(row["id"] for row in choices[seed].values())]
        write(target, dict(created_unix=time.time(), role="test", jobs=jobs, local_seeds=chosen_seeds,
            fresh_model_counts=counts, executor="local_helios_frozen_test", forward_budget=512,
            selection_sha256=sha(selection_path), policy_sha256=sha(policy_path),
            protocol_sha256=sha(OUT / "retained_protocol.json"),
            execution_proof_sha256=sha(OUT / "helios_local_execution_smoke/accepted.json"),
            test_values_read_for_assignment=False, same_seed_choices_use_same_device=True))
    plan = read(target)
    assert plan["selection_sha256"] == sha(selection_path)
    sync_paths([target, ROOT / "scripts/_review_selection_prepare_upload_dirs.py"])
    uid = subprocess.check_output(["ssh", "joms-gpu", "id", "-u"], text=True).strip()
    gid = subprocess.check_output(["ssh", "joms-gpu", "id", "-g"], text=True).strip()
    assert uid.isdigit() and gid.isdigit()
    subprocess.run(["ssh", "joms-gpu", "docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=",
        "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "python3",
        "scripts/_review_selection_prepare_upload_dirs.py", "--role", "test", "--uid", uid, "--gid", gid], check=True)
    if not plan["jobs"]:
        state.update(status="complete_no_local_test_partition", updated_unix=time.time())
        write(status_path, state)
        return
    inputs = OUT / "local_helios_test_inputs"
    inputs.mkdir(exist_ok=True)
    if not (inputs / "plan.json").exists():
        data = Path("/mnt/d/NSD_datasets/review_selection_cache_20261105/helios_local_test/data")
        data.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(data).free
        assert free > policy["minimum_D_free_bytes"]
        write(inputs / "plan.json", dict(created_unix=time.time(), target_host_root=str(data),
            target_container_root=str(data).replace("/mnt/d/NSD_datasets", "/workspace/data"),
            selection_sha256=sha(selection_path), expected_bytes=27_000_000_000,
            available_bytes=free, GPU_used=False, source="server preserved official common test input"))
    sync_paths([inputs / "plan.json", ROOT / "scripts/_review_selection_helios_local_inputs.py",
                Path(__file__)])
    server_pid = subprocess.check_output(["ssh", "joms-gpu", "python3", REMOTE_ROOT +
        "/scripts/_review_selection_helios_test_partition.py", "server_inventory"], text=True).strip()
    state["server_input_inventory_pid"] = int(server_pid)
    if not (inputs / "complete.json").exists():
        if not live(state.get("transfer_pid"), b"_review_selection_helios_local_inputs.py"):
            with (inputs / "transfer.log").open("a") as log:
                child = subprocess.Popen([sys.executable, "-u", "scripts/_review_selection_helios_local_inputs.py",
                    "transfer", "--role", "test"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
            state["transfer_pid"] = child.pid
        while not (inputs / "complete.json").exists():
            assert live(state["transfer_pid"], b"_review_selection_helios_local_inputs.py"), "test input transfer failed"
            state.update(status="copying_frozen_test_inputs", updated_unix=time.time())
            write(status_path, state)
            time.sleep(15)
    queue_state = OUT / "local_helios_test_queue/state.json"
    previous = read(queue_state) if queue_state.exists() else {}
    if not live(previous.get("pid"), b"_review_selection_helios_local_queue.py") and previous.get("status") != "complete_test_partition":
        (OUT / "local_helios_test_queue").mkdir(exist_ok=True)
        with (OUT / "local_helios_test_queue/controller.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "scripts/_review_selection_helios_local_queue.py",
                "run", "--role", "test"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        state["local_test_runner_pid"] = child.pid
    else:
        state["local_test_runner_pid"] = previous.get("pid")
    while True:
        actual = read(queue_state) if queue_state.exists() else {}
        if actual.get("status") == "complete_test_partition":
            state.update(status="complete_local_test_partition", completed_unix=time.time())
            write(status_path, state)
            return
        assert live(state["local_test_runner_pid"], b"_review_selection_helios_local_queue.py"), "local frozen test worker stopped"
        state.update(status="local_test_running", position=actual.get("position", 0),
                     total=len(plan["jobs"]), updated_unix=time.time())
        write(status_path, state)
        time.sleep(30)


if __name__ == "__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "server_inventory"))
    args = parser.parse_args()
    if args.action == "server_inventory":
        server_inventory()
    else:
        try:
            run()
        except Exception as exc:
            path = STATE_ROOT / "state.json"
            state = read(path) if path.exists() else {}
            state.update(status="failed", error=str(exc), updated_unix=time.time())
            write(path, state)
            raise
