#!/usr/bin/env python3
"""완성 경계에서 미배정 ImLPR 시험 입력을 로컬 CPU6에 이어 배정한다."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _review_selection_bev import TEST, read, write, sha

OUT = ROOT / "results/pdv2_review_selection_full_20261105"
HOME = OUT / "local_riv_campaign"


def alive(pid: int | None) -> bool:
    """좀비를 실행 중으로 세지 않는다."""
    try:
        proc = Path(f"/proc/{pid}")
        command = (proc / "cmdline").read_bytes()
        ours = (b"_review_selection_local_test_riv.py" in command or
                b"_review_selection_local_riv_campaign.py" in command)
        return bool(pid) and ours and (proc / "stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def claim() -> Path | None:
    """전송기가 새 배정을 멈췄음을 확인한 뒤 다음 세 입력 중 하나를 맡는다."""
    relay_control = OUT / "test_relay_control.json"
    before = read(relay_control) if relay_control.exists() else {}
    if before.get("hold_new"):
        return None
    nonce = f"local_cpu_{os.getpid()}_{time.time_ns()}"
    hold = dict(before, hold_new=True, local_cpu_nonce=nonce,
                local_cpu_previous_control=before, updated_unix=time.time())
    write(relay_control, hold)
    try:
        deadline = time.time() + 35
        while time.time() < deadline:
            status = read(OUT / "test_relay_status.json")
            if status.get("hold_new") and status["updated_unix"] > hold["updated_unix"]:
                break
            time.sleep(1)
        else:
            return None
        owners_path = OUT / "test_external_input_owners.json"
        owners = read(owners_path) if owners_path.exists() else {}
        pending = []
        for sensor, seq in TEST:
            name = f"imlpr_{sensor}_{seq}"
            path = OUT / "test_input_requests" / (name + ".json")
            if path.exists() and read(path).get("status") != "consumed":
                pending.append((name, sensor, seq))
        for name, sensor, seq in pending[:3]:
            state_path = OUT / "test_relay_status" / (name + ".json")
            state = read(state_path) if state_path.exists() else {}
            if name in status["active"] or state.get("status") in ("synced", "cleaned") or name in owners:
                continue
            if read(OUT / "test_input_requests" / (name + ".json")).get("status") == "consumed":
                continue
            target = OUT / "inputs/test_imlpr/data/test_odom" / f"{sensor}_{seq}"
            if target.exists() and any(target.iterdir()):
                # 기존 실패의 부분 산출물을 다른 실행자에게 넘기지 않는다.
                continue
            free = shutil.disk_usage(OUT / "inputs/rawpack").free
            if free < 200_000_000_000:
                write(HOME / "disk_wait.json", dict(free_bytes=free,
                    required_free_bytes=200_000_000_000, updated_unix=time.time()))
                return None
            plan = dict(created_unix=time.time(), sequence=f"{sensor}/{seq}", method="imlpr",
                processes=6, GPU_used=False, selection_sha256=sha(OUT / "retained/selection_imlpr.json"),
                smoke_sha256=sha(OUT / "local_riv_smoke_verified.json"),
                source_sha256=sha(ROOT / "scripts/_imlpr_official_riv.py"),
                preparation_only_not_model_selection=True, disk_available_bytes=free,
                conservative_increment_budget_bytes=100_000_000_000,
                data_location="D:/NSD_datasets/review_selection_cache_20261105",
                ownership="relay hold_new 확인 후 원래 시험 순서의 다음3개 중 미배정 입력만 선점",
                scheduling_uses_test_metrics=False, relay_pid=status["pid"], nonce=nonce)
            plan_path = HOME / "plans" / (name + ".json")
            assert not plan_path.exists()
            write(plan_path, plan)
            owners[name] = dict(status="reserved", plan_sha256=sha(plan_path), origin="local_CPU",
                sequence=plan["sequence"], controller_pid=os.getpid())
            write(owners_path, owners)
            return plan_path
        return None
    finally:
        if read(relay_control) == hold:
            write(relay_control, before)


def main() -> None:
    """기존 Town03와 전송기 교체가 끝난 뒤 직렬로 CPU6 입력을 복구한다."""
    HOME.mkdir(exist_ok=True)
    lock = (HOME / "campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = HOME / "state.json"
    state = read(state_path) if state_path.exists() else dict(completed=[], active_plan=None)
    state.update(pid=os.getpid(), started_unix=time.time())
    relay_control = OUT / "test_relay_control.json"
    control = read(relay_control) if relay_control.exists() else {}
    if control.get("local_cpu_nonce") and "local_cpu_previous_control" in control:
        old_pid = int(control["local_cpu_nonce"].split("_")[2])
        before = control["local_cpu_previous_control"]
        expected = dict(before, hold_new=True, local_cpu_nonce=control["local_cpu_nonce"],
                        local_cpu_previous_control=before, updated_unix=control["updated_unix"])
        if not alive(old_pid) and control == expected:
            write(relay_control, before)
    while True:
        if (OUT / "retained/report_imlpr.json").exists():
            state.update(status="complete", updated_unix=time.time())
            write(state_path, state)
            return
        control_path = HOME / "control.json"
        if control_path.exists() and read(control_path).get("hold_new"):
            state.update(status="held", updated_unix=time.time())
            write(state_path, state)
            time.sleep(10)
            continue
        old = read(OUT / "local_test_riv_state.json")
        drain = read(OUT / "transfer_drain_state.json")
        if old.get("status") != "complete" or drain.get("status") != "restarted_at_idle_boundary":
            state.update(status="waiting_Town03_and_compressed_relay", updated_unix=time.time())
            write(state_path, state)
            time.sleep(10)
            continue
        if state.get("active_plan"):
            plan_path = ROOT / state["active_plan"]
        else:
            # 소유 기록을 쓴 직후 제어기가 사라진 경우에도 입력을 놓치지 않는다.
            owners = read(OUT / "test_external_input_owners.json")
            reserved = [p for p in (HOME / "plans").glob("*.json")
                        if owners.get(p.stem, {}).get("status") == "reserved"
                        and not alive(owners[p.stem].get("controller_pid"))]
            assert len(reserved) <= 1
            plan_path = reserved[0] if reserved else claim()
            if plan_path is None:
                state.update(status="waiting_unassigned_input", updated_unix=time.time())
                write(state_path, state)
                time.sleep(10)
                continue
            state["active_plan"] = str(plan_path.relative_to(ROOT))
            state.update(status="reserved", updated_unix=time.time())
            write(state_path, state)
        job_state = HOME / "jobs" / plan_path.name
        if job_state.exists() and read(job_state).get("status") == "complete":
            state["completed"].append(state["active_plan"])
            state.update(active_plan=None, child_pid=None, updated_unix=time.time())
            write(state_path, state)
            continue
        if alive(state.get("child_pid")):
            state.update(status="adopted_existing_worker", updated_unix=time.time())
            write(state_path, state)
            time.sleep(10)
            continue
        with (OUT / "logs/local_riv_campaign.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "scripts/_review_selection_local_test_riv.py",
                "--plan", str(plan_path), "--state", str(job_state)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT)
            state.update(status="running", child_pid=child.pid, updated_unix=time.time())
            write(state_path, state)
            code = child.wait()
        if code:
            state.update(status="worker_failed", returncode=code, updated_unix=time.time())
            write(state_path, state)
            raise RuntimeError(f"local CPU input failed: {plan_path}")
        assert read(job_state)["status"] == "complete"
        state["completed"].append(state["active_plan"])
        state.update(active_plan=None, child_pid=None, updated_unix=time.time())
        write(state_path, state)


if __name__ == "__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    main()
