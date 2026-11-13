#!/usr/bin/env python3
"""GPU 소비를 막지 않고 기존 CPU 작업을 끝낸 뒤 빈 전송기만 교체한다."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _review_selection_bev import read, write, sha

OUT = ROOT / "results/pdv2_review_selection_full_20261105"


def children(pid: int) -> list[int]:
    """주 스레드뿐 아니라 전송 작업 스레드가 시작한 자식까지 확인한다."""
    result = set()
    for path in Path(f"/proc/{pid}/task").glob("*/children"):
        try:
            result.update(int(p) for p in path.read_text().split())
        except FileNotFoundError:
            pass
    return sorted(result)


def main() -> None:
    """미시작 요청을 보존하고 두 활성 입력의 완료 경계에서만 교체한다."""
    lock = (OUT / "transfer_drain.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    proof = OUT / "transfer_compression_verified.json"
    assert read(proof)["all_roundtrip_file_SHA_exact"]
    gate_path = OUT / "test_request_gate.json"
    assert not gate_path.exists(), "이전 drain 기록을 먼저 복구한다"
    status_path = OUT / "transfer_drain_state.json"
    old = read(OUT / "test_relay_status.json")
    pid = old["pid"]
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    assert b"_review_selection_full_test_relay.py run" in command
    assert all(name.startswith("imlpr_") for name in old["active"])
    requests = OUT / "test_input_requests"
    initial = {p.name: read(p) for p in requests.glob("imlpr_*.json")}
    active = set(old["active"])
    allowed = [v["sequence"] for name, v in initial.items()
               if name[:-5] in active or v.get("status") == "consumed"]
    nonce = str(int(time.time()))
    backup = OUT / "transfer_drain_backups" / nonce
    backup.mkdir(parents=True)
    gate = dict(method="imlpr", allowed_sequences=allowed, controller_pid=os.getpid(),
                nonce=nonce, purpose="활성 입력 완료 후 무손실 압축 전송기 교체")
    control_path = OUT / "local_gpu_control.json"
    before = read(control_path)
    assert "imlpr" not in before.get("held_methods", [])
    brief_hold = dict(before, held_methods=[*before.get("held_methods", []), "imlpr"],
                      transfer_drain_nonce=nonce)
    manifest = {}
    new_pid = None
    old_stopped = False
    write(OUT / "transfer_drain_plan.json", dict(created_unix=time.time(), nonce=nonce,
        controller_pid=os.getpid(), old_relay_pid=pid, active_before=old["active"],
        backup=str(backup.relative_to(ROOT)), control_before=before,
        compression_proof_sha256=sha(proof),
        recovery="제어기가 사라졌으면 backup의 요청을 같은 이름으로 복원하고 gate를 제거한다. 기존 relay PID부터 확인한다."))
    try:
        # 요청 파일을 옮기는 짧은 경계에서만 새 ImLPR finisher를 막는다.
        write(control_path, brief_hold)
        while (read(OUT / "local_gpu_state.json").get("active") or {}).get("method") == "imlpr":
            time.sleep(1)
        write(gate_path, gate)
        for name, info in initial.items():
            if info["sequence"] in allowed:
                continue
            source = requests / name
            manifest[name] = sha(source)
            source.replace(backup / name)
        write(backup / "manifest.json", manifest)
        assert read(control_path) == brief_hold
        write(control_path, before)
        start = time.time()
        while True:
            state = read(OUT / "test_relay_status.json")
            assert state["pid"] == pid
            assert Path(f"/proc/{pid}").exists(), "기존 relay가 예기치 않게 종료됐다"
            # gate 설치 직전 경합으로 시작한 입력도 중단하지 않고 포함한다.
            for name in state["active"]:
                saved = backup / (name + ".json")
                if saved.exists():
                    info = read(saved)
                    gate["allowed_sequences"].append(info["sequence"])
                    saved.replace(requests / saved.name)
                    write(gate_path, gate)
            child_pids = children(pid)
            write(status_path, dict(status="draining_existing_CPU_jobs", pid=os.getpid(),
                old_relay_pid=pid, active=state["active"], children=child_pids,
                future_requests_preserved=len(list(backup.glob("imlpr_*.json"))),
                GPU_consumption_held=False, elapsed_s=time.time() - start, updated_unix=time.time()))
            if not state["active"] and not child_pids:
                # 미시작 요청은 따로 보존됐고 새 요청은 gate가 막는다.
                time.sleep(1)
                if read(OUT / "test_relay_status.json")["active"] or children(pid):
                    continue
                os.kill(pid, signal.SIGTERM)
                old_stopped = True
                for _ in range(50):
                    proc = Path(f"/proc/{pid}")
                    if not proc.exists() or (proc / "stat").read_text().split()[2] == "Z":
                        break
                    time.sleep(0.1)
                else:
                    raise RuntimeError("빈 relay 종료 확인 실패")
                break
            time.sleep(5)
        for saved in backup.glob("imlpr_*.json"):
            assert sha(saved) == manifest[saved.name]
            dest = requests / saved.name
            assert not dest.exists()
            saved.replace(dest)
        gate_path.unlink()
        log = (OUT / "logs/test_relay_compressed.log").open("a")
        child = subprocess.Popen([sys.executable, "-u", "scripts/_review_selection_full_test_relay.py",
            "run", "--workers", "2"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        new_pid = child.pid
        write(status_path, dict(status="restarted_at_idle_boundary", pid=os.getpid(),
            old_relay_pid=pid, new_relay_pid=new_pid, current_jobs_interrupted=False,
            GPU_consumption_held=False, all_future_request_files_SHA_exact=True,
            compression_proof_sha256=sha(proof), updated_unix=time.time()))
    except Exception as exc:
        # 데이터와 요청을 남겨 어떤 실패에서도 누락을 숨기지 않는다.
        for saved in backup.glob("imlpr_*.json"):
            dest = requests / saved.name
            if not dest.exists():
                saved.replace(dest)
            else:
                assert sha(saved) == sha(dest)
        gate_path.unlink(missing_ok=True)
        if read(control_path) == brief_hold:
            write(control_path, before)
        write(status_path, dict(status="failed_restored_requests", error=str(exc),
            old_relay_pid=pid, old_stopped=old_stopped, new_relay_pid=new_pid, updated_unix=time.time()))
        raise


if __name__ == "__main__":
    main()
