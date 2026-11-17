#!/usr/bin/env python3
"""ImLPR 입력 대기 뒤에만 DiSCO를 검산하고 로컬 후순위 작업열로 넘긴다."""
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
REMOTE = "/home/qkrrlans00/Neural-Spectral-Codec"
REL = OUT.relative_to(ROOT)


def remote_json(name: str) -> dict:
    """서버의 현재 제어·상태 기록을 읽는다."""
    import json
    return json.loads(subprocess.check_output(["ssh", "joms-gpu", "cat", REMOTE + "/" + str(REL / name)]))


def put_remote(name: str, value: dict) -> None:
    """이번 실험의 서버 제어 기록만 전송한다."""
    path = OUT / ("disco_transfer_" + name)
    write(path, value)
    subprocess.run(["rsync", "-aO", "--no-owner", "--no-group", "--no-perms", str(path),
                    "joms-gpu:" + REMOTE + "/" + str(REL / name)], check=True)


def background(script: str, arguments: list, name: str) -> int:
    """로그를 남기는 독립 세션을 시작한다."""
    log = (OUT / "logs" / name).open("a")
    child = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / script), *arguments], cwd=ROOT,
                             stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
    return child.pid


def main() -> None:
    """검산된 DiSCO 작업만 입력 대기 시간에 추가하고 완료를 전달한다."""
    lock = (OUT / "disco_adoption.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = OUT / "disco_adoption_state.json"
    handoff = OUT / "disco_local_handoff.json"
    if not handoff.exists():
        assert read(OUT / "disco_raw_restore_state.json")["all_files_exact"]
        while True:
            state = read(OUT / "local_gpu_state.json")
            ready = (OUT / "retained/selection_imlpr.json").exists()
            idle = state["status"] in ("waiting_input_or_smoke", "complete") and not state.get("active")
            if ready and idle:
                break
            write(status_path, dict(status="waiting_ImLPR_frozen_and_GPU_idle", pid=os.getpid(),
                                    imlpr_position=state["positions"].get("imlpr"), updated_unix=time.time()))
            time.sleep(15)
        server_state = remote_json("queue_state.json")
        assert 2 <= server_state["positions"]["disco"] <= 540
        assert not server_state.get("active") or server_state["active"].get("method") != "disco"
        local_control = read(OUT / "local_gpu_control.json")
        assert not local_control.get("held_methods"), "별도 보류 지시가 있으면 인계하지 않는다"
        nonce = "disco_local_" + str(int(time.time()))
        held = dict(local_control, held_methods=read(OUT / "local_gpu_plan.json")["methods"],
                    opportunistic_handoff_nonce=nonce, updated_unix=time.time())
        write(OUT / "local_gpu_control.json", held)
        while True:
            assert read(OUT / "local_gpu_control.json") == held, "다른 제어 변경을 존중해 인계를 중단한다"
            state = read(OUT / "local_gpu_state.json")
            if state["status"] in ("held", "complete") and not state.get("active"):
                break
            time.sleep(2)
        old_pid = state["pid"]
        proc = Path(f"/proc/{old_pid}")
        if proc.exists():
            assert b"_review_selection_full_local_queue.py" in (proc / "cmdline").read_bytes()
            assert not (proc / "task" / str(old_pid) / "children").read_text().strip()
            os.kill(old_pid, signal.SIGTERM)
            for _ in range(50):
                if not proc.exists() or (proc / "stat").read_text().split()[2] == "Z":
                    break
                time.sleep(0.1)
        adopted = False
        before = {name: read(OUT / name) for name in
                  ("local_gpu_plan.json", "local_gpu_state.json", "execution_hosts.json")}
        remote_control = None
        remote_held = None
        policy_created = False
        try:
            write(status_path, dict(status="GPU_smoke", pid=os.getpid(), previous_queue_pid=old_pid, updated_unix=time.time()))
            proofs = []
            for sequence in ("kitti/09", "kitti360/0004"):
                log = OUT / "logs" / ("disco_local_smoke_" + sequence.replace("/", "_") + ".log")
                with log.open("a") as stream:
                    code = subprocess.call(["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=0", "-e", "NVIDIA_TF32_OVERRIDE=0",
                        "-e", "OMP_NUM_THREADS=4", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=4",
                        "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "python3", "-u",
                        "scripts/_review_selection_full_local_smoke.py", "--method", "disco", "--sequence", sequence],
                        stdout=stream, stderr=subprocess.STDOUT)
                if code:
                    raise RuntimeError(f"DiSCO smoke failed: {sequence}, code {code}")
                proof_path = OUT / "local_gpu_smoke" / ("disco_" + sequence.replace("/", "_") + ".json")
                proof = read(proof_path)
                assert proof["all_arrays_exact"], (sequence, proof["differences"])
                proofs.append(dict(path=str(proof_path.relative_to(ROOT)), sha256=sha(proof_path)))
            current = read(OUT / "local_gpu_control.json")
            assert current.get("opportunistic_handoff_nonce") == nonce
            remote_control = remote_json("queue_control.json")
            remote_held = dict(remote_control,
                held_streams=list(dict.fromkeys([*remote_control["held_streams"], "disco"])),
                local_disco_owner_nonce=nonce, updated_unix=time.time())
            put_remote("queue_control.json", remote_held)
            # 이미 끝난 서버 원장은 실제 경계까지 복사하고 해시를 검산한다.
            # smoke 동안 서버가 다른 원장을 끝내도 중복 추론하지 않는다.
            while True:
                server_state = remote_json("queue_state.json")
                if not server_state.get("active") or server_state["active"].get("method") != "disco":
                    break
                time.sleep(2)
            start_position = server_state["positions"]["disco"]
            assert 2 <= start_position <= 540
            local_val = OUT / "retained/val/disco"
            subprocess.run(["rsync", "-aO", "--no-owner", "--no-group", "--no-perms",
                "--include=*/", "--include=*.json", "--include=*.npz", "--exclude=*",
                "joms-gpu:" + REMOTE + "/" + str(REL / "retained/val/disco") + "/",
                str(local_val) + "/"], check=True)
            completed_records = list(local_val.glob("*/*/*.json"))
            ledger_records = [read(path) for path in completed_records if "ledger_sha256" in read(path)]
            for record in ledger_records:
                assert sha(ROOT / record["ledger"]) == record["ledger_sha256"]
            assert len(ledger_records) >= start_position
            policy = OUT / "same_runtime_original_policy_disco.json"
            assert not policy.exists()
            write(policy, dict(created_unix=time.time(), method="disco", test_used_for_policy=False,
                rule="선택이 바뀐 시드는 AP/R1/원래모델을 같은 로컬 GPU·보존 polar·원래 FP32와 batch로 계산한다.",
                historical_original="원래 원장은 보존하고 현재 같은 원래모델과의 차이를 별도 열로 분리한다.",
                candidate_selection="동결한 검증60후보·9시퀀스·원래 L2와 집계 유지; 시험값으로 고르지 않는다."))
            policy_created = True
            plan = read(OUT / "local_gpu_plan.json")
            write(OUT / "local_gpu_plan_before_disco.json", plan)
            plan["methods"].append("disco")
            plan["start_positions"]["disco"] = start_position
            plan["disco_handoff_nonce"] = nonce
            write(OUT / "local_gpu_plan.json", plan)
            state = read(OUT / "local_gpu_state.json")
            state["positions"]["disco"] = start_position
            state.update(status="adding_validated_low_priority_DiSCO", active=None, updated_unix=time.time())
            write(OUT / "local_gpu_state.json", state)
            hosts = read(OUT / "execution_hosts.json")
            hosts["disco"] = "local"
            write(OUT / "execution_hosts.json", hosts)
            proof = dict(method="disco", all_arrays_exact=True, accepted_for_execution=True,
                         reference_completed_server_ledgers=2, cases=proofs,
                         scope="같은 원래 모델·batch·점수의 전체10510질의 원장 동치", completed_unix=time.time())
            write(OUT / "local_gpu_smoke/disco.json", proof)
            write(handoff, dict(created_unix=time.time(), nonce=nonce, server_start_position=start_position,
                local_start_position=start_position, restored_ledger_count=len(ledger_records), validation_jobs=540, local_priority=["seqot", "imlpr", "disco"],
                smoke_sha256=sha(OUT / "local_gpu_smoke/disco.json"), candidate_set_unchanged=True,
                server_control_before=remote_control, old_local_queue_pid=old_pid))
            local_control["synchronous_methods"] = list(dict.fromkeys([*local_control.get("synchronous_methods", []), "disco"]))
            local_control["method_priority"] = ["seqot", "imlpr", "disco"]
            adopted = True
        except Exception as exc:
            if read(OUT / "local_gpu_control.json") == held:
                for name, value in before.items():
                    write(OUT / name, value)
                if policy_created:
                    policy.rename(policy.with_name(policy.stem + ".not_adopted_" + nonce + ".json"))
            if remote_held is not None and remote_json("queue_control.json") == remote_held:
                put_remote("queue_control.json", remote_control)
            write(status_path, dict(status="not_adopted", error=str(exc), updated_unix=time.time()))
        finally:
            current = read(OUT / "local_gpu_control.json")
            if current == held:
                write(OUT / "local_gpu_control.json", local_control)
                restarted = background("_review_selection_full_local_queue.py", [], "local_queue_after_disco_smoke.log")
                write(OUT / "disco_smoke_queue_restart.json", dict(pid=restarted, adopted=adopted, updated_unix=time.time()))
        if not adopted:
            return
        restore_pid = background("_review_selection_restore_disco_test.py", ["restore"], "disco_test_input_restore.log")
        write(status_path, dict(status="adopted_running", pid=os.getpid(), test_restore_pid=restore_pid, updated_unix=time.time()))
    report = OUT / "retained/report_disco.json"
    while not report.exists():
        time.sleep(20)
    with (OUT / "logs/independent_verification_disco.log").open("a") as log:
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts/_review_selection_full_verify.py"),
                        "--method", "disco"], stdout=log, stderr=subprocess.STDOUT, check=True)
    from _review_selection_full_delivery import summarize
    from _review_selection_full_local_queue import sync_outputs
    summarize("disco")
    sync_outputs("disco", dict(kind="select_and_test"))
    subprocess.run(["rsync", "-aO", "--no-owner", "--no-group", "--no-perms",
        str(OUT / "independent_verification_disco.json"), str(OUT / "final_delivery_disco.json"),
        "joms-gpu:" + REMOTE + "/" + str(REL) + "/"], check=True)
    control = remote_json("queue_control.json")
    if control.get("local_disco_owner_nonce") == read(handoff)["nonce"]:
        control["held_streams"] = [s for s in control["held_streams"] if s != "disco"]
        control["local_disco_complete"] = time.time()
        put_remote("queue_control.json", control)
    write(status_path, dict(status="complete", completed_unix=time.time(), main_table_changed=False))


if __name__ == "__main__":
    main()
