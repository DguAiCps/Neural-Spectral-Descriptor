#!/usr/bin/env python3
"""ImLPR 유휴 경계에서 HeLiOS 전체 검증 시퀀스의 실행 동치를 검사한다."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "src")]
from _review_selection_bev import read, write, sha, metrics

OUT = ROOT / "results/pdv2_review_selection_full_20261105"
SMOKE_ROOT = OUT / "helios_local_execution_smoke"


def worker(sequence: str) -> None:
    """별도 폴더에서 256장씩 원래 FP32·batch1 특징을 저장한다."""
    from _review_selection_local_reservation import require_available
    require_available()
    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"
    import torch
    import _review_selection_full_eval as evaluator
    from _causal_bevplace_pr import ledger

    torch.set_num_threads(4)
    plan = read(SMOKE_ROOT / "plan.json")
    assert sequence in plan["sequences"]
    assert sha(OUT / "retained_protocol.json") == plan["protocol_sha256"]
    row = plan["candidate"]
    assert sha(ROOT / row["path"]) == row["sha256"]
    sensor, seq = sequence.split("/")
    key = f"{sensor}_{seq}"
    dest = SMOKE_ROOT / key
    dest.mkdir(exist_ok=True)
    if (dest / "comparison.json").exists():
        return
    if key == "kitti360_0004":
        original_configure = evaluator.configure

        def configure(method: str):
            module = original_configure(method)
            module.DATA = Path(plan["extra_input_container_root"])
            return module

        evaluator.configure = configure
    module = evaluator.configure("helios")
    input_dir = module.seq_dir("test_odom", sensor, seq)
    ids, positions = evaluator.positions(sensor, seq, "val")
    verified = dest / "input_verified.json"
    if not verified.exists():
        expected = read(SMOKE_ROOT / "server_input_hashes.json")["sequences"][key]
        hashes = {}
        for sid in ids:
            filename = f"{int(sid):06d}.bin"
            relative = str((OUT / "inputs/helios/data/test_odom" / key / "LiDAR" / filename).relative_to(ROOT))
            digest = sha(input_dir / "LiDAR" / filename)
            assert digest == expected[relative], filename
            hashes[filename] = digest
        write(verified, dict(frames=len(ids), all_input_files_SHA_exact=True,
                             source_inventory_sha256=sha(SMOKE_ROOT / "server_input_hashes.json"),
                             cache_sha256=sha(evaluator.cache(sensor, seq)), files=hashes,
                             checked_unix=time.time()))
    args = argparse.Namespace(method="helios", role="val", scope="retained", seed=1,
                              candidate=row["id"], sequence=sequence, forward_budget=256)
    desc, audit = evaluator.descriptors(args, row, ids, dest)
    assert desc.dtype == np.float32 and np.isfinite(desc).all()
    tick = time.time()
    actual = ledger(desc, positions)
    ix = np.searchsorted(actual["query_idx"], evaluator.query_base(sensor, seq, "val")["query_idx"])
    actual = {field: values[ix] for field, values in actual.items()}
    ref_record = read(SMOKE_ROOT / "reference" / f"{key}.json")
    ref_path = SMOKE_ROOT / "reference" / f"dump_{key}.npz"
    assert sha(ref_path) == ref_record["ledger_sha256"]
    with np.load(ref_path) as source:
        reference = {field: source[field] for field in source.files if field != "meta"}
    differences = {field: dict(n_mismatch=int(np.count_nonzero(values != reference[field])),
                               max_abs=float(np.max(np.abs(values - reference[field]))))
                   for field, values in actual.items() if not np.array_equal(values, reference[field])}
    current_metrics = metrics(dict(actual, score_negl2=actual["score_cos"]))
    reference_metrics = metrics(dict(reference, score_negl2=reference["score_cos"]))
    metric_deltas = {field: (None if value is None else value - reference_metrics[field])
                     for field, value in current_metrics.items()}
    np.savez_compressed(dest / "local_ledger.npz", **actual)
    write(dest / "comparison.json", dict(completed_unix=time.time(), sequence=sequence,
        candidate=row, gpu=torch.cuda.get_device_name(0), frames=len(ids), queries=len(ix),
        all_arrays_exact=not differences, differences=differences,
        all_winners_exact=np.array_equal(actual["j_star"], reference["j_star"]),
        current_metrics=current_metrics, reference_metrics=reference_metrics,
        metric_deltas=metric_deltas, audit=audit, search_s=time.time() - tick,
        descriptor_bytes_sha256=hashlib.sha256(np.ascontiguousarray(desc).tobytes()).hexdigest(),
        reference_ledger_sha256=sha(ref_path), local_ledger_sha256=sha(dest / "local_ledger.npz"),
        plan_sha256=sha(SMOKE_ROOT / "plan.json"), precision="기존 공식 FP32·batch1, TF32 override 0",
        accepted_for_execution=False, note="전체 대조 뒤 별도 판정 전에는 후보 분담을 바꾸지 않는다"))


def imlpr_ready() -> bool:
    """완성 입력이 있지만 아직 소비하지 않은 ImLPR 작업을 우선한다."""
    for request in (OUT / "test_input_requests").glob("imlpr_*.json"):
        info = read(request)
        if info.get("status") == "consumed":
            continue
        sensor, seq = info["sequence"].split("/")
        if (OUT / f"inputs/test_imlpr/data/test_odom/{sensor}_{seq}/transfer_complete.json").exists():
            return True
    return False


def prepare_inputs() -> None:
    """서버의 동일 검증 입력 1.2 GB를 여유 있는 D:에만 복사·검산한다."""
    plan = read(SMOKE_ROOT / "plan.json")
    target = Path(plan["extra_input_host_root"]) / "test_odom/kitti360_0004"
    target.mkdir(parents=True, exist_ok=True)
    remote = ("joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/" +
              str(OUT.relative_to(ROOT)) + "/inputs/helios/data/test_odom/kitti360_0004/")
    subprocess.run(["rsync", "-aO", "--no-owner", "--no-group", "--no-perms",
                    "--partial", remote, str(target) + "/"], check=True)
    expected = read(SMOKE_ROOT / "server_input_hashes.json")["sequences"]["kitti360_0004"]
    for source, digest in expected.items():
        assert sha(target / "LiDAR" / Path(source).name) == digest
    write(SMOKE_ROOT / "extra_input_transfer_verified.json", dict(frames=len(expected),
        all_input_files_SHA_exact=True, source_inventory_sha256=sha(SMOKE_ROOT / "server_input_hashes.json"),
        target=str(target), completed_unix=time.time()))


def run() -> None:
    """GPU 작업을 중단하지 않고 256장 경계마다 우선 작업에 반환한다."""
    lock = (SMOKE_ROOT / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = SMOKE_ROOT / "state.json"
    control_path = OUT / "local_gpu_control.json"
    plan = read(SMOKE_ROOT / "plan.json")
    state = read(state_path) if state_path.exists() else dict(chunks=0)
    stale = read(control_path)
    held_path = SMOKE_ROOT / "held_control.json"
    if held_path.exists():
        previous = read(held_path)
        old_proc = Path(f"/proc/{previous['pid']}/cmdline")
        old_alive = old_proc.exists() and b"_review_selection_helios_local_smoke.py" in old_proc.read_bytes()
        if not old_alive and stale == previous["held"]:
            write(control_path, previous["before"])
    state.update(pid=os.getpid(), started_unix=time.time(), status="waiting_idle")
    for sequence in plan["sequences"]:
        key = sequence.replace("/", "_")
        while not (SMOKE_ROOT / key / "comparison.json").exists():
            before = read(control_path)
            queue = read(OUT / "local_gpu_state.json")
            input_ready = key == "kitti_09" or (SMOKE_ROOT / "extra_input_transfer_verified.json").exists()
            if queue.get("active") or before.get("held_methods") or imlpr_ready() or not input_ready:
                state.update(status="waiting_priority_or_input", sequence=sequence, updated_unix=time.time())
                write(state_path, state)
                time.sleep(5)
                continue
            nonce = f"helios_smoke_{time.time_ns()}"
            held = dict(before, held_methods=read(OUT / "local_gpu_plan.json")["methods"],
                        helios_smoke_nonce=nonce, updated_unix=time.time())
            write(SMOKE_ROOT / "held_control.json", dict(before=before, held=held, pid=os.getpid()))
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
                    raise RuntimeError("GPU 유휴 확인 실패")
                if imlpr_ready():
                    continue
                command = ["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=0",
                    "-e", "NVIDIA_TF32_OVERRIDE=0", "-e", "OMP_NUM_THREADS=4",
                    "-e", "MKL_NUM_THREADS=4", "-e", "OPENBLAS_NUM_THREADS=1",
                    "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "python3", "-u",
                    "scripts/_review_selection_helios_local_smoke.py", "worker", "--sequence", sequence]
                with (SMOKE_ROOT / "worker.log").open("a") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    state.update(status="GPU_chunk", sequence=sequence, child_pid=child.pid,
                                 updated_unix=time.time())
                    write(state_path, state)
                    code = child.wait()
                assert code in (0, 75), code
                state.update(chunks=state["chunks"] + 1, last_returncode=code,
                             updated_unix=time.time())
                write(state_path, state)
            finally:
                if read(control_path) == held:
                    write(control_path, before)
                else:
                    raise RuntimeError("외부 제어 변경 발견; 원래 제어를 덮어쓰지 않음")
            time.sleep(2)
    state.update(status="all_comparisons_complete", candidate_assignment_changed=False,
                 completed_unix=time.time())
    write(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("worker", "run", "inputs"))
    parser.add_argument("--sequence")
    arguments = parser.parse_args()
    if arguments.action == "inputs":
        prepare_inputs()
    elif arguments.action == "run":
        run()
    else:
        worker(arguments.sequence)
