#!/usr/bin/env python3
"""배정 뒤에만 보조 GPU에서 미시작 후보를 중복 없이 검증한다."""
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
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "src")]
from _review_selection_bev import read, write, sha
from _review_selection_full_eval import OUT, candidate, descriptors, positions, query_base, SCORES
from _review_selection_full_inputs import cache
from _review_selection_full_queue import command


def smoke(seed: int, name: str, key: str) -> None:
    """같은 공식 HeLiOS 입력 전체의 서술자·승자·최종 점수를 대조한다."""
    import torch
    from _causal_bevplace_pr import ledger
    torch.set_num_threads(4)
    sensor, seq = key.split("/")
    row = candidate("helios", seed, name, "retained")
    assert sha(ROOT / row["path"]) == row["sha256"]
    reference = OUT / f"retained/val/helios/s{seed}/{name}/{sensor}_{seq}.json"
    record = read(reference)
    assert sha(ROOT / record["ledger"]) == record["ledger_sha256"]
    dest = OUT / "aux_gpu_smoke" / f"s{seed}_{name}_{sensor}_{seq}"
    dest.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(method="helios", role="val", sequence=key, seed=seed,
                           candidate=name, scope="retained", forward_budget=2048)
    ids, pos = positions(sensor, seq, "val")
    tick = time.time()
    desc, audit = descriptors(args, row, ids, dest)
    digest = hashlib.sha256()
    digest.update(json.dumps([list(desc.shape), str(desc.dtype), sha(cache(sensor, seq)),
                             sha(ROOT / "scripts/_causal_bevplace_pr.py")]).encode())
    digest.update(np.ascontiguousarray(desc).tobytes())
    same_descriptors = digest.hexdigest() == record["audit"]["descriptor_sha256"]
    actual = ledger(desc, pos)
    base = query_base(sensor, seq, "val")
    ix = np.searchsorted(actual["query_idx"], base["query_idx"])
    differences = {}
    actual = {field: values[ix] for field, values in actual.items()}
    with np.load(ROOT / record["ledger"]) as old:
        fields = [k for k in old.files if k != "meta"]
        for field in fields:
            differences[field] = dict(n_mismatch=int(np.count_nonzero(actual[field] != old[field])),
                max_abs=float(np.max(np.abs(actual[field] - old[field]))))
        from _review_selection_bev import metrics
        original_metrics = metrics(dict({k: old[k] for k in fields}, score_negl2=old["score_cos"]))
    new_metrics = metrics(dict(actual, score_negl2=actual["score_cos"]))
    np.savez_compressed(dest / "actual_ledger.npz", **actual)
    exact = all(value["n_mismatch"] == 0 for value in differences.values())
    proof = dict(all_arrays_exact=exact, same_descriptor_bytes=same_descriptors,
                 gpu=torch.cuda.get_device_name(0), visible_device=os.environ.get("CUDA_VISIBLE_DEVICES"),
                 device_uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
                 checkpoint_sha256=row["sha256"], reference_json=str(reference.relative_to(ROOT)),
                 reference_ledger_sha256=record["ledger_sha256"], fields=fields,
                 sequence=key, frames=len(ids), queries=len(ix), seconds=time.time() - tick,
                 cumulative_forward_s=audit["descriptor_inference_cumulative_s"],
                 forward_budget=2048, differences=differences, actual_metrics=new_metrics,
                 reference_metrics=original_metrics, descriptor_sha256=digest.hexdigest(),
                 reference_descriptor_sha256=record["audit"]["descriptor_sha256"],
                 accepted=exact and same_descriptors,
                 peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                 completed_unix=time.time(), scorer_sha256=sha(ROOT / "scripts/_causal_bevplace_pr.py"))
    write(dest / "comparison.json", proof)
    if not proof["accepted"]:
        write(OUT / "aux_gpu_smoke/rejected.json", proof)
        print(json.dumps(proof), flush=True)
        raise SystemExit(3)
    write(dest / "verified.json", proof)
    write(OUT / "aux_gpu_smoke/verified.json", proof)
    print(json.dumps(proof), flush=True)


def smoke_loop(seed: int, name: str, key: str) -> None:
    """주 GPU와 같은2048프레임 재시작을 유지하고 완료·기각을 기록한다."""
    home = OUT / "aux_gpu_smoke"
    home.mkdir(exist_ok=True)
    lock = (home / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    proposal = read(OUT / "aux_gpu_assignment_proposal.json")
    assert proposal["CUDA_VISIBLE_DEVICES"] == os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert sha(ROOT / proposal["smoke"]["reference"]) == proposal["smoke"]["reference_sha256"]
    state = dict(pid=os.getpid(), started_unix=time.time(), chunks=0,
        forward_budget=2048, acceptance="original proposal: descriptor bytes and all9098query arrays exact")
    while True:
        command_line = [sys.executable, "-u", str(Path(__file__)), "smoke", "--seed", str(seed),
            "--candidate", name, "--sequence", key]
        with (home / "worker.log").open("a") as log:
            child = subprocess.Popen(command_line, stdout=log, stderr=subprocess.STDOUT)
            state.update(status="running", child_pid=child.pid, updated_unix=time.time())
            write(home / "state.json", state)
            code = child.wait()
        state.update(chunks=state["chunks"]+1, returncode=code, updated_unix=time.time())
        if code != 75:
            state["status"] = "verified" if code == 0 else "rejected" if code == 3 else "failed"
            write(home / "state.json", state)
            if code not in (0, 3):
                raise RuntimeError(f"aux GPU smoke failed: {code}")
            return
        write(home / "state.json", state)


def run() -> None:
    """고정 분담의 작업만 실행하며 전체 선택·시험은 주 작업열에 맡긴다."""
    plan_path = OUT / "aux_gpu_plan.json"
    plan = read(plan_path)
    assert plan["authorization"] and plan["gpu"] == os.environ["CUDA_VISIBLE_DEVICES"]
    assert sha(OUT / "retained_protocol.json") == plan["protocol_sha256"]
    proof_path = OUT / "aux_gpu_smoke/verified.json"
    proof = read(proof_path)
    assert proof["all_arrays_exact"] and proof["same_descriptor_bytes"]
    assert proof["visible_device"] == plan["gpu"]
    assert sha(proof_path) == plan["smoke_sha256"]
    lock = (OUT / "aux_gpu_queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = OUT / "aux_gpu_state.json"
    state = read(state_path) if state_path.exists() else dict(position=0, failures={})
    state.update(pid=os.getpid(), started_unix=time.time(), plan_sha256=sha(plan_path))
    while state["position"] < len(plan["jobs"]):
        control_path = OUT / "aux_gpu_control.json"
        control = read(control_path) if control_path.exists() else {}
        if control.get("hold", False):
            state.update(status="held", active=None, updated_unix=time.time())
            write(state_path, state)
            time.sleep(10)
            continue
        assert sha(plan_path) == state["plan_sha256"]
        job = plan["jobs"][state["position"]]
        log = OUT / "logs" / f"aux_gpu_{state['position']:05d}.log"
        env = dict(os.environ, REVIEW_SELECTION_EXECUTOR=plan["executor"], SEED=str(job["seed"]),
                   OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4")
        tick = time.time()
        with log.open("a") as handle:
            child = subprocess.Popen(command(job), env=env, stdout=handle, stderr=subprocess.STDOUT)
            state.update(status="running", active=dict(job, child_pid=child.pid), updated_unix=time.time())
            write(state_path, state)
            code = child.wait()
        if code == 0:
            state["position"] += 1
        elif code not in (75, 76):
            key = str(state["position"])
            state["failures"][key] = state["failures"].get(key, 0) + 1
            if state["failures"][key] >= 3:
                state.update(status="worker_failed", updated_unix=time.time())
                write(state_path, state)
                raise RuntimeError(f"auxiliary worker failed: {job}")
        state.update(last_result=dict(returncode=code, elapsed_s=time.time() - tick), updated_unix=time.time())
        write(state_path, state)
        if code == 76:
            time.sleep(30)
    state.update(status="complete", active=None, updated_unix=time.time())
    write(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "smoke_loop", "run"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--candidate")
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.action == "run":
        run()
    elif args.action == "smoke_loop":
        smoke_loop(args.seed, args.candidate, args.sequence)
    else:
        smoke(args.seed, args.candidate, args.sequence)
