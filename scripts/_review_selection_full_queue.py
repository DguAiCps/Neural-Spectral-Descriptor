#!/usr/bin/env python3
"""전체 보존후보 검증·BEV120에포크·고정 후 시험을 영속 작업열로 실행한다."""
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
from _review_selection_bev import read, write, sha, VAL

OUT = ROOT / "results/pdv2_review_selection_full_20261105"
METHODS = ("disco", "seqot", "helios", "imlpr", "deepring", "ringsharp", "bevplace")


def prepare() -> None:
    """관측 결과와 무관한 모든 후보·작업 순서를 디스크에 고정한다."""
    target = OUT / "full_queue.json"
    if target.exists():
        return
    protocol = read(OUT / "retained_protocol.json")
    streams = {m: [] for m in METHODS}
    for method in METHODS:
        seeds = protocol["methods"][method]
        candidates = [(seed,row) for seed, entry in seeds.items() for row in entry["candidates"]]
        candidates.sort(key=lambda r: (r[1]["epoch"], int(r[0]), r[1]["id"]))
        for seed, candidate in candidates:
            for sensor, seq in VAL:
                streams[method].append(dict(kind="retained", method=method, seed=int(seed),
                    candidate=candidate["id"], sequence=sensor+"/"+seq))
        streams[method].append(dict(kind="select_and_test", method=method, scope="retained"))
    streams["bev_rerun"] = []
    for epoch in range(40):
        for seed in (1,2,3):
            streams["bev_rerun"].append(dict(kind="train_bev", seed=seed, epoch=epoch))
            for sensor, seq in VAL:
                streams["bev_rerun"].append(dict(kind="rerun_val", method="bevplace", seed=seed,
                    epoch=epoch, candidate=f"epoch_{epoch:03d}", sequence=sensor+"/"+seq))
    streams["bev_rerun"].append(dict(kind="select_and_test", method="bevplace", scope="rerun"))
    write(target,dict(created_unix=time.time(), source_protocol_sha256=sha(OUT/"retained_protocol.json"),
        scheduling="매 stream에서 한 작업씩 round-robin; 무거운 원장 작업은 저장된 질의 경계에서75로 반환해 다음순환에 재개",
        external_initial_training=dict(host_pid=1520447, seed=1, epoch=0),
        partial_return_code=75, streams=streams,
        retained_validation_jobs=sum(len(streams[m])-1 for m in METHODS),
        bev_training_epochs=120, bev_validation_jobs=1080,
        finalization_jobs=8, test_policy="방법별 전체 후보 검증이 끝난 뒤만 선택을 동결하고 선택된 고유 모델의46시험을 진행"))
    write(OUT/"queue_state.json",dict(created_unix=time.time(), positions={k:0 for k in streams}, completed=0,
        next_stream=0, status="full_scope_running", attempts={}, active=None))


def command(job: dict) -> list:
    """작업 하나를 고유 실행기로 연결한다."""
    if job["kind"] == "train_bev":
        return [sys.executable,"-u",str(ROOT/"scripts/_review_selection_full_bev_train.py"),"epoch",
                "--seed",str(job["seed"]),"--epoch",str(job["epoch"])]
    if job["kind"] == "select_and_test":
        return [sys.executable,"-u",str(ROOT/"scripts/_review_selection_full_finish.py"),
                "--method",job["method"],"--scope",job["scope"]]
    controls=read(OUT/"queue_control.json") if (OUT/"queue_control.json").exists() else {}
    budget=controls.get("query_budget",{}).get(job["method"],128)
    cmd=[sys.executable,"-u",str(ROOT/"scripts/_review_selection_full_eval.py"),"--method",job["method"],
         "--seed",str(job["seed"]),"--candidate",job["candidate"],"--sequence",job["sequence"],
         "--role","val","--query-budget",str(budget)]
    if job["kind"] == "rerun_val":
        cmd += ["--scope","rerun"]
    return cmd


def run() -> None:
    """파일 잠금·상태·로그를 보존하며 작업열을 끝까지 실행한다."""
    prepare()
    lock=(OUT/"queue.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(OUT/"full_queue.json")
    assert sha(OUT/"retained_protocol.json")==plan["source_protocol_sha256"]
    state=read(OUT/"queue_state.json")
    state["runner_pid"]=os.getpid()
    state["runner_started_unix"]=time.time()
    write(OUT/"queue_state.json",state)
    # 이미 시작한 최초 에포크와 중복 실행하지 않는다. 이후에는 이 runner만 GPU0을 쓴다.
    manifest=OUT/"bev_rerun/s1/epochs/epoch_000.pth.json"
    initial_status=OUT/"bev_rerun/s1/status.json"
    while not manifest.exists() and initial_status.exists():
        pid=read(initial_status).get("pid")
        if not pid or not Path(f"/proc/{pid}").exists():
            break
        state.update(status="waiting_initial_epoch",active={"external_container_pid":pid},updated_unix=time.time())
        write(OUT/"queue_state.json",state)
        time.sleep(10)
    names=["bev_rerun",*METHODS]
    while True:
        control_path = OUT / "queue_control.json"
        control = read(control_path) if control_path.exists() else {}
        names = control.get("stream_order", names)
        held = set(control.get("held_streams", []))
        assert set(names) == set(plan["streams"])
        waiting = state.setdefault("input_wait_until", {})
        eligible = {s for s in names if s not in held and
                    state["positions"][s] < len(plan["streams"][s]) and waiting.get(s, 0) <= time.time()}
        for tier in control.get("priority_tiers", []):
            available = eligible.intersection(tier)
            if available:
                eligible = available
                break
        progressed=False
        for offset in range(len(names)):
            slot=(state["next_stream"]+offset)%len(names)
            stream=names[slot]
            if stream not in eligible:
                continue
            pos=state["positions"][stream]
            if pos>=len(plan["streams"][stream]):
                continue
            job=plan["streams"][stream][pos]
            cmd=command(job)
            if not Path(cmd[2]).exists():
                state.update(status="waiting_worker_implementation",active=job,updated_unix=time.time())
                write(OUT/"queue_state.json",state)
                continue
            jid=f"{stream}_{pos:05d}"
            attempt=state["attempts"].get(jid,0)+1
            state["attempts"][jid]=attempt
            state.update(status="running",active=dict(job,job_id=jid,attempt=attempt),updated_unix=time.time())
            write(OUT/"queue_state.json",state)
            log=OUT/"logs"/f"queue_{jid}.log"
            log.parent.mkdir(exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",SEED=str(job.get("seed",1)),
                     OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="4")
            tick=time.time()
            with log.open("a") as handle:
                child=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,env=env)
                state["active"]["child_pid"]=child.pid
                write(OUT/"queue_state.json",state)
                code=child.wait()
            elapsed=time.time()-tick
            state["last_result"]=dict(job_id=jid,returncode=code,elapsed_s=elapsed,finished_unix=time.time())
            if code==0:
                state["positions"][stream]=pos+1
                state["completed"]+=1
                progressed=True
            elif code==76:
                waiting[stream] = time.time() + 60
            elif code!=75:
                failures=state.setdefault("failure_counts",{})
                failures[jid]=failures.get(jid,0)+1
                if failures[jid]>=3:
                    state.update(status="worker_failed",active=job,updated_unix=time.time())
                    write(OUT/"queue_state.json",state)
                    raise RuntimeError(f"{jid} failed 3 times; resume after fixing cause; see {log}")
                waiting[stream]=time.time()+10
            state["next_stream"]=(slot+1)%len(names)
            state["updated_unix"]=time.time()
            write(OUT/"queue_state.json",state)
            print("QUEUE",jid,"rc",code,"seconds",round(elapsed,1),"completed",state["completed"],flush=True)
            break
        else:
            if all(state["positions"][s]>=len(plan["streams"][s]) for s in names):
                state.update(status="complete",active=None,updated_unix=time.time())
                write(OUT/"queue_state.json",state)
                return
            if all(s in held or state["positions"][s]>=len(plan["streams"][s]) for s in names):
                state.update(status="held_pending_work",active=None,held_streams=sorted(held),updated_unix=time.time())
                write(OUT/"queue_state.json",state)
            time.sleep(10)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","run"))
    args=parser.parse_args()
    prepare() if args.action=="prepare" else run()
