#!/usr/bin/env python3
"""GPU간 수치 차이와 가까운 ImLPR 선택을 시험 전에 서버에서 재검산한다."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"src")]
from _review_selection_bev import read,write,sha,VAL
OUT=ROOT/"results/pdv2_review_selection_full_20261105"
REMOTE_ROOT="/home/qkrrlans00/Neural-Spectral-Codec"
REL=OUT.relative_to(ROOT)


def worker(seed: int, name: str, sequence: str) -> None:
    """같은 평가기를 별도 검산 출력 폴더에 실행한다."""
    os.environ["SEED"]=str(seed)
    import _review_selection_full_eval as evaluate
    evaluate.folder=lambda method,seed,name,scope,role: OUT/"numerical_margin"/role/method/f"s{seed}"/name
    evaluate.main(argparse.Namespace(method="imlpr",seed=seed,candidate=name,sequence=sequence,scope="retained",
                                    role="val",query_budget=128,query_offset=0,forward_budget=2048))


def remote_json(name: str) -> dict:
    import json
    return json.loads(subprocess.check_output(["ssh","joms-gpu","cat",REMOTE_ROOT+"/"+str(REL/name)]))


def put_json(name: str, value: dict) -> None:
    path=OUT/("margin_transfer_"+name)
    write(path,value)
    subprocess.run(["rsync","-aO","--no-owner","--no-group","--no-perms",str(path),
                    "joms-gpu:"+REMOTE_ROOT+"/"+str(REL/name)],check=True)


def run() -> None:
    """요청이 생기면 서버의 저장 경계에서 잠시 인계받고 기존 제어를 복원한다."""
    lock=(OUT/"margin_audit.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=OUT/"margin_audit_status.json"
    request=OUT/"imlpr_numerical_margin_request.json"
    while not request.exists():
        write(status,dict(status="waiting_validation_margin",pid=os.getpid(),updated_unix=time.time()));time.sleep(30)
    target=OUT/"imlpr_numerical_override.json"
    if target.exists():return
    spec=read(OUT/"retained_protocol.json")
    seeds=sorted(set(r["seed"] for r in read(request)["close"]))
    plan=dict(created_unix=time.time(),seeds={s:spec["methods"]["imlpr"][s]["candidates"] for s in seeds},
              request_sha256=sha(request),protocol_sha256=sha(OUT/"retained_protocol.json"),validation_sequences=VAL,test_not_read=True)
    write(OUT/"imlpr_numerical_plan.json",plan)
    # 명세와 검산에 필요한 원천을 먼저 동기화한다.
    subprocess.run(["rsync","-a",str(Path(__file__)),str(ROOT/"scripts/_review_selection_full_eval.py"),
                    "joms-gpu:"+REMOTE_ROOT+"/scripts/"],check=True)
    put_json("restart_nonheli_inputs.json",read(OUT/"restart_nonheli_inputs.json"))
    for _,seq in [row for row in VAL if row[0]=="helipr"]:
        name="delivered_riv_verified/helipr_"+seq+".json"
        proof=read(OUT/name)
        assert proof["all_files_exact"]
        subprocess.run(["ssh","joms-gpu","mkdir","-p",REMOTE_ROOT+"/"+str(REL/"delivered_riv_verified")],check=True)
        put_json(name,proof)
    put_json("imlpr_numerical_plan.json",plan)
    control=remote_json("queue_control.json")
    nonce="imlpr_margin_"+str(int(time.time()))
    held=dict(control,held_streams=control["stream_order"],numerical_hold_nonce=nonce,
              reason="검증 선택 격차가교차GPU확인범위에가까워같은서버에서선택전재검산",updated_unix=time.time())
    put_json("queue_control.json",held)
    try:
        write(status,dict(status="waiting_server_boundary",pid=os.getpid(),seeds=seeds,updated_unix=time.time()))
        while remote_json("queue_state.json")["status"]!="held_pending_work":time.sleep(10)
        records={}
        for seed,candidates in plan["seeds"].items():
            records[seed]={}
            for row in candidates:
                records[seed][row["id"]]={}
                for sensor,seq in VAL:
                    write(status,dict(status="server_validation",pid=os.getpid(),seed=seed,candidate=row["id"],sequence=sensor+"/"+seq,updated_unix=time.time()))
                    with (OUT/"logs"/f"margin_s{seed}_{row['id']}_{sensor}_{seq}.log").open("a") as log:
                        subprocess.run(["ssh","joms-gpu","docker","exec","-e","CUDA_VISIBLE_DEVICES=0","-e",f"SEED={seed}",
                            "-e","OMP_NUM_THREADS=4","-e","OPENBLAS_NUM_THREADS=1","-e","MKL_NUM_THREADS=4","-w","/workspace/Neural-Spectral-Codec",
                            "ring_baseline","python3","-u","scripts/_review_selection_full_margin_audit.py","worker","--seed",seed,
                            "--candidate",row["id"],"--sequence",sensor+"/"+seq],stdout=log,stderr=subprocess.STDOUT,check=True)
                    home=OUT/"numerical_margin/val/imlpr"/f"s{seed}"/row["id"];home.mkdir(parents=True,exist_ok=True)
                    subprocess.run(["rsync","-aO","--no-owner","--no-group","--no-perms",
                                    "joms-gpu:"+REMOTE_ROOT+"/"+str(home.relative_to(ROOT))+"/",str(home)+"/"],check=True)
                    path=home/f"{sensor}_{seq}.json";record=read(path)
                    assert sha(ROOT/record["ledger"])==record["ledger_sha256"]
                    records[seed][row["id"]][sensor+"/"+seq]=str(path.relative_to(ROOT))
        write(target,dict(completed_unix=time.time(),records=records,plan_sha256=sha(OUT/"imlpr_numerical_plan.json"),test_not_read=True,
                          reason="가까운 선택의 모든보존후보를같은서버GPU/원래점수로검증; 후보집합·시험값불변"))
        write(status,dict(status="complete",pid=os.getpid(),updated_unix=time.time()))
    finally:
        current=remote_json("queue_control.json")
        if current.get("numerical_hold_nonce")==nonce:put_json("queue_control.json",control)


if __name__=="__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=("run","worker"))
    parser.add_argument("--seed",type=int);parser.add_argument("--candidate");parser.add_argument("--sequence")
    args=parser.parse_args()
    if args.action=="run":run()
    else:worker(args.seed,args.candidate,args.sequence)
