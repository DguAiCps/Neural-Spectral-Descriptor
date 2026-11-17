#!/usr/bin/env python3
"""로컬 GPU에 배정한 SeqOT·ImLPR 작업을 중복 없이 이어 실행한다."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from _review_selection_bev import read,write,sha
from _review_selection_full_queue import command

OUT=ROOT/"results/pdv2_review_selection_full_20261105"


def available_imlpr(job: dict) -> bool:
    """해시 검증된 입력의 작업만 순서를 앞당긴다. 후보 목록은 유지한다."""
    if job["kind"]!="retained": return False
    sensor,seq=job["sequence"].split("/")
    marker=(OUT/"delivered_riv_verified"/f"{sensor}_{seq}.json" if sensor=="helipr"
            else OUT/"restart_nonheli_inputs.json")
    if not marker.exists():return False
    info=read(marker)
    return info.get("all_files_exact",False) if sensor=="helipr" else info.get("status")=="complete"


def sync_outputs(method: str, job: dict) -> None:
    """새로 완성한 원장과 선택 기록만 서버로 복사한다."""
    paths=[]
    if job["kind"]=="retained":
        sensor,seq=job["sequence"].split("/")
        home=OUT/f"retained/val/{method}/s{job['seed']}/{job['candidate']}"
        paths=[home/f"{sensor}_{seq}.json",home/f"dump_{sensor}_{seq}.npz"]
    else:
        for name in ("selection","validation","report"):
            path=OUT/"retained"/f"{name}_{method}.json"
            if path.exists():paths.append(path)
        home=OUT/"retained/test"/method
        if home.exists():paths.extend(p for p in home.rglob("*") if p.is_file() and p.suffix in (".json",".npz"))
    paths=[p for p in paths if p.exists()]
    if not paths:return
    listing=OUT/"local_upload_files.txt"
    listing.write_text("\n".join(str(p.relative_to(ROOT)) for p in paths)+"\n")
    subprocess.run(["rsync","-aO","--no-owner","--no-group","--no-perms","--files-from="+str(listing),str(ROOT)+"/",
        "joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/"],check=True)


def main() -> None:
    """서버 경계의 위치부터 고정 작업열을 이어 실행한다."""
    lock=(OUT/"local_gpu_queue.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(OUT/"local_gpu_plan.json")
    assert sha(OUT/"retained_protocol.json")==plan["retained_protocol_sha256"]
    assert sha(OUT/"full_queue.json")==plan["full_queue_sha256"]
    full=read(OUT/"full_queue.json")
    state_path=OUT/"local_gpu_state.json"
    state=read(state_path) if state_path.exists() else dict(positions=plan["start_positions"],completed=0,failures={})
    state.update(pid=os.getpid(),started_unix=time.time())
    waiting={}
    while any(state["positions"][m]<len(full["streams"][m]) for m in plan["methods"]):
        control_path=OUT/"local_gpu_control.json"
        control=read(control_path) if control_path.exists() else {}
        held=set(control.get("held_methods",[]))
        from _review_selection_local_reservation import held as user_gpu_reserved
        if user_gpu_reserved():
            held.update(plan["methods"])
        eligible=[]
        for name in plan["methods"]:
            if name in held: continue
            proof=OUT/"local_gpu_smoke"/f"{name}.json"
            if state["positions"][name]>=len(full["streams"][name]) or waiting.get(name,0)>time.time(): continue
            if not proof.exists(): continue
            if not read(proof).get("accepted_for_execution",read(proof).get("all_arrays_exact",False)): continue
            eligible.append(name)
        if not eligible:
            state.update(status="held" if held else "waiting_input_or_smoke",active=None,
                         held_methods=sorted(held),updated_unix=time.time());write(state_path,state)
            time.sleep(10);continue
        method=eligible[0];stream=full["streams"][method]
        position=state["positions"][method]
        if method=="imlpr" and control.get("validation_skip_ahead",False) and stream[position]["kind"]=="retained" and not available_imlpr(stream[position]):
            ahead=[]
            for index in range(position+1,len(stream)):
                proposed=stream[index]
                if not available_imlpr(proposed):continue
                sensor,seq=proposed["sequence"].split("/")
                record=OUT/f"retained/val/imlpr/s{proposed['seed']}/{proposed['candidate']}/{sensor}_{seq}.json"
                if not record.exists():ahead.append(index)
            if ahead:position=ahead[0]
        job=stream[position];jid=f"{method}_{position:05d}"
        inner=command(job);script=str(Path(inner[2]).relative_to(ROOT))
        args=["docker","exec","-e","CUDA_VISIBLE_DEVICES=0","-e",f"SEED={job.get('seed',1)}",
            "-e","OMP_NUM_THREADS=4","-e","OPENBLAS_NUM_THREADS=1","-e","MKL_NUM_THREADS=4",
            "-w","/workspace/Neural-Spectral-Codec","ring_baseline","python3","-u",script,*inner[3:]]
        tick=time.time();log=OUT/"logs"/f"local_queue_{jid}.log"
        with log.open("a") as handle:
            child=subprocess.Popen(args,stdout=handle,stderr=subprocess.STDOUT)
            state.update(status="running",active=dict(job,job_id=jid,host_child_pid=child.pid),updated_unix=time.time())
            write(state_path,state);code=child.wait()
        asynchronous=control.get("async_upload",False) and method not in control.get("synchronous_methods",[])
        if code in (0,75) and not asynchronous:
            try: sync_outputs(method,job)
            except subprocess.CalledProcessError as exc:
                state.update(status="upload_retry",upload_error=str(exc),updated_unix=time.time())
                write(state_path,state);time.sleep(10);continue
        if code==0:
            if position==state["positions"][method]:
                state["positions"][method]+=1;state["completed"]+=1
                state.setdefault("precomputed_jobs",[])
                if jid in state["precomputed_jobs"]:state["precomputed_jobs"].remove(jid)
            elif jid not in state.setdefault("precomputed_jobs",[]):state["precomputed_jobs"].append(jid)
        elif code==76:
            waiting[method]=time.time()+60
        elif code not in (75,76):
            state["failures"][jid]=state["failures"].get(jid,0)+1
            if state["failures"][jid]>=3:
                state.update(status="worker_failed",updated_unix=time.time());write(state_path,state)
                raise RuntimeError(f"{jid} failed three times; see {log}")
        state.update(last_result=dict(job_id=jid,returncode=code,elapsed_s=time.time()-tick),updated_unix=time.time())
        write(state_path,state)
        import json
        with (OUT/"local_gpu_history.jsonl").open("a") as history:
            history.write(json.dumps(dict(state["last_result"],finished_unix=time.time()))+"\n")
        if not asynchronous:
            subprocess.run(["rsync","-a",str(state_path),"joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/"+str(OUT.relative_to(ROOT))+"/"],check=False)
        print("LOCAL_QUEUE",jid,"code",code,"seconds",round(time.time()-tick,2),flush=True)

    state.update(status="complete",active=None,updated_unix=time.time());write(state_path,state)


if __name__=="__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    main()
