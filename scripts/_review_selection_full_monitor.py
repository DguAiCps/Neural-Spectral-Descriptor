#!/usr/bin/env python3
"""로컬 입력 복구·서버 상태를 감시하고 완료 입력 전송을 재개한다."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from _review_selection_bev import read,write
from _review_selection_full_inputs import OUT,INPUT


def alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def main(pid: int) -> None:
    """진행 파일을 동기화하며 입력 생성 실패는 완료 표시 없이 재시도한다."""
    status=OUT/"monitor_status.json"
    previous=read(status) if status.exists() else {}
    synced=set(previous.get("synced",[]))
    remote="joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/"
    spec=read(OUT/"input_relay_plan.json")
    jobs=[(m,r["sensor"],r["seq"]) for m in spec["methods"] for r in spec["rows"]]
    while True:
        write(status,dict(status="scanning_and_syncing",pid=os.getpid(),input_runner_pid=pid,
                          synced=sorted(synced),updated_unix=time.time()))
        completed=[]
        for m,s,q in jobs:
            key=f"{m}_{s}_{q}"; marker=INPUT/"relay_status"/f"{key}.json"
            if not marker.exists() or read(marker).get("status")!="complete": continue
            completed.append(key)
            hosts=read(OUT/"execution_hosts.json") if (OUT/"execution_hosts.json").exists() else {}
            if m=="imlpr" and hosts.get(m)=="local":
                # 이 입력은 서버에서 로컬로 복구한다. 검증 전 로컬 사본을 역전송하지 않는다.
                synced.add(key)
                continue
            if key in synced: continue
            folder=INPUT/m/("data/test_odom/"+s+"_"+q if m in ("helios","imlpr") else s+"_"+q)
            in_flight=False
            for proc in Path("/proc").iterdir():
                if not proc.name.isdigit(): continue
                try: command=(proc/"cmdline").read_bytes()
                except (FileNotFoundError,PermissionError): continue
                if command.startswith(b"rsync\x00") and str(folder).encode() in command:
                    in_flight=True; break
            if in_flight: continue
            rel=folder.relative_to(ROOT)
            code=subprocess.call(["rsync","-az","--exclude=.*","--exclude=*.tmp*",str(folder),remote+str(rel.parent)+"/"])
            if code==0: synced.add(key)
        names=["queue_state.json","catalog_progress.json","retained_protocol.json"]
        subprocess.run(["rsync","-a",*[remote+str(OUT.relative_to(ROOT)/n) for n in names],str(OUT)+"/"],check=False)
        for part in ("retained","bev_rerun"):
            destination=OUT/part; destination.mkdir(exist_ok=True)
            # 로컬 소유 방법은 완성 원장+JSON을 uploader가 서버로 보낸다.
            # 서버의 JSON만 역복사하면 NPZ 없는 완료 마커가 생길 수 있다.
            hosts=read(OUT/"execution_hosts.json") if (OUT/"execution_hosts.json").exists() else {}
            excluded=[]
            if part=="retained":
                for method,host in hosts.items():
                    if host!="local":continue
                    excluded.extend(f"--exclude=/{role}/{method}/***" for role in ("val","test"))
                    excluded.extend(f"--exclude=/{name}_{method}.json" for name in ("selection","validation","report"))
            subprocess.run(["rsync","-a","--ignore-missing-args",*excluded,"--include=*/","--include=*.json","--exclude=*",
                remote+str(OUT.relative_to(ROOT)/part)+"/",str(destination)+"/"],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
        requests=OUT/"test_input_requests"; requests.mkdir(exist_ok=True)
        subprocess.run(["rsync","-a","--ignore-missing-args",remote+str(OUT.relative_to(ROOT)/"test_input_requests")+"/",str(requests)+"/"],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
        # 선택 후 입력은 별도 재개 가능한 test_relay가 처리한다.
        restart_control=OUT/"restart_control.json"
        auto_restart=(read(restart_control).get("input_autorestart",True)
                      if restart_control.exists() else True)
        if not alive(pid) and len(completed)<len(jobs) and auto_restart:
            log=(OUT/"logs/input_relay_restarts.log").open("a")
            handoff=OUT/"val_riv_handoff.json"
            runner=([sys.executable,"-u",str(ROOT/"scripts/_review_selection_full_val_riv.py"),"run"]
                    if handoff.exists() and read(handoff).get("previous_workers_stopped") else
                    [sys.executable,"-u",str(ROOT/"scripts/_review_selection_full_relay.py"),"run","--workers","2"])
            child=subprocess.Popen(runner,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            pid=child.pid
        ready=all(f"{m}_{s}_{q}" in synced for m,s,q in jobs)
        write(status,dict(status="inputs_complete" if ready else "running",pid=os.getpid(),
            input_runner_pid=pid,input_runner_alive=alive(pid),total_jobs=len(jobs),cpu_completed=completed,
            synced=sorted(synced),updated_unix=time.time()))
        time.sleep(30)


if __name__=="__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--adopt",type=int,required=True)
    main(p.parse_args().adopt)
