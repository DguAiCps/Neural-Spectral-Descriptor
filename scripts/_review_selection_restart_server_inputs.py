#!/usr/bin/env python3
"""재부팅 복구 입력을 검산·전송하고 서버 CPU RIV 작업열을 지속한다."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from _review_selection_bev import read,write,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/pdv2_review_selection_full_20261105"
REL=OUT.relative_to(ROOT)
REMOTE_ROOT="/home/qkrrlans00/Neural-Spectral-Codec"
REMOTE="joms-gpu:"+REMOTE_ROOT+"/"
CONTAINER_ROOT="/workspace/Neural-Spectral-Codec"
KEYS=["Roundabout01","Roundabout02","Roundabout03"]


def copy_to_server(source: Path, target: Path) -> None:
    subprocess.run(["ssh","joms-gpu","mkdir","-p",REMOTE_ROOT+"/"+str(target)],check=True)
    subprocess.run(["rsync","-aO","--no-owner","--no-group","--no-perms",str(source),REMOTE+str(target)+"/"],check=True)


def cpu(sequence: str, smoke: bool, log) -> None:
    args=["ssh","joms-gpu","docker","exec","-e","CUDA_VISIBLE_DEVICES=","-e","OMP_NUM_THREADS=1",
          "-e","OPENBLAS_NUM_THREADS=1","-e","MKL_NUM_THREADS=1","-w",CONTAINER_ROOT,"ring_baseline",
          "python3","-u","scripts/_review_selection_full_server_riv.py","--role","val","--sequence",sequence,
          "--processes","3" if smoke else "24"]
    if smoke:args.append("--smoke")
    subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)


def main() -> None:
    """완료된 입력만 넘기며 서로 다른 GPU 작업에는 관여하지 않는다."""
    lock=(OUT/"restart_server_inputs.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=OUT/"restart_server_inputs_state.json"
    current=read(status) if status.exists() else dict(transferred=[],completed=[])
    current.update(pid=os.getpid(),started_unix=time.time())
    try:
        for seq in KEYS:
            if seq in current["transferred"]:continue
            home=OUT/"inputs/rawpack/val"/f"helipr_{seq}"
            while not (home/"verified.json").exists():
                current.update(status="waiting_rawpack",sequence=seq,updated_unix=time.time());write(status,current);time.sleep(10)
            assert read(home/"verified.json")["done_sha256"]==sha(home/"done.json")
            current.update(status="transferring_rawpack",sequence=seq,updated_unix=time.time());write(status,current)
            copy_to_server(home,REL/"inputs/rawpack/val")
            current["transferred"].append(seq);write(status,current)
        smoke=OUT/"server_riv_smoke"
        reference=smoke/"reference";reference.mkdir(parents=True,exist_ok=True)
        old=read(OUT/"riv_cache_audit.json")
        for item in old["cases"]:
            pos=item["position"];source=OUT/f"inputs/imlpr/data/test_odom/helipr_Roundabout01/{pos:06d}.npy"
            assert sha(source)==item["reference_sha256"]
            target=reference/source.name
            if not target.exists():target.write_bytes(source.read_bytes())
        copy_to_server(reference,REL/"server_riv_smoke")
        remote_proof=REMOTE+str(REL/"server_riv_smoke/validated.json")
        if not (smoke/"validated.json").exists():
            current.update(status="server_cpu_equivalence_smoke",updated_unix=time.time());write(status,current)
            with (OUT/"logs/server_riv_smoke.log").open("a") as log:cpu("helipr/Roundabout01",True,log)
            subprocess.run(["rsync","-aO","--no-owner","--no-group","--no-perms",remote_proof,str(smoke)+"/"],check=True)
        assert read(smoke/"validated.json")["all_arrays_exact"]
        handoff=OUT/"val_riv_server_handoff.json"
        write(handoff,dict(created_unix=time.time(),local_workers_stopped=True,reason="WSL reboot stopped local CPU workers; they were not restarted",
                           original_RIV_smoke_exact=True,server_CPU_processes=24,GPU_assignments_unchanged=True))
        copy_to_server(handoff,REL)
        for seq in KEYS:
            if seq in current["completed"]:continue
            home=OUT/"inputs/imlpr/data/test_odom"/f"helipr_{seq}"
            current.update(status="copying_preserved_riv_before_server_compute",sequence=seq,updated_unix=time.time());write(status,current)
            # 완성·검산된 로컬 파일만 서버에 보내서 재계산을 피한다.
            remote_home=REMOTE_ROOT+"/"+str(REL/"inputs/imlpr/data/test_odom"/f"helipr_{seq}")
            finished=subprocess.call(["ssh","joms-gpu","test","-f",remote_home+"/done.json"])==0
            if not finished:
                if not (OUT/"val_riv_prefetch_launch.json").exists():
                    copy_to_server(home,REL/"inputs/imlpr/data/test_odom")
                current.update(status="server_CPU_RIV_running",sequence=seq,updated_unix=time.time());write(status,current)
                with (OUT/"logs"/f"server_riv_val_{seq}.log").open("a") as log:cpu("helipr/"+seq,False,log)
            while not (OUT/"restart_imlpr_storage.json").exists() or read(OUT/"restart_imlpr_storage.json")["status"]!="complete":
                current.update(status="waiting_verified_storage_migration_before_download",updated_unix=time.time());write(status,current);time.sleep(10)
            current.update(status="receiving_complete_riv",sequence=seq,updated_unix=time.time());write(status,current)
            subprocess.run(["rsync","-aO","--no-owner","--no-group","--no-perms","--exclude=.*","--exclude=*.tmp*",
                            REMOTE+str(REL/"inputs/imlpr/data/test_odom"/f"helipr_{seq}")+"/",str(home)+"/"],check=True)
            marker=OUT/"inputs/relay_status"/f"imlpr_helipr_{seq}.json"
            data=subprocess.check_output(["ssh","joms-gpu","cat",REMOTE_ROOT+"/"+str(marker.relative_to(ROOT))],text=True)
            write(marker,json.loads(data))
            current["completed"].append(seq);write(status,current)
        current.update(status="complete",updated_unix=time.time());write(status,current)
    except Exception as exc:
        current.update(status="failed_resumable",error=repr(exc),updated_unix=time.time());write(status,current)
        raise


if __name__=="__main__":main()
